from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import train_locomotion as tl
import transition_autoencoder as tae


def apply_config_dict(cfg: tl.TrainConfig, values: dict) -> None:
    valid = {field.name for field in fields(tl.TrainConfig)}
    for key, value in values.items():
        if key not in valid:
            continue
        current = getattr(cfg, key)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(cfg, key, value)


def load_controller(
    checkpoint_path: Path,
    clips: list[tl.MotionClip],
    cfg: tl.TrainConfig,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    input_dim, output_dim = tl.make_batch_dims(clips[0], cfg)
    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def evenly_spaced_starts(max_start: int, count: int) -> list[int]:
    if max_start <= 1:
        return [1]
    if count <= 0 or count >= max_start:
        return list(range(1, max_start + 1))
    values = np.linspace(1, max_start, num=count, dtype=np.int64)
    return sorted(set(int(x) for x in values.tolist()))


@torch.no_grad()
def collect_generated_features(
    model: torch.nn.Module,
    clips: list[tl.MotionClip],
    cfg: tl.TrainConfig,
    device: torch.device,
    starts_per_clip: int,
    rollout_steps: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for clip in clips:
        if clip.cyclic_animation:
            max_start = max(1, clip.cyclic_period - 1)
            steps = rollout_steps if rollout_steps > 0 else max_start
        else:
            steps = rollout_steps if rollout_steps > 0 else max(1, clip.T - 2)
            max_start = max(1, clip.T - steps - 1)
        starts = evenly_spaced_starts(max_start, starts_per_clip)
        prev_idx = torch.tensor(starts, dtype=torch.long, device=device) - 1
        cur_idx = torch.tensor(starts, dtype=torch.long, device=device)
        prev_pose = tl.get_pose_from_clip(clip, prev_idx, device)
        cur_pose = tl.get_pose_from_clip(clip, cur_idx, device)
        prev_pose, cur_pose = tl.maybe_apply_initial_offsets(clip, prev_idx, cur_idx, prev_pose, cur_pose, cfg, device)
        for _step in range(steps):
            inp = tl.build_input(clip, prev_idx, cur_idx, prev_pose, cur_pose, cfg, device)
            raw_out = tl.predict_next_raw(model, inp, cur_pose, cfg)
            pred_pose, _raw_pose = tl.output_to_pose(raw_out, clip)
            target_idx = cur_idx + 1
            root_pos, root_rot, _yaw, _heading = tl.root_state(clip, target_idx, cfg, device)
            _global_pos, _global_rot, pred_canon = tl.fk_from_pose(clip, root_pos, root_rot, pred_pose, device)
            next_pose = {
                "pelvis_pos": pred_pose["pelvis_pos"],
                "pelvis_rot6": pred_pose["pelvis_rot6"],
                "nonpelvis_rot6": pred_pose["nonpelvis_rot6"],
                "canon_pos": pred_canon,
                "contacts": pred_pose["contacts"],
            }
            features = tae.transition_feature_from_next_pose(
                clip,
                prev_idx,
                cur_idx,
                prev_pose,
                cur_pose,
                next_pose,
                cfg,
                device,
            )
            chunks.append(features.detach().cpu())
            prev_pose = cur_pose
            cur_pose = next_pose
            prev_idx = cur_idx
            cur_idx = target_idx
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def energy(model: torch.nn.Module, x: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
    values = []
    model.eval()
    for start in range(0, x.shape[0], batch_size):
        batch = x[start : start + batch_size]
        recon = model(batch)
        target = model.target(batch) if hasattr(model, "target") else batch
        values.append(tae.reconstruction_loss_rows(model, recon, target, loss_type="mse"))
    return torch.cat(values, dim=0)


def write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sample_feature_rows(features: torch.Tensor, max_rows: int, seed: int) -> torch.Tensor:
    if max_rows <= 0 or features.shape[0] <= max_rows:
        return features
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    indices = torch.randperm(features.shape[0], generator=gen)[:max_rows]
    return features.index_select(0, indices)


@torch.no_grad()
def keep_low_energy_fakes(
    fake_features: torch.Tensor,
    init_ckpt: dict | None,
    schema: dict[str, int],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    keep_fraction: float,
) -> torch.Tensor:
    keep_fraction = float(keep_fraction)
    if init_ckpt is None or keep_fraction <= 0.0 or keep_fraction >= 1.0 or fake_features.shape[0] <= 1:
        return fake_features
    ae_cfg = tae.AEConfig(**init_ckpt["config"])
    prior = tae.TransitionAutoencoder(schema["total_dim"], ae_cfg, schema).to(device)
    missing, unexpected = prior.load_state_dict(init_ckpt["model"], strict=False)
    if missing or unexpected:
        print(f"hard-negative score prior partial load missing={missing} unexpected={unexpected}", flush=True)
    prior.eval()
    fake_norm = tae.normalise(fake_features, mean.cpu(), std.cpu()).to(device)
    scores = energy(prior, fake_norm).detach().cpu()
    keep = max(1, int(math.ceil(fake_features.shape[0] * keep_fraction)))
    indices = torch.argsort(scores)[:keep]
    return fake_features.index_select(0, indices)


def save_checkpoint(
    path: Path,
    model: tae.TransitionAutoencoder,
    ae_cfg: tae.AEConfig,
    locomotion_cfg: tl.TrainConfig,
    schema: dict[str, int],
    mean: torch.Tensor,
    std: torch.Tensor,
    clips: list[tl.MotionClip],
    epoch: int,
    best: float,
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(ae_cfg),
            "locomotion_config": asdict(locomotion_cfg),
            "schema": schema,
            "mean": mean.detach().cpu(),
            "std": std.detach().cpu(),
            "metadata": {
                "npz_folder": str(Path(clips[0].path).parent),
                "body_names": clips[0].body_names,
                "parents_body": clips[0].parents_body.tolist(),
                **metadata,
            },
            "epoch": epoch,
            "best": best,
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    start_time = time.perf_counter()
    tl.set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    folder = tl.resolve_path(args.folder_path)
    controller_checkpoint = tl.resolve_path(args.model_checkpoint)
    controller_ckpt = torch.load(controller_checkpoint, map_location="cpu", weights_only=False)
    locomotion_cfg = tl.TrainConfig()
    apply_config_dict(locomotion_cfg, controller_ckpt.get("config", {}))
    locomotion_cfg.device = str(device)
    locomotion_cfg.cyclic_animation = args.cyclic_animation
    locomotion_cfg.use_torch_compile = False
    init_prior_path = tl.resolve_path(args.init_prior_checkpoint) if args.init_prior_checkpoint else None
    init_ckpt = None
    if init_prior_path is not None:
        init_ckpt = torch.load(init_prior_path, map_location="cpu", weights_only=False)
        apply_config_dict(locomotion_cfg, init_ckpt.get("locomotion_config", {}))
    tl.apply_cuda_performance_settings(locomotion_cfg, device)

    if args.periodic_folder_path or args.nonperiodic_folder_path:
        clip_specs = tl.clip_specs_from_folders(
            args.folder_path,
            args.periodic_folder_path or None,
            args.nonperiodic_folder_path or None,
        )
        clips = tl.load_clips_from_specs(clip_specs, locomotion_cfg)
    else:
        clips = tl.load_clips(folder, locomotion_cfg)
    controller = load_controller(controller_checkpoint, clips, locomotion_cfg, device)
    real_features, _clip_ids, _cur_idx = tae.collect_clean_feature_rows(clips, locomotion_cfg, device)
    fake_features = collect_generated_features(
        controller,
        clips,
        locomotion_cfg,
        device,
        args.fake_starts_per_clip,
        args.fake_rollout_steps,
    )

    if init_prior_path is not None:
        assert init_ckpt is not None
        ae_cfg = tae.AEConfig(**init_ckpt["config"])
    else:
        ae_cfg = tae.AEConfig(
            folder_path=args.folder_path,
            run_name=args.run_name,
            output_dir=args.output_dir,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            num_hidden_layers=args.num_hidden_layers,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            seed=args.seed,
            device=args.device,
            cyclic_animation=args.cyclic_animation,
            input_noise_std=args.input_noise_std,
            std_floor=args.std_floor,
        )
    ae_cfg.folder_path = args.folder_path
    ae_cfg.run_name = args.run_name
    ae_cfg.output_dir = args.output_dir
    ae_cfg.learning_rate = args.learning_rate
    ae_cfg.weight_decay = args.weight_decay
    ae_cfg.batch_size = args.batch_size
    ae_cfg.max_epochs = args.max_epochs
    ae_cfg.seed = args.seed
    ae_cfg.device = args.device
    ae_cfg.cyclic_animation = args.cyclic_animation
    ae_cfg.input_noise_std = args.input_noise_std
    ae_cfg.std_floor = args.std_floor
    ae_cfg.compatibility_weight = 1.0 if args.compatibility_fake_weight > 0.0 else 0.0

    schema = tae.transition_schema(clips[0], locomotion_cfg)
    if real_features.shape[1] != schema["total_dim"] or fake_features.shape[1] != schema["total_dim"]:
        raise RuntimeError(
            f"feature dim mismatch real={real_features.shape} fake={fake_features.shape} schema={schema['total_dim']}"
        )

    mean = real_features.mean(dim=0)
    std = real_features.std(dim=0).clamp_min(ae_cfg.std_floor)
    raw_fake_count = int(fake_features.shape[0])
    fake_features = keep_low_energy_fakes(
        fake_features,
        init_ckpt,
        schema,
        mean,
        std,
        device,
        args.hard_negative_keep_fraction,
    )
    kept_fake_count = int(fake_features.shape[0])
    buffer_loaded_count = 0
    fake_buffer_path = tl.resolve_path(args.fake_buffer_path) if args.fake_buffer_path else None
    if fake_buffer_path is not None and fake_buffer_path.exists():
        payload = torch.load(fake_buffer_path, map_location="cpu", weights_only=False)
        buffered = payload["features"] if isinstance(payload, dict) and "features" in payload else payload
        if not isinstance(buffered, torch.Tensor):
            raise TypeError(f"fake buffer did not contain a tensor: {fake_buffer_path}")
        if buffered.ndim != 2 or buffered.shape[1] != schema["total_dim"]:
            raise RuntimeError(
                f"fake buffer dim mismatch {tuple(buffered.shape)} expected (*,{schema['total_dim']}) at {fake_buffer_path}"
            )
        buffer_loaded_count = int(buffered.shape[0])
        fake_features = torch.cat((buffered.float(), fake_features.cpu().float()), dim=0)
    if fake_buffer_path is not None:
        fake_features = sample_feature_rows(fake_features.cpu().float(), args.fake_buffer_max_rows, args.seed + 1009)
        fake_buffer_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "features": fake_features,
                "schema": schema,
                "updated_at": time.time(),
                "source_controller_checkpoint": str(controller_checkpoint),
                "raw_fake_count": raw_fake_count,
                "kept_fake_count": kept_fake_count,
                "buffer_loaded_count": buffer_loaded_count,
            },
            fake_buffer_path,
        )
        print(
            f"fake buffer {fake_buffer_path} loaded={buffer_loaded_count} raw={raw_fake_count} "
            f"kept={kept_fake_count} total={fake_features.shape[0]}",
            flush=True,
        )
    real_norm = tae.normalise(real_features, mean, std).to(device)
    fake_norm = tae.normalise(fake_features, mean, std).to(device)
    mean = mean.to(device)
    std = std.to(device)

    model = tae.TransitionAutoencoder(schema["total_dim"], ae_cfg, schema).to(device)
    if init_prior_path is not None:
        init_ckpt = torch.load(init_prior_path, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(init_ckpt["model"], strict=False)
        if missing or unexpected:
            print(f"init prior partial load missing={missing} unexpected={unexpected}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=ae_cfg.learning_rate, weight_decay=ae_cfg.weight_decay)
    run_dir = tl.resolve_path(args.output_dir) / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    writer = SummaryWriter(run_dir / "tb")
    metadata = {
        "training_type": "model_aware_transition_ae",
        "source_controller_checkpoint": str(controller_checkpoint),
        "init_prior_checkpoint": str(init_prior_path) if init_prior_path is not None else "",
        "fake_starts_per_clip": args.fake_starts_per_clip,
        "fake_rollout_steps": args.fake_rollout_steps,
        "fake_margin": args.fake_margin,
        "fake_weight": args.fake_weight,
        "real_weight": args.real_weight,
        "compatibility_real_weight": args.compatibility_real_weight,
        "compatibility_fake_weight": args.compatibility_fake_weight,
        "fake_buffer_path": str(fake_buffer_path) if fake_buffer_path is not None else "",
        "fake_buffer_max_rows": args.fake_buffer_max_rows,
        "hard_negative_keep_fraction": args.hard_negative_keep_fraction,
        "raw_fake_count": raw_fake_count,
        "kept_fake_count": kept_fake_count,
        "buffer_loaded_count": buffer_loaded_count,
    }
    (run_dir / "config.json").write_text(
        json.dumps({"ae_config": asdict(ae_cfg), "metadata": metadata}, indent=2),
        encoding="utf-8",
    )

    print(
        f"model_aware_ae run={args.run_name} real={real_norm.shape[0]} fake={fake_norm.shape[0]} "
        f"dim={real_norm.shape[1]} margin={args.fake_margin:g} init={init_prior_path}",
        flush=True,
    )

    real_indices = torch.arange(real_norm.shape[0], device=device)
    fake_indices = torch.arange(fake_norm.shape[0], device=device)
    rows: list[dict[str, float]] = []
    best = math.inf
    stalls = 0
    last_epoch = 0
    for epoch in range(1, args.max_epochs + 1):
        last_epoch = epoch
        model.train()
        perm = real_indices[torch.randperm(real_indices.numel(), device=device)]
        loss_sum = torch.zeros((), device=device)
        real_sum = torch.zeros((), device=device)
        fake_sum = torch.zeros((), device=device)
        fake_energy_sum = torch.zeros((), device=device)
        compat_real_sum = torch.zeros((), device=device)
        compat_fake_sum = torch.zeros((), device=device)
        count = 0
        for start in range(0, perm.numel(), args.batch_size):
            real_idx = perm[start : start + args.batch_size]
            real_batch = real_norm.index_select(0, real_idx)
            fake_idx = fake_indices[torch.randint(0, fake_indices.numel(), real_idx.shape, device=device)]
            fake_batch = fake_norm.index_select(0, fake_idx)
            noisy_real = real_batch
            if args.input_noise_std > 0.0:
                noisy_real = real_batch + args.input_noise_std * torch.randn_like(real_batch)
            real_recon = model(noisy_real)
            fake_recon = model(fake_batch)
            real_loss = tae.reconstruction_loss(model, real_recon, model.target(real_batch), loss_type="huber")
            fake_err = tae.reconstruction_loss_rows(model, fake_recon, model.target(fake_batch), loss_type="mse")
            fake_loss = F.relu(args.fake_margin - fake_err).mean()
            compat_real_loss = torch.zeros((), device=device)
            compat_fake_loss = torch.zeros((), device=device)
            if model.has_compatibility_head() and args.compatibility_fake_weight > 0.0:
                real_logits = model.compatibility_logits(real_batch)
                fake_logits = model.compatibility_logits(fake_batch)
                compat_real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                compat_fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            loss = (
                args.real_weight * real_loss
                + args.fake_weight * fake_loss
                + args.compatibility_real_weight * compat_real_loss
                + args.compatibility_fake_weight * compat_fake_loss
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum = loss_sum + loss.detach()
            real_sum = real_sum + real_loss.detach()
            fake_sum = fake_sum + fake_loss.detach()
            fake_energy_sum = fake_energy_sum + fake_err.mean().detach()
            compat_real_sum = compat_real_sum + compat_real_loss.detach()
            compat_fake_sum = compat_fake_sum + compat_fake_loss.detach()
            count += 1

        train_loss = float((loss_sum / max(1, count)).cpu())
        train_real = float((real_sum / max(1, count)).cpu())
        train_fake = float((fake_sum / max(1, count)).cpu())
        train_fake_energy = float((fake_energy_sum / max(1, count)).cpu())
        train_compat_real = float((compat_real_sum / max(1, count)).cpu())
        train_compat_fake = float((compat_fake_sum / max(1, count)).cpu())
        improved = train_loss < best - args.min_delta
        stalls = 0 if improved else stalls + 1
        if train_loss < best:
            best = train_loss
            save_checkpoint(
                ckpt_dir / "checkpoint_best.pt",
                model,
                ae_cfg,
                locomotion_cfg,
                schema,
                mean,
                std,
                clips,
                epoch,
                best,
                metadata,
            )
        if epoch == 1 or epoch % args.eval_every_epochs == 0:
            real_e = energy(model, real_norm)
            fake_e = energy(model, fake_norm)
            if model.has_compatibility_head():
                real_logits = model.compatibility_logits(real_norm)
                fake_logits = model.compatibility_logits(fake_norm)
                real_compat_acc = float((real_logits > 0.0).float().mean().cpu())
                fake_compat_acc = float((fake_logits < 0.0).float().mean().cpu())
                real_compat_penalty = float(F.softplus(-real_logits).mean().cpu())
                fake_compat_penalty = float(F.softplus(-fake_logits).mean().cpu())
            else:
                real_compat_acc = fake_compat_acc = real_compat_penalty = fake_compat_penalty = 0.0
            row = {
                "epoch": float(epoch),
                "loss": train_loss,
                "real_loss": train_real,
                "fake_hinge": train_fake,
                "real_energy_mean": float(real_e.mean().cpu()),
                "real_energy_p95": float(torch.quantile(real_e, 0.95).cpu()),
                "fake_energy_mean": float(fake_e.mean().cpu()),
                "fake_energy_p05": float(torch.quantile(fake_e, 0.05).cpu()),
                "fake_margin_success": float((fake_e > args.fake_margin).float().mean().cpu()),
                "gap": float((fake_e.mean() - real_e.mean()).cpu()),
                "compat_real_acc": real_compat_acc,
                "compat_fake_acc": fake_compat_acc,
                "compat_real_penalty": real_compat_penalty,
                "compat_fake_penalty": fake_compat_penalty,
            }
            rows.append(row)
            write_rows(run_dir / "model_aware_ae_report.csv", rows)
            print(
                f"epoch={epoch:04d} loss={train_loss:.6g} real={train_real:.6g} "
                f"fake_hinge={train_fake:.6g} real_e={row['real_energy_mean']:.6g} "
                f"fake_e={row['fake_energy_mean']:.6g} fake_ok={row['fake_margin_success']:.3f} "
                f"compat_real={real_compat_acc:.3f} compat_fake={fake_compat_acc:.3f} "
                f"stalls={stalls} elapsed_s={time.perf_counter() - start_time:.1f}",
                flush=True,
            )
        elif epoch % 10 == 0:
            print(
                f"epoch={epoch:04d} loss={train_loss:.6g} real={train_real:.6g} "
                f"fake_hinge={train_fake:.6g} fake_batch_e={train_fake_energy:.6g} "
                f"compat_real={train_compat_real:.6g} compat_fake={train_compat_fake:.6g} stalls={stalls}",
                flush=True,
            )
        writer.add_scalar("loss/train_total", train_loss, epoch)
        writer.add_scalar("loss/real_reconstruction", train_real, epoch)
        writer.add_scalar("loss/fake_hinge", train_fake, epoch)
        writer.add_scalar("loss/compat_real", train_compat_real, epoch)
        writer.add_scalar("loss/compat_fake", train_compat_fake, epoch)
        writer.add_scalar("energy/fake_batch_mean", train_fake_energy, epoch)
        writer.add_scalar("loss/best", best, epoch)
        if args.stall_patience_epochs > 0 and stalls >= args.stall_patience_epochs:
            print(f"stopped on stall epoch={epoch} best={best:.6g}", flush=True)
            break

    save_checkpoint(
        ckpt_dir / "checkpoint_last.pt",
        model,
        ae_cfg,
        locomotion_cfg,
        schema,
        mean,
        std,
        clips,
        last_epoch,
        best,
        metadata,
    )
    writer.close()
    print(f"model_aware_ae run_dir={run_dir}", flush=True)
    print(f"model_aware_ae best_checkpoint={ckpt_dir / 'checkpoint_best.pt'}", flush=True)
    print(f"model_aware_ae last_checkpoint={ckpt_dir / 'checkpoint_last.pt'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a transition AE against real transitions and model-generated fakes.")
    parser.add_argument("--folder-path", required=True)
    parser.add_argument("--periodic-folder-path", default="")
    parser.add_argument("--nonperiodic-folder-path", default="")
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--init-prior-checkpoint", default="")
    parser.add_argument("--run-name", default="model_aware_transition_ae")
    parser.add_argument("--output-dir", default="training/runs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cyclic-animation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--input-noise-std", type=float, default=0.0)
    parser.add_argument("--std-floor", type=float, default=1e-4)
    parser.add_argument("--fake-margin", type=float, default=0.02)
    parser.add_argument("--fake-weight", type=float, default=1.0)
    parser.add_argument("--real-weight", type=float, default=1.0)
    parser.add_argument("--compatibility-real-weight", type=float, default=1.0)
    parser.add_argument("--compatibility-fake-weight", type=float, default=0.0)
    parser.add_argument("--fake-starts-per-clip", type=int, default=16)
    parser.add_argument("--fake-rollout-steps", type=int, default=0)
    parser.add_argument(
        "--fake-buffer-path",
        default="",
        help="Optional persistent CPU tensor buffer for model-generated hard negatives.",
    )
    parser.add_argument("--fake-buffer-max-rows", type=int, default=200000)
    parser.add_argument(
        "--hard-negative-keep-fraction",
        type=float,
        default=1.0,
        help="If an init prior is present, keep only this lowest-energy fraction of fresh generated fakes.",
    )
    parser.add_argument("--eval-every-epochs", type=int, default=10)
    parser.add_argument("--stall-patience-epochs", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
