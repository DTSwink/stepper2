from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import train_locomotion as tl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AEConfig:
    folder_path: str = "data/npz_final"
    periodic_folder_path: str = ""
    nonperiodic_folder_path: str = ""
    run_name: str = "transition_ae"
    date_prefix_run_name: bool = True
    output_dir: str = "training/runs"
    latent_dim: int = 64
    hidden_dim: int = 512
    num_hidden_layers: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    max_epochs: int = 2000
    seed: int = 1234
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cyclic_animation: bool = False
    input_noise_std: float = 0.0
    std_floor: float = 1e-4
    target_loss_reduction: float = 0.995
    stall_patience_epochs: int = 120
    min_delta: float = 1e-6
    tier_eval_every_epochs: int = 25
    timed_checkpoint_interval_minutes: float = 30.0
    compatibility_weight: float = 0.0
    compatibility_direction_weight: float = 1.0
    compatibility_temporal_weight: float = 1.0
    compatibility_temporal_min_skip: int = 2
    compatibility_temporal_max_skip: int = 8
    compatibility_hidden_dim: int = 256
    compatibility_num_hidden_layers: int = 2


class TransitionAutoencoder(nn.Module):
    def __init__(self, dim: int, cfg: AEConfig):
        super().__init__()
        encoder: list[nn.Module] = []
        in_dim = dim
        for _ in range(cfg.num_hidden_layers):
            encoder += [nn.Linear(in_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.GELU()]
            in_dim = cfg.hidden_dim
        encoder += [nn.Linear(in_dim, cfg.latent_dim), nn.GELU()]
        decoder: list[nn.Module] = []
        in_dim = cfg.latent_dim
        for _ in range(cfg.num_hidden_layers):
            decoder += [nn.Linear(in_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.GELU()]
            in_dim = cfg.hidden_dim
        decoder += [nn.Linear(in_dim, dim)]
        self.net = nn.Sequential(*(encoder + decoder))
        self.compatibility_head = None
        if cfg.compatibility_weight > 0.0:
            layers: list[nn.Module] = []
            in_dim = dim
            for _ in range(cfg.compatibility_num_hidden_layers):
                layers += [
                    nn.Linear(in_dim, cfg.compatibility_hidden_dim),
                    nn.LayerNorm(cfg.compatibility_hidden_dim),
                    nn.GELU(),
                ]
                in_dim = cfg.compatibility_hidden_dim
            layers += [nn.Linear(in_dim, 1)]
            self.compatibility_head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def has_compatibility_head(self) -> bool:
        return self.compatibility_head is not None

    def compatibility_logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.compatibility_head is None:
            raise RuntimeError("This transition autoencoder was created without a compatibility head.")
        return self.compatibility_head(x).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def transition_schema(clip: tl.MotionClip, cfg: tl.TrainConfig) -> dict[str, int]:
    pose_dim = 3 + 6 + clip.J * 3 + clip.Jn * 6 + (2 if cfg.use_contact_state else 0)
    velocity_dim = 3 + clip.J * 3
    input_dim, output_dim = tl.make_batch_dims(clip, cfg)
    next_canon_dim = clip.J * 3
    next_velocity_dim = 3 + clip.J * 3
    return {
        "pose_dim": pose_dim,
        "velocity_dim": velocity_dim,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "next_canon_dim": next_canon_dim,
        "next_velocity_dim": next_velocity_dim,
        "total_dim": input_dim + output_dim + next_canon_dim + next_velocity_dim,
        "input_root_start": pose_dim * 2 + velocity_dim,
        "input_root_end": input_dim,
        "next_output_start": input_dim,
        "next_canon_start": input_dim + output_dim,
        "next_velocity_start": input_dim + output_dim + next_canon_dim,
    }


def transition_feature_from_next_pose(
    clip: tl.MotionClip,
    prev_idx: torch.Tensor,
    cur_idx: torch.Tensor,
    prev_pose: dict[str, torch.Tensor],
    cur_pose: dict[str, torch.Tensor],
    next_pose: dict[str, torch.Tensor],
    cfg: tl.TrainConfig,
    device: torch.device,
) -> torch.Tensor:
    model_input = tl.build_input(clip, prev_idx, cur_idx, prev_pose, cur_pose, cfg, device)
    next_output = tl.pose_target_output(next_pose)
    pelvis_next_vel = (next_pose["pelvis_pos"] - cur_pose["pelvis_pos"]) / cfg.pose_delta_scale_final
    joint_next_vel = (next_pose["canon_pos"] - cur_pose["canon_pos"]).reshape(cur_idx.shape[0], -1)
    joint_next_vel = joint_next_vel / cfg.pose_delta_scale_final
    return torch.cat(
        (
            model_input,
            next_output,
            next_pose["canon_pos"].reshape(cur_idx.shape[0], -1),
            pelvis_next_vel,
            joint_next_vel,
        ),
        dim=-1,
    )


def clean_transition_features(
    clip: tl.MotionClip,
    cur_idx: torch.Tensor,
    cfg: tl.TrainConfig,
    device: torch.device,
) -> torch.Tensor:
    prev_idx = cur_idx - 1
    next_idx = cur_idx + 1
    prev_pose = tl.get_pose_from_clip(clip, prev_idx, device)
    cur_pose = tl.get_pose_from_clip(clip, cur_idx, device)
    next_pose = tl.get_pose_from_clip(clip, next_idx, device)
    return transition_feature_from_next_pose(clip, prev_idx, cur_idx, prev_pose, cur_pose, next_pose, cfg, device)


def transition_features_with_offset(
    clip: tl.MotionClip,
    cur_idx: torch.Tensor,
    next_offset: int,
    cfg: tl.TrainConfig,
    device: torch.device,
) -> torch.Tensor:
    prev_idx = cur_idx - 1
    next_idx = cur_idx + next_offset
    prev_pose = tl.get_pose_from_clip(clip, prev_idx, device)
    cur_pose = tl.get_pose_from_clip(clip, cur_idx, device)
    next_pose = tl.get_pose_from_clip(clip, next_idx, device)
    return transition_feature_from_next_pose(clip, prev_idx, cur_idx, prev_pose, cur_pose, next_pose, cfg, device)


def collect_clean_feature_rows(
    clips: list[tl.MotionClip],
    cfg: tl.TrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chunks = []
    clip_chunks = []
    idx_chunks = []
    for ci, clip in enumerate(clips):
        stop = clip.cyclic_period if clip.cyclic_animation else clip.T - cfg.future_window
        if stop <= 1:
            continue
        idx = torch.arange(1, stop, dtype=torch.long, device=device)
        chunks.append(clean_transition_features(clip, idx, cfg, device).detach().cpu())
        clip_chunks.append(torch.full((idx.numel(),), ci, dtype=torch.long))
        idx_chunks.append(idx.detach().cpu())
    return torch.cat(chunks, dim=0), torch.cat(clip_chunks, dim=0), torch.cat(idx_chunks, dim=0)


def collect_clean_features(clips: list[tl.MotionClip], cfg: tl.TrainConfig, device: torch.device) -> torch.Tensor:
    return collect_clean_feature_rows(clips, cfg, device)[0]


def collect_temporal_skip_features(
    clips: list[tl.MotionClip],
    cfg: tl.TrainConfig,
    device: torch.device,
    min_skip: int,
    max_skip: int,
) -> torch.Tensor:
    chunks = []
    min_skip = max(2, int(min_skip))
    max_skip = max(min_skip, int(max_skip))
    for clip in clips:
        stop = clip.cyclic_period if clip.cyclic_animation else min(clip.T - max_skip, clip.T - cfg.future_window)
        if stop <= 1:
            continue
        idx = torch.arange(1, stop, dtype=torch.long, device=device)
        for skip in range(min_skip, max_skip + 1):
            chunks.append(transition_features_with_offset(clip, idx, skip, cfg, device).detach().cpu())
    if not chunks:
        return torch.empty((0, transition_schema(clips[0], cfg)["total_dim"]), dtype=torch.float32)
    return torch.cat(chunks, dim=0)


def normalise(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def denormalise(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std + mean


def alteration_mask(schema: dict[str, int], device: torch.device) -> torch.Tensor:
    mask = torch.ones(schema["total_dim"], device=device)
    mask[schema["input_root_start"] : schema["input_root_end"]] = 0.0
    return mask


def make_statue_tier(x: torch.Tensor, schema: dict[str, int]) -> torch.Tensor:
    y = x.clone()
    pose_dim = schema["pose_dim"]
    input_dim = schema["input_dim"]
    output_start = schema["next_output_start"]
    canon_start = schema["next_canon_start"]
    vel_start = schema["next_velocity_start"]
    output_dim = schema["output_dim"]
    canon_dim = schema["next_canon_dim"]
    vel_dim = schema["next_velocity_dim"]

    current_pose = y[:, :pose_dim]
    current_output = torch.cat(
        (
            current_pose[:, :9],
            current_pose[:, 9 + canon_dim :],
        ),
        dim=-1,
    )
    if current_output.shape[-1] < output_dim:
        current_output = torch.cat(
            (
                current_output,
                torch.zeros(
                    (current_output.shape[0], output_dim - current_output.shape[-1]),
                    dtype=current_output.dtype,
                    device=current_output.device,
                ),
            ),
            dim=-1,
        )
    y[:, output_start : output_start + output_dim] = current_output
    y[:, canon_start : canon_start + canon_dim] = current_pose[:, 9 : 9 + canon_dim]
    y[:, vel_start : vel_start + vel_dim] = 0.0
    return y


def make_tiers(x_norm: torch.Tensor, schema: dict[str, int]) -> dict[str, torch.Tensor]:
    mask = alteration_mask(schema, x_norm.device).unsqueeze(0)
    tier2 = x_norm + 0.05 * torch.randn_like(x_norm) * mask
    noisy_bad = x_norm + 0.75 * torch.randn_like(x_norm) * mask
    statue = make_statue_tier(x_norm, schema)
    tier3 = torch.where((torch.arange(x_norm.shape[0], device=x_norm.device)[:, None] % 2) == 0, statue, noisy_bad)
    tier4 = torch.randn_like(x_norm)
    tier4[:, schema["input_root_start"] : schema["input_root_end"]] = x_norm[
        :, schema["input_root_start"] : schema["input_root_end"]
    ]
    return {
        "tier1_clean": x_norm,
        "tier2_slight": tier2,
        "tier3_bad": tier3,
        "tier4_noise": tier4,
    }


@torch.no_grad()
def reconstruction_errors(model: nn.Module, x_norm: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
    values = []
    model.eval()
    for start in range(0, x_norm.shape[0], batch_size):
        batch = x_norm[start : start + batch_size]
        recon = model(batch)
        values.append(F.mse_loss(recon, batch, reduction="none").mean(dim=-1))
    return torch.cat(values, dim=0)


@torch.no_grad()
def tier_report(model: nn.Module, x_norm: torch.Tensor, schema: dict[str, int]) -> dict[str, float]:
    tiers = make_tiers(x_norm, schema)
    report: dict[str, float] = {}
    for name, values in tiers.items():
        err = reconstruction_errors(model, values)
        report[f"{name}_mean"] = float(err.mean().cpu())
        report[f"{name}_p95"] = float(torch.quantile(err, 0.95).cpu())
    return report


def save_tier_report(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sample_direction_negatives(
    x_norm: torch.Tensor,
    clip_ids: torch.Tensor,
    batch_indices: torch.Tensor,
    schema: dict[str, int],
) -> torch.Tensor:
    batch = x_norm.index_select(0, batch_indices)
    batch_clip_ids = clip_ids.index_select(0, batch_indices)
    body_indices = torch.randint(0, x_norm.shape[0], batch_indices.shape, device=x_norm.device)
    for _ in range(16):
        same_clip = clip_ids.index_select(0, body_indices) == batch_clip_ids
        if not bool(same_clip.any()):
            break
        body_indices[same_clip] = torch.randint(0, x_norm.shape[0], (int(same_clip.sum().item()),), device=x_norm.device)
    negative = x_norm.index_select(0, body_indices).clone()
    root_start = schema["input_root_start"]
    root_end = schema["input_root_end"]
    negative[:, root_start:root_end] = batch[:, root_start:root_end]
    return negative


def compatibility_bce_loss(
    model: TransitionAutoencoder,
    positive: torch.Tensor,
    direction_negative: torch.Tensor | None,
    temporal_negative: torch.Tensor | None,
    cfg: AEConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = positive.new_zeros(())
    if not model.has_compatibility_head() or cfg.compatibility_weight <= 0.0:
        return zero, {
            "compat_pos_loss": zero,
            "compat_direction_loss": zero,
            "compat_temporal_loss": zero,
            "compat_pos_acc": zero,
            "compat_direction_acc": zero,
            "compat_temporal_acc": zero,
        }

    pos_logits = model.compatibility_logits(positive)
    pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
    total = pos_loss
    direction_loss = zero
    temporal_loss = zero
    direction_acc = zero
    temporal_acc = zero
    if direction_negative is not None and cfg.compatibility_direction_weight > 0.0:
        direction_logits = model.compatibility_logits(direction_negative)
        direction_loss = F.binary_cross_entropy_with_logits(direction_logits, torch.zeros_like(direction_logits))
        total = total + cfg.compatibility_direction_weight * direction_loss
        direction_acc = (direction_logits < 0.0).float().mean()
    if temporal_negative is not None and cfg.compatibility_temporal_weight > 0.0:
        temporal_logits = model.compatibility_logits(temporal_negative)
        temporal_loss = F.binary_cross_entropy_with_logits(temporal_logits, torch.zeros_like(temporal_logits))
        total = total + cfg.compatibility_temporal_weight * temporal_loss
        temporal_acc = (temporal_logits < 0.0).float().mean()
    return total, {
        "compat_pos_loss": pos_loss.detach(),
        "compat_direction_loss": direction_loss.detach(),
        "compat_temporal_loss": temporal_loss.detach(),
        "compat_pos_acc": (pos_logits > 0.0).float().mean().detach(),
        "compat_direction_acc": direction_acc.detach(),
        "compat_temporal_acc": temporal_acc.detach(),
    }


def train(args: argparse.Namespace) -> None:
    cfg = AEConfig()
    for field in cfg.__dataclass_fields__:
        value = getattr(args, field, None)
        if value is not None:
            setattr(cfg, field, value)
    if cfg.date_prefix_run_name:
        cfg.run_name = tl.date_prefixed_run_name(cfg.run_name)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    tl.apply_cuda_performance_settings(tl.TrainConfig(device=cfg.device), device)
    locomotion_cfg = tl.TrainConfig()
    locomotion_cfg.cyclic_animation = cfg.cyclic_animation
    clip_specs = tl.clip_specs_from_folders(
        cfg.folder_path,
        cfg.periodic_folder_path or None,
        cfg.nonperiodic_folder_path or None,
    )
    clips = tl.load_clips_from_specs(clip_specs, locomotion_cfg)
    clean, feature_clip_ids, _feature_cur_idx = collect_clean_feature_rows(clips, locomotion_cfg, device)
    skip_features = None
    if cfg.compatibility_weight > 0.0 and cfg.compatibility_temporal_weight > 0.0:
        skip_features = collect_temporal_skip_features(
            clips,
            locomotion_cfg,
            device,
            cfg.compatibility_temporal_min_skip,
            cfg.compatibility_temporal_max_skip,
        )
    mean = clean.mean(dim=0)
    std = clean.std(dim=0).clamp_min(cfg.std_floor)
    x_norm = normalise(clean, mean, std).to(device)
    clip_ids = feature_clip_ids.to(device)
    skip_norm = normalise(skip_features, mean, std).to(device) if skip_features is not None else None
    mean = mean.to(device)
    std = std.to(device)
    schema = transition_schema(clips[0], locomotion_cfg)

    run_dir = tl.resolve_path(cfg.output_dir) / cfg.run_name
    ckpt_dir = run_dir / "checkpoints"
    writer = SummaryWriter(run_dir / "tb")
    model = TransitionAutoencoder(schema["total_dim"], cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    print(
        f"transition_ae samples={x_norm.shape[0]} dim={x_norm.shape[1]} latent={cfg.latent_dim} "
        f"compat={cfg.compatibility_weight:g} device={device} "
        f"folders={[(str(tl.npz_folder_from_path(path)), cyclic) for path, cyclic in clip_specs]}",
        flush=True,
    )
    best = math.inf
    baseline = None
    target = None
    stalls = 0
    rows: list[dict[str, float]] = []
    indices = torch.arange(x_norm.shape[0], device=device)
    start_time = time.perf_counter()
    timed_interval_seconds = 60.0 * max(0.0, float(cfg.timed_checkpoint_interval_minutes))
    next_timed_checkpoint_at = start_time + timed_interval_seconds if timed_interval_seconds > 0.0 else math.inf
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        perm = indices[torch.randperm(indices.numel(), device=device)]
        loss_sum = torch.zeros((), device=device)
        recon_sum = torch.zeros((), device=device)
        compat_sum = torch.zeros((), device=device)
        compat_parts_sum: dict[str, torch.Tensor] = {}
        loss_count = 0
        for start in range(0, perm.numel(), cfg.batch_size):
            batch_indices = perm[start : start + cfg.batch_size]
            batch = x_norm.index_select(0, batch_indices)
            noisy = batch
            if cfg.input_noise_std > 0.0:
                noisy = batch + cfg.input_noise_std * torch.randn_like(batch)
            recon = model(noisy)
            recon_loss = F.huber_loss(recon, batch)
            compat_loss = torch.zeros((), device=device)
            compat_parts: dict[str, torch.Tensor] = {}
            if cfg.compatibility_weight > 0.0:
                direction_negative = sample_direction_negatives(x_norm, clip_ids, batch_indices, schema)
                temporal_negative = None
                if skip_norm is not None:
                    skip_indices = torch.randint(0, skip_norm.shape[0], batch_indices.shape, device=device)
                    temporal_negative = skip_norm.index_select(0, skip_indices)
                compat_loss, compat_parts = compatibility_bce_loss(
                    model,
                    batch,
                    direction_negative,
                    temporal_negative,
                    cfg,
                )
            loss = recon_loss + cfg.compatibility_weight * compat_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum = loss_sum + loss.detach()
            recon_sum = recon_sum + recon_loss.detach()
            compat_sum = compat_sum + compat_loss.detach()
            for key, value in compat_parts.items():
                compat_parts_sum[key] = compat_parts_sum.get(key, torch.zeros((), device=device)) + value
            loss_count += 1
        train_loss = float((loss_sum / max(1, loss_count)).cpu())
        train_recon = float((recon_sum / max(1, loss_count)).cpu())
        train_compat = float((compat_sum / max(1, loss_count)).cpu())
        if baseline is None:
            baseline = train_loss
            target = baseline * (1.0 - cfg.target_loss_reduction)
        improved = train_loss < best - cfg.min_delta
        stalls = 0 if improved else stalls + 1
        if train_loss < best:
            best = train_loss
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(cfg),
                    "locomotion_config": asdict(locomotion_cfg),
                    "schema": schema,
                    "mean": mean.detach().cpu(),
                    "std": std.detach().cpu(),
                    "metadata": {
                        "npz_folders": [
                            {"path": str(tl.npz_folder_from_path(path)), "cyclic": cyclic}
                            for path, cyclic in clip_specs
                        ],
                        "body_names": clips[0].body_names,
                        "parents_body": clips[0].parents_body.tolist(),
                    },
                    "epoch": epoch,
                    "best": best,
                },
                ckpt_dir / "checkpoint_best.pt",
            )
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/reconstruction", train_recon, epoch)
        writer.add_scalar("loss/compatibility", train_compat, epoch)
        writer.add_scalar("loss/best", best, epoch)
        for key, value in compat_parts_sum.items():
            writer.add_scalar(f"compatibility/{key}", float((value / max(1, loss_count)).cpu()), epoch)
        if epoch == 1 or epoch % cfg.tier_eval_every_epochs == 0:
            report = tier_report(model, x_norm, schema)
            row = {"epoch": epoch, "train_loss": train_loss, "best": best, **report}
            rows.append(row)
            save_tier_report(run_dir / "tier_report.csv", rows)
            for key, value in report.items():
                writer.add_scalar(f"tiers/{key}", value, epoch)
            print(
                "epoch={:04d} loss={:.6g} best={:.6g} tiers clean={:.6g} slight={:.6g} bad={:.6g} noise={:.6g}".format(
                    epoch,
                    train_loss,
                    best,
                    report["tier1_clean_mean"],
                    report["tier2_slight_mean"],
                    report["tier3_bad_mean"],
                    report["tier4_noise_mean"],
                ),
                flush=True,
            )
        elif epoch % 10 == 0:
            print(
                f"epoch={epoch:04d} loss={train_loss:.6g} recon={train_recon:.6g} "
                f"compat={train_compat:.6g} best={best:.6g} stalls={stalls}",
                flush=True,
            )
        if target is not None and train_loss <= target:
            print(f"target reached epoch={epoch} loss={train_loss:.6g} target={target:.6g}", flush=True)
            break
        now_perf = time.perf_counter()
        if now_perf >= next_timed_checkpoint_at:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(cfg),
                    "locomotion_config": asdict(locomotion_cfg),
                    "schema": schema,
                    "mean": mean.detach().cpu(),
                    "std": std.detach().cpu(),
                    "metadata": {
                        "npz_folders": [
                            {"path": str(tl.npz_folder_from_path(path)), "cyclic": cyclic}
                            for path, cyclic in clip_specs
                        ],
                        "body_names": clips[0].body_names,
                        "parents_body": clips[0].parents_body.tolist(),
                    },
                    "epoch": epoch,
                    "best": best,
                },
                ckpt_dir / f"checkpoint_time_{stamp}_epoch_{epoch:06d}.pt",
            )
            while next_timed_checkpoint_at <= now_perf:
                next_timed_checkpoint_at += timed_interval_seconds
        if cfg.stall_patience_epochs > 0 and stalls >= cfg.stall_patience_epochs:
            print(f"stopped on stall epoch={epoch} best={best:.6g}", flush=True)
            break
    final_report = tier_report(model, x_norm, schema)
    rows.append({"epoch": epoch, "train_loss": train_loss, "best": best, **final_report})
    save_tier_report(run_dir / "tier_report.csv", rows)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    for name, field in AEConfig.__dataclass_fields__.items():
        default = field.default
        arg = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(arg, action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument(arg, type=type(default), default=None)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
