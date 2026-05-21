from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, fields
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

try:
    from .bootstrap import PROJECT_ROOT, ensure_paths
    from .naming import checkpoint_path, ik_run_id
    from . import ik_core as tl
    from . import train as supervised
    from . import train_ae_prior as rollout_data
    from .train_simple_autoencoder import SimpleAEConfig, SimpleAutoencoder
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    from naming import checkpoint_path, ik_run_id
    import ik_core as tl
    import train as supervised
    import train_ae_prior as rollout_data
    from train_simple_autoencoder import SimpleAEConfig, SimpleAutoencoder

ensure_paths()


RUNS_DIR = PROJECT_ROOT / "training" / "runs"
DEFAULT_AE_GLOB = "*_ik_simple_ae_*"
SIMPLE_CONTROLLER_AE_KIND = "simple_controller_io_autoencoder"
LEGACY_SIMPLE_AE_KIND = "simple_" + "ag" + "ent_io_autoencoder"
BATCH_SIZE = 4096
ROLLOUT_SCHEDULE = (1, 2, 4, 8, 16, 32)
ROLLOUT_STAGE_STEPS = (1000, 1000, 1500, 2000, 2500, 4000)
ROLLOUT_K = 32
LEARNING_RATE = 3e-4
LOG_EVERY = 250


def apply_config_dict(cfg: tl.TrainConfig, values: dict) -> None:
    valid = {field.name for field in fields(tl.TrainConfig)}
    for key, value in values.items():
        if key not in valid:
            continue
        current = getattr(cfg, key)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(cfg, key, value)


def latest_simple_ae_checkpoint() -> Path:
    candidates: list[Path] = []
    for run_dir in RUNS_DIR.glob(DEFAULT_AE_GLOB):
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue
        best = sorted(ckpt_dir.glob("*_best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        candidates.extend(best)
    if not candidates:
        raise FileNotFoundError(f"No simple AE checkpoints found under {RUNS_DIR}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0].resolve()


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_simple_ae(path: Path, device: torch.device) -> tuple[SimpleAutoencoder, torch.Tensor, torch.Tensor, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("kind") not in {SIMPLE_CONTROLLER_AE_KIND, LEGACY_SIMPLE_AE_KIND}:
        raise ValueError(f"Not a simple controller IO AE checkpoint: {path}")
    ae_cfg = SimpleAEConfig(**ckpt["config"])
    schema = dict(ckpt["schema"])
    model = SimpleAutoencoder(int(schema["total_dim"]), ae_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    mean = ckpt["mean"].to(device=device, dtype=torch.float32)
    std = ckpt["std"].to(device=device, dtype=torch.float32).clamp_min(1e-8)
    return model, mean, std, ckpt


def make_cfg(device: torch.device, ae_ckpt: dict) -> tl.TrainConfig:
    cfg = tl.TrainConfig()
    apply_config_dict(cfg, ae_ckpt.get("locomotion_config", {}))
    cfg.pose_representation = tl.IK_POSE_REPRESENTATION
    cfg.predict_residual = True
    cfg.zero_init_output = True
    cfg.hidden_dim = supervised.HIDDEN_DIM
    cfg.num_hidden_layers = supervised.NUM_HIDDEN_LAYERS
    cfg.learning_rate = LEARNING_RATE
    cfg.batch_size = BATCH_SIZE
    cfg.live_viewer = False
    cfg.visual_reporter = False
    cfg.update_comparison_on_exit = False
    cfg.use_torch_compile = False
    cfg.device = str(device)
    return cfg


def rollout_stage_for_step(step: int) -> tuple[int, int, int, int]:
    start = 1
    for stage_idx, (rollout_k, stage_steps) in enumerate(zip(ROLLOUT_SCHEDULE, ROLLOUT_STAGE_STEPS)):
        end = start + int(stage_steps) - 1
        if int(step) <= end:
            return stage_idx, int(rollout_k), start, end
        start = end + 1
    return len(ROLLOUT_SCHEDULE) - 1, int(ROLLOUT_SCHEDULE[-1]), start, start


def ae_score_rows(
    ae: SimpleAutoencoder,
    mean: torch.Tensor,
    std: torch.Tensor,
    controller_input: torch.Tensor,
    predicted_output: torch.Tensor,
) -> torch.Tensor:
    feature = torch.cat((controller_input, predicted_output), dim=-1)
    x = (feature - mean) / std
    recon = ae(x)
    return (recon - x).square().mean(dim=-1)


def pure_ae_rollout_loss(
    model: torch.nn.Module,
    ae: SimpleAutoencoder,
    mean: torch.Tensor,
    std: torch.Tensor,
    store: rollout_data.ClipStore,
    cfg: tl.TrainConfig,
    rollout_k: int,
    batch_size: int,
    start_pools: dict[int, supervised.StartPool],
) -> torch.Tensor:
    rollout_k = max(1, int(rollout_k))
    original_batch_size = max(1, int(batch_size))
    effective_k = supervised.sample_effective_rollout_k(original_batch_size, rollout_k, store.device)
    clip_ids, starts = supervised.sample_rollout_rows(start_pools, effective_k)
    prev_idx = starts - 1
    cur_idx = starts
    prev_vec, prev_pelvis, prev_markers = supervised.target_state(store, clip_ids, prev_idx)
    cur_vec, cur_pelvis, cur_markers = supervised.target_state(store, clip_ids, cur_idx)
    row_weight = (1.0 / effective_k.float()) / float(original_batch_size)
    total_loss = torch.zeros((), dtype=torch.float32, device=store.device)

    for step in range(rollout_k):
        inp = supervised.build_ik_input(
            store,
            clip_ids,
            cur_idx,
            prev_vec,
            cur_vec,
            prev_pelvis,
            cur_pelvis,
            prev_markers,
            cur_markers,
            cfg,
        )
        raw = supervised.model_forward(model, inp, cur_vec, cfg)
        score = ae_score_rows(ae, mean, std, inp, raw)
        total_loss = total_loss + (score * row_weight).sum()
        if step + 1 >= rollout_k:
            break
        continuing = effective_k > (step + 1)
        rows = continuing.nonzero(as_tuple=False).flatten()
        if rows.numel() == 0:
            break
        raw = raw.index_select(0, rows)
        clip_ids = clip_ids.index_select(0, rows)
        prev_vec = cur_vec.index_select(0, rows)
        prev_pelvis = cur_pelvis.index_select(0, rows)
        prev_markers = cur_markers.index_select(0, rows)
        cur_vec, cur_pelvis, cur_markers = supervised.predicted_state_from_raw(raw, store)
        prev_idx = cur_idx.index_select(0, rows)
        cur_idx = prev_idx + 1
        effective_k = effective_k.index_select(0, rows)
        row_weight = row_weight.index_select(0, rows)
    return total_loss


@torch.no_grad()
def validation_ae_score(
    model: torch.nn.Module,
    ae: SimpleAutoencoder,
    mean: torch.Tensor,
    std: torch.Tensor,
    store: rollout_data.ClipStore,
    cfg: tl.TrainConfig,
    rollout_k: int,
    pool_clip_ids: torch.Tensor,
    pool_max_starts: torch.Tensor,
) -> float:
    clip_ids, starts = supervised.validation_starts(store, supervised.VALIDATION_ROWS, pool_clip_ids, pool_max_starts)
    prev_idx = starts - 1
    cur_idx = starts
    prev_vec, prev_pelvis, prev_markers = supervised.target_state(store, clip_ids, prev_idx)
    cur_vec, cur_pelvis, cur_markers = supervised.target_state(store, clip_ids, cur_idx)
    total = 0.0
    count = 0
    for step in range(max(1, int(rollout_k))):
        inp = supervised.build_ik_input(
            store,
            clip_ids,
            cur_idx,
            prev_vec,
            cur_vec,
            prev_pelvis,
            cur_pelvis,
            prev_markers,
            cur_markers,
            cfg,
        )
        raw = supervised.model_forward(model, inp, cur_vec, cfg)
        score = ae_score_rows(ae, mean, std, inp, raw)
        total += float(score.sum().detach().cpu())
        count += int(score.numel())
        if step + 1 >= int(rollout_k):
            break
        prev_vec = cur_vec
        prev_pelvis = cur_pelvis
        prev_markers = cur_markers
        cur_vec, cur_pelvis, cur_markers = supervised.predicted_state_from_raw(raw, store)
        prev_idx = cur_idx
        cur_idx = cur_idx + 1
    return total / float(max(1, count))


def save_controller_checkpoint(
    run_dir: Path,
    run_id: str,
    tag: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best: float,
    rollout_k: int,
    cfg: tl.TrainConfig,
    metadata: dict,
) -> Path:
    path = checkpoint_path(run_dir, run_id, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tl.checkpoint_payload(model, optimizer, step, best, rollout_k, cfg, metadata), path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IK controller with only the simple AE prior loss.")
    parser.add_argument("--npz", default=None)
    parser.add_argument("--periodic-folder", default=None)
    parser.add_argument("--nonperiodic-folder", default=None)
    parser.add_argument("--ae-checkpoint", default=None)
    parser.add_argument("--run-label", default="walkF_simple_ae_controller")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    ae_path = resolve_path(args.ae_checkpoint) if args.ae_checkpoint else latest_simple_ae_checkpoint()
    ae, mean, std, ae_ckpt = load_simple_ae(ae_path, device)
    cfg = make_cfg(device, ae_ckpt)
    specs = supervised.resolve_clip_specs(args.npz, args.periodic_folder, args.nonperiodic_folder)
    clips = supervised.load_clips(specs, cfg)
    input_dim, output_dim = tl.make_batch_dims(clips[0], cfg)
    expected_dim = int(ae_ckpt["schema"]["total_dim"])
    if input_dim + output_dim != expected_dim:
        raise ValueError(f"AE dim {expected_dim} does not match controller feature dim {input_dim + output_dim}")

    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    optimizer = supervised.make_adamw(model.parameters(), LEARNING_RATE, device, weight_decay=0.0, capturable=False)
    store = rollout_data.ClipStore(clips, cfg, device)

    stage_cache: dict[int, dict[str, object]] = {}
    for stage_k in ROLLOUT_SCHEDULE:
        rollout_values = supervised.rollout_values_for(stage_k) if supervised.mixed_rollout_enabled(stage_k) else (int(stage_k),)
        start_pools = supervised.build_start_pools(store, rollout_values, require_all_clips=False)
        max_pool_clip_ids, max_pool_starts = start_pools[int(stage_k)]
        row_count = int(max_pool_starts.sum().detach().cpu())
        batch_size = min(BATCH_SIZE, row_count)
        stage_cache[int(stage_k)] = {
            "rollout_values": rollout_values,
            "start_pools": start_pools,
            "max_pool_clip_ids": max_pool_clip_ids,
            "max_pool_starts": max_pool_starts,
            "row_count": row_count,
            "batch_size": batch_size,
            "rollout_stats": supervised.rollout_stat_summary(batch_size, int(stage_k)),
        }

    run_id = ik_run_id(args.run_label)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    final_cache = stage_cache[int(ROLLOUT_K)]
    metadata = {
        "npz_paths": [str(path) for path, _cyclic in specs],
        "npz_folders": [{"path": str(path.parent), "cyclic": bool(cyclic)} for path, cyclic in specs],
        "simple_ae_checkpoint": str(ae_path),
        "tensorboard_logdir": str(run_dir / "tb"),
        "policy": {
            "loss": "pure_simple_ae_reconstruction",
            "supervised_loss_weight": 0.0,
            "pose_representation": tl.IK_POSE_REPRESENTATION,
            "mixed_rollout_at_max": True,
        },
        "rollout_schedule": [int(k) for k in ROLLOUT_SCHEDULE],
        "rollout_stage_steps": [int(n) for n in ROLLOUT_STAGE_STEPS],
        "rollout_k": int(ROLLOUT_K),
        "row_count": int(final_cache["row_count"]),
        "batch_size": int(final_cache["batch_size"]),
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
    }
    config_payload = {"config": asdict(cfg), "metadata": metadata}
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    writer = SummaryWriter(log_dir=str(run_dir / "tb"), flush_secs=1)
    writer.add_text("config/json", f"```json\n{json.dumps(config_payload, indent=2)}\n```", 0)
    print(f"pure_ae_controller run={run_id} ae={ae_path} tensorboard_logdir={run_dir / 'tb'}", flush=True)

    best = float("inf")
    init_path = save_controller_checkpoint(run_dir, run_id, "init", model, optimizer, 0, best, 0, cfg, metadata)
    print(f"saved initial checkpoint {init_path}", flush=True)
    start = time.perf_counter()
    total_steps = sum(int(x) for x in ROLLOUT_STAGE_STEPS)
    for step in range(1, total_steps + 1):
        stage_idx, stage_k, stage_start, stage_end = rollout_stage_for_step(step)
        stage_step = step - stage_start + 1
        stage_steps = stage_end - stage_start + 1
        lr = supervised.stage_learning_rate(LEARNING_RATE, stage_step, stage_steps)
        supervised.set_optimizer_lr(optimizer, lr)
        cache = stage_cache[int(stage_k)]
        model.train()
        loss = pure_ae_rollout_loss(
            model,
            ae,
            mean,
            std,
            store,
            cfg,
            int(stage_k),
            int(cache["batch_size"]),
            cache["start_pools"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step == stage_start or step % LOG_EVERY == 0 or step == total_steps:
            model.eval()
            val_ae = validation_ae_score(
                model,
                ae,
                mean,
                std,
                store,
                cfg,
                int(stage_k),
                cache["max_pool_clip_ids"],
                cache["max_pool_starts"],
            )
            mean_err, max_err = supervised.rollout_joint_error(
                model,
                store,
                cfg,
                int(stage_k),
                cache["max_pool_clip_ids"],
                cache["max_pool_starts"],
            )
            if val_ae < best:
                best = val_ae
                save_controller_checkpoint(run_dir, run_id, "best", model, optimizer, step, best, int(stage_k), cfg, metadata)
            latest = save_controller_checkpoint(
                run_dir, run_id, "latest", model, optimizer, step, best, int(stage_k), cfg, metadata
            )
            elapsed = time.perf_counter() - start
            stats = cache["rollout_stats"]
            best_log = best if best < float("inf") else val_ae
            print(
                f"step={step:05d} K={stage_k} ae_loss={float(loss.detach().cpu()):.6g} "
                f"val_ae={val_ae:.6g} best_ae={best_log:.6g} "
                f"gt_mean_m={mean_err:.6f} gt_max_m={max_err:.6f} "
                f"effK_mean={stats['effective_k_mean']:.2f} lr={lr:.3g} elapsed_s={elapsed:.1f}",
                flush=True,
            )
            writer.add_scalar("loss/train_ae", float(loss.detach().cpu()), step)
            writer.add_scalar("loss/val_ae", val_ae, step)
            writer.add_scalar("loss/best_ae", best_log, step)
            writer.add_scalar("eval/rollout_mean_m", mean_err, step)
            writer.add_scalar("eval/rollout_max_m", max_err, step)
            writer.add_scalar("curriculum/rollout_k", int(stage_k), step)
            writer.add_scalar("train/effective_rollout_k_mean", stats["effective_k_mean"], step)
            writer.add_scalar("time/elapsed_s", elapsed, step)
            writer.flush()
            print(f"checkpoint_latest={latest}", flush=True)

    last = save_controller_checkpoint(run_dir, run_id, "last", model, optimizer, total_steps, best, ROLLOUT_K, cfg, metadata)
    writer.close()
    print(f"saved {last}", flush=True)


if __name__ == "__main__":
    main()
