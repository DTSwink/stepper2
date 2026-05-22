from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

try:
    from .bootstrap import PROJECT_ROOT, ensure_paths
    from .naming import checkpoint_path, ik_run_id
    from . import excess_envelope as env
    from . import ik_core as tl
    from . import train_simple_ae_controller as ctl
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    from naming import checkpoint_path, ik_run_id
    import excess_envelope as env
    import ik_core as tl
    import train_simple_ae_controller as ctl

ensure_paths()


RUNS_DIR = PROJECT_ROOT / "training" / "runs"
PERIODIC_FOLDER = PROJECT_ROOT / "ue5" / "animations_omni_only_full" / "npz_final"
NONPERIODIC_FOLDER = PROJECT_ROOT / "ue5" / "animations_transitions_only_full_trimmed" / "npz_final"
SCHEDULE = (1, 2, 8, 16, 32)
LOG_EVERY_STEPS = 100
MIN_STAGE_SECONDS = {1: 45.0, 2: 45.0, 8: 90.0, 16: 120.0, 32: 180.0}
STALL_PATIENCE_LOGS = {1: 8, 2: 8, 8: 10, 16: 10, 32: 24}
FINAL_STALL_PATIENCE_LOGS = 36
MAX_STAGE_SECONDS = {1: math.inf, 2: math.inf, 8: math.inf, 16: math.inf, 32: math.inf}
MIN_DELTA_FRACTION = 1e-3
EVAL_BATCHES = 32


def latest_checkpoint_for_label(label: str, tag: str = "best") -> Path | None:
    matches = sorted(RUNS_DIR.glob(f"*_ik_{label}/checkpoints/*_{tag}.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def make_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    return device


def full_specs() -> list[tuple[Path, bool]]:
    return ctl.resolve_clip_specs(None, str(PERIODIC_FOLDER), str(NONPERIODIC_FOLDER))


def load_store_from_ae(ae_ckpt: dict, device: torch.device) -> tuple[tl.TrainConfig, ctl.SimpleClipStore]:
    cfg = ctl.make_cfg(device, ae_ckpt)
    clips = ctl.load_clips(full_specs(), cfg)
    return cfg, ctl.SimpleClipStore(clips, cfg, device)


def run_vanilla_ae(label: str) -> Path:
    existing = latest_checkpoint_for_label(label, "best")
    if existing is not None:
        print(f"reuse full AE {existing}", flush=True)
        return existing
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "training" / "ik" / "train_simple_autoencoder.py"),
        "--periodic-folder",
        str(PERIODIC_FOLDER),
        "--nonperiodic-folder",
        str(NONPERIODIC_FOLDER),
        "--run-label",
        label,
    ]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    ctl.refresh_tensorboard_async()
    found = latest_checkpoint_for_label(label, "best")
    if found is None:
        raise FileNotFoundError(f"AE training finished but no best checkpoint was found for {label}")
    return found


def pose_from_vec(vec: torch.Tensor, store: ctl.SimpleClipStore) -> dict[str, torch.Tensor]:
    pose, _ = tl.output_to_pose(vec, store.prototype)
    return pose


def fk_by_clip(
    store: ctl.SimpleClipStore,
    clip_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    pose: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_pos = torch.empty((clip_ids.shape[0], store.J, 3), dtype=root_pos.dtype, device=store.device)
    out_rot = torch.empty((clip_ids.shape[0], store.J, 3, 3), dtype=root_pos.dtype, device=store.device)
    out_canon = torch.empty((clip_ids.shape[0], store.J, 3), dtype=root_pos.dtype, device=store.device)
    for clip_id in clip_ids.unique().tolist():
        rows = (clip_ids == int(clip_id)).nonzero(as_tuple=False).flatten()
        pos, rot, canon = tl.fk_from_pose(
            store.clips[int(clip_id)],
            root_pos.index_select(0, rows),
            root_rot.index_select(0, rows),
            ctl.pose_rows(pose, rows),
            store.device,
        )
        out_pos[rows] = pos
        out_rot[rows] = rot
        out_canon[rows] = canon
    return out_pos, out_rot, out_canon


def rollout_loss(
    model: torch.nn.Module,
    ae: torch.nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    store: ctl.SimpleClipStore,
    rollout_k: int,
    batch_size: int,
    start_pools: dict[int, ctl.StartPool],
    envelope: dict[str, object] | None,
    linear_weight: float,
    angular_weight: float,
    random_init_pose: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    max_k = max(1, int(rollout_k))
    batch_size = max(1, int(batch_size))
    effective_k = ctl.sample_effective_rollout_k(batch_size, max_k, store.device)
    clip_ids, starts = ctl.sample_rollout_rows(start_pools, effective_k)
    cur_idx = starts
    prev_idx = cur_idx - 1
    state_clip_ids = clip_ids
    state_starts = starts
    if random_init_pose:
        init_pool = start_pools.get(1) or ctl.build_start_pool(store, 1)
        state_clip_ids, state_starts = ctl.sample_from_pool(init_pool, batch_size)
    prev_vec, prev_pelvis, prev_payload = ctl.target_state(store, state_clip_ids, state_starts - 1)
    cur_vec, cur_pelvis, cur_payload = ctl.target_state(store, state_clip_ids, state_starts)
    row_weight = (1.0 / effective_k.float()) / float(batch_size)
    total_loss = torch.zeros((), dtype=torch.float32, device=store.device)
    ae_total = torch.zeros_like(total_loss)
    linear_total = torch.zeros_like(total_loss)
    angular_total = torch.zeros_like(total_loss)
    active_total = torch.zeros_like(total_loss)

    for step in range(max_k):
        active = effective_k > step
        active_f = active.float()
        inp = ctl.build_controller_input(
            store, clip_ids, cur_idx, prev_vec, cur_vec, prev_pelvis, cur_pelvis, prev_payload, cur_payload
        )
        raw = ctl.model_forward(model, inp, cur_vec, store.cfg)
        pred_vec = ctl.clean_output_vector(raw, store)
        ae_rows = ctl.ae_score_rows(ae, mean, std, inp, pred_vec)
        step_rows = ae_rows
        ae_total = ae_total + (ae_rows * row_weight * active_f).sum()
        if envelope is not None and (linear_weight > 0.0 or angular_weight > 0.0):
            cur_pose = pose_from_vec(cur_vec, store)
            pred_pose = pose_from_vec(pred_vec, store)
            cur_root_pos, cur_root_rot, _cur_yaw, _cur_heading = store.root_state(clip_ids, cur_idx)
            next_root_pos, next_root_rot, _next_yaw, _next_heading = store.root_state(clip_ids, cur_idx + 1)
            cur_pos, cur_rot, _cur_canon = fk_by_clip(store, clip_ids, cur_root_pos, cur_root_rot, cur_pose)
            next_pos, next_rot, pred_canon = fk_by_clip(store, clip_ids, next_root_pos, next_root_rot, pred_pose)
            linear_rows, angular_rows = env.envelope_excess_rows(
                store,
                envelope,  # type: ignore[arg-type]
                cur_pos,
                cur_rot,
                next_pos,
                next_rot,
                clip_ids,
                cur_idx,
            )
            linear_total = linear_total + (linear_rows * row_weight * active_f).sum()
            angular_total = angular_total + (angular_rows * row_weight * active_f).sum()
            step_rows = step_rows + float(linear_weight) * linear_rows + float(angular_weight) * angular_rows
        else:
            pred_pose = None
            pred_canon = None
        total_loss = total_loss + (step_rows * row_weight * active_f).sum()
        active_total = active_total + (row_weight * active_f).sum()
        if step + 1 >= max_k:
            break

        continuing = effective_k > (step + 1)
        if envelope is not None and (linear_weight > 0.0 or angular_weight > 0.0):
            assert pred_pose is not None and pred_canon is not None
            next_pose = tl.next_pose_from_prediction(pred_pose, pred_canon)
            next_vec = tl.pose_target_output(next_pose)
            next_pelvis = next_vec[:, :3]
            next_payload = next_vec[:, ctl.payload_slice(store)]
        else:
            next_vec, next_pelvis, next_payload = ctl.predicted_state_from_vector(pred_vec, store)
        mask = continuing[:, None]
        prev_vec = torch.where(mask, cur_vec, prev_vec)
        prev_pelvis = torch.where(mask, cur_pelvis, prev_pelvis)
        prev_payload = torch.where(mask, cur_payload, prev_payload)
        cur_vec = torch.where(mask, next_vec, cur_vec)
        cur_pelvis = torch.where(mask, next_pelvis, cur_pelvis)
        cur_payload = torch.where(mask, next_payload, cur_payload)
        cur_idx = torch.where(continuing, cur_idx + 1, cur_idx)

    return total_loss, {
        "ae": ae_total,
        "linear": linear_total,
        "angular": angular_total,
        "active": active_total.clamp_min(1e-8),
    }


class EnvelopeStepper:
    kind = "eager_envelope"

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        ae: torch.nn.Module,
        mean: torch.Tensor,
        std: torch.Tensor,
        store: ctl.SimpleClipStore,
        rollout_k: int,
        batch_size: int,
        start_pools: dict[int, ctl.StartPool],
        envelope: dict[str, object] | None,
        linear_weight: float,
        angular_weight: float,
        random_init_pose: bool,
    ):
        self.model = model
        self.optimizer = optimizer
        self.ae = ae
        self.mean = mean
        self.std = std
        self.store = store
        self.rollout_k = int(rollout_k)
        self.batch_size = int(batch_size)
        self.start_pools = start_pools
        self.envelope = envelope
        self.linear_weight = float(linear_weight)
        self.angular_weight = float(angular_weight)
        self.random_init_pose = bool(random_init_pose)
        self.last_parts: dict[str, float] = {}

    def step(self) -> torch.Tensor:
        loss, parts = rollout_loss(
            self.model,
            self.ae,
            self.mean,
            self.std,
            self.store,
            self.rollout_k,
            self.batch_size,
            self.start_pools,
            self.envelope,
            self.linear_weight,
            self.angular_weight,
            self.random_init_pose,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.last_parts = {name: float(value.detach().cpu()) for name, value in parts.items()}
        return loss.detach()


@torch.no_grad()
def estimate_loss_means(
    model: torch.nn.Module,
    ae: torch.nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    store: ctl.SimpleClipStore,
    envelope: dict[str, object],
    batches: int,
) -> dict[str, float]:
    rollout_k = int(ctl.ROLLOUT_K)
    values = ctl.rollout_values_for(rollout_k)
    start_pools = ctl.build_start_pools(store, values)
    batch_size = min(int(ctl.BATCH_SIZE), int(start_pools[rollout_k].row_count))
    sums = {"ae": 0.0, "linear": 0.0, "angular": 0.0}
    model_was_training = model.training
    model.eval()
    for _ in range(max(1, int(batches))):
        _loss, parts = rollout_loss(
            model,
            ae,
            mean,
            std,
            store,
            rollout_k,
            batch_size,
            start_pools,
            envelope,
            1.0,
            1.0,
            False,
        )
        sums["ae"] += float(parts["ae"].detach().cpu())
        sums["linear"] += float(parts["linear"].detach().cpu())
        sums["angular"] += float(parts["angular"].detach().cpu())
    model.train(model_was_training)
    denom = float(max(1, int(batches)))
    return {f"mean_{name}": value / denom for name, value in sums.items()}


def load_controller_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
) -> tuple[int, float, int, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except Exception as exc:
        print(f"optimizer state not loaded from {checkpoint_path}: {exc}", flush=True)
    return int(ckpt.get("epoch", 0)), float(ckpt.get("best_val", math.inf)), int(ckpt.get("rollout_k", 0)), dict(ckpt.get("metadata", {}))


def make_stage_stepper(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ae: torch.nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    store: ctl.SimpleClipStore,
    stage_k: int,
    batch_size: int,
    start_pools: dict[int, ctl.StartPool],
    envelope: dict[str, object] | None,
    linear_weight: float,
    angular_weight: float,
    random_init_pose: bool,
) -> object:
    if envelope is None and linear_weight == 0.0 and angular_weight == 0.0 and not random_init_pose:
        return ctl.make_pure_ae_stepper(model, optimizer, ae, mean, std, store, stage_k, batch_size, start_pools)
    return EnvelopeStepper(
        model,
        optimizer,
        ae,
        mean,
        std,
        store,
        stage_k,
        batch_size,
        start_pools,
        envelope,
        linear_weight,
        angular_weight,
        random_init_pose,
    )


def train_controller_adaptive(
    label: str,
    ae_path: Path,
    device: torch.device,
    init_checkpoint: Path | None = None,
    envelope: dict[str, object] | None = None,
    linear_weight: float = 0.0,
    angular_weight: float = 0.0,
    random_init_pose: bool = False,
    start_at_k32: bool = False,
) -> Path:
    existing = latest_checkpoint_for_label(label, "last")
    if existing is not None:
        print(f"reuse controller {existing}", flush=True)
        return existing
    ae, mean, std, ae_ckpt = ctl.load_simple_ae(ae_path, device)
    cfg, store = load_store_from_ae(ae_ckpt, device)
    input_dim, output_dim = tl.make_batch_dims(store.prototype, cfg)
    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    optimizer = ctl.make_adamw(model.parameters(), ctl.LEARNING_RATE, device, capturable=bool(device.type == "cuda" and envelope is None and not random_init_pose))
    base_step = 0
    prior_metadata: dict = {}
    if init_checkpoint is not None:
        base_step, _best, loaded_k, prior_metadata = load_controller_checkpoint(model, optimizer, init_checkpoint)
        print(f"loaded init checkpoint {init_checkpoint} epoch={base_step} K={loaded_k}", flush=True)

    run_id = ik_run_id(label)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"), flush_secs=1)
    metadata = {
        "npz_paths": [str(path) for path, _cyclic in full_specs()],
        "npz_folders": [{"path": str(path.parent), "cyclic": bool(cyclic)} for path, cyclic in full_specs()],
        "simple_ae_checkpoint": str(ae_path),
        "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else "",
        "random_init_pose": bool(random_init_pose),
        "policy": {
            "loss": "simple_ae_output_reconstruction_with_optional_envelope",
            "linear_excess_loss_weight": float(linear_weight),
            "angular_excess_loss_weight": float(angular_weight),
            "pose_representation": tl.IK_POSE_REPRESENTATION,
            "adaptive_stall_curriculum": True,
        },
        "prior_metadata": prior_metadata,
    }
    (run_dir / "config.json").write_text(json.dumps({"config": asdict(cfg), "metadata": metadata}, indent=2), encoding="utf-8")
    writer.add_text("config/json", f"```json\n{json.dumps({'config': asdict(cfg), 'metadata': metadata}, indent=2)}\n```", 0)
    writer.flush()
    ctl.refresh_tensorboard_async()

    step = int(base_step)
    best_global = math.inf
    init_tag = "init_from_checkpoint" if init_checkpoint is not None else "init"
    ctl.save_controller_checkpoint(run_dir, run_id, init_tag, model, optimizer, step, best_global, 0, cfg, metadata)
    schedule = (32,) if start_at_k32 else SCHEDULE
    t0 = time.perf_counter()
    last_loss = math.inf
    final_path: Path | None = None
    for stage_i, stage_k in enumerate(schedule):
        ctl.set_optimizer_lr(optimizer, ctl.stage_learning_rate(int(stage_k)))
        rollout_values = ctl.rollout_values_for(int(stage_k)) if ctl.mixed_rollout_enabled(int(stage_k)) else (int(stage_k),)
        start_pools = ctl.build_start_pools(store, rollout_values)
        max_pool = start_pools[int(stage_k)]
        batch_size = min(int(ctl.BATCH_SIZE), int(max_pool.row_count))
        stepper = make_stage_stepper(
            model,
            optimizer,
            ae,
            mean,
            std,
            store,
            int(stage_k),
            batch_size,
            start_pools,
            envelope,
            linear_weight,
            angular_weight,
            random_init_pose,
        )
        stage_start = time.perf_counter()
        stage_logs = 0
        stalls = 0
        best_stage = math.inf
        print(f"{label}: stage K={stage_k} batch={batch_size} stepper={getattr(stepper, 'kind', 'unknown')}", flush=True)
        while True:
            block_parts: list[dict[str, float]] = []
            for _ in range(LOG_EVERY_STEPS):
                step += 1
                loss = stepper.step()  # type: ignore[attr-defined]
                last_loss = float(loss.detach().cpu())
                if hasattr(stepper, "last_parts"):
                    block_parts.append(dict(getattr(stepper, "last_parts")))
            stage_logs += 1
            improved = last_loss < best_stage * (1.0 - MIN_DELTA_FRACTION)
            if improved or not math.isfinite(best_stage):
                best_stage = last_loss
                stalls = 0
            else:
                stalls += 1
            best_global = min(best_global, last_loss)
            stats = ctl.rollout_stat_summary(batch_size, int(stage_k))
            writer.add_scalar("loss/train_total", last_loss, step)
            writer.add_scalar("loss/train_ae", last_loss, step)
            writer.add_scalar("curriculum/rollout_k", int(stage_k), step)
            writer.add_scalar("curriculum/effective_rollout_k_mean", float(stats["effective_k_mean"]), step)
            writer.add_scalar("curriculum/effective_rollout_k_max", float(stats["effective_k_max"]), step)
            writer.add_scalar("curriculum/stalls", stalls, step)
            writer.add_scalar("time/elapsed_s", time.perf_counter() - t0, step)
            if block_parts:
                raw_ae = sum(p.get("ae", 0.0) for p in block_parts) / len(block_parts)
                raw_linear = sum(p.get("linear", 0.0) for p in block_parts) / len(block_parts)
                raw_angular = sum(p.get("angular", 0.0) for p in block_parts) / len(block_parts)
                writer.add_scalar("monitor/raw_ae_loss", raw_ae, step)
                writer.add_scalar("monitor/raw_linear_excess", raw_linear, step)
                writer.add_scalar("monitor/raw_angular_excess", raw_angular, step)
                writer.add_scalar("loss/weighted_linear_excess", raw_linear * float(linear_weight), step)
                writer.add_scalar("loss/weighted_angular_excess", raw_angular * float(angular_weight), step)
            writer.flush()
            final_path = ctl.save_controller_checkpoint(run_dir, run_id, "latest", model, optimizer, step, last_loss, int(stage_k), cfg, metadata)
            elapsed_stage = time.perf_counter() - stage_start
            print(
                f"{label}: step={step} K={stage_k} loss={last_loss:.6g} best_stage={best_stage:.6g} "
                f"stalls={stalls} elapsed_stage_s={elapsed_stage:.1f}",
                flush=True,
            )
            is_final = stage_i == len(schedule) - 1
            patience = FINAL_STALL_PATIENCE_LOGS if is_final else STALL_PATIENCE_LOGS[int(stage_k)]
            min_time = MIN_STAGE_SECONDS[int(stage_k)]
            max_time = MAX_STAGE_SECONDS[int(stage_k)]
            if elapsed_stage >= min_time and stalls >= patience:
                break
            if elapsed_stage >= max_time:
                print(f"{label}: max stage time reached for K={stage_k}", flush=True)
                break
        final_path = ctl.save_controller_checkpoint(run_dir, run_id, f"stage_K{int(stage_k)}", model, optimizer, step, last_loss, int(stage_k), cfg, metadata)
        final_path = ctl.save_controller_checkpoint(run_dir, run_id, "last", model, optimizer, step, last_loss, int(stage_k), cfg, metadata)
        del stepper
    writer.close()
    assert final_path is not None
    return final_path


def compute_refinement_weights(
    baseline_checkpoint: Path,
    ae_path: Path,
    device: torch.device,
    envelope: dict[str, object],
) -> dict[str, float]:
    ae, mean, std, ae_ckpt = ctl.load_simple_ae(ae_path, device)
    cfg, store = load_store_from_ae(ae_ckpt, device)
    input_dim, output_dim = tl.make_batch_dims(store.prototype, cfg)
    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    optimizer = ctl.make_adamw(model.parameters(), ctl.LEARNING_RATE, device)
    load_controller_checkpoint(model, optimizer, baseline_checkpoint)
    means = estimate_loss_means(model, ae, mean, std, store, envelope, EVAL_BATCHES)
    mean_ae = means["mean_ae"]
    mean_linear = means["mean_linear"]
    mean_angular = means["mean_angular"]
    linear_weight = 0.0 if mean_linear <= 1e-12 else 0.1 * mean_ae / mean_linear
    angular_weight = 0.0 if mean_angular <= 1e-12 else 0.1 * mean_ae / mean_angular
    out = {
        **means,
        "linear_weight": float(linear_weight),
        "angular_weight": float(angular_weight),
    }
    print(json.dumps(out, indent=2), flush=True)
    return out


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full IK vanilla AE baseline, envelope refinement, and random-init run.")
    parser.add_argument("--phase", choices=("all", "baseline", "refine", "final"), default="all")
    parser.add_argument("--ae-label", default="full_vanilla_ae_all")
    parser.add_argument("--baseline-label", default="full_vanilla_ae_controller_baseline")
    parser.add_argument("--refined-label", default="full_vanilla_ae_controller_refined")
    parser.add_argument("--final-label", default="full_vanilla_ae_controller_random_init")
    parser.add_argument("--ae-checkpoint", default="")
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--refined-checkpoint", default="")
    parser.add_argument("--weights-json", default="")
    args = parser.parse_args()

    device = make_device()
    ae_path = Path(args.ae_checkpoint).resolve() if args.ae_checkpoint else run_vanilla_ae(args.ae_label)
    ae, _mean, _std, ae_ckpt = ctl.load_simple_ae(ae_path, device)
    del ae
    _cfg, store = load_store_from_ae(ae_ckpt, device)
    envelope = env.load_or_build_excess_envelope(store)
    sanity = env.groundtruth_sanity(store, envelope)
    print(f"envelope metadata {json.dumps(envelope['metadata'], indent=2)}", flush=True)
    print(f"envelope GT sanity {json.dumps(sanity, indent=2)}", flush=True)
    if max(sanity.values()) > 1e-6:
        raise RuntimeError(f"GT exceeds envelope unexpectedly: {sanity}")

    weights_path = RUNS_DIR / "cache" / "ik_excess_envelopes" / "latest_full_refinement_weights.json"
    baseline_path = Path(args.baseline_checkpoint).resolve() if args.baseline_checkpoint else None
    refined_path = Path(args.refined_checkpoint).resolve() if args.refined_checkpoint else None
    if args.phase in ("all", "baseline") and baseline_path is None:
        baseline_path = train_controller_adaptive(args.baseline_label, ae_path, device)
    if args.phase in ("all", "refine"):
        if baseline_path is None:
            raise ValueError("refine phase needs --baseline-checkpoint")
        weights = compute_refinement_weights(baseline_path, ae_path, device, envelope)
        save_json(weights_path, weights)
        refined_path = train_controller_adaptive(
            args.refined_label,
            ae_path,
            device,
            init_checkpoint=baseline_path,
            envelope=envelope,
            linear_weight=weights["linear_weight"],
            angular_weight=weights["angular_weight"],
            start_at_k32=True,
        )
    if args.phase in ("all", "final"):
        if refined_path is None:
            if not args.refined_checkpoint:
                raise ValueError("final phase needs --refined-checkpoint")
            refined_path = Path(args.refined_checkpoint).resolve()
        if args.weights_json:
            weights = json.loads(Path(args.weights_json).read_text(encoding="utf-8"))
        else:
            weights = json.loads(weights_path.read_text(encoding="utf-8"))
        train_controller_adaptive(
            args.final_label,
            ae_path,
            device,
            init_checkpoint=refined_path,
            envelope=envelope,
            linear_weight=float(weights["linear_weight"]),
            angular_weight=float(weights["angular_weight"]),
            random_init_pose=True,
            start_at_k32=True,
        )


if __name__ == "__main__":
    main()
