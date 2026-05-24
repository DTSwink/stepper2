from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

try:
    from .bootstrap import PROJECT_ROOT, ensure_paths
    from .naming import checkpoint_path, clean_label, ik_run_id
    from . import excess_envelope as env
    from . import ik_core as tl
    from . import train_simple_ae_controller as ctl
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    from naming import checkpoint_path, clean_label, ik_run_id
    import excess_envelope as env
    import ik_core as tl
    import train_simple_ae_controller as ctl

ensure_paths()


RUNS_DIR = PROJECT_ROOT / "training" / "runs"
CRASHED_RUNS_DIR = RUNS_DIR / "_crashed_ik"
PERIODIC_FOLDER = PROJECT_ROOT / "ue5" / "animations_omni_only_full" / "npz_final"
NONPERIODIC_FOLDER = PROJECT_ROOT / "ue5" / "animations_transitions_only_full_trimmed" / "npz_final"
DATASET_NPZ_TEXT: str | None = None
DATASET_PERIODIC_TEXT: str | None = str(PERIODIC_FOLDER)
DATASET_NONPERIODIC_TEXT: str | None = str(NONPERIODIC_FOLDER)
SCHEDULE = (1, 2, 8, 16, 32, 64)
LOG_EVERY_STEPS = 100
ENVELOPE_BATCH_SIZE = 1024
CONTROLLER_LOSS_SCALE = 500.0
AE_LOSS_WEIGHT = 0.19147315761749842
BASE_GRAD_CLIP_NORM = 1.0
PERIODIC_CHECKPOINT_SECONDS = 30.0 * 60.0
MIN_STAGE_SECONDS = {1: 45.0, 2: 45.0, 8: 90.0, 16: 120.0, 32: 180.0, 64: 240.0}
STALL_PATIENCE_LOGS = {1: 8, 2: 8, 8: 10, 16: 10, 32: 24, 64: 32}
FINAL_STALL_PATIENCE_LOGS = 96
MAX_STAGE_SECONDS = {1: math.inf, 2: math.inf, 8: math.inf, 16: math.inf, 32: math.inf, 64: math.inf}
MIN_DELTA_FRACTION = 5e-4
STALL_RECENT_LOGS = {1: 3, 2: 3, 8: 4, 16: 6, 32: 8, 64: 10}
STALL_LOOKBACK_LOGS = {1: 6, 2: 6, 8: 8, 16: 12, 32: 24, 64: 32}
EVAL_BATCHES = 32
OBJECTIVE_CYCLE_STEPS = 5
SUPERVISED_STEPS_PER_CYCLE = 0
SUPERVISED_K1_LOSS_WEIGHT = 9.548846199989088
FINAL_STAGE_RUNS_UNTIL_MANUAL_STOP = True


def scaled_loss(value: torch.Tensor) -> torch.Tensor:
    return value * float(CONTROLLER_LOSS_SCALE)


def controller_grad_clip_norm() -> float:
    return float(BASE_GRAD_CLIP_NORM) * float(CONTROLLER_LOSS_SCALE)


def stage_trend_improved(history: list[float], rollout_k: int) -> bool:
    recent_count = int(STALL_RECENT_LOGS[int(rollout_k)])
    left_count = int(STALL_LOOKBACK_LOGS[int(rollout_k)])
    if len(history) < recent_count + left_count:
        return True
    recent = history[-recent_count:]
    left = history[-(recent_count + left_count) : -recent_count]
    recent_mean = sum(recent) / float(len(recent))
    left_mean = sum(left) / float(len(left))
    return recent_mean < left_mean * (1.0 - float(MIN_DELTA_FRACTION))


def latest_checkpoint_for_label(label: str, tag: str = "best") -> Path | None:
    matches = sorted(
        RUNS_DIR.glob(f"*_ik_{clean_label(label)}/checkpoints/*_{tag}.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def run_dirs_for_label(label: str) -> list[Path]:
    pattern = f"*_ik_{clean_label(label)}"
    return sorted((p for p in RUNS_DIR.glob(pattern) if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)


def checkpoint_in_run(run_dir: Path, tag: str) -> Path | None:
    matches = sorted((run_dir / "checkpoints").glob(f"*_{tag}.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def archive_run_dir(run_dir: Path, reason: str) -> None:
    CRASHED_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    target = CRASHED_RUNS_DIR / run_dir.name
    if target.exists():
        target = CRASHED_RUNS_DIR / f"{run_dir.name}_{int(time.time())}"
    status = {
        "state": "archived",
        "reason": reason,
        "source": str(run_dir),
        "archived_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    (run_dir / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    shutil.move(str(run_dir), str(target))
    print(f"archived incomplete run {run_dir.name} -> {target.name}: {reason}", flush=True)


def clean_controller_run_state(label: str) -> Path | None:
    """Return a resume checkpoint and move no-progress crashed runs out of TensorBoard."""
    run_dirs = run_dirs_for_label(label)
    completed = next((checkpoint_in_run(run_dir, "last") for run_dir in run_dirs if checkpoint_in_run(run_dir, "last") is not None), None)
    if completed is not None:
        for run_dir in run_dirs:
            if checkpoint_in_run(run_dir, "last") is None:
                archive_run_dir(run_dir, "superseded incomplete run; completed checkpoint already exists")
        return completed

    resume: Path | None = None
    for run_dir in run_dirs:
        last = checkpoint_in_run(run_dir, "last")
        if last is not None:
            continue
        latest = checkpoint_in_run(run_dir, "latest")
        if latest is not None and resume is None:
            resume = latest
            continue
        if latest is None:
            archive_run_dir(run_dir, "no latest/last checkpoint; restarting this phase cleanly")
    return completed or resume


def write_run_status(run_dir: Path, payload: dict) -> None:
    data = {
        **payload,
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "pid": os.getpid(),
    }
    (run_dir / "run_status.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def make_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    return device


def configure_dataset(npz_text: str | None, periodic_text: str | None, nonperiodic_text: str | None) -> None:
    global DATASET_NPZ_TEXT, DATASET_PERIODIC_TEXT, DATASET_NONPERIODIC_TEXT
    if npz_text or periodic_text or nonperiodic_text:
        DATASET_NPZ_TEXT = npz_text or None
        DATASET_PERIODIC_TEXT = periodic_text or None
        DATASET_NONPERIODIC_TEXT = nonperiodic_text or None
    else:
        DATASET_NPZ_TEXT = None
        DATASET_PERIODIC_TEXT = str(PERIODIC_FOLDER)
        DATASET_NONPERIODIC_TEXT = str(NONPERIODIC_FOLDER)


def full_specs() -> list[tuple[Path, bool]]:
    return ctl.resolve_clip_specs(DATASET_NPZ_TEXT, DATASET_PERIODIC_TEXT, DATASET_NONPERIODIC_TEXT)


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


def graph_root_cycle_count(store: ctl.SimpleClipStore, rollout_k: int) -> int:
    if not bool(store.cyclic.any().detach().cpu()):
        return 1
    cyclic_periods = store.periods[store.cyclic].detach().cpu()
    min_period = max(1, int(cyclic_periods.min().item()))
    return max(1, int(math.ceil((int(rollout_k) + 1) / float(min_period))) + 1)


def root_state_fixed_cycles(
    store: ctl.SimpleClipStore,
    clip_ids: torch.Tensor,
    idx: torch.Tensor,
    max_cycles: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    frame = store.frame_index(clip_ids, idx)
    base_pos = store.root_pos.index_select(0, frame)
    base_rot = store.root_rot.index_select(0, frame)
    root0_pos = store.root0_pos.index_select(0, clip_ids)
    root0_rot = store.root0_rot.index_select(0, clip_ids)
    root0_inv = store.root0_inv.index_select(0, clip_ids)
    periods = store.periods.index_select(0, clip_ids).clamp_min(1)
    cyclic = store.cyclic.index_select(0, clip_ids)
    cycles = torch.where(cyclic, torch.div(idx, periods, rounding_mode="floor"), torch.zeros_like(idx))

    rel_pos = torch.matmul((base_pos - root0_pos).unsqueeze(1), root0_inv).squeeze(1)
    rel_rot = base_rot @ root0_inv
    cycle_pos = store.cycle_pos.index_select(0, clip_ids)
    cycle_rot = store.cycle_rot.index_select(0, clip_ids)
    for cycle in range(max(1, int(max_cycles))):
        mask = (cycles > cycle).reshape(-1, 1)
        next_pos = torch.matmul(rel_pos.unsqueeze(1), cycle_rot).squeeze(1) + cycle_pos
        next_rot = rel_rot @ cycle_rot
        rel_pos = torch.where(mask, next_pos, rel_pos)
        rel_rot = torch.where(mask.unsqueeze(-1), next_rot, rel_rot)
    return torch.matmul(rel_pos.unsqueeze(1), root0_rot).squeeze(1) + root0_pos, rel_rot @ root0_rot


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
    pose_noise_amount: float = 0.0,
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
    if pose_noise_amount > 0.0:
        cur_vec = ctl.add_pose_noise_to_vector(store, cur_vec, pose_noise_amount)
        cur_vec, cur_pelvis, cur_payload = ctl.predicted_state_from_vector(cur_vec, store)
    row_weight = (1.0 / effective_k.float()) / float(batch_size)
    total_loss = torch.zeros((), dtype=torch.float32, device=store.device)
    ae_total = torch.zeros_like(total_loss)
    linear_total = torch.zeros_like(total_loss)
    angular_total = torch.zeros_like(total_loss)
    active_total = torch.zeros_like(total_loss)
    has_envelope_loss = envelope is not None and (linear_weight > 0.0 or angular_weight > 0.0)
    carried_cur_foot_pos: torch.Tensor | None = None
    carried_cur_foot_rot: torch.Tensor | None = None
    carried_cur_root_pos: torch.Tensor | None = None
    carried_cur_root_rot: torch.Tensor | None = None

    for step in range(max_k):
        active = effective_k > step
        inp = ctl.build_controller_input(
            store, clip_ids, cur_idx, prev_vec, cur_vec, prev_pelvis, cur_pelvis, prev_payload, cur_payload
        )
        raw = ctl.model_forward(model, inp, cur_vec, store.cfg)
        pred_vec = ctl.clean_output_vector(raw, store)
        ae_rows = ctl.ae_score_rows(ae, mean, std, inp, pred_vec)
        ae_rows = ae_rows * float(AE_LOSS_WEIGHT)
        step_rows = ae_rows
        active_f = active.float()
        ae_total = ae_total + (ae_rows * row_weight * active_f).sum()
        if has_envelope_loss:
            if carried_cur_root_pos is None or carried_cur_root_rot is None:
                cur_root_pos, cur_root_rot, _cur_yaw, _cur_heading = store.root_state(clip_ids, cur_idx)
            else:
                cur_root_pos = carried_cur_root_pos
                cur_root_rot = carried_cur_root_rot
            next_root_pos, next_root_rot, _next_yaw, _next_heading = store.root_state(clip_ids, cur_idx + 1)
            if carried_cur_foot_pos is None or carried_cur_foot_rot is None:
                cur_foot_pos, cur_foot_rot = env.ik_foot_toe_state_from_vec(store, cur_root_pos, cur_root_rot, cur_vec)
            else:
                cur_foot_pos = carried_cur_foot_pos
                cur_foot_rot = carried_cur_foot_rot
            next_foot_pos, next_foot_rot = env.ik_foot_toe_state_from_vec(store, next_root_pos, next_root_rot, pred_vec)
            linear_rows, angular_rows = env.envelope_excess_ik_state_rows(
                store,
                envelope,  # type: ignore[arg-type]
                cur_foot_pos,
                cur_foot_rot,
                next_foot_pos,
                next_foot_rot,
                clip_ids,
                cur_idx,
            )
            linear_total = linear_total + (linear_rows * row_weight * active_f).sum()
            angular_total = angular_total + (angular_rows * row_weight * active_f).sum()
            step_rows = step_rows + float(linear_weight) * linear_rows + float(angular_weight) * angular_rows
        total_loss = total_loss + (step_rows * row_weight * active_f).sum()
        active_total = active_total + (row_weight * active_f).sum()
        if step + 1 >= max_k:
            break

        continuing = effective_k > (step + 1)
        rows = continuing.nonzero(as_tuple=False).flatten()
        if rows.numel() == 0:
            break
        selected_clip_ids = clip_ids.index_select(0, rows)
        selected_cur_idx = cur_idx.index_select(0, rows)
        reset = ctl.training_reset_rows(
            store,
            selected_clip_ids,
            selected_cur_idx,
            torch.ones_like(selected_cur_idx, dtype=torch.bool),
        )
        next_vec, next_pelvis, next_payload = ctl.predicted_state_from_vector(pred_vec.index_select(0, rows), store)
        reset_starts = ctl.sample_same_clip_training_starts(store, selected_clip_ids)
        reset_prev_vec, reset_prev_pelvis, reset_prev_payload = ctl.target_state(store, selected_clip_ids, reset_starts - 1)
        reset_cur_vec, reset_cur_pelvis, reset_cur_payload = ctl.target_state(store, selected_clip_ids, reset_starts)
        if pose_noise_amount > 0.0:
            reset_cur_vec = ctl.add_pose_noise_to_vector(store, reset_cur_vec, pose_noise_amount)
            reset_cur_vec, reset_cur_pelvis, reset_cur_payload = ctl.predicted_state_from_vector(reset_cur_vec, store)
        reset_mask = reset[:, None]
        clip_ids = selected_clip_ids
        prev_vec = torch.where(reset_mask, reset_prev_vec, cur_vec.index_select(0, rows))
        prev_pelvis = torch.where(reset_mask, reset_prev_pelvis, cur_pelvis.index_select(0, rows))
        prev_payload = torch.where(reset_mask, reset_prev_payload, cur_payload.index_select(0, rows))
        cur_vec = torch.where(reset_mask, reset_cur_vec, next_vec)
        cur_pelvis = torch.where(reset_mask, reset_cur_pelvis, next_pelvis)
        cur_payload = torch.where(reset_mask, reset_cur_payload, next_payload)
        cur_idx = torch.where(reset, reset_starts, selected_cur_idx + 1)
        effective_k = effective_k.index_select(0, rows)
        row_weight = row_weight.index_select(0, rows)
        carried_cur_foot_pos = None
        carried_cur_foot_rot = None
        carried_cur_root_pos = None
        carried_cur_root_rot = None

    return scaled_loss(total_loss), {
        "ae": ae_total,
        "linear": linear_total,
        "angular": angular_total,
        "active": active_total.clamp_min(1e-8),
    }


def supervised_k1_loss(
    model: torch.nn.Module,
    store: ctl.SimpleClipStore,
    batch_size: int,
    start_pool: ctl.StartPool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size = max(1, int(batch_size))
    clip_ids, cur_idx = ctl.sample_from_pool(start_pool, batch_size)
    prev_vec, prev_pelvis, prev_payload = ctl.target_state(store, clip_ids, cur_idx - 1)
    cur_vec, cur_pelvis, cur_payload = ctl.target_state(store, clip_ids, cur_idx)
    inp = ctl.build_controller_input(
        store,
        clip_ids,
        cur_idx,
        prev_vec,
        cur_vec,
        prev_pelvis,
        cur_pelvis,
        prev_payload,
        cur_payload,
    )
    raw = ctl.model_forward(model, inp, cur_vec, store.cfg)
    pred_vec = ctl.clean_output_vector(raw, store)
    target = store.get_target_output(clip_ids, cur_idx + 1)
    supervised = (pred_vec - target).square().mean(dim=-1).mean() * float(SUPERVISED_K1_LOSS_WEIGHT)
    return scaled_loss(supervised), {"supervised": supervised}


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
        pose_noise_amount: float = 0.0,
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
        self.pose_noise_amount = float(pose_noise_amount)
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
            self.pose_noise_amount,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), controller_grad_clip_norm())
        self.optimizer.step()
        self.last_parts = {name: value.detach() for name, value in parts.items()}
        return loss.detach()


def rollout_loss_static(
    model: torch.nn.Module,
    ae: torch.nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    store: ctl.SimpleClipStore,
    rollout_k: int,
    batch_size: int,
    effective_k: torch.Tensor,
    clip_ids: torch.Tensor,
    starts: torch.Tensor,
    envelope: dict[str, object],
    linear_weight: float,
    angular_weight: float,
    root_cycle_count: int,
    reset_starts_by_step: torch.Tensor | None = None,
    start_cur_vec_override: torch.Tensor | None = None,
    reset_cur_vec_by_step: torch.Tensor | None = None,
    pose_noise_amount: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_k = max(1, int(rollout_k))
    cur_idx = starts
    prev_idx = cur_idx - 1
    prev_vec, prev_pelvis, prev_payload = ctl.target_state(store, clip_ids, prev_idx)
    cur_vec, cur_pelvis, cur_payload = ctl.target_state(store, clip_ids, cur_idx)
    if start_cur_vec_override is not None:
        cur_vec = start_cur_vec_override
        cur_vec, cur_pelvis, cur_payload = ctl.predicted_state_from_vector(cur_vec, store)
    elif pose_noise_amount > 0.0:
        cur_vec = ctl.add_pose_noise_to_vector(store, cur_vec, pose_noise_amount)
        cur_vec, cur_pelvis, cur_payload = ctl.predicted_state_from_vector(cur_vec, store)
    row_weight = (1.0 / effective_k.float()) / float(max(1, int(batch_size)))
    total_loss = torch.zeros((), dtype=torch.float32, device=store.device)
    ae_total = torch.zeros_like(total_loss)
    linear_total = torch.zeros_like(total_loss)
    angular_total = torch.zeros_like(total_loss)
    active_total = torch.zeros_like(total_loss)
    cur_root_pos, cur_root_rot = root_state_fixed_cycles(store, clip_ids, cur_idx, root_cycle_count)
    cur_foot_pos, cur_foot_rot = env.ik_foot_toe_state_from_vec(store, cur_root_pos, cur_root_rot, cur_vec)

    for step in range(max_k):
        active = effective_k > step
        active_f = active.float()
        inp = ctl.build_controller_input(
            store, clip_ids, cur_idx, prev_vec, cur_vec, prev_pelvis, cur_pelvis, prev_payload, cur_payload
        )
        raw = ctl.model_forward(model, inp, cur_vec, store.cfg)
        pred_vec = ctl.clean_output_vector(raw, store)
        ae_rows = ctl.ae_score_rows(ae, mean, std, inp, pred_vec)
        ae_rows = ae_rows * float(AE_LOSS_WEIGHT)
        next_root_pos, next_root_rot = root_state_fixed_cycles(store, clip_ids, cur_idx + 1, root_cycle_count)
        next_foot_pos, next_foot_rot = env.ik_foot_toe_state_from_vec(store, next_root_pos, next_root_rot, pred_vec)
        linear_rows, angular_rows = env.envelope_excess_ik_state_rows(
            store,
            envelope,
            cur_foot_pos,
            cur_foot_rot,
            next_foot_pos,
            next_foot_rot,
            clip_ids,
            cur_idx,
        )
        ae_total = ae_total + (ae_rows * row_weight * active_f).sum()
        linear_loss = (linear_rows * row_weight * active_f).sum()
        angular_loss = (angular_rows * row_weight * active_f).sum()
        linear_total = linear_total + linear_loss
        angular_total = angular_total + angular_loss
        total_loss = total_loss + (
            (ae_rows * row_weight * active_f).sum()
            + float(linear_weight) * linear_loss
            + float(angular_weight) * angular_loss
        )
        active_total = active_total + (row_weight * active_f).sum()
        if step + 1 >= max_k:
            break

        continuing = effective_k > (step + 1)
        next_vec, next_pelvis, next_payload = ctl.predicted_state_from_vector(pred_vec, store)
        reset = ctl.training_reset_rows(store, clip_ids, cur_idx, continuing)
        advance = continuing & (~reset)
        reset_starts = (
            reset_starts_by_step[step]
            if reset_starts_by_step is not None
            else ctl.sample_same_clip_training_starts(store, clip_ids)
        )
        reset_prev_vec, reset_prev_pelvis, reset_prev_payload = ctl.target_state(store, clip_ids, reset_starts - 1)
        reset_cur_vec, reset_cur_pelvis, reset_cur_payload = ctl.target_state(store, clip_ids, reset_starts)
        if reset_cur_vec_by_step is not None:
            reset_cur_vec = reset_cur_vec_by_step[step]
            reset_cur_vec, reset_cur_pelvis, reset_cur_payload = ctl.predicted_state_from_vector(reset_cur_vec, store)
        elif pose_noise_amount > 0.0:
            reset_cur_vec = ctl.add_pose_noise_to_vector(store, reset_cur_vec, pose_noise_amount)
            reset_cur_vec, reset_cur_pelvis, reset_cur_payload = ctl.predicted_state_from_vector(reset_cur_vec, store)
        reset_root_pos, reset_root_rot = root_state_fixed_cycles(store, clip_ids, reset_starts, root_cycle_count)
        reset_foot_pos, reset_foot_rot = env.ik_foot_toe_state_from_vec(store, reset_root_pos, reset_root_rot, reset_cur_vec)
        reset_mask = reset[:, None]
        advance_mask = advance[:, None]
        reset_mask_3 = reset[:, None, None]
        advance_mask_3 = advance[:, None, None]
        reset_mask_rot = reset[:, None, None, None]
        advance_mask_rot = advance[:, None, None, None]
        prev_vec = torch.where(reset_mask, reset_prev_vec, torch.where(advance_mask, cur_vec, prev_vec))
        prev_pelvis = torch.where(reset_mask, reset_prev_pelvis, torch.where(advance_mask, cur_pelvis, prev_pelvis))
        prev_payload = torch.where(reset_mask, reset_prev_payload, torch.where(advance_mask, cur_payload, prev_payload))
        cur_vec = torch.where(reset_mask, reset_cur_vec, torch.where(advance_mask, next_vec, cur_vec))
        cur_pelvis = torch.where(reset_mask, reset_cur_pelvis, torch.where(advance_mask, next_pelvis, cur_pelvis))
        cur_payload = torch.where(reset_mask, reset_cur_payload, torch.where(advance_mask, next_payload, cur_payload))
        cur_idx = torch.where(reset, reset_starts, torch.where(continuing, cur_idx + 1, cur_idx))
        cur_root_pos = torch.where(reset_mask, reset_root_pos, torch.where(advance_mask, next_root_pos, cur_root_pos))
        cur_root_rot = torch.where(reset_mask_3, reset_root_rot, torch.where(advance_mask_3, next_root_rot, cur_root_rot))
        cur_foot_pos = torch.where(reset_mask_3, reset_foot_pos, torch.where(advance_mask_3, next_foot_pos, cur_foot_pos))
        cur_foot_rot = torch.where(reset_mask_rot, reset_foot_rot, torch.where(advance_mask_rot, next_foot_rot, cur_foot_rot))
    return scaled_loss(total_loss), ae_total, linear_total, angular_total, active_total.clamp_min(1e-8)


class CudaGraphEnvelopeStep:
    kind = "cuda_graph_static_envelope"

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
        envelope: dict[str, object],
        linear_weight: float,
        angular_weight: float,
        pose_noise_amount: float = 0.0,
    ):
        if store.device.type != "cuda":
            raise RuntimeError("CudaGraphEnvelopeStep requires CUDA")
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
        self.pose_noise_amount = float(pose_noise_amount)
        self.root_cycle_count = graph_root_cycle_count(store, rollout_k)
        self.vec_dim = 3 + 6 + store.Jcore * 6 + store.ik_payload_dim
        self.effective_k = torch.empty((self.batch_size,), dtype=torch.long, device=store.device)
        self.clip_ids = torch.empty_like(self.effective_k)
        self.starts = torch.empty_like(self.effective_k)
        self.reset_starts_by_step = torch.empty(
            (max(1, int(self.rollout_k)), self.batch_size), dtype=torch.long, device=store.device
        )
        self.start_cur_vec = torch.empty((self.batch_size, self.vec_dim), dtype=torch.float32, device=store.device)
        self.reset_cur_vec_by_step = torch.empty(
            (max(1, int(self.rollout_k)), self.batch_size, self.vec_dim), dtype=torch.float32, device=store.device
        )
        self.loss = torch.zeros((), dtype=torch.float32, device=store.device)
        self.ae_part = torch.zeros_like(self.loss)
        self.linear_part = torch.zeros_like(self.loss)
        self.angular_part = torch.zeros_like(self.loss)
        self.active_part = torch.zeros_like(self.loss)
        self.last_parts: dict[str, torch.Tensor] = {}
        self.graph = torch.cuda.CUDAGraph()
        self._capture()

    def _sample_into_static_buffers(self) -> None:
        effective_k = ctl.sample_effective_rollout_k(self.batch_size, self.rollout_k, self.store.device)
        clip_ids, starts = ctl.sample_rollout_rows(self.start_pools, effective_k)
        self.effective_k.copy_(effective_k)
        self.clip_ids.copy_(clip_ids)
        self.starts.copy_(starts)
        self.reset_starts_by_step.copy_(
            ctl.sample_same_clip_training_starts_by_step(self.store, clip_ids, self.reset_starts_by_step.shape[0])
        )
        if self.pose_noise_amount > 0.0:
            cur_vec = self.store.get_target_output(clip_ids, starts)
            self.start_cur_vec.copy_(ctl.add_pose_noise_to_vector(self.store, cur_vec, self.pose_noise_amount))
            flat_clip_ids = clip_ids.unsqueeze(0).expand(self.reset_starts_by_step.shape[0], -1).reshape(-1)
            flat_starts = self.reset_starts_by_step.reshape(-1)
            reset_cur_vec = self.store.get_target_output(flat_clip_ids, flat_starts)
            reset_cur_vec = ctl.add_pose_noise_to_vector(self.store, reset_cur_vec, self.pose_noise_amount)
            self.reset_cur_vec_by_step.copy_(reset_cur_vec.reshape_as(self.reset_cur_vec_by_step))

    def _loss_parts(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return rollout_loss_static(
            self.model,
            self.ae,
            self.mean,
            self.std,
            self.store,
            self.rollout_k,
            self.batch_size,
            self.effective_k,
            self.clip_ids,
            self.starts,
            self.envelope,
            self.linear_weight,
            self.angular_weight,
            self.root_cycle_count,
            self.reset_starts_by_step,
            self.start_cur_vec if self.pose_noise_amount > 0.0 else None,
            self.reset_cur_vec_by_step if self.pose_noise_amount > 0.0 else None,
            self.pose_noise_amount,
        )

    def _capture(self) -> None:
        self._sample_into_static_buffers()
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(2):
                self._sample_into_static_buffers()
                self.optimizer.zero_grad(set_to_none=False)
                loss, _ae, _linear, _angular, _active = self._loss_parts()
                loss.backward()
                self.optimizer.step()
                del loss, _ae, _linear, _angular, _active
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()
        self.optimizer.zero_grad(set_to_none=False)
        with torch.cuda.graph(self.graph):
            self.optimizer.zero_grad(set_to_none=False)
            self.loss, self.ae_part, self.linear_part, self.angular_part, self.active_part = self._loss_parts()
            self.loss.backward()
            self.optimizer.step()

    def step(self) -> torch.Tensor:
        self._sample_into_static_buffers()
        self.graph.replay()
        self.last_parts = {
            "ae": self.ae_part.detach(),
            "linear": self.linear_part.detach(),
            "angular": self.angular_part.detach(),
            "active": self.active_part.detach(),
        }
        return self.loss.detach()


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
    start_pools = ctl.build_training_start_pools(store, values)
    batch_size = min(int(ctl.BATCH_SIZE), ENVELOPE_BATCH_SIZE, int(start_pools[rollout_k].row_count))
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
    pose_noise_amount: float = 0.0,
    disable_cuda_graph: bool = False,
) -> object:
    if envelope is None and linear_weight == 0.0 and angular_weight == 0.0 and not random_init_pose and pose_noise_amount <= 0.0:
        if disable_cuda_graph:
            return ctl.EagerPureAEStep(model, optimizer, ae, mean, std, store, stage_k, batch_size, start_pools)
        return ctl.make_pure_ae_stepper(model, optimizer, ae, mean, std, store, stage_k, batch_size, start_pools)
    if (
        envelope is not None
        and (linear_weight > 0.0 or angular_weight > 0.0)
        and not random_init_pose
        and store.device.type == "cuda"
        and not disable_cuda_graph
    ):
        return CudaGraphEnvelopeStep(
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
            pose_noise_amount,
        )
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
        pose_noise_amount,
    )


class SupervisedK1Stepper:
    kind = "eager_supervised_k1"

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        store: ctl.SimpleClipStore,
        batch_size: int,
        start_pool: ctl.StartPool,
    ):
        self.model = model
        self.optimizer = optimizer
        self.store = store
        self.batch_size = int(batch_size)
        self.start_pool = start_pool
        self.last_parts: dict[str, torch.Tensor] = {}

    def step(self) -> torch.Tensor:
        loss, parts = supervised_k1_loss(self.model, self.store, self.batch_size, self.start_pool)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), controller_grad_clip_norm())
        self.optimizer.step()
        self.last_parts = {name: value.detach() for name, value in parts.items()}
        return loss.detach()


class MixedObjectiveStepper:
    kind = "mixed_objective"

    def __init__(
        self,
        primary_stepper: object,
        supervised_stepper: SupervisedK1Stepper,
        supervised_steps_per_cycle: int,
        objective_cycle_steps: int,
    ):
        self.primary_stepper = primary_stepper
        self.supervised_stepper = supervised_stepper
        self.supervised_steps_per_cycle = max(0, int(supervised_steps_per_cycle))
        self.objective_cycle_steps = max(1, int(objective_cycle_steps))
        if self.supervised_steps_per_cycle > self.objective_cycle_steps:
            raise ValueError("supervised_steps_per_cycle cannot exceed objective_cycle_steps")
        self.last_parts: dict[str, torch.Tensor | float] = {}
        self.last_mode = "primary"

    def use_supervised(self, step: int) -> bool:
        if self.supervised_steps_per_cycle <= 0:
            return False
        slot = (max(1, int(step)) - 1) % self.objective_cycle_steps
        return slot < self.supervised_steps_per_cycle

    def step(self, step: int) -> torch.Tensor:
        if self.use_supervised(step):
            self.last_mode = "supervised"
            loss = self.supervised_stepper.step()
            self.last_parts = self.supervised_stepper.last_parts
            return loss
        self.last_mode = "primary"
        loss = self.primary_stepper.step()  # type: ignore[attr-defined]
        self.last_parts = dict(getattr(self.primary_stepper, "last_parts", {}))
        if not self.last_parts:
            kind = str(getattr(self.primary_stepper, "kind", ""))
            pure_ae_raw_loss = kind in {"cuda_graph_static_masked", "eager"}
            ae_part = loss.detach() if pure_ae_raw_loss else loss.detach() / float(CONTROLLER_LOSS_SCALE)
            self.last_parts = {"ae": ae_part}
        return loss


def log_loss_scalars(
    writer: SummaryWriter,
    step: int,
    parts: dict[str, float],
    linear_weight: float,
    angular_weight: float,
) -> None:
    scale = float(CONTROLLER_LOSS_SCALE)
    writer.add_scalar("loss/ae_score", float(parts.get("ae", 0.0)) * scale, int(step))
    writer.add_scalar("loss/linear_slide_weighted", float(parts.get("linear", 0.0)) * float(linear_weight) * scale, int(step))
    writer.add_scalar("loss/angular_slide_weighted", float(parts.get("angular", 0.0)) * float(angular_weight) * scale, int(step))
    writer.add_scalar("loss/supervised", float(parts.get("supervised", 0.0)) * scale, int(step))


def tensor_to_float(value: object) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


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
    supervised_steps_per_cycle: int = SUPERVISED_STEPS_PER_CYCLE,
    objective_cycle_steps: int = OBJECTIVE_CYCLE_STEPS,
    pose_noise_amount: float = 0.0,
    disable_cuda_graph: bool = False,
) -> Path:
    controller_state = clean_controller_run_state(label)
    if controller_state is not None:
        if controller_state.name.endswith("_last.pt"):
            print(f"reuse controller {controller_state}", flush=True)
            return controller_state
        init_checkpoint = controller_state
        print(f"resume partial controller from {controller_state}", flush=True)
    ae, mean, std, ae_ckpt = ctl.load_simple_ae(ae_path, device)
    cfg, store = load_store_from_ae(ae_ckpt, device)
    input_dim, output_dim = tl.make_batch_dims(store.prototype, cfg)
    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    optimizer = ctl.make_adamw(
        model.parameters(),
        ctl.LEARNING_RATE,
        device,
        capturable=bool(device.type == "cuda" and not random_init_pose and not disable_cuda_graph),
    )
    base_step = 0
    loaded_k = 0
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
            "controller_loss_scale": float(CONTROLLER_LOSS_SCALE),
            "ae_loss_weight": float(AE_LOSS_WEIGHT),
            "linear_excess_loss_weight": float(linear_weight),
            "angular_excess_loss_weight": float(angular_weight),
            "supervised_steps_per_cycle": int(supervised_steps_per_cycle),
            "objective_cycle_steps": int(objective_cycle_steps),
            "supervised_rollout_k": 1,
            "supervised_k1_loss_weight": float(SUPERVISED_K1_LOSS_WEIGHT),
            "pose_noise_amount": float(pose_noise_amount),
            "pose_noise_recipe": {
                "pos_sigma_m_at_1": float(ctl.POSE_NOISE_POS_SIGMA_M_AT_1),
                "rot_sigma_deg_at_1": float(ctl.POSE_NOISE_ROT_SIGMA_DEG_AT_1),
                "pole_toe_sigma_at_1": float(ctl.POSE_NOISE_SCALAR_SIGMA_AT_1),
            },
            "pose_representation": tl.IK_POSE_REPRESENTATION,
            "adaptive_stall_curriculum": True,
            "final_stage_runs_until_manual_stop": bool(FINAL_STAGE_RUNS_UNTIL_MANUAL_STOP),
            "disable_cuda_graph": bool(disable_cuda_graph),
        },
        "training_constants": {
            "schedule": [int(k) for k in SCHEDULE],
            "max_mixed_rollout_k": int(ctl.ROLLOUT_K),
            "stage_learning_rates": {str(int(k)): float(ctl.stage_learning_rate(int(k))) for k in SCHEDULE},
            "dynamic_start_rule": "training rows need one-step starts; noncyclic rows reset randomly inside the same clip when they reach the usable end",
            "periodic_checkpoint_seconds": float(PERIODIC_CHECKPOINT_SECONDS),
        },
        "prior_metadata": prior_metadata,
    }
    (run_dir / "config.json").write_text(json.dumps({"config": asdict(cfg), "metadata": metadata}, indent=2), encoding="utf-8")
    writer.add_text("config/json", f"```json\n{json.dumps({'config': asdict(cfg), 'metadata': metadata}, indent=2)}\n```", 0)
    writer.flush()
    ctl.refresh_tensorboard_async()

    step = int(base_step)
    writer.flush()
    ctl.refresh_tensorboard_async()
    write_run_status(run_dir, {"state": "running", "label": label, "init_checkpoint": str(init_checkpoint or "")})
    best_global = math.inf
    init_tag = "init_from_checkpoint" if init_checkpoint is not None else "init"
    ctl.save_controller_checkpoint(run_dir, run_id, init_tag, model, optimizer, step, best_global, 0, cfg, metadata)
    if start_at_k32:
        schedule = (int(SCHEDULE[-1]),)
    else:
        min_stage = max(1, int(loaded_k or 1))
        schedule = tuple(k for k in SCHEDULE if int(k) >= min_stage) or (int(SCHEDULE[-1]),)
    last_loss = math.inf
    final_path: Path | None = None
    last_periodic_checkpoint_time = time.perf_counter()
    for stage_i, stage_k in enumerate(schedule):
        ctl.set_optimizer_lr(optimizer, ctl.stage_learning_rate(int(stage_k)))
        rollout_values = ctl.rollout_values_for(int(stage_k)) if ctl.mixed_rollout_enabled(int(stage_k)) else (int(stage_k),)
        start_pools = ctl.build_training_start_pools(store, rollout_values)
        max_pool = start_pools[int(stage_k)]
        batch_size = min(int(ctl.BATCH_SIZE), int(max_pool.row_count))
        if envelope is not None and (linear_weight > 0.0 or angular_weight > 0.0):
            batch_size = min(batch_size, ENVELOPE_BATCH_SIZE)
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
            pose_noise_amount,
            disable_cuda_graph,
        )
        if int(supervised_steps_per_cycle) > 0:
            supervised_pool = ctl.build_start_pool(store, 1)
            supervised_batch = min(batch_size, int(supervised_pool.row_count))
            supervised_stepper = SupervisedK1Stepper(model, optimizer, store, supervised_batch, supervised_pool)
            stepper = MixedObjectiveStepper(
                stepper,
                supervised_stepper,
                int(supervised_steps_per_cycle),
                int(objective_cycle_steps),
            )
        stage_start = time.perf_counter()
        stage_logs = 0
        stalls = 0
        best_stage = math.inf
        stage_loss_history: list[float] = []
        print(f"{label}: stage K={stage_k} batch={batch_size} stepper={getattr(stepper, 'kind', 'unknown')}", flush=True)
        while True:
            chunk_parts = {"ae": 0.0, "linear": 0.0, "angular": 0.0, "supervised": 0.0}
            chunk_counts = {"ae": 0, "linear": 0, "angular": 0, "supervised": 0}
            chunk_loss = 0.0
            for _ in range(LOG_EVERY_STEPS):
                step += 1
                if isinstance(stepper, MixedObjectiveStepper):
                    loss = stepper.step(step)
                else:
                    loss = stepper.step()  # type: ignore[attr-defined]
                last_loss = float(loss.detach().cpu())
                chunk_loss += last_loss
                parts = dict(getattr(stepper, "last_parts")) if hasattr(stepper, "last_parts") else {}
                if not parts:
                    pure_ae_stepper = str(getattr(stepper, "kind", "")) in {"cuda_graph_static_masked", "eager"}
                    ae_part = last_loss * float(AE_LOSS_WEIGHT) if pure_ae_stepper else last_loss / float(CONTROLLER_LOSS_SCALE)
                    parts = {"ae": ae_part}
                for name in chunk_parts:
                    if name in parts:
                        chunk_parts[name] += tensor_to_float(parts[name])
                        chunk_counts[name] += 1
            stage_logs += 1
            chunk_loss_mean = chunk_loss / float(LOG_EVERY_STEPS)
            chunk_part_means = {
                name: (value / float(chunk_counts[name]) if chunk_counts[name] else 0.0)
                for name, value in chunk_parts.items()
            }
            log_loss_scalars(writer, step, chunk_part_means, linear_weight, angular_weight)
            stage_loss_history.append(chunk_loss_mean)
            improved = stage_trend_improved(stage_loss_history, int(stage_k))
            if improved or not math.isfinite(best_stage):
                stalls = 0
            else:
                stalls += 1
            best_stage = min(best_stage, chunk_loss_mean)
            best_global = min(best_global, chunk_loss_mean)
            writer.flush()
            final_path = ctl.save_controller_checkpoint(run_dir, run_id, "latest", model, optimizer, step, chunk_loss_mean, int(stage_k), cfg, metadata)
            now = time.perf_counter()
            if now - last_periodic_checkpoint_time >= float(PERIODIC_CHECKPOINT_SECONDS):
                periodic_path = ctl.save_controller_checkpoint(
                    run_dir,
                    run_id,
                    f"periodic_step{int(step)}",
                    model,
                    optimizer,
                    step,
                    chunk_loss_mean,
                    int(stage_k),
                    cfg,
                    metadata,
                )
                last_periodic_checkpoint_time = now
                print(f"saved periodic checkpoint {periodic_path}", flush=True)
            elapsed_stage = time.perf_counter() - stage_start
            print(
                f"{label}: step={step} K={stage_k} loss={chunk_loss_mean:.6g} best_stage={best_stage:.6g} "
                f"ae={chunk_part_means['ae']:.6g} linear={chunk_part_means['linear']:.6g} "
                f"angular={chunk_part_means['angular']:.6g} supervised={chunk_part_means['supervised']:.6g} "
                f"stalls={stalls} elapsed_stage_s={elapsed_stage:.1f}",
                flush=True,
            )
            is_final = stage_i == len(schedule) - 1
            patience = FINAL_STALL_PATIENCE_LOGS if is_final else STALL_PATIENCE_LOGS[int(stage_k)]
            min_time = MIN_STAGE_SECONDS[int(stage_k)]
            max_time = MAX_STAGE_SECONDS[int(stage_k)]
            if elapsed_stage >= min_time and stalls >= patience:
                if is_final and FINAL_STAGE_RUNS_UNTIL_MANUAL_STOP:
                    stalls = 0
                else:
                    break
            if elapsed_stage >= max_time:
                if is_final and FINAL_STAGE_RUNS_UNTIL_MANUAL_STOP:
                    print(f"{label}: max stage time ignored for final manual-stop K={stage_k}", flush=True)
                else:
                    print(f"{label}: max stage time reached for K={stage_k}", flush=True)
                    break
        final_path = ctl.save_controller_checkpoint(run_dir, run_id, f"stage_K{int(stage_k)}", model, optimizer, step, best_stage, int(stage_k), cfg, metadata)
        final_path = ctl.save_controller_checkpoint(run_dir, run_id, "last", model, optimizer, step, best_stage, int(stage_k), cfg, metadata)
        del stepper
    assert final_path is not None
    writer.close()
    write_run_status(run_dir, {"state": "complete", "label": label, "last_checkpoint": str(final_path)})
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
    parser.add_argument("--baseline-label", default="full_vanilla_ae_controller_baseline_stall")
    parser.add_argument("--refined-label", default="full_vanilla_ae_controller_refined_stall")
    parser.add_argument("--final-label", default="full_vanilla_ae_controller_random_init_stall")
    parser.add_argument("--ae-checkpoint", default="")
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--refined-checkpoint", default="")
    parser.add_argument("--weights-json", default="")
    parser.add_argument("--npz", default="", help="Optional cyclic .npz path or semicolon-separated paths.")
    parser.add_argument("--periodic-folder", default="", help="Optional periodic .npz folder/path list.")
    parser.add_argument("--nonperiodic-folder", default="", help="Optional nonperiodic .npz folder/path list.")
    parser.add_argument("--supervised-steps-per-cycle", type=int, default=SUPERVISED_STEPS_PER_CYCLE)
    parser.add_argument("--objective-cycle-steps", type=int, default=OBJECTIVE_CYCLE_STEPS)
    parser.add_argument("--pose-noise", type=float, default=0.0, help="Noise amount applied to rollout seed/reset poses.")
    parser.add_argument("--disable-cuda-graph", action="store_true", help="Use stable eager training instead of CUDA graph capture.")
    args = parser.parse_args()
    configure_dataset(args.npz, args.periodic_folder, args.nonperiodic_folder)

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
        baseline_path = train_controller_adaptive(
            args.baseline_label,
            ae_path,
            device,
            supervised_steps_per_cycle=int(args.supervised_steps_per_cycle),
            objective_cycle_steps=int(args.objective_cycle_steps),
            pose_noise_amount=float(args.pose_noise),
            disable_cuda_graph=bool(args.disable_cuda_graph),
        )
    if args.phase in ("all", "refine"):
        if baseline_path is None:
            raise ValueError("refine phase needs --baseline-checkpoint")
        if args.weights_json:
            weights = json.loads(Path(args.weights_json).read_text(encoding="utf-8"))
        else:
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
            supervised_steps_per_cycle=int(args.supervised_steps_per_cycle),
            objective_cycle_steps=int(args.objective_cycle_steps),
            pose_noise_amount=float(args.pose_noise),
            disable_cuda_graph=bool(args.disable_cuda_graph),
        )
    if args.phase in ("all", "final"):
        if refined_path is None:
            if not args.refined_checkpoint:
                raise ValueError("final phase needs --refined-checkpoint")
            refined_path = Path(args.refined_checkpoint).resolve()
        if args.weights_json:
            weights = json.loads(Path(args.weights_json).read_text(encoding="utf-8"))
        else:
            if not weights_path.exists():
                raise FileNotFoundError(f"Refinement weights not found: {weights_path}")
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
            supervised_steps_per_cycle=int(args.supervised_steps_per_cycle),
            objective_cycle_steps=int(args.objective_cycle_steps),
            pose_noise_amount=float(args.pose_noise),
            disable_cuda_graph=bool(args.disable_cuda_graph),
        )


if __name__ == "__main__":
    main()
