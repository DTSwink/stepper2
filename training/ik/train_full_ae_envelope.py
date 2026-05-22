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
    from . import tensorboard_log as tb_log
    from . import train_simple_ae_controller as ctl
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    from naming import checkpoint_path, clean_label, ik_run_id
    import excess_envelope as env
    import ik_core as tl
    import tensorboard_log as tb_log
    import train_simple_ae_controller as ctl

ensure_paths()


RUNS_DIR = PROJECT_ROOT / "training" / "runs"
CRASHED_RUNS_DIR = RUNS_DIR / "_crashed_ik"
PERIODIC_FOLDER = PROJECT_ROOT / "ue5" / "animations_omni_only_full" / "npz_final"
NONPERIODIC_FOLDER = PROJECT_ROOT / "ue5" / "animations_transitions_only_full_trimmed" / "npz_final"
DATASET_NPZ_TEXT: str | None = None
DATASET_PERIODIC_TEXT: str | None = str(PERIODIC_FOLDER)
DATASET_NONPERIODIC_TEXT: str | None = str(NONPERIODIC_FOLDER)
SCHEDULE = (1, 2, 8, 16, 32)
LOG_EVERY_STEPS = 100
ENVELOPE_BATCH_SIZE = 1024
CONTROLLER_LOSS_SCALE = 500.0
BASE_GRAD_CLIP_NORM = 1.0
MIN_STAGE_SECONDS = {1: 45.0, 2: 45.0, 8: 90.0, 16: 120.0, 32: 180.0}
STALL_PATIENCE_LOGS = {1: 8, 2: 8, 8: 10, 16: 10, 32: 24}
FINAL_STALL_PATIENCE_LOGS = 96
MAX_STAGE_SECONDS = {1: math.inf, 2: math.inf, 8: math.inf, 16: math.inf, 32: math.inf}
MIN_DELTA_FRACTION = 5e-4
STALL_RECENT_LOGS = {1: 3, 2: 3, 8: 4, 16: 6, 32: 8}
STALL_LOOKBACK_LOGS = {1: 6, 2: 6, 8: 8, 16: 12, 32: 24}
EVAL_BATCHES = 32


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
        next_vec, next_pelvis, next_payload = ctl.predicted_state_from_vector(pred_vec, store)
        if has_envelope_loss:
            carried_cur_foot_pos = next_foot_pos.index_select(0, rows)
            carried_cur_foot_rot = next_foot_rot.index_select(0, rows)
            carried_cur_root_pos = next_root_pos.index_select(0, rows)
            carried_cur_root_rot = next_root_rot.index_select(0, rows)
        clip_ids = clip_ids.index_select(0, rows)
        prev_vec = cur_vec.index_select(0, rows)
        prev_pelvis = cur_pelvis.index_select(0, rows)
        prev_payload = cur_payload.index_select(0, rows)
        cur_vec = next_vec.index_select(0, rows)
        cur_pelvis = next_pelvis.index_select(0, rows)
        cur_payload = next_payload.index_select(0, rows)
        cur_idx = cur_idx.index_select(0, rows) + 1
        effective_k = effective_k.index_select(0, rows)
        row_weight = row_weight.index_select(0, rows)

    return scaled_loss(total_loss), {
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_k = max(1, int(rollout_k))
    cur_idx = starts
    prev_idx = cur_idx - 1
    prev_vec, prev_pelvis, prev_payload = ctl.target_state(store, clip_ids, prev_idx)
    cur_vec, cur_pelvis, cur_payload = ctl.target_state(store, clip_ids, cur_idx)
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
        mask = continuing[:, None]
        mask_3 = continuing[:, None, None]
        mask_rot = continuing[:, None, None, None]
        next_vec, next_pelvis, next_payload = ctl.predicted_state_from_vector(pred_vec, store)
        prev_vec = torch.where(mask, cur_vec, prev_vec)
        prev_pelvis = torch.where(mask, cur_pelvis, prev_pelvis)
        prev_payload = torch.where(mask, cur_payload, prev_payload)
        cur_vec = torch.where(mask, next_vec, cur_vec)
        cur_pelvis = torch.where(mask, next_pelvis, cur_pelvis)
        cur_payload = torch.where(mask, next_payload, cur_payload)
        cur_idx = torch.where(continuing, cur_idx + 1, cur_idx)
        cur_root_pos = torch.where(mask, next_root_pos, cur_root_pos)
        cur_root_rot = torch.where(mask_3, next_root_rot, cur_root_rot)
        cur_foot_pos = torch.where(mask_3, next_foot_pos, cur_foot_pos)
        cur_foot_rot = torch.where(mask_rot, next_foot_rot, cur_foot_rot)
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
        self.root_cycle_count = graph_root_cycle_count(store, rollout_k)
        self.effective_k = torch.empty((self.batch_size,), dtype=torch.long, device=store.device)
        self.clip_ids = torch.empty_like(self.effective_k)
        self.starts = torch.empty_like(self.effective_k)
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
    start_pools = ctl.build_start_pools(store, values)
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
) -> object:
    if envelope is None and linear_weight == 0.0 and angular_weight == 0.0 and not random_init_pose:
        return ctl.make_pure_ae_stepper(model, optimizer, ae, mean, std, store, stage_k, batch_size, start_pools)
    if (
        envelope is not None
        and (linear_weight > 0.0 or angular_weight > 0.0)
        and not random_init_pose
        and store.device.type == "cuda"
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
    )


def log_loss_scalars(
    writer: SummaryWriter,
    step: int,
    total_loss: float,
    parts: dict[str, float] | None,
    linear_weight: float,
    angular_weight: float,
    has_envelope_loss: bool,
) -> None:
    if not tb_log.should_log_controller_step(step):
        return
    tb_log.log_controller_loss(
        writer,
        step=step,
        total_loss=total_loss,
        parts=parts,
        linear_weight=linear_weight,
        angular_weight=angular_weight,
        has_envelope_loss=has_envelope_loss,
        loss_scale=CONTROLLER_LOSS_SCALE,
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
    optimizer = ctl.make_adamw(model.parameters(), ctl.LEARNING_RATE, device, capturable=bool(device.type == "cuda" and not random_init_pose))
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
    tb_log.log_controller_start(writer, rollout_k=32 if start_at_k32 else 0, linear_weight=linear_weight, angular_weight=angular_weight)
    writer.flush()
    ctl.refresh_tensorboard_async()
    write_run_status(run_dir, {"state": "running", "label": label, "init_checkpoint": str(init_checkpoint or "")})
    best_global = math.inf
    init_tag = "init_from_checkpoint" if init_checkpoint is not None else "init"
    ctl.save_controller_checkpoint(run_dir, run_id, init_tag, model, optimizer, step, best_global, 0, cfg, metadata)
    if start_at_k32:
        schedule = (32,)
    else:
        min_stage = max(1, int(loaded_k or 1))
        schedule = tuple(k for k in SCHEDULE if int(k) >= min_stage) or (int(SCHEDULE[-1]),)
    t0 = time.perf_counter()
    last_loss = math.inf
    final_path: Path | None = None
    for stage_i, stage_k in enumerate(schedule):
        ctl.set_optimizer_lr(optimizer, ctl.stage_learning_rate(int(stage_k)))
        rollout_values = ctl.rollout_values_for(int(stage_k)) if ctl.mixed_rollout_enabled(int(stage_k)) else (int(stage_k),)
        start_pools = ctl.build_start_pools(store, rollout_values)
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
        )
        stage_start = time.perf_counter()
        stage_logs = 0
        stalls = 0
        best_stage = math.inf
        stage_loss_history: list[float] = []
        has_envelope_loss = envelope is not None and (linear_weight > 0.0 or angular_weight > 0.0)
        print(f"{label}: stage K={stage_k} batch={batch_size} stepper={getattr(stepper, 'kind', 'unknown')}", flush=True)
        while True:
            for _ in range(LOG_EVERY_STEPS):
                step += 1
                loss = stepper.step()  # type: ignore[attr-defined]
                last_loss = float(loss.detach().cpu())
                parts = dict(getattr(stepper, "last_parts")) if hasattr(stepper, "last_parts") else None
                log_loss_scalars(writer, step, last_loss, parts, linear_weight, angular_weight, has_envelope_loss)
            stage_logs += 1
            stage_loss_history.append(last_loss)
            improved = stage_trend_improved(stage_loss_history, int(stage_k))
            if improved or not math.isfinite(best_stage):
                stalls = 0
            else:
                stalls += 1
            best_stage = min(best_stage, last_loss)
            best_global = min(best_global, last_loss)
            stats = ctl.rollout_stat_summary(batch_size, int(stage_k))
            tb_log.log_controller_curriculum(
                writer,
                step=step,
                rollout_k=int(stage_k),
                effective_rollout_k_mean=float(stats["effective_k_mean"]),
                effective_rollout_k_max=float(stats["effective_k_max"]),
                stalls=stalls,
                start_time=t0,
            )
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
        baseline_path = train_controller_adaptive(args.baseline_label, ae_path, device)
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
        )


if __name__ == "__main__":
    main()
