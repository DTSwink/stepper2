from __future__ import annotations

import argparse
import json
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
    from . import ik_core as tl
    from . import train_ae_prior as rollout_data
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    from naming import checkpoint_path, ik_run_id
    import ik_core as tl
    import train_ae_prior as rollout_data

ensure_paths()


BATCH_SIZE = 4096
MAX_ROLLOUT_K = 32
ROLLOUT_K = 32
ROLLOUT_SCHEDULE = (1, 2, 4, 8, 16, 32)
ROLLOUT_STAGE_STEPS = (500, 500, 750, 1000, 1250, 3000)
TRAIN_STEPS = sum(ROLLOUT_STAGE_STEPS)
LEARNING_RATE = 3e-4
HIDDEN_DIM = 512
NUM_HIDDEN_LAYERS = 2
ROOT_LOOKAHEAD_STEPS = 1
VALIDATION_ROWS = 2048
USE_CUDA_GRAPH = True
USE_CUDA_AMP = False
LR_STAGE_DECAYS = ((0.60, 1.0 / 3.0), (0.85, 0.1))
RUNS_DIR = PROJECT_ROOT / "training" / "runs"
DEFAULT_WALK_F = PROJECT_ROOT / "ue5" / "animations_omni_only_full" / "npz_final" / "M_Neutral_Walk_Loop_F.npz"
TB_DIR_NAME = "tb"


StartPool = tuple[torch.Tensor, torch.Tensor]


def make_cfg(device: torch.device) -> tl.TrainConfig:
    cfg = tl.TrainConfig()
    cfg.pose_representation = "ik_markers"
    cfg.cyclic_animation = True
    cfg.predict_residual = False
    cfg.zero_init_output = False
    cfg.hidden_dim = HIDDEN_DIM
    cfg.num_hidden_layers = NUM_HIDDEN_LAYERS
    cfg.learning_rate = LEARNING_RATE
    cfg.batch_size = BATCH_SIZE
    cfg.root_lookahead_steps = ROOT_LOOKAHEAD_STEPS
    cfg.live_viewer = False
    cfg.visual_reporter = False
    cfg.update_comparison_on_exit = False
    cfg.use_torch_compile = False
    cfg.device = str(device)
    return cfg


def resolve_npz(path_text: str | None) -> Path:
    if path_text:
        path = Path(path_text)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if DEFAULT_WALK_F.exists():
        return DEFAULT_WALK_F.resolve()
    raise FileNotFoundError(f"Default walk-forward clip not found: {DEFAULT_WALK_F}")


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def npz_paths_from_text(path_text: str) -> list[Path]:
    paths: list[Path] = []
    for part in str(path_text).split(";"):
        if not part.strip():
            continue
        path = resolve_path(part.strip())
        if path.is_dir():
            found = sorted(path.glob("*.npz"))
            if not found:
                raise FileNotFoundError(f"No .npz files found in requested folder {path}")
            paths.extend(found)
        else:
            if not path.exists():
                raise FileNotFoundError(f"Requested NPZ path does not exist: {path}")
            if path.suffix.lower() != ".npz":
                raise ValueError(f"Requested path is not an .npz file: {path}")
            paths.append(path)
    return paths


def resolve_clip_specs(
    npz_text: str | None,
    periodic_text: str | None,
    nonperiodic_text: str | None,
) -> list[tuple[Path, bool]]:
    specs: list[tuple[Path, bool]] = []
    requested_any = any(str(text or "").strip() for text in (npz_text, periodic_text, nonperiodic_text))
    for path in npz_paths_from_text(periodic_text or ""):
        specs.append((path, True))
    for path in npz_paths_from_text(nonperiodic_text or ""):
        specs.append((path, False))
    if specs:
        return specs
    if npz_text:
        specs = [(path, True) for path in npz_paths_from_text(npz_text)]
        if specs:
            return specs
    if requested_any:
        raise FileNotFoundError("No NPZ files resolved from the requested dataset arguments")
    return [(resolve_npz(None), True)]


def load_clips(specs: list[tuple[Path, bool]], cfg: tl.TrainConfig) -> list[tl.MotionClip]:
    clips = [tl.MotionClip(path, cfg, cyclic_animation=cyclic) for path, cyclic in specs]
    first = clips[0].body_names
    first_parents = clips[0].parents_body_list
    for clip in clips[1:]:
        if clip.body_names != first or clip.parents_body_list != first_parents:
            raise ValueError(f"Skeleton mismatch: {clip.path} vs {clips[0].path}")
    return clips


def strict_rollout_start_max(clip: tl.MotionClip, cfg: tl.TrainConfig, rollout_k: int) -> int:
    if clip.cyclic_animation:
        return int(clip.cyclic_period) - 1
    return int(clip.T) - rollout_data.transition_feature_horizon(cfg) - max(1, int(rollout_k))


def build_start_pool(
    store: rollout_data.ClipStore,
    rollout_k: int,
    require_all_clips: bool = True,
) -> StartPool:
    clip_ids: list[int] = []
    max_starts: list[int] = []
    rejected: list[str] = []
    for clip_id, clip in enumerate(store.clips):
        max_start = strict_rollout_start_max(clip, store.cfg, rollout_k)
        if max_start < 1:
            rejected.append(str(clip.path))
            continue
        clip_ids.append(clip_id)
        max_starts.append(max_start)
    if rejected and require_all_clips:
        shown = "; ".join(rejected[:8])
        suffix = f"; ... {len(rejected) - 8} more" if len(rejected) > 8 else ""
        raise ValueError(f"Clips without a full K={rollout_k} rollout window: {shown}{suffix}")
    if not clip_ids:
        raise ValueError("No valid full-window rollout starts found.")
    return (
        torch.tensor(clip_ids, dtype=torch.long, device=store.device),
        torch.tensor(max_starts, dtype=torch.long, device=store.device),
    )


def rollout_values_for(max_k: int) -> tuple[int, ...]:
    values = []
    k = max(1, int(max_k))
    while k > 1:
        values.append(k)
        k = max(1, k // 2)
    values.append(1)
    return tuple(dict.fromkeys(values))


def fractal_rollout_probs(values: tuple[int, ...], device: torch.device) -> torch.Tensor:
    if len(values) == 1:
        return torch.ones((1,), dtype=torch.float32, device=device)
    probs = [0.5 ** (i + 1) for i in range(len(values) - 1)]
    probs.append(0.5 ** (len(values) - 1))
    out = torch.tensor(probs, dtype=torch.float32, device=device)
    return out / out.sum()


def rollout_stat_summary(batch_size: int, rollout_k: int) -> dict[str, float]:
    values = rollout_values_for(rollout_k) if mixed_rollout_enabled(rollout_k) else (max(1, int(rollout_k)),)
    remaining = max(1, int(batch_size))
    counts: list[int] = []
    for _value in values[:-1]:
        count = remaining // 2
        counts.append(count)
        remaining -= count
    counts.append(remaining)
    total = float(sum(counts))
    return {
        "effective_k_mean": sum(float(k) * float(c) for k, c in zip(values, counts)) / total,
        "effective_k_max": float(max(values)),
    }


def effective_rollout_stage_steps(train_steps: int) -> tuple[int, ...]:
    steps = [int(v) for v in ROLLOUT_STAGE_STEPS]
    scheduled = sum(steps)
    if int(train_steps) > scheduled:
        steps[-1] += int(train_steps) - scheduled
    return tuple(steps)


def rollout_stage_for_step(step: int, train_steps: int = TRAIN_STEPS) -> tuple[int, int, int, int]:
    start = 1
    stage_steps_all = effective_rollout_stage_steps(train_steps)
    for stage_idx, (rollout_k, stage_steps) in enumerate(zip(ROLLOUT_SCHEDULE, stage_steps_all)):
        end = start + int(stage_steps) - 1
        if int(step) <= end:
            return stage_idx, int(rollout_k), start, end
        start = end + 1
    final_start = 1 + sum(stage_steps_all[:-1])
    final_end = sum(stage_steps_all)
    return len(ROLLOUT_SCHEDULE) - 1, int(ROLLOUT_SCHEDULE[-1]), final_start, final_end


def mixed_rollout_enabled(rollout_k: int) -> bool:
    return int(rollout_k) >= int(MAX_ROLLOUT_K)


def build_start_pools(
    store: rollout_data.ClipStore,
    rollout_values: tuple[int, ...],
    require_all_clips: bool = True,
) -> dict[int, StartPool]:
    return {int(k): build_start_pool(store, int(k), require_all_clips=require_all_clips) for k in rollout_values}


def sample_start_pool(
    pool_clip_ids: torch.Tensor,
    pool_max_starts: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = max(1, int(batch_size))
    total_rows = int(pool_max_starts.sum().detach().cpu())
    if batch_size == total_rows:
        clip_chunks: list[torch.Tensor] = []
        start_chunks: list[torch.Tensor] = []
        for clip_id, max_start in zip(pool_clip_ids.tolist(), pool_max_starts.tolist()):
            starts = torch.arange(1, int(max_start) + 1, dtype=torch.long, device=pool_clip_ids.device)
            clip_chunks.append(torch.full_like(starts, int(clip_id)))
            start_chunks.append(starts)
        return torch.cat(clip_chunks, dim=0), torch.cat(start_chunks, dim=0)
    rows = torch.randint(0, pool_clip_ids.numel(), (batch_size,), device=pool_clip_ids.device)
    clip_ids = pool_clip_ids.index_select(0, rows)
    max_starts = pool_max_starts.index_select(0, rows)
    starts = (torch.rand(rows.shape[0], device=pool_clip_ids.device) * max_starts.float()).floor().long() + 1
    return clip_ids, starts


def sample_effective_rollout_k(batch_size: int, rollout_k: int, device: torch.device) -> torch.Tensor:
    rollout_k = max(1, int(rollout_k))
    batch_size = max(1, int(batch_size))
    if not mixed_rollout_enabled(rollout_k):
        return torch.full((batch_size,), rollout_k, dtype=torch.long, device=device)
    values = rollout_values_for(rollout_k)
    remaining = batch_size
    chunks = []
    for value in values[:-1]:
        count = remaining // 2
        if count > 0:
            chunks.append(torch.full((count,), int(value), dtype=torch.long, device=device))
        remaining -= count
    chunks.append(torch.full((remaining,), int(values[-1]), dtype=torch.long, device=device))
    effective_k = torch.cat(chunks, dim=0)
    return effective_k.index_select(0, torch.randperm(batch_size, device=device))


def sample_rollout_rows(
    start_pools: dict[int, StartPool],
    effective_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    clip_ids = torch.empty_like(effective_k)
    starts = torch.empty_like(effective_k)
    for rollout_k, (pool_clip_ids, pool_max_starts) in start_pools.items():
        rows = (effective_k == int(rollout_k)).nonzero(as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        row_clip_ids, row_starts = sample_start_pool(pool_clip_ids, pool_max_starts, int(rows.numel()))
        clip_ids[rows] = row_clip_ids
        starts[rows] = row_starts
    return clip_ids, starts


def start_pool_summary(start_pools: dict[int, StartPool]) -> dict[str, dict[str, int]]:
    summary = {}
    for rollout_k, (pool_clip_ids, pool_max_starts) in start_pools.items():
        summary[str(int(rollout_k))] = {
            "eligible_clip_count": int(pool_clip_ids.numel()),
            "row_count": int(pool_max_starts.sum().detach().cpu()),
        }
    return summary


def make_adamw(
    params,
    lr: float,
    device: torch.device,
    weight_decay: float = 0.0,
    capturable: bool = False,
) -> torch.optim.Optimizer:
    kwargs = {"lr": lr, "weight_decay": weight_decay}
    if device.type == "cuda":
        kwargs["fused"] = True
        if capturable:
            kwargs["capturable"] = True
    try:
        return torch.optim.AdamW(params, **kwargs)
    except (RuntimeError, TypeError):
        kwargs.pop("fused", None)
        if device.type != "cuda":
            kwargs.pop("capturable", None)
        return torch.optim.AdamW(params, **kwargs)


def stage_learning_rate(base_lr: float, stage_step: int, stage_steps: int) -> float:
    progress = float(max(0, int(stage_step) - 1)) / float(max(1, int(stage_steps) - 1))
    lr = float(base_lr)
    for threshold, multiplier in LR_STAGE_DECAYS:
        if progress >= float(threshold):
            lr = float(base_lr) * float(multiplier)
    return lr


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def release_stepper(stepper: object | None, device: torch.device) -> None:
    if stepper is None:
        return
    del stepper
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def make_summary_writer(run_dir: Path) -> tuple[SummaryWriter, Path]:
    tb_dir = run_dir / TB_DIR_NAME
    tb_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(tb_dir), flush_secs=1), tb_dir


def assert_tensorboard_event_file(tb_dir: Path) -> None:
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline:
        event_files = list(tb_dir.glob("events.out.tfevents*"))
        if any(path.stat().st_size > 0 for path in event_files):
            return
        time.sleep(0.05)
    raise RuntimeError(f"TensorBoard event file was not created in {tb_dir}")


def refresh_tensorboard_async() -> None:
    script = PROJECT_ROOT / "training" / "ik" / "launch_tensorboard_latest.ps1"
    if not script.exists():
        return
    try:
        subprocess.Popen(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
            ],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
    except Exception as exc:
        print(f"tensorboard refresh skipped: {exc}", flush=True)


def payload_slice(store: rollout_data.ClipStore) -> slice:
    start = 3 + 6 + int(store.Jcore) * 6
    return slice(start, start + int(store.ik_payload_dim))


def target_state(
    store: rollout_data.ClipStore,
    clip_ids: torch.Tensor,
    idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vec = store.get_target_output(clip_ids, idx)
    payload = vec[:, payload_slice(store)]
    return vec, vec[:, :3], payload


def predicted_state_from_raw(
    raw: torch.Tensor,
    store: rollout_data.ClipStore,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b = raw.shape[0]
    cursor = 0
    pelvis_pos = raw[:, cursor : cursor + 3]
    cursor += 3
    pelvis_rot6 = tl.clean_6d(raw[:, cursor : cursor + 6])
    cursor += 6
    core_dim = int(store.Jcore) * 6
    core_rot6 = tl.clean_6d(raw[:, cursor : cursor + core_dim].reshape(-1, 6)).reshape(b, int(store.Jcore), 6)
    cursor += core_dim
    payload_dim = int(store.ik_payload_dim)
    payload = tl.clean_ik_payload(raw[:, cursor : cursor + payload_dim])
    vec = torch.cat((pelvis_pos, pelvis_rot6, core_rot6.reshape(b, -1), payload), dim=-1)
    return vec, pelvis_pos, payload


def model_forward(
    model: torch.nn.Module,
    inp: torch.Tensor,
    cur_vec: torch.Tensor,
    cfg: tl.TrainConfig,
) -> torch.Tensor:
    if USE_CUDA_AMP and inp.is_cuda:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            raw = model(inp).float()
    else:
        raw = model(inp)
    if cfg.predict_residual:
        raw = cur_vec + raw
    return raw


def build_ik_input(
    store: rollout_data.ClipStore,
    clip_ids: torch.Tensor,
    cur_idx: torch.Tensor,
    prev_vec: torch.Tensor,
    cur_vec: torch.Tensor,
    prev_pelvis: torch.Tensor,
    cur_pelvis: torch.Tensor,
    prev_markers: torch.Tensor,
    cur_markers: torch.Tensor,
    cfg: tl.TrainConfig,
) -> torch.Tensor:
    pelvis_vel = (cur_pelvis - prev_pelvis) / cfg.pose_delta_scale_final
    marker_vel = (cur_markers - prev_markers).reshape(cur_idx.shape[0], -1) / cfg.pose_delta_scale_final
    root_features = store.get_input_root_features(clip_ids, cur_idx)
    return torch.cat((cur_vec, prev_vec, pelvis_vel, marker_vel, root_features), dim=-1)


def supervised_rollout_loss(
    model: torch.nn.Module,
    store: rollout_data.ClipStore,
    cfg: tl.TrainConfig,
    rollout_k: int,
    batch_size: int,
    start_pools: dict[int, StartPool],
) -> torch.Tensor:
    rollout_k = max(1, int(rollout_k))
    original_batch_size = max(1, int(batch_size))
    effective_k = sample_effective_rollout_k(original_batch_size, rollout_k, store.device)
    clip_ids, starts = sample_rollout_rows(start_pools, effective_k)
    prev_idx = starts - 1
    cur_idx = starts
    prev_vec, prev_pelvis, prev_markers = target_state(store, clip_ids, prev_idx)
    cur_vec, cur_pelvis, cur_markers = target_state(store, clip_ids, cur_idx)
    row_weight = (1.0 / effective_k.float()) / float(original_batch_size)
    total_loss = torch.zeros((), dtype=torch.float32, device=store.device)
    for step in range(rollout_k):
        inp = build_ik_input(
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
        raw = model_forward(model, inp, cur_vec, cfg)
        next_vec, next_pelvis, next_markers = predicted_state_from_raw(raw, store)
        target_idx = cur_idx + 1
        target = store.get_target_output(clip_ids, target_idx)
        # Loss must use the canonical cleaned state that rollout feeds back.
        # Raw 6D rotations can be numerically far while decoding to the same pose.
        row_loss = (next_vec - target).square().mean(dim=-1)
        total_loss = total_loss + (row_loss * row_weight).sum()
        if step + 1 >= rollout_k:
            break
        continuing = effective_k > (step + 1)
        rows = continuing.nonzero(as_tuple=False).flatten()
        if rows.numel() == 0:
            break
        next_clip_ids = clip_ids.index_select(0, rows)
        next_target_idx = target_idx.index_select(0, rows)
        prev_vec = cur_vec.index_select(0, rows)
        prev_pelvis = cur_pelvis.index_select(0, rows)
        prev_markers = cur_markers.index_select(0, rows)
        cur_vec = next_vec.index_select(0, rows)
        cur_pelvis = next_pelvis.index_select(0, rows)
        cur_markers = next_markers.index_select(0, rows)
        clip_ids = next_clip_ids
        prev_idx = cur_idx.index_select(0, rows)
        cur_idx = next_target_idx
        effective_k = effective_k.index_select(0, rows)
        row_weight = row_weight.index_select(0, rows)
    return total_loss


def supervised_rollout_loss_static(
    model: torch.nn.Module,
    store: rollout_data.ClipStore,
    cfg: tl.TrainConfig,
    rollout_k: int,
    batch_size: int,
    effective_k: torch.Tensor,
    clip_ids: torch.Tensor,
    starts: torch.Tensor,
) -> torch.Tensor:
    cur_idx = starts
    prev_idx = starts - 1
    prev_vec, prev_pelvis, prev_markers = target_state(store, clip_ids, prev_idx)
    cur_vec, cur_pelvis, cur_markers = target_state(store, clip_ids, cur_idx)
    row_weight = (1.0 / effective_k.float()) / float(batch_size)
    total_loss = torch.zeros((), dtype=torch.float32, device=store.device)
    for step in range(max(1, int(rollout_k))):
        inp = build_ik_input(
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
        raw = model_forward(model, inp, cur_vec, cfg)
        next_vec, next_pelvis, next_markers = predicted_state_from_raw(raw, store)
        active = effective_k > step
        target_idx = torch.where(active, cur_idx + 1, cur_idx)
        target = store.get_target_output(clip_ids, target_idx)
        # Keep static/CUDA-graph loss identical to the eager path.
        row_loss = (next_vec - target).square().mean(dim=-1)
        total_loss = total_loss + (row_loss * row_weight * active.float()).sum()
        if step + 1 >= rollout_k:
            break
        continuing = effective_k > (step + 1)
        continuing_vec = continuing[:, None]
        continuing_markers = continuing[:, None]
        prev_vec = torch.where(continuing_vec, cur_vec, prev_vec)
        prev_pelvis = torch.where(continuing_vec, cur_pelvis, prev_pelvis)
        prev_markers = torch.where(continuing_markers, cur_markers, prev_markers)
        cur_vec = torch.where(continuing_vec, next_vec, cur_vec)
        cur_pelvis = torch.where(continuing_vec, next_pelvis, cur_pelvis)
        cur_markers = torch.where(continuing_markers, next_markers, cur_markers)
        cur_idx = torch.where(continuing, cur_idx + 1, cur_idx)
    return total_loss


def sample_rollout_batch(
    store: rollout_data.ClipStore,
    rollout_k: int,
    batch_size: int,
    start_pools: dict[int, StartPool],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    effective_k = sample_effective_rollout_k(batch_size, rollout_k, store.device)
    clip_ids, starts = sample_rollout_rows(start_pools, effective_k)
    return effective_k, clip_ids, starts


class EagerSupervisedStep:
    kind = "eager"

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        store: rollout_data.ClipStore,
        cfg: tl.TrainConfig,
        rollout_k: int,
        batch_size: int,
        start_pools: dict[int, StartPool],
    ):
        self.model = model
        self.optimizer = optimizer
        self.store = store
        self.cfg = cfg
        self.rollout_k = int(rollout_k)
        self.batch_size = int(batch_size)
        self.start_pools = start_pools

    def step(self) -> torch.Tensor:
        loss = supervised_rollout_loss(self.model, self.store, self.cfg, self.rollout_k, self.batch_size, self.start_pools)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss.detach()


class CudaGraphSupervisedStep:
    kind = "cuda_graph_static_masked"

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        store: rollout_data.ClipStore,
        cfg: tl.TrainConfig,
        rollout_k: int,
        batch_size: int,
        start_pools: dict[int, StartPool],
    ):
        if store.device.type != "cuda":
            raise RuntimeError("CudaGraphSupervisedStep requires a CUDA store")
        self.model = model
        self.optimizer = optimizer
        self.store = store
        self.cfg = cfg
        self.rollout_k = int(rollout_k)
        self.batch_size = int(batch_size)
        self.start_pools = start_pools
        self.effective_k = torch.empty((self.batch_size,), dtype=torch.long, device=store.device)
        self.clip_ids = torch.empty_like(self.effective_k)
        self.starts = torch.empty_like(self.effective_k)
        self.loss = torch.zeros((), dtype=torch.float32, device=store.device)
        self.graph = torch.cuda.CUDAGraph()
        self._capture()

    def _sample_into_static_buffers(self) -> None:
        effective_k, clip_ids, starts = sample_rollout_batch(
            self.store,
            self.rollout_k,
            self.batch_size,
            self.start_pools,
        )
        self.effective_k.copy_(effective_k)
        self.clip_ids.copy_(clip_ids)
        self.starts.copy_(starts)

    def _loss(self) -> torch.Tensor:
        return supervised_rollout_loss_static(
            self.model,
            self.store,
            self.cfg,
            self.rollout_k,
            self.batch_size,
            self.effective_k,
            self.clip_ids,
            self.starts,
        )

    def _capture(self) -> None:
        self._sample_into_static_buffers()
        for _ in range(3):
            self.optimizer.zero_grad(set_to_none=False)
            loss = self._loss()
            loss.backward()
            self.optimizer.step()
            del loss
        torch.cuda.synchronize()
        self.optimizer.zero_grad(set_to_none=False)
        with torch.cuda.graph(self.graph):
            self.optimizer.zero_grad(set_to_none=False)
            self.loss = self._loss()
            self.loss.backward()
            self.optimizer.step()

    def step(self) -> torch.Tensor:
        self._sample_into_static_buffers()
        self.graph.replay()
        return self.loss.detach()


def make_supervised_stepper(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    store: rollout_data.ClipStore,
    cfg: tl.TrainConfig,
    rollout_k: int,
    batch_size: int,
    start_pools: dict[int, StartPool],
    use_cuda_graph: bool = USE_CUDA_GRAPH,
) -> EagerSupervisedStep | CudaGraphSupervisedStep:
    if bool(use_cuda_graph) and store.device.type == "cuda":
        return CudaGraphSupervisedStep(model, optimizer, store, cfg, rollout_k, batch_size, start_pools)
    return EagerSupervisedStep(model, optimizer, store, cfg, rollout_k, batch_size, start_pools)


def validation_starts(
    store: rollout_data.ClipStore,
    max_rows: int,
    pool_clip_ids: torch.Tensor,
    pool_max_starts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    per_clip = max(1, int(max_rows) // max(1, int(pool_clip_ids.numel())))
    clip_chunks: list[torch.Tensor] = []
    start_chunks: list[torch.Tensor] = []
    for clip_id, max_start in zip(pool_clip_ids.tolist(), pool_max_starts.tolist()):
        take = min(int(max_start), per_clip)
        if take == int(max_start):
            starts = torch.arange(1, int(max_start) + 1, dtype=torch.long, device=store.device)
        else:
            starts = torch.linspace(1, int(max_start), steps=take, device=store.device).round().long().unique()
        clip_chunks.append(torch.full((starts.numel(),), int(clip_id), dtype=torch.long, device=store.device))
        start_chunks.append(starts)
    return torch.cat(clip_chunks, dim=0), torch.cat(start_chunks, dim=0)


@torch.no_grad()
def rollout_joint_error(
    model: torch.nn.Module,
    store: rollout_data.ClipStore,
    cfg: tl.TrainConfig,
    rollout_k: int,
    pool_clip_ids: torch.Tensor,
    pool_max_starts: torch.Tensor,
) -> tuple[float, float]:
    rollout_k = max(1, int(rollout_k))
    clip_ids, starts = validation_starts(store, VALIDATION_ROWS, pool_clip_ids, pool_max_starts)
    prev_idx = starts - 1
    cur_idx = starts
    prev_pose = store.get_pose(clip_ids, prev_idx)
    cur_pose = store.get_pose(clip_ids, cur_idx)
    total_error = 0.0
    total_frames = 0
    max_error = 0.0
    for step in range(rollout_k):
        inp = rollout_data.store_build_input(store, clip_ids, prev_idx, cur_idx, prev_pose, cur_pose, cfg)
        raw = tl.predict_next_raw(model, inp, cur_pose, cfg)
        pred_pose, _raw_pose = tl.output_to_pose(raw, store.prototype)
        target_idx = cur_idx + 1
        root_pos, root_rot, _yaw, _heading = store.root_state(clip_ids, target_idx)
        pred_global, pred_canon = store.fk_positions_from_pose(clip_ids, root_pos, root_rot, pred_pose)
        target_pose = store.get_pose(clip_ids, target_idx)
        target_global, _target_canon = store.fk_positions_from_pose(clip_ids, root_pos, root_rot, target_pose)
        per_frame = (pred_global - target_global).norm(dim=-1).mean(dim=-1)
        total_error += float(per_frame.sum().cpu())
        total_frames += int(per_frame.numel())
        max_error = max(max_error, float(per_frame.max().cpu()))
        if step + 1 == rollout_k:
            continue
        prev_pose = cur_pose
        cur_pose = tl.next_pose_from_prediction(pred_pose, pred_canon)
        prev_idx = cur_idx
        cur_idx = target_idx
    if total_frames == 0:
        return 0.0, 0.0
    return total_error / float(total_frames), max_error


def save_named_checkpoint(
    run_dir: Path,
    run_id: str,
    tag: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best: float,
    cfg: tl.TrainConfig,
    metadata: dict,
) -> Path:
    path = checkpoint_path(run_dir, run_id, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tl.checkpoint_payload(model, optimizer, step, best, ROLLOUT_K, cfg, metadata), path)
    return path


def write_readable_config(
    run_dir: Path,
    args: argparse.Namespace,
    cfg: tl.TrainConfig,
    metadata: dict,
    *,
    init_checkpoint: Path | None,
    train_steps: int,
    start_step: int,
) -> None:
    important = {
        "run_label": str(args.run_label),
        "mode": "supervised",
        "init_checkpoint": str(init_checkpoint or ""),
        "start_step": int(start_step),
        "train_steps": int(train_steps),
        "npz": str(args.npz or ""),
        "periodic_folder": str(args.periodic_folder or ""),
        "nonperiodic_folder": str(args.nonperiodic_folder or ""),
        "batch_size": int(metadata["batch_size"]),
        "clip_id_count": int(metadata["clip_id_count"]),
        "row_count": int(metadata["row_count"]),
        "rollout_schedule": metadata["rollout_schedule"],
        "rollout_stage_steps": metadata["rollout_stage_steps"],
        "learning_rate": float(LEARNING_RATE),
        "hidden_dim": int(HIDDEN_DIM),
        "num_hidden_layers": int(NUM_HIDDEN_LAYERS),
        "use_cuda_graph": bool(USE_CUDA_GRAPH),
        "effective_use_cuda_graph": bool(metadata.get("effective_use_cuda_graph", USE_CUDA_GRAPH)),
        "cuda_amp": bool(USE_CUDA_AMP),
        "checkpoint_policy": "init, latest every logged step, best at K32, last",
    }
    full = {
        "important": important,
        "args": vars(args),
        "training_constants": {
            "BATCH_SIZE": BATCH_SIZE,
            "MAX_ROLLOUT_K": MAX_ROLLOUT_K,
            "ROLLOUT_K": ROLLOUT_K,
            "ROLLOUT_SCHEDULE": ROLLOUT_SCHEDULE,
            "ROLLOUT_STAGE_STEPS": ROLLOUT_STAGE_STEPS,
            "TRAIN_STEPS": TRAIN_STEPS,
            "LEARNING_RATE": LEARNING_RATE,
            "HIDDEN_DIM": HIDDEN_DIM,
            "NUM_HIDDEN_LAYERS": NUM_HIDDEN_LAYERS,
            "ROOT_LOOKAHEAD_STEPS": ROOT_LOOKAHEAD_STEPS,
            "VALIDATION_ROWS": VALIDATION_ROWS,
            "USE_CUDA_GRAPH": USE_CUDA_GRAPH,
            "effective_use_cuda_graph": bool(metadata.get("effective_use_cuda_graph", USE_CUDA_GRAPH)),
            "USE_CUDA_AMP": USE_CUDA_AMP,
            "LR_STAGE_DECAYS": LR_STAGE_DECAYS,
        },
        "metadata": metadata,
        "train_config": asdict(cfg),
    }
    (run_dir / "config_readable.json").write_text(json.dumps(full, indent=2), encoding="utf-8")


def load_init_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint: Path,
    load_optimizer: bool,
) -> tuple[int, float, int, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if load_optimizer and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as exc:
            print(f"optimizer state not loaded from {checkpoint}: {exc}", flush=True)
    return (
        int(ckpt.get("epoch", 0)),
        float(ckpt.get("best_val", float("inf"))),
        int(ckpt.get("rollout_k", 0)),
        dict(ckpt.get("metadata", {})),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Contained IK supervised rollout trainer with per-row random sampling.")
    parser.add_argument("--npz", default=None, help="NPZ file, NPZ folder, or semicolon-separated NPZ list.")
    parser.add_argument("--periodic-folder", default=None, help="Periodic NPZ folder/list; clips are sampled cyclically.")
    parser.add_argument("--nonperiodic-folder", default=None, help="Nonperiodic NPZ folder/list; clips are not sampled cyclically.")
    parser.add_argument("--run-label", default="walkF_supervised")
    parser.add_argument("--init-checkpoint", default="", help="Optional controller checkpoint to initialize from.")
    parser.add_argument("--train-steps", type=int, default=TRAIN_STEPS, help="Override supervised training step count.")
    parser.add_argument("--load-optimizer", action="store_true", help="Also load optimizer state from --init-checkpoint.")
    parser.add_argument(
        "--resume-step-from-checkpoint",
        action="store_true",
        help="Continue the schedule from the checkpoint epoch instead of using it only as initialization.",
    )
    parser.add_argument("--disable-cuda-graph", action="store_true", help="Use eager supervised updates instead of CUDA graph replay.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        try:
            torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
        except AttributeError:
            pass

    cfg = make_cfg(device)
    clip_specs = resolve_clip_specs(args.npz, args.periodic_folder, args.nonperiodic_folder)
    clips = load_clips(clip_specs, cfg)
    input_dim, output_dim = tl.make_batch_dims(clips[0], cfg)
    use_cuda_graph = bool(USE_CUDA_GRAPH and not args.disable_cuda_graph)
    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    optimizer = make_adamw(
        model.parameters(),
        LEARNING_RATE,
        device,
        weight_decay=0.0,
        capturable=bool(use_cuda_graph and device.type == "cuda"),
    )
    base_step = 0
    init_best = float("inf")
    init_rollout_k = 0
    init_metadata: dict = {}
    init_checkpoint = resolve_path(args.init_checkpoint) if args.init_checkpoint else None
    if init_checkpoint is not None:
        base_step, init_best, init_rollout_k, init_metadata = load_init_checkpoint(
            model,
            optimizer,
            init_checkpoint,
            bool(args.load_optimizer),
        )
        print(f"loaded init checkpoint {init_checkpoint} epoch={base_step} K={init_rollout_k}", flush=True)
    start_step = int(base_step) if bool(args.resume_step_from_checkpoint) else 0
    store = rollout_data.ClipStore(clips, cfg, device)
    stage_cache: dict[int, dict[str, object]] = {}
    for stage_k in ROLLOUT_SCHEDULE:
        rollout_values = rollout_values_for(stage_k) if mixed_rollout_enabled(stage_k) else (int(stage_k),)
        start_pools = build_start_pools(store, rollout_values, require_all_clips=False)
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
            "rollout_stats": rollout_stat_summary(batch_size, int(stage_k)),
        }
    final_cache = stage_cache[int(ROLLOUT_K)]

    run_id = ik_run_id(args.run_label)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    train_steps = max(1, int(args.train_steps))
    metadata = {
        "npz_paths": [str(path) for path, _cyclic in clip_specs],
        "npz_folders": [
            {"path": str(path.parent), "cyclic": bool(cyclic)}
            for path, cyclic in clip_specs
        ],
        "tensorboard_logdir": str(run_dir / TB_DIR_NAME),
        "policy": {
            "pose_representation": "ik_markers",
            "gpu_resident_rollout": True,
            "random_clip_per_row": True,
            "mixed_rollout_at_max": True,
            "training_step": "staged_cuda_graph_static_masked" if use_cuda_graph and device.type == "cuda" else "staged_eager",
            "cuda_amp": bool(USE_CUDA_AMP and device.type == "cuda"),
            "stage_lr_decays": [(float(t), float(m)) for t, m in LR_STAGE_DECAYS],
            "cyclic": True,
            "checkpoint_naming": "YYYYMMDD_HHMMSS_ik_<label>_<tag>.pt",
            "ik_payload_dim": int(clips[0].ik_payload_dim),
            "ik_payload_layout": "hand pos3+rot6+pole, hand pos3+rot6+pole, foot pos3+rot6+pole+toe, foot pos3+rot6+pole+toe",
        },
        "rollout_k": int(ROLLOUT_K),
        "max_rollout_k": int(MAX_ROLLOUT_K),
        "rollout_schedule": [int(k) for k in ROLLOUT_SCHEDULE],
        "rollout_stage_steps": [int(n) for n in effective_rollout_stage_steps(train_steps)],
        "rollout_values": [int(k) for k in final_cache["rollout_values"]],
        "fractal_rollout_probabilities": {
            str(int(k)): float(p)
            for k, p in zip(
                final_cache["rollout_values"],
                fractal_rollout_probs(final_cache["rollout_values"], torch.device("cpu")).tolist(),
            )
        },
        "start_pools": {str(k): start_pool_summary(stage_cache[k]["start_pools"]) for k in stage_cache},
        "row_count": int(final_cache["row_count"]),
        "eligible_clip_count": int(final_cache["max_pool_clip_ids"].numel()),
        "batch_size": int(final_cache["batch_size"]),
        "clip_id_count": len(clips),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "init_checkpoint": str(init_checkpoint or ""),
        "source_epoch": int(base_step),
        "source_rollout_k": int(init_rollout_k),
        "resume_step_from_checkpoint": bool(args.resume_step_from_checkpoint),
        "start_step": int(start_step),
        "train_steps": int(train_steps),
        "effective_use_cuda_graph": bool(use_cuda_graph),
        "source_metadata": init_metadata,
    }
    config_payload = {"config": asdict(cfg), "metadata": metadata}
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    write_readable_config(
        run_dir,
        args,
        cfg,
        metadata,
        init_checkpoint=init_checkpoint,
        train_steps=train_steps,
        start_step=start_step,
    )
    writer, tb_dir = make_summary_writer(run_dir)
    print(f"tensorboard_logdir={tb_dir}", flush=True)
    writer.add_text("config/json", f"```json\n{json.dumps(config_payload, indent=2)}\n```", 0)
    writer.add_text("run/id", run_id, 0)
    writer.add_scalar("run/started", 1.0, 0)
    writer.add_scalar("curriculum/rollout_k", int(ROLLOUT_SCHEDULE[0]), 0)
    writer.add_scalar("train/effective_rollout_k_mean", float(ROLLOUT_SCHEDULE[0]), 0)
    writer.add_scalar("train/effective_rollout_k_max", float(ROLLOUT_SCHEDULE[0]), 0)
    writer.flush()
    assert_tensorboard_event_file(tb_dir)
    refresh_tensorboard_async()

    best = init_best
    save_named_checkpoint(run_dir, run_id, "init", model, optimizer, start_step, best, cfg, metadata)
    start = time.perf_counter()
    current_stage_k = -1
    current_lr = None
    stepper: EagerSupervisedStep | CudaGraphSupervisedStep | None = None
    if start_step >= train_steps:
        print(f"checkpoint already at step={start_step}, train_steps={train_steps}; writing last checkpoint", flush=True)
    for step in range(start_step + 1, train_steps + 1):
        stage_idx, stage_k, stage_start, stage_end = rollout_stage_for_step(step, train_steps)
        stage_step = step - stage_start + 1
        stage_steps = stage_end - stage_start + 1
        lr = stage_learning_rate(LEARNING_RATE, stage_step, stage_steps)
        lr_changed = current_lr is None or abs(float(lr) - float(current_lr)) > 1e-16
        if lr_changed:
            set_optimizer_lr(optimizer, lr)
            current_lr = float(lr)
        if stage_k != current_stage_k or lr_changed:
            release_stepper(stepper, device)
            stepper = None
            cache = stage_cache[int(stage_k)]
            stepper = make_supervised_stepper(
                model,
                optimizer,
                store,
                cfg,
                int(stage_k),
                int(cache["batch_size"]),
                cache["start_pools"],
                use_cuda_graph=use_cuda_graph,
            )
            current_stage_k = int(stage_k)
            print(
                f"stage={stage_idx + 1}/{len(ROLLOUT_SCHEDULE)} "
                f"K={stage_k} steps={stage_start}-{stage_end} "
                f"batch={int(cache['batch_size'])} lr={float(current_lr):.3g} stepper={stepper.kind}",
                flush=True,
            )
        assert stepper is not None
        loss = stepper.step()
        if step == stage_end or step == train_steps:
            stage_path = save_named_checkpoint(
                run_dir,
                run_id,
                f"stage_K{int(stage_k)}",
                model,
                optimizer,
                step,
                best,
                cfg,
                metadata,
            )
            print(f"saved stage checkpoint {stage_path}", flush=True)

        if step == 1 or step == stage_start or step % 250 == 0 or step == train_steps:
            cache = stage_cache[int(stage_k)]
            model.eval()
            mean_err, max_err = rollout_joint_error(
                model,
                store,
                cfg,
                int(stage_k),
                cache["max_pool_clip_ids"],
                cache["max_pool_starts"],
            )
            model.train()
            if int(stage_k) == int(ROLLOUT_K) and mean_err < best:
                best = mean_err
                save_named_checkpoint(run_dir, run_id, "best", model, optimizer, step, best, cfg, metadata)
            save_named_checkpoint(run_dir, run_id, "latest", model, optimizer, step, best, cfg, metadata)
            elapsed = time.perf_counter() - start
            rollout_stats = cache["rollout_stats"]
            best_to_log = best if best < float("inf") else mean_err
            print(
                f"step={step:05d} loss={float(loss.detach().cpu()):.6g} "
                f"K={stage_k} "
                f"rollout_mean_m={mean_err:.6f} rollout_max_m={max_err:.6f} best_m={best_to_log:.6f} "
                f"effK_mean={rollout_stats['effective_k_mean']:.2f} effK_max={rollout_stats['effective_k_max']:.0f} "
                f"lr={float(current_lr):.3g} elapsed_s={elapsed:.1f}",
                flush=True,
            )
            loss_value = float(loss.detach().cpu())
            writer.add_scalar("train/loss", loss_value, step)
            writer.add_scalar("loss/supervised", loss_value, step)
            writer.add_scalar("eval/rollout_mean_m", mean_err, step)
            writer.add_scalar("eval/rollout_max_m", max_err, step)
            writer.add_scalar("eval/best_m", best_to_log, step)
            writer.add_scalar("curriculum/rollout_k", int(stage_k), step)
            writer.add_scalar("train/effective_rollout_k_mean", rollout_stats["effective_k_mean"], step)
            writer.add_scalar("train/effective_rollout_k_max", rollout_stats["effective_k_max"], step)
            writer.add_scalar("curriculum/effective_rollout_k_mean", rollout_stats["effective_k_mean"], step)
            writer.add_scalar("curriculum/effective_rollout_k_max", rollout_stats["effective_k_max"], step)
            writer.add_scalar("time/elapsed_s", elapsed, step)
            writer.flush()

    last = save_named_checkpoint(run_dir, run_id, "last", model, optimizer, train_steps, best, cfg, metadata)
    writer.close()
    assert_tensorboard_event_file(tb_dir)
    refresh_tensorboard_async()
    print(f"saved {last}", flush=True)


if __name__ == "__main__":
    main()
