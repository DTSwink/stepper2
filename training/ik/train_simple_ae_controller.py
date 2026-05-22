from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

try:
    from .bootstrap import PROJECT_ROOT, ensure_paths
    from .naming import checkpoint_path, ik_run_id
    from . import ik_core as tl
    from .train_simple_autoencoder import SimpleAEConfig, SimpleAutoencoder
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    from naming import checkpoint_path, ik_run_id
    import ik_core as tl
    from train_simple_autoencoder import SimpleAEConfig, SimpleAutoencoder

ensure_paths()


DEFAULT_WALK_F = PROJECT_ROOT / "ue5" / "animations_omni_only_full" / "npz_final" / "M_Neutral_Walk_Loop_F.npz"
RUNS_DIR = PROJECT_ROOT / "training" / "runs"
DEFAULT_AE_GLOB = "*_ik_simple_ae_*"
SIMPLE_CONTROLLER_AE_KIND = "simple_controller_io_autoencoder"
LEGACY_SIMPLE_AE_KIND = "simple_" + "ag" + "ent_io_autoencoder"

BATCH_SIZE = 4096
ROLLOUT_SCHEDULE = (1, 2, 8, 16, 32)
ROLLOUT_STAGE_STEPS = (3000, 1000, 1500, 1500, 2500)
ROLLOUT_K = 32
LEARNING_RATE = 1e-4
STAGE_LEARNING_RATES = {
    1: 1e-4,
    2: 8e-5,
    8: 5e-5,
    16: 2e-5,
    32: 1e-5,
}
LOG_EVERY = 250
VALIDATION_ROWS = 256
RUN_FK_DIAGNOSTIC = False
AE_SCORE_OUTPUT_ONLY = True
NAN_METRIC = float("nan")


def refresh_tensorboard_async() -> None:
    script = PROJECT_ROOT / "training" / "ik" / "launch_tensorboard_latest.ps1"
    if not script.exists():
        return
    kwargs = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
    }
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        subprocess.Popen(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            cwd=str(PROJECT_ROOT),
            **kwargs,
        )
    except Exception as exc:
        print(f"tensorboard refresh skipped: {exc}", flush=True)


@dataclass(frozen=True)
class StartPool:
    clip_ids: torch.Tensor
    starts: torch.Tensor

    @property
    def row_count(self) -> int:
        return int(self.starts.numel())


class SimpleClipStore:
    def __init__(self, clips: list[tl.MotionClip], cfg: tl.TrainConfig, device: torch.device):
        if not clips:
            raise ValueError("SimpleClipStore needs at least one clip")
        first = clips[0]
        for clip in clips[1:]:
            if clip.body_names != first.body_names or clip.parents_body_list != first.parents_body_list:
                raise ValueError(f"Skeleton mismatch: {clip.path} vs {first.path}")

        self.clips = clips
        self.cfg = cfg
        self.device = device
        self.prototype = first
        self.J = int(first.J)
        self.Jcore = int(first.Jcore)
        self.ik_payload_dim = int(getattr(first, "ik_payload_dim", 0))
        self.pelvis = int(first.pelvis)

        tensors = [clip.tensors(device) for clip in clips]
        lengths = [int(clip.T) for clip in clips]
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)

        self.frame_offsets = torch.tensor(offsets[:-1], dtype=torch.long, device=device)
        self.lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.periods = torch.tensor([int(clip.cyclic_period) for clip in clips], dtype=torch.long, device=device)
        self.cyclic = torch.tensor([bool(clip.cyclic_animation) for clip in clips], dtype=torch.bool, device=device)

        self.root_pos = torch.cat([t["root_pos"] for t in tensors], dim=0)
        self.root_rot = torch.cat([t["root_rot"] for t in tensors], dim=0)
        self.pelvis_local_pos = torch.cat([t["pelvis_local_pos"] for t in tensors], dim=0)
        self.pelvis_rot6 = torch.cat([t["pelvis_rot6"] for t in tensors], dim=0)
        self.non_pelvis_rot6 = torch.cat([t["non_pelvis_rot6"] for t in tensors], dim=0)
        self.core_non_pelvis_rot6 = torch.cat([t["core_non_pelvis_rot6"] for t in tensors], dim=0)
        self.canonical_pos = torch.cat([t["canonical_pos"] for t in tensors], dim=0)
        self.ik_payload = torch.cat([t["ik_payload"] for t in tensors], dim=0)

        self.root0_pos = torch.stack([t["root_pos"][0] for t in tensors], dim=0)
        self.root0_rot = torch.stack([t["root_rot"][0] for t in tensors], dim=0)
        self.root0_inv = self.root0_rot.transpose(-1, -2)
        self.end_pos = torch.stack([t["root_pos"][clip.cyclic_period] for clip, t in zip(clips, tensors)], dim=0)
        self.end_rot = torch.stack([t["root_rot"][clip.cyclic_period] for clip, t in zip(clips, tensors)], dim=0)
        self.cycle_pos = torch.matmul((self.end_pos - self.root0_pos).unsqueeze(1), self.root0_inv).squeeze(1)
        self.cycle_rot = self.end_rot @ self.root0_inv

        self.target_output = self._build_target_output()
        self.input_root_features = self._build_input_root_features()

    def _build_target_output(self) -> torch.Tensor:
        b = self.pelvis_local_pos.shape[0]
        return torch.cat(
            (
                self.pelvis_local_pos,
                self.pelvis_rot6,
                self.core_non_pelvis_rot6.reshape(b, -1),
                self.ik_payload,
            ),
            dim=-1,
        )

    def _build_input_root_features(self) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        future_steps = int(self.cfg.future_window)
        feature_dim = 3 + future_steps * 4
        for clip_id, clip in enumerate(self.clips):
            features = torch.zeros((int(clip.T), feature_dim), dtype=torch.float32, device=self.device)
            if int(clip.T) <= 1:
                chunks.append(features)
                continue

            if clip.cyclic_animation:
                period = max(1, int(clip.cyclic_period))
                rows = torch.arange(period, dtype=torch.long, device=self.device)
                cur_idx = rows.clone()
                cur_idx[0] = period
            else:
                max_cur = int(clip.T) - transition_feature_horizon(self.cfg) - 1
                if max_cur < 1:
                    chunks.append(features)
                    continue
                rows = torch.arange(1, max_cur + 1, dtype=torch.long, device=self.device)
                cur_idx = rows

            clip_ids = torch.full((cur_idx.numel(),), clip_id, dtype=torch.long, device=self.device)
            prev_idx = cur_idx - 1
            prev_pos, _prev_rot, prev_yaw, prev_heading = self.root_state(clip_ids, prev_idx)
            cur_pos, _cur_rot, cur_yaw, cur_heading = self.root_state(clip_ids, cur_idx)
            delta_local = torch.matmul((cur_pos - prev_pos).unsqueeze(1), prev_heading).squeeze(1)
            root_feat = torch.stack(
                (
                    delta_local[:, 0] / self.cfg.max_speed_scale_final,
                    delta_local[:, 2] / self.cfg.max_speed_scale_final,
                    tl.wrap_angle(cur_yaw - prev_yaw) / self.cfg.max_turn_rate_scale_final,
                ),
                dim=-1,
            )

            future_offsets = torch.arange(1, future_steps + 1, device=self.device, dtype=cur_idx.dtype)
            flat_clip_ids = clip_ids.reshape(-1, 1).expand(-1, future_steps).reshape(-1)
            flat_idx = (cur_idx.reshape(-1, 1) + future_offsets.reshape(1, future_steps)).reshape(-1)
            fut_pos, _fut_rot, fut_yaw, _fut_heading = self.root_state(flat_clip_ids, flat_idx)
            fut_pos = fut_pos.reshape(cur_idx.numel(), future_steps, 3)
            fut_yaw = fut_yaw.reshape(cur_idx.numel(), future_steps)
            fut_local = torch.matmul((fut_pos - cur_pos[:, None, :]).unsqueeze(-2), cur_heading[:, None]).squeeze(-2)
            scale = future_offsets.to(dtype=fut_local.dtype).reshape(1, future_steps) * self.cfg.max_speed_scale_final
            dyaw = tl.wrap_angle(fut_yaw - cur_yaw[:, None])
            future_feat = torch.stack(
                (
                    torch.clamp(fut_local[:, :, 0] / scale, -2.0, 2.0),
                    torch.clamp(fut_local[:, :, 2] / scale, -2.0, 2.0),
                    torch.cos(dyaw),
                    torch.sin(dyaw),
                ),
                dim=-1,
            ).reshape(cur_idx.numel(), future_steps * 4)
            features[rows] = torch.cat((root_feat, future_feat), dim=-1)
            chunks.append(features)
        return torch.cat(chunks, dim=0)

    def frame_index(self, clip_ids: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        clip_ids = clip_ids.to(self.device).long()
        idx = idx.to(self.device).long()
        periods = self.periods.index_select(0, clip_ids).clamp_min(1)
        cyclic = self.cyclic.index_select(0, clip_ids)
        logical = torch.where(cyclic, torch.remainder(idx, periods), idx)
        return self.frame_offsets.index_select(0, clip_ids) + logical

    def root_state(self, clip_ids: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        clip_ids = clip_ids.to(self.device).long()
        idx = idx.to(self.device).long()
        frame = self.frame_index(clip_ids, idx)
        base_pos = self.root_pos.index_select(0, frame)
        base_rot = self.root_rot.index_select(0, frame)
        root0_pos = self.root0_pos.index_select(0, clip_ids)
        root0_rot = self.root0_rot.index_select(0, clip_ids)
        root0_inv = self.root0_inv.index_select(0, clip_ids)
        periods = self.periods.index_select(0, clip_ids).clamp_min(1)
        cyclic = self.cyclic.index_select(0, clip_ids)
        cycles = torch.where(cyclic, torch.div(idx, periods, rounding_mode="floor"), torch.zeros_like(idx))

        rel_pos = torch.matmul((base_pos - root0_pos).unsqueeze(1), root0_inv).squeeze(1)
        rel_rot = base_rot @ root0_inv
        cycle_pos = self.cycle_pos.index_select(0, clip_ids)
        cycle_rot = self.cycle_rot.index_select(0, clip_ids)
        max_cycles = int(cycles.max().detach().cpu()) if cycles.numel() else 0
        for cycle in range(max_cycles):
            mask = (cycles > cycle).reshape(-1, 1)
            next_pos = torch.matmul(rel_pos.unsqueeze(1), cycle_rot).squeeze(1) + cycle_pos
            next_rot = rel_rot @ cycle_rot
            rel_pos = torch.where(mask, next_pos, rel_pos)
            rel_rot = torch.where(mask.unsqueeze(-1), next_rot, rel_rot)

        pos = torch.matmul(rel_pos.unsqueeze(1), root0_rot).squeeze(1) + root0_pos
        rot = rel_rot @ root0_rot
        yaw = tl.heading_yaw_from_root(rot)
        heading = tl.yaw_to_row_matrix(yaw)
        return pos, rot, yaw, heading

    def get_pose(self, clip_ids: torch.Tensor, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        frame = self.frame_index(clip_ids, idx)
        return {
            "pelvis_pos": self.pelvis_local_pos.index_select(0, frame),
            "pelvis_rot6": self.pelvis_rot6.index_select(0, frame),
            "nonpelvis_rot6": self.non_pelvis_rot6.index_select(0, frame),
            "canon_pos": self.canonical_pos.index_select(0, frame),
            "core_nonpelvis_rot6": self.core_non_pelvis_rot6.index_select(0, frame),
            "ik_payload": self.ik_payload.index_select(0, frame),
        }

    def get_target_output(self, clip_ids: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return self.target_output.index_select(0, self.frame_index(clip_ids, idx))

    def get_input_root_features(self, clip_ids: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return self.input_root_features.index_select(0, self.frame_index(clip_ids, idx))


def apply_config_dict(cfg: tl.TrainConfig, values: dict) -> None:
    valid = {field.name for field in fields(tl.TrainConfig)}
    for key, value in values.items():
        if key not in valid:
            continue
        current = getattr(cfg, key)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(cfg, key, value)


def make_cfg(device: torch.device, ae_ckpt: dict) -> tl.TrainConfig:
    cfg = tl.TrainConfig()
    apply_config_dict(cfg, ae_ckpt.get("locomotion_config", {}))
    cfg.pose_representation = tl.IK_POSE_REPRESENTATION
    cfg.predict_residual = True
    cfg.zero_init_output = True
    cfg.hidden_dim = 512
    cfg.num_hidden_layers = 2
    cfg.learning_rate = LEARNING_RATE
    cfg.batch_size = BATCH_SIZE
    cfg.live_viewer = False
    cfg.visual_reporter = False
    cfg.update_comparison_on_exit = False
    cfg.use_torch_compile = False
    cfg.device = str(device)
    return cfg


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def npz_paths_from_text(path_text: str) -> list[Path]:
    paths: list[Path] = []
    for raw_part in str(path_text or "").split(";"):
        part = raw_part.strip()
        if not part:
            continue
        path = resolve_path(part)
        if path.is_dir():
            found = sorted(path.glob("*.npz"))
            if not found:
                raise FileNotFoundError(f"No .npz files found in {path}")
            paths.extend(found)
        else:
            if not path.exists():
                raise FileNotFoundError(f"NPZ path does not exist: {path}")
            if path.suffix.lower() != ".npz":
                raise ValueError(f"Expected .npz file, got: {path}")
            paths.append(path)
    return paths


def resolve_clip_specs(
    npz_text: str | None,
    periodic_text: str | None,
    nonperiodic_text: str | None,
) -> list[tuple[Path, bool]]:
    specs: list[tuple[Path, bool]] = []
    for path in npz_paths_from_text(periodic_text or ""):
        specs.append((path, True))
    for path in npz_paths_from_text(nonperiodic_text or ""):
        specs.append((path, False))
    if specs:
        return specs
    if npz_text:
        return [(path, True) for path in npz_paths_from_text(npz_text)]
    if not DEFAULT_WALK_F.exists():
        raise FileNotFoundError(f"Default walk-forward NPZ not found: {DEFAULT_WALK_F}")
    return [(DEFAULT_WALK_F.resolve(), True)]


def load_clips(specs: list[tuple[Path, bool]], cfg: tl.TrainConfig) -> list[tl.MotionClip]:
    clips = [tl.MotionClip(path, cfg, cyclic_animation=cyclic) for path, cyclic in specs]
    first_names = clips[0].body_names
    first_parents = clips[0].parents_body_list
    for clip in clips[1:]:
        if clip.body_names != first_names or clip.parents_body_list != first_parents:
            raise ValueError(f"Skeleton mismatch: {clip.path} vs {clips[0].path}")
    return clips


def latest_simple_ae_checkpoint() -> Path:
    candidates: list[Path] = []
    for run_dir in RUNS_DIR.glob(DEFAULT_AE_GLOB):
        ckpt_dir = run_dir / "checkpoints"
        if ckpt_dir.exists():
            candidates.extend(sorted(ckpt_dir.glob("*_best.pt"), key=lambda p: p.stat().st_mtime, reverse=True))
    if not candidates:
        raise FileNotFoundError(f"No simple AE checkpoints found under {RUNS_DIR}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0].resolve()


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


def transition_feature_horizon(cfg: tl.TrainConfig) -> int:
    root_lookahead_steps = max(0, int(getattr(cfg, "root_lookahead_steps", 0)))
    return max(int(cfg.future_window), root_lookahead_steps + 1)


def max_start_for_clip(clip: tl.MotionClip, cfg: tl.TrainConfig, rollout_k: int) -> int:
    if clip.cyclic_animation:
        return int(clip.cyclic_period) - 1
    return int(clip.T) - transition_feature_horizon(cfg) - max(1, int(rollout_k))


def build_start_pool(store: SimpleClipStore, rollout_k: int) -> StartPool:
    clip_chunks: list[torch.Tensor] = []
    start_chunks: list[torch.Tensor] = []
    for clip_id, clip in enumerate(store.clips):
        max_start = max_start_for_clip(clip, store.cfg, rollout_k)
        if max_start < 1:
            continue
        starts = torch.arange(1, max_start + 1, dtype=torch.long, device=store.device)
        clip_chunks.append(torch.full_like(starts, int(clip_id)))
        start_chunks.append(starts)
    if not start_chunks:
        raise ValueError(f"No valid full-window rollout starts found for K={rollout_k}")
    return StartPool(torch.cat(clip_chunks, dim=0), torch.cat(start_chunks, dim=0))


def rollout_values_for(max_k: int) -> tuple[int, ...]:
    values: list[int] = []
    k = max(1, int(max_k))
    while k > 1:
        values.append(k)
        k = max(1, k // 2)
    values.append(1)
    return tuple(dict.fromkeys(values))


def mixed_rollout_enabled(rollout_k: int) -> bool:
    return int(rollout_k) >= int(ROLLOUT_K)


def build_start_pools(store: SimpleClipStore, rollout_values: tuple[int, ...]) -> dict[int, StartPool]:
    return {int(k): build_start_pool(store, int(k)) for k in rollout_values}


def sample_from_pool(pool: StartPool, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    count = max(1, int(count))
    if count == pool.row_count:
        order = torch.randperm(pool.row_count, device=pool.starts.device)
        return pool.clip_ids.index_select(0, order), pool.starts.index_select(0, order)
    rows = torch.randint(0, pool.row_count, (count,), device=pool.starts.device)
    return pool.clip_ids.index_select(0, rows), pool.starts.index_select(0, rows)


def sample_effective_rollout_k(batch_size: int, rollout_k: int, device: torch.device) -> torch.Tensor:
    batch_size = max(1, int(batch_size))
    rollout_k = max(1, int(rollout_k))
    if not mixed_rollout_enabled(rollout_k):
        return torch.full((batch_size,), rollout_k, dtype=torch.long, device=device)
    values = rollout_values_for(rollout_k)
    remaining = batch_size
    chunks: list[torch.Tensor] = []
    for value in values[:-1]:
        count = remaining // 2
        if count:
            chunks.append(torch.full((count,), int(value), dtype=torch.long, device=device))
        remaining -= count
    chunks.append(torch.full((remaining,), int(values[-1]), dtype=torch.long, device=device))
    effective_k = torch.cat(chunks, dim=0)
    return effective_k.index_select(0, torch.randperm(batch_size, device=device))


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


def sample_rollout_rows(start_pools: dict[int, StartPool], effective_k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    clip_ids = torch.empty_like(effective_k)
    starts = torch.empty_like(effective_k)
    for rollout_k, pool in start_pools.items():
        rows = (effective_k == int(rollout_k)).nonzero(as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        row_clip_ids, row_starts = sample_from_pool(pool, int(rows.numel()))
        clip_ids[rows] = row_clip_ids
        starts[rows] = row_starts
    return clip_ids, starts


def stage_for_step(step: int) -> tuple[int, int, int, int]:
    start = 1
    for stage_idx, (rollout_k, stage_steps) in enumerate(zip(ROLLOUT_SCHEDULE, ROLLOUT_STAGE_STEPS)):
        end = start + int(stage_steps) - 1
        if int(step) <= end:
            return stage_idx, int(rollout_k), start, end
        start = end + 1
    return len(ROLLOUT_SCHEDULE) - 1, int(ROLLOUT_SCHEDULE[-1]), start, start


def stage_learning_rate(rollout_k: int) -> float:
    return float(STAGE_LEARNING_RATES.get(int(rollout_k), LEARNING_RATE))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def make_adamw(params, lr: float, device: torch.device, capturable: bool = False) -> torch.optim.Optimizer:
    kwargs = {"lr": lr, "weight_decay": 0.0}
    if device.type == "cuda":
        kwargs["fused"] = True
        if capturable:
            kwargs["capturable"] = True
    try:
        return torch.optim.AdamW(params, **kwargs)
    except (RuntimeError, TypeError):
        kwargs.pop("fused", None)
        kwargs.pop("capturable", None)
        return torch.optim.AdamW(params, **kwargs)


def payload_slice(store: SimpleClipStore) -> slice:
    start = 3 + 6 + store.Jcore * 6
    return slice(start, start + store.ik_payload_dim)


def target_state(store: SimpleClipStore, clip_ids: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vec = store.get_target_output(clip_ids, idx)
    return vec, vec[:, :3], vec[:, payload_slice(store)]


def fast_clean_6d(d6: torch.Tensor) -> torch.Tensor:
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-8)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1, eps=1e-8)
    return torch.cat((b1, b2), dim=-1)


def fast_clean_ik_payload(payload: torch.Tensor) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for spec in tl.IK_PAYLOAD_SLICES:
        pos_slice = spec["pos"]
        rot_slice = spec["rot6"]
        pole_slice = spec["pole"]
        toe_slice = spec["toe_float"]
        assert isinstance(pos_slice, slice)
        assert isinstance(rot_slice, slice)
        assert isinstance(pole_slice, slice)
        parts.append(payload[:, pos_slice])
        parts.append(fast_clean_6d(payload[:, rot_slice]))
        parts.append(payload[:, pole_slice])
        if toe_slice is not None:
            assert isinstance(toe_slice, slice)
            parts.append(payload[:, toe_slice])
    return torch.cat(parts, dim=-1)


def clean_output_vector(raw: torch.Tensor, store: SimpleClipStore) -> torch.Tensor:
    b = raw.shape[0]
    cursor = 0
    pelvis_pos = raw[:, cursor : cursor + 3]
    cursor += 3
    pelvis_rot6 = fast_clean_6d(raw[:, cursor : cursor + 6])
    cursor += 6
    core_dim = store.Jcore * 6
    core_rot6 = fast_clean_6d(raw[:, cursor : cursor + core_dim].reshape(-1, 6)).reshape(b, store.Jcore, 6)
    cursor += core_dim
    payload = fast_clean_ik_payload(raw[:, cursor : cursor + store.ik_payload_dim])
    return torch.cat((pelvis_pos, pelvis_rot6, core_rot6.reshape(b, -1), payload), dim=-1)


def predicted_state_from_raw(raw: torch.Tensor, store: SimpleClipStore) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vec = clean_output_vector(raw, store)
    return vec, vec[:, :3], vec[:, payload_slice(store)]


def predicted_state_from_vector(vec: torch.Tensor, store: SimpleClipStore) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return vec, vec[:, :3], vec[:, payload_slice(store)]


def model_forward(model: torch.nn.Module, inp: torch.Tensor, cur_vec: torch.Tensor, cfg: tl.TrainConfig) -> torch.Tensor:
    raw = model(inp)
    if cfg.predict_residual:
        raw = cur_vec + raw
    return raw


def build_controller_input(
    store: SimpleClipStore,
    clip_ids: torch.Tensor,
    cur_idx: torch.Tensor,
    prev_vec: torch.Tensor,
    cur_vec: torch.Tensor,
    prev_pelvis: torch.Tensor,
    cur_pelvis: torch.Tensor,
    prev_payload: torch.Tensor,
    cur_payload: torch.Tensor,
) -> torch.Tensor:
    pelvis_vel = (cur_pelvis - prev_pelvis) / store.cfg.pose_delta_scale_final
    payload_vel = (cur_payload - prev_payload).reshape(cur_idx.shape[0], -1) / store.cfg.pose_delta_scale_final
    root_features = store.get_input_root_features(clip_ids, cur_idx)
    return torch.cat((cur_vec, prev_vec, pelvis_vel, payload_vel, root_features), dim=-1)


def ae_score_rows(
    ae: SimpleAutoencoder,
    mean: torch.Tensor,
    std: torch.Tensor,
    controller_input: torch.Tensor,
    predicted_output: torch.Tensor,
) -> torch.Tensor:
    input_dim = int(controller_input.shape[-1])
    feature = torch.cat((controller_input, predicted_output), dim=-1)
    x = (feature - mean) / std
    recon = ae(x)
    if AE_SCORE_OUTPUT_ONLY:
        return (recon[:, input_dim:] - x[:, input_dim:]).square().mean(dim=-1)
    return (recon - x).square().mean(dim=-1)


def pure_ae_rollout_loss(
    model: torch.nn.Module,
    ae: SimpleAutoencoder,
    mean: torch.Tensor,
    std: torch.Tensor,
    store: SimpleClipStore,
    rollout_k: int,
    batch_size: int,
    start_pools: dict[int, StartPool],
) -> torch.Tensor:
    max_k = max(1, int(rollout_k))
    original_batch_size = max(1, int(batch_size))
    effective_k = sample_effective_rollout_k(original_batch_size, max_k, store.device)
    clip_ids, starts = sample_rollout_rows(start_pools, effective_k)
    cur_idx = starts
    prev_vec, prev_pelvis, prev_payload = target_state(store, clip_ids, cur_idx - 1)
    cur_vec, cur_pelvis, cur_payload = target_state(store, clip_ids, cur_idx)
    row_weight = (1.0 / effective_k.float()) / float(original_batch_size)
    total_loss = torch.zeros((), dtype=torch.float32, device=store.device)

    for step in range(max_k):
        inp = build_controller_input(
            store, clip_ids, cur_idx, prev_vec, cur_vec, prev_pelvis, cur_pelvis, prev_payload, cur_payload
        )
        raw = model_forward(model, inp, cur_vec, store.cfg)
        pred_vec = clean_output_vector(raw, store)
        total_loss = total_loss + (ae_score_rows(ae, mean, std, inp, pred_vec) * row_weight).sum()
        if step + 1 >= max_k:
            break

        continuing = effective_k > (step + 1)
        rows = continuing.nonzero(as_tuple=False).flatten()
        if rows.numel() == 0:
            break
        pred_vec = pred_vec.index_select(0, rows)
        clip_ids = clip_ids.index_select(0, rows)
        prev_vec = cur_vec.index_select(0, rows)
        prev_pelvis = cur_pelvis.index_select(0, rows)
        prev_payload = cur_payload.index_select(0, rows)
        cur_vec, cur_pelvis, cur_payload = predicted_state_from_vector(pred_vec, store)
        cur_idx = cur_idx.index_select(0, rows) + 1
        effective_k = effective_k.index_select(0, rows)
        row_weight = row_weight.index_select(0, rows)
    return total_loss


def pure_ae_rollout_loss_static(
    model: torch.nn.Module,
    ae: SimpleAutoencoder,
    mean: torch.Tensor,
    std: torch.Tensor,
    store: SimpleClipStore,
    rollout_k: int,
    batch_size: int,
    effective_k: torch.Tensor,
    clip_ids: torch.Tensor,
    starts: torch.Tensor,
) -> torch.Tensor:
    max_k = max(1, int(rollout_k))
    cur_idx = starts
    prev_vec, prev_pelvis, prev_payload = target_state(store, clip_ids, cur_idx - 1)
    cur_vec, cur_pelvis, cur_payload = target_state(store, clip_ids, cur_idx)
    row_weight = (1.0 / effective_k.float()) / float(max(1, int(batch_size)))
    total_loss = torch.zeros((), dtype=torch.float32, device=store.device)

    for step in range(max_k):
        inp = build_controller_input(
            store, clip_ids, cur_idx, prev_vec, cur_vec, prev_pelvis, cur_pelvis, prev_payload, cur_payload
        )
        raw = model_forward(model, inp, cur_vec, store.cfg)
        pred_vec = clean_output_vector(raw, store)
        active = effective_k > step
        total_loss = total_loss + (ae_score_rows(ae, mean, std, inp, pred_vec) * row_weight * active.float()).sum()
        if step + 1 >= max_k:
            break

        continuing = effective_k > (step + 1)
        next_vec, next_pelvis, next_payload = predicted_state_from_vector(pred_vec, store)
        mask = continuing[:, None]
        prev_vec = torch.where(mask, cur_vec, prev_vec)
        prev_pelvis = torch.where(mask, cur_pelvis, prev_pelvis)
        prev_payload = torch.where(mask, cur_payload, prev_payload)
        cur_vec = torch.where(mask, next_vec, cur_vec)
        cur_pelvis = torch.where(mask, next_pelvis, cur_pelvis)
        cur_payload = torch.where(mask, next_payload, cur_payload)
        cur_idx = torch.where(continuing, cur_idx + 1, cur_idx)
    return total_loss


class CudaGraphPureAEStep:
    kind = "cuda_graph_static_masked"

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        ae: SimpleAutoencoder,
        mean: torch.Tensor,
        std: torch.Tensor,
        store: SimpleClipStore,
        rollout_k: int,
        batch_size: int,
        start_pools: dict[int, StartPool],
    ):
        if store.device.type != "cuda":
            raise RuntimeError("CudaGraphPureAEStep requires CUDA")
        self.model = model
        self.optimizer = optimizer
        self.ae = ae
        self.mean = mean
        self.std = std
        self.store = store
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
        effective_k = sample_effective_rollout_k(self.batch_size, self.rollout_k, self.store.device)
        clip_ids, starts = sample_rollout_rows(self.start_pools, effective_k)
        self.effective_k.copy_(effective_k)
        self.clip_ids.copy_(clip_ids)
        self.starts.copy_(starts)

    def _loss(self) -> torch.Tensor:
        return pure_ae_rollout_loss_static(
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
        )

    def _capture(self) -> None:
        self._sample_into_static_buffers()
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(2):
                self._sample_into_static_buffers()
                self.optimizer.zero_grad(set_to_none=False)
                loss = self._loss()
                loss.backward()
                self.optimizer.step()
                del loss
        torch.cuda.current_stream().wait_stream(side_stream)
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


class EagerPureAEStep:
    kind = "eager"

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        ae: SimpleAutoencoder,
        mean: torch.Tensor,
        std: torch.Tensor,
        store: SimpleClipStore,
        rollout_k: int,
        batch_size: int,
        start_pools: dict[int, StartPool],
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

    def step(self) -> torch.Tensor:
        loss = pure_ae_rollout_loss(
            self.model,
            self.ae,
            self.mean,
            self.std,
            self.store,
            self.rollout_k,
            self.batch_size,
            self.start_pools,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss.detach()


def make_pure_ae_stepper(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ae: SimpleAutoencoder,
    mean: torch.Tensor,
    std: torch.Tensor,
    store: SimpleClipStore,
    rollout_k: int,
    batch_size: int,
    start_pools: dict[int, StartPool],
) -> CudaGraphPureAEStep | EagerPureAEStep:
    if store.device.type == "cuda":
        return CudaGraphPureAEStep(model, optimizer, ae, mean, std, store, rollout_k, batch_size, start_pools)
    return EagerPureAEStep(model, optimizer, ae, mean, std, store, rollout_k, batch_size, start_pools)


def validation_rows(pool: StartPool, max_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    if pool.row_count <= max_rows:
        return pool.clip_ids, pool.starts
    rows = torch.linspace(0, pool.row_count - 1, steps=max_rows, device=pool.starts.device).round().long().unique()
    return pool.clip_ids.index_select(0, rows), pool.starts.index_select(0, rows)


@torch.no_grad()
def validation_ae_score(
    model: torch.nn.Module,
    ae: SimpleAutoencoder,
    mean: torch.Tensor,
    std: torch.Tensor,
    store: SimpleClipStore,
    rollout_k: int,
    pool: StartPool,
) -> float:
    clip_ids, starts = validation_rows(pool, VALIDATION_ROWS)
    cur_idx = starts
    prev_vec, prev_pelvis, prev_payload = target_state(store, clip_ids, cur_idx - 1)
    cur_vec, cur_pelvis, cur_payload = target_state(store, clip_ids, cur_idx)
    total = 0.0
    count = 0
    for step in range(max(1, int(rollout_k))):
        inp = build_controller_input(
            store, clip_ids, cur_idx, prev_vec, cur_vec, prev_pelvis, cur_pelvis, prev_payload, cur_payload
        )
        raw = model_forward(model, inp, cur_vec, store.cfg)
        score = ae_score_rows(ae, mean, std, inp, clean_output_vector(raw, store))
        total += float(score.sum().detach().cpu())
        count += int(score.numel())
        if step + 1 >= int(rollout_k):
            break
        prev_vec = cur_vec
        prev_pelvis = cur_pelvis
        prev_payload = cur_payload
        cur_vec, cur_pelvis, cur_payload = predicted_state_from_raw(raw, store)
        cur_idx = cur_idx + 1
    return total / float(max(1, count))


def pose_rows(pose: dict[str, torch.Tensor], rows: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value.index_select(0, rows) for key, value in pose.items()}


@torch.no_grad()
def fk_positions_by_clip(
    store: SimpleClipStore,
    clip_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    pose: dict[str, torch.Tensor],
) -> torch.Tensor:
    out = torch.empty((clip_ids.shape[0], store.J, 3), dtype=root_pos.dtype, device=store.device)
    for clip_id in clip_ids.unique().tolist():
        rows = (clip_ids == int(clip_id)).nonzero(as_tuple=False).flatten()
        pos, _rot, _canon = tl.fk_from_pose(
            store.clips[int(clip_id)],
            root_pos.index_select(0, rows),
            root_rot.index_select(0, rows),
            pose_rows(pose, rows),
            store.device,
        )
        out[rows] = pos
    return out


@torch.no_grad()
def rollout_joint_error(
    model: torch.nn.Module,
    store: SimpleClipStore,
    rollout_k: int,
    pool: StartPool,
) -> tuple[float, float]:
    clip_ids, starts = validation_rows(pool, VALIDATION_ROWS)
    cur_idx = starts
    prev_vec, prev_pelvis, prev_payload = target_state(store, clip_ids, cur_idx - 1)
    cur_vec, cur_pelvis, cur_payload = target_state(store, clip_ids, cur_idx)
    total_error = 0.0
    total_frames = 0
    max_error = 0.0
    for step in range(max(1, int(rollout_k))):
        inp = build_controller_input(
            store, clip_ids, cur_idx, prev_vec, cur_vec, prev_pelvis, cur_pelvis, prev_payload, cur_payload
        )
        raw = model_forward(model, inp, cur_vec, store.cfg)
        pred_pose, _raw_pose = tl.output_to_pose(raw, store.prototype)
        target_idx = cur_idx + 1
        root_pos, root_rot, _yaw, _heading = store.root_state(clip_ids, target_idx)
        pred_global = fk_positions_by_clip(store, clip_ids, root_pos, root_rot, pred_pose)
        target_global = fk_positions_by_clip(store, clip_ids, root_pos, root_rot, store.get_pose(clip_ids, target_idx))
        per_frame = (pred_global - target_global).norm(dim=-1).mean(dim=-1)
        total_error += float(per_frame.sum().detach().cpu())
        total_frames += int(per_frame.numel())
        max_error = max(max_error, float(per_frame.max().detach().cpu()))
        if step + 1 >= int(rollout_k):
            break
        prev_vec = cur_vec
        prev_pelvis = cur_pelvis
        prev_payload = cur_payload
        cur_vec, cur_pelvis, cur_payload = predicted_state_from_raw(raw, store)
        cur_idx = target_idx
    return total_error / float(max(1, total_frames)), max_error


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
    specs = resolve_clip_specs(args.npz, args.periodic_folder, args.nonperiodic_folder)
    clips = load_clips(specs, cfg)
    input_dim, output_dim = tl.make_batch_dims(clips[0], cfg)
    expected_dim = int(ae_ckpt["schema"]["total_dim"])
    if input_dim + output_dim != expected_dim:
        raise ValueError(f"AE dim {expected_dim} does not match controller feature dim {input_dim + output_dim}")

    store = SimpleClipStore(clips, cfg, device)
    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    optimizer = make_adamw(model.parameters(), LEARNING_RATE, device, capturable=bool(device.type == "cuda"))

    stage_cache: dict[int, dict[str, object]] = {}
    for stage_k in ROLLOUT_SCHEDULE:
        rollout_values = rollout_values_for(stage_k) if mixed_rollout_enabled(stage_k) else (int(stage_k),)
        start_pools = build_start_pools(store, rollout_values)
        max_pool = start_pools[int(stage_k)]
        batch_size = min(BATCH_SIZE, max_pool.row_count)
        stage_cache[int(stage_k)] = {
            "rollout_values": rollout_values,
            "start_pools": start_pools,
            "max_pool": max_pool,
            "row_count": max_pool.row_count,
            "batch_size": batch_size,
            "rollout_stats": rollout_stat_summary(batch_size, int(stage_k)),
            "pool_rows": {str(int(k)): int(pool.row_count) for k, pool in start_pools.items()},
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
            "loss": "simple_ae_output_reconstruction",
            "ae_score_output_only": bool(AE_SCORE_OUTPUT_ONLY),
            "mixed_rollout_at_max": True,
            "pose_representation": tl.IK_POSE_REPRESENTATION,
            "test_set": False,
            "checkpoint_selection": "latest_stage_last",
        },
        "rollout_schedule": [int(k) for k in ROLLOUT_SCHEDULE],
        "rollout_stage_steps": [int(n) for n in ROLLOUT_STAGE_STEPS],
        "rollout_k": int(ROLLOUT_K),
        "row_count": int(final_cache["row_count"]),
        "batch_size": int(final_cache["batch_size"]),
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
        "start_pool_rows": {str(k): v["pool_rows"] for k, v in stage_cache.items()},
        "stage_learning_rates": {str(k): float(stage_learning_rate(k)) for k in ROLLOUT_SCHEDULE},
    }
    config_payload = {"config": asdict(cfg), "metadata": metadata}
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    writer = SummaryWriter(log_dir=str(run_dir / "tb"), flush_secs=1)
    writer.add_text("config/json", f"```json\n{json.dumps(config_payload, indent=2)}\n```", 0)
    writer.flush()
    refresh_tensorboard_async()

    print(f"simple_ae_controller run={run_id} ae={ae_path} tensorboard_logdir={run_dir / 'tb'}", flush=True)
    last_loss = float("inf")
    init_path = save_controller_checkpoint(run_dir, run_id, "init", model, optimizer, 0, last_loss, 0, cfg, metadata)
    print(f"saved initial checkpoint {init_path}", flush=True)

    start = time.perf_counter()
    total_steps = sum(int(x) for x in ROLLOUT_STAGE_STEPS)
    step = 0
    for stage_idx, stage_k in enumerate(ROLLOUT_SCHEDULE):
        stage_steps = int(ROLLOUT_STAGE_STEPS[stage_idx])
        lr = stage_learning_rate(int(stage_k))
        set_optimizer_lr(optimizer, lr)
        cache = stage_cache[int(stage_k)]
        model.train()
        stepper = make_pure_ae_stepper(
            model,
            optimizer,
            ae,
            mean,
            std,
            store,
            int(stage_k),
            int(cache["batch_size"]),
            cache["start_pools"],
        )
        print(
            f"stage={stage_idx + 1}/{len(ROLLOUT_SCHEDULE)} K={stage_k} "
            f"steps={stage_steps} batch={int(cache['batch_size'])} lr={lr:.3g} stepper={stepper.kind}",
            flush=True,
        )
        for stage_step in range(1, stage_steps + 1):
            step += 1
            loss = stepper.step()
            last_loss = float(loss.detach().cpu())
            if stage_step == 1 or stage_step % LOG_EVERY == 0 or stage_step == stage_steps:
                model.eval()
                mean_err = NAN_METRIC
                max_err = NAN_METRIC
                if RUN_FK_DIAGNOSTIC:
                    mean_err, max_err = rollout_joint_error(model, store, int(stage_k), cache["max_pool"])
                latest = save_controller_checkpoint(
                    run_dir, run_id, "latest", model, optimizer, step, last_loss, int(stage_k), cfg, metadata
                )
                elapsed = time.perf_counter() - start
                stats = cache["rollout_stats"]
                gt_text = (
                    f"gt_mean_m={mean_err:.6f} gt_max_m={max_err:.6f}"
                    if RUN_FK_DIAGNOSTIC
                    else "gt_diag=off"
                )
                print(
                    f"step={step:05d} K={stage_k} train_loss={last_loss:.6g} "
                    f"{gt_text} effK_mean={stats['effective_k_mean']:.2f} lr={lr:.3g} elapsed_s={elapsed:.1f}",
                    flush=True,
                )
                writer.add_scalar("loss/train_ae", last_loss, step)
                if RUN_FK_DIAGNOSTIC:
                    writer.add_scalar("eval/rollout_mean_m", mean_err, step)
                    writer.add_scalar("eval/rollout_max_m", max_err, step)
                writer.add_scalar("curriculum/rollout_k", int(stage_k), step)
                writer.add_scalar("train/effective_rollout_k_mean", stats["effective_k_mean"], step)
                writer.add_scalar("train/effective_rollout_k_max", stats["effective_k_max"], step)
                writer.add_scalar("time/elapsed_s", elapsed, step)
                writer.flush()
                print(f"checkpoint_latest={latest}", flush=True)
                model.train()
        stage_path = save_controller_checkpoint(
            run_dir, run_id, f"stage_K{int(stage_k)}", model, optimizer, step, last_loss, int(stage_k), cfg, metadata
        )
        print(f"saved stage checkpoint {stage_path}", flush=True)
        del stepper

    last = save_controller_checkpoint(run_dir, run_id, "last", model, optimizer, total_steps, last_loss, ROLLOUT_K, cfg, metadata)
    writer.close()
    print(f"saved {last}", flush=True)


if __name__ == "__main__":
    main()
