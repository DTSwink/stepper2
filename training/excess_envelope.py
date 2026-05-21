from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

import contact_physics as cp
import train_locomotion as tl


@dataclass(frozen=True)
class ExcessEnvelopeConfig:
    margin: float = 1.05
    knn: int = 32
    cache_dir: str = "training/runs/cache/excess_envelopes"
    chunk_size: int = 4096


def _clip_has_hard_idle_envelope(clip: tl.MotionClip) -> bool:
    return "idle" in Path(clip.path).stem.lower()


def signed_horizontal_angle(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a2 = a[..., [0, 2]]
    b2 = b[..., [0, 2]]
    an = torch.linalg.norm(a2, dim=-1)
    bn = torch.linalg.norm(b2, dim=-1)
    dot = (a2 * b2).sum(dim=-1)
    cross = a2[..., 0] * b2[..., 1] - a2[..., 1] * b2[..., 0]
    angle = torch.atan2(cross, dot)
    valid = torch.logical_and(an > eps, bn > eps)
    return torch.where(valid, angle, torch.zeros_like(angle))


@torch.no_grad()
def root_window_feature(
    clip: tl.MotionClip,
    cur_idx: torch.Tensor,
    cfg: tl.TrainConfig,
    device: torch.device,
) -> torch.Tensor:
    """Return [yaw_delta_to_window_end, horizontal_bend_angle] for each frame."""
    cur_idx = cur_idx.to(device=device, dtype=torch.long)
    prev_idx = cur_idx - 1
    fut_idx = cur_idx + int(cfg.future_window)
    if not clip.cyclic_animation:
        fut_idx = torch.clamp(fut_idx, max=int(clip.T) - 1)
    prev_pos, _prev_rot, prev_yaw, _prev_heading = tl.root_state(clip, prev_idx, cfg, device)
    cur_pos, _cur_rot, _cur_yaw, _cur_heading = tl.root_state(clip, cur_idx, cfg, device)
    fut_pos, _fut_rot, fut_yaw, _fut_heading = tl.root_state(clip, fut_idx, cfg, device)
    yaw_delta = tl.wrap_angle(fut_yaw - prev_yaw)
    bend = signed_horizontal_angle(cur_pos - prev_pos, fut_pos - cur_pos)
    return torch.stack((yaw_delta, bend), dim=-1)


@torch.no_grad()
def clip_excess_reference_values(
    clip: tl.MotionClip,
    cfg: tl.TrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return flat frame indices, root features, slide reference, and yaw reference.

    Rows correspond to transitions cur -> cur+1. Frame 0 is skipped because the
    model input convention needs t-1. Non-cyclic clips stop before future-root
    information would run past the end. Cyclic clips use [1, cyclic_period-1].
    """
    if clip.cyclic_animation:
        end = int(clip.cyclic_period) - 1
    else:
        end = int(clip.T) - int(cfg.future_window) - 1
    if end < 1:
        empty_i = torch.empty((0,), dtype=torch.long, device=device)
        empty_f = torch.empty((0, 2), dtype=torch.float32, device=device)
        empty_v = torch.empty((0,), dtype=torch.float32, device=device)
        return empty_i, empty_f, empty_v, empty_v
    idx = torch.arange(1, end + 1, dtype=torch.long, device=device)
    cur_pos, cur_rot = tl.global_from_clip(clip, idx, cfg, device)
    next_pos, next_rot = tl.global_from_clip(clip, idx + 1, cfg, device)
    foot_indices = tuple(int(x) for x in clip.foot_indices_tensor.tolist())
    toe_indices = tuple(int(x) for x in clip.toe_indices_tensor.tolist())
    slide_speeds = cp.foot_slide_speeds(
        cur_pos,
        cur_rot,
        next_pos,
        next_rot,
        foot_indices,
        toe_indices,
        clip.fps,
    )
    yaw_speeds = cp.foot_vertical_yaw_speeds(
        cur_pos,
        cur_rot,
        next_pos,
        next_rot,
        foot_indices,
        toe_indices,
        clip.fps,
    )
    slide_reference, planted_foot, _heights = cp.planted_foot_values(
        slide_speeds,
        cur_pos,
        cur_rot,
        foot_indices,
        toe_indices,
    )
    yaw_reference = yaw_speeds.gather(-1, planted_foot.unsqueeze(-1)).squeeze(-1)
    features = root_window_feature(clip, idx, cfg, device)
    return idx, features, slide_reference, yaw_reference


def _cache_key(
    clips: list[tl.MotionClip],
    real_clip_indices: list[int],
    synthetic_clip_indices: set[int],
    cfg: tl.TrainConfig,
    env_cfg: ExcessEnvelopeConfig,
) -> str:
    payload = {
        "version": 9,
        "idle_hard_zero_envelope": True,
        "planted_foot_rule": "lowest_foot_collider_point",
        "future_window": int(cfg.future_window),
        "fps": int(cfg.fps),
        "position_unit_scale": float(cfg.position_unit_scale),
        "margin": float(env_cfg.margin),
        "knn": int(env_cfg.knn),
        "clips": [
            {
                "path": str(clip.path.resolve()),
                "mtime_ns": int(clip.path.stat().st_mtime_ns),
                "size": int(clip.path.stat().st_size),
                "cyclic": bool(clip.cyclic_animation),
                "period": int(clip.cyclic_period),
                "synthetic": int(i in synthetic_clip_indices),
                "real_source": int(i in set(real_clip_indices)),
            }
            for i, clip in enumerate(clips)
        ],
    }
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


@torch.no_grad()
def _knn_situation_upper_bound(
    target_features: torch.Tensor,
    source_features: torch.Tensor,
    source_values: torch.Tensor,
    k: int,
    chunk_size: int,
) -> torch.Tensor:
    if source_features.numel() == 0:
        raise ValueError("excess envelope needs at least one real ground-truth transition")
    k = max(1, min(int(k), int(source_features.shape[0])))
    chunks = []
    source_features_n = source_features / torch.pi
    target_features_n = target_features / torch.pi
    for start in range(0, int(target_features.shape[0]), int(chunk_size)):
        chunk = target_features_n[start : start + int(chunk_size)]
        dist = torch.cdist(chunk, source_features_n)
        kth = torch.topk(dist, k=k, largest=False, dim=-1).values[:, -1]
        near = dist <= (kth[:, None] + 1e-8)
        values = source_values.unsqueeze(0).expand_as(dist)
        chunks.append(torch.where(near, values, torch.full_like(values, -torch.inf)).amax(dim=-1))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def build_excess_envelope(
    clips: list[tl.MotionClip],
    cfg: tl.TrainConfig,
    device: torch.device,
    synthetic_clip_indices: set[int] | None = None,
    real_clip_indices: list[int] | None = None,
    env_cfg: ExcessEnvelopeConfig | None = None,
) -> dict[str, torch.Tensor | dict[str, float | int | str]]:
    synthetic_clip_indices = synthetic_clip_indices or set()
    real_clip_indices = real_clip_indices or [i for i in range(len(clips)) if i not in synthetic_clip_indices]
    env_cfg = env_cfg or ExcessEnvelopeConfig()
    frame_count = sum(int(clip.T) for clip in clips)
    flat_slide_values = torch.zeros((frame_count,), dtype=torch.float32, device=device)
    flat_yaw_values = torch.zeros((frame_count,), dtype=torch.float32, device=device)
    flat_features = torch.zeros((frame_count, 2), dtype=torch.float32, device=device)
    valid_mask = torch.zeros((frame_count,), dtype=torch.bool, device=device)
    idle_valid_mask = torch.zeros((frame_count,), dtype=torch.bool, device=device)
    real_valid_mask = torch.zeros((frame_count,), dtype=torch.bool, device=device)
    source_features = []
    source_slide = []
    source_yaw = []
    offsets = []
    offset = 0
    for clip in clips:
        offsets.append(offset)
        offset += int(clip.T)
    for ci, clip in enumerate(clips):
        idx, features, slide_reference, yaw_reference = clip_excess_reference_values(clip, cfg, device)
        if idx.numel() == 0:
            continue
        flat = idx + int(offsets[ci])
        flat_features.index_copy_(0, flat, features)
        flat_slide_values.index_copy_(0, flat, slide_reference)
        flat_yaw_values.index_copy_(0, flat, yaw_reference)
        valid_mask.index_fill_(0, flat, True)
        if _clip_has_hard_idle_envelope(clip):
            idle_valid_mask.index_fill_(0, flat, True)
        if ci in real_clip_indices:
            real_valid_mask.index_fill_(0, flat, True)
            source_features.append(features)
            source_slide.append(slide_reference)
            source_yaw.append(yaw_reference)
    if not source_features:
        raise ValueError("excess envelope could not find any real clip transitions")
    real_features = torch.cat(source_features, dim=0)
    real_slide = torch.cat(source_slide, dim=0)
    real_yaw = torch.cat(source_yaw, dim=0)
    target_features = flat_features[valid_mask]
    slide_bound_valid = _knn_situation_upper_bound(
        target_features,
        real_features,
        real_slide,
        env_cfg.knn,
        env_cfg.chunk_size,
    )
    yaw_bound_valid = _knn_situation_upper_bound(
        target_features,
        real_features,
        real_yaw,
        env_cfg.knn,
        env_cfg.chunk_size,
    )
    margin = float(env_cfg.margin)
    slide_bound = torch.zeros_like(flat_slide_values)
    yaw_bound = torch.zeros_like(flat_yaw_values)
    slide_bound[valid_mask] = slide_bound_valid * margin
    yaw_bound[valid_mask] = yaw_bound_valid * margin
    # Invalid frames should never be queried in normal training, but use a safe,
    # permissive fallback to avoid exploding diagnostics if a bug does query them.
    slide_bound[~valid_mask] = real_slide.max() * margin
    yaw_bound[~valid_mask] = real_yaw.max() * margin
    # Idle is a deliberate exception: the target behavior is planted feet, not
    # permission to reproduce tiny GT capture/retarget jitter.
    slide_bound[idle_valid_mask] = 0.0
    yaw_bound[idle_valid_mask] = 0.0
    return {
        "slide_bound_mps": slide_bound,
        "yaw_excess_bound_radps": yaw_bound,
        "features": flat_features,
        "valid_mask": valid_mask,
        "real_valid_mask": real_valid_mask,
        "groundtruth_slide_mps": flat_slide_values,
        "groundtruth_yaw_excess_radps": flat_yaw_values,
        "metadata": {
            "source_real_transitions": int(real_features.shape[0]),
            "target_transitions": int(valid_mask.sum().item()),
            "margin": float(env_cfg.margin),
            "knn": int(env_cfg.knn),
            "max_real_slide_mps": float(real_slide.max().detach().cpu()),
            "max_real_yaw_excess_radps": float(real_yaw.max().detach().cpu()),
            "linear_bound_reduction": "max_nearest_same_situation",
            "angular_bound_reduction": "max_nearest_same_situation",
            "planted_foot_rule": "lowest_foot_collider_point",
            "idle_hard_zero_envelope": 1,
            "idle_hard_zero_frames": int(idle_valid_mask.sum().item()),
            "cache_version": 9,
        },
    }


def load_or_build_excess_envelope(
    clips: list[tl.MotionClip],
    cfg: tl.TrainConfig,
    device: torch.device,
    synthetic_clip_indices: set[int] | None = None,
    real_clip_indices: list[int] | None = None,
    env_cfg: ExcessEnvelopeConfig | None = None,
) -> dict[str, torch.Tensor | dict[str, float | int | str]]:
    synthetic_clip_indices = synthetic_clip_indices or set()
    real_clip_indices = real_clip_indices or [i for i in range(len(clips)) if i not in synthetic_clip_indices]
    env_cfg = env_cfg or ExcessEnvelopeConfig()
    cache_dir = tl.resolve_path(env_cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(clips, real_clip_indices, synthetic_clip_indices, cfg, env_cfg)
    cache_path = cache_dir / f"excess_envelope_{key}.pt"
    if cache_path.exists():
        cached = torch.load(cache_path, map_location=device, weights_only=False)
        for name in (
            "slide_bound_mps",
            "yaw_excess_bound_radps",
            "features",
            "valid_mask",
            "real_valid_mask",
            "groundtruth_slide_mps",
            "groundtruth_yaw_excess_radps",
        ):
            if name not in cached:
                continue
            cached[name] = cached[name].to(device)
        cached["metadata"]["cache_path"] = str(cache_path)
        cached["metadata"]["cache_hit"] = 1
        return cached
    built = build_excess_envelope(
        clips,
        cfg,
        device,
        synthetic_clip_indices=synthetic_clip_indices,
        real_clip_indices=real_clip_indices,
        env_cfg=env_cfg,
    )
    save_obj = {
        name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for name, value in built.items()
    }
    torch.save(save_obj, cache_path)
    built["metadata"]["cache_path"] = str(cache_path)
    built["metadata"]["cache_hit"] = 0
    return built


def envelope_excess_loss(values: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    return F.relu(values - bounds).mean()
