from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from .bootstrap import PROJECT_ROOT, ensure_paths
    from . import ik_core as tl
    from . import train_simple_ae_controller as ctl
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    import ik_core as tl
    import train_simple_ae_controller as ctl

ensure_paths()

import contact_physics as cp


CACHE_VERSION = 3


@dataclass(frozen=True)
class ExcessEnvelopeConfig:
    margin: float = 1.05
    knn: int = 32
    cache_dir: str = "training/runs/cache/ik_excess_envelopes"
    chunk_size: int = 4096


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
def root_situation_feature(
    clip: tl.MotionClip,
    cur_idx: torch.Tensor,
    cfg: tl.TrainConfig,
    device: torch.device,
) -> torch.Tensor:
    cur_idx = cur_idx.to(device=device, dtype=torch.long)
    prev_idx = cur_idx - 1
    fut_idx = cur_idx + int(cfg.future_window)
    if not clip.cyclic_animation:
        fut_idx = torch.clamp(fut_idx, max=int(clip.T) - 1)
    prev_pos, _prev_rot, prev_yaw, _prev_heading = tl.root_state(clip, prev_idx, cfg, device)
    cur_pos, _cur_rot, _cur_yaw, _cur_heading = tl.root_state(clip, cur_idx, cfg, device)
    fut_pos, _fut_rot, fut_yaw, _fut_heading = tl.root_state(clip, fut_idx, cfg, device)
    yaw_delta = tl.wrap_angle(fut_yaw - prev_yaw) / torch.pi
    bend = signed_horizontal_angle(cur_pos - prev_pos, fut_pos - cur_pos) / torch.pi
    global_pos, _global_rot = tl.global_from_clip(clip, cur_idx, cfg, device)
    left = int(clip.body_names.index("foot_l"))
    right = int(clip.body_names.index("foot_r"))
    foot_distance = torch.linalg.norm(global_pos[:, left, [0, 2]] - global_pos[:, right, [0, 2]], dim=-1)
    return torch.stack((yaw_delta, bend, foot_distance), dim=-1)


def runtime_situation_feature(
    store: ctl.SimpleClipStore,
    cur_pos: torch.Tensor,
    clip_ids: torch.Tensor,
    cur_idx: torch.Tensor,
) -> torch.Tensor:
    prev_idx = cur_idx - 1
    fut_idx = cur_idx + int(store.cfg.future_window)
    prev_root_pos, _prev_rot, prev_yaw, _prev_heading = store.root_state(clip_ids, prev_idx)
    cur_root_pos, _cur_rot, _cur_yaw, _cur_heading = store.root_state(clip_ids, cur_idx)
    fut_root_pos, _fut_rot, fut_yaw, _fut_heading = store.root_state(clip_ids, fut_idx)
    yaw_delta = tl.wrap_angle(fut_yaw - prev_yaw) / torch.pi
    bend = signed_horizontal_angle(cur_root_pos - prev_root_pos, fut_root_pos - cur_root_pos) / torch.pi
    left = int(store.prototype.body_names.index("foot_l"))
    right = int(store.prototype.body_names.index("foot_r"))
    foot_distance = torch.linalg.norm(cur_pos[:, left, [0, 2]] - cur_pos[:, right, [0, 2]], dim=-1)
    return torch.stack((yaw_delta, bend, foot_distance), dim=-1)


@torch.no_grad()
def clip_reference_values(
    clip: tl.MotionClip,
    cfg: tl.TrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if clip.cyclic_animation:
        end = int(clip.cyclic_period) - 1
    else:
        end = int(clip.T) - ctl.transition_feature_horizon(cfg) - 1
    if end < 1:
        empty_i = torch.empty((0,), dtype=torch.long, device=device)
        empty_f = torch.empty((0, 3), dtype=torch.float32, device=device)
        empty_v = torch.empty((0,), dtype=torch.float32, device=device)
        return empty_i, empty_f, empty_v, empty_v
    idx = torch.arange(1, end + 1, dtype=torch.long, device=device)
    cur_pos, cur_rot = tl.global_from_clip(clip, idx, cfg, device)
    next_pos, next_rot = tl.global_from_clip(clip, idx + 1, cfg, device)
    foot_indices = tuple(int(clip.body_names.index(name)) for name in ("foot_l", "foot_r"))
    toe_indices = tuple(int(clip.body_names.index(name)) for name in ("ball_l", "ball_r"))
    slide_speeds = cp.foot_slide_speeds(cur_pos, cur_rot, next_pos, next_rot, foot_indices, toe_indices, clip.fps)
    yaw_speeds = cp.foot_vertical_yaw_speeds(cur_pos, cur_rot, next_pos, next_rot, foot_indices, toe_indices, clip.fps)
    slide_reference, planted_foot, _heights = cp.planted_foot_values(
        slide_speeds,
        cur_pos,
        cur_rot,
        foot_indices,
        toe_indices,
    )
    yaw_reference = yaw_speeds.gather(-1, planted_foot.unsqueeze(-1)).squeeze(-1)
    features = root_situation_feature(clip, idx, cfg, device)
    return idx, features, slide_reference, yaw_reference


def _cache_key(
    clips: list[tl.MotionClip],
    cfg: tl.TrainConfig,
    env_cfg: ExcessEnvelopeConfig,
) -> str:
    payload = {
        "version": CACHE_VERSION,
        "future_window": int(cfg.future_window),
        "position_unit_scale": float(cfg.position_unit_scale),
        "margin": float(env_cfg.margin),
        "knn": int(env_cfg.knn),
        "situation_feature": "yaw_bend_runtime_foot_distance_xz",
        "clips": [
            {
                "path": str(clip.path.resolve()),
                "mtime_ns": int(clip.path.stat().st_mtime_ns),
                "size": int(clip.path.stat().st_size),
                "cyclic": bool(clip.cyclic_animation),
                "period": int(clip.cyclic_period),
            }
            for clip in clips
        ],
    }
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


@torch.no_grad()
def _knn_upper_bound(
    target_features: torch.Tensor,
    source_features: torch.Tensor,
    source_values: torch.Tensor,
    k: int,
    chunk_size: int,
) -> torch.Tensor:
    if source_features.numel() == 0:
        raise ValueError("excess envelope needs at least one ground-truth transition")
    k = max(1, min(int(k), int(source_features.shape[0])))
    chunks: list[torch.Tensor] = []
    for start in range(0, int(target_features.shape[0]), int(chunk_size)):
        chunk = target_features[start : start + int(chunk_size)]
        dist = torch.cdist(chunk, source_features)
        kth = torch.topk(dist, k=k, largest=False, dim=-1).values[:, -1]
        near = dist <= (kth[:, None] + 1e-8)
        values = source_values.unsqueeze(0).expand_as(dist)
        chunks.append(torch.where(near, values, torch.full_like(values, -torch.inf)).amax(dim=-1))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def build_excess_envelope(
    store: ctl.SimpleClipStore,
    env_cfg: ExcessEnvelopeConfig | None = None,
) -> dict[str, torch.Tensor | dict[str, float | int | str]]:
    env_cfg = env_cfg or ExcessEnvelopeConfig()
    frame_count = int(store.lengths.sum().detach().cpu())
    device = store.device
    flat_features = torch.zeros((frame_count, 3), dtype=torch.float32, device=device)
    flat_linear = torch.zeros((frame_count,), dtype=torch.float32, device=device)
    flat_angular = torch.zeros((frame_count,), dtype=torch.float32, device=device)
    valid_mask = torch.zeros((frame_count,), dtype=torch.bool, device=device)
    source_features: list[torch.Tensor] = []
    source_linear: list[torch.Tensor] = []
    source_angular: list[torch.Tensor] = []
    offsets = store.frame_offsets.detach().cpu().tolist()
    for clip_id, clip in enumerate(store.clips):
        idx, features, linear, angular = clip_reference_values(clip, store.cfg, device)
        if idx.numel() == 0:
            continue
        flat = idx + int(offsets[clip_id])
        flat_features.index_copy_(0, flat, features)
        flat_linear.index_copy_(0, flat, linear)
        flat_angular.index_copy_(0, flat, angular)
        valid_mask.index_fill_(0, flat, True)
        source_features.append(features)
        source_linear.append(linear)
        source_angular.append(angular)
    if not source_features:
        raise ValueError("excess envelope could not find any valid clip transitions")

    real_features = torch.cat(source_features, dim=0)
    real_linear = torch.cat(source_linear, dim=0)
    real_angular = torch.cat(source_angular, dim=0)
    return {
        "groundtruth_linear_mps": flat_linear,
        "groundtruth_angular_radps": flat_angular,
        "features": flat_features,
        "source_features": real_features,
        "source_linear_mps": real_linear,
        "source_angular_radps": real_angular,
        "valid_mask": valid_mask,
        "metadata": {
            "source_transitions": int(real_features.shape[0]),
            "target_transitions": int(valid_mask.sum().item()),
            "margin": float(env_cfg.margin),
            "knn": int(env_cfg.knn),
            "max_real_linear_mps": float(real_linear.max().detach().cpu()),
            "max_real_angular_radps": float(real_angular.max().detach().cpu()),
            "situation_feature": "yaw_delta/pi,bend_angle/pi,runtime_horizontal_foot_distance_xz_m",
            "bound_lookup": "runtime_knn_situation_feature_no_frame_index",
            "cache_version": CACHE_VERSION,
        },
    }


def load_or_build_excess_envelope(
    store: ctl.SimpleClipStore,
    env_cfg: ExcessEnvelopeConfig | None = None,
) -> dict[str, torch.Tensor | dict[str, float | int | str]]:
    env_cfg = env_cfg or ExcessEnvelopeConfig()
    cache_dir = (PROJECT_ROOT / env_cfg.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(store.clips, store.cfg, env_cfg)
    cache_path = cache_dir / f"ik_excess_envelope_{key}.pt"
    if cache_path.exists():
        cached = torch.load(cache_path, map_location=store.device, weights_only=False)
        for name, value in list(cached.items()):
            if isinstance(value, torch.Tensor):
                cached[name] = value.to(store.device)
        cached["metadata"]["cache_path"] = str(cache_path)
        cached["metadata"]["cache_hit"] = 1
        return cached
    built = build_excess_envelope(store, env_cfg)
    torch.save(
        {name: value.detach().cpu() if isinstance(value, torch.Tensor) else value for name, value in built.items()},
        cache_path,
    )
    built["metadata"]["cache_path"] = str(cache_path)
    built["metadata"]["cache_hit"] = 0
    return built


def envelope_excess_rows(
    store: ctl.SimpleClipStore,
    envelope: dict[str, torch.Tensor | dict[str, float | int | str]],
    cur_pos: torch.Tensor,
    cur_rot: torch.Tensor,
    next_pos: torch.Tensor,
    next_rot: torch.Tensor,
    clip_ids: torch.Tensor,
    cur_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    foot_indices = tuple(int(store.prototype.body_names.index(name)) for name in ("foot_l", "foot_r"))
    toe_indices = tuple(int(store.prototype.body_names.index(name)) for name in ("ball_l", "ball_r"))
    linear_speeds = cp.foot_slide_speeds(cur_pos, cur_rot, next_pos, next_rot, foot_indices, toe_indices, store.prototype.fps)
    angular_speeds = cp.foot_vertical_yaw_speeds(cur_pos, cur_rot, next_pos, next_rot, foot_indices, toe_indices, store.prototype.fps)
    linear_selected, planted, _heights = cp.planted_foot_values(linear_speeds, cur_pos, cur_rot, foot_indices, toe_indices)
    angular_selected = angular_speeds.gather(-1, planted.unsqueeze(-1)).squeeze(-1)
    features = runtime_situation_feature(store, cur_pos, clip_ids, cur_idx).detach()
    metadata = envelope["metadata"]  # type: ignore[assignment]
    assert isinstance(metadata, dict)
    k = int(metadata.get("knn", 32))
    linear_bound = _knn_upper_bound(
        features,
        envelope["source_features"],  # type: ignore[arg-type]
        envelope["source_linear_mps"],  # type: ignore[arg-type]
        k,
        int(features.shape[0]),
    ) * float(metadata.get("margin", 1.05))
    angular_bound = _knn_upper_bound(
        features,
        envelope["source_features"],  # type: ignore[arg-type]
        envelope["source_angular_radps"],  # type: ignore[arg-type]
        k,
        int(features.shape[0]),
    ) * float(metadata.get("margin", 1.05))
    return F.relu(linear_selected - linear_bound), F.relu(angular_selected - angular_bound)


@torch.no_grad()
def groundtruth_sanity(
    store: ctl.SimpleClipStore,
    envelope: dict[str, torch.Tensor | dict[str, float | int | str]],
) -> dict[str, float]:
    valid = envelope["valid_mask"].bool()  # type: ignore[union-attr]
    linear_gt = envelope["groundtruth_linear_mps"][valid]  # type: ignore[index,union-attr]
    angular_gt = envelope["groundtruth_angular_radps"][valid]  # type: ignore[index,union-attr]
    features = envelope["features"][valid]  # type: ignore[index,union-attr]
    metadata = envelope["metadata"]  # type: ignore[assignment]
    assert isinstance(metadata, dict)
    k = int(metadata.get("knn", 32))
    margin = float(metadata.get("margin", 1.05))
    linear_bound = _knn_upper_bound(
        features,
        envelope["source_features"],  # type: ignore[arg-type]
        envelope["source_linear_mps"],  # type: ignore[arg-type]
        k,
        4096,
    ) * margin
    angular_bound = _knn_upper_bound(
        features,
        envelope["source_features"],  # type: ignore[arg-type]
        envelope["source_angular_radps"],  # type: ignore[arg-type]
        k,
        4096,
    ) * margin
    linear_excess = F.relu(linear_gt - linear_bound)
    angular_excess = F.relu(angular_gt - angular_bound)
    return {
        "gt_linear_excess_mean": float(linear_excess.mean().detach().cpu()),
        "gt_linear_excess_p95": float(torch.quantile(linear_excess, 0.95).detach().cpu()),
        "gt_linear_excess_max": float(linear_excess.max().detach().cpu()),
        "gt_angular_excess_mean": float(angular_excess.mean().detach().cpu()),
        "gt_angular_excess_p95": float(torch.quantile(angular_excess, 0.95).detach().cpu()),
        "gt_angular_excess_max": float(angular_excess.max().detach().cpu()),
    }
