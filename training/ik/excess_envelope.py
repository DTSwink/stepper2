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

try:
    from . import contact_physics as cp
except ImportError:
    import contact_physics as cp


CACHE_VERSION = 12
SOLE_SAMPLE_GRID_VALUES = (
    (-1.0, -1.0),
    (-1.0, 0.0),
    (-1.0, 1.0),
    (0.0, -1.0),
    (0.0, 0.0),
    (0.0, 1.0),
    (1.0, -1.0),
    (1.0, 0.0),
    (1.0, 1.0),
)
_SOLE_SAMPLE_GRID_CACHE: dict[tuple[str, torch.dtype], torch.Tensor] = {}
_LEG_START_INDEX_CACHE: dict[tuple[tuple[int, ...], str], torch.Tensor] = {}


@dataclass(frozen=True)
class ExcessEnvelopeConfig:
    margin: float = 1.05
    knn: int = 1
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


def runtime_situation_feature_from_feet(
    store: ctl.SimpleClipStore,
    cur_foot_toe_pos: torch.Tensor,
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
    foot_dx = cur_foot_toe_pos[:, 0, 0] - cur_foot_toe_pos[:, 2, 0]
    foot_dz = cur_foot_toe_pos[:, 0, 2] - cur_foot_toe_pos[:, 2, 2]
    foot_distance = torch.sqrt(foot_dx.square() + foot_dz.square() + 1e-12)
    return torch.stack((yaw_delta, bend, foot_distance), dim=-1)


def _leg_base_positions_root(store: ctl.SimpleClipStore, vec: torch.Tensor) -> torch.Tensor:
    clip = store.prototype
    tensors = clip.tensors(store.device)
    b = int(vec.shape[0])
    dtype = vec.dtype
    cursor = 0
    pelvis_pos = vec[:, cursor : cursor + 3]
    cursor += 3
    pelvis_rot = tl.rotation_6d_to_matrix(vec[:, cursor : cursor + 6])
    cursor += 6
    leg_starts = [int(spec["start"]) for spec in clip.ik_limb_specs if str(spec["kind"]) == "leg"]
    if all(int(clip.parents_body_list[start]) == int(clip.pelvis) for start in leg_starts):
        key = (tuple(leg_starts), str(store.device))
        leg_start_tensor = _LEG_START_INDEX_CACHE.get(key)
        if leg_start_tensor is None:
            leg_start_tensor = torch.tensor(leg_starts, dtype=torch.long, device=store.device)
            _LEG_START_INDEX_CACHE[key] = leg_start_tensor
        offsets = tensors["local_offsets"].to(dtype=dtype).index_select(
            0,
            leg_start_tensor,
        )
        return pelvis_pos[:, None, :] + torch.matmul(offsets.reshape(1, len(leg_starts), 3), pelvis_rot)

    core_dim = clip.Jcore * 6
    core_raw = vec[:, cursor : cursor + core_dim].reshape(b, clip.Jcore, 6)
    needed: set[int] = set()
    for start in leg_starts:
        j = int(start)
        while j >= 0:
            needed.add(j)
            j = int(clip.parents_body_list[j])
    local_offsets = tensors["local_offsets"].to(dtype=dtype)
    identity = torch.eye(3, dtype=dtype, device=store.device).expand(b, 3, 3)
    pos_root: dict[int, torch.Tensor] = {}
    rot_root: dict[int, torch.Tensor] = {}
    for j in sorted(needed):
        if j == int(clip.pelvis):
            local_offset = pelvis_pos
            local_rot = pelvis_rot
        else:
            local_offset = local_offsets[j].reshape(1, 3).expand(b, 3)
            if j in clip.core_nonpelvis_map:
                local_rot = tl.rotation_6d_to_matrix(core_raw[:, int(clip.core_nonpelvis_map[j])])
            else:
                local_rot = identity
        parent = int(clip.parents_body_list[j])
        if parent < 0:
            pos_root[j] = local_offset
            rot_root[j] = local_rot
        else:
            parent_pos = pos_root[parent]
            parent_rot = rot_root[parent]
            pos_root[j] = torch.matmul(local_offset.unsqueeze(1), parent_rot).squeeze(1) + parent_pos
            rot_root[j] = local_rot @ parent_rot
    return torch.stack([pos_root[int(start)] for start in leg_starts], dim=1)


def ik_foot_toe_state_from_vec(
    store: ctl.SimpleClipStore,
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    vec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    clip = store.prototype
    tensors = clip.tensors(store.device)
    payload = vec[:, ctl.payload_slice(store)]
    base_root = _leg_base_positions_root(store, vec)
    leg_specs = tuple(spec for spec in tl.IK_PAYLOAD_SLICES if str(spec["kind"]) == "leg")
    left_pos = leg_specs[0]["pos"]
    right_pos = leg_specs[1]["pos"]
    left_rot = leg_specs[0]["rot6"]
    right_rot = leg_specs[1]["rot6"]
    assert isinstance(left_pos, slice) and isinstance(right_pos, slice)
    assert isinstance(left_rot, slice) and isinstance(right_rot, slice)

    b = int(vec.shape[0])
    end_root = torch.stack((payload[:, left_pos], payload[:, right_pos]), dim=1)
    end_rot6 = torch.stack((payload[:, left_rot], payload[:, right_rot]), dim=1).reshape(b * 2, 6)
    end_rot_root = tl.rotation_6d_to_matrix(end_rot6).reshape(b, 2, 3, 3)
    lengths = tensors["ik_limb_lengths"][2:4].to(dtype=vec.dtype)
    l1 = lengths[:, 0].reshape(1, 2, 1)
    l2 = lengths[:, 1].reshape(1, 2, 1)
    delta = end_root - base_root
    axis = tl.normalize(delta)
    d = torch.linalg.norm(delta, dim=-1, keepdim=True)
    min_d = (l1 - l2).abs() + 1e-5
    max_d = l1 + l2 - 1e-5
    end_root = base_root + axis * d.clamp_min(1e-8).clamp(min=min_d, max=max_d)

    toe_offset = tensors["ik_toe_offsets"][2:4].to(dtype=vec.dtype).reshape(1, 2, 3)
    toe_pos_root = end_root + torch.matmul(toe_offset.unsqueeze(-2), end_rot_root).squeeze(-2)
    foot_pos_world = torch.matmul(end_root, root_rot) + root_pos[:, None, :]
    toe_pos_world = torch.matmul(toe_pos_root, root_rot) + root_pos[:, None, :]
    foot_rot_world = end_rot_root @ root_rot[:, None, :, :]
    positions = torch.stack((foot_pos_world[:, 0], toe_pos_world[:, 0], foot_pos_world[:, 1], toe_pos_world[:, 1]), dim=1)
    rotations = torch.stack((foot_rot_world[:, 0], foot_rot_world[:, 0], foot_rot_world[:, 1], foot_rot_world[:, 1]), dim=1)
    return positions, rotations


def _sole_sample_grid(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (str(device), dtype)
    cached = _SOLE_SAMPLE_GRID_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(SOLE_SAMPLE_GRID_VALUES, dtype=dtype, device=device)
        _SOLE_SAMPLE_GRID_CACHE[key] = cached
    return cached


def _compact_box_specs(
    positions: torch.Tensor,
    rotations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg = cp.DEFAULT_GEOMETRY
    foot = positions[:, (0, 2)]
    toe = positions[:, (1, 3)]
    toe_vec = toe - foot
    b, sides, _ = toe_vec.shape

    foot_forward, foot_side, foot_up = cp.basis_axes_from_direction(
        rotations[:, (0, 2)].reshape(b * sides, 3, 3),
        toe_vec.reshape(b * sides, 3),
        1,
    )
    toe_forward, toe_side, toe_up = cp.basis_axes_from_direction(
        rotations[:, (1, 3)].reshape(b * sides, 3, 3),
        toe_vec.reshape(b * sides, 3),
        0,
    )
    foot_forward = foot_forward.reshape(b, sides, 3)
    foot_side = foot_side.reshape(b, sides, 3)
    foot_up = foot_up.reshape(b, sides, 3)
    toe_forward = toe_forward.reshape(b, sides, 3)
    toe_side = toe_side.reshape(b, sides, 3)
    toe_up = toe_up.reshape(b, sides, 3)

    foot_center = toe - foot_forward * (cfg.foot_length * 0.5) + foot_up * cfg.sole_vertical_offset
    toe_center = toe + toe_forward * (cfg.toe_length * 0.5) + toe_up * cfg.sole_vertical_offset
    centers = torch.stack((foot_center, toe_center), dim=2)
    forward = torch.stack((foot_forward, toe_forward), dim=2)
    side = torch.stack((foot_side, toe_side), dim=2)
    up = torch.stack((foot_up, toe_up), dim=2)
    dims = torch.tensor(
        (
            (cfg.foot_length, cfg.foot_width, cfg.foot_height),
            (cfg.toe_length, cfg.toe_width, cfg.toe_height),
        ),
        dtype=positions.dtype,
        device=positions.device,
    )
    return centers, forward, side, up, dims


def _fixed_sole_points(
    specs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    center, forward, side, up, dims = specs
    grid = _sole_sample_grid(center.device, center.dtype).reshape(1, 1, 1, -1, 2)
    half = dims[:, :2].reshape(1, 1, 2, 1, 2) * 0.5
    offsets = grid * half
    sole_center = center - up * (dims[:, 2].reshape(1, 1, 2, 1) * 0.5)
    return (
        sole_center.unsqueeze(-2)
        + forward.unsqueeze(-2) * offsets[..., 0:1]
        + side.unsqueeze(-2) * offsets[..., 1:2]
    )


def compact_slide_yaw_selected_from_specs(
    cur_pos: torch.Tensor,
    cur_rot: torch.Tensor,
    next_pos: torch.Tensor,
    next_rot: torch.Tensor,
    cur_specs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    next_specs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    fps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    cur_points = _fixed_sole_points(cur_specs)
    next_points = _fixed_sole_points(next_specs)
    linear = torch.linalg.norm((next_points - cur_points)[..., [0, 2]], dim=-1).flatten(2).amin(dim=-1) * float(fps)
    planted = cur_points[..., 1].flatten(2).amin(dim=-1).argmin(dim=-1)

    left_delta = cur_rot[:, 0].transpose(-1, -2) @ next_rot[:, 0]
    right_delta = cur_rot[:, 2].transpose(-1, -2) @ next_rot[:, 2]
    left_yaw = torch.atan2(left_delta[:, 0, 2] - left_delta[:, 2, 0], left_delta[:, 0, 0] + left_delta[:, 2, 2]).abs()
    right_yaw = torch.atan2(right_delta[:, 0, 2] - right_delta[:, 2, 0], right_delta[:, 0, 0] + right_delta[:, 2, 2]).abs()
    angular = torch.stack((left_yaw, right_yaw), dim=-1) * float(fps)
    linear_selected = linear.gather(-1, planted.unsqueeze(-1)).squeeze(-1)
    angular_selected = angular.gather(-1, planted.unsqueeze(-1)).squeeze(-1)
    return linear_selected, angular_selected


def compact_slide_yaw_selected(
    cur_pos: torch.Tensor,
    cur_rot: torch.Tensor,
    next_pos: torch.Tensor,
    next_rot: torch.Tensor,
    fps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    linear, angular, _planted = compact_slide_yaw_selected_with_planted(cur_pos, cur_rot, next_pos, next_rot, fps)
    return linear, angular


def compact_slide_yaw_selected_with_planted(
    cur_pos: torch.Tensor,
    cur_rot: torch.Tensor,
    next_pos: torch.Tensor,
    next_rot: torch.Tensor,
    fps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cur_points = cur_pos.reshape(cur_pos.shape[0], 2, 2, 3)
    next_points = next_pos.reshape(next_pos.shape[0], 2, 2, 3)
    dx = next_points[..., 0] - cur_points[..., 0]
    dz = next_points[..., 2] - cur_points[..., 2]
    linear = torch.sqrt(dx.square() + dz.square() + 1e-12).amin(dim=-1) * float(fps)
    planted = cur_points[..., 1].amin(dim=-1).argmin(dim=-1)

    delta = cur_rot.transpose(-1, -2) @ next_rot
    yaw_delta = torch.atan2(delta[:, :, 0, 2] - delta[:, :, 2, 0], delta[:, :, 0, 0] + delta[:, :, 2, 2])
    angular_parts = yaw_delta.abs() * float(fps)
    angular_left = torch.maximum(angular_parts[:, 0], angular_parts[:, 1])
    angular_right = torch.maximum(angular_parts[:, 2], angular_parts[:, 3])
    angular = torch.stack((angular_left, angular_right), dim=-1)
    return (
        linear.gather(-1, planted.unsqueeze(-1)).squeeze(-1),
        angular.gather(-1, planted.unsqueeze(-1)).squeeze(-1),
        planted,
    )


@torch.no_grad()
def clip_reference_values(
    clip: tl.MotionClip,
    cfg: tl.TrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if clip.cyclic_animation:
        end = int(clip.cyclic_period) - 1
    else:
        end = int(clip.T) - ctl.transition_feature_horizon(cfg) - 1
    if end < 1:
        empty_i = torch.empty((0,), dtype=torch.long, device=device)
        empty_f = torch.empty((0, 3), dtype=torch.float32, device=device)
        empty_v = torch.empty((0,), dtype=torch.float32, device=device)
        return empty_i, empty_f, empty_v, empty_v, empty_i
    idx = torch.arange(1, end + 1, dtype=torch.long, device=device)
    cur_pos, cur_rot = tl.global_from_clip(clip, idx, cfg, device)
    next_pos, next_rot = tl.global_from_clip(clip, idx + 1, cfg, device)
    foot_indices = tuple(int(clip.body_names.index(name)) for name in ("foot_l", "foot_r"))
    toe_indices = tuple(int(clip.body_names.index(name)) for name in ("ball_l", "ball_r"))
    order = (foot_indices[0], toe_indices[0], foot_indices[1], toe_indices[1])
    slide_reference, yaw_reference, planted = compact_slide_yaw_selected_with_planted(
        cur_pos[:, order],
        cur_rot[:, order],
        next_pos[:, order],
        next_rot[:, order],
        clip.fps,
    )
    features = root_situation_feature(clip, idx, cfg, device)
    return idx, features, slide_reference, yaw_reference, planted


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
        "situation_feature": "clip_id_plus_yaw_bend_runtime_foot_distance_xz_planted_side",
        "bound_lookup": "animation_dependent_nearest_same_planted_side_foot_ball_points_foot_yaw",
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
def _animation_upper_bounds(
    envelope: dict[str, torch.Tensor | dict[str, float | int | str]],
    features: torch.Tensor,
    clip_ids: torch.Tensor,
    planted_side: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    metadata = envelope["metadata"]  # type: ignore[assignment]
    assert isinstance(metadata, dict)
    clip_features = envelope["clip_source_features"].index_select(0, clip_ids)  # type: ignore[union-attr]
    clip_linear = envelope["clip_source_linear_mps"].index_select(0, clip_ids)  # type: ignore[union-attr]
    clip_angular = envelope["clip_source_angular_radps"].index_select(0, clip_ids)  # type: ignore[union-attr]
    clip_counts = envelope["clip_source_counts"].index_select(0, clip_ids)  # type: ignore[union-attr]
    n_sources = int(clip_features.shape[1])
    if "clip_source_row_index" in envelope:
        source_rows = envelope["clip_source_row_index"]  # type: ignore[assignment]
    else:
        source_rows = torch.arange(n_sources, dtype=torch.long, device=features.device).reshape(1, n_sources)
    valid = source_rows < clip_counts.reshape(-1, 1)
    if planted_side is not None and "clip_source_planted_side" in envelope:
        clip_planted = envelope["clip_source_planted_side"].index_select(0, clip_ids)  # type: ignore[union-attr]
        side_valid = valid & (clip_planted == planted_side.reshape(-1, 1))
        valid = torch.where(side_valid.any(dim=-1, keepdim=True), side_valid, valid)
    dist = (clip_features - features[:, None, :]).square().sum(dim=-1)
    dist = dist.masked_fill(~valid, torch.inf)
    k = min(int(metadata.get("knn", 32)), n_sources)
    nearest = torch.topk(dist, k=k, largest=False, dim=-1).indices
    clip_linear = clip_linear.masked_fill(~valid, -torch.inf)
    clip_angular = clip_angular.masked_fill(~valid, -torch.inf)
    return clip_linear.gather(1, nearest).amax(dim=-1), clip_angular.gather(1, nearest).amax(dim=-1)


@torch.no_grad()
def build_excess_envelope(
    store: ctl.SimpleClipStore,
    env_cfg: ExcessEnvelopeConfig | None = None,
) -> dict[str, torch.Tensor | dict[str, float | int | str]]:
    env_cfg = env_cfg or ExcessEnvelopeConfig()
    frame_count = int(store.lengths.sum().detach().cpu())
    device = store.device
    flat_features = torch.zeros((frame_count, 3), dtype=torch.float32, device=device)
    flat_root_yaw_bend = torch.zeros((frame_count, 2), dtype=torch.float32, device=device)
    flat_linear = torch.zeros((frame_count,), dtype=torch.float32, device=device)
    flat_angular = torch.zeros((frame_count,), dtype=torch.float32, device=device)
    flat_planted = torch.full((frame_count,), -1, dtype=torch.long, device=device)
    flat_clip_ids = torch.full((frame_count,), -1, dtype=torch.long, device=device)
    valid_mask = torch.zeros((frame_count,), dtype=torch.bool, device=device)
    clip_features: list[torch.Tensor] = []
    clip_linear_values: list[torch.Tensor] = []
    clip_angular_values: list[torch.Tensor] = []
    clip_planted_values: list[torch.Tensor] = []
    offsets = store.frame_offsets.detach().cpu().tolist()
    for clip_id, clip in enumerate(store.clips):
        if clip.cyclic_animation:
            root_rows = torch.arange(0, int(clip.cyclic_period), dtype=torch.long, device=device)
            root_idx = root_rows.clone()
            if root_idx.numel() > 0:
                root_idx[0] = int(clip.cyclic_period)
        else:
            max_cur = int(clip.T) - ctl.transition_feature_horizon(store.cfg) - 1
            root_rows = torch.arange(1, max(1, max_cur + 1), dtype=torch.long, device=device) if max_cur >= 1 else torch.empty((0,), dtype=torch.long, device=device)
            root_idx = root_rows
        if root_rows.numel() > 0:
            root_features = root_situation_feature(clip, root_idx, store.cfg, device)[:, :2]
            flat_root_yaw_bend.index_copy_(0, root_rows + int(offsets[clip_id]), root_features)

        idx, features, linear, angular, planted = clip_reference_values(clip, store.cfg, device)
        if idx.numel() == 0:
            clip_features.append(torch.empty((0, 3), dtype=torch.float32, device=device))
            clip_linear_values.append(torch.empty((0,), dtype=torch.float32, device=device))
            clip_angular_values.append(torch.empty((0,), dtype=torch.float32, device=device))
            clip_planted_values.append(torch.empty((0,), dtype=torch.long, device=device))
            continue
        flat = idx + int(offsets[clip_id])
        flat_features.index_copy_(0, flat, features)
        flat_linear.index_copy_(0, flat, linear)
        flat_angular.index_copy_(0, flat, angular)
        flat_planted.index_copy_(0, flat, planted)
        flat_clip_ids.index_fill_(0, flat, int(clip_id))
        valid_mask.index_fill_(0, flat, True)
        clip_features.append(features)
        clip_linear_values.append(linear)
        clip_angular_values.append(angular)
        clip_planted_values.append(planted)
    valid_clip_features = [features for features in clip_features if features.numel() > 0]
    if not valid_clip_features:
        raise ValueError("excess envelope could not find any valid clip transitions")

    real_linear = torch.cat([linear for linear in clip_linear_values if linear.numel() > 0], dim=0)
    real_angular = torch.cat([angular for angular in clip_angular_values if angular.numel() > 0], dim=0)
    source_transition_count = sum(int(features.shape[0]) for features in clip_features)
    max_clip_rows = max(int(features.shape[0]) for features in clip_features)
    clip_count = len(store.clips)
    per_clip_features = torch.zeros((clip_count, max_clip_rows, 3), dtype=torch.float32, device=device)
    per_clip_linear = torch.zeros((clip_count, max_clip_rows), dtype=torch.float32, device=device)
    per_clip_angular = torch.zeros((clip_count, max_clip_rows), dtype=torch.float32, device=device)
    per_clip_planted = torch.full((clip_count, max_clip_rows), -1, dtype=torch.long, device=device)
    per_clip_counts = torch.zeros((clip_count,), dtype=torch.long, device=device)
    for clip_id, (features, linear, angular, planted) in enumerate(
        zip(clip_features, clip_linear_values, clip_angular_values, clip_planted_values)
    ):
        count = int(features.shape[0])
        per_clip_counts[clip_id] = count
        if count:
            per_clip_features[clip_id, :count] = features
            per_clip_linear[clip_id, :count] = linear
            per_clip_angular[clip_id, :count] = angular
            per_clip_planted[clip_id, :count] = planted
    return {
        "groundtruth_linear_mps": flat_linear,
        "groundtruth_angular_radps": flat_angular,
        "groundtruth_planted_side": flat_planted,
        "features": flat_features,
        "root_yaw_bend": flat_root_yaw_bend,
        "frame_clip_ids": flat_clip_ids,
        "clip_source_features": per_clip_features,
        "clip_source_linear_mps": per_clip_linear,
        "clip_source_angular_radps": per_clip_angular,
        "clip_source_planted_side": per_clip_planted,
        "clip_source_counts": per_clip_counts,
        "clip_source_row_index": torch.arange(max_clip_rows, dtype=torch.long, device=device).reshape(1, max_clip_rows),
        "valid_mask": valid_mask,
        "metadata": {
            "source_transitions": int(source_transition_count),
            "target_transitions": int(valid_mask.sum().item()),
            "margin": float(env_cfg.margin),
            "knn": int(env_cfg.knn),
            "max_real_linear_mps": float(real_linear.max().detach().cpu()),
            "max_real_angular_radps": float(real_angular.max().detach().cpu()),
            "situation_feature": "clip_id,yaw_delta/pi,bend_angle/pi,runtime_horizontal_foot_distance_xz_m,selected_lower_side",
            "bound_lookup": "animation_dependent_nearest_same_planted_side_foot_ball_points_foot_yaw_no_frame_index",
            "cache_version": CACHE_VERSION,
            "max_clip_source_rows": int(max_clip_rows),
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
    order = (foot_indices[0], toe_indices[0], foot_indices[1], toe_indices[1])
    linear_selected, angular_selected, planted = compact_slide_yaw_selected_with_planted(
        cur_pos[:, order],
        cur_rot[:, order],
        next_pos[:, order],
        next_rot[:, order],
        store.prototype.fps,
    )
    features = runtime_situation_feature(store, cur_pos, clip_ids, cur_idx).detach()
    metadata = envelope["metadata"]  # type: ignore[assignment]
    assert isinstance(metadata, dict)
    linear_bound, angular_bound = _animation_upper_bounds(envelope, features, clip_ids, planted)
    margin = float(metadata.get("margin", 1.05))
    linear_bound = linear_bound * margin
    angular_bound = angular_bound * margin
    return F.relu(linear_selected - linear_bound), F.relu(angular_selected - angular_bound)


def envelope_excess_ik_state_rows(
    store: ctl.SimpleClipStore,
    envelope: dict[str, torch.Tensor | dict[str, float | int | str]],
    cur_pos: torch.Tensor,
    cur_rot: torch.Tensor,
    next_pos: torch.Tensor,
    next_rot: torch.Tensor,
    clip_ids: torch.Tensor,
    cur_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    linear_selected, angular_selected, linear_bound, angular_bound = envelope_values_ik_state_rows(
        store, envelope, cur_pos, cur_rot, next_pos, next_rot, clip_ids, cur_idx
    )
    return F.relu(linear_selected - linear_bound), F.relu(angular_selected - angular_bound)


def envelope_values_ik_state_rows(
    store: ctl.SimpleClipStore,
    envelope: dict[str, torch.Tensor | dict[str, float | int | str]],
    cur_pos: torch.Tensor,
    cur_rot: torch.Tensor,
    next_pos: torch.Tensor,
    next_rot: torch.Tensor,
    clip_ids: torch.Tensor,
    cur_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    linear_selected, angular_selected, planted = compact_slide_yaw_selected_with_planted(
        cur_pos,
        cur_rot,
        next_pos,
        next_rot,
        store.prototype.fps,
    )
    if "root_yaw_bend" in envelope:
        frame = store.frame_index(clip_ids, cur_idx)
        root_yaw_bend = envelope["root_yaw_bend"].index_select(0, frame)  # type: ignore[union-attr]
        foot_dx = cur_pos[:, 0, 0] - cur_pos[:, 2, 0]
        foot_dz = cur_pos[:, 0, 2] - cur_pos[:, 2, 2]
        foot_distance = torch.sqrt(foot_dx.square() + foot_dz.square() + 1e-12).unsqueeze(-1)
        features = torch.cat((root_yaw_bend, foot_distance), dim=-1).detach()
    else:
        features = runtime_situation_feature_from_feet(store, cur_pos, clip_ids, cur_idx).detach()
    metadata = envelope["metadata"]  # type: ignore[assignment]
    assert isinstance(metadata, dict)
    linear_bound, angular_bound = _animation_upper_bounds(envelope, features, clip_ids, planted)
    margin = float(metadata.get("margin", 1.05))
    linear_bound = linear_bound * margin
    angular_bound = angular_bound * margin
    return linear_selected, angular_selected, linear_bound, angular_bound


def envelope_excess_ik_rows(
    store: ctl.SimpleClipStore,
    envelope: dict[str, torch.Tensor | dict[str, float | int | str]],
    cur_root_pos: torch.Tensor,
    cur_root_rot: torch.Tensor,
    cur_vec: torch.Tensor,
    next_root_pos: torch.Tensor,
    next_root_rot: torch.Tensor,
    next_vec: torch.Tensor,
    clip_ids: torch.Tensor,
    cur_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cur_pos, cur_rot = ik_foot_toe_state_from_vec(store, cur_root_pos, cur_root_rot, cur_vec)
    next_pos, next_rot = ik_foot_toe_state_from_vec(store, next_root_pos, next_root_rot, next_vec)
    return envelope_excess_ik_state_rows(store, envelope, cur_pos, cur_rot, next_pos, next_rot, clip_ids, cur_idx)


@torch.no_grad()
def groundtruth_sanity(
    store: ctl.SimpleClipStore,
    envelope: dict[str, torch.Tensor | dict[str, float | int | str]],
) -> dict[str, float]:
    valid = envelope["valid_mask"].bool()  # type: ignore[union-attr]
    linear_gt = envelope["groundtruth_linear_mps"][valid]  # type: ignore[index,union-attr]
    angular_gt = envelope["groundtruth_angular_radps"][valid]  # type: ignore[index,union-attr]
    features = envelope["features"][valid]  # type: ignore[index,union-attr]
    clip_ids = envelope["frame_clip_ids"][valid]  # type: ignore[index,union-attr]
    planted = envelope["groundtruth_planted_side"][valid]  # type: ignore[index,union-attr]
    metadata = envelope["metadata"]  # type: ignore[assignment]
    assert isinstance(metadata, dict)
    margin = float(metadata.get("margin", 1.05))
    linear_bound, angular_bound = _animation_upper_bounds(envelope, features, clip_ids, planted)
    linear_bound = linear_bound * margin
    angular_bound = angular_bound * margin
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
