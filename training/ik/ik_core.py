from __future__ import annotations

# Put your NPZ folder path here. Relative paths are resolved from the stepper
# project root, not from the current shell directory.
folder_path = "data/fbx/npz_final"

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

try:
    from .bootstrap import PROJECT_ROOT, ensure_paths
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths

ensure_paths()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import body_mass
import contact_physics as cp


@dataclass
class TrainConfig:
    fps: int = 30
    # Cascadeur/Unreal FBX data is usually centimeters. Training in meters keeps
    # MAX_SPEED_SCALE=5.0 interpretable as 5 m/s. Set to 1.0 for raw FBX units.
    position_unit_scale: float = 0.01
    max_speed_scale: float = 5.0
    max_turn_rate_per_sec_scale: float = math.radians(720.0)
    pose_delta_scale: float = 2.0
    future_window_seconds: float = 0.25
    cyclic_animation: bool = False

    hidden_dim: int = 512
    num_hidden_layers: int = 2
    activation: str = "GELU"
    learning_rate: float = 1e-4
    lr_schedule: str = "adaptive_plateau"
    lr_min_factor: float = 0.05
    lr_stage_decay: float = 1.0
    lr_warmup_epochs: int = 0
    lr_plateau_factor: float = 0.7
    lr_plateau_patience_epochs: int = 12
    lr_plateau_threshold: float = 1e-3
    lr_plateau_cooldown_epochs: int = 0
    lr_reset_on_rollout_advance: bool = True
    weight_decay: float = 0.0
    batch_size: int = 64
    max_epochs: int = 2000
    val_fraction: float = 0.1
    seed: int = 1234
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    allow_tf32: bool = True
    use_torch_compile: bool = False
    torch_compile_mode: str = "default"
    show_progress: bool = False
    save_last_every_epochs: int = 5
    save_best_every_epochs: int = 0
    writer_flush_every_epochs: int = 5
    predict_residual: bool = True
    zero_init_output: bool = True
    target_loss_reduction: float = 0.98
    stop_at_target_loss_reduction: bool = False
    max_train_seconds: float = 0.0
    profile_timing: bool = False
    profile_sync_cuda: bool = False
    disable_validation: bool = True

    rollout_schedule: tuple[int, ...] = (1, 2, 4, 8)
    curriculum_threshold: float = 1e-3
    curriculum_min_epochs: int = 0
    curriculum_max_epochs_per_stage: int = 0
    curriculum_patience_epochs: int = 5
    curriculum_stall_patience_epochs: int = 0
    curriculum_min_delta: float = 1e-5
    stop_on_final_stall: bool = False

    alpha0_pelvis_location: float = 1.0
    alpha1_pelvis_rotation: float = 1.0
    alpha2_pose_rotation: float = 1.0
    alpha3_pose_6d_aux: float = 0.1
    alpha4_end_effector_location: float = 10.0
    alpha5_end_effector_rotation: float = 0.5
    alpha6_full_body_location: float = 1.0
    alpha7_contact_label: float = 10.0
    alpha8_foot_penetration: float = 8000.0
    alpha9_foot_sliding: float = 2.0
    alpha10_freefall: float = 0.0
    alpha11_contact_height: float = 3000.0
    alpha12_termination: float = 0.0
    enable_contact_physics_losses: bool = False
    training_loop: str = "sampled"
    agent_sampling: str = "random"
    agent_min_cohort_steps: int = 8
    gradient_accumulation_batches: int = 1
    periodic_sampling_weight: float = 1.0
    nonperiodic_sampling_weight: float = 1.0
    synthetic_agent_fraction: float = 0.0
    init_pose_sampling: str = "same_clip"
    agent_fixed_start_frame: int = 0
    live_viewer: bool = True
    live_viewer_max_agents: int = 4
    live_viewer_start_visualizing: bool = False
    live_viewer_close_on_exit: bool = False
    visual_reporter: bool = False
    visual_report_interval_seconds: float = 60.0
    visual_report_device: str = "cpu"
    visual_report_max_frames: int = 180
    update_comparison_on_exit: bool = True
    comparison_output_path: str = "training/runs/model_comparisons/model_comparison.html"
    comparison_device: str = "cpu"
    comparison_max_frames: int = 0
    enable_early_termination: bool = False
    restart_on_termination: bool = True
    reset_exhausted_agents: bool = True
    freefall_body_height_offset_m: float = 0.0
    freefall_initial_offset_history: int = 1
    freefall_initial_contacts_off: bool = True
    enable_freefall_termination: bool = False
    ae_loss_weight: float = 1.0
    ae_row_top_fraction: float = 0.0
    ae_row_top_weight: float = 0.0
    slide_excess_loss_weight: float = 0.0
    slide_excess_threshold_mps: float = 0.0
    slide_excess_gt_margin: float = 1.05
    turn_slide_bound_divisor: float = 1.0
    excess_envelope_enabled: bool = True
    excess_envelope_knn: int = 32
    excess_envelope_margin: float = 1.05
    excess_envelope_cache_dir: str = "training/runs/cache/excess_envelopes"
    yaw_excess_loss_weight: float = 0.0
    yaw_excess_scale_radps: float = 1.0
    yaw_excess_scale_checkpoint: str = ""
    motion_floor_loss_weight: float = 0.0
    motion_floor_margin: float = 0.9
    include_transition_foot_motion: bool = False
    foot_slide_scale_mps: float = 1.0
    transition_yaw_scale_radps: float = 10.0
    root_lookahead_steps: int = 0
    pose_representation: str = "rot6"
    ik_marker_bones: tuple[str, ...] = ("hand_l", "hand_r", "foot_l", "ball_l", "foot_r", "ball_r")

    end_effector_bones: tuple[str, ...] = ("foot_l", "ball_l", "foot_r", "ball_r")
    exclude_bone_prefixes: tuple[str, ...] = ("ik_", "weapon_")
    exclude_bone_names: tuple[str, ...] = ("root", "attach")

    checkpoint_every_epochs: int = 500
    timed_checkpoint_interval_minutes: float = 30.0
    run_name: str = "locomotion_mlp"
    output_dir: str = "training/runs"

    @property
    def max_speed_scale_final(self) -> float:
        return self.max_speed_scale / self.fps

    @property
    def max_turn_rate_scale_final(self) -> float:
        return self.max_turn_rate_per_sec_scale / self.fps

    @property
    def pose_delta_scale_final(self) -> float:
        return self.pose_delta_scale / self.fps

    @property
    def future_window(self) -> int:
        return max(1, int(round(self.future_window_seconds * self.fps)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def date_prefixed_run_name(run_name: str, now: float | None = None) -> str:
    text = str(run_name).strip() or "run"
    if "_ik_" in text:
        text = text.split("_ik_", 1)[1]
    if len(text) >= 15 and text[:8].isdigit() and text[8] == "_" and text[9:15].isdigit():
        text = text[16:] if len(text) > 15 and text[15] == "_" else text[15:]
        if text.startswith("ik_"):
            text = text[3:]
    while len(text) >= 8 and text[:8].isdigit() and (len(text) == 8 or text[8] in "_/\\"):
        text = text[9:] if len(text) > 8 else "run"
        text = text.strip("_/\\") or "run"
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now or time.time()))
    text = text.strip("_/\\") or "run"
    return f"{stamp}_ik_{text}"


def ik_checkpoint_path(path: Path, run_name: str | None = None) -> Path:
    path = Path(path)
    if path.suffix != ".pt" or not path.stem.startswith("checkpoint_"):
        return path
    run_id = Path(str(run_name or path.parent.parent.name)).name
    if "_ik_" not in run_id:
        run_id = date_prefixed_run_name(run_id)
    tag = path.stem.removeprefix("checkpoint_")
    return path.with_name(f"{run_id}_{tag}.pt")


def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)


def rotmat_to_6d(rot: torch.Tensor) -> torch.Tensor:
    # Row-vector convention: store the first two basis rows.
    return rot[..., :2, :].reshape(*rot.shape[:-2], 6)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    # Row-vector Gram-Schmidt. This matches rotmat_to_6d above.
    # If the network emits degenerate 6D rows, choose a stable orthogonal
    # fallback instead of returning a collapsed "rotation" matrix.
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    a1_norm = torch.linalg.norm(a1, dim=-1, keepdim=True)
    fallback_b1 = torch.zeros_like(a1)
    fallback_b1[..., 0] = 1.0
    b1 = torch.where(a1_norm > 1e-8, normalize(a1), fallback_b1)

    projected = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    projected_norm = torch.linalg.norm(projected, dim=-1, keepdim=True)

    fallback_axis = F.one_hot(b1.abs().argmin(dim=-1), num_classes=3).to(dtype=d6.dtype, device=d6.device)
    fallback_projected = fallback_axis - (b1 * fallback_axis).sum(dim=-1, keepdim=True) * b1
    b2 = torch.where(projected_norm > 1e-8, normalize(projected), normalize(fallback_projected))
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def clean_6d(d6: torch.Tensor) -> torch.Tensor:
    return rotmat_to_6d(rotation_6d_to_matrix(d6))


def geodesic_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Row-vector relative rotation: R_delta = pred * target^-1 = pred * target^T.
    delta = pred @ target.transpose(-1, -2)
    trace = delta.diagonal(dim1=-1, dim2=-2).sum(dim=-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos).mean()


def geodesic_angles(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Row-vector relative rotation: R_delta = pred * target^-1 = pred * target^T.
    delta = pred @ target.transpose(-1, -2)
    trace = delta.diagonal(dim1=-1, dim2=-2).sum(dim=-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos)


def weighted_batch_mean(values: torch.Tensor, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    per_sample = values.reshape(values.shape[0], -1).mean(dim=-1)
    if sample_weight is None:
        return per_sample.mean()
    weights = sample_weight.to(device=values.device, dtype=values.dtype)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)


def yaw_to_row_matrix(yaw: torch.Tensor) -> torch.Tensor:
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    row0 = torch.stack((c, z, s), dim=-1)
    row1 = torch.stack((z, o, z), dim=-1)
    row2 = torch.stack((-s, z, c), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def heading_yaw_from_root(root_rot: torch.Tensor) -> torch.Tensor:
    # UE mannequin root convention in this project: local +Z is vertical, and
    # local -Y is the character/root heading. Project it onto the XZ ground plane.
    forward = -root_rot[..., 1, :]
    return torch.atan2(forward[..., 0], forward[..., 2])


def should_keep_bone(name: str, cfg: TrainConfig) -> bool:
    if name in cfg.exclude_bone_names:
        return False
    return not any(name.startswith(prefix) for prefix in cfg.exclude_bone_prefixes)


def axis_up_axis(arrays: np.lib.npyio.NpzFile) -> int:
    if "axis_up_axis" not in arrays.files:
        return 2
    return int(arrays["axis_up_axis"])


def canonicalize_positions(pos: torch.Tensor, up_axis: int) -> torch.Tensor:
    if up_axis == 3:
        # UE/FBX Z-up convention in this project: source +Z is vertical and
        # source -Y is forward. Convert to the training convention: +Y up,
        # +Z forward, preserving a right-handed row-vector basis.
        return torch.stack((pos[..., 0], pos[..., 2], -pos[..., 1]), dim=-1)
    return pos


def canonicalize_rotations(rot: torch.Tensor, up_axis: int) -> torch.Tensor:
    if up_axis != 3:
        return rot
    # Row-vector convention. With p_c = p_s P, rotations transform as
    # R_c = P^-1 R_s P.
    p = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=rot.dtype,
        device=rot.device,
    )
    return p.transpose(0, 1) @ rot @ p


IK_POSE_REPRESENTATION = "ik_markers"
IK_LIMB_SPECS = (
    {"side": "l", "kind": "arm", "start": "upperarm_l", "mid": "lowerarm_l", "end": "hand_l", "toe": None},
    {"side": "r", "kind": "arm", "start": "upperarm_r", "mid": "lowerarm_r", "end": "hand_r", "toe": None},
    {"side": "l", "kind": "leg", "start": "thigh_l", "mid": "calf_l", "end": "foot_l", "toe": "ball_l"},
    {"side": "r", "kind": "leg", "start": "thigh_r", "mid": "calf_r", "end": "foot_r", "toe": "ball_r"},
)
IK_PAYLOAD_DIM = 42
IK_POLE_ALPHA = math.pi / 2.0
IK_TOE_ALPHA = math.pi / 2.0
IK_CHARACTER_FORWARD = (0.0, -1.0, 0.0)
IK_FOOT_LOCAL_SIDE_AXIS = (0.0, 1.0, 0.0)


def uses_ik_markers(value: object) -> bool:
    return str(value).lower().strip() == IK_POSE_REPRESENTATION


def ik_chain_payload_slices() -> tuple[dict[str, object], ...]:
    cursor = 0
    slices: list[dict[str, object]] = []
    for spec in IK_LIMB_SPECS:
        pos = slice(cursor, cursor + 3)
        cursor += 3
        rot6 = slice(cursor, cursor + 6)
        cursor += 6
        pole = slice(cursor, cursor + 1)
        cursor += 1
        toe = None
        if spec["kind"] == "leg":
            toe = slice(cursor, cursor + 1)
            cursor += 1
        slices.append({**spec, "pos": pos, "rot6": rot6, "pole": pole, "toe_float": toe})
    if cursor != IK_PAYLOAD_DIM:
        raise RuntimeError(f"IK payload layout is {cursor} dims, expected {IK_PAYLOAD_DIM}")
    return tuple(slices)


IK_PAYLOAD_SLICES = ik_chain_payload_slices()


def root_relative_positions(
    global_pos: torch.Tensor,
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(global_pos - root_pos[:, None, :], root_rot.transpose(-1, -2))


def root_relative_to_world(
    local_pos: torch.Tensor,
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(local_pos, root_rot) + root_pos[:, None, :]


def matrix_from_direction(
    direction: torch.Tensor,
    up_hint: torch.Tensor,
    forward_axis: int = 0,
) -> torch.Tensor:
    forward = normalize(direction)
    fallback_forward = torch.zeros_like(forward)
    fallback_forward[..., 2] = 1.0
    forward = torch.where(
        torch.linalg.norm(direction, dim=-1, keepdim=True) > 1e-8,
        forward,
        fallback_forward,
    )
    up = up_hint - (up_hint * forward).sum(dim=-1, keepdim=True) * forward
    fallback_up = torch.zeros_like(up)
    fallback_up[..., 1] = 1.0
    up = torch.where(torch.linalg.norm(up, dim=-1, keepdim=True) > 1e-8, up, fallback_up)
    up = normalize(up - (up * forward).sum(dim=-1, keepdim=True) * forward)
    side = normalize(torch.cross(up, forward, dim=-1))
    up = normalize(torch.cross(forward, side, dim=-1))
    if forward_axis == 1:
        return torch.stack((up, forward, side), dim=-2)
    return torch.stack((forward, side, up), dim=-2)


def solve_two_bone_positions(
    base: torch.Tensor,
    preferred_mid: torch.Tensor,
    target: torch.Tensor,
    upper_len: torch.Tensor,
    lower_len: torch.Tensor,
    pole_hint: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    eps = 1e-6
    total_len = (upper_len + lower_len).clamp_min(eps)
    delta = target - base
    dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
    direction = normalize(delta)
    fallback_dir = normalize(preferred_mid - base)
    direction = torch.where(dist > eps, direction, fallback_dir)
    clamped_dist = dist.clamp(min=eps)
    max_reach = (total_len * 0.999).unsqueeze(-1)
    target_clamped = base + direction * torch.minimum(dist, max_reach)
    clamped_dist = torch.minimum(clamped_dist, max_reach).clamp_min(eps)
    a = (upper_len.square() - lower_len.square() + clamped_dist.squeeze(-1).square()) / (
        2.0 * clamped_dist.squeeze(-1)
    )
    h_sq = (upper_len.square() - a.square()).clamp_min(0.0)
    pole = pole_hint - (pole_hint * direction).sum(dim=-1, keepdim=True) * direction
    preferred_pole = preferred_mid - base - ((preferred_mid - base) * direction).sum(dim=-1, keepdim=True) * direction
    pole = torch.where(torch.linalg.norm(pole, dim=-1, keepdim=True) > eps, pole, preferred_pole)
    fallback_axis = F.one_hot(direction.abs().argmin(dim=-1), num_classes=3).to(dtype=direction.dtype, device=direction.device)
    fallback_pole = fallback_axis - (fallback_axis * direction).sum(dim=-1, keepdim=True) * direction
    pole = torch.where(torch.linalg.norm(pole, dim=-1, keepdim=True) > eps, pole, fallback_pole)
    pole = normalize(pole)
    mid = base + direction * a.unsqueeze(-1) + pole * torch.sqrt(h_sq).unsqueeze(-1)
    return mid, target_clamped


def project_to_plane(v: torch.Tensor, normal: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    n = normalize(normal, eps)
    return normalize(v - n * (v * n).sum(dim=-1, keepdim=True), eps)


def rotate_around_axis(v: torch.Tensor, axis: torch.Tensor, angle: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    axis = normalize(axis, eps)
    c = torch.cos(angle).unsqueeze(-1)
    s = torch.sin(angle).unsqueeze(-1)
    return (
        v * c
        + torch.cross(axis, v, dim=-1) * s
        + axis * (axis * v).sum(dim=-1, keepdim=True) * (1.0 - c)
    )


def stable_perpendicular(axis: torch.Tensor) -> torch.Tensor:
    axis = normalize(axis)
    up = axis.new_tensor((0.0, 0.0, 1.0)).expand_as(axis)
    side = axis.new_tensor((1.0, 0.0, 0.0)).expand_as(axis)
    ref = torch.where(axis[..., 2:3].abs() < 0.8, up, side)
    return normalize(torch.cross(axis, ref, dim=-1))


def swing_only_transport(
    rest_axis: torch.Tensor,
    current_axis: torch.Tensor,
    rest_pole: torch.Tensor,
) -> torch.Tensor:
    a = normalize(rest_axis)
    b = normalize(current_axis)
    if a.ndim < b.ndim:
        a = a.reshape((1,) * (b.ndim - a.ndim) + a.shape).expand_as(b)
    pole = rest_pole.expand_as(b) if rest_pole.ndim < b.ndim else rest_pole
    c = (a * b).sum(dim=-1).clamp(-1.0, 1.0)
    same = c > 0.999999
    opposite = c < -0.999999

    swing_axis = normalize(torch.cross(a, b, dim=-1))
    swing_angle = torch.acos(c)
    normal_out = rotate_around_axis(pole, swing_axis, swing_angle)

    opposite_axis = stable_perpendicular(a)
    opposite_out = rotate_around_axis(pole, opposite_axis, torch.full_like(c, math.pi))
    return torch.where(
        same.unsqueeze(-1),
        pole,
        torch.where(opposite.unsqueeze(-1), opposite_out, normal_out),
    )


def encode_pole_float(
    base_pos: torch.Tensor,
    mid_pos: torch.Tensor,
    end_pos: torch.Tensor,
    rest_axis: torch.Tensor,
    rest_pole: torch.Tensor,
    alpha: float = IK_POLE_ALPHA,
) -> torch.Tensor:
    axis = normalize(end_pos - base_pos)
    natural = project_to_plane(swing_only_transport(rest_axis, axis, rest_pole), axis)
    actual = project_to_plane(mid_pos - base_pos, axis)
    sin_v = (torch.cross(natural, actual, dim=-1) * axis).sum(dim=-1)
    cos_v = (natural * actual).sum(dim=-1)
    return torch.atan2(sin_v, cos_v) / float(alpha)


def solve_two_bone_with_pole(
    base_pos: torch.Tensor,
    end_pos: torch.Tensor,
    l1: torch.Tensor,
    l2: torch.Tensor,
    rest_axis: torch.Tensor,
    rest_pole: torch.Tensor,
    pole_float: torch.Tensor,
    alpha: float = IK_POLE_ALPHA,
) -> torch.Tensor:
    raw = end_pos - base_pos
    d_raw = torch.linalg.norm(raw, dim=-1, keepdim=True)
    axis = normalize(raw)
    min_d = (l1 - l2).abs().unsqueeze(-1) + 1e-5
    max_d = (l1 + l2).unsqueeze(-1) - 1e-5
    d = d_raw.clamp_min(1e-8).clamp(min=min_d, max=max_d)

    natural = project_to_plane(swing_only_transport(rest_axis, axis, rest_pole), axis)
    theta = pole_float.reshape(-1) * float(alpha)
    pole = rotate_around_axis(natural, axis, theta)

    a = (l1.unsqueeze(-1).square() - l2.unsqueeze(-1).square() + d.square()) / (2.0 * d)
    h = torch.sqrt((l1.unsqueeze(-1).square() - a.square()).clamp_min(0.0))
    return base_pos + axis * a + pole * h


def root_relative_rotations(global_rot: torch.Tensor, root_rot: torch.Tensor) -> torch.Tensor:
    return global_rot @ root_rot[:, None, :, :].transpose(-1, -2)


def root_relative_rot_to_world(rot_root: torch.Tensor, root_rot: torch.Tensor) -> torch.Tensor:
    return rot_root @ root_rot


def axis_angle_to_row_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    batch = angle.shape[0]
    if axis.ndim == 1:
        axis = axis.reshape(1, 3).expand(batch, 3)
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device).expand(batch, 3, 3)
    axis_rows = axis.reshape(batch, 1, 3).expand(batch, 3, 3)
    angle_rows = angle.reshape(batch, 1).expand(batch, 3)
    return rotate_around_axis(eye, axis_rows, angle_rows)


def rotation_from_axis_with_reference(
    local_axis: torch.Tensor,
    world_axis: torch.Tensor,
    reference_rot: torch.Tensor,
) -> torch.Tensor:
    b = world_axis.shape[0]
    local_axis = normalize(local_axis.to(dtype=world_axis.dtype, device=world_axis.device))
    if local_axis.ndim == 1:
        local_axis = local_axis.reshape(1, 3).expand(b, 3)
    local_side = stable_perpendicular(local_axis)
    local_up = normalize(torch.cross(local_axis, local_side, dim=-1))
    local_basis = torch.stack((local_axis, local_side, local_up), dim=-2)

    world_main = normalize(world_axis)
    ref_side = torch.matmul(local_side.unsqueeze(1), reference_rot).squeeze(1)
    world_side_raw = ref_side - world_main * (ref_side * world_main).sum(dim=-1, keepdim=True)
    fallback_side = stable_perpendicular(world_main)
    world_side = torch.where(
        torch.linalg.norm(world_side_raw, dim=-1, keepdim=True) > 1e-8,
        normalize(world_side_raw),
        fallback_side,
    )
    world_up = normalize(torch.cross(world_main, world_side, dim=-1))
    world_basis = torch.stack((world_main, world_side, world_up), dim=-2)
    return local_basis.transpose(-1, -2) @ world_basis


def rotation_from_axis_and_pole(
    local_axis: torch.Tensor,
    world_axis: torch.Tensor,
    local_pole: torch.Tensor,
    world_pole: torch.Tensor,
) -> torch.Tensor:
    b = world_axis.shape[0]
    local_axis = normalize(local_axis.to(dtype=world_axis.dtype, device=world_axis.device))
    if local_axis.ndim == 1:
        local_axis = local_axis.reshape(1, 3).expand(b, 3)
    local_pole = local_pole.to(dtype=world_axis.dtype, device=world_axis.device)
    if local_pole.ndim == 1:
        local_pole = local_pole.reshape(1, 3).expand(b, 3)
    local_side = project_to_plane(local_pole, local_axis)
    local_up = normalize(torch.cross(local_axis, local_side, dim=-1))
    local_basis = torch.stack((local_axis, local_side, local_up), dim=-2)

    world_main = normalize(world_axis)
    world_side = project_to_plane(world_pole, world_main)
    world_up = normalize(torch.cross(world_main, world_side, dim=-1))
    world_basis = torch.stack((world_main, world_side, world_up), dim=-2)
    return local_basis.transpose(-1, -2) @ world_basis


def clean_ik_payload(payload: torch.Tensor) -> torch.Tensor:
    if payload.shape[-1] != IK_PAYLOAD_DIM:
        raise ValueError(f"IK payload must have {IK_PAYLOAD_DIM} dims, got {payload.shape[-1]}")
    parts: list[torch.Tensor] = []
    cursor = 0
    for spec in IK_PAYLOAD_SLICES:
        parts.append(payload[:, cursor : cursor + 3])
        cursor += 3
        rot_slice = spec["rot6"]
        assert isinstance(rot_slice, slice)
        parts.append(clean_6d(payload[:, rot_slice]))
        cursor += 6
        parts.append(payload[:, cursor : cursor + 1])
        cursor += 1
        toe_slice = spec["toe_float"]
        if toe_slice is not None:
            parts.append(payload[:, cursor : cursor + 1])
            cursor += 1
    return torch.cat(parts, dim=-1)


def split_ik_payload(payload: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if payload.shape[-1] != IK_PAYLOAD_DIM:
        raise ValueError(f"IK payload must have {IK_PAYLOAD_DIM} dims, got {payload.shape[-1]}")
    ee_pos: list[torch.Tensor] = []
    ee_rot6: list[torch.Tensor] = []
    pole: list[torch.Tensor] = []
    toe: list[torch.Tensor] = []
    for spec in IK_PAYLOAD_SLICES:
        pos_slice = spec["pos"]
        rot_slice = spec["rot6"]
        pole_slice = spec["pole"]
        toe_slice = spec["toe_float"]
        assert isinstance(pos_slice, slice)
        assert isinstance(rot_slice, slice)
        assert isinstance(pole_slice, slice)
        ee_pos.append(payload[:, pos_slice])
        ee_rot6.append(payload[:, rot_slice])
        pole.append(payload[:, pole_slice].squeeze(-1))
        if toe_slice is not None:
            assert isinstance(toe_slice, slice)
            toe.append(payload[:, toe_slice].squeeze(-1))
    toe_tensor = torch.stack(toe, dim=1) if toe else payload.new_empty((payload.shape[0], 0))
    return torch.stack(ee_pos, dim=1), torch.stack(ee_rot6, dim=1), torch.stack(pole, dim=1), toe_tensor


class MotionClip:
    def __init__(self, path: Path, cfg: TrainConfig, cyclic_animation: bool | None = None):
        arrays = np.load(path)
        self.path = path
        self.pose_representation = str(cfg.pose_representation).lower().strip()
        if self.pose_representation not in ("rot6", IK_POSE_REPRESENTATION):
            raise ValueError(f"unknown pose_representation={cfg.pose_representation!r}; use 'rot6' or 'ik_markers'")
        self.cyclic_animation = bool(cfg.cyclic_animation if cyclic_animation is None else cyclic_animation)
        self.source_up_axis = axis_up_axis(arrays)
        self.bone_names = [str(x) for x in arrays["bone_names"]]
        self.parents_full = arrays["parents"].astype(np.int64)
        self.fps = float(arrays["fps"])
        if abs(self.fps - cfg.fps) > 1e-3:
            raise ValueError(f"{path} is {self.fps} FPS, config expects {cfg.fps} FPS")

        self.keep_full = [i for i, n in enumerate(self.bone_names) if should_keep_bone(n, cfg)]
        self.body_names = [self.bone_names[i] for i in self.keep_full]
        if "pelvis" not in self.body_names:
            raise ValueError(f"{path} does not contain a kept pelvis bone")
        self.pelvis = self.body_names.index("pelvis")
        self.non_pelvis = [i for i, n in enumerate(self.body_names) if n != "pelvis"]
        self.ik_marker_names: list[str] = []
        self.ik_marker_indices: list[int] = []
        self.ik_marker_indices_tensor = torch.empty((0,), dtype=torch.long)
        self.end_effectors = []
        for name in cfg.end_effector_bones:
            if name not in self.body_names:
                raise ValueError(f"{path} is missing end effector bone {name!r}")
            self.end_effectors.append(self.body_names.index(name))
        self.end_effectors_tensor = torch.tensor(self.end_effectors, dtype=torch.long)
        self.foot_indices = []
        self.toe_indices = []
        for foot_name, toe_name in (("foot_l", "ball_l"), ("foot_r", "ball_r")):
            if foot_name not in self.body_names or toe_name not in self.body_names:
                raise ValueError(f"{path} is missing contact bones {foot_name!r}/{toe_name!r}")
            self.foot_indices.append(self.body_names.index(foot_name))
            self.toe_indices.append(self.body_names.index(toe_name))
        self.foot_indices_tensor = torch.tensor(self.foot_indices, dtype=torch.long)
        self.toe_indices_tensor = torch.tensor(self.toe_indices, dtype=torch.long)
        full_to_body = {full_i: body_i for body_i, full_i in enumerate(self.keep_full)}
        parents_body = []
        for full_i in self.keep_full:
            parent = int(self.parents_full[full_i])
            parents_body.append(full_to_body.get(parent, -1))
        self.parents_body = torch.tensor(parents_body, dtype=torch.long)

        has_model_space = all(
            key in arrays.files
            for key in (
                "model_global_joint_pos_m",
                "model_global_matrix",
                "model_local_matrix",
                "model_lcl_translation_m",
                "model_default_lcl_translation_m",
            )
        )
        model_local_rot6_full = None
        if has_model_space:
            global_pos_full = torch.tensor(arrays["model_global_joint_pos_m"], dtype=torch.float32)
            global_rot_full = torch.tensor(arrays["model_global_matrix"][:, :, :3, :3], dtype=torch.float32)
            local_rot_full = torch.tensor(arrays["model_local_matrix"][:, :, :3, :3], dtype=torch.float32)
            lcl_translation_full = torch.tensor(arrays["model_lcl_translation_m"], dtype=torch.float32)
            default_lcl_translation_full = torch.tensor(arrays["model_default_lcl_translation_m"], dtype=torch.float32)
            if "model_local_rotation_6d" in arrays.files:
                model_local_rot6_full = torch.tensor(arrays["model_local_rotation_6d"], dtype=torch.float32)
        else:
            global_pos_full = canonicalize_positions(
                torch.tensor(arrays["global_joint_pos"], dtype=torch.float32) * cfg.position_unit_scale,
                self.source_up_axis,
            )
            global_rot_full = canonicalize_rotations(
                torch.tensor(arrays["global_matrix"][:, :, :3, :3], dtype=torch.float32),
                self.source_up_axis,
            )
            local_rot_full = canonicalize_rotations(
                torch.tensor(arrays["local_matrix"][:, :, :3, :3], dtype=torch.float32),
                self.source_up_axis,
            )
            lcl_translation_full = canonicalize_positions(
                torch.tensor(arrays["fbx_lcl_translation"], dtype=torch.float32) * cfg.position_unit_scale,
                self.source_up_axis,
            )
            default_lcl_translation_full = (
                torch.tensor(arrays["default_lcl_translation"], dtype=torch.float32) * cfg.position_unit_scale
            )
            default_lcl_translation_full = canonicalize_positions(default_lcl_translation_full, self.source_up_axis)

        root_index = self.bone_names.index("root")
        self.root_pos = global_pos_full[:, root_index]
        self.root_rot = global_rot_full[:, root_index]
        self.root_yaw = heading_yaw_from_root(self.root_rot)
        self.root_heading_rot = yaw_to_row_matrix(self.root_yaw)
        self.T = int(global_pos_full.shape[0])
        self.cyclic_period = max(1, self.T - 1)

        keep = torch.tensor(self.keep_full, dtype=torch.long)
        self.global_pos = global_pos_full.index_select(1, keep)
        self.global_rot = global_rot_full.index_select(1, keep)
        self.local_rot = local_rot_full.index_select(1, keep)
        if model_local_rot6_full is not None:
            self.local_rot6 = model_local_rot6_full.index_select(1, keep)
        else:
            self.local_rot6 = rotmat_to_6d(self.local_rot)
        self.local_offsets = default_lcl_translation_full.index_select(0, keep)
        self.pelvis_local_pos = lcl_translation_full[:, self.keep_full[self.pelvis]]
        self.pelvis_rot_mat = self.local_rot[:, self.pelvis]
        self.non_pelvis_rot_mat = self.local_rot[:, self.non_pelvis]
        self.pelvis_rot6 = self.local_rot6[:, self.pelvis]
        self.non_pelvis_rot6 = self.local_rot6[:, self.non_pelvis]
        self.parents_body_list = [int(parent) for parent in self.parents_body.tolist()]
        rest_pos_list: list[torch.Tensor] = []
        for j, parent in enumerate(self.parents_body_list):
            if parent < 0:
                rest_pos_list.append(self.local_offsets[j])
            else:
                rest_pos_list.append(rest_pos_list[parent] + self.local_offsets[j])
        self.rest_body_pos = torch.stack(rest_pos_list, dim=0)
        self.ik_controlled_names = set()
        self.ik_limb_specs: list[dict[str, object]] = []
        if uses_ik_markers(self.pose_representation):
            for spec in IK_LIMB_SPECS:
                names = [str(spec["start"]), str(spec["mid"]), str(spec["end"])]
                if spec.get("toe") is not None:
                    names.append(str(spec["toe"]))
                missing = [name for name in names if name not in self.body_names]
                if missing:
                    raise ValueError(f"{path} is missing IK chain bones: {missing}")
                mapped = {
                    "side": str(spec["side"]),
                    "kind": str(spec["kind"]),
                    "start": self.body_names.index(str(spec["start"])),
                    "mid": self.body_names.index(str(spec["mid"])),
                    "end": self.body_names.index(str(spec["end"])),
                    "toe": self.body_names.index(str(spec["toe"])) if spec.get("toe") is not None else None,
                }
                self.ik_limb_specs.append(mapped)
                self.ik_controlled_names.update(names)
            self.ik_marker_names = ["hand_l", "hand_r", "foot_l", "ball_l", "foot_r", "ball_r"]
            self.ik_marker_indices = [self.body_names.index(name) for name in self.ik_marker_names]
            self.ik_marker_indices_tensor = torch.tensor(self.ik_marker_indices, dtype=torch.long)
        self.ik_controlled = [self.body_names.index(name) for name in sorted(self.ik_controlled_names) if name in self.body_names]
        self.ik_controlled_set = set(self.ik_controlled)
        self.core_non_pelvis = [i for i in self.non_pelvis if i not in self.ik_controlled_set]
        self.core_nonpelvis_map = {bone_index: i for i, bone_index in enumerate(self.core_non_pelvis)}
        self.core_non_pelvis_rot6 = self.local_rot6[:, self.core_non_pelvis] if self.core_non_pelvis else torch.empty((self.T, 0, 6))
        self.mass_weights = torch.tensor(body_mass.bone_masses_for_names(self.body_names), dtype=torch.float32)

        root_delta = self.global_pos - self.root_pos[:, None, :]
        # Row-vector convention: root_heading_rot maps world deltas into the
        # root-heading frame. Keep this in the same basis as root/future deltas.
        self.canonical_pos = torch.einsum("tjc,tcd->tjd", root_delta, self.root_heading_rot)
        self.root_relative_pos = root_relative_positions(self.global_pos, self.root_pos, self.root_rot)
        self.ik_payload_dim = IK_PAYLOAD_DIM if uses_ik_markers(self.pose_representation) else 0
        self.ik_marker_pos = torch.empty((self.T, 0, 3), dtype=torch.float32)
        self.ik_payload = torch.empty((self.T, 0), dtype=torch.float32)
        self.ik_ee_pos_root = torch.empty((self.T, 0, 3), dtype=torch.float32)
        self.ik_ee_rot6_root = torch.empty((self.T, 0, 6), dtype=torch.float32)
        self.ik_pole_float = torch.empty((self.T, 0), dtype=torch.float32)
        self.ik_toe_float = torch.empty((self.T, 0), dtype=torch.float32)
        self.ik_rest_axis = torch.empty((0, 3), dtype=torch.float32)
        self.ik_rest_pole = torch.empty((0, 3), dtype=torch.float32)
        self.ik_limb_lengths = torch.empty((0, 2), dtype=torch.float32)
        self.ik_local_pole_axis = torch.empty((0, 2, 3), dtype=torch.float32)
        self.ik_toe_offsets = torch.empty((0, 3), dtype=torch.float32)
        self.ik_toe_axis = torch.empty((0, 3), dtype=torch.float32)
        if uses_ik_markers(self.pose_representation):
            root_inv = self.root_rot.transpose(-1, -2)
            forward = torch.tensor(IK_CHARACTER_FORWARD, dtype=torch.float32)
            axis_candidates = torch.tensor(
                (
                    (1.0, 0.0, 0.0),
                    (-1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, -1.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (0.0, 0.0, -1.0),
                ),
                dtype=torch.float32,
            )
            payload_chunks: list[torch.Tensor] = []
            ee_pos_chunks: list[torch.Tensor] = []
            ee_rot6_chunks: list[torch.Tensor] = []
            pole_chunks: list[torch.Tensor] = []
            toe_chunks: list[torch.Tensor] = []
            rest_axis_chunks: list[torch.Tensor] = []
            rest_pole_chunks: list[torch.Tensor] = []
            length_chunks: list[torch.Tensor] = []
            local_pole_axis_chunks: list[torch.Tensor] = []
            toe_offset_chunks: list[torch.Tensor] = []
            toe_axis_chunks: list[torch.Tensor] = []
            for spec in self.ik_limb_specs:
                kind = str(spec["kind"])
                start = int(spec["start"])
                mid = int(spec["mid"])
                end = int(spec["end"])
                toe = spec.get("toe")

                ee_pos = self.root_relative_pos[:, end]
                ee_rot_root = self.global_rot[:, end] @ root_inv
                ee_rot6 = rotmat_to_6d(ee_rot_root)
                rest_axis = normalize(self.rest_body_pos[end] - self.rest_body_pos[start])
                rest_pole = -forward if kind == "arm" else forward
                pole_float = encode_pole_float(
                    self.root_relative_pos[:, start],
                    self.root_relative_pos[:, mid],
                    self.root_relative_pos[:, end],
                    rest_axis,
                    rest_pole,
                ).unsqueeze(-1)

                payload_chunks.extend((ee_pos, ee_rot6, pole_float))
                ee_pos_chunks.append(ee_pos)
                ee_rot6_chunks.append(ee_rot6)
                pole_chunks.append(pole_float.squeeze(-1))
                rest_axis_chunks.append(rest_axis)
                rest_pole_chunks.append(rest_pole)
                chain_axis_gt = normalize(self.global_pos[:, end] - self.global_pos[:, start])
                chain_pole_gt = project_to_plane(self.global_pos[:, mid] - self.global_pos[:, start], chain_axis_gt)
                start_axis = normalize(self.local_offsets[mid]).expand(self.T, 3)
                mid_axis = normalize(self.local_offsets[end]).expand(self.T, 3)
                start_local_pole = torch.matmul(
                    chain_pole_gt.unsqueeze(1),
                    self.global_rot[:, start].transpose(-1, -2),
                ).squeeze(1)
                mid_local_pole = torch.matmul(
                    chain_pole_gt.unsqueeze(1),
                    self.global_rot[:, mid].transpose(-1, -2),
                ).squeeze(1)
                start_local_pole = project_to_plane(start_local_pole, start_axis)[0]
                mid_local_pole = project_to_plane(mid_local_pole, mid_axis)[0]
                local_pole_axis_chunks.append(torch.stack((start_local_pole, mid_local_pole), dim=0))
                length_chunks.append(
                    torch.stack(
                        (
                            torch.linalg.norm(self.local_offsets[mid]),
                            torch.linalg.norm(self.local_offsets[end]),
                        )
                    )
                )
                if toe is not None:
                    toe_i = int(toe)
                    best_toe_axis = axis_candidates[0]
                    best_toe_err = float("inf")
                    for candidate in axis_candidates:
                        axis = candidate.expand(self.T, 3)
                        ref = stable_perpendicular(axis)
                        rotated_ref = torch.matmul(ref.unsqueeze(1), self.local_rot[:, toe_i]).squeeze(1)
                        rotated_ref = project_to_plane(rotated_ref, axis)
                        sin_v = (torch.cross(ref, rotated_ref, dim=-1) * axis).sum(dim=-1)
                        cos_v = (ref * rotated_ref).sum(dim=-1)
                        toe_angle = torch.atan2(sin_v, cos_v)
                        recon = axis_angle_to_row_matrix(axis, toe_angle)
                        err_f = float(geodesic_angles(recon, self.local_rot[:, toe_i]).mean().detach().cpu())
                        if err_f < best_toe_err:
                            best_toe_err = err_f
                            best_toe_axis = candidate
                    axis = best_toe_axis.expand(self.T, 3)
                    ref = stable_perpendicular(axis)
                    rotated_ref = torch.matmul(ref.unsqueeze(1), self.local_rot[:, toe_i]).squeeze(1)
                    rotated_ref = project_to_plane(rotated_ref, axis)
                    sin_v = (torch.cross(ref, rotated_ref, dim=-1) * axis).sum(dim=-1)
                    cos_v = (ref * rotated_ref).sum(dim=-1)
                    toe_float = (torch.atan2(sin_v, cos_v) / IK_TOE_ALPHA).unsqueeze(-1)
                    payload_chunks.append(toe_float)
                    toe_chunks.append(toe_float.squeeze(-1))
                    toe_offset_chunks.append(self.local_offsets[toe_i])
                else:
                    best_toe_axis = torch.tensor(IK_FOOT_LOCAL_SIDE_AXIS, dtype=torch.float32)
                    toe_offset_chunks.append(torch.zeros(3, dtype=torch.float32))
                toe_axis_chunks.append(best_toe_axis)

            self.ik_payload = torch.cat(payload_chunks, dim=-1)
            self.ik_ee_pos_root = torch.stack(ee_pos_chunks, dim=1)
            self.ik_ee_rot6_root = torch.stack(ee_rot6_chunks, dim=1)
            self.ik_pole_float = torch.stack(pole_chunks, dim=1)
            self.ik_toe_float = torch.stack(toe_chunks, dim=1) if toe_chunks else torch.empty((self.T, 0))
            self.ik_rest_axis = torch.stack(rest_axis_chunks, dim=0)
            self.ik_rest_pole = torch.stack(rest_pole_chunks, dim=0)
            self.ik_limb_lengths = torch.stack(length_chunks, dim=0)
            self.ik_local_pole_axis = torch.stack(local_pole_axis_chunks, dim=0)
            self.ik_toe_offsets = torch.stack(toe_offset_chunks, dim=0)
            self.ik_toe_axis = torch.stack(toe_axis_chunks, dim=0)
            self.ik_marker_pos = self.root_relative_pos.index_select(1, self.ik_marker_indices_tensor)
            if "model_ik_payload" in arrays.files:
                required = (
                    "model_ik_rest_axis",
                    "model_ik_rest_pole",
                    "model_ik_limb_lengths",
                    "model_ik_local_pole_axis",
                    "model_ik_toe_offsets",
                    "model_ik_toe_axis",
                )
                missing = [key for key in required if key not in arrays.files]
                if missing:
                    raise ValueError(f"{path} is missing model IK metadata: {missing}")
                model_payload = torch.tensor(arrays["model_ik_payload"], dtype=torch.float32)
                if model_payload.shape != (self.T, IK_PAYLOAD_DIM):
                    raise ValueError(
                        f"{path} model_ik_payload shape {tuple(model_payload.shape)} "
                        f"does not match {(self.T, IK_PAYLOAD_DIM)}"
                    )
                self.ik_payload = model_payload
                self.ik_ee_pos_root, self.ik_ee_rot6_root, self.ik_pole_float, self.ik_toe_float = split_ik_payload(
                    self.ik_payload
                )
                self.ik_rest_axis = torch.tensor(arrays["model_ik_rest_axis"], dtype=torch.float32)
                self.ik_rest_pole = torch.tensor(arrays["model_ik_rest_pole"], dtype=torch.float32)
                self.ik_limb_lengths = torch.tensor(arrays["model_ik_limb_lengths"], dtype=torch.float32)
                self.ik_local_pole_axis = torch.tensor(arrays["model_ik_local_pole_axis"], dtype=torch.float32)
                self.ik_toe_offsets = torch.tensor(arrays["model_ik_toe_offsets"], dtype=torch.float32)
                self.ik_toe_axis = torch.tensor(arrays["model_ik_toe_axis"], dtype=torch.float32)

        self.J = len(self.body_names)
        self.Jn = len(self.non_pelvis)
        self.Jcore = len(self.core_non_pelvis)
        self.Jmarkers = len(self.ik_marker_indices)
        self.nonpelvis_map = {bone_index: i for i, bone_index in enumerate(self.non_pelvis)}
        self._device_cache: dict[str, dict[str, torch.Tensor]] = {}

    def pose_at(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        pose = {
            "pelvis_pos": self.pelvis_local_pos[idx],
            "pelvis_rot6": self.pelvis_rot6[idx],
            "nonpelvis_rot6": self.non_pelvis_rot6[idx],
            "canon_pos": self.canonical_pos[idx],
        }
        if uses_ik_markers(self.pose_representation):
            pose["core_nonpelvis_rot6"] = self.core_non_pelvis_rot6[idx]
            pose["ik_payload"] = self.ik_payload[idx]
        return pose

    def tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        key = str(device)
        cached = self._device_cache.get(key)
        if cached is None:
            cached = {
                "root_pos": self.root_pos.to(device),
                "root_rot": self.root_rot.to(device),
                "root_yaw": self.root_yaw.to(device),
                "root_heading_rot": self.root_heading_rot.to(device),
                "global_pos": self.global_pos.to(device),
                "global_rot": self.global_rot.to(device),
                "local_offsets": self.local_offsets.to(device),
                "pelvis_local_pos": self.pelvis_local_pos.to(device),
                "pelvis_rot_mat": self.pelvis_rot_mat.to(device),
                "non_pelvis_rot_mat": self.non_pelvis_rot_mat.to(device),
                "pelvis_rot6": self.pelvis_rot6.to(device),
                "non_pelvis_rot6": self.non_pelvis_rot6.to(device),
                "core_non_pelvis_rot6": self.core_non_pelvis_rot6.to(device),
                "canonical_pos": self.canonical_pos.to(device),
                "root_relative_pos": self.root_relative_pos.to(device),
                "ik_marker_pos": self.ik_marker_pos.to(device),
                "ik_payload": self.ik_payload.to(device),
                "ik_ee_pos_root": self.ik_ee_pos_root.to(device),
                "ik_ee_rot6_root": self.ik_ee_rot6_root.to(device),
                "ik_pole_float": self.ik_pole_float.to(device),
                "ik_toe_float": self.ik_toe_float.to(device),
                "ik_rest_axis": self.ik_rest_axis.to(device),
                "ik_rest_pole": self.ik_rest_pole.to(device),
                "ik_limb_lengths": self.ik_limb_lengths.to(device),
                "ik_local_pole_axis": self.ik_local_pole_axis.to(device),
                "ik_toe_offsets": self.ik_toe_offsets.to(device),
                "ik_toe_axis": self.ik_toe_axis.to(device),
                "mass_weights": self.mass_weights.to(device),
                "end_effectors": self.end_effectors_tensor.to(device),
                "foot_indices": self.foot_indices_tensor.to(device),
                "toe_indices": self.toe_indices_tensor.to(device),
                "ik_marker_indices": self.ik_marker_indices_tensor.to(device),
            }
            self._device_cache[key] = cached
        return cached


class MotionIndexDataset(Dataset):
    def __init__(self, clips: list[MotionClip], cfg: TrainConfig, split: str, max_rollout: int):
        self.items: list[tuple[int, int]] = []
        for ci, clip in enumerate(clips):
            max_start = clip_rollout_max_start(clip, max_rollout, cfg)
            if max_start < 1:
                continue
            starts = list(range(1, max_start + 1))
            random.Random(cfg.seed + ci).shuffle(starts)
            if cfg.disable_validation or cfg.val_fraction <= 0.0:
                chosen = starts
            else:
                val_count = max(1, int(round(len(starts) * cfg.val_fraction))) if len(starts) > 1 else 0
                chosen = starts[:val_count] if split == "val" else starts[val_count:]
            self.items.extend((ci, s) for s in chosen)
        if not self.items:
            raise ValueError(
                f"No {split} samples. Need longer clips or smaller future_window_seconds/max rollout."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[int, int]:
        return self.items[index]


class MLPController(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, cfg: TrainConfig):
        super().__init__()
        act_cls = getattr(nn, cfg.activation)
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(cfg.num_hidden_layers):
            layers.append(nn.Linear(in_dim, cfg.hidden_dim))
            layers.append(nn.LayerNorm(cfg.hidden_dim))
            layers.append(act_cls())
            in_dim = cfg.hidden_dim
        output = nn.Linear(in_dim, output_dim)
        if cfg.zero_init_output:
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def body_pose_vector(
    pose: dict[str, torch.Tensor],
) -> torch.Tensor:
    b = pose["pelvis_pos"].shape[0]
    if "ik_payload" in pose:
        parts = [
            pose["pelvis_pos"],
            pose["pelvis_rot6"],
            pose["core_nonpelvis_rot6"].reshape(b, -1),
            pose["ik_payload"],
        ]
    else:
        parts = [
            pose["pelvis_pos"],
            pose["pelvis_rot6"],
            pose["canon_pos"].reshape(b, -1),
            pose["nonpelvis_rot6"].reshape(b, -1),
        ]
    return torch.cat(parts, dim=-1)


def output_to_pose(raw: torch.Tensor, clip: MotionClip) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    b = raw.shape[0]
    cursor = 0
    pelvis_pos = raw[:, cursor : cursor + 3]
    cursor += 3
    pelvis_rot6_raw = raw[:, cursor : cursor + 6]
    cursor += 6
    if uses_ik_markers(clip.pose_representation):
        core_end = cursor + clip.Jcore * 6
        core_rot6_raw = raw[:, cursor:core_end].reshape(b, clip.Jcore, 6)
        cursor = core_end
        payload_end = cursor + clip.ik_payload_dim
        ik_payload_raw = raw[:, cursor:payload_end]
        ik_payload = clean_ik_payload(ik_payload_raw)
        cursor = payload_end
        core_rot6 = clean_6d(core_rot6_raw.reshape(-1, 6)).reshape(b, clip.Jcore, 6)
        nonpelvis_rot6 = torch.zeros((b, clip.Jn, 6), dtype=raw.dtype, device=raw.device)
        if clip.Jn > 0:
            nonpelvis_rot6[..., 0] = 1.0
            nonpelvis_rot6[..., 4] = 1.0
        for body_i, core_slot in clip.core_nonpelvis_map.items():
            nonpelvis_slot = clip.nonpelvis_map[body_i]
            nonpelvis_rot6[:, nonpelvis_slot] = core_rot6[:, core_slot]
        nonpelvis_rot6_raw = nonpelvis_rot6
    else:
        nonpelvis_end = cursor + clip.Jn * 6
        nonpelvis_rot6_raw = raw[:, cursor:nonpelvis_end].reshape(b, clip.Jn, 6)
        cursor = nonpelvis_end
        core_rot6_raw = None
        core_rot6 = None
        ik_payload_raw = None
        ik_payload = None
    pelvis_rot6 = clean_6d(pelvis_rot6_raw)
    if not uses_ik_markers(clip.pose_representation):
        nonpelvis_rot6 = clean_6d(nonpelvis_rot6_raw.reshape(-1, 6)).reshape(b, clip.Jn, 6)
    clean_pose = {
        "pelvis_pos": pelvis_pos,
        "pelvis_rot6": pelvis_rot6,
        "nonpelvis_rot6": nonpelvis_rot6,
    }
    raw_pose = {
        "pelvis_rot6": pelvis_rot6_raw,
        "nonpelvis_rot6": nonpelvis_rot6_raw,
    }
    if uses_ik_markers(clip.pose_representation):
        assert core_rot6 is not None and core_rot6_raw is not None and ik_payload is not None
        clean_pose["core_nonpelvis_rot6"] = core_rot6
        clean_pose["ik_payload"] = ik_payload
        raw_pose["core_nonpelvis_rot6"] = core_rot6_raw
        raw_pose["ik_payload"] = ik_payload_raw
    return clean_pose, raw_pose


def pose_target_output(pose: dict[str, torch.Tensor]) -> torch.Tensor:
    b = pose["pelvis_pos"].shape[0]
    if "ik_payload" in pose:
        return torch.cat(
            (
                pose["pelvis_pos"],
                pose["pelvis_rot6"],
                pose["core_nonpelvis_rot6"].reshape(b, -1),
                pose["ik_payload"],
            ),
            dim=-1,
        )
    return torch.cat(
        (
            pose["pelvis_pos"],
            pose["pelvis_rot6"],
            pose["nonpelvis_rot6"].reshape(b, -1),
        ),
        dim=-1,
    )


def next_pose_from_prediction(pred_pose: dict[str, torch.Tensor], canon_pos: torch.Tensor) -> dict[str, torch.Tensor]:
    pose = {
        "pelvis_pos": pred_pose["pelvis_pos"],
        "pelvis_rot6": pred_pose["pelvis_rot6"],
        "nonpelvis_rot6": pred_pose["nonpelvis_rot6"],
        "canon_pos": canon_pos,
    }
    if "ik_payload" in pred_pose:
        pose["core_nonpelvis_rot6"] = pred_pose["core_nonpelvis_rot6"]
        pose["ik_payload"] = pred_pose["ik_payload"]
    return pose


def predict_next_raw(
    model: nn.Module,
    inp: torch.Tensor,
    cur_pose: dict[str, torch.Tensor],
    cfg: TrainConfig,
) -> torch.Tensor:
    raw = model(inp)
    if cfg.predict_residual:
        raw = pose_target_output(cur_pose) + raw
    return raw


def fk_from_pose(
    clip: MotionClip,
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    pose: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b = root_pos.shape[0]
    pelvis_rot = rotation_6d_to_matrix(pose["pelvis_rot6"])
    tensors = clip.tensors(device)
    if "ik_payload" in pose:
        core_rot = rotation_6d_to_matrix(pose["core_nonpelvis_rot6"])
        offsets = tensors["local_offsets"].unsqueeze(0).expand(b, -1, -1).clone()
        offsets[:, clip.pelvis] = pose["pelvis_pos"]
        identity = torch.eye(3, dtype=root_rot.dtype, device=device).expand(b, 3, 3)

        global_pos_list: list[torch.Tensor] = []
        global_rot_list: list[torch.Tensor] = []
        for j in range(clip.J):
            if j == clip.pelvis:
                local_rot_j = pelvis_rot
            elif j in clip.core_nonpelvis_map:
                local_rot_j = core_rot[:, clip.core_nonpelvis_map[j]]
            else:
                local_rot_j = identity
            parent = clip.parents_body_list[j]
            if parent < 0:
                rot_j = local_rot_j @ root_rot
                pos_j = torch.matmul(offsets[:, j].unsqueeze(1), root_rot).squeeze(1) + root_pos
            else:
                parent_rot = global_rot_list[parent]
                parent_pos = global_pos_list[parent]
                rot_j = local_rot_j @ parent_rot
                pos_j = torch.matmul(offsets[:, j].unsqueeze(1), parent_rot).squeeze(1) + parent_pos
            global_rot_list.append(rot_j)
            global_pos_list.append(pos_j)

        up_hint = torch.matmul(
            torch.tensor([0.0, 1.0, 0.0], dtype=root_rot.dtype, device=device).view(1, 1, 3),
            root_rot,
        ).squeeze(1)
        payload = pose["ik_payload"]
        cursor = 0
        changed_bones: set[int] = set()
        for limb_i, spec in enumerate(clip.ik_limb_specs):
            start = int(spec["start"])
            mid = int(spec["mid"])
            end = int(spec["end"])
            end_root = payload[:, cursor : cursor + 3]
            cursor += 3
            end_rot_root = rotation_6d_to_matrix(payload[:, cursor : cursor + 6])
            cursor += 6
            pole_float = payload[:, cursor]
            cursor += 1

            base = global_pos_list[start]
            base_root = torch.matmul((base - root_pos).unsqueeze(1), root_rot.transpose(-1, -2)).squeeze(1)
            l1 = tensors["ik_limb_lengths"][limb_i, 0].to(dtype=root_rot.dtype).expand(b)
            l2 = tensors["ik_limb_lengths"][limb_i, 1].to(dtype=root_rot.dtype).expand(b)
            delta = end_root - base_root
            axis = normalize(delta)
            d = torch.linalg.norm(delta, dim=-1, keepdim=True)
            min_d = (l1 - l2).abs().unsqueeze(-1) + 1e-5
            max_d = (l1 + l2).unsqueeze(-1) - 1e-5
            d_clamped = d.clamp_min(1e-8).clamp(min=min_d, max=max_d)
            end_root = base_root + axis * d_clamped
            rest_axis = tensors["ik_rest_axis"][limb_i].to(dtype=root_rot.dtype)
            rest_pole = tensors["ik_rest_pole"][limb_i].to(dtype=root_rot.dtype)
            natural_pole = project_to_plane(swing_only_transport(rest_axis, axis, rest_pole), axis)
            pole_root = rotate_around_axis(natural_pole, axis, pole_float * IK_POLE_ALPHA)
            mid_root = solve_two_bone_with_pole(
                base_root,
                end_root,
                l1,
                l2,
                rest_axis,
                rest_pole,
                pole_float,
            )
            solved_mid = root_relative_to_world(mid_root.unsqueeze(1), root_pos, root_rot).squeeze(1)
            solved_end = root_relative_to_world(end_root.unsqueeze(1), root_pos, root_rot).squeeze(1)
            end_rot_world = root_relative_rot_to_world(end_rot_root, root_rot)
            world_pole = torch.matmul(pole_root.unsqueeze(1), root_rot).squeeze(1)

            global_pos_list[mid] = solved_mid
            global_pos_list[end] = solved_end
            local_pole_axis = tensors["ik_local_pole_axis"][limb_i].to(dtype=root_rot.dtype)
            mid_rot_world = rotation_from_axis_and_pole(
                tensors["local_offsets"][end],
                solved_end - solved_mid,
                local_pole_axis[1],
                world_pole,
            )
            start_rot_world = rotation_from_axis_and_pole(
                tensors["local_offsets"][mid],
                solved_mid - base,
                local_pole_axis[0],
                world_pole,
            )
            global_rot_list[start] = start_rot_world
            global_rot_list[mid] = mid_rot_world
            global_rot_list[end] = end_rot_world
            changed_bones.update((start, mid, end))
            toe = spec.get("toe")
            if toe is not None:
                toe_i = int(toe)
                toe_float = payload[:, cursor]
                cursor += 1
                toe_offset = tensors["ik_toe_offsets"][limb_i].to(dtype=root_rot.dtype).reshape(1, 3).expand(b, 3)
                toe_pos_root = end_root + torch.matmul(toe_offset.unsqueeze(1), end_rot_root).squeeze(1)
                toe_axis = tensors["ik_toe_axis"][limb_i].to(dtype=root_rot.dtype).reshape(1, 3).expand(b, 3)
                toe_hinge = axis_angle_to_row_matrix(toe_axis, toe_float * IK_TOE_ALPHA)
                toe_rot_root = toe_hinge @ end_rot_root
                global_pos_list[toe_i] = root_relative_to_world(toe_pos_root.unsqueeze(1), root_pos, root_rot).squeeze(1)
                global_rot_list[toe_i] = root_relative_rot_to_world(toe_rot_root, root_rot)
                changed_bones.add(toe_i)

        dirty_bones = set(changed_bones)
        for j in range(clip.J):
            parent = clip.parents_body_list[j]
            if parent < 0 or parent not in dirty_bones or j in clip.ik_controlled_set:
                continue
            if j == clip.pelvis:
                local_rot_j = pelvis_rot
            elif j in clip.core_nonpelvis_map:
                local_rot_j = core_rot[:, clip.core_nonpelvis_map[j]]
            else:
                local_rot_j = identity
            parent_rot = global_rot_list[parent]
            parent_pos = global_pos_list[parent]
            global_rot_list[j] = local_rot_j @ parent_rot
            global_pos_list[j] = torch.matmul(offsets[:, j].unsqueeze(1), parent_rot).squeeze(1) + parent_pos
            dirty_bones.add(j)

        global_pos = torch.stack(global_pos_list, dim=1)
        global_rot = torch.stack(global_rot_list, dim=1)
        root_yaw = heading_yaw_from_root(root_rot)
        heading = yaw_to_row_matrix(root_yaw)
        canon = torch.einsum("bjc,bcd->bjd", global_pos - root_pos[:, None, :], heading)
        return global_pos, global_rot, canon

    nonpelvis_rot = rotation_6d_to_matrix(pose["nonpelvis_rot6"])

    offsets = tensors["local_offsets"].unsqueeze(0).expand(b, -1, -1).clone()
    offsets[:, clip.pelvis] = pose["pelvis_pos"]

    global_pos_list: list[torch.Tensor] = []
    global_rot_list: list[torch.Tensor] = []
    for j in range(clip.J):
        local_rot_j = pelvis_rot if j == clip.pelvis else nonpelvis_rot[:, clip.nonpelvis_map[j]]
        parent = clip.parents_body_list[j]
        if parent < 0:
            rot_j = local_rot_j @ root_rot
            pos_j = torch.matmul(offsets[:, j].unsqueeze(1), root_rot).squeeze(1) + root_pos
        else:
            parent_rot = global_rot_list[parent]
            parent_pos = global_pos_list[parent]
            rot_j = local_rot_j @ parent_rot
            pos_j = torch.matmul(offsets[:, j].unsqueeze(1), parent_rot).squeeze(1) + parent_pos
        global_rot_list.append(rot_j)
        global_pos_list.append(pos_j)

    global_pos = torch.stack(global_pos_list, dim=1)
    global_rot = torch.stack(global_rot_list, dim=1)
    root_yaw = heading_yaw_from_root(root_rot)
    heading = yaw_to_row_matrix(root_yaw)
    canon = torch.einsum("bjc,bcd->bjd", global_pos - root_pos[:, None, :], heading)
    return global_pos, global_rot, canon


def clone_pose(pose: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in pose.items()}


def apply_initial_body_height_offset(
    clip: MotionClip,
    pose: dict[str, torch.Tensor],
    idx: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    offset = float(cfg.freefall_body_height_offset_m)
    if offset == 0.0:
        return pose
    out = clone_pose(pose)
    idx_device = idx.to(device)
    root_pos, root_rot, _yaw, _heading = root_state(clip, idx, cfg, device)
    world_delta = torch.zeros((idx_device.shape[0], 3), dtype=out["pelvis_pos"].dtype, device=device)
    world_delta[:, 1] = offset
    local_delta = torch.matmul(world_delta.unsqueeze(1), root_rot.transpose(-1, -2)).squeeze(1)
    out["pelvis_pos"] = out["pelvis_pos"] + local_delta
    _global_pos, _global_rot, canon = fk_from_pose(clip, root_pos, root_rot, out, device)
    out["canon_pos"] = canon
    return out


def maybe_apply_initial_offsets(
    clip: MotionClip,
    prev_idx: torch.Tensor,
    cur_idx: torch.Tensor,
    prev_pose: dict[str, torch.Tensor],
    cur_pose: dict[str, torch.Tensor],
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if cfg.freefall_body_height_offset_m == 0.0:
        return prev_pose, cur_pose
    history = max(1, int(cfg.freefall_initial_offset_history))
    if history >= 2:
        prev_pose = apply_initial_body_height_offset(clip, prev_pose, prev_idx, cfg, device)
    cur_pose = apply_initial_body_height_offset(clip, cur_pose, cur_idx, cfg, device)
    return prev_pose, cur_pose


def logical_pose_index(clip: MotionClip, idx: torch.Tensor, device: torch.device) -> torch.Tensor:
    idx = idx.to(device)
    if not clip.cyclic_animation:
        return idx
    return torch.remainder(idx, clip.cyclic_period)


def clip_rollout_max_start(clip: MotionClip, rollout_k: int, cfg: TrainConfig | None = None) -> int:
    if clip.cyclic_animation:
        return int(clip.cyclic_period) - 1
    if cfg is not None:
        return int(clip.T) - int(cfg.future_window) - int(rollout_k)
    return int(clip.T) - int(rollout_k) - 1


def clip_can_rollout(clip: MotionClip, rollout_k: int, cfg: TrainConfig | None = None) -> bool:
    return clip_rollout_max_start(clip, rollout_k, cfg) >= 1


def root_state(
    clip: MotionClip,
    idx: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = clip.tensors(device)
    idx = idx.to(device)
    if not clip.cyclic_animation:
        pos = tensors["root_pos"].index_select(0, idx)
        rot = tensors["root_rot"].index_select(0, idx)
        yaw = tensors["root_yaw"].index_select(0, idx)
        heading = tensors["root_heading_rot"].index_select(0, idx)
        return pos, rot, yaw, heading

    period = int(clip.cyclic_period)
    base_idx = torch.remainder(idx, period)
    cycles = torch.div(idx, period, rounding_mode="floor")
    root_pos = tensors["root_pos"]
    root_rot = tensors["root_rot"]
    base_pos = root_pos.index_select(0, base_idx)
    base_rot = root_rot.index_select(0, base_idx)

    root0_pos = root_pos[0]
    root0_rot = root_rot[0]
    end_pos = root_pos[period]
    end_rot = root_rot[period]
    root0_inv = root0_rot.transpose(-1, -2)
    base_rel_pos = torch.matmul((base_pos - root0_pos).unsqueeze(1), root0_inv).squeeze(1)
    base_rel_rot = base_rot @ root0_inv
    cycle_pos = torch.matmul((end_pos - root0_pos).unsqueeze(0), root0_inv).squeeze(0)
    cycle_rot = end_rot @ root0_inv

    rel_pos = base_rel_pos
    rel_rot = base_rel_rot
    max_cycles = int(cycles.max().detach().cpu()) if cycles.numel() else 0
    for cycle in range(max_cycles):
        mask = (cycles > cycle).reshape(-1, 1)
        next_pos = torch.matmul(rel_pos.unsqueeze(1), cycle_rot).squeeze(1) + cycle_pos
        next_rot = rel_rot @ cycle_rot
        rel_pos = torch.where(mask, next_pos, rel_pos)
        rel_rot = torch.where(mask.unsqueeze(-1), next_rot, rel_rot)

    pos = torch.matmul(rel_pos.unsqueeze(1), root0_rot).squeeze(1) + root0_pos
    rot = rel_rot @ root0_rot
    yaw = heading_yaw_from_root(rot)
    heading = yaw_to_row_matrix(yaw)
    return pos, rot, yaw, heading


def global_from_clip(
    clip: MotionClip,
    idx: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not clip.cyclic_animation:
        tensors = clip.tensors(device)
        idx_device = idx.to(device)
        return (
            tensors["global_pos"].index_select(0, idx_device),
            tensors["global_rot"].index_select(0, idx_device),
        )
    pose = get_pose_from_clip(clip, idx, device)
    root_pos, root_rot, _yaw, _heading = root_state(clip, idx, cfg, device)
    global_pos, global_rot, _canon = fk_from_pose(clip, root_pos, root_rot, pose, device)
    return global_pos, global_rot


def root_delta_feature(clip: MotionClip, prev_idx: torch.Tensor, cur_idx: torch.Tensor, cfg: TrainConfig, device) -> torch.Tensor:
    prev_pos, _prev_rot, prev_yaw, prev_heading = root_state(clip, prev_idx, cfg, device)
    cur_pos, _cur_rot, cur_yaw, _cur_heading = root_state(clip, cur_idx, cfg, device)
    delta_local = torch.matmul((cur_pos - prev_pos).unsqueeze(1), prev_heading).squeeze(1)
    dx = delta_local[:, 0] / cfg.max_speed_scale_final
    dz = delta_local[:, 2] / cfg.max_speed_scale_final
    yaw_delta = wrap_angle(cur_yaw - prev_yaw)
    dyaw = yaw_delta / cfg.max_turn_rate_scale_final
    return torch.stack((dx, dz, dyaw), dim=-1)


def future_root_features(clip: MotionClip, cur_idx: torch.Tensor, cfg: TrainConfig, device) -> torch.Tensor:
    feats = []
    cur_idx = cur_idx.to(device)
    cur_pos, _cur_rot, cur_yaw, cur_heading = root_state(clip, cur_idx, cfg, device)
    for k in range(1, cfg.future_window + 1):
        fut_idx = cur_idx + k if clip.cyclic_animation else torch.clamp(cur_idx + k, max=clip.T - 1)
        fut_pos, _fut_rot, fut_yaw, _fut_heading = root_state(clip, fut_idx, cfg, device)
        fut_local = torch.matmul((fut_pos - cur_pos).unsqueeze(1), cur_heading).squeeze(1)
        scale_k = k * cfg.max_speed_scale_final
        dx = torch.clamp(fut_local[:, 0] / scale_k, -2.0, 2.0)
        dz = torch.clamp(fut_local[:, 2] / scale_k, -2.0, 2.0)
        dyaw = wrap_angle(fut_yaw - cur_yaw)
        feats.append(torch.stack((dx, dz, torch.cos(dyaw), torch.sin(dyaw)), dim=-1))
    return torch.cat(feats, dim=-1)


def build_input(
    clip: MotionClip,
    prev_idx: torch.Tensor,
    cur_idx: torch.Tensor,
    prev_pose: dict[str, torch.Tensor],
    cur_pose: dict[str, torch.Tensor],
    cfg: TrainConfig,
    device: torch.device,
) -> torch.Tensor:
    current = body_pose_vector(cur_pose)
    previous = body_pose_vector(prev_pose)
    pelvis_vel = (cur_pose["pelvis_pos"] - prev_pose["pelvis_pos"]) / cfg.pose_delta_scale_final
    if "ik_payload" in cur_pose:
        joint_vel = (
            cur_pose["ik_payload"] - prev_pose["ik_payload"]
        ).reshape(cur_idx.shape[0], -1) / cfg.pose_delta_scale_final
    else:
        joint_vel = (cur_pose["canon_pos"] - prev_pose["canon_pos"]).reshape(cur_idx.shape[0], -1) / cfg.pose_delta_scale_final
    root_feat = root_delta_feature(clip, prev_idx, cur_idx, cfg, device)
    future_feat = future_root_features(clip, cur_idx, cfg, device)
    return torch.cat((current, previous, pelvis_vel, joint_vel, root_feat, future_feat), dim=-1)


def get_pose_from_clip(clip: MotionClip, idx: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    tensors = clip.tensors(device)
    idx = logical_pose_index(clip, idx, device)
    pose = {
        "pelvis_pos": tensors["pelvis_local_pos"].index_select(0, idx),
        "pelvis_rot_mat": tensors["pelvis_rot_mat"].index_select(0, idx),
        "nonpelvis_rot_mat": tensors["non_pelvis_rot_mat"].index_select(0, idx),
        "pelvis_rot6": tensors["pelvis_rot6"].index_select(0, idx),
        "nonpelvis_rot6": tensors["non_pelvis_rot6"].index_select(0, idx),
        "canon_pos": tensors["canonical_pos"].index_select(0, idx),
    }
    if uses_ik_markers(clip.pose_representation):
        pose["core_nonpelvis_rot6"] = tensors["core_non_pelvis_rot6"].index_select(0, idx)
        pose["ik_payload"] = tensors["ik_payload"].index_select(0, idx)
    return pose


def blend_pose_by_mask(
    primary: dict[str, torch.Tensor],
    replacement: dict[str, torch.Tensor],
    use_replacement: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        key: torch.where(
            use_replacement.reshape((use_replacement.shape[0],) + (1,) * (value.ndim - 1)),
            replacement[key],
            value,
        )
        for key, value in primary.items()
        if key in replacement
    }


def compute_losses(
    clip: MotionClip,
    prev_pose: dict[str, torch.Tensor],
    cur_pose: dict[str, torch.Tensor],
    pred_pose: dict[str, torch.Tensor],
    raw_pose: dict[str, torch.Tensor],
    target_pose: dict[str, torch.Tensor],
    prev_idx: torch.Tensor,
    cur_idx: torch.Tensor,
    target_idx: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    b = target_idx.shape[0]
    tensors = clip.tensors(device)
    root_pos, root_rot, _target_yaw, _target_heading = root_state(clip, target_idx, cfg, device)
    pred_global_pos, pred_global_rot, pred_canon = fk_from_pose(clip, root_pos, root_rot, pred_pose, device)

    prev_root_pos, prev_root_rot, _prev_yaw, _prev_heading = root_state(clip, prev_idx, cfg, device)
    cur_root_pos, cur_root_rot, _cur_yaw, _cur_heading = root_state(clip, cur_idx, cfg, device)
    prev_global_pos, _prev_global_rot, _prev_canon = fk_from_pose(clip, prev_root_pos, prev_root_rot, prev_pose, device)
    cur_global_pos, cur_global_rot, _cur_canon = fk_from_pose(clip, cur_root_pos, cur_root_rot, cur_pose, device)

    target_global_pos, target_global_rot = global_from_clip(clip, target_idx, cfg, device)

    pelvis_loc = weighted_batch_mean(
        F.huber_loss(pred_pose["pelvis_pos"], target_pose["pelvis_pos"], reduction="none"), sample_weight
    )
    pred_pelvis_rot = rotation_6d_to_matrix(pred_pose["pelvis_rot6"])
    target_pelvis_rot = target_pose.get("pelvis_rot_mat")
    if target_pelvis_rot is None:
        target_pelvis_rot = rotation_6d_to_matrix(target_pose["pelvis_rot6"])
    pelvis_rot = weighted_batch_mean(
        geodesic_angles(pred_pelvis_rot, target_pelvis_rot).unsqueeze(-1),
        sample_weight,
    )
    zero_loss = pred_pose["pelvis_pos"].new_zeros(())
    marker_loc = zero_loss
    if "ik_payload" in pred_pose:
        if clip.Jcore > 0:
            pred_core_rot = rotation_6d_to_matrix(pred_pose["core_nonpelvis_rot6"].reshape(-1, 6))
            target_core_rot = rotation_6d_to_matrix(target_pose["core_nonpelvis_rot6"].reshape(-1, 6))
            pose_rot_angles = geodesic_angles(
                pred_core_rot,
                target_core_rot,
            ).reshape(b, clip.Jcore)
            pose_rot = weighted_batch_mean(pose_rot_angles, sample_weight)
            pose_aux = weighted_batch_mean(
                (pred_pose["core_nonpelvis_rot6"] - target_pose["core_nonpelvis_rot6"]).square(),
                sample_weight,
            )
        else:
            pose_rot = zero_loss
            pose_aux = zero_loss
        marker_loc = weighted_batch_mean(
            (pred_pose["ik_payload"] - target_pose["ik_payload"]).square(),
            sample_weight,
        )
        pose_aux = (
            pose_aux
            + weighted_batch_mean((pred_pose["pelvis_rot6"] - target_pose["pelvis_rot6"]).square(), sample_weight)
            + marker_loc
        )
    else:
        pred_nonpelvis_rot = rotation_6d_to_matrix(pred_pose["nonpelvis_rot6"].reshape(-1, 6))
        target_nonpelvis_rot = target_pose.get("nonpelvis_rot_mat")
        if target_nonpelvis_rot is None:
            target_nonpelvis_rot = rotation_6d_to_matrix(target_pose["nonpelvis_rot6"].reshape(-1, 6))
        else:
            target_nonpelvis_rot = target_nonpelvis_rot.reshape(-1, 3, 3)
        pose_rot_angles = geodesic_angles(
            pred_nonpelvis_rot,
            target_nonpelvis_rot,
        ).reshape(b, clip.Jn)
        pose_rot = weighted_batch_mean(pose_rot_angles, sample_weight)
        pose_aux = weighted_batch_mean(
            (pred_pose["nonpelvis_rot6"] - target_pose["nonpelvis_rot6"]).square(), sample_weight
        ) + weighted_batch_mean(
            (pred_pose["pelvis_rot6"] - target_pose["pelvis_rot6"]).square(), sample_weight
        )
    ee_idx = tensors["end_effectors"]
    ee_delta = pred_global_pos.index_select(1, ee_idx) - target_global_pos.index_select(1, ee_idx)
    ee_loc = weighted_batch_mean(ee_delta.square().sum(dim=-1), sample_weight)
    ee_rot = weighted_batch_mean(
        geodesic_angles(
        pred_global_rot.index_select(1, ee_idx).reshape(-1, 3, 3),
        target_global_rot.index_select(1, ee_idx).reshape(-1, 3, 3),
        ).reshape(b, -1),
        sample_weight,
    )
    full_body_loc = weighted_batch_mean(pred_global_pos.sub(target_global_pos).square().sum(dim=-1), sample_weight)
    zero = pred_pose["pelvis_pos"].new_zeros(())
    contact_prob = pred_pose["pelvis_pos"].new_zeros((b, 2))
    contact_label = zero
    foot_penetration = zero
    foot_sliding = zero
    contact_height = zero
    freefall = zero
    termination = zero
    term_mask = torch.zeros((b,), dtype=torch.bool, device=device)
    term_severity = torch.zeros((b,), dtype=pred_pose["pelvis_pos"].dtype, device=device)
    foot_heights = torch.zeros((b, 2), dtype=pred_pose["pelvis_pos"].dtype, device=device)
    foot_speeds = torch.zeros_like(foot_heights)
    foot_horizontal_speeds = torch.zeros_like(foot_heights)
    freefall_rel = torch.zeros((b,), dtype=pred_pose["pelvis_pos"].dtype, device=device)

    if cfg.enable_contact_physics_losses:
        raise NotImplementedError("Contact losses were removed from the IK harness.")

    need_foot_geometry = (
        cfg.enable_early_termination
        or cfg.alpha12_termination != 0.0
        or (
            cfg.enable_contact_physics_losses
            and (
                cfg.alpha8_foot_penetration != 0.0
                or cfg.alpha9_foot_sliding != 0.0
                or cfg.alpha11_contact_height != 0.0
            )
        )
    )
    need_freefall = cfg.alpha10_freefall != 0.0 or cfg.enable_freefall_termination
    geom = cp.DEFAULT_GEOMETRY
    if need_foot_geometry:
        geom = cp.DEFAULT_GEOMETRY
        foot_indices = tuple(clip.foot_indices)
        toe_indices = tuple(clip.toe_indices)
        foot_heights, _lowest_points = cp.foot_lowest_heights_and_points(
            pred_global_pos, pred_global_rot, foot_indices, toe_indices, geom
        )
        foot_speeds = cp.foot_contact_point_speeds(
            cur_global_pos, cur_global_rot, pred_global_pos, pred_global_rot, foot_indices, toe_indices, clip.fps, geom
        )
        foot_horizontal_speeds = cp.foot_slide_speeds(
            cur_global_pos, cur_global_rot, pred_global_pos, pred_global_rot, foot_indices, toe_indices, clip.fps, geom
        )
        foot_penetration = weighted_batch_mean(F.relu(-foot_heights).square(), sample_weight)
        foot_sliding = weighted_batch_mean(contact_prob * F.relu(foot_speeds - geom.speed_threshold_mps).square(), sample_weight)
        contact_height = weighted_batch_mean(contact_prob * F.relu(foot_heights - geom.height_threshold_m).square(), sample_weight)

    if need_freefall:
        prev_com = cp.center_of_mass(prev_global_pos, tensors["mass_weights"])
        cur_com = cp.center_of_mass(cur_global_pos, tensors["mass_weights"])
        pred_com = cp.center_of_mass(pred_global_pos, tensors["mass_weights"])
        no_contact_prob = (1.0 - contact_prob[:, 0]) * (1.0 - contact_prob[:, 1])
        _freefall_unweighted, freefall_rel = cp.freefall_loss(prev_com, cur_com, pred_com, no_contact_prob, clip.fps, geom)
        dt = 1.0 / float(clip.fps)
        expected_y = cur_com[:, 1] + (cur_com[:, 1] - prev_com[:, 1]) - 0.5 * geom.gravity_mps2 * dt * dt
        freefall = weighted_batch_mean((no_contact_prob * (pred_com[:, 1] - expected_y).square()).unsqueeze(-1), sample_weight)

    if need_foot_geometry or cfg.enable_early_termination or cfg.alpha12_termination != 0.0:
        term_mask = cp.termination_mask(
            foot_heights, foot_speeds, contact_prob, freefall_rel, geom, cfg.enable_freefall_termination
        )
        term_severity = cp.termination_severity(
            foot_heights, foot_speeds, contact_prob, freefall_rel, geom, cfg.enable_freefall_termination
        )
        termination = weighted_batch_mean(term_severity.unsqueeze(-1), sample_weight) * float(clip.T)
    total = (
        cfg.alpha0_pelvis_location * pelvis_loc
        + cfg.alpha1_pelvis_rotation * pelvis_rot
        + cfg.alpha2_pose_rotation * pose_rot
        + cfg.alpha3_pose_6d_aux * pose_aux
        + cfg.alpha4_end_effector_location * ee_loc
        + cfg.alpha5_end_effector_rotation * ee_rot
        + cfg.alpha6_full_body_location * full_body_loc
        + cfg.alpha4_end_effector_location * marker_loc
    )
    if cfg.enable_contact_physics_losses:
        total = (
            total
            + cfg.alpha7_contact_label * contact_label
            + cfg.alpha8_foot_penetration * foot_penetration
            + cfg.alpha9_foot_sliding * foot_sliding
            + cfg.alpha10_freefall * freefall
            + cfg.alpha11_contact_height * contact_height
            + cfg.alpha12_termination * termination
        )
    losses = {
        "pelvis_location": pelvis_loc.detach(),
        "pelvis_rotation": pelvis_rot.detach(),
        "pose_rotation": pose_rot.detach(),
        "pose_6d_aux": pose_aux.detach(),
        "ik_marker_location": marker_loc.detach(),
        "end_effector_location": ee_loc.detach(),
        "end_effector_rotation": ee_rot.detach(),
        "full_body_location": full_body_loc.detach(),
        "contact_label": contact_label.detach(),
        "foot_penetration": foot_penetration.detach(),
        "foot_sliding": foot_sliding.detach(),
        "contact_height": contact_height.detach(),
        "freefall": freefall.detach(),
        "termination": termination.detach(),
        "termination_rate": term_mask.float().mean().detach(),
        "termination_severity": term_severity.detach().mean(),
        "contact_prob_mean": contact_prob.detach().mean(),
        "foot_height_min": foot_heights.detach().amin(),
        "foot_speed_mean": foot_speeds.detach().mean(),
        "foot_horizontal_speed_mean": foot_horizontal_speeds.detach().mean(),
        "freefall_relative_error": freefall_rel.detach().mean(),
    }
    next_pose = next_pose_from_prediction(pred_pose, pred_canon)
    return total, losses, next_pose, term_mask


def run_batch(
    model: nn.Module,
    clips: list[MotionClip],
    batch: list[torch.Tensor],
    cfg: TrainConfig,
    rollout_k: int,
    device: torch.device,
    train: bool,
    live_bridge: LiveTrainingBridge | None = None,
    epoch: int = 0,
    phase: str = "train",
) -> tuple[torch.Tensor, dict[str, float]]:
    clip_indices, starts = batch
    # Group by clip so variable skeleton metadata remains simple and explicit.
    total_loss = torch.zeros((), device=device)
    accum: dict[str, torch.Tensor] = {}
    live_rows = set(range(min(live_bridge.max_agents, int(clip_indices.shape[0])))) if live_bridge is not None else set()
    live_sequences: dict[int, dict[str, list[np.ndarray]]] = {}
    live_clip: MotionClip | None = None

    def append_live_frame(
        clip: MotionClip,
        rows: list[int],
        local_rows: list[int],
        frame_idx: torch.Tensor,
        pose: dict[str, torch.Tensor],
    ) -> None:
        nonlocal live_clip
        if live_bridge is None or not local_rows:
            return
        if live_clip is None:
            live_clip = clip
        if live_clip is not clip:
            return
        with torch.no_grad():
            local_t = torch.tensor(local_rows, dtype=torch.long, device=device)
            idx_sel = frame_idx.index_select(0, local_t)
            pose_sel = index_pose(pose, local_t)
            root_pos, root_rot, _yaw, _heading = root_state(clip, idx_sel, cfg, device)
            pred_pos, pred_rot, _canon = fk_from_pose(clip, root_pos, root_rot, pose_sel, device)
            gt_pos, gt_rot = global_from_clip(clip, idx_sel, cfg, device)
            pred_pos_np = pred_pos.detach().cpu().numpy()
            pred_rot_np = pred_rot.detach().cpu().numpy()
            gt_pos_np = gt_pos.detach().cpu().numpy()
            gt_rot_np = gt_rot.detach().cpu().numpy()
        for i, row in enumerate(rows):
            seq = live_sequences.setdefault(row, {"pred_pos": [], "pred_rot": [], "gt_pos": [], "gt_rot": []})
            seq["pred_pos"].append(pred_pos_np[i])
            seq["pred_rot"].append(pred_rot_np[i])
            seq["gt_pos"].append(gt_pos_np[i])
            seq["gt_rot"].append(gt_rot_np[i])

    groups = {}
    for row, ci in enumerate(clip_indices.tolist()):
        groups.setdefault(ci, []).append(row)

    group_count = 0
    for ci, rows in groups.items():
        clip = clips[ci]
        row_t = torch.tensor(rows, dtype=torch.long)
        start = starts[row_t].long().to(device)
        prev_idx = start - 1
        cur_idx = start
        prev_pose = get_pose_from_clip(clip, prev_idx, device)
        cur_pose = get_pose_from_clip(clip, cur_idx, device)
        prev_pose, cur_pose = maybe_apply_initial_offsets(
            clip, prev_idx, cur_idx, prev_pose, cur_pose, cfg, device
        )
        alive = torch.ones(start.shape[0], device=device)
        live_local_rows = [local_i for local_i, row in enumerate(rows) if row in live_rows]
        live_global_rows = [rows[local_i] for local_i in live_local_rows]
        append_live_frame(clip, live_global_rows, live_local_rows, cur_idx, cur_pose)

        group_loss = torch.zeros((), device=device)
        for step in range(rollout_k):
            if cfg.enable_early_termination and not cfg.restart_on_termination and bool((alive <= 0.0).all()):
                break
            inp = build_input(clip, prev_idx, cur_idx, prev_pose, cur_pose, cfg, device)
            raw_out = predict_next_raw(model, inp, cur_pose, cfg)
            pred_pose, raw_pose = output_to_pose(raw_out, clip)
            target_idx = cur_idx + 1
            target_pose = get_pose_from_clip(clip, target_idx, device)
            step_loss, parts, next_pose, term_mask = compute_losses(
                clip,
                prev_pose,
                cur_pose,
                pred_pose,
                raw_pose,
                target_pose,
                prev_idx,
                cur_idx,
                target_idx,
                cfg,
                device,
                alive if cfg.enable_early_termination else None,
            )
            append_live_frame(clip, live_global_rows, live_local_rows, target_idx, next_pose)
            group_loss = group_loss + step_loss / rollout_k
            for key, value in parts.items():
                accum[key] = accum.get(key, torch.zeros((), device=device)) + value / rollout_k
            if cfg.enable_early_termination:
                dead = term_mask.detach()
                if cfg.restart_on_termination and step + 1 < rollout_k and bool(dead.any()):
                    remaining_steps = rollout_k - step - 1
                    max_start = clip_rollout_max_start(clip, remaining_steps, cfg)
                    if max_start < 1:
                        dead = torch.zeros_like(dead)
                        alive = alive & (~dead)
                        continue
                    restart_start = torch.randint(1, max_start + 1, cur_idx.shape, device=device)
                    restart_prev_idx = restart_start - 1
                    restart_cur_idx = restart_start
                    restart_prev_pose = get_pose_from_clip(clip, restart_prev_idx, device)
                    restart_cur_pose = get_pose_from_clip(clip, restart_cur_idx, device)
                    restart_prev_pose, restart_cur_pose = maybe_apply_initial_offsets(
                        clip,
                        restart_prev_idx,
                        restart_cur_idx,
                        restart_prev_pose,
                        restart_cur_pose,
                        cfg,
                        device,
                    )
                    prev_pose = blend_pose_by_mask(
                        cur_pose,
                        restart_prev_pose,
                        dead,
                    )
                    cur_pose = blend_pose_by_mask(
                        next_pose,
                        restart_cur_pose,
                        dead,
                    )
                    prev_idx = torch.where(dead, restart_prev_idx, cur_idx)
                    cur_idx = torch.where(dead, restart_cur_idx, target_idx)
                    alive = torch.ones_like(alive)
                    continue
                alive = alive * (~dead).to(device=device, dtype=alive.dtype)
            prev_pose = cur_pose
            cur_pose = next_pose
            prev_idx = cur_idx
            cur_idx = target_idx

        total_loss = total_loss + group_loss
        group_count += 1

    total_loss = total_loss / max(1, group_count)
    scalars = {k: float(v.detach().cpu() / max(1, group_count)) for k, v in accum.items()}
    scalars["total"] = float(total_loss.detach().cpu())
    if live_bridge is not None and live_clip is not None and live_sequences:
        ordered_rows = sorted(live_sequences)[: live_bridge.max_agents]
        frame_count = min(len(live_sequences[row]["pred_pos"]) for row in ordered_rows)
        if frame_count > 0:
            live_bridge.write_snapshot(
                clip=live_clip,
                epoch=epoch,
                rollout_k=rollout_k,
                phase=phase,
                train_total=scalars["total"],
                pred_pos=np.stack([
                    np.stack(live_sequences[row]["pred_pos"][:frame_count], axis=0) for row in ordered_rows
                ], axis=0),
                pred_rot=np.stack([
                    np.stack(live_sequences[row]["pred_rot"][:frame_count], axis=0) for row in ordered_rows
                ], axis=0),
                gt_pos=np.stack([
                    np.stack(live_sequences[row]["gt_pos"][:frame_count], axis=0) for row in ordered_rows
                ], axis=0),
                gt_rot=np.stack([
                    np.stack(live_sequences[row]["gt_rot"][:frame_count], axis=0) for row in ordered_rows
                ], axis=0),
            )
    return total_loss, scalars


def parse_path_list(text: str | None) -> list[Path]:
    if text is None:
        return []
    return [resolve_path(part.strip()) for part in str(text).split(";") if part.strip()]


def npz_folder_from_path(folder: Path) -> Path:
    folder = resolve_path(folder)
    if any(folder.glob("*.npz")):
        return folder
    final_folder = folder / "npz_final"
    if final_folder.exists() and any(final_folder.glob("*.npz")):
        return final_folder
    return folder


def load_clips_from_specs(specs: list[tuple[Path, bool | None]], cfg: TrainConfig) -> list[MotionClip]:
    paths_with_flags: list[tuple[Path, bool | None]] = []
    for folder, cyclic in specs:
        npz_folder = npz_folder_from_path(folder)
        paths = sorted(npz_folder.glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"No .npz files found in {npz_folder}")
        paths_with_flags.extend((path, cyclic) for path in paths)
    if not paths_with_flags:
        raise FileNotFoundError("No .npz files found in requested motion folders")
    clips = [MotionClip(path, cfg, cyclic_animation=cyclic) for path, cyclic in paths_with_flags]
    first_names = clips[0].body_names
    first_parents = clips[0].parents_body_list
    for clip in clips[1:]:
        if clip.body_names != first_names or clip.parents_body_list != first_parents:
            raise ValueError(
                f"Skeleton mismatch: {clip.path} does not match {clips[0].path}; "
                "all NPZs in one IK run must share bone names and parent topology"
            )
    return clips


def load_clips(folder: Path, cfg: TrainConfig) -> list[MotionClip]:
    return load_clips_from_specs([(folder, None)], cfg)


def clip_specs_from_folders(
    folder_path: str | Path | None,
    periodic_folder_path: str | None,
    nonperiodic_folder_path: str | None,
) -> list[tuple[Path, bool | None]]:
    specs: list[tuple[Path, bool | None]] = []
    for folder in parse_path_list(periodic_folder_path):
        specs.append((folder, True))
    for folder in parse_path_list(nonperiodic_folder_path):
        specs.append((folder, False))
    if specs:
        return specs
    if folder_path is None:
        return [(resolve_path(folder_path or "data/npz_final"), None)]
    return [(resolve_path(folder_path), None)]


def make_batch_dims(clip: MotionClip, cfg: TrainConfig) -> tuple[int, int]:
    if uses_ik_markers(cfg.pose_representation):
        pose_dim = 3 + 6 + clip.Jcore * 6 + clip.ik_payload_dim
        velocity_dim = 3 + clip.ik_payload_dim
        output_dim = 3 + 6 + clip.Jcore * 6 + clip.ik_payload_dim
    else:
        pose_dim = 3 + 6 + clip.J * 3 + clip.Jn * 6
        velocity_dim = 3 + clip.J * 3
        output_dim = 3 + 6 + clip.Jn * 6
    input_dim = pose_dim * 2 + velocity_dim + 3 + cfg.future_window * 4
    return input_dim, output_dim


def unwrap_compiled_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def apply_cuda_performance_settings(cfg: TrainConfig, device: torch.device) -> None:
    if cfg.allow_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def maybe_compile_model(
    model: nn.Module,
    input_dim: int,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[nn.Module, bool]:
    if not cfg.use_torch_compile:
        return model, False
    if not hasattr(torch, "compile"):
        print("torch.compile disabled: this PyTorch build does not expose torch.compile")
        return model, False
    try:
        compiled_model = torch.compile(model, mode=cfg.torch_compile_mode)
        probe = torch.zeros(max(1, int(cfg.batch_size)), input_dim, device=device)
        probe.requires_grad_(True)
        out = compiled_model(probe)
        out.square().mean().backward()
        compiled_model.zero_grad(set_to_none=True)
        return compiled_model, True
    except Exception as exc:
        print(f"torch.compile disabled after forward/backward probe: {exc}")
        return model, False


def parse_rollout_schedule(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("rollout schedule cannot be empty")
    if any(value < 1 for value in values):
        raise ValueError("rollout schedule values must be >= 1")
    return values


def learning_rate_for_epoch(cfg: TrainConfig, epoch: int, stage_epoch: int, rollout_idx: int) -> float:
    base_lr = float(cfg.learning_rate)
    if cfg.lr_warmup_epochs > 0 and epoch <= cfg.lr_warmup_epochs:
        return base_lr * max(1.0 / cfg.lr_warmup_epochs, epoch / cfg.lr_warmup_epochs)
    min_factor = max(0.0, min(1.0, float(cfg.lr_min_factor)))
    schedule = cfg.lr_schedule
    if schedule in ("constant", "adaptive_plateau"):
        factor = 1.0
    elif schedule == "stage_decay":
        factor = float(cfg.lr_stage_decay) ** rollout_idx
    elif schedule == "cosine":
        denom = max(1, cfg.max_epochs - max(0, cfg.lr_warmup_epochs))
        progress = min(1.0, max(0.0, (epoch - max(0, cfg.lr_warmup_epochs)) / denom))
        factor = min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))
    elif schedule == "stage_cosine":
        stage_len = max(1, int(cfg.curriculum_max_epochs_per_stage))
        progress = min(1.0, max(0.0, (stage_epoch - 1) / max(1, stage_len - 1)))
        cosine = min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))
        factor = (float(cfg.lr_stage_decay) ** rollout_idx) * cosine
    else:
        raise ValueError(f"unknown lr_schedule={schedule!r}")
    return max(base_lr * min_factor, base_lr * factor)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


@dataclass
class AdaptiveLrState:
    best: float = float("inf")
    bad_epochs: int = 0
    cooldown: int = 0


def reset_adaptive_lr(optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> AdaptiveLrState:
    set_optimizer_lr(optimizer, cfg.learning_rate)
    return AdaptiveLrState()


def step_adaptive_lr(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    state: AdaptiveLrState,
    metric: float,
) -> tuple[AdaptiveLrState, bool]:
    if not math.isfinite(metric):
        return state, False
    improved = state.best == float("inf") or metric < state.best * (1.0 - cfg.lr_plateau_threshold)
    if improved:
        state.best = metric
        state.bad_epochs = 0
        return state, False
    if state.cooldown > 0:
        state.cooldown -= 1
        return state, False
    state.bad_epochs += 1
    if state.bad_epochs < cfg.lr_plateau_patience_epochs:
        return state, False
    old_lr = float(optimizer.param_groups[0]["lr"])
    min_lr = float(cfg.learning_rate) * max(0.0, min(1.0, float(cfg.lr_min_factor)))
    new_lr = max(min_lr, old_lr * float(cfg.lr_plateau_factor))
    state.bad_epochs = 0
    state.cooldown = max(0, int(cfg.lr_plateau_cooldown_epochs))
    if new_lr >= old_lr - 1e-16:
        return state, False
    set_optimizer_lr(optimizer, new_lr)
    return state, True


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_val: float, rollout_k: int, cfg, metadata) -> None:
    path = ik_checkpoint_path(path, getattr(cfg, "run_name", None))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, optimizer, epoch, best_val, rollout_k, cfg, metadata), path)
    size_mb = path.stat().st_size / (1024.0 * 1024.0)
    print(
        f"new saved checkpoint at {path} epoch={epoch} K={rollout_k} "
        f"best_val={best_val:.8g} size_mb={size_mb:.2f}",
        flush=True,
    )


def checkpoint_payload(model, optimizer, epoch: int, best_val: float, rollout_k: int, cfg, metadata) -> dict:
    return {
        "epoch": epoch,
        "best_val": best_val,
        "rollout_k": rollout_k,
        "config": asdict(cfg),
        "metadata": metadata,
        "model": unwrap_compiled_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
    }


def clone_checkpoint_payload(payload: dict) -> dict:
    cloned = dict(payload)
    cloned["model"] = {k: v.detach().cpu().clone() for k, v in payload["model"].items()}
    cloned["optimizer"] = payload["optimizer"]
    return cloned


def save_payload(path: Path, payload: dict) -> None:
    config = payload.get("config", {})
    run_name = config.get("run_name") if isinstance(config, dict) else None
    path = ik_checkpoint_path(path, run_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    size_mb = path.stat().st_size / (1024.0 * 1024.0)
    epoch = payload.get("epoch", "?")
    rollout_k = payload.get("rollout_k", "?")
    best_val = payload.get("best_val", "?")
    if isinstance(best_val, float):
        best_text = f"{best_val:.8g}"
    else:
        best_text = str(best_val)
    print(
        f"new saved checkpoint at {path} epoch={epoch} K={rollout_k} "
        f"best_val={best_text} size_mb={size_mb:.2f}",
        flush=True,
    )


def update_model_comparison_html(clip_path: Path, ckpt_dir: Path, cfg: TrainConfig) -> None:
    checkpoint = ckpt_dir / "checkpoint_best.pt"
    if not checkpoint.exists():
        checkpoint = ckpt_dir / "checkpoint_last.pt"
    if not checkpoint.exists():
        print("model comparison refresh skipped: no checkpoint was written", flush=True)
        return

    output = resolve_path(cfg.comparison_output_path)
    script = PROJECT_ROOT / "training" / "visualize_model.py"
    cmd = [
        sys.executable,
        str(script),
        "--npz-path",
        str(clip_path),
        "--checkpoint-path",
        str(checkpoint),
        "--output-path",
        str(output),
        "--device",
        str(cfg.comparison_device),
    ]
    if cfg.comparison_max_frames > 0:
        cmd.extend(["--max-frames", str(cfg.comparison_max_frames)])
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"model comparison refresh failed: {exc}", flush=True)
        return
    if result.returncode != 0:
        print(f"model comparison refresh failed with exit code {result.returncode}", flush=True)
        if result.stderr.strip():
            print(result.stderr.strip(), flush=True)
        return
    print(
        f"model comparison refreshed output={output} checkpoint={checkpoint}",
        flush=True,
    )


class TimingProfiler:
    def __init__(self, enabled: bool, device: torch.device | None = None, sync_cuda: bool = False) -> None:
        self.enabled = enabled
        self.device = device
        self.sync_cuda = sync_cuda
        self.seconds: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def _sync(self) -> None:
        if (
            self.sync_cuda
            and self.device is not None
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize(self.device)

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.seconds[name] = self.seconds.get(name, 0.0) + (time.perf_counter() - start)
            self.counts[name] = self.counts.get(name, 0) + 1

    def write_csv(self, path: Path, total_seconds: float) -> None:
        if not self.enabled:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        known = sum(self.seconds.values())
        rows = [("section", "seconds", "percent", "count")]
        for key, seconds in sorted(self.seconds.items(), key=lambda item: item[1], reverse=True):
            percent = 0.0 if total_seconds <= 0.0 else seconds * 100.0 / total_seconds
            rows.append((key, f"{seconds:.6f}", f"{percent:.3f}", str(self.counts.get(key, 0))))
        overhead = max(0.0, total_seconds - known)
        rows.append(("unprofiled_overhead", f"{overhead:.6f}", f"{(overhead * 100.0 / total_seconds) if total_seconds > 0.0 else 0.0:.3f}", ""))
        rows.append(("total_wall", f"{total_seconds:.6f}", "100.000", ""))
        path.write_text("\n".join(",".join(row) for row in rows) + "\n", encoding="utf-8")


def replace_with_retry(tmp: Path, target: Path, attempts: int = 12, delay_seconds: float = 0.01) -> bool:
    for attempt in range(attempts):
        try:
            tmp.replace(target)
            return True
        except PermissionError:
            if attempt == attempts - 1:
                break
            time.sleep(delay_seconds)
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    return False


def write_json_atomic(path: Path, data: dict) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return replace_with_retry(tmp, path)


class LiveTrainingBridge:
    def __init__(self, run_dir: Path, cfg: TrainConfig) -> None:
        self.run_dir = run_dir
        self.cfg = cfg
        self.live_dir = run_dir / "live_training"
        self.control_path = self.live_dir / "control.json"
        self.snapshot_path = self.live_dir / "snapshot.npz"
        self.status_path = self.live_dir / "status.json"
        self.loss_history_path = self.live_dir / "loss_history.csv"
        self.process: subprocess.Popen | None = None
        self.visualize = bool(cfg.live_viewer_start_visualizing)
        self.stop_requested = False
        self.control_mtime = 0.0
        self.max_agents = max(1, int(cfg.live_viewer_max_agents))
        self.loss_history_file = None

    def start(self) -> None:
        self.live_dir.mkdir(parents=True, exist_ok=True)
        if not self.loss_history_path.exists():
            self.loss_history_path.write_text("epoch,elapsed_seconds,rollout_k,train_total\n", encoding="utf-8")
        self.loss_history_file = self.loss_history_path.open("a", encoding="utf-8", buffering=1)
        write_json_atomic(
            self.control_path,
            {
                "visualize": bool(self.cfg.live_viewer_start_visualizing),
                "show_ground_truth": True,
                "stop": False,
                "updated_at": time.time(),
            },
        )
        script = PROJECT_ROOT / "training" / "live_training_viewer.py"
        cmd = [sys.executable, str(script), "--run-dir", str(self.run_dir)]
        if self.cfg.live_viewer_start_visualizing:
            cmd.append("--start-visualizing")
        stdout = open(self.live_dir / "viewer_stdout.log", "a", encoding="utf-8")
        stderr = open(self.live_dir / "viewer_stderr.log", "a", encoding="utf-8")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=stdout,
                stderr=stderr,
                creationflags=creationflags,
            )
        except OSError as exc:
            print(f"live training viewer disabled: {exc}", flush=True)
            self.process = None

    def poll_control(self) -> bool:
        try:
            mtime = self.control_path.stat().st_mtime
            data = json.loads(self.control_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            self.visualize = False
            self.stop_requested = False
            return False
        self.control_mtime = mtime
        self.visualize = bool(data.get("visualize", False))
        self.stop_requested = bool(data.get("stop", False))
        return self.visualize

    def write_loss_point(self, epoch: int, rollout_k: int, train_total: float, elapsed_seconds: float) -> None:
        if self.loss_history_file is None:
            return
        try:
            self.loss_history_file.write(
                f"{int(epoch)},{float(elapsed_seconds):.6f},{int(rollout_k)},{float(train_total):.9g}\n"
            )
        except OSError:
            pass

    def write_status(self, epoch: int, rollout_k: int, train_total: float | None = None) -> None:
        _ok = write_json_atomic(
            self.status_path,
            {
                "epoch": int(epoch),
                "rollout_k": int(rollout_k),
                "train_total": None if train_total is None else float(train_total),
                "updated_at": time.time(),
            },
        )

    def write_snapshot(
        self,
        *,
        clip: "MotionClip",
        epoch: int,
        rollout_k: int,
        phase: str,
        train_total: float,
        pred_pos: np.ndarray,
        pred_rot: np.ndarray,
        gt_pos: np.ndarray,
        gt_rot: np.ndarray,
    ) -> None:
        tmp = self.snapshot_path.with_suffix(".npz.tmp")
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("wb") as f:
            np.savez(
                f,
                epoch=np.asarray([int(epoch)], dtype=np.int64),
                rollout_k=np.asarray([int(rollout_k)], dtype=np.int64),
                phase=np.asarray([phase]),
                train_total=np.asarray([float(train_total)], dtype=np.float32),
                fps=np.asarray([float(clip.fps)], dtype=np.float32),
                source_up_axis=np.asarray([int(clip.source_up_axis)], dtype=np.int32),
                body_names=np.asarray(clip.body_names),
                parents=np.asarray(clip.parents_body, dtype=np.int64),
                pred_pos=pred_pos.astype(np.float32, copy=False),
                pred_rot=pred_rot.astype(np.float32, copy=False),
                gt_pos=gt_pos.astype(np.float32, copy=False),
                gt_rot=gt_rot.astype(np.float32, copy=False),
            )
        replace_with_retry(tmp, self.snapshot_path)

    def close(self) -> None:
        if self.loss_history_file is not None:
            try:
                self.loss_history_file.close()
            except OSError:
                pass
            self.loss_history_file = None
        if self.cfg.live_viewer_close_on_exit and self.process is not None and self.process.poll() is None:
            self.process.terminate()


def index_pose(pose: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    out = {}
    max_index = int(indices.max().item()) if indices.numel() > 0 else -1
    for key, value in pose.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] > max_index:
            out[key] = value.index_select(0, indices)
        else:
            out[key] = value
    return out


def train(args: argparse.Namespace) -> None:
    process_start_time = time.perf_counter()
    cfg = TrainConfig()
    cfg.max_epochs = args.max_epochs if args.max_epochs is not None else cfg.max_epochs
    cfg.batch_size = args.batch_size if args.batch_size is not None else cfg.batch_size
    cfg.future_window_seconds = (
        args.future_window_seconds if args.future_window_seconds is not None else cfg.future_window_seconds
    )
    if args.cyclic_animation is not None:
        cfg.cyclic_animation = bool(args.cyclic_animation)
    if args.device is not None:
        cfg.device = args.device
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.lr_schedule is not None:
        cfg.lr_schedule = args.lr_schedule
    if args.lr_min_factor is not None:
        cfg.lr_min_factor = args.lr_min_factor
    if args.lr_stage_decay is not None:
        cfg.lr_stage_decay = args.lr_stage_decay
    if args.lr_warmup_epochs is not None:
        cfg.lr_warmup_epochs = max(0, int(args.lr_warmup_epochs))
    if args.lr_plateau_factor is not None:
        cfg.lr_plateau_factor = args.lr_plateau_factor
    if args.lr_plateau_patience_epochs is not None:
        cfg.lr_plateau_patience_epochs = max(1, int(args.lr_plateau_patience_epochs))
    if args.lr_plateau_threshold is not None:
        cfg.lr_plateau_threshold = args.lr_plateau_threshold
    if args.lr_plateau_cooldown_epochs is not None:
        cfg.lr_plateau_cooldown_epochs = max(0, int(args.lr_plateau_cooldown_epochs))
    if args.lr_reset_on_rollout_advance is not None:
        cfg.lr_reset_on_rollout_advance = bool(args.lr_reset_on_rollout_advance)
    if args.val_fraction is not None:
        cfg.val_fraction = args.val_fraction
    if args.predict_residual is not None:
        cfg.predict_residual = args.predict_residual
    if args.zero_init_output is not None:
        cfg.zero_init_output = args.zero_init_output
    if args.rollout_schedule is not None:
        cfg.rollout_schedule = parse_rollout_schedule(args.rollout_schedule)
    if args.curriculum_threshold is not None:
        cfg.curriculum_threshold = args.curriculum_threshold
    if args.curriculum_min_epochs is not None:
        cfg.curriculum_min_epochs = args.curriculum_min_epochs
    if args.curriculum_max_epochs_per_stage is not None:
        cfg.curriculum_max_epochs_per_stage = args.curriculum_max_epochs_per_stage
    if args.curriculum_patience_epochs is not None:
        cfg.curriculum_patience_epochs = args.curriculum_patience_epochs
    if args.curriculum_stall_patience_epochs is not None:
        cfg.curriculum_stall_patience_epochs = args.curriculum_stall_patience_epochs
    if args.curriculum_min_delta is not None:
        cfg.curriculum_min_delta = args.curriculum_min_delta
    if args.stop_on_final_stall is not None:
        cfg.stop_on_final_stall = bool(args.stop_on_final_stall)
    if args.alpha6_full_body_location is not None:
        cfg.alpha6_full_body_location = args.alpha6_full_body_location
    if args.alpha7_contact_label is not None:
        cfg.alpha7_contact_label = args.alpha7_contact_label
    if args.alpha8_foot_penetration is not None:
        cfg.alpha8_foot_penetration = args.alpha8_foot_penetration
    if args.alpha9_foot_sliding is not None:
        cfg.alpha9_foot_sliding = args.alpha9_foot_sliding
    if args.alpha10_freefall is not None:
        cfg.alpha10_freefall = args.alpha10_freefall
    if args.alpha11_contact_height is not None:
        cfg.alpha11_contact_height = args.alpha11_contact_height
    if args.alpha12_termination is not None:
        cfg.alpha12_termination = args.alpha12_termination
    if args.contact_physics_losses is not None:
        cfg.enable_contact_physics_losses = bool(args.contact_physics_losses)
    if args.pose_representation is not None:
        cfg.pose_representation = args.pose_representation
    if args.training_loop is not None:
        cfg.training_loop = args.training_loop
    if args.agent_sampling is not None:
        cfg.agent_sampling = args.agent_sampling
    if args.live_viewer is not None:
        cfg.live_viewer = bool(args.live_viewer)
    if args.live_viewer_max_agents is not None:
        cfg.live_viewer_max_agents = max(1, int(args.live_viewer_max_agents))
    if args.live_viewer_start_visualizing:
        cfg.live_viewer_start_visualizing = True
    if args.live_viewer_close_on_exit:
        cfg.live_viewer_close_on_exit = True
    if args.visual_reporter is not None and bool(args.visual_reporter):
        print("visual reporter disabled: use the standalone model viewer instead", flush=True)
    cfg.visual_reporter = False
    if args.visual_report_interval_seconds is not None:
        cfg.visual_report_interval_seconds = max(1.0, float(args.visual_report_interval_seconds))
    if args.visual_report_device is not None:
        cfg.visual_report_device = args.visual_report_device
    if args.visual_report_max_frames is not None:
        cfg.visual_report_max_frames = max(0, int(args.visual_report_max_frames))
    if args.update_comparison_on_exit is not None:
        cfg.update_comparison_on_exit = bool(args.update_comparison_on_exit)
    if args.comparison_output_path is not None:
        cfg.comparison_output_path = args.comparison_output_path
    if args.comparison_device is not None:
        cfg.comparison_device = args.comparison_device
    if args.comparison_max_frames is not None:
        cfg.comparison_max_frames = max(0, int(args.comparison_max_frames))
    if args.enable_early_termination:
        cfg.enable_early_termination = True
    if args.no_restart_on_termination:
        cfg.restart_on_termination = False
    if args.freefall_body_height_offset_m is not None:
        cfg.freefall_body_height_offset_m = args.freefall_body_height_offset_m
    if args.freefall_initial_offset_history is not None:
        cfg.freefall_initial_offset_history = max(1, int(args.freefall_initial_offset_history))
    if args.freefall_initial_contacts_off is not None:
        cfg.freefall_initial_contacts_off = bool(args.freefall_initial_contacts_off)
    if args.no_freefall_termination:
        cfg.enable_freefall_termination = False
    if args.alpha4_end_effector_location is not None:
        cfg.alpha4_end_effector_location = args.alpha4_end_effector_location
    if args.alpha0_pelvis_location is not None:
        cfg.alpha0_pelvis_location = args.alpha0_pelvis_location
    if args.alpha1_pelvis_rotation is not None:
        cfg.alpha1_pelvis_rotation = args.alpha1_pelvis_rotation
    if args.alpha2_pose_rotation is not None:
        cfg.alpha2_pose_rotation = args.alpha2_pose_rotation
    if args.alpha3_pose_6d_aux is not None:
        cfg.alpha3_pose_6d_aux = args.alpha3_pose_6d_aux
    if args.alpha5_end_effector_rotation is not None:
        cfg.alpha5_end_effector_rotation = args.alpha5_end_effector_rotation
    if args.hidden_dim is not None:
        cfg.hidden_dim = args.hidden_dim
    if args.num_hidden_layers is not None:
        cfg.num_hidden_layers = args.num_hidden_layers
    if args.save_last_every_epochs is not None:
        cfg.save_last_every_epochs = args.save_last_every_epochs
    if args.save_best_every_epochs is not None:
        cfg.save_best_every_epochs = args.save_best_every_epochs
    if args.writer_flush_every_epochs is not None:
        cfg.writer_flush_every_epochs = args.writer_flush_every_epochs
    if args.timed_checkpoint_interval_minutes is not None:
        cfg.timed_checkpoint_interval_minutes = max(0.0, float(args.timed_checkpoint_interval_minutes))
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.date_prefix_run_name:
        cfg.run_name = date_prefixed_run_name(cfg.run_name)
    cfg.use_torch_compile = bool(args.compile) and not args.no_compile
    if args.compile_mode is not None:
        cfg.torch_compile_mode = args.compile_mode
    cfg.show_progress = args.progress
    if args.target_loss_reduction is not None:
        cfg.target_loss_reduction = args.target_loss_reduction
    cfg.stop_at_target_loss_reduction = args.stop_at_target_loss_reduction
    if args.max_train_seconds is not None:
        cfg.max_train_seconds = args.max_train_seconds
    cfg.profile_timing = args.profile_timing
    cfg.profile_sync_cuda = args.profile_sync_cuda
    if args.validation is not None:
        cfg.disable_validation = not bool(args.validation)
    set_seed(cfg.seed)

    clip_specs = clip_specs_from_folders(args.folder_path or folder_path, args.periodic_folder_path, args.nonperiodic_folder_path)
    device = torch.device(cfg.device)
    profiler = TimingProfiler(cfg.profile_timing, device, cfg.profile_sync_cuda)
    with profiler.section("setup/load_npz_and_precompute"):
        clips = load_clips_from_specs(clip_specs, cfg)
    schedule = tuple(k for k in cfg.rollout_schedule if all(clip_can_rollout(clip, k, cfg) for clip in clips))
    if not schedule:
        schedule = (1,)
    def make_loaders(max_rollout: int) -> tuple[DataLoader, DataLoader | None]:
        train_ds = MotionIndexDataset(clips, cfg, "train", max_rollout)
        loader_kwargs = {
            "batch_size": cfg.batch_size,
            "num_workers": cfg.num_workers,
            "pin_memory": device.type == "cuda",
            "persistent_workers": cfg.num_workers > 0,
        }
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        val_loader = None
        if not cfg.disable_validation:
            val_ds = MotionIndexDataset(clips, cfg, "val", max_rollout)
            val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
        return train_loader, val_loader

    agent_clip_indices: torch.Tensor | None = None
    agent_starts: torch.Tensor | None = None
    agent_rng = random.Random(cfg.seed + 7919)
    agent_coverage_order: list[tuple[int, int]] = []
    agent_coverage_cursor = 0

    def random_agent_start(max_rollout: int, clip_index: int | None = None) -> tuple[int, int]:
        eligible = [ci for ci, clip in enumerate(clips) if clip_can_rollout(clip, max_rollout, cfg)]
        if not eligible:
            raise ValueError(f"No clips cover rollout K={max_rollout}")
        ci = agent_rng.choice(eligible) if clip_index is None or int(clip_index) not in eligible else int(clip_index)
        max_start = clip_rollout_max_start(clips[ci], max_rollout, cfg)
        return ci, agent_rng.randint(1, max_start)

    def reset_agent_coverage_order() -> None:
        nonlocal agent_coverage_order, agent_coverage_cursor
        agent_coverage_order = list(train_loader.dataset.items)
        agent_rng.shuffle(agent_coverage_order)
        agent_coverage_cursor = 0

    def coverage_agent_start() -> tuple[int, int]:
        nonlocal agent_coverage_cursor
        if not agent_coverage_order or agent_coverage_cursor >= len(agent_coverage_order):
            reset_agent_coverage_order()
        item = agent_coverage_order[agent_coverage_cursor]
        agent_coverage_cursor += 1
        return item

    def reset_agent_state(max_rollout: int) -> None:
        nonlocal agent_clip_indices, agent_starts
        clip_ids = []
        starts = []
        count = cfg.batch_size
        if cfg.agent_sampling == "coverage":
            count = min(cfg.batch_size, len(train_loader.dataset.items))
        for _ in range(count):
            if cfg.agent_sampling == "coverage":
                ci, start = coverage_agent_start()
            else:
                ci, start = random_agent_start(max_rollout)
            clip_ids.append(ci)
            starts.append(start)
        agent_clip_indices = torch.tensor(clip_ids, dtype=torch.long)
        agent_starts = torch.tensor(starts, dtype=torch.long)

    def current_agent_batch() -> tuple[torch.Tensor, torch.Tensor]:
        if agent_clip_indices is None or agent_starts is None:
            reset_agent_state(rollout_k)
        assert agent_clip_indices is not None and agent_starts is not None
        return agent_clip_indices.clone(), agent_starts.clone()

    with profiler.section("setup/model_optimizer_compile"):
        input_dim, output_dim = make_batch_dims(clips[0], cfg)
        apply_cuda_performance_settings(cfg, device)
        model = MLPController(input_dim, output_dim, cfg).to(device)
        model, compile_enabled = maybe_compile_model(model, input_dim, cfg, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if args.resume_checkpoint is not None:
        with profiler.section("setup/load_resume_checkpoint"):
            resume_path = resolve_path(args.resume_checkpoint)
            resume = torch.load(resume_path, map_location=device, weights_only=False)
            unwrap_compiled_model(model).load_state_dict(resume["model"])
        print(f"resumed model weights from {resume_path}")

    run_dir = resolve_path(cfg.output_dir) / cfg.run_name
    ckpt_dir = run_dir / "checkpoints"
    live_bridge = None
    if cfg.live_viewer:
        with profiler.section("setup/live_viewer_launch"):
            live_bridge = LiveTrainingBridge(run_dir, cfg)
            live_bridge.start()
    with profiler.section("setup/tensorboard_writer"):
        writer = SummaryWriter(run_dir / "tb")
    metadata = {
        "npz_folders": [
            {"path": str(npz_folder_from_path(path)), "cyclic": cyclic}
            for path, cyclic in clip_specs
        ],
        "source_npz_paths": [str(clip.path) for clip in clips],
        "body_names": clips[0].body_names,
        "parents_body": clips[0].parents_body.tolist(),
        "pelvis_index": clips[0].pelvis,
        "non_pelvis_indices": clips[0].non_pelvis,
        "end_effector_indices": clips[0].end_effectors,
        "foot_indices": clips[0].foot_indices,
        "toe_indices": clips[0].toe_indices,
        "pose_representation": cfg.pose_representation,
        "ik_marker_names": clips[0].ik_marker_names,
        "ik_marker_indices": clips[0].ik_marker_indices,
        "core_non_pelvis_indices": clips[0].core_non_pelvis,
        "contact_output": False,
        "body_mass_weights": clips[0].mass_weights.tolist(),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "compile_enabled": compile_enabled,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    rollout_idx = 0
    rollout_k = schedule[rollout_idx]
    with profiler.section("setup/build_dataloaders"):
        train_loader, val_loader = make_loaders(rollout_k)
    if cfg.training_loop == "agents":
        reset_agent_state(rollout_k)
    val_sample_text = "disabled" if val_loader is None else str(len(val_loader.dataset))
    if cfg.training_loop == "agents":
        train_sample_text = (
            f"up to {cfg.batch_size} agents from {len(train_loader.dataset)} starts "
            f"({cfg.agent_sampling}, mixed per-row clips)"
        )
    else:
        train_sample_text = str(len(train_loader.dataset))
    print(f"rollout_k={rollout_k} train_samples={train_sample_text} val_samples={val_sample_text}")
    stage_start_epoch = 1
    stable_epochs = 0
    stall_epochs = 0
    best_val = float("inf")
    baseline_val = None
    target_val = None
    start_time = time.perf_counter()
    timed_interval_seconds = 60.0 * max(0.0, float(cfg.timed_checkpoint_interval_minutes))
    next_timed_checkpoint_at = start_time + timed_interval_seconds if timed_interval_seconds > 0.0 else math.inf
    target_reached_epoch = None
    target_reached_seconds = None
    pending_best_payload = None
    stop_requested = False
    adaptive_lr_state = reset_adaptive_lr(optimizer, cfg) if cfg.lr_schedule == "adaptive_plateau" else None

    def flush_pending_best() -> None:
        nonlocal pending_best_payload
        if pending_best_payload is None:
            return
        best_k = int(pending_best_payload["rollout_k"])
        with profiler.section("checkpoint/write_best"):
            save_payload(ckpt_dir / "checkpoint_best.pt", pending_best_payload)
            save_payload(ckpt_dir / f"checkpoint_best_k{best_k:02d}.pt", pending_best_payload)
        pending_best_payload = None

    for epoch in range(1, cfg.max_epochs + 1):
        if live_bridge is not None:
            live_bridge.poll_control()
            if live_bridge.stop_requested:
                print(f"live viewer stop requested before epoch={epoch}", flush=True)
                break
        epoch_start = time.perf_counter()
        stage_epoch = epoch - stage_start_epoch + 1
        if cfg.lr_schedule == "adaptive_plateau":
            current_lr = float(optimizer.param_groups[0]["lr"])
        else:
            current_lr = learning_rate_for_epoch(cfg, epoch, stage_epoch, rollout_idx)
            set_optimizer_lr(optimizer, current_lr)
        model.train()
        train_parts = []
        train_iter = [current_agent_batch()] if cfg.training_loop == "agents" else train_loader
        pbar = tqdm(train_iter, desc=f"epoch {epoch} train K={rollout_k}", leave=False, disable=not cfg.show_progress)
        for batch in pbar:
            live_capture = None
            if live_bridge is not None:
                live_bridge.poll_control()
                if live_bridge.stop_requested:
                    stop_requested = True
                    break
                live_capture = live_bridge if live_bridge.visualize else None
            with profiler.section("train/zero_grad"):
                optimizer.zero_grad(set_to_none=True)
            with profiler.section("train/forward_loss"):
                loss, scalars = run_batch(
                    model,
                    clips,
                    batch,
                    cfg,
                    rollout_k,
                    device,
                    train=True,
                    live_bridge=live_capture,
                    epoch=epoch,
                    phase="train",
                )
            with profiler.section("train/backward"):
                loss.backward()
            with profiler.section("train/clip_grad"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            with profiler.section("train/optimizer_step"):
                optimizer.step()
            if cfg.training_loop == "agents":
                reset_agent_state(rollout_k)
            train_parts.append(scalars)
            if cfg.show_progress:
                pbar.set_postfix(loss=f"{scalars['total']:.4f}")

        if stop_requested and not train_parts:
            print(f"live viewer stop requested during epoch={epoch}", flush=True)
            break

        model.eval()
        val_parts = []
        if val_loader is None:
            val_parts = train_parts
        else:
            with torch.no_grad():
                for batch in val_loader:
                    with profiler.section("validation/forward_loss"):
                        _, scalars = run_batch(model, clips, batch, cfg, rollout_k, device, train=False)
                    val_parts.append(scalars)

        def mean_scalar(parts: list[dict[str, float]], key: str) -> float:
            return float(np.mean([p[key] for p in parts])) if parts else 0.0

        train_total = mean_scalar(train_parts, "total")
        val_total = mean_scalar(val_parts, "total")
        elapsed_seconds = time.perf_counter() - start_time
        if live_bridge is not None:
            live_bridge.write_loss_point(epoch, rollout_k, train_total, elapsed_seconds)
            live_bridge.write_status(epoch, rollout_k, train_total)
        epoch_seconds = time.perf_counter() - epoch_start
        if baseline_val is None:
            baseline_val = val_total
            target_val = baseline_val * (1.0 - cfg.target_loss_reduction)
        reduction = 0.0 if baseline_val <= 0.0 else 1.0 - (val_total / baseline_val)
        with profiler.section("logging/tensorboard_scalars"):
            writer.add_scalar("loss/train_total", train_total, epoch)
            writer.add_scalar("loss/validation_total", val_total, epoch)
            loss_log_keys = [
                "pelvis_location",
                "pelvis_rotation",
                "pose_rotation",
                "pose_6d_aux",
                "end_effector_location",
                "end_effector_rotation",
                "full_body_location",
            ]
            if cfg.enable_contact_physics_losses:
                loss_log_keys.extend([
                "contact_label",
                "foot_penetration",
                "foot_sliding",
                "contact_height",
                "contact_prob_mean",
                "foot_height_min",
                "foot_speed_mean",
                "foot_horizontal_speed_mean",
                ])
            if cfg.alpha10_freefall != 0.0 or cfg.enable_freefall_termination or cfg.freefall_body_height_offset_m != 0.0:
                loss_log_keys.extend(["freefall", "freefall_relative_error"])
            if cfg.alpha12_termination != 0.0 or cfg.enable_early_termination:
                loss_log_keys.extend(["termination", "termination_rate", "termination_severity"])
            for key in loss_log_keys:
                writer.add_scalar(f"loss/train_{key}", mean_scalar(train_parts, key), epoch)
                writer.add_scalar(f"loss/validation_{key}", mean_scalar(val_parts, key), epoch)
            writer.add_scalar("curriculum/rollout_k", rollout_k, epoch)
            writer.add_scalar("optim/learning_rate", current_lr, epoch)
            writer.add_scalar("timing/epoch_seconds", epoch_seconds, epoch)
            writer.add_scalar("timing/elapsed_seconds", elapsed_seconds, epoch)
            writer.add_scalar("timing/validation_loss_reduction", reduction, epoch)
            if cfg.writer_flush_every_epochs > 0 and epoch % cfg.writer_flush_every_epochs == 0:
                writer.flush()

        if cfg.save_last_every_epochs > 0 and epoch % cfg.save_last_every_epochs == 0:
            with profiler.section("checkpoint/write_last_periodic"):
                save_checkpoint(ckpt_dir / "checkpoint_last.pt", model, optimizer, epoch, best_val, rollout_k, cfg, metadata)
        improved_for_stall = val_total < best_val - cfg.curriculum_min_delta
        if val_total < best_val:
            best_val = val_total
            with profiler.section("checkpoint/build_best_payload"):
                payload = checkpoint_payload(model, optimizer, epoch, best_val, rollout_k, cfg, metadata)
                pending_best_payload = clone_checkpoint_payload(payload)
            if cfg.save_best_every_epochs > 0 and epoch % cfg.save_best_every_epochs == 0:
                flush_pending_best()
        stall_epochs = 0 if improved_for_stall else stall_epochs + 1
        if epoch % cfg.checkpoint_every_epochs == 0:
            with profiler.section("checkpoint/write_numbered"):
                save_checkpoint(
                    ckpt_dir / f"checkpoint_epoch_{epoch:06d}.pt",
                    model,
                    optimizer,
                    epoch,
                    best_val,
                    rollout_k,
                    cfg,
                    metadata,
                )
        now_perf = time.perf_counter()
        if now_perf >= next_timed_checkpoint_at:
            with profiler.section("checkpoint/write_timed"):
                stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                save_checkpoint(
                    ckpt_dir / f"checkpoint_time_{stamp}_epoch_{epoch:06d}.pt",
                    model,
                    optimizer,
                    epoch,
                    best_val,
                    rollout_k,
                    cfg,
                    metadata,
                )
            while next_timed_checkpoint_at <= now_perf:
                next_timed_checkpoint_at += timed_interval_seconds

        stage_epochs = stage_epoch
        can_advance_by_loss = (
            val_total <= cfg.curriculum_threshold
            and stage_epochs >= cfg.curriculum_min_epochs
        )
        can_advance_by_epoch_cap = (
            cfg.curriculum_max_epochs_per_stage > 0
            and stage_epochs >= cfg.curriculum_max_epochs_per_stage
        )
        can_advance_by_stall = (
            cfg.curriculum_stall_patience_epochs > 0
            and stage_epochs >= cfg.curriculum_min_epochs
            and stall_epochs >= cfg.curriculum_stall_patience_epochs
        )
        if can_advance_by_loss:
            stable_epochs += 1
        else:
            stable_epochs = 0
        should_advance = (
            stable_epochs >= cfg.curriculum_patience_epochs
            or can_advance_by_epoch_cap
            or can_advance_by_stall
        )
        was_final_stage = rollout_idx == len(schedule) - 1
        lr_reduced = False
        if should_advance and rollout_idx < len(schedule) - 1:
            flush_pending_best()
            reason = "loss" if stable_epochs >= cfg.curriculum_patience_epochs else "epoch_cap"
            if can_advance_by_stall:
                reason = "stall"
            rollout_idx += 1
            rollout_k = schedule[rollout_idx]
            with profiler.section("curriculum/rebuild_dataloaders"):
                train_loader, val_loader = make_loaders(rollout_k)
            if cfg.training_loop == "agents":
                reset_agent_coverage_order()
                reset_agent_state(rollout_k)
            best_val = float("inf")
            stage_start_epoch = epoch + 1
            if cfg.lr_schedule == "adaptive_plateau" and cfg.lr_reset_on_rollout_advance:
                adaptive_lr_state = reset_adaptive_lr(optimizer, cfg)
                current_lr = cfg.learning_rate
            val_sample_text = "disabled" if val_loader is None else str(len(val_loader.dataset))
            if cfg.training_loop == "agents":
                train_sample_text = (
                    f"up to {cfg.batch_size} agents from {len(train_loader.dataset)} starts "
                    f"({cfg.agent_sampling}, mixed per-row clips)"
                )
            else:
                train_sample_text = str(len(train_loader.dataset))
            print(
                f"advanced rollout_k={rollout_k} reason={reason} "
                f"train_samples={train_sample_text} val_samples={val_sample_text}",
                flush=True,
            )
            stable_epochs = 0
            stall_epochs = 0
        elif cfg.lr_schedule == "adaptive_plateau" and adaptive_lr_state is not None:
            adaptive_lr_state, lr_reduced = step_adaptive_lr(optimizer, cfg, adaptive_lr_state, val_total)

        print(
            f"epoch={epoch:04d} K={rollout_k:02d} train={train_total:.6f} "
            f"val={val_total:.6f} best={best_val:.6f} "
            f"reduction={reduction * 100.0:.2f}% stall={stall_epochs} "
            f"lr={current_lr:.3g}{'->' + format(optimizer.param_groups[0]['lr'], '.3g') if lr_reduced else ''} "
            f"epoch_s={epoch_seconds:.2f} elapsed_s={elapsed_seconds:.2f}",
            flush=True,
        )
        if cfg.stop_on_final_stall and was_final_stage and can_advance_by_stall:
            print(
                f"final rollout_k={rollout_k} stopped on validation stall "
                f"after {stall_epochs} epochs without improvement >= {cfg.curriculum_min_delta:g}",
                flush=True,
            )
            break
        if target_reached_epoch is None and target_val is not None and val_total <= target_val:
            target_reached_epoch = epoch
            target_reached_seconds = elapsed_seconds
            print(
                f"target_loss_reduction={cfg.target_loss_reduction * 100.0:.2f}% "
                f"reached at epoch={epoch} elapsed_s={elapsed_seconds:.2f} "
                f"baseline_val={baseline_val:.6f} target_val={target_val:.6f} val={val_total:.6f}",
                flush=True,
            )
            if cfg.stop_at_target_loss_reduction:
                break
        if cfg.max_train_seconds > 0.0 and elapsed_seconds >= cfg.max_train_seconds:
            print(
                f"max_train_seconds={cfg.max_train_seconds:.2f} reached at epoch={epoch} "
                f"elapsed_s={elapsed_seconds:.2f}",
                flush=True,
            )
            break
        if stop_requested:
            print(f"live viewer stop requested after epoch={epoch}", flush=True)
            break

    with profiler.section("logging/tensorboard_close"):
        writer.close()
    flush_pending_best()
    with profiler.section("checkpoint/write_last_final"):
        save_checkpoint(ckpt_dir / "checkpoint_last.pt", model, optimizer, epoch, best_val, rollout_k, cfg, metadata)
    total_seconds = time.perf_counter() - start_time
    profiler.write_csv(run_dir / "timing_profile.csv", time.perf_counter() - process_start_time)
    if cfg.update_comparison_on_exit:
        with profiler.section("postprocess/update_model_comparison"):
            update_model_comparison_html(clips[0].path, ckpt_dir, cfg)
    if target_reached_epoch is None and baseline_val is not None and target_val is not None:
        print(
            f"target_loss_reduction={cfg.target_loss_reduction * 100.0:.2f}% not reached "
            f"after {epoch} epochs elapsed_s={total_seconds:.2f} "
            f"baseline_val={baseline_val:.6f} target_val={target_val:.6f} best_val={best_val:.6f}"
        )
    elif target_reached_epoch is not None:
        print(
            f"timing_summary target_epoch={target_reached_epoch} "
            f"target_elapsed_s={target_reached_seconds:.2f} total_elapsed_s={total_seconds:.2f}"
        )
    if live_bridge is not None:
        live_bridge.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a kinematic locomotion imitator from NPZ motion clips.")
    parser.add_argument("--folder-path", default=None, help="Override top-level folder_path.")
    parser.add_argument(
        "--periodic-folder-path",
        default=None,
        help="Semicolon-separated motion folders that should use cyclic root/pose indexing.",
    )
    parser.add_argument(
        "--nonperiodic-folder-path",
        default=None,
        help="Semicolon-separated motion folders that should never sample starts past clip length minus rollout.",
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--future-window-seconds", type=float, default=None)
    parser.add_argument("--cyclic-animation", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine", "stage_decay", "stage_cosine", "adaptive_plateau"),
        default=None,
    )
    parser.add_argument("--lr-min-factor", type=float, default=None)
    parser.add_argument("--lr-stage-decay", type=float, default=None)
    parser.add_argument("--lr-warmup-epochs", type=int, default=None)
    parser.add_argument("--lr-plateau-factor", type=float, default=None)
    parser.add_argument("--lr-plateau-patience-epochs", type=int, default=None)
    parser.add_argument("--lr-plateau-threshold", type=float, default=None)
    parser.add_argument("--lr-plateau-cooldown-epochs", type=int, default=None)
    parser.add_argument("--lr-reset-on-rollout-advance", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--predict-residual", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--zero-init-output", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rollout-schedule", default=None, help="Comma-separated rollout K values, e.g. 1,2,4,8,16,32.")
    parser.add_argument("--curriculum-threshold", type=float, default=None)
    parser.add_argument("--curriculum-min-epochs", type=int, default=None)
    parser.add_argument("--curriculum-max-epochs-per-stage", type=int, default=None)
    parser.add_argument("--curriculum-patience-epochs", type=int, default=None)
    parser.add_argument("--curriculum-stall-patience-epochs", type=int, default=None)
    parser.add_argument("--curriculum-min-delta", type=float, default=None)
    parser.add_argument("--stop-on-final-stall", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--alpha0-pelvis-location", type=float, default=None)
    parser.add_argument("--alpha1-pelvis-rotation", type=float, default=None)
    parser.add_argument("--alpha2-pose-rotation", type=float, default=None)
    parser.add_argument("--alpha3-pose-6d-aux", type=float, default=None)
    parser.add_argument("--alpha4-end-effector-location", type=float, default=None)
    parser.add_argument("--alpha5-end-effector-rotation", type=float, default=None)
    parser.add_argument("--alpha6-full-body-location", type=float, default=None)
    parser.add_argument("--alpha7-contact-label", type=float, default=None)
    parser.add_argument("--alpha8-foot-penetration", type=float, default=None)
    parser.add_argument("--alpha9-foot-sliding", type=float, default=None)
    parser.add_argument("--alpha10-freefall", type=float, default=None)
    parser.add_argument("--alpha11-contact-height", type=float, default=None)
    parser.add_argument("--alpha12-termination", type=float, default=None)
    parser.add_argument("--contact-physics-losses", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pose-representation", choices=("rot6", "ik_markers"), default=None)
    parser.add_argument("--training-loop", choices=("sampled", "agents"), default=None)
    parser.add_argument("--agent-sampling", choices=("random", "coverage"), default=None)
    parser.add_argument("--live-viewer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--live-viewer-max-agents", type=int, default=None)
    parser.add_argument("--live-viewer-start-visualizing", action="store_true")
    parser.add_argument("--live-viewer-close-on-exit", action="store_true")
    parser.add_argument("--visual-reporter", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--visual-report-interval-seconds", type=float, default=None)
    parser.add_argument("--visual-report-device", default=None)
    parser.add_argument("--visual-report-max-frames", type=int, default=None)
    parser.add_argument("--update-comparison-on-exit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--comparison-output-path", default=None)
    parser.add_argument("--comparison-device", default=None)
    parser.add_argument("--comparison-max-frames", type=int, default=None)
    parser.add_argument("--enable-early-termination", action="store_true")
    parser.add_argument("--no-restart-on-termination", action="store_true")
    parser.add_argument("--freefall-body-height-offset-m", type=float, default=None)
    parser.add_argument("--freefall-initial-offset-history", type=int, choices=(1, 2), default=None)
    parser.add_argument("--freefall-initial-contacts-off", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no-freefall-termination", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-hidden-layers", type=int, default=None)
    parser.add_argument("--save-last-every-epochs", type=int, default=None)
    parser.add_argument("--save-best-every-epochs", type=int, default=None)
    parser.add_argument("--writer-flush-every-epochs", type=int, default=None)
    parser.add_argument("--timed-checkpoint-interval-minutes", type=float, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--date-prefix-run-name", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--compile", action="store_true", help="Try torch.compile after a forward/backward probe.")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile.")
    parser.add_argument("--compile-mode", default=None, help="torch.compile mode, for example default or reduce-overhead.")
    parser.add_argument("--progress", action="store_true", help="Show per-epoch tqdm progress bars.")
    parser.add_argument("--target-loss-reduction", type=float, default=None)
    parser.add_argument("--stop-at-target-loss-reduction", action="store_true")
    parser.add_argument("--max-train-seconds", type=float, default=None)
    parser.add_argument("--profile-timing", action="store_true", help="Write timing_profile.csv in the run directory.")
    parser.add_argument(
        "--profile-sync-cuda",
        action="store_true",
        help="Synchronize CUDA around timed sections for stricter timing. Slower, but more precise.",
    )
    parser.add_argument(
        "--validation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable validation passes. Disabled by default.",
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
