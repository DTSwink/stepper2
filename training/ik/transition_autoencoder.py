from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    from . import contact_physics as cp
    from . import ik_core as tl
except ImportError:
    import contact_physics as cp
    import ik_core as tl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AEConfig:
    folder_path: str = "data/npz_final"
    periodic_folder_path: str = ""
    nonperiodic_folder_path: str = ""
    run_name: str = "transition_ae"
    date_prefix_run_name: bool = True
    output_dir: str = "training/runs"
    latent_dim: int = 64
    hidden_dim: int = 512
    num_hidden_layers: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    max_epochs: int = 2000
    seed: int = 1234
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    sample_mode: str = "rows"
    cyclic_animation: bool = False
    input_noise_std: float = 0.0
    input_noise_mask: str = "all"
    std_floor: float = 1e-4
    target_loss_reduction: float = 0.995
    stall_patience_epochs: int = 120
    min_delta: float = 1e-6
    tier_eval_every_epochs: int = 25
    timed_checkpoint_interval_minutes: float = 30.0
    compatibility_weight: float = 0.0
    compatibility_direction_weight: float = 1.0
    compatibility_temporal_weight: float = 1.0
    compatibility_statue_weight: float = 0.0
    compatibility_damped_weight: float = 0.0
    compatibility_damped_factor: float = 0.35
    compatibility_yaw_body_weight: float = 0.0
    compatibility_yaw_min_degrees: float = 6.0
    compatibility_yaw_max_degrees: float = 25.0
    structured_denoise_statue_weight: float = 0.0
    structured_denoise_damped_weight: float = 0.0
    structured_denoise_damped_factor: float = 0.35
    compatibility_temporal_min_skip: int = 2
    compatibility_temporal_max_skip: int = 8
    compatibility_hidden_dim: int = 256
    compatibility_num_hidden_layers: int = 2
    include_transition_foot_motion: bool = False
    foot_slide_scale_mps: float = 1.0
    transition_yaw_scale_radps: float = 10.0
    transition_foot_motion_loss_weight: float = 1.0
    root_lookahead_steps: int = 0
    pose_representation: str = "rot6"
    pelvis_feature_weight: float = 1.0
    lower_body_feature_weight: float = 1.0
    foot_feature_weight: float = 1.0
    velocity_feature_weight: float = 1.0
    reconstruction_top_fraction: float = 0.0
    reconstruction_top_weight: float = 0.0
    conditional_root_window: bool = False
    condition_body_dropout_prob: float = 0.0


def ae_config_from_dict(values: dict) -> AEConfig:
    data = dict(values)
    legacy_yaw_key = "foot" + "_yaw_scale_radps"
    if legacy_yaw_key in data and "transition_yaw_scale_radps" not in data:
        data["transition_yaw_scale_radps"] = data.pop(legacy_yaw_key)
    return AEConfig(**data)


def conditional_transition_indices(schema: dict[str, int], device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    condition: list[int] = []
    condition.extend(range(0, int(schema["input_dim"])))
    root_lookahead_dim = int(schema.get("root_lookahead_dim", 0))
    if root_lookahead_dim > 0:
        start = int(schema["root_lookahead_start"])
        condition.extend(range(start, start + root_lookahead_dim))
    target_start = int(schema["next_output_start"])
    target_end = int(schema.get("root_lookahead_start", schema["total_dim"]))
    target = list(range(target_start, target_end))
    return (
        torch.tensor(condition, dtype=torch.long, device=device),
        torch.tensor(target, dtype=torch.long, device=device),
    )


class TransitionAutoencoder(nn.Module):
    def __init__(self, dim: int, cfg: AEConfig, schema: dict[str, int] | None = None):
        super().__init__()
        self.conditional_root_window = bool(getattr(cfg, "conditional_root_window", False))
        if self.conditional_root_window:
            if schema is None:
                raise ValueError("conditional_root_window=True requires a transition schema")
            condition_indices, target_indices = conditional_transition_indices(schema)
            self.register_buffer("condition_indices", condition_indices)
            self.register_buffer("target_indices", target_indices)
            net_input_dim = int(condition_indices.numel())
            net_output_dim = int(target_indices.numel())
        else:
            net_input_dim = int(dim)
            net_output_dim = int(dim)
        reconstruction_weights = transition_reconstruction_weights(schema, cfg, net_output_dim, self.conditional_root_window)
        self.register_buffer("reconstruction_weights", reconstruction_weights)
        self.has_weighted_reconstruction = bool(torch.any(reconstruction_weights != 1.0).item())
        self.reconstruction_top_fraction = max(0.0, min(1.0, float(cfg.reconstruction_top_fraction)))
        self.reconstruction_top_weight = max(0.0, float(cfg.reconstruction_top_weight))
        encoder: list[nn.Module] = []
        in_dim = net_input_dim
        for _ in range(cfg.num_hidden_layers):
            encoder += [nn.Linear(in_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.GELU()]
            in_dim = cfg.hidden_dim
        encoder += [nn.Linear(in_dim, cfg.latent_dim), nn.GELU()]
        decoder: list[nn.Module] = []
        in_dim = cfg.latent_dim
        for _ in range(cfg.num_hidden_layers):
            decoder += [nn.Linear(in_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.GELU()]
            in_dim = cfg.hidden_dim
        decoder += [nn.Linear(in_dim, net_output_dim)]
        self.net = nn.Sequential(*(encoder + decoder))
        self.compatibility_head = None
        if cfg.compatibility_weight > 0.0:
            layers: list[nn.Module] = []
            in_dim = dim
            for _ in range(cfg.compatibility_num_hidden_layers):
                layers += [
                    nn.Linear(in_dim, cfg.compatibility_hidden_dim),
                    nn.LayerNorm(cfg.compatibility_hidden_dim),
                    nn.GELU(),
                ]
                in_dim = cfg.compatibility_hidden_dim
            layers += [nn.Linear(in_dim, 1)]
            self.compatibility_head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conditional_root_window:
            x = x.index_select(-1, self.condition_indices)
        return self.net(x)

    def target(self, x: torch.Tensor) -> torch.Tensor:
        if self.conditional_root_window:
            return x.index_select(-1, self.target_indices)
        return x

    def has_compatibility_head(self) -> bool:
        return self.compatibility_head is not None

    def compatibility_logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.compatibility_head is None:
            raise RuntimeError("This transition autoencoder was created without a compatibility head.")
        return self.compatibility_head(x).squeeze(-1)


def transition_reconstruction_weights(
    schema: dict[str, int] | None,
    cfg: AEConfig,
    output_dim: int,
    conditional_root_window: bool,
) -> torch.Tensor:
    weights = torch.ones(int(output_dim), dtype=torch.float32)
    if schema is None:
        return weights
    total_dim = int(schema["total_dim"])
    full_weights = torch.ones(total_dim, dtype=torch.float32)

    def apply(indices: list[int], value: float) -> None:
        if value == 1.0 or not indices:
            return
        idx = torch.tensor([i for i in indices if 0 <= i < total_dim], dtype=torch.long)
        if idx.numel() > 0:
            full_weights[idx] = torch.maximum(full_weights[idx], torch.full_like(full_weights[idx], float(value)))

    def body_sets() -> tuple[set[int], set[int]]:
        names = [str(name).lower() for name in schema.get("body_names", [])]
        lower_keywords = ("thigh", "calf", "leg", "knee", "ankle", "foot", "ball", "toe")
        foot_keywords = ("foot", "ball", "toe")
        lower = {i for i, name in enumerate(names) if any(key in name for key in lower_keywords)}
        foot = {i for i, name in enumerate(names) if any(key in name for key in foot_keywords)}
        foot.update(int(i) for i in schema.get("foot_indices", []))
        foot.update(int(i) for i in schema.get("toe_indices", []))
        lower.update(foot)
        return lower, foot

    lower_body, foot_bodies = body_sets()
    body_count = int(schema.get("body_count", max(1, int(schema["next_canon_dim"]) // 3)))
    nonpelvis = [int(i) for i in schema.get("non_pelvis", list(range(1, body_count)))]
    nonpelvis_slot = {body_i: slot for slot, body_i in enumerate(nonpelvis)}
    pose_dim = int(schema["pose_dim"])
    output_dim_full = int(schema["output_dim"])
    input_dim = int(schema["input_dim"])
    next_output_start = int(schema["next_output_start"])
    next_canon_start = int(schema["next_canon_start"])
    next_velocity_start = int(schema["next_velocity_start"])
    next_canon_dim = int(schema["next_canon_dim"])
    next_velocity_dim = int(schema["next_velocity_dim"])
    velocity_dim = int(schema["velocity_dim"])

    def pose_indices(pose_start: int, include_contact: bool = True) -> tuple[list[int], list[int], list[int]]:
        pelvis = list(range(pose_start, pose_start + 9))
        canon_start = pose_start + 9
        rot_start = canon_start + body_count * 3
        lower: list[int] = []
        foot: list[int] = []
        for body_i in lower_body:
            lower.extend(range(canon_start + body_i * 3, canon_start + body_i * 3 + 3))
            slot = nonpelvis_slot.get(body_i)
            if slot is not None:
                lower.extend(range(rot_start + slot * 6, rot_start + slot * 6 + 6))
        for body_i in foot_bodies:
            foot.extend(range(canon_start + body_i * 3, canon_start + body_i * 3 + 3))
            slot = nonpelvis_slot.get(body_i)
            if slot is not None:
                foot.extend(range(rot_start + slot * 6, rot_start + slot * 6 + 6))
        return pelvis, lower, foot

    pelvis_weight = float(getattr(cfg, "pelvis_feature_weight", 1.0))
    lower_weight = float(getattr(cfg, "lower_body_feature_weight", 1.0))
    foot_weight_cfg = float(getattr(cfg, "foot_feature_weight", 1.0))
    velocity_weight = float(getattr(cfg, "velocity_feature_weight", 1.0))

    for pose_start in (0, pose_dim):
        pelvis_idx, lower_idx, foot_idx = pose_indices(pose_start)
        apply(pelvis_idx, pelvis_weight)
        apply(lower_idx, lower_weight)
        apply(foot_idx, foot_weight_cfg)

    # Current pose-delta input velocity: pelvis velocity followed by all canonical joint velocities.
    input_velocity_start = pose_dim * 2
    apply(list(range(input_velocity_start, input_velocity_start + velocity_dim)), velocity_weight)
    joint_vel_start = input_velocity_start + 3
    lower_vel: list[int] = []
    foot_vel: list[int] = []
    for body_i in lower_body:
        lower_vel.extend(range(joint_vel_start + body_i * 3, joint_vel_start + body_i * 3 + 3))
    for body_i in foot_bodies:
        foot_vel.extend(range(joint_vel_start + body_i * 3, joint_vel_start + body_i * 3 + 3))
    apply(lower_vel, max(lower_weight, velocity_weight))
    apply(foot_vel, max(foot_weight_cfg, velocity_weight))

    # Next output: pelvis position/rotation and non-pelvis rotations.
    apply(list(range(next_output_start, next_output_start + 9)), pelvis_weight)
    next_rot_start = next_output_start + 9
    lower_rot: list[int] = []
    foot_rot: list[int] = []
    for body_i in lower_body:
        slot = nonpelvis_slot.get(body_i)
        if slot is not None:
            lower_rot.extend(range(next_rot_start + slot * 6, next_rot_start + slot * 6 + 6))
    for body_i in foot_bodies:
        slot = nonpelvis_slot.get(body_i)
        if slot is not None:
            foot_rot.extend(range(next_rot_start + slot * 6, next_rot_start + slot * 6 + 6))
    apply(lower_rot, lower_weight)
    apply(foot_rot, foot_weight_cfg)

    # Next canonical positions.
    lower_canon: list[int] = []
    foot_canon: list[int] = []
    for body_i in lower_body:
        lower_canon.extend(range(next_canon_start + body_i * 3, next_canon_start + body_i * 3 + 3))
    for body_i in foot_bodies:
        foot_canon.extend(range(next_canon_start + body_i * 3, next_canon_start + body_i * 3 + 3))
    apply(lower_canon, lower_weight)
    apply(foot_canon, foot_weight_cfg)

    # Next pose-delta velocity.
    apply(list(range(next_velocity_start, next_velocity_start + next_velocity_dim)), velocity_weight)
    next_joint_vel_start = next_velocity_start + 3
    lower_next_vel: list[int] = []
    foot_next_vel: list[int] = []
    for body_i in lower_body:
        lower_next_vel.extend(range(next_joint_vel_start + body_i * 3, next_joint_vel_start + body_i * 3 + 3))
    for body_i in foot_bodies:
        foot_next_vel.extend(range(next_joint_vel_start + body_i * 3, next_joint_vel_start + body_i * 3 + 3))
    apply(lower_next_vel, max(lower_weight, velocity_weight))
    apply(foot_next_vel, max(foot_weight_cfg, velocity_weight))

    foot_dim = int(schema.get("transition_foot_motion_dim", 0))
    foot_weight = float(getattr(cfg, "transition_foot_motion_loss_weight", 1.0))
    if foot_dim > 0 and foot_weight != 1.0:
        foot_start = int(schema["transition_foot_motion_start"])
        foot_end = foot_start + foot_dim
        apply(list(range(foot_start, foot_end)), foot_weight)

    if conditional_root_window:
        _condition, target = conditional_transition_indices(schema)
        return full_weights.index_select(0, target).contiguous()
    return full_weights


def reduce_reconstruction_error_rows(model: nn.Module, error: torch.Tensor) -> torch.Tensor:
    mean_error = error.mean(dim=-1)
    top_fraction = float(getattr(model, "reconstruction_top_fraction", 0.0))
    top_weight = float(getattr(model, "reconstruction_top_weight", 0.0))
    if top_fraction <= 0.0 or top_weight <= 0.0 or error.shape[-1] <= 1:
        return mean_error
    k = max(1, int(math.ceil(error.shape[-1] * min(1.0, top_fraction))))
    top_mean = torch.topk(error, k=k, dim=-1, largest=True).values.mean(dim=-1)
    return mean_error + top_weight * top_mean


def reconstruction_loss(
    model: nn.Module,
    recon: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "huber",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    has_top = float(getattr(model, "reconstruction_top_fraction", 0.0)) > 0.0 and float(
        getattr(model, "reconstruction_top_weight", 0.0)
    ) > 0.0
    has_weights = bool(getattr(model, "has_weighted_reconstruction", False))
    if not has_top and not has_weights:
        if loss_type == "mse":
            return F.mse_loss(recon, target)
        return F.huber_loss(recon, target, delta=huber_delta)
    if loss_type == "mse":
        error = F.mse_loss(recon, target, reduction="none")
    else:
        error = F.huber_loss(recon, target, reduction="none", delta=huber_delta)
    weights = getattr(model, "reconstruction_weights", None)
    if weights is not None:
        error = error * weights.to(device=error.device, dtype=error.dtype)
    return reduce_reconstruction_error_rows(model, error).mean()


def reconstruction_loss_rows(
    model: nn.Module,
    recon: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    has_top = float(getattr(model, "reconstruction_top_fraction", 0.0)) > 0.0 and float(
        getattr(model, "reconstruction_top_weight", 0.0)
    ) > 0.0
    has_weights = bool(getattr(model, "has_weighted_reconstruction", False))
    if not has_top and not has_weights:
        if loss_type == "huber":
            return F.huber_loss(recon, target, reduction="none", delta=huber_delta).mean(dim=-1)
        return F.mse_loss(recon, target, reduction="none").mean(dim=-1)
    if loss_type == "huber":
        error = F.huber_loss(recon, target, reduction="none", delta=huber_delta)
    else:
        error = F.mse_loss(recon, target, reduction="none")
    weights = getattr(model, "reconstruction_weights", None)
    if weights is not None:
        error = error * weights.to(device=error.device, dtype=error.dtype)
    return reduce_reconstruction_error_rows(model, error)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def transition_schema(clip: tl.MotionClip, cfg: tl.TrainConfig) -> dict[str, int]:
    if tl.uses_ik_markers(cfg.pose_representation):
        pose_dim = 3 + 6 + clip.Jcore * 6 + clip.ik_payload_dim + (2 if cfg.use_contact_state else 0)
        velocity_dim = 3 + clip.ik_payload_dim
        ik_marker_dim = clip.ik_payload_dim
        core_rot_dim = clip.Jcore * 6
    else:
        pose_dim = 3 + 6 + clip.J * 3 + clip.Jn * 6 + (2 if cfg.use_contact_state else 0)
        velocity_dim = 3 + clip.J * 3
        ik_marker_dim = 0
        core_rot_dim = 0
    input_dim, output_dim = tl.make_batch_dims(clip, cfg)
    next_canon_dim = clip.J * 3
    next_velocity_dim = velocity_dim
    transition_foot_motion_dim = 4 if getattr(cfg, "include_transition_foot_motion", False) else 0
    root_lookahead_dim = max(0, int(getattr(cfg, "root_lookahead_steps", 0))) * 3
    base_total_dim = input_dim + output_dim + next_canon_dim + next_velocity_dim
    total_without_root_lookahead = base_total_dim + transition_foot_motion_dim
    return {
        "body_count": clip.J,
        "body_names": list(clip.body_names),
        "non_pelvis": [int(i) for i in clip.non_pelvis],
        "foot_indices": [int(i) for i in clip.foot_indices],
        "toe_indices": [int(i) for i in clip.toe_indices],
        "pose_dim": pose_dim,
        "velocity_dim": velocity_dim,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "next_canon_dim": next_canon_dim,
        "next_velocity_dim": next_velocity_dim,
        "transition_foot_motion_dim": transition_foot_motion_dim,
        "root_lookahead_dim": root_lookahead_dim,
        "total_dim": total_without_root_lookahead + root_lookahead_dim,
        "pose_representation": str(cfg.pose_representation),
        "core_non_pelvis": [int(i) for i in clip.core_non_pelvis],
        "core_non_pelvis_count": int(clip.Jcore),
        "core_rot_dim": int(core_rot_dim),
        "ik_payload_dim": int(clip.ik_payload_dim),
        "ik_payload_start": 3 + 6 + core_rot_dim,
        "input_root_start": pose_dim * 2 + velocity_dim,
        "input_root_end": input_dim,
        "next_output_start": input_dim,
        "next_canon_start": input_dim + output_dim,
        "next_velocity_start": input_dim + output_dim + next_canon_dim,
        "transition_foot_motion_start": base_total_dim,
        "root_lookahead_start": total_without_root_lookahead,
        "output_reference_root": tl.OUTPUT_REFERENCE_ROOT,
        "output_prediction_mode": tl.normalized_output_prediction_mode(),
        "state_reference_root": tl.STATE_REFERENCE_ROOT,
    }


def transition_foot_motion_features(
    clip: tl.MotionClip,
    cur_idx: torch.Tensor,
    cur_pose: dict[str, torch.Tensor],
    next_pose: dict[str, torch.Tensor],
    cfg: tl.TrainConfig,
    device: torch.device,
) -> torch.Tensor:
    cur_root_pos, cur_root_rot, _cur_yaw, _cur_heading = tl.root_state(clip, cur_idx, cfg, device)
    next_root_pos, next_root_rot, _next_yaw, _next_heading = tl.root_state(clip, cur_idx + 1, cfg, device)
    cur_pos, cur_rot, _cur_canon = tl.fk_from_pose(clip, cur_root_pos, cur_root_rot, cur_pose, device)
    next_pos, next_rot, _next_canon = tl.fk_from_pose(clip, next_root_pos, next_root_rot, next_pose, device)
    foot_indices = tuple(int(x) for x in clip.foot_indices_tensor.tolist())
    toe_indices = tuple(int(x) for x in clip.toe_indices_tensor.tolist())
    slide = cp.foot_slide_speeds(
        cur_pos,
        cur_rot,
        next_pos,
        next_rot,
        foot_indices,
        toe_indices,
        clip.fps,
    )
    yaw = cp.foot_vertical_yaw_speeds(
        cur_pos,
        cur_rot,
        next_pos,
        next_rot,
        foot_indices,
        toe_indices,
        clip.fps,
    )
    slide_scale = max(float(getattr(cfg, "foot_slide_scale_mps", 1.0)), 1e-6)
    yaw_scale = max(float(getattr(cfg, "transition_yaw_scale_radps", 10.0)), 1e-6)
    return torch.cat((slide / slide_scale, yaw / yaw_scale), dim=-1)


def root_lookahead_features(
    clip: tl.MotionClip,
    cur_idx: torch.Tensor,
    cfg: tl.TrainConfig,
    device: torch.device,
) -> torch.Tensor:
    steps = max(0, int(getattr(cfg, "root_lookahead_steps", 0)))
    if steps <= 0:
        return torch.empty((cur_idx.shape[0], 0), dtype=torch.float32, device=device)
    feats = []
    for offset in range(1, steps + 1):
        prev_idx = cur_idx + offset
        next_idx = cur_idx + offset + 1
        if not clip.cyclic_animation:
            prev_idx = torch.clamp(prev_idx, max=clip.T - 1)
            next_idx = torch.clamp(next_idx, max=clip.T - 1)
        feats.append(tl.root_delta_feature(clip, prev_idx, next_idx, cfg, device))
    return torch.cat(feats, dim=-1)


def transition_feature_from_next_pose(
    clip: tl.MotionClip,
    prev_idx: torch.Tensor,
    cur_idx: torch.Tensor,
    prev_pose: dict[str, torch.Tensor],
    cur_pose: dict[str, torch.Tensor],
    next_pose: dict[str, torch.Tensor],
    cfg: tl.TrainConfig,
    device: torch.device,
) -> torch.Tensor:
    model_input = tl.build_input(clip, prev_idx, cur_idx, prev_pose, cur_pose, cfg, device)
    next_pose_state = next_pose
    cur_root_pos, cur_root_rot, _cur_yaw, _cur_heading = tl.root_state(clip, cur_idx, cfg, device)
    next_root_pos, next_root_rot, _next_yaw, _next_heading = tl.root_state(clip, cur_idx + 1, cfg, device)
    cur_pose_delta = cur_pose
    if tl.output_reference_uses_current_root():
        next_pose = tl.rebase_pose_root(clip, next_pose, next_root_pos, next_root_rot, cur_root_pos, cur_root_rot)
    else:
        cur_pose_delta = tl.rebase_pose_root(clip, cur_pose, cur_root_pos, cur_root_rot, next_root_pos, next_root_rot)
    next_output = tl.pose_target_output(next_pose)
    pelvis_next_vel = (next_pose["pelvis_pos"] - cur_pose_delta["pelvis_pos"]) / cfg.pose_delta_scale_final
    if "ik_payload" in next_pose:
        joint_next_vel = (next_pose["ik_payload"] - cur_pose_delta["ik_payload"]).reshape(cur_idx.shape[0], -1)
    else:
        joint_next_vel = (next_pose["canon_pos"] - cur_pose_delta["canon_pos"]).reshape(cur_idx.shape[0], -1)
    joint_next_vel = joint_next_vel / cfg.pose_delta_scale_final
    parts = [
        model_input,
        next_output,
        next_pose["canon_pos"].reshape(cur_idx.shape[0], -1),
        pelvis_next_vel,
        joint_next_vel,
    ]
    if getattr(cfg, "include_transition_foot_motion", False):
        parts.append(transition_foot_motion_features(clip, cur_idx, cur_pose, next_pose_state, cfg, device))
    if max(0, int(getattr(cfg, "root_lookahead_steps", 0))) > 0:
        parts.append(root_lookahead_features(clip, cur_idx, cfg, device))
    return torch.cat(parts, dim=-1)


def clean_transition_features(
    clip: tl.MotionClip,
    cur_idx: torch.Tensor,
    cfg: tl.TrainConfig,
    device: torch.device,
) -> torch.Tensor:
    prev_idx = cur_idx - 1
    next_idx = cur_idx + 1
    prev_pose = tl.get_pose_from_clip(clip, prev_idx, device)
    cur_pose = tl.get_pose_from_clip(clip, cur_idx, device)
    next_pose = tl.get_pose_from_clip(clip, next_idx, device)
    return transition_feature_from_next_pose(clip, prev_idx, cur_idx, prev_pose, cur_pose, next_pose, cfg, device)


def transition_features_with_offset(
    clip: tl.MotionClip,
    cur_idx: torch.Tensor,
    next_offset: int,
    cfg: tl.TrainConfig,
    device: torch.device,
) -> torch.Tensor:
    prev_idx = cur_idx - 1
    next_idx = cur_idx + next_offset
    prev_pose = tl.get_pose_from_clip(clip, prev_idx, device)
    cur_pose = tl.get_pose_from_clip(clip, cur_idx, device)
    next_pose = tl.get_pose_from_clip(clip, next_idx, device)
    return transition_feature_from_next_pose(clip, prev_idx, cur_idx, prev_pose, cur_pose, next_pose, cfg, device)


def collect_clean_feature_rows(
    clips: list[tl.MotionClip],
    cfg: tl.TrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chunks = []
    clip_chunks = []
    idx_chunks = []
    for ci, clip in enumerate(clips):
        stop = clip.cyclic_period if clip.cyclic_animation else clip.T - cfg.future_window
        if stop <= 1:
            continue
        idx = torch.arange(1, stop, dtype=torch.long, device=device)
        chunks.append(clean_transition_features(clip, idx, cfg, device).detach().cpu())
        clip_chunks.append(torch.full((idx.numel(),), ci, dtype=torch.long))
        idx_chunks.append(idx.detach().cpu())
    return torch.cat(chunks, dim=0), torch.cat(clip_chunks, dim=0), torch.cat(idx_chunks, dim=0)


def balanced_clip_epoch_indices(
    clip_ids: torch.Tensor,
    total_samples: int,
    device: torch.device,
) -> torch.Tensor:
    unique_clip_ids = torch.unique(clip_ids, sorted=True)
    per_clip_rows = [torch.nonzero(clip_ids == clip_id, as_tuple=False).flatten() for clip_id in unique_clip_ids]
    sampled_clip_slots = torch.randint(0, len(per_clip_rows), (total_samples,), device=device)
    out = torch.empty((total_samples,), dtype=torch.long, device=device)
    for slot, rows in enumerate(per_clip_rows):
        mask = sampled_clip_slots == slot
        count = int(mask.sum().item())
        if count == 0:
            continue
        sampled_rows = rows[torch.randint(0, rows.numel(), (count,), device=device)]
        out[mask] = sampled_rows
    return out


def collect_clean_features(clips: list[tl.MotionClip], cfg: tl.TrainConfig, device: torch.device) -> torch.Tensor:
    return collect_clean_feature_rows(clips, cfg, device)[0]


def collect_temporal_skip_features(
    clips: list[tl.MotionClip],
    cfg: tl.TrainConfig,
    device: torch.device,
    min_skip: int,
    max_skip: int,
) -> torch.Tensor:
    chunks = []
    min_skip = max(2, int(min_skip))
    max_skip = max(min_skip, int(max_skip))
    for clip in clips:
        stop = clip.cyclic_period if clip.cyclic_animation else min(clip.T - max_skip, clip.T - cfg.future_window)
        if stop <= 1:
            continue
        idx = torch.arange(1, stop, dtype=torch.long, device=device)
        for skip in range(min_skip, max_skip + 1):
            chunks.append(transition_features_with_offset(clip, idx, skip, cfg, device).detach().cpu())
    if not chunks:
        return torch.empty((0, transition_schema(clips[0], cfg)["total_dim"]), dtype=torch.float32)
    return torch.cat(chunks, dim=0)


def normalise(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def denormalise(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std + mean


def alteration_mask(schema: dict[str, int], device: torch.device) -> torch.Tensor:
    mask = torch.ones(schema["total_dim"], device=device)
    mask[schema["input_root_start"] : schema["input_root_end"]] = 0.0
    root_lookahead_dim = int(schema.get("root_lookahead_dim", 0))
    if root_lookahead_dim > 0:
        start = int(schema["root_lookahead_start"])
        mask[start : start + root_lookahead_dim] = 0.0
    return mask


def input_noise_mask(schema: dict[str, int], mode: str, device: torch.device) -> torch.Tensor:
    mode = str(mode).lower()
    if mode in ("all", "full"):
        return torch.ones(schema["total_dim"], device=device)
    if mode in ("nonroot", "body", "body_transition"):
        return alteration_mask(schema, device)
    if mode in ("none", "off"):
        return torch.zeros(schema["total_dim"], device=device)
    raise ValueError(f"Unknown input_noise_mask={mode!r}. Use all, nonroot, or none.")


def apply_condition_body_dropout(
    x: torch.Tensor,
    schema: dict[str, int],
    probability: float,
) -> torch.Tensor:
    probability = float(max(0.0, min(1.0, probability)))
    if probability <= 0.0:
        return x
    rows = torch.rand((x.shape[0], 1), dtype=x.dtype, device=x.device) < probability
    if not bool(rows.any()):
        return x
    mask = torch.zeros((1, x.shape[-1]), dtype=torch.bool, device=x.device)
    input_dim = int(schema["input_dim"])
    mask[:, :input_dim] = True
    root_start = int(schema["input_root_start"])
    root_end = int(schema["input_root_end"])
    mask[:, root_start:root_end] = False
    y = x.clone()
    y[rows.expand_as(y) & mask.expand_as(y)] = 0.0
    return y


def make_statue_tier(x: torch.Tensor, schema: dict[str, int]) -> torch.Tensor:
    y = x.clone()
    pose_dim = schema["pose_dim"]
    input_dim = schema["input_dim"]
    output_start = schema["next_output_start"]
    canon_start = schema["next_canon_start"]
    vel_start = schema["next_velocity_start"]
    output_dim = schema["output_dim"]
    canon_dim = schema["next_canon_dim"]
    vel_dim = schema["next_velocity_dim"]

    current_pose = y[:, :pose_dim]
    if schema.get("pose_representation") == tl.IK_POSE_REPRESENTATION:
        current_output = current_pose
    else:
        current_output = torch.cat(
            (
                current_pose[:, :9],
                current_pose[:, 9 + canon_dim :],
            ),
            dim=-1,
        )
    if current_output.shape[-1] < output_dim:
        current_output = torch.cat(
            (
                current_output,
                torch.zeros(
                    (current_output.shape[0], output_dim - current_output.shape[-1]),
                    dtype=current_output.dtype,
                    device=current_output.device,
                ),
            ),
            dim=-1,
        )
    y[:, output_start : output_start + output_dim] = current_output
    if schema.get("pose_representation") != tl.IK_POSE_REPRESENTATION:
        y[:, canon_start : canon_start + canon_dim] = current_pose[:, 9 : 9 + canon_dim]
    y[:, vel_start : vel_start + vel_dim] = 0.0
    return y


def make_damped_transition_tier(x: torch.Tensor, schema: dict[str, int], factor: float) -> torch.Tensor:
    statue = make_statue_tier(x, schema)
    factor = float(max(0.0, min(1.0, factor)))
    return statue + factor * (x - statue)


def rotate_flat_vectors(values: torch.Tensor, start: int, vector_count: int, yaw_rot: torch.Tensor) -> None:
    if vector_count <= 0:
        return
    segment = values[:, start : start + vector_count * 3].reshape(values.shape[0], vector_count, 3)
    values[:, start : start + vector_count * 3] = torch.matmul(segment, yaw_rot).reshape(values.shape[0], -1)


def rotate_pelvis_rot6(values: torch.Tensor, start: int, yaw_rot: torch.Tensor) -> None:
    pelvis_rot = tl.rotation_6d_to_matrix(values[:, start : start + 6])
    values[:, start : start + 6] = tl.rotmat_to_6d(pelvis_rot @ yaw_rot)


def make_yaw_body_tier(
    x: torch.Tensor,
    schema: dict[str, int],
    yaw_degrees: float | torch.Tensor,
) -> torch.Tensor:
    """Rotate the body transition around the root vertical axis while keeping root commands unchanged.

    This creates a focused root/body compatibility negative: the local motion can still look human,
    but it no longer matches the commanded root heading. That is exactly the family of failures that
    shows up visually as ice skating on turns/circles.
    """
    y = x.clone()
    pose_dim = int(schema["pose_dim"])
    velocity_dim = int(schema["velocity_dim"])
    input_dim = int(schema["input_dim"])
    output_start = int(schema["next_output_start"])
    canon_start = int(schema["next_canon_start"])
    vel_start = int(schema["next_velocity_start"])
    output_dim = int(schema["output_dim"])
    canon_dim = int(schema["next_canon_dim"])
    next_vel_dim = int(schema["next_velocity_dim"])
    joint_count = int(canon_dim // 3)

    if isinstance(yaw_degrees, torch.Tensor):
        yaw = torch.deg2rad(yaw_degrees.to(device=x.device, dtype=x.dtype)).reshape(-1)
    else:
        yaw = torch.full((x.shape[0],), math.radians(float(yaw_degrees)), dtype=x.dtype, device=x.device)
    if yaw.numel() == 1 and x.shape[0] != 1:
        yaw = yaw.expand(x.shape[0])
    yaw_rot = tl.yaw_to_row_matrix(yaw)
    if schema.get("pose_representation") == tl.IK_POSE_REPRESENTATION:
        raise NotImplementedError("transition yaw augmentation needs an explicit ik_payload rot6-aware path")
        rotate_flat_vectors(y, vel_start, 1, yaw_rot)
        rotate_flat_vectors(y, vel_start + 3, marker_count, yaw_rot)
        _ = input_dim, velocity_dim, next_vel_dim
        return y

    current_pose = 0
    previous_pose = pose_dim
    rotate_flat_vectors(y, current_pose, 1, yaw_rot)
    rotate_flat_vectors(y, current_pose + 9, joint_count, yaw_rot)
    rotate_pelvis_rot6(y, current_pose + 3, yaw_rot)
    rotate_flat_vectors(y, previous_pose, 1, yaw_rot)
    rotate_flat_vectors(y, previous_pose + 9, joint_count, yaw_rot)
    rotate_pelvis_rot6(y, previous_pose + 3, yaw_rot)

    current_velocity = pose_dim * 2
    rotate_flat_vectors(y, current_velocity, 1, yaw_rot)
    rotate_flat_vectors(y, current_velocity + 3, joint_count, yaw_rot)

    rotate_flat_vectors(y, output_start, 1, yaw_rot)
    if output_dim >= 9:
        rotate_pelvis_rot6(y, output_start + 3, yaw_rot)
    rotate_flat_vectors(y, canon_start, joint_count, yaw_rot)
    rotate_flat_vectors(y, vel_start, 1, yaw_rot)
    rotate_flat_vectors(y, vel_start + 3, max(0, (next_vel_dim - 3) // 3), yaw_rot)

    # Keep root features and future-root lookahead untouched on purpose.
    _ = input_dim
    return y


def sample_yaw_body_negative(
    clean: torch.Tensor,
    batch_indices: torch.Tensor,
    schema: dict[str, int],
    mean: torch.Tensor,
    std: torch.Tensor,
    cfg: AEConfig,
) -> torch.Tensor:
    raw = clean.index_select(0, batch_indices)
    lo = abs(float(cfg.compatibility_yaw_min_degrees))
    hi = abs(float(cfg.compatibility_yaw_max_degrees))
    if hi < lo:
        lo, hi = hi, lo
    span = max(0.0, hi - lo)
    angle = lo + span * torch.rand((raw.shape[0],), dtype=raw.dtype, device=raw.device)
    sign = torch.where(
        torch.rand((raw.shape[0],), dtype=raw.dtype, device=raw.device) < 0.5,
        -torch.ones_like(angle),
        torch.ones_like(angle),
    )
    return normalise(make_yaw_body_tier(raw, schema, angle * sign), mean, std)


def make_tiers(x_norm: torch.Tensor, schema: dict[str, int]) -> dict[str, torch.Tensor]:
    mask = alteration_mask(schema, x_norm.device).unsqueeze(0)
    tier2 = x_norm + 0.05 * torch.randn_like(x_norm) * mask
    noisy_bad = x_norm + 0.75 * torch.randn_like(x_norm) * mask
    statue = make_statue_tier(x_norm, schema)
    yaw_body = make_yaw_body_tier(x_norm, schema, 15.0)
    tier3 = torch.where((torch.arange(x_norm.shape[0], device=x_norm.device)[:, None] % 2) == 0, statue, noisy_bad)
    tier4 = torch.randn_like(x_norm)
    tier4[:, schema["input_root_start"] : schema["input_root_end"]] = x_norm[
        :, schema["input_root_start"] : schema["input_root_end"]
    ]
    return {
        "tier1_clean": x_norm,
        "tier2_slight": tier2,
        "tier3_bad": tier3,
        "tier3_yaw_body": yaw_body,
        "tier4_noise": tier4,
    }


@torch.no_grad()
def reconstruction_errors(model: nn.Module, x_norm: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
    values = []
    model.eval()
    for start in range(0, x_norm.shape[0], batch_size):
        batch = x_norm[start : start + batch_size]
        recon = model(batch)
        target = model.target(batch) if hasattr(model, "target") else batch
        values.append(reconstruction_loss_rows(model, recon, target, loss_type="mse"))
    return torch.cat(values, dim=0)


@torch.no_grad()
def tier_report(model: nn.Module, x_norm: torch.Tensor, schema: dict[str, int]) -> dict[str, float]:
    tiers = make_tiers(x_norm, schema)
    report: dict[str, float] = {}
    for name, values in tiers.items():
        err = reconstruction_errors(model, values)
        report[f"{name}_mean"] = float(err.mean().cpu())
        report[f"{name}_p95"] = float(torch.quantile(err, 0.95).cpu())
    return report


def save_tier_report(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sample_direction_negatives(
    x_norm: torch.Tensor,
    clip_ids: torch.Tensor,
    batch_indices: torch.Tensor,
    schema: dict[str, int],
) -> torch.Tensor:
    batch = x_norm.index_select(0, batch_indices)
    batch_clip_ids = clip_ids.index_select(0, batch_indices)
    body_indices = torch.randint(0, x_norm.shape[0], batch_indices.shape, device=x_norm.device)
    for _ in range(16):
        same_clip = clip_ids.index_select(0, body_indices) == batch_clip_ids
        if not bool(same_clip.any()):
            break
        body_indices[same_clip] = torch.randint(0, x_norm.shape[0], (int(same_clip.sum().item()),), device=x_norm.device)
    negative = x_norm.index_select(0, body_indices).clone()
    root_start = schema["input_root_start"]
    root_end = schema["input_root_end"]
    negative[:, root_start:root_end] = batch[:, root_start:root_end]
    root_lookahead_dim = int(schema.get("root_lookahead_dim", 0))
    if root_lookahead_dim > 0:
        lookahead_start = int(schema["root_lookahead_start"])
        lookahead_end = lookahead_start + root_lookahead_dim
        negative[:, lookahead_start:lookahead_end] = batch[:, lookahead_start:lookahead_end]
    return negative


def compatibility_bce_loss(
    model: TransitionAutoencoder,
    positive: torch.Tensor,
    direction_negative: torch.Tensor | None,
    temporal_negative: torch.Tensor | None,
    statue_negative: torch.Tensor | None,
    damped_negative: torch.Tensor | None,
    yaw_body_negative: torch.Tensor | None,
    cfg: AEConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = positive.new_zeros(())
    if not model.has_compatibility_head() or cfg.compatibility_weight <= 0.0:
        return zero, {
            "compat_pos_loss": zero,
            "compat_direction_loss": zero,
            "compat_temporal_loss": zero,
            "compat_statue_loss": zero,
            "compat_damped_loss": zero,
            "compat_yaw_body_loss": zero,
            "compat_pos_acc": zero,
            "compat_direction_acc": zero,
            "compat_temporal_acc": zero,
            "compat_statue_acc": zero,
            "compat_damped_acc": zero,
            "compat_yaw_body_acc": zero,
        }

    pos_logits = model.compatibility_logits(positive)
    pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
    total = pos_loss
    direction_loss = zero
    temporal_loss = zero
    statue_loss = zero
    damped_loss = zero
    direction_acc = zero
    temporal_acc = zero
    statue_acc = zero
    damped_acc = zero
    yaw_body_loss = zero
    yaw_body_acc = zero
    if direction_negative is not None and cfg.compatibility_direction_weight > 0.0:
        direction_logits = model.compatibility_logits(direction_negative)
        direction_loss = F.binary_cross_entropy_with_logits(direction_logits, torch.zeros_like(direction_logits))
        total = total + cfg.compatibility_direction_weight * direction_loss
        direction_acc = (direction_logits < 0.0).float().mean()
    if temporal_negative is not None and cfg.compatibility_temporal_weight > 0.0:
        temporal_logits = model.compatibility_logits(temporal_negative)
        temporal_loss = F.binary_cross_entropy_with_logits(temporal_logits, torch.zeros_like(temporal_logits))
        total = total + cfg.compatibility_temporal_weight * temporal_loss
        temporal_acc = (temporal_logits < 0.0).float().mean()
    if statue_negative is not None and cfg.compatibility_statue_weight > 0.0:
        statue_logits = model.compatibility_logits(statue_negative)
        statue_loss = F.binary_cross_entropy_with_logits(statue_logits, torch.zeros_like(statue_logits))
        total = total + cfg.compatibility_statue_weight * statue_loss
        statue_acc = (statue_logits < 0.0).float().mean()
    if damped_negative is not None and cfg.compatibility_damped_weight > 0.0:
        damped_logits = model.compatibility_logits(damped_negative)
        damped_loss = F.binary_cross_entropy_with_logits(damped_logits, torch.zeros_like(damped_logits))
        total = total + cfg.compatibility_damped_weight * damped_loss
        damped_acc = (damped_logits < 0.0).float().mean()
    if yaw_body_negative is not None and cfg.compatibility_yaw_body_weight > 0.0:
        yaw_body_logits = model.compatibility_logits(yaw_body_negative)
        yaw_body_loss = F.binary_cross_entropy_with_logits(yaw_body_logits, torch.zeros_like(yaw_body_logits))
        total = total + cfg.compatibility_yaw_body_weight * yaw_body_loss
        yaw_body_acc = (yaw_body_logits < 0.0).float().mean()
    return total, {
        "compat_pos_loss": pos_loss.detach(),
        "compat_direction_loss": direction_loss.detach(),
        "compat_temporal_loss": temporal_loss.detach(),
        "compat_statue_loss": statue_loss.detach(),
        "compat_damped_loss": damped_loss.detach(),
        "compat_yaw_body_loss": yaw_body_loss.detach(),
        "compat_pos_acc": (pos_logits > 0.0).float().mean().detach(),
        "compat_direction_acc": direction_acc.detach(),
        "compat_temporal_acc": temporal_acc.detach(),
        "compat_statue_acc": statue_acc.detach(),
        "compat_damped_acc": damped_acc.detach(),
        "compat_yaw_body_acc": yaw_body_acc.detach(),
    }


def train(args: argparse.Namespace) -> None:
    cfg = AEConfig()
    for field in cfg.__dataclass_fields__:
        value = getattr(args, field, None)
        if value is not None:
            setattr(cfg, field, value)
    if cfg.date_prefix_run_name:
        cfg.run_name = tl.date_prefixed_run_name(cfg.run_name)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    tl.apply_cuda_performance_settings(tl.TrainConfig(device=cfg.device), device)
    locomotion_cfg = tl.TrainConfig()
    locomotion_cfg.cyclic_animation = cfg.cyclic_animation
    locomotion_cfg.include_transition_foot_motion = bool(cfg.include_transition_foot_motion)
    locomotion_cfg.foot_slide_scale_mps = float(cfg.foot_slide_scale_mps)
    locomotion_cfg.transition_yaw_scale_radps = float(cfg.transition_yaw_scale_radps)
    locomotion_cfg.root_lookahead_steps = max(0, int(cfg.root_lookahead_steps))
    locomotion_cfg.pose_representation = str(cfg.pose_representation)
    clip_specs = tl.clip_specs_from_folders(
        cfg.folder_path,
        cfg.periodic_folder_path or None,
        cfg.nonperiodic_folder_path or None,
    )
    clips = tl.load_clips_from_specs(clip_specs, locomotion_cfg)
    clean, feature_clip_ids, _feature_cur_idx = collect_clean_feature_rows(clips, locomotion_cfg, device)
    skip_features = None
    if cfg.compatibility_weight > 0.0 and cfg.compatibility_temporal_weight > 0.0:
        skip_features = collect_temporal_skip_features(
            clips,
            locomotion_cfg,
            device,
            cfg.compatibility_temporal_min_skip,
            cfg.compatibility_temporal_max_skip,
        )
    mean = clean.mean(dim=0)
    std = clean.std(dim=0).clamp_min(cfg.std_floor)
    schema = transition_schema(clips[0], locomotion_cfg)
    x_norm = normalise(clean, mean, std).to(device)
    clean_for_yaw_negative = clean.to(device) if cfg.compatibility_yaw_body_weight > 0.0 else clean
    clip_ids = feature_clip_ids.to(device)
    sample_mode = str(cfg.sample_mode).lower().strip()
    if sample_mode not in ("rows", "uniform_clip"):
        raise ValueError("--sample-mode must be 'rows' or 'uniform_clip'")
    skip_norm = normalise(skip_features, mean, std).to(device) if skip_features is not None else None
    statue_norm = None
    if (
        (cfg.compatibility_weight > 0.0 and cfg.compatibility_statue_weight > 0.0)
        or cfg.structured_denoise_statue_weight > 0.0
    ):
        statue_norm = normalise(make_statue_tier(clean, schema), mean, std).to(device)
    damped_norm = None
    if (
        (cfg.compatibility_weight > 0.0 and cfg.compatibility_damped_weight > 0.0)
        or cfg.structured_denoise_damped_weight > 0.0
    ):
        denoise_factor = (
            cfg.structured_denoise_damped_factor
            if cfg.structured_denoise_damped_weight > 0.0
            else cfg.compatibility_damped_factor
        )
        damped_norm = normalise(make_damped_transition_tier(clean, schema, denoise_factor), mean, std).to(device)
    mean = mean.to(device)
    std = std.to(device)
    noise_mask = input_noise_mask(schema, cfg.input_noise_mask, device).unsqueeze(0)

    run_dir = tl.resolve_path(cfg.output_dir) / cfg.run_name
    ckpt_dir = run_dir / "checkpoints"
    writer = SummaryWriter(run_dir / "tb")
    print(f"transition_ae run_dir={run_dir}", flush=True)
    print(f"transition_ae best_checkpoint={ckpt_dir / 'checkpoint_best.pt'}", flush=True)
    model = TransitionAutoencoder(schema["total_dim"], cfg, schema).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    print(
        f"transition_ae samples={x_norm.shape[0]} dim={x_norm.shape[1]} latent={cfg.latent_dim} "
        f"compat={cfg.compatibility_weight:g} sample_mode={sample_mode} device={device} "
        f"folders={[(str(tl.npz_folder_from_path(path)), cyclic) for path, cyclic in clip_specs]}",
        flush=True,
    )
    best = math.inf
    baseline = None
    target = None
    stalls = 0
    rows: list[dict[str, float]] = []
    indices = torch.arange(x_norm.shape[0], device=device)
    start_time = time.perf_counter()
    timed_interval_seconds = 60.0 * max(0.0, float(cfg.timed_checkpoint_interval_minutes))
    next_timed_checkpoint_at = start_time + timed_interval_seconds if timed_interval_seconds > 0.0 else math.inf
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        if sample_mode == "uniform_clip":
            perm = balanced_clip_epoch_indices(clip_ids, indices.numel(), device)
        else:
            perm = indices[torch.randperm(indices.numel(), device=device)]
        loss_sum = torch.zeros((), device=device)
        recon_sum = torch.zeros((), device=device)
        compat_sum = torch.zeros((), device=device)
        structured_statue_sum = torch.zeros((), device=device)
        structured_damped_sum = torch.zeros((), device=device)
        compat_parts_sum: dict[str, torch.Tensor] = {}
        loss_count = 0
        for start in range(0, perm.numel(), cfg.batch_size):
            batch_indices = perm[start : start + cfg.batch_size]
            batch = x_norm.index_select(0, batch_indices)
            noisy = batch
            if cfg.input_noise_std > 0.0:
                noisy = batch + cfg.input_noise_std * torch.randn_like(batch) * noise_mask
            if cfg.conditional_root_window and cfg.condition_body_dropout_prob > 0.0:
                noisy = apply_condition_body_dropout(noisy, schema, cfg.condition_body_dropout_prob)
            recon = model(noisy)
            clean_target = model.target(batch)
            recon_loss = reconstruction_loss(model, recon, clean_target, loss_type="huber")
            structured_statue_loss = torch.zeros((), device=device)
            if statue_norm is not None and cfg.structured_denoise_statue_weight > 0.0:
                statue_input = statue_norm.index_select(0, batch_indices)
                structured_statue_loss = reconstruction_loss(
                    model,
                    model(statue_input),
                    clean_target,
                    loss_type="huber",
                )
            structured_damped_loss = torch.zeros((), device=device)
            if damped_norm is not None and cfg.structured_denoise_damped_weight > 0.0:
                damped_input = damped_norm.index_select(0, batch_indices)
                structured_damped_loss = reconstruction_loss(
                    model,
                    model(damped_input),
                    clean_target,
                    loss_type="huber",
                )
            compat_loss = torch.zeros((), device=device)
            compat_parts: dict[str, torch.Tensor] = {}
            if cfg.compatibility_weight > 0.0:
                direction_negative = sample_direction_negatives(x_norm, clip_ids, batch_indices, schema)
                temporal_negative = None
                if skip_norm is not None:
                    skip_indices = torch.randint(0, skip_norm.shape[0], batch_indices.shape, device=device)
                    temporal_negative = skip_norm.index_select(0, skip_indices)
                statue_negative = None
                if statue_norm is not None:
                    statue_negative = statue_norm.index_select(0, batch_indices)
                damped_negative = None
                if damped_norm is not None:
                    damped_negative = damped_norm.index_select(0, batch_indices)
                yaw_body_negative = None
                if cfg.compatibility_yaw_body_weight > 0.0:
                    yaw_body_negative = sample_yaw_body_negative(clean_for_yaw_negative, batch_indices, schema, mean, std, cfg)
                compat_loss, compat_parts = compatibility_bce_loss(
                    model,
                    batch,
                    direction_negative,
                    temporal_negative,
                    statue_negative,
                    damped_negative,
                    yaw_body_negative,
                    cfg,
                )
            loss = (
                recon_loss
                + cfg.structured_denoise_statue_weight * structured_statue_loss
                + cfg.structured_denoise_damped_weight * structured_damped_loss
                + cfg.compatibility_weight * compat_loss
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum = loss_sum + loss.detach()
            recon_sum = recon_sum + recon_loss.detach()
            structured_statue_sum = structured_statue_sum + structured_statue_loss.detach()
            structured_damped_sum = structured_damped_sum + structured_damped_loss.detach()
            compat_sum = compat_sum + compat_loss.detach()
            for key, value in compat_parts.items():
                compat_parts_sum[key] = compat_parts_sum.get(key, torch.zeros((), device=device)) + value
            loss_count += 1
        train_loss = float((loss_sum / max(1, loss_count)).cpu())
        train_recon = float((recon_sum / max(1, loss_count)).cpu())
        train_structured_statue = float((structured_statue_sum / max(1, loss_count)).cpu())
        train_structured_damped = float((structured_damped_sum / max(1, loss_count)).cpu())
        train_compat = float((compat_sum / max(1, loss_count)).cpu())
        if baseline is None:
            baseline = train_loss
            target = baseline * (1.0 - cfg.target_loss_reduction)
        improved = train_loss < best - cfg.min_delta
        stalls = 0 if improved else stalls + 1
        if train_loss < best:
            best = train_loss
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(cfg),
                    "locomotion_config": asdict(locomotion_cfg),
                    "schema": schema,
                    "mean": mean.detach().cpu(),
                    "std": std.detach().cpu(),
                    "metadata": {
                        "npz_folders": [
                            {"path": str(tl.npz_folder_from_path(path)), "cyclic": cyclic}
                            for path, cyclic in clip_specs
                        ],
                        "body_names": clips[0].body_names,
                        "parents_body": clips[0].parents_body.tolist(),
                    },
                    "epoch": epoch,
                    "best": best,
                },
                tl.ik_checkpoint_path(ckpt_dir / "checkpoint_best.pt", cfg.run_name),
            )
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/reconstruction", train_recon, epoch)
        writer.add_scalar("loss/structured_denoise_statue", train_structured_statue, epoch)
        writer.add_scalar("loss/structured_denoise_damped", train_structured_damped, epoch)
        writer.add_scalar("loss/compatibility", train_compat, epoch)
        writer.add_scalar("loss/best", best, epoch)
        for key, value in compat_parts_sum.items():
            writer.add_scalar(f"compatibility/{key}", float((value / max(1, loss_count)).cpu()), epoch)
        if epoch == 1 or epoch % cfg.tier_eval_every_epochs == 0:
            report = tier_report(model, x_norm, schema)
            row = {"epoch": epoch, "train_loss": train_loss, "best": best, **report}
            rows.append(row)
            save_tier_report(run_dir / "tier_report.csv", rows)
            for key, value in report.items():
                writer.add_scalar(f"tiers/{key}", value, epoch)
            print(
                "epoch={:04d} loss={:.6g} best={:.6g} tiers clean={:.6g} slight={:.6g} bad={:.6g} noise={:.6g}".format(
                    epoch,
                    train_loss,
                    best,
                    report["tier1_clean_mean"],
                    report["tier2_slight_mean"],
                    report["tier3_bad_mean"],
                    report["tier4_noise_mean"],
                ),
                flush=True,
            )
        elif epoch % 10 == 0:
            print(
                f"epoch={epoch:04d} loss={train_loss:.6g} recon={train_recon:.6g} "
                f"struct_statue={train_structured_statue:.6g} struct_damped={train_structured_damped:.6g} "
                f"compat={train_compat:.6g} best={best:.6g} stalls={stalls}",
                flush=True,
            )
        if target is not None and train_loss <= target:
            print(f"target reached epoch={epoch} loss={train_loss:.6g} target={target:.6g}", flush=True)
            break
        now_perf = time.perf_counter()
        if now_perf >= next_timed_checkpoint_at:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(cfg),
                    "locomotion_config": asdict(locomotion_cfg),
                    "schema": schema,
                    "mean": mean.detach().cpu(),
                    "std": std.detach().cpu(),
                    "metadata": {
                        "npz_folders": [
                            {"path": str(tl.npz_folder_from_path(path)), "cyclic": cyclic}
                            for path, cyclic in clip_specs
                        ],
                        "body_names": clips[0].body_names,
                        "parents_body": clips[0].parents_body.tolist(),
                    },
                    "epoch": epoch,
                    "best": best,
                },
                tl.ik_checkpoint_path(ckpt_dir / f"checkpoint_time_{stamp}_epoch_{epoch:06d}.pt", cfg.run_name),
            )
            while next_timed_checkpoint_at <= now_perf:
                next_timed_checkpoint_at += timed_interval_seconds
        if cfg.stall_patience_epochs > 0 and stalls >= cfg.stall_patience_epochs:
            print(f"stopped on stall epoch={epoch} best={best:.6g}", flush=True)
            break
    final_report = tier_report(model, x_norm, schema)
    rows.append({"epoch": epoch, "train_loss": train_loss, "best": best, **final_report})
    save_tier_report(run_dir / "tier_report.csv", rows)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    for name, field in AEConfig.__dataclass_fields__.items():
        default = field.default
        arg = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(arg, action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument(arg, type=type(default), default=None)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
