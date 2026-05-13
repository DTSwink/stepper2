from __future__ import annotations

import argparse

import torch

import train_locomotion as tl
import transition_autoencoder as base


AEConfig = base.AEConfig
TransitionAutoencoder = base.TransitionAutoencoder


def transition_schema(clip: tl.MotionClip, cfg: tl.TrainConfig) -> dict[str, int]:
    velocity_dim = 3 + clip.J * 3
    contact_dim = 2
    root_dim = 3
    future_dim = cfg.future_window * 4
    rot_delta_dim = clip.J * 6
    current_contact_start = velocity_dim
    root_start = current_contact_start + contact_dim
    next_velocity_start = root_start + root_dim + future_dim
    next_contact_start = next_velocity_start + velocity_dim
    rot_delta_start = next_contact_start + contact_dim
    return {
        "velocity_dim": velocity_dim,
        "contact_dim": contact_dim,
        "root_dim": root_dim,
        "future_dim": future_dim,
        "next_velocity_dim": velocity_dim,
        "rot_delta_dim": rot_delta_dim,
        "current_velocity_start": 0,
        "current_contact_start": current_contact_start,
        "root_start": root_start,
        "root_end": root_start + root_dim + future_dim,
        "next_velocity_start": next_velocity_start,
        "next_contact_start": next_contact_start,
        "rot_delta_start": rot_delta_start,
        "total_dim": rot_delta_start + rot_delta_dim,
    }


def local_rotation_delta_6d(cur_pose: dict[str, torch.Tensor], next_pose: dict[str, torch.Tensor]) -> torch.Tensor:
    b = cur_pose["pelvis_rot6"].shape[0]
    cur_pelvis = tl.rotation_6d_to_matrix(cur_pose["pelvis_rot6"])
    next_pelvis = tl.rotation_6d_to_matrix(next_pose["pelvis_rot6"])
    pelvis_delta = next_pelvis @ cur_pelvis.transpose(-1, -2)

    cur_non = tl.rotation_6d_to_matrix(cur_pose["nonpelvis_rot6"].reshape(-1, 6)).reshape(b, -1, 3, 3)
    next_non = tl.rotation_6d_to_matrix(next_pose["nonpelvis_rot6"].reshape(-1, 6)).reshape(b, -1, 3, 3)
    non_delta = next_non @ cur_non.transpose(-1, -2)
    return torch.cat(
        (
            tl.rotmat_to_6d(pelvis_delta),
            tl.rotmat_to_6d(non_delta).reshape(b, -1),
        ),
        dim=-1,
    )


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
    b = cur_idx.shape[0]
    cur_pelvis_vel = (cur_pose["pelvis_pos"] - prev_pose["pelvis_pos"]) / cfg.pose_delta_scale_final
    cur_joint_vel = (cur_pose["canon_pos"] - prev_pose["canon_pos"]).reshape(b, -1) / cfg.pose_delta_scale_final
    next_pelvis_vel = (next_pose["pelvis_pos"] - cur_pose["pelvis_pos"]) / cfg.pose_delta_scale_final
    next_joint_vel = (next_pose["canon_pos"] - cur_pose["canon_pos"]).reshape(b, -1) / cfg.pose_delta_scale_final
    root_feat = tl.root_delta_feature(clip, prev_idx, cur_idx, cfg, device)
    future_feat = tl.future_root_features(clip, cur_idx, cfg, device)
    rot_delta = local_rotation_delta_6d(cur_pose, next_pose)
    return torch.cat(
        (
            cur_pelvis_vel,
            cur_joint_vel,
            cur_pose["contacts"],
            root_feat,
            future_feat,
            next_pelvis_vel,
            next_joint_vel,
            next_pose["contacts"],
            rot_delta,
        ),
        dim=-1,
    )


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


def collect_clean_features(clips: list[tl.MotionClip], cfg: tl.TrainConfig, device: torch.device) -> torch.Tensor:
    chunks = []
    for clip in clips:
        stop = clip.cyclic_period if cfg.cyclic_animation else clip.T - 1
        idx = torch.arange(1, stop, dtype=torch.long, device=device)
        chunks.append(clean_transition_features(clip, idx, cfg, device).detach().cpu())
    return torch.cat(chunks, dim=0)


def alteration_mask(schema: dict[str, int], device: torch.device) -> torch.Tensor:
    mask = torch.ones(schema["total_dim"], device=device)
    mask[schema["root_start"] : schema["root_end"]] = 0.0
    return mask


def make_tiers(x_norm: torch.Tensor, schema: dict[str, int]) -> dict[str, torch.Tensor]:
    mask = alteration_mask(schema, x_norm.device).unsqueeze(0)
    tier2 = x_norm + 0.05 * torch.randn_like(x_norm) * mask
    tier3 = x_norm + 0.75 * torch.randn_like(x_norm) * mask
    vel_start = schema["next_velocity_start"]
    next_contact_start = schema["next_contact_start"]
    rot_start = schema["rot_delta_start"]
    statue = x_norm.clone()
    statue[:, vel_start:rot_start] = 0.0
    statue[:, next_contact_start:rot_start] = x_norm[:, schema["current_contact_start"] : schema["current_contact_start"] + schema["contact_dim"]]
    statue[:, rot_start:] = 0.0
    choose_statue = (torch.arange(x_norm.shape[0], device=x_norm.device)[:, None] % 2) == 0
    tier3 = torch.where(choose_statue, statue, tier3)
    tier4 = torch.randn_like(x_norm)
    tier4[:, schema["root_start"] : schema["root_end"]] = x_norm[:, schema["root_start"] : schema["root_end"]]
    return {
        "tier1_clean": x_norm,
        "tier2_slight": tier2,
        "tier3_bad": tier3,
        "tier4_noise": tier4,
    }


def train(args: argparse.Namespace) -> None:
    old_transition_schema = base.transition_schema
    old_transition_feature_from_next_pose = base.transition_feature_from_next_pose
    old_clean_transition_features = base.clean_transition_features
    old_collect_clean_features = base.collect_clean_features
    old_alteration_mask = base.alteration_mask
    old_make_tiers = base.make_tiers
    try:
        base.transition_schema = transition_schema
        base.transition_feature_from_next_pose = transition_feature_from_next_pose
        base.clean_transition_features = clean_transition_features
        base.collect_clean_features = collect_clean_features
        base.alteration_mask = alteration_mask
        base.make_tiers = make_tiers
        base.train(args)
    finally:
        base.transition_schema = old_transition_schema
        base.transition_feature_from_next_pose = old_transition_feature_from_next_pose
        base.clean_transition_features = old_clean_transition_features
        base.collect_clean_features = old_collect_clean_features
        base.alteration_mask = old_alteration_mask
        base.make_tiers = old_make_tiers


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
