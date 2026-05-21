from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import contact_physics as cp
import train_locomotion as tl
import train_locomotion_ae_prior as ae_prior
import transition_autoencoder as tae


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class LabMetrics:
    ae_mean: float
    target_pose_mse: float
    final_joint_rmse_m: float
    final_ee_rmse_m: float
    contact_slide_mean_mps: float
    contact_slide_p95_mps: float
    both_fast_rate: float
    both_fast_max_mps: float
    both_ground_sliding_rate: float
    both_ground_sliding_max_mps: float
    planted_available_rate: float
    both_airborne_rate: float
    dominance_mean: float
    dominance_min: float
    dominant_switches: float
    solo_motion_rate: float
    left_solo_rate: float
    right_solo_rate: float
    left_path_m: float
    right_path_m: float
    left_swing_path_m: float
    right_swing_path_m: float
    left_lift_max_m: float
    right_lift_max_m: float
    success: float


def resolve(path: str | Path) -> Path:
    return tl.resolve_path(str(path))


def apply_checkpoint_locomotion_config(cfg: tl.TrainConfig, checkpoint: dict) -> None:
    valid = set(tl.TrainConfig.__dataclass_fields__.keys())
    for key, value in checkpoint.get("locomotion_config", {}).items():
        if key in valid:
            setattr(cfg, key, value)


def repeat_pose(pose: dict[str, torch.Tensor], count: int) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in pose.items():
        if key.endswith("_mat"):
            continue
        out[key] = value.repeat((count,) + (1,) * (value.ndim - 1))
    return out


def select_pose_from_clip(clip: tl.MotionClip, idx: int, device: torch.device, count: int = 1) -> dict[str, torch.Tensor]:
    pose = tl.get_pose_from_clip(clip, torch.full((1,), int(idx), dtype=torch.long, device=device), device)
    return repeat_pose(pose, count)


def transition_prior_score(
    priors: list[dict[str, object]],
    clip: tl.MotionClip,
    prev_idx: torch.Tensor,
    cur_idx: torch.Tensor,
    prev_pose: dict[str, torch.Tensor],
    cur_pose: dict[str, torch.Tensor],
    next_pose: dict[str, torch.Tensor],
    cfg: tl.TrainConfig,
    device: torch.device,
    loss_type: str,
    huber_delta: float,
) -> torch.Tensor:
    rows = []
    for prior_info in priors:
        if prior_info.get("kind") == "window":
            continue
        prior_cfg = ae_prior.prior_transition_cfg(prior_info, cfg)
        features = tae.transition_feature_from_next_pose(
            clip,
            prev_idx,
            cur_idx,
            prev_pose,
            cur_pose,
            next_pose,
            prior_cfg,
            device,
        )
        score = ae_prior.ae_score_rows(
            prior_info["model"],
            prior_info["mean"],
            prior_info["std"],
            features,
            loss_type,
            huber_delta,
        )
        rows.append(score * float(prior_info.get("weight", 1.0)))
    if not rows:
        return torch.zeros((cur_idx.shape[0],), dtype=torch.float32, device=device)
    return torch.stack(rows, dim=0).sum(dim=0)


def foot_motion_terms(
    cur_pos: torch.Tensor,
    cur_rot: torch.Tensor,
    next_pos: torch.Tensor,
    next_rot: torch.Tensor,
    foot_indices: tuple[int, int],
    toe_indices: tuple[int, int],
    fps: float,
    height_threshold_m: float,
    contact_temperature_m: float,
    both_speed_threshold_mps: float,
    loss_kind: str,
    contact_slide_loss_weight: float,
    both_fast_loss_weight: float,
    speed_product_loss_weight: float,
    speed_product_scale_mps: float,
    both_ground_product_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    speeds = cp.foot_slide_speeds(cur_pos, cur_rot, next_pos, next_rot, foot_indices, toe_indices, fps)
    cur_h, _cur_points = cp.foot_lowest_heights_and_points(cur_pos, cur_rot, foot_indices, toe_indices)
    next_h, next_points = cp.foot_lowest_heights_and_points(next_pos, next_rot, foot_indices, toe_indices)
    contact_h = torch.minimum(cur_h, next_h)
    temperature = max(float(contact_temperature_m), 1e-5)
    soft_contact = torch.sigmoid((float(height_threshold_m) - contact_h) / temperature)
    contact_slide = soft_contact * speeds
    smooth_both = -0.03 * torch.logsumexp(-speeds / 0.03, dim=-1)
    both_fast = F.relu(smooth_both - float(both_speed_threshold_mps))
    product_scale = max(float(speed_product_scale_mps), 1e-5)
    speed_product = (speeds[..., 0] * speeds[..., 1]) / (product_scale * product_scale)
    both_contact = soft_contact[..., 0] * soft_contact[..., 1]
    planted_available = soft_contact.max(dim=-1).values
    no_planted = F.relu(0.75 - planted_available)
    plant_logits = (float(height_threshold_m) - contact_h) / temperature
    plant_weights = torch.softmax(plant_logits, dim=-1)
    planted_slide = (plant_weights * speeds.square()).sum(dim=-1)
    contact_slide_loss = contact_slide.square().sum(dim=-1)
    if loss_kind == "planted_lowest":
        selected, _planted, _heights = cp.planted_foot_values(speeds, cur_pos, cur_rot, foot_indices, toe_indices)
        loss = selected.square()
    elif loss_kind == "soft_contact":
        loss = contact_slide_loss_weight * contact_slide_loss + both_fast_loss_weight * both_fast.square() + no_planted.square()
    elif loss_kind == "one_planted_soft":
        loss = planted_slide + contact_slide_loss_weight * contact_slide_loss + both_fast_loss_weight * both_fast.square() + no_planted.square()
    else:
        raise ValueError(f"unknown slide loss kind: {loss_kind}")
    loss = (
        loss
        + float(speed_product_loss_weight) * speed_product.square()
        + float(both_ground_product_loss_weight) * both_contact * speed_product.square()
    )
    return loss, {
        "speeds": speeds,
        "heights": next_h,
        "points": next_points,
        "soft_contact": soft_contact,
        "contact_slide": contact_slide,
        "both_fast": both_fast,
        "speed_product": speed_product,
        "both_contact": both_contact,
        "planted_available": planted_available,
        "plant_weights": plant_weights,
    }


def target_losses(
    pred_pose: dict[str, torch.Tensor],
    pred_global_pos: torch.Tensor,
    pred_global_rot: torch.Tensor,
    target_pose: dict[str, torch.Tensor],
    target_global_pos: torch.Tensor,
    target_global_rot: torch.Tensor,
    clip: tl.MotionClip,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pose_mse = F.mse_loss(tl.pose_target_output(pred_pose), tl.pose_target_output(target_pose), reduction="none")
    pose_mse_rows = pose_mse.mean(dim=-1)
    joint_sq = (pred_global_pos - target_global_pos).square().sum(dim=-1)
    joint_rmse_rows = joint_sq.mean(dim=-1).sqrt()
    joint_weights = torch.ones((pred_global_pos.shape[1],), dtype=pred_global_pos.dtype, device=pred_global_pos.device)
    foot_toe = torch.tensor(
        [int(clip.foot_indices[0]), int(clip.foot_indices[1]), int(clip.toe_indices[0]), int(clip.toe_indices[1])],
        dtype=torch.long,
        device=pred_global_pos.device,
    )
    joint_weights.index_fill_(0, foot_toe, float(args.target_foot_global_weight))
    weighted_joint_rmse_rows = (
        (joint_sq * joint_weights.unsqueeze(0)).sum(dim=-1) / joint_weights.sum().clamp_min(1e-6)
    ).sqrt()
    ee = clip.end_effectors_tensor.to(pred_global_pos.device)
    ee_rmse_rows = (
        pred_global_pos.index_select(1, ee) - target_global_pos.index_select(1, ee)
    ).square().sum(dim=-1).mean(dim=-1).sqrt()
    rot_angles = tl.geodesic_angles(
        pred_global_rot.reshape(-1, 3, 3),
        target_global_rot.reshape(-1, 3, 3),
    ).reshape(pred_global_pos.shape[0], -1)
    rot_weights = torch.ones((pred_global_rot.shape[1],), dtype=pred_global_rot.dtype, device=pred_global_rot.device)
    rot_weights.index_fill_(0, foot_toe, float(args.target_foot_rot_weight))
    rot_rows = (rot_angles * rot_weights.unsqueeze(0)).sum(dim=-1) / rot_weights.sum().clamp_min(1e-6)
    target_rows = (
        float(args.target_pose_component_weight) * pose_mse_rows
        + float(args.target_joint_component_weight) * weighted_joint_rmse_rows
        + float(args.target_rot_component_weight) * rot_rows
    )
    return target_rows, {
        "pose_mse": pose_mse_rows,
        "joint_rmse": joint_rmse_rows,
        "weighted_joint_rmse": weighted_joint_rmse_rows,
        "ee_rmse": ee_rmse_rows,
    }


def rollout(
    model: torch.nn.Module,
    priors: list[dict[str, object]],
    target_clip: tl.MotionClip,
    init_clips: list[tl.MotionClip],
    init_pairs: list[tuple[int, int]],
    cfg: tl.TrainConfig,
    device: torch.device,
    rollout_k: int,
    args: argparse.Namespace,
    train: bool,
) -> tuple[torch.Tensor, LabMetrics, dict[str, np.ndarray]]:
    batch = len(init_pairs)
    target_prev_idx = torch.zeros((batch,), dtype=torch.long, device=device)
    target_cur_idx = torch.zeros((batch,), dtype=torch.long, device=device)
    prev_pose_parts: dict[str, list[torch.Tensor]] = {}
    cur_pose_parts: dict[str, list[torch.Tensor]] = {}
    for clip_i, frame_i in init_pairs:
        init_clip = init_clips[int(clip_i)]
        frame = int(frame_i)
        prev = tl.get_pose_from_clip(init_clip, torch.tensor([frame], dtype=torch.long, device=device), device)
        cur = tl.get_pose_from_clip(init_clip, torch.tensor([frame], dtype=torch.long, device=device), device)
        for key, value in repeat_pose(prev, 1).items():
            prev_pose_parts.setdefault(key, []).append(value)
        for key, value in repeat_pose(cur, 1).items():
            cur_pose_parts.setdefault(key, []).append(value)
    prev_pose = {key: torch.cat(values, dim=0) for key, values in prev_pose_parts.items()}
    cur_pose = {key: torch.cat(values, dim=0) for key, values in cur_pose_parts.items()}
    init_root_pos, init_root_rot, _init_yaw, _init_heading = tl.root_state(target_clip, target_cur_idx, cfg, device)
    init_global_pos, init_global_rot, _init_canon = tl.fk_from_pose(
        target_clip,
        init_root_pos,
        init_root_rot,
        cur_pose,
        device,
    )
    foot_pairs = torch.tensor(
        [
            [int(target_clip.foot_indices[0]), int(target_clip.toe_indices[0])],
            [int(target_clip.foot_indices[1]), int(target_clip.toe_indices[1])],
        ],
        dtype=torch.long,
        device=device,
    )
    init_heights, _init_points = cp.foot_lowest_heights_and_points(
        init_global_pos,
        init_global_rot,
        tuple(target_clip.foot_indices),
        tuple(target_clip.toe_indices),
    )
    target0_global_pos, _target0_global_rot = tl.global_from_clip(target_clip, target_cur_idx, cfg, device)
    pair_idx = foot_pairs.reshape(-1)
    init_pair_pos = init_global_pos.index_select(1, pair_idx).reshape(batch, 2, 2, 3)
    target_pair_pos = target0_global_pos.index_select(1, pair_idx).reshape(batch, 2, 2, 3)
    foot_target_distance = torch.linalg.norm(init_pair_pos - target_pair_pos, dim=-1).mean(dim=-1)
    first_mover = (foot_target_distance + 0.25 * init_heights.clamp_min(0.0)).argmax(dim=-1)

    total = torch.zeros((), dtype=torch.float32, device=device)
    ae_rows_all = []
    target_pose_rows_all = []
    joint_rows_all = []
    ee_rows_all = []
    contact_slide_all = []
    both_fast_all = []
    speed_all = []
    height_all = []
    point_all = []
    contact_all = []

    for _step in range(int(rollout_k)):
        inp = tl.build_input(target_clip, target_prev_idx, target_cur_idx, prev_pose, cur_pose, cfg, device)
        raw = tl.predict_next_raw(model, inp, cur_pose, cfg)
        pred_pose, _raw_pose = tl.output_to_pose(raw, target_clip)
        target_next_idx = target_cur_idx + 1
        root_pos, root_rot, _yaw, _heading = tl.root_state(target_clip, target_next_idx, cfg, device)
        pred_global_pos, pred_global_rot, pred_canon = tl.fk_from_pose(target_clip, root_pos, root_rot, pred_pose, device)
        next_pose = {
            "pelvis_pos": pred_pose["pelvis_pos"],
            "pelvis_rot6": pred_pose["pelvis_rot6"],
            "nonpelvis_rot6": pred_pose["nonpelvis_rot6"],
            "canon_pos": pred_canon,
            "contacts": pred_pose["contacts"],
        }

        target_pose = tl.get_pose_from_clip(target_clip, target_next_idx, device)
        target_global_pos, target_global_rot = tl.global_from_clip(target_clip, target_next_idx, cfg, device)
        target_rows, target_parts = target_losses(
            next_pose,
            pred_global_pos,
            pred_global_rot,
            target_pose,
            target_global_pos,
            target_global_rot,
            target_clip,
            args,
        )
        ae_rows = transition_prior_score(
            priors,
            target_clip,
            target_prev_idx,
            target_cur_idx,
            prev_pose,
            cur_pose,
            next_pose,
            cfg,
            device,
            args.ae_score_loss,
            args.ae_huber_delta,
        )
        cur_root_pos, cur_root_rot, _cur_yaw, _cur_heading = tl.root_state(target_clip, target_cur_idx, cfg, device)
        cur_global_pos, cur_global_rot, _cur_canon = tl.fk_from_pose(target_clip, cur_root_pos, cur_root_rot, cur_pose, device)
        slide_rows, slide_parts = foot_motion_terms(
            cur_global_pos,
            cur_global_rot,
            pred_global_pos,
            pred_global_rot,
            tuple(target_clip.foot_indices),
            tuple(target_clip.toe_indices),
            float(target_clip.fps),
            args.contact_height_m,
            args.contact_temperature_m,
            args.both_speed_threshold_mps,
            args.slide_loss_kind,
            args.contact_slide_loss_weight,
            args.both_fast_loss_weight,
            args.speed_product_loss_weight,
            args.speed_product_scale_mps,
            args.both_ground_product_loss_weight,
        )
        plan_rows = torch.zeros_like(target_rows)
        if args.foot_plan_loss_weight > 0.0:
            if args.plan_mode == "soft_farthest":
                pair_idx = foot_pairs.reshape(-1)
                cur_pair_pos = cur_global_pos.index_select(1, pair_idx).reshape(batch, 2, 2, 3)
                target_pair_pos = target_global_pos.index_select(1, pair_idx).reshape(batch, 2, 2, 3)
                pred_pair_pos = pred_global_pos.index_select(1, pair_idx).reshape(batch, 2, 2, 3)
                current_distance = torch.linalg.norm(cur_pair_pos - target_pair_pos, dim=-1).mean(dim=-1)
                temp = max(float(args.plan_temperature_m), 1e-5)
                mover_weights = torch.softmax(current_distance / temp, dim=-1)
                planter_weights = torch.softmax(-current_distance / temp, dim=-1)
                foot_target_rows = (pred_pair_pos - target_pair_pos).square().sum(dim=-1).mean(dim=-1)
                mover_target_rows = (mover_weights * foot_target_rows).sum(dim=-1)
                planted_speed_rows = (planter_weights * slide_parts["speeds"].square()).sum(dim=-1)
                planted_height_rows = (
                    planter_weights * F.relu(slide_parts["heights"] - float(args.contact_height_m)).square()
                ).sum(dim=-1)
                ambiguity_rows = (mover_weights[:, 0] * mover_weights[:, 1]) * 4.0
                plan_rows = (
                    mover_target_rows
                    + float(args.plan_planted_slide_weight) * planted_speed_rows
                    + float(args.plan_planted_height_weight) * planted_height_rows
                    + float(args.plan_commitment_weight) * ambiguity_rows
                )
            else:
                if args.plan_mode == "phase":
                    phase_split = max(1, min(int(rollout_k) - 1, int(round(float(args.plan_switch_fraction) * int(rollout_k)))))
                    first = first_mover
                    second = 1 - first_mover
                    mover = torch.where(
                        torch.full_like(first_mover, bool(_step < phase_split), dtype=torch.bool),
                        first,
                        second,
                    )
                elif args.plan_mode == "farthest":
                    first = None
                    phase_split = 0
                    pair_idx = foot_pairs.reshape(-1)
                    cur_pair_pos = cur_global_pos.index_select(1, pair_idx).reshape(batch, 2, 2, 3)
                    target_pair_pos = target_global_pos.index_select(1, pair_idx).reshape(batch, 2, 2, 3)
                    current_distance = torch.linalg.norm(cur_pair_pos - target_pair_pos, dim=-1).mean(dim=-1)
                    mover = current_distance.argmax(dim=-1)
                else:
                    raise ValueError(f"unknown plan mode: {args.plan_mode}")
                planter = 1 - mover
                mover_pairs = foot_pairs.index_select(0, mover.reshape(-1)).reshape(batch, 2)
                gather_idx = mover_pairs[:, :, None].expand(-1, -1, 3)
                mover_pred = pred_global_pos.gather(1, gather_idx)
                mover_target = target_global_pos.gather(1, gather_idx)
                mover_target_rows = (mover_pred - mover_target).square().sum(dim=-1).mean(dim=-1)
                planted_speed = slide_parts["speeds"].gather(-1, planter.unsqueeze(-1)).squeeze(-1)
                plan_rows = mover_target_rows + float(args.plan_planted_slide_weight) * planted_speed.square()
                if args.plan_mode == "phase" and _step >= phase_split:
                    first_pairs = foot_pairs.index_select(0, first.reshape(-1)).reshape(batch, 2)
                    first_idx = first_pairs[:, :, None].expand(-1, -1, 3)
                    first_pred = pred_global_pos.gather(1, first_idx)
                    first_target = target_global_pos.gather(1, first_idx)
                    first_target_rows = (first_pred - first_target).square().sum(dim=-1).mean(dim=-1)
                    plan_rows = plan_rows + float(args.plan_settled_foot_weight) * first_target_rows
                if _step == int(rollout_k) - 1 and float(args.plan_terminal_feet_weight) > 0.0:
                    pair_idx = foot_pairs.reshape(-1)
                    pred_pair_pos = pred_global_pos.index_select(1, pair_idx).reshape(batch, 2, 2, 3)
                    target_pair_pos = target_global_pos.index_select(1, pair_idx).reshape(batch, 2, 2, 3)
                    terminal_feet_rows = (pred_pair_pos - target_pair_pos).square().sum(dim=-1).mean(dim=(-1, -2))
                    plan_rows = plan_rows + float(args.plan_terminal_feet_weight) * terminal_feet_rows

        target_weight = float(args.target_loss_weight)
        if _step == int(rollout_k) - 1:
            target_weight += float(args.terminal_target_loss_weight)
        step_rows = (
            target_weight * target_rows
            + args.ae_loss_weight * ae_rows
            + args.slide_loss_weight * slide_rows
            + args.foot_plan_loss_weight * plan_rows
        )
        total = total + step_rows.mean() / max(1, int(rollout_k))

        ae_rows_all.append(ae_rows.detach())
        target_pose_rows_all.append(target_parts["pose_mse"].detach())
        joint_rows_all.append(target_parts["joint_rmse"].detach())
        ee_rows_all.append(target_parts["ee_rmse"].detach())
        contact_slide_all.append(slide_parts["contact_slide"].detach())
        both_fast_all.append(slide_parts["both_fast"].detach())
        speed_all.append(slide_parts["speeds"].detach())
        height_all.append(slide_parts["heights"].detach())
        point_all.append(slide_parts["points"].detach())
        contact_all.append(slide_parts["soft_contact"].detach())

        prev_pose = cur_pose
        cur_pose = next_pose
        target_prev_idx = target_cur_idx
        target_cur_idx = target_next_idx

    metrics, traces = summarize_rollout(
        torch.stack(ae_rows_all, dim=0),
        torch.stack(target_pose_rows_all, dim=0),
        torch.stack(joint_rows_all, dim=0),
        torch.stack(ee_rows_all, dim=0),
        torch.stack(contact_slide_all, dim=0),
        torch.stack(both_fast_all, dim=0),
        torch.stack(speed_all, dim=0),
        torch.stack(height_all, dim=0),
        torch.stack(point_all, dim=0),
        torch.stack(contact_all, dim=0),
        args,
    )
    return total, metrics, traces


def summarize_rollout(
    ae_rows: torch.Tensor,
    target_pose_rows: torch.Tensor,
    joint_rows: torch.Tensor,
    ee_rows: torch.Tensor,
    contact_slide: torch.Tensor,
    both_fast: torch.Tensor,
    speeds: torch.Tensor,
    heights: torch.Tensor,
    points: torch.Tensor,
    contacts: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[LabMetrics, dict[str, np.ndarray]]:
    with torch.no_grad():
        speed_min = speeds.min(dim=-1).values
        both_fast_mask = speed_min > float(args.metric_speed_threshold_mps)
        both_ground = heights.max(dim=-1).values < float(args.metric_ground_height_m)
        both_ground_sliding = torch.logical_and(both_ground, both_fast_mask)
        left_moving = speeds[..., 0] > float(args.metric_speed_threshold_mps)
        right_moving = speeds[..., 1] > float(args.metric_speed_threshold_mps)
        left_solo = torch.logical_and(left_moving, ~right_moving)
        right_solo = torch.logical_and(right_moving, ~left_moving)
        solo_motion = torch.logical_or(left_solo, right_solo)
        planted_available = contacts.max(dim=-1).values
        both_airborne = planted_available < 0.3
        moving = speeds.max(dim=-1).values > float(args.metric_speed_threshold_mps)
        dominance = speeds.max(dim=-1).values / speeds.sum(dim=-1).clamp_min(1e-6)
        dominance_when_moving = dominance[moving]
        if dominance_when_moving.numel() == 0:
            dominance_when_moving = torch.ones((1,), dtype=dominance.dtype, device=dominance.device)
        dominant_foot = speeds.argmax(dim=-1)
        switches = []
        for batch_i in range(speeds.shape[1]):
            moving_i = moving[:, batch_i]
            dom_i = dominant_foot[:, batch_i][moving_i]
            if dom_i.numel() <= 1:
                switches.append(torch.zeros((), device=speeds.device))
            else:
                switches.append((dom_i[1:] != dom_i[:-1]).float().sum())
        switches_t = torch.stack(switches)
        disp = torch.linalg.norm(points[1:] - points[:-1], dim=-1) if points.shape[0] > 1 else torch.zeros_like(speeds)
        swing = heights[:-1] > float(args.metric_lift_height_m) if heights.shape[0] > 1 else torch.zeros_like(heights, dtype=torch.bool)
        left_path = disp[..., 0].sum(dim=0).mean() if disp.numel() else torch.zeros((), device=speeds.device)
        right_path = disp[..., 1].sum(dim=0).mean() if disp.numel() else torch.zeros((), device=speeds.device)
        left_swing = torch.where(swing[..., 0], disp[..., 0], torch.zeros_like(disp[..., 0])).sum(dim=0).mean() if disp.numel() else torch.zeros((), device=speeds.device)
        right_swing = torch.where(swing[..., 1], disp[..., 1], torch.zeros_like(disp[..., 1])).sum(dim=0).mean() if disp.numel() else torch.zeros((), device=speeds.device)
        contact_slide_scalar = contact_slide.sum(dim=-1)
        flat_contact_slide = contact_slide_scalar.reshape(-1)
        contact_p95 = torch.quantile(flat_contact_slide, 0.95) if flat_contact_slide.numel() else torch.zeros((), device=speeds.device)
        both_ground_values = torch.where(both_ground_sliding, speed_min, torch.zeros_like(speed_min))
        success = (
            float(joint_rows[-1].mean().detach().cpu()) <= float(args.success_final_joint_rmse_m)
            and float(both_fast_mask.float().mean().detach().cpu()) <= float(args.success_both_fast_rate)
            and float(both_ground_sliding.float().mean().detach().cpu()) <= float(args.success_both_ground_sliding_rate)
            and float((planted_available > 0.7).float().mean().detach().cpu()) >= float(args.success_planted_available_rate)
            and float(left_solo.float().mean().detach().cpu()) >= float(args.success_solo_foot_rate)
            and float(right_solo.float().mean().detach().cpu()) >= float(args.success_solo_foot_rate)
        )
        metrics = LabMetrics(
            ae_mean=float(ae_rows.mean().detach().cpu()),
            target_pose_mse=float(target_pose_rows.mean().detach().cpu()),
            final_joint_rmse_m=float(joint_rows[-1].mean().detach().cpu()),
            final_ee_rmse_m=float(ee_rows[-1].mean().detach().cpu()),
            contact_slide_mean_mps=float(contact_slide_scalar.mean().detach().cpu()),
            contact_slide_p95_mps=float(contact_p95.detach().cpu()),
            both_fast_rate=float(both_fast_mask.float().mean().detach().cpu()),
            both_fast_max_mps=float(speed_min.max().detach().cpu()),
            both_ground_sliding_rate=float(both_ground_sliding.float().mean().detach().cpu()),
            both_ground_sliding_max_mps=float(both_ground_values.max().detach().cpu()),
            planted_available_rate=float((planted_available > 0.7).float().mean().detach().cpu()),
            both_airborne_rate=float(both_airborne.float().mean().detach().cpu()),
            dominance_mean=float(dominance_when_moving.mean().detach().cpu()),
            dominance_min=float(dominance_when_moving.min().detach().cpu()),
            dominant_switches=float(switches_t.float().mean().detach().cpu()),
            solo_motion_rate=float(solo_motion.float().mean().detach().cpu()),
            left_solo_rate=float(left_solo.float().mean().detach().cpu()),
            right_solo_rate=float(right_solo.float().mean().detach().cpu()),
            left_path_m=float(left_path.detach().cpu()),
            right_path_m=float(right_path.detach().cpu()),
            left_swing_path_m=float(left_swing.detach().cpu()),
            right_swing_path_m=float(right_swing.detach().cpu()),
            left_lift_max_m=float(heights[..., 0].max().detach().cpu()),
            right_lift_max_m=float(heights[..., 1].max().detach().cpu()),
            success=1.0 if success else 0.0,
        )
        traces = {
            "speeds_mps": speeds.detach().cpu().numpy(),
            "heights_m": heights.detach().cpu().numpy(),
            "soft_contact": contacts.detach().cpu().numpy(),
            "contact_slide_mps": contact_slide.detach().cpu().numpy(),
            "joint_rmse_m": joint_rows.detach().cpu().numpy(),
            "ee_rmse_m": ee_rows.detach().cpu().numpy(),
        }
        return metrics, traces


def make_init_pairs(
    mode: str,
    batch_size: int,
    init_clips: list[tl.MotionClip],
    fixed_clip_index: int,
    fixed_frames: list[int],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    if mode == "fixed":
        frames = [int(frame) for frame in fixed_frames] or [0]
        return [(int(fixed_clip_index), frames[row % len(frames)]) for row in range(int(batch_size))]
    if mode == "mixed":
        fixed_count = max(1, int(batch_size) // 2)
        random_count = max(0, int(batch_size) - fixed_count)
        return make_init_pairs("fixed", fixed_count, init_clips, fixed_clip_index, fixed_frames, rng) + make_init_pairs(
            "random", random_count, init_clips, fixed_clip_index, fixed_frames, rng
        )
    pairs = []
    for _ in range(int(batch_size)):
        ci = int(rng.integers(0, len(init_clips)))
        clip = init_clips[ci]
        high = max(1, int(clip.cyclic_period if clip.cyclic_animation else clip.T))
        frame = int(rng.integers(0, high))
        pairs.append((ci, frame))
    return pairs


def select_balanced_walk_frames(clip: tl.MotionClip) -> list[int]:
    device = torch.device("cpu")
    tensors = clip.tensors(device)
    heights, points = cp.foot_lowest_heights_and_points(
        tensors["global_pos"],
        tensors["global_rot"],
        tuple(clip.foot_indices),
        tuple(clip.toe_indices),
    )
    local = torch.einsum(
        "tfc,tcd->tfd",
        points - tensors["root_pos"][:, None, :],
        tensors["root_heading_rot"],
    )
    separation = torch.linalg.norm(local[:, 0, [0, 2]] - local[:, 1, [0, 2]], dim=-1)
    left_score = heights[:, 0] - heights[:, 1] + 0.25 * separation
    right_score = heights[:, 1] - heights[:, 0] + 0.25 * separation
    return [int(left_score.argmax().item()), int(right_score.argmax().item())]


def parse_fixed_frames(text: str, walk_clip: tl.MotionClip, fallback: int) -> list[int]:
    if not text.strip() or text.strip().lower() == "auto":
        return select_balanced_walk_frames(walk_clip)
    frames = [int(part.strip()) for part in text.replace(";", ",").split(",") if part.strip()]
    return frames or [int(fallback)]


def write_plot(run_dir: Path, traces: dict[str, np.ndarray], title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    speeds = traces["speeds_mps"][:, 0, :]
    heights = traces["heights_m"][:, 0, :]
    contact = traces["soft_contact"][:, 0, :]
    x = np.arange(speeds.shape[0])
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(x, speeds[:, 0], label="left")
    axes[0].plot(x, speeds[:, 1], label="right")
    axes[0].set_ylabel("slide m/s")
    axes[0].legend(loc="upper right")
    axes[1].plot(x, heights[:, 0], label="left")
    axes[1].plot(x, heights[:, 1], label="right")
    axes[1].set_ylabel("lowest height m")
    axes[2].plot(x, contact[:, 0], label="left")
    axes[2].plot(x, contact[:, 1], label="right")
    axes[2].set_ylabel("soft contact")
    axes[2].set_xlabel("rollout step")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(run_dir / "foot_timeline.png", dpi=140)
    plt.close(fig)


def save_checkpoint(run_dir: Path, model: torch.nn.Module, opt: torch.optim.Optimizer, epoch: int, best: float, cfg: tl.TrainConfig, metadata: dict) -> None:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tl.save_checkpoint(ckpt_dir / "checkpoint_best.pt", model, opt, epoch, best, int(metadata["rollout_k"]), cfg, metadata)
    tl.save_checkpoint(ckpt_dir / "checkpoint_last.pt", model, opt, epoch, best, int(metadata["rollout_k"]), cfg, metadata)


def save_last_checkpoint(run_dir: Path, model: torch.nn.Module, opt: torch.optim.Optimizer, epoch: int, score: float, cfg: tl.TrainConfig, metadata: dict) -> None:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tl.save_checkpoint(ckpt_dir / "checkpoint_last.pt", model, opt, epoch, score, int(metadata["rollout_k"]), cfg, metadata)


def contract_score(metrics: LabMetrics) -> float:
    return (
        (0.0 if metrics.success >= 1.0 else 1000.0)
        + metrics.final_joint_rmse_m
        + metrics.both_fast_rate
        + metrics.both_ground_sliding_rate
        - 0.01 * metrics.solo_motion_rate
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Contained idle recovery foot-slide lab.")
    parser.add_argument("--idle-npz", default="training/runs/micro_datasets/idle_only/periodic/M_Neutral_Stand_Idle_Loop.npz")
    parser.add_argument("--walk-npz", default="training/runs/micro_datasets/walk_forward_only/periodic/M_Neutral_Walk_Loop_F.npz")
    parser.add_argument("--init-folder", default="", help="Optional semicolon-separated folders used for random-init validation/training.")
    parser.add_argument("--prior-checkpoint", default="training/runs/20260521_051229_simple_condroot1_full_recon_e160/checkpoints/checkpoint_best.pt")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-dir", default="training/runs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rollout-k", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--train-init-mode", choices=("fixed", "random", "mixed"), default="fixed")
    parser.add_argument("--selection-mode", choices=("fixed", "random", "combined"), default="fixed")
    parser.add_argument("--validate-random-inits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fixed-frame", type=int, default=0)
    parser.add_argument("--fixed-frames", default="auto", help="Comma-separated walk init frames, or auto for left/right swing extremes.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--target-loss-weight", type=float, default=1.0)
    parser.add_argument("--target-pose-component-weight", type=float, default=1.0)
    parser.add_argument("--target-joint-component-weight", type=float, default=0.5)
    parser.add_argument("--target-rot-component-weight", type=float, default=0.05)
    parser.add_argument("--target-foot-global-weight", type=float, default=1.0)
    parser.add_argument("--target-foot-rot-weight", type=float, default=1.0)
    parser.add_argument("--ae-loss-weight", type=float, default=0.3)
    parser.add_argument("--slide-loss-weight", type=float, default=40.0)
    parser.add_argument(
        "--slide-loss-kind",
        choices=("soft_contact", "planted_lowest", "one_planted_soft"),
        default="one_planted_soft",
    )
    parser.add_argument("--terminal-target-loss-weight", type=float, default=3.0)
    parser.add_argument("--foot-plan-loss-weight", type=float, default=0.0)
    parser.add_argument("--plan-planted-slide-weight", type=float, default=10.0)
    parser.add_argument("--plan-planted-height-weight", type=float, default=0.0)
    parser.add_argument("--plan-commitment-weight", type=float, default=0.0)
    parser.add_argument("--plan-settled-foot-weight", type=float, default=0.0)
    parser.add_argument("--plan-terminal-feet-weight", type=float, default=0.0)
    parser.add_argument("--plan-temperature-m", type=float, default=0.06)
    parser.add_argument("--plan-mode", choices=("phase", "farthest", "soft_farthest"), default="phase")
    parser.add_argument("--plan-switch-fraction", type=float, default=0.5)
    parser.add_argument("--contact-slide-loss-weight", type=float, default=0.35)
    parser.add_argument("--both-fast-loss-weight", type=float, default=1.0)
    parser.add_argument("--speed-product-loss-weight", type=float, default=0.0)
    parser.add_argument("--speed-product-scale-mps", type=float, default=0.10)
    parser.add_argument("--both-ground-product-loss-weight", type=float, default=0.0)
    parser.add_argument("--contact-height-m", type=float, default=0.035)
    parser.add_argument("--contact-temperature-m", type=float, default=0.015)
    parser.add_argument("--both-speed-threshold-mps", type=float, default=0.025)
    parser.add_argument("--metric-speed-threshold-mps", type=float, default=0.05)
    parser.add_argument("--metric-ground-height-m", type=float, default=0.035)
    parser.add_argument("--metric-lift-height-m", type=float, default=0.05)
    parser.add_argument("--success-final-joint-rmse-m", type=float, default=0.075)
    parser.add_argument("--success-both-fast-rate", type=float, default=0.05)
    parser.add_argument("--success-both-ground-sliding-rate", type=float, default=0.02)
    parser.add_argument("--success-planted-available-rate", type=float, default=0.90)
    parser.add_argument("--success-solo-foot-rate", type=float, default=0.05)
    parser.add_argument("--ae-score-loss", choices=("mse", "huber"), default="mse")
    parser.add_argument("--ae-huber-delta", type=float, default=1.0)
    parser.add_argument("--save-last-every", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np_rng = np.random.default_rng(int(args.seed))
    device = torch.device(args.device)
    prior_path = resolve(args.prior_checkpoint)
    prior_ckpt = torch.load(prior_path, map_location=device, weights_only=False)
    cfg = tl.TrainConfig()
    apply_checkpoint_locomotion_config(cfg, prior_ckpt)
    cfg.device = str(device)
    cfg.hidden_dim = int(args.hidden_dim)
    cfg.num_hidden_layers = int(args.num_hidden_layers)
    cfg.learning_rate = float(args.learning_rate)
    cfg.use_contact_state = bool(getattr(cfg, "use_contact_state", False))
    cfg.zero_contact_state = bool(getattr(cfg, "zero_contact_state", False))
    cfg.predict_residual = True
    cfg.zero_init_output = True
    cfg.cyclic_animation = True

    idle_clip = tl.MotionClip(resolve(args.idle_npz), cfg, cyclic_animation=True)
    walk_clip = tl.MotionClip(resolve(args.walk_npz), cfg, cyclic_animation=True)
    fixed_frames = parse_fixed_frames(args.fixed_frames, walk_clip, int(args.fixed_frame))
    init_clips = [walk_clip]
    if args.init_folder.strip():
        for folder_text in args.init_folder.replace(";", ",").split(","):
            folder_text = folder_text.strip()
            if not folder_text:
                continue
            for path in sorted(resolve(folder_text).glob("*.npz")):
                init_clips.append(tl.MotionClip(path, cfg, cyclic_animation=True))

    input_dim, output_dim = tl.make_batch_dims(idle_clip, cfg)
    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=0.0)
    if args.resume_checkpoint.strip():
        resume_path = resolve(args.resume_checkpoint)
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        tl.unwrap_compiled_model(model).load_state_dict(resume["model"])
        print(f"resumed model weights from {resume_path}", flush=True)
    priors = ae_prior.load_prior_bundle([prior_path], device)

    run_name = args.run_name.strip() or time.strftime("%Y%m%d_%H%M%S_idle_recovery_slide_lab")
    run_dir = resolve(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(run_dir / "tb")
    metadata = {
        "run_name": run_name,
        "idle_npz": str(resolve(args.idle_npz)),
        "walk_npz": str(resolve(args.walk_npz)),
        "prior_checkpoint": str(prior_path),
        "rollout_k": int(args.rollout_k),
        "args": vars(args),
        "resume_checkpoint": str(resolve(args.resume_checkpoint)) if args.resume_checkpoint.strip() else "",
        "locomotion_config": asdict(cfg),
        "metric_contract": {
            "final_joint_rmse_m_max": float(args.success_final_joint_rmse_m),
            "both_fast_rate_max": float(args.success_both_fast_rate),
            "both_ground_sliding_rate_max": float(args.success_both_ground_sliding_rate),
            "planted_available_rate_min": float(args.success_planted_available_rate),
        },
        "fixed_init_frames": fixed_frames,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"fixed_init_frames={fixed_frames}", flush=True)
    fixed_pairs = make_init_pairs("fixed", int(args.batch_size), init_clips, 0, fixed_frames, np_rng)
    random_eval_pairs = make_init_pairs("random", int(args.batch_size), init_clips, 0, fixed_frames, np_rng)
    best_score = math.inf
    best_metrics: LabMetrics | None = None
    best_traces: dict[str, np.ndarray] | None = None
    start_time = time.perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_pairs = make_init_pairs(args.train_init_mode, int(args.batch_size), init_clips, 0, fixed_frames, np_rng)
        opt.zero_grad(set_to_none=True)
        loss, train_metrics, _train_traces = rollout(
            model,
            priors,
            idle_clip,
            init_clips,
            train_pairs,
            cfg,
            device,
            int(args.rollout_k),
            args,
            train=True,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            _eval_loss, fixed_metrics, fixed_traces = rollout(
                model,
                priors,
                idle_clip,
                init_clips,
                fixed_pairs,
                cfg,
                device,
                int(args.rollout_k),
                args,
                train=False,
            )
            random_epoch_metrics = None
            random_epoch_traces = None
            if args.selection_mode in {"random", "combined"}:
                _random_epoch_loss, random_epoch_metrics, random_epoch_traces = rollout(
                    model,
                    priors,
                    idle_clip,
                    init_clips,
                    random_eval_pairs,
                    cfg,
                    device,
                    int(args.rollout_k),
                    args,
                    train=False,
                )
        if args.selection_mode == "random" and random_epoch_metrics is not None:
            selection_metrics = random_epoch_metrics
            selection_traces = random_epoch_traces if random_epoch_traces is not None else fixed_traces
            score = contract_score(random_epoch_metrics)
        elif args.selection_mode == "combined" and random_epoch_metrics is not None:
            selection_metrics = fixed_metrics if fixed_metrics.success < 1.0 else random_epoch_metrics
            selection_traces = fixed_traces if fixed_metrics.success < 1.0 else (random_epoch_traces or fixed_traces)
            score = contract_score(fixed_metrics) + contract_score(random_epoch_metrics)
        else:
            selection_metrics = fixed_metrics
            selection_traces = fixed_traces
            score = contract_score(fixed_metrics)
        if score < best_score:
            best_score = score
            best_metrics = selection_metrics
            best_traces = selection_traces
            save_checkpoint(run_dir, model, opt, epoch, best_score, cfg, metadata)
            (run_dir / "best_metrics.json").write_text(json.dumps(asdict(selection_metrics), indent=2), encoding="utf-8")
            np.savez_compressed(run_dir / "best_traces.npz", **selection_traces)
            write_plot(run_dir, selection_traces, f"{run_name} epoch {epoch}")
        elif int(args.save_last_every) > 0 and epoch % int(args.save_last_every) == 0:
            save_last_checkpoint(run_dir, model, opt, epoch, score, cfg, metadata)

        for prefix, metrics in (("train", train_metrics), ("fixed", fixed_metrics)):
            for key, value in asdict(metrics).items():
                writer.add_scalar(f"{prefix}/{key}", float(value), epoch)
        if random_epoch_metrics is not None:
            for key, value in asdict(random_epoch_metrics).items():
                writer.add_scalar(f"random_eval/{key}", float(value), epoch)
        writer.add_scalar("loss/train_total", float(loss.detach().cpu()), epoch)
        if epoch == 1 or epoch % 10 == 0 or fixed_metrics.success >= 1.0:
            print(
            f"epoch={epoch:04d} loss={float(loss.detach().cpu()):.6g} "
            f"fixed_success={fixed_metrics.success:.0f} final_joint={fixed_metrics.final_joint_rmse_m:.5f} "
                f"both_fast={fixed_metrics.both_fast_rate:.3f} both_ground_slide={fixed_metrics.both_ground_sliding_rate:.3f} "
                f"planted={fixed_metrics.planted_available_rate:.3f} dominance={fixed_metrics.dominance_mean:.3f} "
                f"solo=({fixed_metrics.left_solo_rate:.3f},{fixed_metrics.right_solo_rate:.3f}) "
                f"switches={fixed_metrics.dominant_switches:.2f} elapsed_s={time.perf_counter() - start_time:.1f}",
                flush=True,
            )
            if random_epoch_metrics is not None:
                print(
                    f"  random_eval success={random_epoch_metrics.success:.0f} final_joint={random_epoch_metrics.final_joint_rmse_m:.5f} "
                    f"both_fast={random_epoch_metrics.both_fast_rate:.3f} "
                    f"both_ground_slide={random_epoch_metrics.both_ground_sliding_rate:.3f} "
                    f"planted={random_epoch_metrics.planted_available_rate:.3f} "
                    f"solo=({random_epoch_metrics.left_solo_rate:.3f},{random_epoch_metrics.right_solo_rate:.3f})",
                    flush=True,
                )
        selected_success = (
            fixed_metrics.success >= 1.0
            if args.selection_mode == "fixed" or random_epoch_metrics is None
            else random_epoch_metrics.success >= 1.0
            if args.selection_mode == "random"
            else fixed_metrics.success >= 1.0 and random_epoch_metrics.success >= 1.0
        )
        if selected_success and epoch >= 20:
            break

    random_metrics = None
    random_traces = None
    if bool(args.validate_random_inits):
        model.eval()
        random_pairs = random_eval_pairs
        with torch.no_grad():
            _random_loss, random_metrics, random_traces = rollout(
                model,
                priors,
                idle_clip,
                init_clips,
                random_pairs,
                cfg,
                device,
                int(args.rollout_k),
                args,
                train=False,
            )
        (run_dir / "random_init_metrics.json").write_text(json.dumps(asdict(random_metrics), indent=2), encoding="utf-8")
        np.savez_compressed(run_dir / "random_init_traces.npz", **random_traces)
        for key, value in asdict(random_metrics).items():
            writer.add_scalar(f"random_init/{key}", float(value), int(args.epochs))
        print(
            f"random_init success={random_metrics.success:.0f} final_joint={random_metrics.final_joint_rmse_m:.5f} "
            f"both_fast={random_metrics.both_fast_rate:.3f} both_ground_slide={random_metrics.both_ground_sliding_rate:.3f} "
            f"planted={random_metrics.planted_available_rate:.3f} dominance={random_metrics.dominance_mean:.3f} "
            f"solo=({random_metrics.left_solo_rate:.3f},{random_metrics.right_solo_rate:.3f})",
            flush=True,
        )

    writer.close()
    summary = {
        "best_fixed": asdict(best_metrics) if best_metrics is not None else None,
        "random_init": asdict(random_metrics) if random_metrics is not None else None,
        "run_dir": str(run_dir),
        "best_checkpoint": str(run_dir / "checkpoints" / "checkpoint_best.pt"),
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if best_traces is not None:
        write_plot(run_dir, best_traces, f"{run_name} best fixed")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
