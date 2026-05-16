from __future__ import annotations

import argparse
import copy
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import contact_physics as cp
import train_locomotion as tl
import transition_autoencoder as tae
from visual_report_bridge import VisualReportBridge


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def refresh_live_viewer(args: argparse.Namespace, checkpoint_path: Path) -> None:
    if not args.live_viewer:
        return
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "training" / "visualize_model.py"),
        "--npz-path",
        str(tl.resolve_path(args.live_npz_path)),
        "--checkpoint-path",
        str(checkpoint_path),
        "--output-path",
        str(tl.resolve_path(args.live_output_path)),
    ]
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        print(f"live viewer refresh skipped: {exc}", flush=True)


def load_prior(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = tae.AEConfig(**ckpt["config"])
    model = tae.TransitionAutoencoder(int(ckpt["schema"]["total_dim"]), cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, ckpt


def load_prior_bundle(paths: list[Path], device: torch.device):
    bundle = []
    for path in paths:
        prior, ckpt = load_prior(path, device)
        bundle.append(
            {
                "path": path,
                "model": prior,
                "mean": ckpt["mean"].to(device),
                "std": ckpt["std"].to(device),
            }
        )
    return bundle


def ae_score(
    prior,
    mean,
    std,
    features: torch.Tensor,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
    compatibility_weight: float = 0.0,
) -> torch.Tensor:
    x = (features - mean) / std
    recon = prior(x)
    if loss_type == "huber":
        error = F.huber_loss(recon, x, reduction="none", delta=huber_delta)
    else:
        error = F.mse_loss(recon, x, reduction="none")
    score = error.mean(dim=-1)
    if compatibility_weight > 0.0 and hasattr(prior, "has_compatibility_head") and prior.has_compatibility_head():
        compatibility = F.softplus(-prior.compatibility_logits(x))
        score = score + compatibility_weight * compatibility
    return score.mean()


class AnyStartDataset(torch.utils.data.Dataset):
    def __init__(self, clips: list[tl.MotionClip], cfg: tl.TrainConfig, requested_k: int):
        self.items: list[tuple[int, int]] = []
        for ci, clip in enumerate(clips):
            max_start = clip_sample_start_max(clip, cfg, requested_k, cfg.agent_min_cohort_steps)
            if max_start < 1:
                continue
            starts = list(range(1, max_start + 1))
            random.Random(cfg.seed + ci).shuffle(starts)
            self.items.extend((ci, s) for s in starts)
        if not self.items:
            raise ValueError("No one-step starts found. Need clips with at least 3 frames.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[int, int]:
        return self.items[index]


def clip_future_safe_current_max(clip: tl.MotionClip, cfg: tl.TrainConfig) -> int:
    if clip.cyclic_animation:
        return int(clip.cyclic_period) - 1
    return int(clip.T) - int(cfg.future_window) - 1


def clip_any_start_max(clip: tl.MotionClip, cfg: tl.TrainConfig) -> int:
    if clip.cyclic_animation:
        return int(clip.cyclic_period) - 1
    return clip_future_safe_current_max(clip, cfg)


def clip_sample_start_max(
    clip: tl.MotionClip,
    cfg: tl.TrainConfig,
    requested_k: int,
    min_cohort_steps: int = 1,
) -> int:
    if clip.cyclic_animation:
        return int(clip.cyclic_period) - 1
    requested_k = max(1, int(requested_k))
    full_k_max_start = int(clip.T) - int(cfg.future_window) - requested_k
    max_start = full_k_max_start if full_k_max_start >= 1 else clip_any_start_max(clip, cfg)
    min_cohort_steps = max(1, int(min_cohort_steps))
    if min_cohort_steps > 1:
        guaranteed_max_start = int(clip.T) - int(cfg.future_window) - min_cohort_steps
        if guaranteed_max_start >= 1:
            max_start = min(max_start, guaranteed_max_start)
    return max_start


def clip_supports_any_start(clip: tl.MotionClip, cfg: tl.TrainConfig) -> bool:
    return clip_any_start_max(clip, cfg) >= 1


def expected_active_rollout_steps(
    clip: tl.MotionClip,
    cfg: tl.TrainConfig,
    requested_k: int,
    min_cohort_steps: int = 1,
) -> float:
    requested_k = max(1, int(requested_k))
    if clip.cyclic_animation:
        return float(requested_k)
    max_start = clip_sample_start_max(clip, cfg, requested_k, min_cohort_steps)
    if max_start < 1:
        return 0.0
    total = 0
    for start in range(1, max_start + 1):
        total += max(1, min(requested_k, clip_future_safe_current_max(clip, cfg) - start + 1))
    return float(total) / float(max_start)


def clip_is_idle(clip: tl.MotionClip) -> bool:
    return "idle" in Path(clip.path).stem.lower()


@torch.no_grad()
def compute_groundtruth_support_slide_threshold(
    clips: list[tl.MotionClip],
    cfg: tl.TrainConfig,
    device: torch.device,
    margin: float,
) -> float:
    max_support = 0.0
    for clip in clips:
        limit = clip.cyclic_period if clip.cyclic_animation else clip.T - 1
        idx = torch.arange(0, limit + 1, dtype=torch.long, device=device)
        pos, rot = tl.global_from_clip(clip, idx, cfg, device)
        foot_indices = tuple(int(x) for x in clip.foot_indices_tensor.tolist())
        toe_indices = tuple(int(x) for x in clip.toe_indices_tensor.tolist())
        speeds = cp.foot_slide_speeds(
            pos[:-1],
            rot[:-1],
            pos[1:],
            rot[1:],
            foot_indices,
            toe_indices,
            clip.fps,
        )
        support = speeds.reshape(-1) if clip_is_idle(clip) else speeds.amin(dim=-1)
        max_support = max(max_support, float(support.max().detach().cpu()))
    return max_support * float(margin)


def simple_generated_footslide_loss(
    clip: tl.MotionClip,
    cur_pos: torch.Tensor,
    cur_rot: torch.Tensor,
    next_pos: torch.Tensor,
    next_rot: torch.Tensor,
    threshold_mps: float,
) -> torch.Tensor:
    foot_indices = tuple(int(x) for x in clip.foot_indices_tensor.tolist())
    toe_indices = tuple(int(x) for x in clip.toe_indices_tensor.tolist())
    speeds = cp.foot_slide_speeds(
        cur_pos,
        cur_rot,
        next_pos,
        next_rot,
        foot_indices,
        toe_indices,
        clip.fps,
    )
    support_speed = speeds.mean(dim=-1) if clip_is_idle(clip) else speeds.amin(dim=-1)
    return F.relu(support_speed - float(threshold_mps)).mean()


class PackedClips:
    """Dense multi-clip storage for Isaac-style per-agent random rollouts."""

    def __init__(self, clips: list[tl.MotionClip], cfg: tl.TrainConfig, device: torch.device):
        if not clips:
            raise ValueError("PackedClips needs at least one clip")
        first = clips[0]
        for clip in clips[1:]:
            if clip.body_names != first.body_names or clip.parents_body_list != first.parents_body_list:
                raise ValueError("Packed agent rollouts require all clips to share the same reduced skeleton")
        self.clips = clips
        self.cfg = cfg
        self.device = device
        self.prototype = first
        self.J = first.J
        self.Jn = first.Jn
        self.pelvis = first.pelvis
        self.nonpelvis_map = first.nonpelvis_map
        self.parents_body_list = first.parents_body_list
        self.foot_indices = tuple(int(x) for x in first.foot_indices_tensor.tolist())
        self.toe_indices = tuple(int(x) for x in first.toe_indices_tensor.tolist())
        self.fps = float(first.fps)

        lengths = [clip.T for clip in clips]
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + int(length))
        self.frame_offsets = torch.tensor(offsets[:-1], dtype=torch.long, device=device)
        self.lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.periods = torch.tensor([clip.cyclic_period for clip in clips], dtype=torch.long, device=device)
        self.cyclic = torch.tensor([clip.cyclic_animation for clip in clips], dtype=torch.bool, device=device)
        self.idle = torch.tensor([clip_is_idle(clip) for clip in clips], dtype=torch.bool, device=device)
        self.future_safe_max = torch.where(
            self.cyclic,
            self.periods - 1,
            self.lengths - int(cfg.future_window) - 1,
        )

        tensors = [clip.tensors(device) for clip in clips]
        self.root_pos = torch.cat([t["root_pos"] for t in tensors], dim=0)
        self.root_rot = torch.cat([t["root_rot"] for t in tensors], dim=0)
        self.pelvis_local_pos = torch.cat([t["pelvis_local_pos"] for t in tensors], dim=0)
        self.pelvis_rot6 = torch.cat([t["pelvis_rot6"] for t in tensors], dim=0)
        self.non_pelvis_rot6 = torch.cat([t["non_pelvis_rot6"] for t in tensors], dim=0)
        self.canonical_pos = torch.cat([t["canonical_pos"] for t in tensors], dim=0)
        self.contacts = torch.cat([t["contacts"] for t in tensors], dim=0)
        self.local_offsets = torch.stack([t["local_offsets"] for t in tensors], dim=0)

        self.root0_pos = torch.stack([t["root_pos"][0] for t in tensors], dim=0)
        self.root0_rot = torch.stack([t["root_rot"][0] for t in tensors], dim=0)
        self.root0_inv = self.root0_rot.transpose(-1, -2)
        self.end_pos = torch.stack([t["root_pos"][clip.cyclic_period] for clip, t in zip(clips, tensors)], dim=0)
        self.end_rot = torch.stack([t["root_rot"][clip.cyclic_period] for clip, t in zip(clips, tensors)], dim=0)
        self.cycle_pos = torch.matmul(
            (self.end_pos - self.root0_pos).unsqueeze(1),
            self.root0_inv,
        ).squeeze(1)
        self.cycle_rot = self.end_rot @ self.root0_inv

        self.stage_clip_indices = torch.arange(len(clips), dtype=torch.long, device=device)
        self.stage_clip_probs = torch.full((len(clips),), 1.0 / len(clips), dtype=torch.float32, device=device)

    def update_stage_sampling(self, indices: list[int], weights: list[float]) -> None:
        if not indices:
            raise ValueError("Packed stage sampling needs at least one clip")
        idx = torch.tensor(indices, dtype=torch.long, device=self.device)
        w = torch.tensor(weights, dtype=torch.float32, device=self.device).clamp_min(0.0)
        if float(w.sum().detach().cpu()) <= 0.0:
            w = torch.ones_like(w)
        self.stage_clip_indices = idx
        self.stage_clip_probs = w / w.sum()

    def sample_starts(self, count: int, requested_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        choices = torch.multinomial(self.stage_clip_probs, int(count), replacement=True)
        clip_ids = self.stage_clip_indices.index_select(0, choices)
        max_start = self.clip_sample_start_max(clip_ids, requested_k)
        starts = (torch.rand(int(count), device=self.device) * max_start.float()).floor().long() + 1
        return clip_ids, starts

    def clip_sample_start_max(self, clip_ids: torch.Tensor, requested_k: int) -> torch.Tensor:
        clip_ids = clip_ids.to(self.device)
        cyclic = self.cyclic.index_select(0, clip_ids)
        period_max = self.periods.index_select(0, clip_ids) - 1
        lengths = self.lengths.index_select(0, clip_ids)
        any_start_max = lengths - int(self.cfg.future_window) - 1
        requested_k = max(1, int(requested_k))
        full_k_max = lengths - int(self.cfg.future_window) - requested_k
        max_start = torch.where(full_k_max >= 1, full_k_max, any_start_max)
        min_steps = max(1, int(self.cfg.agent_min_cohort_steps))
        guaranteed = lengths - int(self.cfg.future_window) - min_steps
        max_start = torch.where(guaranteed >= 1, torch.minimum(max_start, guaranteed), max_start)
        return torch.where(cyclic, period_max, max_start).clamp_min(1)

    def frame_index(self, clip_ids: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        clip_ids = clip_ids.to(self.device).long()
        idx = idx.to(self.device).long()
        periods = self.periods.index_select(0, clip_ids).clamp_min(1)
        cyclic = self.cyclic.index_select(0, clip_ids)
        logical = torch.where(cyclic, torch.remainder(idx, periods), idx)
        return self.frame_offsets.index_select(0, clip_ids) + logical

    def get_pose(self, clip_ids: torch.Tensor, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        frame = self.frame_index(clip_ids, idx)
        return {
            "pelvis_pos": self.pelvis_local_pos.index_select(0, frame),
            "pelvis_rot6": self.pelvis_rot6.index_select(0, frame),
            "nonpelvis_rot6": self.non_pelvis_rot6.index_select(0, frame),
            "canon_pos": self.canonical_pos.index_select(0, frame),
            "contacts": self.contacts.index_select(0, frame),
        }

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

    def fk_from_pose(
        self,
        clip_ids: torch.Tensor,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        pose: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = root_pos.shape[0]
        pelvis_rot = tl.rotation_6d_to_matrix(pose["pelvis_rot6"])
        nonpelvis_rot = tl.rotation_6d_to_matrix(pose["nonpelvis_rot6"])
        offsets = self.local_offsets.index_select(0, clip_ids.to(self.device).long()).clone()
        offsets[:, self.pelvis] = pose["pelvis_pos"]

        global_pos_list: list[torch.Tensor] = []
        global_rot_list: list[torch.Tensor] = []
        for j in range(self.J):
            local_rot_j = pelvis_rot if j == self.pelvis else nonpelvis_rot[:, self.nonpelvis_map[j]]
            parent = self.parents_body_list[j]
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
        root_yaw = tl.heading_yaw_from_root(root_rot)
        heading = tl.yaw_to_row_matrix(root_yaw)
        canon = torch.einsum("bjc,bcd->bjd", global_pos - root_pos[:, None, :], heading)
        return global_pos, global_rot, canon


def packed_build_input(
    packed: PackedClips,
    clip_ids: torch.Tensor,
    prev_idx: torch.Tensor,
    cur_idx: torch.Tensor,
    prev_pose: dict[str, torch.Tensor],
    cur_pose: dict[str, torch.Tensor],
    cfg: tl.TrainConfig,
) -> torch.Tensor:
    current = tl.body_pose_vector(cur_pose, cfg.use_contact_state, cfg.zero_contact_state)
    previous = tl.body_pose_vector(prev_pose, cfg.use_contact_state, cfg.zero_contact_state)
    pelvis_vel = (cur_pose["pelvis_pos"] - prev_pose["pelvis_pos"]) / cfg.pose_delta_scale_final
    joint_vel = (cur_pose["canon_pos"] - prev_pose["canon_pos"]).reshape(cur_idx.shape[0], -1) / cfg.pose_delta_scale_final
    prev_pos, _prev_rot, prev_yaw, prev_heading = packed.root_state(clip_ids, prev_idx)
    cur_pos, _cur_rot, cur_yaw, cur_heading = packed.root_state(clip_ids, cur_idx)
    delta_local = torch.matmul((cur_pos - prev_pos).unsqueeze(1), prev_heading).squeeze(1)
    root_feat = torch.stack(
        (
            delta_local[:, 0] / cfg.max_speed_scale_final,
            delta_local[:, 2] / cfg.max_speed_scale_final,
            tl.wrap_angle(cur_yaw - prev_yaw) / cfg.max_turn_rate_scale_final,
        ),
        dim=-1,
    )
    future_feats = []
    for k in range(1, cfg.future_window + 1):
        fut_pos, _fut_rot, fut_yaw, _fut_heading = packed.root_state(clip_ids, cur_idx + k)
        fut_local = torch.matmul((fut_pos - cur_pos).unsqueeze(1), cur_heading).squeeze(1)
        scale_k = k * cfg.max_speed_scale_final
        dyaw = tl.wrap_angle(fut_yaw - cur_yaw)
        future_feats.append(
            torch.stack(
                (
                    torch.clamp(fut_local[:, 0] / scale_k, -2.0, 2.0),
                    torch.clamp(fut_local[:, 2] / scale_k, -2.0, 2.0),
                    torch.cos(dyaw),
                    torch.sin(dyaw),
                ),
                dim=-1,
            )
        )
    future_feat = torch.cat(future_feats, dim=-1)
    return torch.cat((current, previous, pelvis_vel, joint_vel, root_feat, future_feat), dim=-1)


def packed_transition_feature_from_next_pose(
    packed: PackedClips,
    clip_ids: torch.Tensor,
    prev_idx: torch.Tensor,
    cur_idx: torch.Tensor,
    prev_pose: dict[str, torch.Tensor],
    cur_pose: dict[str, torch.Tensor],
    next_pose: dict[str, torch.Tensor],
    cfg: tl.TrainConfig,
) -> torch.Tensor:
    model_input = packed_build_input(packed, clip_ids, prev_idx, cur_idx, prev_pose, cur_pose, cfg)
    next_output = tl.pose_target_output(next_pose)
    pelvis_next_vel = (next_pose["pelvis_pos"] - cur_pose["pelvis_pos"]) / cfg.pose_delta_scale_final
    joint_next_vel = (next_pose["canon_pos"] - cur_pose["canon_pos"]).reshape(cur_idx.shape[0], -1)
    joint_next_vel = joint_next_vel / cfg.pose_delta_scale_final
    return torch.cat(
        (
            model_input,
            next_output,
            next_pose["canon_pos"].reshape(cur_idx.shape[0], -1),
            pelvis_next_vel,
            joint_next_vel,
        ),
        dim=-1,
    )


def packed_generated_footslide_loss(
    packed: PackedClips,
    cur_pos: torch.Tensor,
    cur_rot: torch.Tensor,
    next_pos: torch.Tensor,
    next_rot: torch.Tensor,
    threshold_mps: float,
    clip_ids: torch.Tensor,
) -> torch.Tensor:
    speeds = cp.foot_slide_speeds(
        cur_pos,
        cur_rot,
        next_pos,
        next_rot,
        packed.foot_indices,
        packed.toe_indices,
        packed.fps,
    )
    idle_mask = packed.idle.index_select(0, clip_ids.to(packed.device).long())
    support_speed = torch.where(idle_mask, speeds.mean(dim=-1), speeds.amin(dim=-1))
    return F.relu(support_speed - float(threshold_mps)).mean()


def run_batch_ae_truncated(
    model: torch.nn.Module,
    priors: list[dict[str, object]],
    clips: list[tl.MotionClip],
    batch: list[torch.Tensor],
    cfg: tl.TrainConfig,
    rollout_k: int,
    device: torch.device,
    compute_diagnostics: bool = True,
    compatibility_score_weight: float = 0.0,
    reset_sampler=None,
) -> tuple[torch.Tensor, dict[str, float]]:
    clip_indices, starts = batch[0], batch[1]
    init_clip_indices = batch[2] if len(batch) >= 4 else None
    init_starts = batch[3] if len(batch) >= 4 else None
    total_loss = torch.zeros((), device=device)
    groups = {}
    for row, ci in enumerate(clip_indices.tolist()):
        groups.setdefault(ci, []).append(row)
    scores: list[torch.Tensor] = []
    motion_sizes: list[torch.Tensor] = []
    joint_rmses: list[torch.Tensor] = []
    ee_rmses: list[torch.Tensor] = []
    output_mses: list[torch.Tensor] = []
    footslide_losses: list[torch.Tensor] = []
    active_step_counts: list[torch.Tensor] = []

    def select_pose_rows(pose: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return {key: value[mask] for key, value in pose.items()}

    def assign_pose_rows(pose: dict[str, torch.Tensor], row_indices: torch.Tensor, source: dict[str, torch.Tensor]) -> None:
        for key in pose:
            pose[key][row_indices] = source[key]

    def gather_initial_poses(row_t: torch.Tensor, fallback_clip: tl.MotionClip, fallback_start: torch.Tensor):
        if init_clip_indices is None or init_starts is None:
            prev_idx_local = fallback_start - 1
            cur_idx_local = fallback_start
            prev_pose_local = tl.get_pose_from_clip(fallback_clip, prev_idx_local, device)
            cur_pose_local = tl.get_pose_from_clip(fallback_clip, cur_idx_local, device)
            return tl.maybe_apply_initial_offsets(
                fallback_clip,
                prev_idx_local,
                cur_idx_local,
                prev_pose_local,
                cur_pose_local,
                cfg,
                device,
            )

        local_init_clip_indices = init_clip_indices[row_t]
        local_init_starts = init_starts[row_t]
        prev_pose_acc = None
        cur_pose_acc = None
        for init_ci in sorted(set(local_init_clip_indices.tolist())):
            local_rows = (local_init_clip_indices == init_ci).nonzero(as_tuple=False).flatten()
            local_rows_dev = local_rows.to(device)
            local_start = local_init_starts[local_rows].long().to(device)
            local_prev_idx = local_start - 1
            local_cur_idx = local_start
            init_clip = clips[int(init_ci)]
            prev_pose_local = tl.get_pose_from_clip(init_clip, local_prev_idx, device)
            cur_pose_local = tl.get_pose_from_clip(init_clip, local_cur_idx, device)
            prev_pose_local, cur_pose_local = tl.maybe_apply_initial_offsets(
                init_clip,
                local_prev_idx,
                local_cur_idx,
                prev_pose_local,
                cur_pose_local,
                cfg,
                device,
            )
            if prev_pose_acc is None:
                prev_pose_acc = {
                    key: torch.empty(
                        (row_t.numel(), *value.shape[1:]),
                        dtype=value.dtype,
                        device=device,
                    )
                    for key, value in prev_pose_local.items()
                }
                cur_pose_acc = {
                    key: torch.empty(
                        (row_t.numel(), *value.shape[1:]),
                        dtype=value.dtype,
                        device=device,
                    )
                    for key, value in cur_pose_local.items()
                }
            for key in prev_pose_acc:
                prev_pose_acc[key][local_rows_dev] = prev_pose_local[key]
                cur_pose_acc[key][local_rows_dev] = cur_pose_local[key]
        return prev_pose_acc, cur_pose_acc

    for ci, rows in groups.items():
        clip = clips[ci]
        row_t = torch.tensor(rows, dtype=torch.long)
        start = starts[row_t].long().to(device)
        prev_idx = start - 1
        cur_idx = start
        prev_pose, cur_pose = gather_initial_poses(row_t, clip, start)
        group_loss = torch.zeros((), device=device)
        cur_root_pos, cur_root_rot, _cur_yaw, _cur_heading = tl.root_state(clip, cur_idx, cfg, device)
        cur_global_pos, cur_global_rot, _cur_canon = tl.fk_from_pose(
            clip, cur_root_pos, cur_root_rot, cur_pose, device
        )
        if reset_sampler is not None:
            effective_lengths = torch.full_like(cur_idx, int(rollout_k))
        elif clip.cyclic_animation:
            effective_lengths = torch.full_like(cur_idx, int(rollout_k))
        else:
            effective_lengths = torch.clamp(clip_future_safe_current_max(clip, cfg) - cur_idx + 1, min=1, max=int(rollout_k))
        group_steps = int(effective_lengths.max().item())
        active_step_counts.append(effective_lengths.float().sum().detach())
        for step in range(group_steps):
            if reset_sampler is not None and step > 0 and (not clip.cyclic_animation):
                expired = cur_idx > clip_future_safe_current_max(clip, cfg)
                if bool(expired.any()):
                    expired_rows = expired.nonzero(as_tuple=False).flatten()
                    remaining_steps = int(rollout_k) - step
                    max_start = clip_sample_start_max(clip, cfg, remaining_steps, cfg.agent_min_cohort_steps)
                    reset_start = torch.randint(
                        1,
                        max_start + 1,
                        (int(expired_rows.numel()),),
                        dtype=cur_idx.dtype,
                        device=device,
                    )
                    reset_prev_idx = reset_start - 1
                    reset_cur_idx = reset_start
                    reset_prev_pose = tl.get_pose_from_clip(clip, reset_prev_idx, device)
                    reset_cur_pose = tl.get_pose_from_clip(clip, reset_cur_idx, device)
                    reset_prev_pose, reset_cur_pose = tl.maybe_apply_initial_offsets(
                        clip,
                        reset_prev_idx,
                        reset_cur_idx,
                        reset_prev_pose,
                        reset_cur_pose,
                        cfg,
                        device,
                    )
                    prev_idx[expired_rows] = reset_prev_idx
                    cur_idx[expired_rows] = reset_cur_idx
                    assign_pose_rows(prev_pose, expired_rows, reset_prev_pose)
                    assign_pose_rows(cur_pose, expired_rows, reset_cur_pose)
                    cur_root_pos, cur_root_rot, _cur_yaw, _cur_heading = tl.root_state(clip, cur_idx, cfg, device)
                    cur_global_pos, cur_global_rot, _cur_canon = tl.fk_from_pose(
                        clip, cur_root_pos, cur_root_rot, cur_pose, device
                    )
            active = step < effective_lengths
            if not bool(active.any()):
                continue
            inp = tl.build_input(clip, prev_idx, cur_idx, prev_pose, cur_pose, cfg, device)
            raw_out = tl.predict_next_raw(model, inp, cur_pose, cfg)
            pred_pose, raw_pose = tl.output_to_pose(raw_out, clip)
            target_idx = torch.where(active, cur_idx + 1, cur_idx)
            tensors = clip.tensors(device)
            root_pos, root_rot, _yaw, _heading = tl.root_state(clip, target_idx, cfg, device)
            pred_global_pos, pred_global_rot, pred_canon = tl.fk_from_pose(clip, root_pos, root_rot, pred_pose, device)
            next_pose = {
                "pelvis_pos": pred_pose["pelvis_pos"],
                "pelvis_rot6": pred_pose["pelvis_rot6"],
                "nonpelvis_rot6": pred_pose["nonpelvis_rot6"],
                "canon_pos": pred_canon,
                "contacts": pred_pose["contacts"],
            }
            features = tae.transition_feature_from_next_pose(
                clip, prev_idx, cur_idx, prev_pose, cur_pose, next_pose, cfg, device
            )
            prior_scores = [
                ae_score(
                    prior_info["model"],
                    prior_info["mean"],
                    prior_info["std"],
                    features[active],
                    cfg.ae_score_loss,
                    cfg.ae_huber_delta,
                    compatibility_score_weight,
                )
                for prior_info in priors
            ]
            score = torch.stack(prior_scores).mean()
            step_loss = cfg.ae_loss_weight * score
            simple_slide_loss = torch.zeros((), device=device)
            if cfg.simple_footslide_loss_weight > 0.0:
                simple_slide_loss = simple_generated_footslide_loss(
                    clip,
                    cur_global_pos[active],
                    cur_global_rot[active],
                    pred_global_pos[active],
                    pred_global_rot[active],
                    cfg.simple_footslide_threshold_mps,
                )
                step_loss = step_loss + cfg.simple_footslide_loss_weight * simple_slide_loss
            term_mask = torch.zeros(cur_idx.shape[0], dtype=torch.bool, device=device)
            if cfg.enable_contact_physics_losses:
                target_pose = tl.get_pose_from_clip(clip, target_idx[active], device)
                physics_cfg = copy.copy(cfg)
                physics_cfg.alpha0_pelvis_location = 0.0
                physics_cfg.alpha1_pelvis_rotation = 0.0
                physics_cfg.alpha2_pose_rotation = 0.0
                physics_cfg.alpha3_pose_6d_aux = 0.0
                physics_cfg.alpha4_end_effector_location = 0.0
                physics_cfg.alpha5_end_effector_rotation = 0.0
                physics_cfg.alpha6_full_body_location = 0.0
                physics_loss, _physics_parts, _physics_next_pose, active_term_mask = tl.compute_losses(
                    clip,
                    select_pose_rows(prev_pose, active),
                    select_pose_rows(cur_pose, active),
                    select_pose_rows(pred_pose, active),
                    select_pose_rows(raw_pose, active),
                    target_pose,
                    prev_idx[active],
                    cur_idx[active],
                    target_idx[active],
                    physics_cfg,
                    device,
                )
                term_mask[active] = active_term_mask
                step_loss = step_loss + physics_loss
            group_loss = group_loss + step_loss / rollout_k
            scores.append(score.detach())
            if cfg.simple_footslide_loss_weight > 0.0:
                footslide_losses.append(simple_slide_loss.detach())
            motion_sizes.append((next_pose["canon_pos"][active] - cur_pose["canon_pos"][active]).square().mean().sqrt().detach())
            if compute_diagnostics:
                target_pose = tl.get_pose_from_clip(clip, target_idx[active], device)
                target_global_pos, _target_global_rot = tl.global_from_clip(clip, target_idx[active], cfg, device)
                joint_rmses.append((pred_global_pos[active] - target_global_pos).square().sum(dim=-1).mean().sqrt().detach())
                ee_idx = tensors["end_effectors"]
                ee_rmses.append(
                    (
                        pred_global_pos[active].index_select(1, ee_idx)
                        - target_global_pos.index_select(1, ee_idx)
                    )
                    .square()
                    .sum(dim=-1)
                    .mean()
                    .sqrt()
                    .detach()
                )
                output_mses.append(
                    F.mse_loss(
                        tl.pose_target_output(select_pose_rows(next_pose, active)),
                        tl.pose_target_output(target_pose),
                    )
                    .detach()
                )
            if cfg.enable_early_termination and cfg.restart_on_termination and step + 1 < rollout_k and bool(term_mask.any()):
                remaining_steps = rollout_k - step - 1
                max_start = tl.clip_rollout_max_start(clip, remaining_steps, cfg)
                if max_start < 1:
                    term_mask = torch.zeros_like(term_mask)
                    continue
                restart_start = torch.randint(1, max_start + 1, cur_idx.shape, device=device)
                restart_prev_idx = restart_start - 1
                restart_cur_idx = restart_start
                restart_prev_pose = tl.get_pose_from_clip(clip, restart_prev_idx, device)
                restart_cur_pose = tl.get_pose_from_clip(clip, restart_cur_idx, device)
                restart_prev_pose, restart_cur_pose = tl.maybe_apply_initial_offsets(
                    clip,
                    restart_prev_idx,
                    restart_cur_idx,
                    restart_prev_pose,
                    restart_cur_pose,
                    cfg,
                    device,
                )
                prev_pose = tl.blend_pose_by_mask(cur_pose, restart_prev_pose, term_mask)
                cur_pose = tl.blend_pose_by_mask(next_pose, restart_cur_pose, term_mask)
                prev_idx = torch.where(term_mask, restart_prev_idx, cur_idx)
                cur_idx = torch.where(term_mask, restart_cur_idx, target_idx)
                cur_root_pos, cur_root_rot, _cur_yaw, _cur_heading = tl.root_state(clip, cur_idx, cfg, device)
                cur_global_pos, cur_global_rot, _cur_canon = tl.fk_from_pose(
                    clip, cur_root_pos, cur_root_rot, cur_pose, device
                )
                continue
            prev_pose = tl.blend_pose_by_mask(prev_pose, cur_pose, active)
            cur_pose = tl.blend_pose_by_mask(cur_pose, next_pose, active)
            prev_idx = torch.where(active, cur_idx, prev_idx)
            cur_idx = torch.where(active, target_idx, cur_idx)
            cur_global_pos = torch.where(active.reshape(-1, 1, 1), pred_global_pos, cur_global_pos)
            cur_global_rot = torch.where(active.reshape(-1, 1, 1, 1), pred_global_rot, cur_global_rot)
        total_loss = total_loss + group_loss
    total_loss = total_loss / max(1, len(groups))
    def mean_metric(values: list[torch.Tensor]) -> float:
        if not values:
            return 0.0
        return float(torch.stack(values).mean().cpu())

    return total_loss, {
        "total": float(total_loss.detach().cpu()),
        "ae_score": mean_metric(scores),
        "canon_step_rms": mean_metric(motion_sizes),
        "joint_rmse": mean_metric(joint_rmses),
        "ee_rmse": mean_metric(ee_rmses),
        "output_mse": mean_metric(output_mses),
        "simple_footslide": mean_metric(footslide_losses),
        "active_fraction": (
            float(torch.stack(active_step_counts).sum().item()) / max(1.0, float(rollout_k * len(clip_indices)))
            if active_step_counts
            else 0.0
        ),
    }


def run_batch_ae_resetting(
    model: torch.nn.Module,
    priors: list[dict[str, object]],
    clips: list[tl.MotionClip],
    batch: list[torch.Tensor],
    cfg: tl.TrainConfig,
    rollout_k: int,
    device: torch.device,
    reset_sampler,
    compute_diagnostics: bool = True,
    compatibility_score_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    clip_indices, starts = batch[0].long().cpu(), batch[1].long().cpu()
    init_clip_indices = batch[2].long().cpu() if len(batch) >= 4 else None
    init_starts = batch[3].long().cpu() if len(batch) >= 4 else None
    batch_size = int(clip_indices.shape[0])
    cur_clip_indices = clip_indices.clone()
    prev_idx_cpu = starts - 1
    cur_idx_cpu = starts.clone()
    total_loss = torch.zeros((), device=device)

    score_values: list[torch.Tensor] = []
    score_counts: list[int] = []
    motion_values: list[torch.Tensor] = []
    motion_counts: list[int] = []
    joint_values: list[torch.Tensor] = []
    joint_counts: list[int] = []
    ee_values: list[torch.Tensor] = []
    ee_counts: list[int] = []
    output_values: list[torch.Tensor] = []
    output_counts: list[int] = []
    footslide_values: list[torch.Tensor] = []
    footslide_counts: list[int] = []

    def select_pose_rows(pose: dict[str, torch.Tensor], rows_dev: torch.Tensor) -> dict[str, torch.Tensor]:
        return {key: value.index_select(0, rows_dev) for key, value in pose.items()}

    def assign_pose_rows(target: dict[str, torch.Tensor], rows_dev: torch.Tensor, source: dict[str, torch.Tensor]) -> None:
        for key in target:
            target[key][rows_dev] = source[key]

    def rows_by_clip(clip_ids: torch.Tensor) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = {}
        for row, ci in enumerate(clip_ids.tolist()):
            groups.setdefault(int(ci), []).append(row)
        return groups

    def gather_pose_pair(clip_ids: torch.Tensor, starts_cpu: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        prev_pose_acc = None
        cur_pose_acc = None
        for ci, rows in rows_by_clip(clip_ids).items():
            rows_cpu = torch.tensor(rows, dtype=torch.long)
            rows_dev = rows_cpu.to(device)
            local_start = starts_cpu.index_select(0, rows_cpu).to(device)
            local_prev_idx = local_start - 1
            local_cur_idx = local_start
            clip = clips[ci]
            prev_pose_local = tl.get_pose_from_clip(clip, local_prev_idx, device)
            cur_pose_local = tl.get_pose_from_clip(clip, local_cur_idx, device)
            prev_pose_local, cur_pose_local = tl.maybe_apply_initial_offsets(
                clip,
                local_prev_idx,
                local_cur_idx,
                prev_pose_local,
                cur_pose_local,
                cfg,
                device,
            )
            if prev_pose_acc is None:
                prev_pose_acc = {
                    key: torch.empty((clip_ids.numel(), *value.shape[1:]), dtype=value.dtype, device=device)
                    for key, value in prev_pose_local.items()
                }
                cur_pose_acc = {
                    key: torch.empty((clip_ids.numel(), *value.shape[1:]), dtype=value.dtype, device=device)
                    for key, value in cur_pose_local.items()
                }
            assign_pose_rows(prev_pose_acc, rows_dev, prev_pose_local)
            assign_pose_rows(cur_pose_acc, rows_dev, cur_pose_local)
        assert prev_pose_acc is not None and cur_pose_acc is not None
        return prev_pose_acc, cur_pose_acc

    if init_clip_indices is None or init_starts is None:
        prev_pose, cur_pose = gather_pose_pair(cur_clip_indices, starts)
    else:
        prev_pose, cur_pose = gather_pose_pair(init_clip_indices, init_starts)

    def reset_rows(rows: list[int], remaining_steps: int) -> None:
        nonlocal cur_clip_indices, prev_idx_cpu, cur_idx_cpu, prev_pose, cur_pose
        if not rows or remaining_steps <= 0:
            return
        rows_cpu = torch.tensor(rows, dtype=torch.long)
        rows_dev = rows_cpu.to(device)
        new_clip_ids, new_starts = reset_sampler(len(rows), max(1, int(remaining_steps)))
        new_clip_ids = new_clip_ids.long().cpu()
        new_starts = new_starts.long().cpu()
        new_prev_pose, new_cur_pose = gather_pose_pair(new_clip_ids, new_starts)
        cur_clip_indices[rows_cpu] = new_clip_ids
        prev_idx_cpu[rows_cpu] = new_starts - 1
        cur_idx_cpu[rows_cpu] = new_starts
        assign_pose_rows(prev_pose, rows_dev, new_prev_pose)
        assign_pose_rows(cur_pose, rows_dev, new_cur_pose)

    def weighted_metric(values: list[torch.Tensor], counts: list[int]) -> float:
        if not values:
            return 0.0
        denom = max(1, int(sum(counts)))
        return float((torch.stack(values).sum() / denom).cpu())

    for step in range(int(rollout_k)):
        if step > 0:
            expired_rows = [
                row
                for row, ci in enumerate(cur_clip_indices.tolist())
                if (not clips[int(ci)].cyclic_animation)
                and int(cur_idx_cpu[row]) > clip_future_safe_current_max(clips[int(ci)], cfg)
            ]
            reset_rows(expired_rows, int(rollout_k) - step)

        step_loss_sum = torch.zeros((), device=device)
        step_count = 0
        for ci, rows in rows_by_clip(cur_clip_indices).items():
            clip = clips[ci]
            rows_cpu = torch.tensor(rows, dtype=torch.long)
            rows_dev = rows_cpu.to(device)
            n_rows = int(rows_cpu.numel())
            local_prev_idx = prev_idx_cpu.index_select(0, rows_cpu).to(device)
            local_cur_idx = cur_idx_cpu.index_select(0, rows_cpu).to(device)
            local_target_idx = local_cur_idx + 1
            local_prev_pose = select_pose_rows(prev_pose, rows_dev)
            local_cur_pose = select_pose_rows(cur_pose, rows_dev)

            inp = tl.build_input(clip, local_prev_idx, local_cur_idx, local_prev_pose, local_cur_pose, cfg, device)
            raw_out = tl.predict_next_raw(model, inp, local_cur_pose, cfg)
            pred_pose, raw_pose = tl.output_to_pose(raw_out, clip)
            root_pos, root_rot, _yaw, _heading = tl.root_state(clip, local_target_idx, cfg, device)
            pred_global_pos, pred_global_rot, pred_canon = tl.fk_from_pose(clip, root_pos, root_rot, pred_pose, device)
            next_pose = {
                "pelvis_pos": pred_pose["pelvis_pos"],
                "pelvis_rot6": pred_pose["pelvis_rot6"],
                "nonpelvis_rot6": pred_pose["nonpelvis_rot6"],
                "canon_pos": pred_canon,
                "contacts": pred_pose["contacts"],
            }
            features = tae.transition_feature_from_next_pose(
                clip,
                local_prev_idx,
                local_cur_idx,
                local_prev_pose,
                local_cur_pose,
                next_pose,
                cfg,
                device,
            )
            prior_scores = [
                ae_score(
                    prior_info["model"],
                    prior_info["mean"],
                    prior_info["std"],
                    features,
                    cfg.ae_score_loss,
                    cfg.ae_huber_delta,
                    compatibility_score_weight,
                )
                for prior_info in priors
            ]
            score = torch.stack(prior_scores).mean()
            step_loss = cfg.ae_loss_weight * score
            simple_slide_loss = torch.zeros((), device=device)
            cur_root_pos, cur_root_rot, _cur_yaw, _cur_heading = tl.root_state(clip, local_cur_idx, cfg, device)
            cur_global_pos, cur_global_rot, _cur_canon = tl.fk_from_pose(
                clip, cur_root_pos, cur_root_rot, local_cur_pose, device
            )
            if cfg.simple_footslide_loss_weight > 0.0:
                simple_slide_loss = simple_generated_footslide_loss(
                    clip,
                    cur_global_pos,
                    cur_global_rot,
                    pred_global_pos,
                    pred_global_rot,
                    cfg.simple_footslide_threshold_mps,
                )
                step_loss = step_loss + cfg.simple_footslide_loss_weight * simple_slide_loss
                footslide_values.append(simple_slide_loss.detach() * n_rows)
                footslide_counts.append(n_rows)

            if cfg.enable_contact_physics_losses:
                target_pose = tl.get_pose_from_clip(clip, local_target_idx, device)
                physics_cfg = copy.copy(cfg)
                physics_cfg.alpha0_pelvis_location = 0.0
                physics_cfg.alpha1_pelvis_rotation = 0.0
                physics_cfg.alpha2_pose_rotation = 0.0
                physics_cfg.alpha3_pose_6d_aux = 0.0
                physics_cfg.alpha4_end_effector_location = 0.0
                physics_cfg.alpha5_end_effector_rotation = 0.0
                physics_cfg.alpha6_full_body_location = 0.0
                physics_loss, _physics_parts, _physics_next_pose, _term_mask = tl.compute_losses(
                    clip,
                    local_prev_pose,
                    local_cur_pose,
                    pred_pose,
                    raw_pose,
                    target_pose,
                    local_prev_idx,
                    local_cur_idx,
                    local_target_idx,
                    physics_cfg,
                    device,
                )
                step_loss = step_loss + physics_loss

            step_loss_sum = step_loss_sum + step_loss * n_rows
            step_count += n_rows
            score_values.append(score.detach() * n_rows)
            score_counts.append(n_rows)
            motion_values.append((next_pose["canon_pos"] - local_cur_pose["canon_pos"]).square().mean().sqrt().detach() * n_rows)
            motion_counts.append(n_rows)
            if compute_diagnostics:
                tensors = clip.tensors(device)
                target_pose = tl.get_pose_from_clip(clip, local_target_idx, device)
                target_global_pos, _target_global_rot = tl.global_from_clip(clip, local_target_idx, cfg, device)
                joint_values.append((pred_global_pos - target_global_pos).square().sum(dim=-1).mean().sqrt().detach() * n_rows)
                joint_counts.append(n_rows)
                ee_idx = tensors["end_effectors"]
                ee_values.append(
                    (
                        pred_global_pos.index_select(1, ee_idx) - target_global_pos.index_select(1, ee_idx)
                    )
                    .square()
                    .sum(dim=-1)
                    .mean()
                    .sqrt()
                    .detach()
                    * n_rows
                )
                ee_counts.append(n_rows)
                output_values.append(
                    F.mse_loss(tl.pose_target_output(next_pose), tl.pose_target_output(target_pose)).detach() * n_rows
                )
                output_counts.append(n_rows)

            assign_pose_rows(prev_pose, rows_dev, local_cur_pose)
            assign_pose_rows(cur_pose, rows_dev, next_pose)
            prev_idx_cpu[rows_cpu] = local_cur_idx.detach().cpu()
            cur_idx_cpu[rows_cpu] = local_target_idx.detach().cpu()

        if step_count > 0:
            total_loss = total_loss + (step_loss_sum / step_count) / max(1, int(rollout_k))

    return total_loss, {
        "total": float(total_loss.detach().cpu()),
        "ae_score": weighted_metric(score_values, score_counts),
        "canon_step_rms": weighted_metric(motion_values, motion_counts),
        "joint_rmse": weighted_metric(joint_values, joint_counts),
        "ee_rmse": weighted_metric(ee_values, ee_counts),
        "output_mse": weighted_metric(output_values, output_counts),
        "simple_footslide": weighted_metric(footslide_values, footslide_counts),
        "active_fraction": 1.0,
    }


def run_batch_ae_packed(
    model: torch.nn.Module,
    priors: list[dict[str, object]],
    packed: PackedClips,
    batch: list[torch.Tensor],
    cfg: tl.TrainConfig,
    rollout_k: int,
    device: torch.device,
    compute_diagnostics: bool = True,
    compatibility_score_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    clip_ids = batch[0].long().to(device)
    starts = batch[1].long().to(device)
    init_clip_ids = batch[2].long().to(device) if len(batch) >= 4 else None
    init_starts = batch[3].long().to(device) if len(batch) >= 4 else None
    prev_idx = starts - 1
    cur_idx = starts.clone()
    if init_clip_ids is None or init_starts is None:
        prev_pose = packed.get_pose(clip_ids, prev_idx)
        cur_pose = packed.get_pose(clip_ids, cur_idx)
    else:
        prev_pose = packed.get_pose(init_clip_ids, init_starts - 1)
        cur_pose = packed.get_pose(init_clip_ids, init_starts)

    total_loss = torch.zeros((), device=device)
    score_values: list[torch.Tensor] = []
    motion_values: list[torch.Tensor] = []
    joint_values: list[torch.Tensor] = []
    ee_values: list[torch.Tensor] = []
    output_values: list[torch.Tensor] = []
    footslide_values: list[torch.Tensor] = []

    def reset_rows(mask: torch.Tensor, remaining_steps: int) -> None:
        nonlocal clip_ids, prev_idx, cur_idx, prev_pose, cur_pose
        if not bool(mask.any()) or remaining_steps <= 0:
            return
        rows = mask.nonzero(as_tuple=False).flatten()
        new_clip_ids, new_starts = packed.sample_starts(int(rows.numel()), max(1, int(remaining_steps)))
        new_prev_pose = packed.get_pose(new_clip_ids, new_starts - 1)
        new_cur_pose = packed.get_pose(new_clip_ids, new_starts)
        clip_ids[rows] = new_clip_ids
        prev_idx[rows] = new_starts - 1
        cur_idx[rows] = new_starts
        for key in prev_pose:
            replaced_prev = prev_pose[key].clone()
            replaced_cur = cur_pose[key].clone()
            replaced_prev[rows] = new_prev_pose[key]
            replaced_cur[rows] = new_cur_pose[key]
            prev_pose[key] = replaced_prev
            cur_pose[key] = replaced_cur

    for step in range(int(rollout_k)):
        if step > 0:
            safe_max = packed.future_safe_max.index_select(0, clip_ids)
            expired = torch.logical_and(~packed.cyclic.index_select(0, clip_ids), cur_idx > safe_max)
            reset_rows(expired, int(rollout_k) - step)

        inp = packed_build_input(packed, clip_ids, prev_idx, cur_idx, prev_pose, cur_pose, cfg)
        raw_out = tl.predict_next_raw(model, inp, cur_pose, cfg)
        pred_pose, raw_pose = tl.output_to_pose(raw_out, packed.prototype)
        target_idx = cur_idx + 1
        root_pos, root_rot, _yaw, _heading = packed.root_state(clip_ids, target_idx)
        pred_global_pos, pred_global_rot, pred_canon = packed.fk_from_pose(clip_ids, root_pos, root_rot, pred_pose)
        next_pose = {
            "pelvis_pos": pred_pose["pelvis_pos"],
            "pelvis_rot6": pred_pose["pelvis_rot6"],
            "nonpelvis_rot6": pred_pose["nonpelvis_rot6"],
            "canon_pos": pred_canon,
            "contacts": pred_pose["contacts"],
        }
        features = packed_transition_feature_from_next_pose(
            packed,
            clip_ids,
            prev_idx,
            cur_idx,
            prev_pose,
            cur_pose,
            next_pose,
            cfg,
        )
        prior_scores = [
            ae_score(
                prior_info["model"],
                prior_info["mean"],
                prior_info["std"],
                features,
                cfg.ae_score_loss,
                cfg.ae_huber_delta,
                compatibility_score_weight,
            )
            for prior_info in priors
        ]
        score = torch.stack(prior_scores).mean()
        step_loss = cfg.ae_loss_weight * score
        simple_slide_loss = torch.zeros((), device=device)
        cur_root_pos, cur_root_rot, _cur_yaw, _cur_heading = packed.root_state(clip_ids, cur_idx)
        cur_global_pos, cur_global_rot, _cur_canon = packed.fk_from_pose(clip_ids, cur_root_pos, cur_root_rot, cur_pose)
        if cfg.simple_footslide_loss_weight > 0.0:
            simple_slide_loss = packed_generated_footslide_loss(
                packed,
                cur_global_pos,
                cur_global_rot,
                pred_global_pos,
                pred_global_rot,
                cfg.simple_footslide_threshold_mps,
                clip_ids,
            )
            step_loss = step_loss + cfg.simple_footslide_loss_weight * simple_slide_loss

        if cfg.enable_contact_physics_losses:
            raise NotImplementedError("Packed AE rollouts currently require --no-contact-physics-losses")

        total_loss = total_loss + step_loss / max(1, int(rollout_k))
        score_values.append(score.detach())
        footslide_values.append(simple_slide_loss.detach())
        motion_values.append((next_pose["canon_pos"] - cur_pose["canon_pos"]).square().mean().sqrt().detach())
        if compute_diagnostics:
            target_pose = packed.get_pose(clip_ids, target_idx)
            target_root_pos, target_root_rot, _target_yaw, _target_heading = packed.root_state(clip_ids, target_idx)
            target_global_pos, _target_global_rot, _target_canon = packed.fk_from_pose(
                clip_ids,
                target_root_pos,
                target_root_rot,
                target_pose,
            )
            joint_values.append((pred_global_pos - target_global_pos).square().sum(dim=-1).mean().sqrt().detach())
            ee_idx = packed.prototype.end_effectors_tensor.to(device)
            ee_values.append(
                (
                    pred_global_pos.index_select(1, ee_idx)
                    - target_global_pos.index_select(1, ee_idx)
                )
                .square()
                .sum(dim=-1)
                .mean()
                .sqrt()
                .detach()
            )
            output_values.append(
                F.mse_loss(tl.pose_target_output(next_pose), tl.pose_target_output(target_pose)).detach()
            )

        prev_pose = cur_pose
        cur_pose = next_pose
        prev_idx = cur_idx
        cur_idx = target_idx

    def mean_metric(values: list[torch.Tensor]) -> float:
        if not values:
            return 0.0
        return float(torch.stack(values).mean().cpu())

    return total_loss, {
        "total": float(total_loss.detach().cpu()),
        "ae_score": mean_metric(score_values),
        "canon_step_rms": mean_metric(motion_values),
        "joint_rmse": mean_metric(joint_values),
        "ee_rmse": mean_metric(ee_values),
        "output_mse": mean_metric(output_values),
        "simple_footslide": mean_metric(footslide_values),
        "active_fraction": 1.0,
    }


def run_batch_ae(
    model: torch.nn.Module,
    priors: list[dict[str, object]],
    clips: list[tl.MotionClip],
    batch: list[torch.Tensor],
    cfg: tl.TrainConfig,
    rollout_k: int,
    device: torch.device,
    compute_diagnostics: bool = True,
    compatibility_score_weight: float = 0.0,
    reset_sampler=None,
    packed: PackedClips | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if packed is not None:
        return run_batch_ae_packed(
            model,
            priors,
            packed,
            batch,
            cfg,
            rollout_k,
            device,
            compute_diagnostics=compute_diagnostics,
            compatibility_score_weight=compatibility_score_weight,
        )
    return run_batch_ae_truncated(
        model,
        priors,
        clips,
        batch,
        cfg,
        rollout_k,
        device,
        compute_diagnostics=compute_diagnostics,
        compatibility_score_weight=compatibility_score_weight,
        reset_sampler=reset_sampler,
    )


def train(args: argparse.Namespace) -> None:
    process_start_time = time.perf_counter()
    cfg = tl.TrainConfig()
    cfg.hidden_dim = args.hidden_dim
    cfg.num_hidden_layers = args.num_hidden_layers
    cfg.learning_rate = args.learning_rate
    cfg.batch_size = args.batch_size
    cfg.max_epochs = args.max_epochs
    cfg.rollout_schedule = tl.parse_rollout_schedule(args.rollout_schedule)
    cfg.curriculum_max_epochs_per_stage = args.curriculum_max_epochs_per_stage
    cfg.curriculum_stall_patience_epochs = args.curriculum_stall_patience_epochs
    cfg.curriculum_min_delta = args.curriculum_min_delta
    cfg.curriculum_min_epochs = args.curriculum_min_epochs
    cfg.val_fraction = 0.0
    cfg.disable_validation = True
    cfg.use_torch_compile = bool(args.compile) and not args.no_compile
    cfg.torch_compile_mode = args.compile_mode
    cfg.predict_residual = args.predict_residual
    cfg.zero_init_output = args.zero_init_output
    cfg.run_name = args.run_name
    if args.date_prefix_run_name:
        cfg.run_name = tl.date_prefixed_run_name(cfg.run_name)
    cfg.device = args.device
    cfg.cyclic_animation = args.cyclic_animation
    cfg.training_loop = args.training_loop
    cfg.agent_sampling = args.agent_sampling
    cfg.agent_min_cohort_steps = max(1, int(args.agent_min_cohort_steps))
    cfg.gradient_accumulation_batches = max(1, int(args.gradient_accumulation_batches))
    cfg.periodic_sampling_weight = max(0.0, float(args.periodic_sampling_weight))
    cfg.nonperiodic_sampling_weight = max(0.0, float(args.nonperiodic_sampling_weight))
    cfg.init_pose_sampling = args.init_pose_sampling
    cfg.enable_contact_physics_losses = not args.no_contact_physics_losses
    cfg.enable_early_termination = args.enable_early_termination
    cfg.restart_on_termination = not args.no_restart_on_termination
    cfg.reset_exhausted_agents = bool(args.reset_exhausted_agents)
    cfg.freefall_body_height_offset_m = args.freefall_body_height_offset_m
    cfg.freefall_initial_offset_history = max(1, int(args.freefall_initial_offset_history))
    cfg.freefall_initial_contacts_off = bool(args.freefall_initial_contacts_off)
    cfg.alpha7_contact_label = args.alpha7_contact_label
    cfg.alpha8_foot_penetration = args.alpha8_foot_penetration
    cfg.alpha9_foot_sliding = args.alpha9_foot_sliding
    cfg.alpha10_freefall = args.alpha10_freefall
    cfg.alpha11_contact_height = args.alpha11_contact_height
    cfg.alpha12_termination = args.alpha12_termination
    cfg.ae_loss_weight = args.ae_loss_weight
    cfg.ae_score_loss = args.ae_score_loss
    cfg.ae_huber_delta = args.ae_huber_delta
    cfg.simple_footslide_loss_weight = args.simple_footslide_loss_weight
    cfg.simple_footslide_threshold_mps = args.simple_footslide_threshold_mps
    cfg.simple_footslide_gt_margin = args.simple_footslide_gt_margin
    cfg.timed_checkpoint_interval_minutes = max(0.0, float(args.timed_checkpoint_interval_minutes))
    if cfg.init_pose_sampling != "same_clip" and cfg.training_loop != "agents":
        raise ValueError("--init-pose-sampling random_dataset currently requires --training-loop agents")
    tl.set_seed(cfg.seed)
    device = torch.device(cfg.device)
    tl.apply_cuda_performance_settings(cfg, device)
    profiler = tl.TimingProfiler(args.profile_timing, device, args.profile_sync_cuda)

    clip_specs = tl.clip_specs_from_folders(args.folder_path, args.periodic_folder_path, args.nonperiodic_folder_path)
    with profiler.section("setup/load_npz_and_prior"):
        clips = tl.load_clips_from_specs(clip_specs, cfg)
        prior_paths = [tl.resolve_path(args.prior_checkpoint)]
        prior_paths.extend(tl.resolve_path(path) for path in args.extra_prior_checkpoint)
        priors = load_prior_bundle(prior_paths, device)
        if cfg.simple_footslide_loss_weight > 0.0 and cfg.simple_footslide_threshold_mps <= 0.0:
            cfg.simple_footslide_threshold_mps = compute_groundtruth_support_slide_threshold(
                clips,
                cfg,
                device,
                cfg.simple_footslide_gt_margin,
            )
            print(
                f"auto simple_footslide_threshold_mps={cfg.simple_footslide_threshold_mps:.6g} "
                f"(margin={cfg.simple_footslide_gt_margin:.3f})",
                flush=True,
            )

    with profiler.section("setup/model_optimizer_compile"):
        input_dim, output_dim = tl.make_batch_dims(clips[0], cfg)
        model = tl.MLPController(input_dim, output_dim, cfg).to(device)
        model, compile_enabled = tl.maybe_compile_model(model, input_dim, cfg, device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if args.resume_checkpoint:
        resume_path = tl.resolve_path(args.resume_checkpoint)
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        tl.unwrap_compiled_model(model).load_state_dict(resume["model"])
        print(f"resumed model weights from {resume_path}", flush=True)
        if args.resume_optimizer:
            if "optimizer" not in resume:
                raise KeyError(f"checkpoint has no optimizer state: {resume_path}")
            opt.load_state_dict(resume["optimizer"])
            print(f"resumed optimizer state from {resume_path}", flush=True)
    run_dir = tl.resolve_path(cfg.output_dir) / cfg.run_name
    ckpt_dir = run_dir / "checkpoints"
    writer = SummaryWriter(run_dir / "tb")
    schedule = tuple(max(1, int(k)) for k in cfg.rollout_schedule) or (1,)
    visual_report_bridge = None
    if args.visual_reporter:
        try:
            visual_report_bridge = VisualReportBridge(
                run_dir,
                npz_path=clips[0].path,
                interval_seconds=args.visual_report_interval_seconds,
                device=args.visual_report_device,
                max_frames=args.visual_report_max_frames,
            )
            visual_report_bridge.start()
        except Exception as exc:
            print(f"visual reporter disabled: {exc}", flush=True)
    metadata = {
        "npz_folders": [
            {"path": str(tl.npz_folder_from_path(path)), "cyclic": cyclic}
            for path, cyclic in clip_specs
        ],
        "body_names": clips[0].body_names,
        "parents_body": clips[0].parents_body.tolist(),
        "pelvis_index": clips[0].pelvis,
        "non_pelvis_indices": clips[0].non_pelvis,
        "end_effector_indices": clips[0].end_effectors,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "ae_prior_checkpoint": str(prior_paths[0]),
        "ae_prior_checkpoints": [str(path) for path in prior_paths],
        "ae_prior_weights": [1.0 / len(prior_paths) for _path in prior_paths],
        "loss_type": "pure_transition_ae_prior",
        "ae_score_loss": args.ae_score_loss,
        "ae_huber_delta": args.ae_huber_delta,
        "simple_footslide_loss_weight": cfg.simple_footslide_loss_weight,
        "simple_footslide_threshold_mps": cfg.simple_footslide_threshold_mps,
        "simple_footslide_gt_margin": cfg.simple_footslide_gt_margin,
        "init_pose_sampling": cfg.init_pose_sampling,
        "agent_min_cohort_steps": cfg.agent_min_cohort_steps,
        "gradient_accumulation_batches": cfg.gradient_accumulation_batches,
        "periodic_sampling_weight": cfg.periodic_sampling_weight,
        "nonperiodic_sampling_weight": cfg.nonperiodic_sampling_weight,
        "rollout_truncation_policy": (
            "nonperiodic rows reset independently to fresh starts in the same clip before target or future-root frames are missing; periodic rows use cyclic indexing"
            if cfg.reset_exhausted_agents
            else "nonperiodic rows contribute until their future-root window would pass clip end; periodic rows use full requested K"
        ),
        "agent_clip_sampling_policy": (
            (
                "packed per-agent random clips: periodic group total weight "
                f"{cfg.periodic_sampling_weight:g}, nonperiodic group total weight {cfg.nonperiodic_sampling_weight:g}"
            )
            if cfg.reset_exhausted_agents
            else "weighted by inverse expected active rollout steps to reduce truncation imbalance"
        ),
        "packed_agent_rollout": bool(
            args.packed_agent_rollout
            and
            cfg.training_loop == "agents"
            and cfg.agent_sampling == "random"
            and args.agent_batch_clips != 1
            and not cfg.enable_contact_physics_losses
        ),
        "final_stage_random_rollout": bool(args.final_stage_random_rollout),
        "final_stage_random_rollout_choices": list(schedule),
        "compile_enabled": compile_enabled,
    }

    rollout_idx = 0
    if args.initial_rollout_k is not None:
        initial_k = max(1, int(args.initial_rollout_k))
        if initial_k not in schedule:
            raise ValueError(f"--initial-rollout-k {initial_k} is not present in --rollout-schedule {schedule}")
        rollout_idx = schedule.index(initial_k)
    rollout_k = schedule[rollout_idx]
    packed = None
    if (
        args.packed_agent_rollout
        and
        cfg.training_loop == "agents"
        and cfg.agent_sampling == "random"
        and args.agent_batch_clips != 1
        and not cfg.enable_contact_physics_losses
    ):
        with profiler.section("setup/pack_clips"):
            packed = PackedClips(clips, cfg, device)
    agent_rng = random.Random(cfg.seed + 4817)
    agent_coverage_order: list[tuple[int, int]] = []
    agent_coverage_cursor = 0
    stage_clip_indices: list[int] = []
    stage_clip_weights: list[float] = []

    def make_loader(max_rollout: int) -> tuple[tl.MotionIndexDataset, DataLoader]:
        dataset = AnyStartDataset(clips, cfg, max_rollout)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        return dataset, loader

    def refresh_stage_sampling(max_rollout: int) -> None:
        nonlocal stage_clip_indices, stage_clip_weights
        stage_clip_indices = [ci for ci, clip in enumerate(clips) if clip_supports_any_start(clip, cfg)]
        if not stage_clip_indices:
            raise ValueError("No clips support one-step training")
        periodic_indices = [ci for ci in stage_clip_indices if clips[ci].cyclic_animation]
        nonperiodic_indices = [ci for ci in stage_clip_indices if not clips[ci].cyclic_animation]
        group_counts = {
            True: max(1, len(periodic_indices)),
            False: max(1, len(nonperiodic_indices)),
        }
        group_weights = {
            True: cfg.periodic_sampling_weight,
            False: cfg.nonperiodic_sampling_weight,
        }
        if cfg.reset_exhausted_agents:
            stage_clip_weights = [
                group_weights[clips[ci].cyclic_animation] / group_counts[clips[ci].cyclic_animation]
                for ci in stage_clip_indices
            ]
        else:
            expected_steps = [
                expected_active_rollout_steps(clips[ci], cfg, max_rollout, cfg.agent_min_cohort_steps)
                for ci in stage_clip_indices
            ]
            stage_clip_weights = [
                group_weights[clips[ci].cyclic_animation]
                / group_counts[clips[ci].cyclic_animation]
                / max(1e-6, steps)
                for ci, steps in zip(stage_clip_indices, expected_steps)
            ]
        if not any(weight > 0.0 for weight in stage_clip_weights):
            stage_clip_weights = [1.0 for _ in stage_clip_indices]
        if packed is not None:
            packed.update_stage_sampling(stage_clip_indices, stage_clip_weights)

    def reset_agent_coverage_order(dataset: tl.MotionIndexDataset) -> None:
        nonlocal agent_coverage_order, agent_coverage_cursor
        agent_coverage_order = list(dataset.items)
        agent_rng.shuffle(agent_coverage_order)
        agent_coverage_cursor = 0

    def coverage_agent_start(dataset: tl.MotionIndexDataset) -> tuple[int, int]:
        nonlocal agent_coverage_cursor
        if not agent_coverage_order or agent_coverage_cursor >= len(agent_coverage_order):
            reset_agent_coverage_order(dataset)
        item = agent_coverage_order[agent_coverage_cursor]
        agent_coverage_cursor += 1
        return item

    def random_agent_start(max_rollout: int, clip_index: int | None = None) -> tuple[int, int]:
        if clip_index is None:
            ci = agent_rng.choices(stage_clip_indices, weights=stage_clip_weights, k=1)[0]
        else:
            ci = int(clip_index)
        max_start = clip_sample_start_max(clips[ci], cfg, max_rollout, cfg.agent_min_cohort_steps)
        return ci, agent_rng.randint(1, max_start)

    def random_initial_pose_start() -> tuple[int, int]:
        ci = agent_rng.randrange(len(clips))
        max_start = max(1, clips[ci].cyclic_period - 1 if clips[ci].cyclic_animation else clips[ci].T - 1)
        return ci, agent_rng.randint(1, max_start)

    def random_agent_reset_batch(count: int, max_rollout: int) -> tuple[torch.Tensor, torch.Tensor]:
        ci = agent_rng.choices(stage_clip_indices, weights=stage_clip_weights, k=1)[0]
        max_start = clip_sample_start_max(clips[ci], cfg, max_rollout, cfg.agent_min_cohort_steps)
        clip_ids = [ci] * int(count)
        starts = []
        for _ in range(int(count)):
            starts.append(agent_rng.randint(1, max_start))
        return torch.tensor(clip_ids, dtype=torch.long), torch.tensor(starts, dtype=torch.long)

    def agent_batch(dataset: tl.MotionIndexDataset, max_rollout: int) -> tuple[torch.Tensor, ...]:
        clip_ids = []
        starts = []
        init_clip_ids = []
        init_starts = []
        fixed_clip = None
        if args.agent_batch_clips == 1 and cfg.agent_sampling == "random" and len(clips) > 1:
            fixed_clip = agent_rng.choices(stage_clip_indices, weights=stage_clip_weights, k=1)[0]
        for _ in range(cfg.batch_size):
            if cfg.agent_sampling == "coverage":
                ci, start = coverage_agent_start(dataset)
            else:
                ci, start = random_agent_start(max_rollout, fixed_clip)
            clip_ids.append(ci)
            starts.append(start)
            if cfg.init_pose_sampling == "random_dataset":
                init_ci, init_start = random_initial_pose_start()
                init_clip_ids.append(init_ci)
                init_starts.append(init_start)
        if cfg.init_pose_sampling == "random_dataset":
            return (
                torch.tensor(clip_ids, dtype=torch.long),
                torch.tensor(starts, dtype=torch.long),
                torch.tensor(init_clip_ids, dtype=torch.long),
                torch.tensor(init_starts, dtype=torch.long),
            )
        return torch.tensor(clip_ids, dtype=torch.long), torch.tensor(starts, dtype=torch.long)

    def sample_effective_rollout_k() -> int:
        if not args.final_stage_random_rollout or rollout_idx != len(schedule) - 1:
            return int(rollout_k)
        return int(schedule[agent_rng.randrange(len(schedule))])

    refresh_stage_sampling(rollout_k)
    dataset, loader = make_loader(rollout_k)
    print(
        f"pure_ae run={cfg.run_name} priors={len(priors)} primary_prior={args.prior_checkpoint} K={rollout_k} "
        f"samples={len(dataset)} loop={cfg.training_loop} packed={packed is not None}",
        flush=True,
    )
    best = math.inf
    stalls = 0
    stage_start_epoch = 1
    stage_batches = 0
    start_time = time.perf_counter()
    timed_interval_seconds = 60.0 * max(0.0, float(cfg.timed_checkpoint_interval_minutes))
    next_timed_checkpoint_at = start_time + timed_interval_seconds if timed_interval_seconds > 0.0 else math.inf
    last_diagnostics = {
        "joint_rmse": 0.0,
        "ee_rmse": 0.0,
        "output_mse": 0.0,
    }
    for epoch in range(1, cfg.max_epochs + 1):
        parts = []
        model.train()
        compute_diagnostics = (
            args.best_metric != "ae_score"
            or args.diagnostic_metrics_every_epochs == 1
            or (
                args.diagnostic_metrics_every_epochs > 1
                and (epoch == 1 or epoch % args.diagnostic_metrics_every_epochs == 0)
            )
        )
        if cfg.training_loop == "agents":
            train_iter = []
            rollout_iter = []
            for _ in range(max(1, args.agent_batches_per_epoch) * cfg.gradient_accumulation_batches):
                effective_k = sample_effective_rollout_k()
                train_iter.append(agent_batch(dataset, effective_k))
                rollout_iter.append(effective_k)
        else:
            train_iter = list(loader)
            rollout_iter = [int(rollout_k) for _ in train_iter]
        stage_batches += len(train_iter)
        for accum_start in range(0, len(train_iter), cfg.gradient_accumulation_batches):
            accum_batches = train_iter[accum_start : accum_start + cfg.gradient_accumulation_batches]
            accum_rollout = rollout_iter[accum_start : accum_start + cfg.gradient_accumulation_batches]
            with profiler.section("train/zero_grad"):
                opt.zero_grad(set_to_none=True)
            for batch, effective_k in zip(accum_batches, accum_rollout):
                with profiler.section("train/forward_loss"):
                    loss, scalars = run_batch_ae(
                        model,
                        priors,
                        clips,
                        batch,
                        cfg,
                        effective_k,
                        device,
                        compute_diagnostics=compute_diagnostics,
                        compatibility_score_weight=args.compatibility_score_weight,
                        reset_sampler=random_agent_reset_batch if cfg.reset_exhausted_agents else None,
                        packed=packed,
                    )
                with profiler.section("train/backward"):
                    (loss / max(1, len(accum_batches))).backward()
                parts.append(scalars)
            with profiler.section("train/clip_grad"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            with profiler.section("train/optimizer_step"):
                opt.step()
        train_total = float(np.mean([p["total"] for p in parts]))
        train_score = float(np.mean([p["ae_score"] for p in parts]))
        motion_rms = float(np.mean([p["canon_step_rms"] for p in parts]))
        footslide_loss = float(np.mean([p["simple_footslide"] for p in parts]))
        active_fraction = float(np.mean([p["active_fraction"] for p in parts]))
        effective_rollout_mean = float(np.mean(rollout_iter)) if rollout_iter else float(rollout_k)
        effective_rollout_max = float(np.max(rollout_iter)) if rollout_iter else float(rollout_k)
        if compute_diagnostics:
            last_diagnostics["joint_rmse"] = float(np.mean([p["joint_rmse"] for p in parts]))
            last_diagnostics["ee_rmse"] = float(np.mean([p["ee_rmse"] for p in parts]))
            last_diagnostics["output_mse"] = float(np.mean([p["output_mse"] for p in parts]))
        joint_rmse = last_diagnostics["joint_rmse"]
        ee_rmse = last_diagnostics["ee_rmse"]
        output_mse = last_diagnostics["output_mse"]
        selection_metric = {
            "ae_score": train_total,
            "joint_rmse": joint_rmse,
            "ee_rmse": ee_rmse,
            "output_mse": output_mse,
        }[args.best_metric]
        writer.add_scalar("loss/train_total", train_total, epoch)
        writer.add_scalar("loss/ae_score", train_score, epoch)
        writer.add_scalar(
            "loss/weighted_simple_footslide",
            cfg.simple_footslide_loss_weight * footslide_loss,
            epoch,
        )
        writer.add_scalar("curriculum/active_fraction", active_fraction, epoch)
        writer.add_scalar("optim/gradient_accumulation_batches", cfg.gradient_accumulation_batches, epoch)
        writer.add_scalar("monitor/simple_footslide_threshold_mps", cfg.simple_footslide_threshold_mps, epoch)
        writer.add_scalar("motion/canon_step_rms", motion_rms, epoch)
        if compute_diagnostics:
            writer.add_scalar("accuracy/joint_rmse_m", joint_rmse, epoch)
            writer.add_scalar("accuracy/end_effector_rmse_m", ee_rmse, epoch)
            writer.add_scalar("accuracy/output_mse", output_mse, epoch)
        writer.add_scalar(f"selection/{args.best_metric}", selection_metric, epoch)
        writer.add_scalar("curriculum/rollout_k", rollout_k, epoch)
        writer.add_scalar("curriculum/effective_rollout_k_mean", effective_rollout_mean, epoch)
        writer.add_scalar("curriculum/effective_rollout_k_max", effective_rollout_max, epoch)
        eligible_clip_count = max(1, sum(clip_supports_any_start(clip, cfg) for clip in clips))
        min_stage_batches = (
            math.ceil(float(args.curriculum_min_eligible_clip_visits) * eligible_clip_count)
            if args.curriculum_min_eligible_clip_visits > 0.0
            else 0
        )
        writer.add_scalar("curriculum/stage_batches", stage_batches, epoch)
        writer.add_scalar("curriculum/eligible_clips", eligible_clip_count, epoch)
        if min_stage_batches > 0:
            writer.add_scalar("curriculum/min_stage_batches", min_stage_batches, epoch)
        improved = selection_metric < best - cfg.curriculum_min_delta
        stalls = 0 if improved else stalls + 1
        if selection_metric < best:
            best = selection_metric
            tl.save_checkpoint(ckpt_dir / "checkpoint_best.pt", model, opt, epoch, best, rollout_k, cfg, metadata)
            tl.save_checkpoint(ckpt_dir / f"checkpoint_best_k{rollout_k:02d}.pt", model, opt, epoch, best, rollout_k, cfg, metadata)
        save_live_period = args.save_live_every_epochs
        if args.visual_reporter and save_live_period <= 0:
            save_live_period = args.visual_report_save_every_epochs
        if save_live_period > 0 and epoch % save_live_period == 0:
            last_path = ckpt_dir / "checkpoint_last.pt"
            tl.save_checkpoint(last_path, model, opt, epoch, best, rollout_k, cfg, metadata)
            refresh_live_viewer(args, last_path)
        now_perf = time.perf_counter()
        if now_perf >= next_timed_checkpoint_at:
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            tl.save_checkpoint(
                ckpt_dir / f"checkpoint_time_{stamp}_epoch_{epoch:06d}.pt",
                model,
                opt,
                epoch,
                best,
                rollout_k,
                cfg,
                metadata,
            )
            while next_timed_checkpoint_at <= now_perf:
                next_timed_checkpoint_at += timed_interval_seconds
        stage_epochs = epoch - stage_start_epoch + 1
        min_stage_reached = stage_epochs >= cfg.curriculum_min_epochs and stage_batches >= min_stage_batches
        threshold_reached = (
            args.curriculum_threshold is not None
            and min_stage_reached
            and train_total <= float(args.curriculum_threshold)
        )
        can_advance = threshold_reached or (
            cfg.curriculum_max_epochs_per_stage > 0
            and stage_epochs >= cfg.curriculum_max_epochs_per_stage
            and stage_batches >= min_stage_batches
        ) or (
            cfg.curriculum_stall_patience_epochs > 0
            and min_stage_reached
            and stalls >= cfg.curriculum_stall_patience_epochs
        )
        print(
            f"epoch={epoch:04d} K={rollout_k:02d} ae={train_total:.6g} best_{args.best_metric}={best:.6g} "
            f"ae_score={train_score:.6g} footslide={footslide_loss:.6g} "
            f"joint_rmse={joint_rmse:.6g} ee_rmse={ee_rmse:.6g} motion_rms={motion_rms:.6g} "
            f"active={active_fraction:.3f} stalls={stalls} stage_batches={stage_batches}/{min_stage_batches} "
            f"eligible_clips={eligible_clip_count} effective_k={effective_rollout_mean:.1f} "
            f"elapsed_s={time.perf_counter() - start_time:.1f}",
            flush=True,
        )
        if can_advance and rollout_idx < len(schedule) - 1:
            rollout_idx += 1
            rollout_k = schedule[rollout_idx]
            refresh_stage_sampling(rollout_k)
            dataset, loader = make_loader(rollout_k)
            reset_agent_coverage_order(dataset)
            best = math.inf
            stalls = 0
            stage_start_epoch = epoch + 1
            stage_batches = 0
            print(f"advanced rollout_k={rollout_k} samples={len(dataset)}", flush=True)
        elif can_advance and args.stop_on_final_stall:
            print(
                f"stopped on final stall epoch={epoch} K={rollout_k} "
                f"best_{args.best_metric}={best:.6g}",
                flush=True,
            )
            break
        if args.max_train_seconds > 0 and time.perf_counter() - start_time >= args.max_train_seconds:
            print(f"max_train_seconds reached {args.max_train_seconds}", flush=True)
            break
    last_path = ckpt_dir / "checkpoint_last.pt"
    tl.save_checkpoint(last_path, model, opt, epoch, best, rollout_k, cfg, metadata)
    refresh_live_viewer(args, last_path)
    writer.close()
    profiler.write_csv(run_dir / "timing_profile.csv", time.perf_counter() - process_start_time)
    if visual_report_bridge is not None:
        visual_report_bridge.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-path", default="data/npz_final")
    parser.add_argument(
        "--periodic-folder-path",
        default=None,
        help="Semicolon-separated motion folders that should use cyclic root/pose indexing.",
    )
    parser.add_argument(
        "--nonperiodic-folder-path",
        default=None,
        help="Semicolon-separated motion folders that use non-cyclic indexing; exhausted rows can reset to fresh starts.",
    )
    parser.add_argument("--prior-checkpoint", required=True)
    parser.add_argument(
        "--extra-prior-checkpoint",
        action="append",
        default=[],
        help="Additional frozen AE prior checkpoint. All priors are averaged with equal weight.",
    )
    parser.add_argument("--run-name", default="locomotion_pure_ae")
    parser.add_argument("--date-prefix-run-name", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cyclic-animation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--training-loop", choices=("sampled", "agents"), default="sampled")
    parser.add_argument("--agent-sampling", choices=("random", "coverage"), default="random")
    parser.add_argument("--agent-batches-per-epoch", type=int, default=1)
    parser.add_argument(
        "--packed-agent-rollout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use dense packed tensors for per-agent random multi-clip rollouts.",
    )
    parser.add_argument(
        "--agent-batch-clips",
        type=int,
        default=0,
        help=(
            "With random agent sampling, 0 means per-agent random clips using the packed rollout path; "
            "1 keeps the legacy one-clip cohort path."
        ),
    )
    parser.add_argument(
        "--agent-min-cohort-steps",
        type=int,
        default=8,
        help="For non-periodic one-clip agent cohorts, avoid starts that would force a reset before this many frames when possible.",
    )
    parser.add_argument(
        "--gradient-accumulation-batches",
        type=int,
        default=1,
        help="Average this many one-clip cohorts before one optimizer step.",
    )
    parser.add_argument(
        "--periodic-sampling-weight",
        type=float,
        default=1.0,
        help="Total sampling weight assigned to cyclic/periodic clips as a group.",
    )
    parser.add_argument(
        "--nonperiodic-sampling-weight",
        type=float,
        default=1.0,
        help="Total sampling weight assigned to non-cyclic transition clips as a group.",
    )
    parser.add_argument(
        "--init-pose-sampling",
        choices=("same_clip", "random_dataset"),
        default="same_clip",
        help="Initial body pose source. random_dataset keeps the sampled root clip but initializes the pose from any clip.",
    )
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--rollout-schedule", default="1")
    parser.add_argument(
        "--initial-rollout-k",
        type=int,
        default=None,
        help="Start a resumed curriculum at this scheduled K while keeping the full schedule for final-stage sampling.",
    )
    parser.add_argument("--curriculum-max-epochs-per-stage", type=int, default=120)
    parser.add_argument("--curriculum-stall-patience-epochs", type=int, default=60)
    parser.add_argument("--curriculum-min-epochs", type=int, default=30)
    parser.add_argument("--curriculum-min-delta", type=float, default=1e-6)
    parser.add_argument(
        "--curriculum-min-eligible-clip-visits",
        type=float,
        default=0.0,
        help="Minimum one-clip agent batches per eligible clip before rollout K can advance.",
    )
    parser.add_argument(
        "--curriculum-threshold",
        type=float,
        default=None,
        help="Optional train-total threshold for advancing rollout K after curriculum-min-epochs.",
    )
    parser.add_argument(
        "--final-stage-random-rollout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="At the last scheduled K, sample each agent microbatch rollout length from the rollout schedule rungs.",
    )
    parser.add_argument("--stop-on-final-stall", action="store_true")
    parser.add_argument("--max-train-seconds", type=float, default=0.0)
    parser.add_argument("--timed-checkpoint-interval-minutes", type=float, default=30.0)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Try torch.compile after a forward/backward probe.")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile.")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--best-metric", choices=("ae_score", "joint_rmse", "ee_rmse", "output_mse"), default="ae_score")
    parser.add_argument(
        "--diagnostic-metrics-every-epochs",
        type=int,
        default=10,
        help="Compute visual/GT diagnostic RMSE every N epochs for AE-prior runs. Set 1 for every epoch, 0 to disable unless best-metric needs it.",
    )
    parser.add_argument("--ae-score-loss", choices=("mse", "huber"), default="mse")
    parser.add_argument("--ae-huber-delta", type=float, default=1.0)
    parser.add_argument("--save-live-every-epochs", type=int, default=20)
    parser.add_argument("--live-viewer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--live-npz-path", default="data/npz_final/testcasc.npz")
    parser.add_argument("--live-output-path", default="training/runs/model_comparisons/model_comparison.html")
    parser.add_argument("--visual-reporter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visual-report-save-every-epochs", type=int, default=20)
    parser.add_argument("--visual-report-interval-seconds", type=float, default=60.0)
    parser.add_argument("--visual-report-device", default="cpu")
    parser.add_argument("--visual-report-max-frames", type=int, default=180)
    parser.add_argument("--profile-timing", action="store_true", help="Write timing_profile.csv in the run directory.")
    parser.add_argument(
        "--profile-sync-cuda",
        action="store_true",
        help="Synchronize CUDA around timed sections for stricter timing. Slower, but more precise.",
    )
    parser.add_argument("--predict-residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--zero-init-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-contact-physics-losses", action="store_true")
    parser.add_argument("--enable-early-termination", action="store_true")
    parser.add_argument("--no-restart-on-termination", action="store_true")
    parser.add_argument(
        "--reset-exhausted-agents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For long AE rollouts, reset non-periodic rows before target/future-root frames would pass clip end.",
    )
    parser.add_argument("--freefall-body-height-offset-m", type=float, default=0.0)
    parser.add_argument("--freefall-initial-offset-history", type=int, choices=(1, 2), default=1)
    parser.add_argument("--freefall-initial-contacts-off", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--alpha7-contact-label", type=float, default=5.0)
    parser.add_argument("--alpha8-foot-penetration", type=float, default=1700.0)
    parser.add_argument("--alpha9-foot-sliding", type=float, default=1.0)
    parser.add_argument("--alpha10-freefall", type=float, default=1700.0)
    parser.add_argument("--alpha11-contact-height", type=float, default=245.0)
    parser.add_argument("--alpha12-termination", type=float, default=0.07)
    parser.add_argument("--ae-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--simple-footslide-loss-weight",
        type=float,
        default=0.0,
        help="Extra simple generated-geometry support-foot slide loss. Default off.",
    )
    parser.add_argument(
        "--simple-footslide-threshold-mps",
        type=float,
        default=0.0,
        help="Zero-loss support slide threshold. If <=0, use max ground-truth support slide times margin.",
    )
    parser.add_argument(
        "--simple-footslide-gt-margin",
        type=float,
        default=1.05,
        help="Multiplier for the auto threshold from ground-truth support slide.",
    )
    parser.add_argument(
        "--compatibility-score-weight",
        type=float,
        default=0.0,
        help="Extra model-training penalty from a compatible transition prior head, if the AE checkpoint has one.",
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
