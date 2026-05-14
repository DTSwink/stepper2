from __future__ import annotations

# Constraint-only locomotion experiment. Relative paths resolve from project root.
folder_path = "ue5/test/npz_final"

import argparse
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import contact_physics as cp
import train_locomotion as tl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ConstraintConfig:
    loss_mode: str = "root_warmup"
    com_horizontal_threshold_m: float = 0.75
    com_root_horizontal_threshold_m: float = 0.75
    head_foot_horizontal_threshold_m: float = 1.0
    pelvis_axis_limit_deg: float = 90.0
    pelvis_height_ceiling_m: float = 2.0
    bone_height_floor_m: float = 0.0
    bone_height_floor_scale_m: float = 0.1
    foot_contact_speed_threshold_mps: float = 0.005
    hover_weight: float = 1.0
    slide_weight: float = 1.0
    penetration_weight: float = 1.0
    com_weight: float = 1.0
    com_root_weight: float = 1.0
    head_foot_weight: float = 1.0
    pelvis_axis_weight: float = 1.0
    pelvis_height_ceiling_weight: float = 1.0
    bone_height_floor_weight: float = 1.0
    both_unpinned_weight: float = 5000.0
    bad_contact_gate_weight: float = 0.1
    bad_contact_gate_margin: float = 0.05
    contact_on_threshold: float = 0.8
    target_loss: float = 1e-6


def constraint_losses(
    clip: tl.MotionClip,
    cur_pose: dict[str, torch.Tensor],
    pred_pose: dict[str, torch.Tensor],
    cur_idx: torch.Tensor,
    target_idx: torch.Tensor,
    cfg: tl.TrainConfig,
    constraint_cfg: ConstraintConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    tensors = clip.tensors(device)
    cur_idx = cur_idx.to(device)
    target_idx = target_idx.to(device)
    cur_root_pos = tensors["root_pos"].index_select(0, cur_idx)
    cur_root_rot = tensors["root_rot"].index_select(0, cur_idx)
    target_root_pos = tensors["root_pos"].index_select(0, target_idx)
    target_root_rot = tensors["root_rot"].index_select(0, target_idx)

    cur_global_pos, cur_global_rot, _cur_canon = tl.fk_from_pose(clip, cur_root_pos, cur_root_rot, cur_pose, device)
    pred_global_pos, pred_global_rot, pred_canon = tl.fk_from_pose(
        clip, target_root_pos, target_root_rot, pred_pose, device
    )

    foot_indices = tuple(clip.foot_indices)
    toe_indices = tuple(clip.toe_indices)
    contact_prob = pred_pose["contacts"]
    com_scale = max(float(constraint_cfg.com_horizontal_threshold_m), 1e-6)
    com_root_scale = max(float(constraint_cfg.com_root_horizontal_threshold_m), 1e-6)
    head_foot_scale = max(float(constraint_cfg.head_foot_horizontal_threshold_m), 1e-6)
    pelvis_height_scale = max(float(constraint_cfg.pelvis_height_ceiling_m), 1e-6)
    bone_height_floor_scale = max(float(constraint_cfg.bone_height_floor_scale_m), 1e-6)

    pinned = (contact_prob >= constraint_cfg.contact_on_threshold).to(contact_prob.dtype)
    zero = pred_pose["pelvis_pos"].new_zeros(())
    hover = zero
    slide = zero
    penetration = zero
    bad_contact_gate = zero
    contact_point_speeds = pred_pose["pelvis_pos"].new_zeros((pred_pose["pelvis_pos"].shape[0], 2))

    if constraint_cfg.loss_mode == "root_warmup":
        foot_joint_indices = torch.tensor((*foot_indices, *toe_indices), dtype=torch.long, device=device)
        foot_points = pred_global_pos.index_select(1, foot_joint_indices)
        foot_mean = foot_points.mean(dim=1)
        foot_heights = foot_points[:, :, 1]
    else:
        geom = cp.DEFAULT_GEOMETRY
        height_scale = max(float(geom.height_threshold_m), 1e-6)
        speed_threshold = float(constraint_cfg.foot_contact_speed_threshold_mps)
        speed_scale = max(speed_threshold, 1e-6)
        foot_heights, lowest_points = cp.foot_lowest_heights_and_points(
            pred_global_pos, pred_global_rot, foot_indices, toe_indices, geom
        )
        contact_point_speeds = cp.foot_contact_point_speeds(
            cur_global_pos, cur_global_rot, pred_global_pos, pred_global_rot, foot_indices, toe_indices, clip.fps, geom
        )
        hover_per_foot = F.relu((foot_heights - geom.height_threshold_m) / height_scale).square()
        slide_per_foot = F.relu((contact_point_speeds - speed_threshold) / speed_scale).square()
        hover = (pinned * hover_per_foot).mean()
        slide = (pinned * slide_per_foot).mean()
        penetration = F.relu((-foot_heights) / height_scale).square().mean()
        bad_contact = (hover_per_foot + slide_per_foot).detach()
        gate_floor = max(constraint_cfg.contact_on_threshold - constraint_cfg.bad_contact_gate_margin, 0.0)
        gate_scale = max(1.0 - gate_floor, 1e-6)
        bad_contact_gate = (F.relu((contact_prob - gate_floor) / gate_scale).square() * bad_contact).mean()
        foot_mean = lowest_points.mean(dim=1)

    com = cp.center_of_mass(pred_global_pos, tensors["mass_weights"])
    com_horizontal = torch.linalg.norm((com - foot_mean)[:, [0, 2]], dim=-1)
    com_support = F.relu((com_horizontal - constraint_cfg.com_horizontal_threshold_m) / com_scale).square().mean()
    com_root_horizontal = torch.linalg.norm((com - target_root_pos)[:, [0, 2]], dim=-1)
    com_root = (
        F.relu((com_root_horizontal - constraint_cfg.com_root_horizontal_threshold_m) / com_root_scale)
        .square()
        .mean()
    )
    head_index = clip.body_names.index("head")
    head_pos = pred_global_pos[:, head_index]
    head_foot_horizontal = torch.linalg.norm((head_pos - foot_mean)[:, [0, 2]], dim=-1)
    head_foot = (
        F.relu((head_foot_horizontal - constraint_cfg.head_foot_horizontal_threshold_m) / head_foot_scale)
        .square()
        .mean()
    )

    pelvis_rot = tl.rotation_6d_to_matrix(pred_pose["pelvis_rot6"])
    neutral_pelvis_rot = clip.local_rot[0, clip.pelvis].to(device=device, dtype=pelvis_rot.dtype)
    axis_dots = (pelvis_rot * neutral_pelvis_rot.unsqueeze(0)).sum(dim=-1)
    axis_cos_limit = math.cos(math.radians(constraint_cfg.pelvis_axis_limit_deg))
    pelvis_axis = F.relu(axis_cos_limit - axis_dots).square().mean()
    pelvis_height = pred_global_pos[:, clip.pelvis, 1]
    pelvis_height_ceiling = (
        F.relu((pelvis_height - constraint_cfg.pelvis_height_ceiling_m) / pelvis_height_scale)
        .square()
        .mean()
    )
    bone_heights = pred_global_pos[:, :, 1]
    bone_height_floor = (
        F.relu((constraint_cfg.bone_height_floor_m - bone_heights) / bone_height_floor_scale)
        .square()
        .mean()
    )

    max_contact = contact_prob.max(dim=-1).values
    both_unpinned = F.relu(
        (constraint_cfg.contact_on_threshold - max_contact) / max(constraint_cfg.contact_on_threshold, 1e-6)
    ).square().mean()

    if constraint_cfg.loss_mode == "root_warmup":
        hover = slide = penetration = both_unpinned = bad_contact_gate = zero
        total = (
            constraint_cfg.com_weight * com_support
            + constraint_cfg.com_root_weight * com_root
            + constraint_cfg.head_foot_weight * head_foot
            + constraint_cfg.pelvis_axis_weight * pelvis_axis
            + constraint_cfg.pelvis_height_ceiling_weight * pelvis_height_ceiling
            + constraint_cfg.bone_height_floor_weight * bone_height_floor
        )
    else:
        total = (
            constraint_cfg.hover_weight * hover
            + constraint_cfg.slide_weight * slide
            + constraint_cfg.penetration_weight * penetration
            + constraint_cfg.com_weight * com_support
            + constraint_cfg.com_root_weight * com_root
            + constraint_cfg.bone_height_floor_weight * bone_height_floor
            + constraint_cfg.both_unpinned_weight * both_unpinned
            + constraint_cfg.bad_contact_gate_weight * bad_contact_gate
        )
    parts = {
        "hover": hover.detach(),
        "slide": slide.detach(),
        "penetration": penetration.detach(),
        "com_support": com_support.detach(),
        "com_root": com_root.detach(),
        "head_foot": head_foot.detach(),
        "pelvis_axis": pelvis_axis.detach(),
        "pelvis_height_ceiling": pelvis_height_ceiling.detach(),
        "bone_height_floor": bone_height_floor.detach(),
        "both_unpinned": both_unpinned.detach(),
        "bad_contact_gate": bad_contact_gate.detach(),
        "contact_prob_mean": contact_prob.detach().mean(),
        "contact_prob_max_mean": max_contact.detach().mean(),
        "pinned_rate": pinned.detach().mean(),
        "pinned_any_rate": pinned.detach().amax(dim=-1).mean(),
        "foot_height_min": foot_heights.detach().amin(),
        "foot_height_max": foot_heights.detach().amax(),
        "contact_point_speed_mean": contact_point_speeds.detach().mean(),
        "contact_point_speed_max": contact_point_speeds.detach().amax(),
        "com_horizontal_max": com_horizontal.detach().amax(),
        "com_root_horizontal_max": com_root_horizontal.detach().amax(),
        "head_foot_horizontal_max": head_foot_horizontal.detach().amax(),
        "pelvis_axis_min_dot": axis_dots.detach().amin(),
        "pelvis_height_max": pelvis_height.detach().amax(),
        "bone_height_min": bone_heights.detach().amin(),
    }
    next_pose = {
        "pelvis_pos": pred_pose["pelvis_pos"],
        "pelvis_rot6": pred_pose["pelvis_rot6"],
        "nonpelvis_rot6": pred_pose["nonpelvis_rot6"],
        "canon_pos": pred_canon,
        "contacts": pred_pose["contacts"],
    }
    return total, parts, next_pose


def run_batch(
    model: torch.nn.Module,
    clips: list[tl.MotionClip],
    batch: list[torch.Tensor],
    cfg: tl.TrainConfig,
    constraint_cfg: ConstraintConfig,
    rollout_k: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    clip_indices, starts = batch
    total_loss = torch.zeros((), device=device)
    accum: dict[str, torch.Tensor] = {}
    groups: dict[int, list[int]] = {}
    for row, ci in enumerate(clip_indices.tolist()):
        groups.setdefault(int(ci), []).append(row)

    for ci, rows in groups.items():
        clip = clips[ci]
        row_t = torch.tensor(rows, dtype=torch.long)
        start = starts[row_t].long().to(device)
        prev_idx = start - 1
        cur_idx = start
        prev_pose = tl.get_pose_from_clip(clip, prev_idx, device)
        cur_pose = tl.get_pose_from_clip(clip, cur_idx, device)
        group_loss = torch.zeros((), device=device)
        for _step in range(rollout_k):
            inp = tl.build_input(clip, prev_idx, cur_idx, prev_pose, cur_pose, cfg, device)
            raw_out = tl.predict_next_raw(model, inp, cur_pose, cfg)
            pred_pose, _raw_pose = tl.output_to_pose(raw_out, clip)
            target_idx = cur_idx + 1
            step_loss, parts, next_pose = constraint_losses(
                clip, cur_pose, pred_pose, cur_idx, target_idx, cfg, constraint_cfg, device
            )
            group_loss = group_loss + step_loss / rollout_k
            for key, value in parts.items():
                accum[key] = accum.get(key, torch.zeros((), device=device)) + value / rollout_k
            prev_pose = cur_pose
            cur_pose = next_pose
            prev_idx = cur_idx
            cur_idx = target_idx
        total_loss = total_loss + group_loss

    group_count = max(1, len(groups))
    total_loss = total_loss / group_count
    scalars = {k: float(v.detach().cpu() / group_count) for k, v in accum.items()}
    scalars["total"] = float(total_loss.detach().cpu())
    return total_loss, scalars


def make_agent_batch(
    clips: list[tl.MotionClip],
    batch_size: int,
    rollout_k: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    clip_ids = []
    starts = []
    for _ in range(batch_size):
        ci = rng.randrange(len(clips))
        max_start = max(1, clips[ci].T - rollout_k - 1)
        clip_ids.append(ci)
        starts.append(rng.randint(1, max_start))
    return torch.tensor(clip_ids, dtype=torch.long), torch.tensor(starts, dtype=torch.long)


def train(args: argparse.Namespace) -> None:
    cfg = tl.TrainConfig()
    cfg.val_fraction = 0.0
    cfg.disable_validation = True
    cfg.hidden_dim = args.hidden_dim
    cfg.num_hidden_layers = args.num_hidden_layers
    cfg.batch_size = args.batch_size
    cfg.max_epochs = args.max_epochs
    cfg.learning_rate = args.learning_rate
    cfg.lr_schedule = args.lr_schedule
    cfg.lr_min_factor = args.lr_min_factor
    cfg.lr_plateau_patience_epochs = args.lr_plateau_patience_epochs
    cfg.lr_plateau_factor = args.lr_plateau_factor
    cfg.rollout_schedule = tl.parse_rollout_schedule(args.rollout_schedule)
    cfg.curriculum_max_epochs_per_stage = args.curriculum_max_epochs_per_stage
    cfg.curriculum_stall_patience_epochs = args.curriculum_stall_patience_epochs
    cfg.curriculum_min_delta = args.curriculum_min_delta
    cfg.predict_residual = args.predict_residual
    cfg.zero_init_output = args.zero_init_output
    cfg.use_contact_state = args.contact_state
    cfg.zero_contact_state = args.zero_contact_state
    cfg.device = args.device
    cfg.use_torch_compile = args.torch_compile
    cfg.live_viewer = False
    cfg.update_comparison_on_exit = True
    cfg.comparison_output_path = args.comparison_output_path
    cfg.comparison_device = args.comparison_device
    cfg.run_name = args.run_name or f"constraint_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    constraint_cfg = ConstraintConfig(
        loss_mode=args.loss_mode,
        com_horizontal_threshold_m=args.com_horizontal_threshold_m,
        com_root_horizontal_threshold_m=args.com_root_horizontal_threshold_m,
        head_foot_horizontal_threshold_m=args.head_foot_horizontal_threshold_m,
        pelvis_axis_limit_deg=args.pelvis_axis_limit_deg,
        pelvis_height_ceiling_m=args.pelvis_height_ceiling_m,
        bone_height_floor_m=args.bone_height_floor_m,
        bone_height_floor_scale_m=args.bone_height_floor_scale_m,
        foot_contact_speed_threshold_mps=args.foot_contact_speed_threshold_mps,
        hover_weight=args.hover_weight,
        slide_weight=args.slide_weight,
        penetration_weight=args.penetration_weight,
        com_weight=args.com_weight,
        com_root_weight=args.com_root_weight,
        head_foot_weight=args.head_foot_weight,
        pelvis_axis_weight=args.pelvis_axis_weight,
        pelvis_height_ceiling_weight=args.pelvis_height_ceiling_weight,
        bone_height_floor_weight=args.bone_height_floor_weight,
        both_unpinned_weight=args.both_unpinned_weight,
        bad_contact_gate_weight=args.bad_contact_gate_weight,
        bad_contact_gate_margin=args.bad_contact_gate_margin,
        contact_on_threshold=args.contact_on_threshold,
        target_loss=args.target_loss,
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available() and cfg.device.startswith("cuda"):
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(cfg.device)
    if cfg.allow_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    folder = tl.resolve_path(args.folder_path or folder_path)
    clips = tl.load_clips(folder, cfg)
    input_dim, output_dim = tl.make_batch_dims(clips[0], cfg)
    model: torch.nn.Module = tl.MLPController(input_dim, output_dim, cfg).to(device)
    if args.resume_checkpoint:
        resume_path = tl.resolve_path(args.resume_checkpoint)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"resumed model={resume_path}", flush=True)
    if cfg.use_torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode=cfg.torch_compile_mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    lr_state = tl.reset_adaptive_lr(optimizer, cfg)

    run_dir = tl.resolve_path(args.output_dir) / cfg.run_name
    ckpt_dir = run_dir / "checkpoints"
    writer = SummaryWriter(str(run_dir / "tb"))
    metadata = {
        "mode": "constraint_only",
        "folder": str(folder),
        "clips": [str(clip.path) for clip in clips],
        "input_dim": input_dim,
        "output_dim": output_dim,
        "constraint_config": asdict(constraint_cfg),
    }
    (run_dir / "config.json").write_text(
        tl.json.dumps({"train_config": asdict(cfg), "metadata": metadata}, indent=2),
        encoding="utf-8",
    )

    schedule = tuple(k for k in cfg.rollout_schedule if k <= min(clip.T - 2 for clip in clips)) or (1,)
    rollout_idx = 0
    rollout_k = schedule[rollout_idx]
    rng = random.Random(cfg.seed + 991)
    dataset = tl.MotionIndexDataset(clips, cfg, "train", rollout_k)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")

    best = math.inf
    stage_best = math.inf
    stalls = 0
    stage_start_epoch = 1
    start_time = time.perf_counter()
    print(
        f"constraint_only run={cfg.run_name} folder={folder} schedule={schedule} "
        f"samples={len(dataset)} device={device}",
        flush=True,
    )

    try:
        for epoch in range(1, cfg.max_epochs + 1):
            model.train()
            if args.training_loop == "agents":
                train_iter = [make_agent_batch(clips, cfg.batch_size, rollout_k, rng) for _ in range(args.agent_batches_per_epoch)]
            else:
                train_iter = loader
            parts = []
            current_lr = float(optimizer.param_groups[0]["lr"])
            for batch in train_iter:
                loss, scalars = run_batch(model, clips, batch, cfg, constraint_cfg, rollout_k, device)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()
                parts.append(scalars)

            def mean_scalar(key: str) -> float:
                return float(np.mean([p.get(key, 0.0) for p in parts])) if parts else 0.0

            train_total = mean_scalar("total")
            elapsed = time.perf_counter() - start_time
            for key in [
                "total",
                "hover",
                "slide",
                "penetration",
                "com_support",
                "com_root",
                "head_foot",
                "pelvis_axis",
                "pelvis_height_ceiling",
                "bone_height_floor",
                "both_unpinned",
                "bad_contact_gate",
                "contact_prob_mean",
                "contact_prob_max_mean",
                "pinned_rate",
                "pinned_any_rate",
                "foot_height_min",
                "foot_height_max",
                "contact_point_speed_mean",
                "contact_point_speed_max",
                "com_horizontal_max",
                "com_root_horizontal_max",
                "head_foot_horizontal_max",
                "pelvis_axis_min_dot",
                "pelvis_height_max",
                "bone_height_min",
            ]:
                writer.add_scalar(f"loss/train_{key}", mean_scalar(key), epoch)
            writer.add_scalar("curriculum/rollout_k", rollout_k, epoch)
            writer.add_scalar("optim/learning_rate", current_lr, epoch)
            writer.add_scalar("timing/elapsed_seconds", elapsed, epoch)
            if epoch % 5 == 0:
                writer.flush()

            if train_total < best:
                best = train_total
                tl.save_checkpoint(ckpt_dir / "checkpoint_best.pt", model, optimizer, epoch, best, rollout_k, cfg, metadata)
            if epoch % args.save_last_every_epochs == 0:
                tl.save_checkpoint(ckpt_dir / "checkpoint_last.pt", model, optimizer, epoch, best, rollout_k, cfg, metadata)

            improved = train_total < stage_best - cfg.curriculum_min_delta
            if improved:
                stage_best = train_total
                stalls = 0
            else:
                stalls += 1
            if cfg.lr_schedule == "adaptive_plateau":
                lr_state, _changed = tl.step_adaptive_lr(optimizer, cfg, lr_state, train_total)

            if epoch == 1 or epoch % args.print_every_epochs == 0 or train_total <= constraint_cfg.target_loss:
                print(
                    f"epoch={epoch:04d} K={rollout_k} loss={train_total:.8g} "
                    f"hover={mean_scalar('hover'):.3g} slide={mean_scalar('slide'):.3g} "
                    f"pen={mean_scalar('penetration'):.3g} comFoot={mean_scalar('com_support'):.3g} "
                    f"comRoot={mean_scalar('com_root'):.3g} "
                    f"headFoot={mean_scalar('head_foot'):.3g} "
                    f"pelvisAxis={mean_scalar('pelvis_axis'):.3g} "
                    f"pelvisCeil={mean_scalar('pelvis_height_ceiling'):.3g} "
                    f"boneFloor={mean_scalar('bone_height_floor'):.3g} "
                    f"unpinned={mean_scalar('both_unpinned'):.3g} "
                    f"badGate={mean_scalar('bad_contact_gate'):.3g} "
                    f"lr={float(optimizer.param_groups[0]['lr']):.3g}",
                    flush=True,
                )

            can_advance = train_total <= constraint_cfg.target_loss or (
                cfg.curriculum_stall_patience_epochs > 0
                and stalls >= cfg.curriculum_stall_patience_epochs
                and epoch - stage_start_epoch + 1 >= max(1, cfg.curriculum_max_epochs_per_stage)
            )
            force_advance = (
                cfg.curriculum_max_epochs_per_stage > 0
                and epoch - stage_start_epoch + 1 >= cfg.curriculum_max_epochs_per_stage
            )
            if rollout_idx < len(schedule) - 1 and (can_advance or force_advance):
                rollout_idx += 1
                rollout_k = schedule[rollout_idx]
                dataset = tl.MotionIndexDataset(clips, cfg, "train", rollout_k)
                loader = DataLoader(
                    dataset,
                    batch_size=cfg.batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=device.type == "cuda",
                )
                stage_best = math.inf
                best = math.inf
                stalls = 0
                stage_start_epoch = epoch + 1
                if cfg.lr_schedule == "adaptive_plateau" and cfg.lr_reset_on_rollout_advance:
                    lr_state = tl.reset_adaptive_lr(optimizer, cfg)
                print(f"advanced rollout_k={rollout_k} samples={len(dataset)}", flush=True)
                continue

            if rollout_idx == len(schedule) - 1 and train_total <= constraint_cfg.target_loss:
                print(f"target loss reached: {train_total:.8g}", flush=True)
                break
            if args.max_train_seconds > 0 and elapsed >= args.max_train_seconds:
                print(f"max_train_seconds reached: {args.max_train_seconds}", flush=True)
                break
    finally:
        tl.save_checkpoint(ckpt_dir / "checkpoint_last.pt", model, optimizer, epoch, best, rollout_k, cfg, metadata)
        writer.flush()
        writer.close()

    tl.update_model_comparison_html(clips[0].path, ckpt_dir, cfg)
    print(f"done run={cfg.run_name} best={best:.8g} dir={run_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Constraint-only locomotion trainer with no supervised pose losses.")
    parser.add_argument("--folder-path", default=None)
    parser.add_argument("--output-dir", default="training/runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--lr-schedule", choices=("constant", "adaptive_plateau", "cosine", "stage_cosine"), default="adaptive_plateau")
    parser.add_argument("--lr-min-factor", type=float, default=0.05)
    parser.add_argument("--lr-plateau-patience-epochs", type=int, default=15)
    parser.add_argument("--lr-plateau-factor", type=float, default=0.7)
    parser.add_argument("--rollout-schedule", default="1,2,4,8")
    parser.add_argument("--curriculum-max-epochs-per-stage", type=int, default=80)
    parser.add_argument("--curriculum-stall-patience-epochs", type=int, default=20)
    parser.add_argument("--curriculum-min-delta", type=float, default=1e-7)
    parser.add_argument("--training-loop", choices=("sampled", "agents"), default="agents")
    parser.add_argument("--agent-batches-per-epoch", type=int, default=4)
    parser.add_argument("--predict-residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--zero-init-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contact-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--zero-contact-state", action="store_true")
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--loss-mode", choices=("root_warmup", "contact_physics"), default="root_warmup")
    parser.add_argument("--com-horizontal-threshold-m", type=float, default=0.75)
    parser.add_argument("--com-root-horizontal-threshold-m", type=float, default=0.75)
    parser.add_argument("--head-foot-horizontal-threshold-m", type=float, default=1.0)
    parser.add_argument("--pelvis-axis-limit-deg", type=float, default=90.0)
    parser.add_argument("--pelvis-height-ceiling-m", type=float, default=2.0)
    parser.add_argument("--bone-height-floor-m", type=float, default=0.0)
    parser.add_argument("--bone-height-floor-scale-m", type=float, default=0.1)
    parser.add_argument("--foot-contact-speed-threshold-mps", type=float, default=0.005)
    parser.add_argument("--hover-weight", type=float, default=1.0)
    parser.add_argument("--slide-weight", type=float, default=1.0)
    parser.add_argument("--penetration-weight", type=float, default=1.0)
    parser.add_argument("--com-weight", type=float, default=1.0)
    parser.add_argument("--com-root-weight", type=float, default=1.0)
    parser.add_argument("--head-foot-weight", type=float, default=1.0)
    parser.add_argument("--pelvis-axis-weight", type=float, default=1.0)
    parser.add_argument("--pelvis-height-ceiling-weight", type=float, default=1.0)
    parser.add_argument("--bone-height-floor-weight", type=float, default=1.0)
    parser.add_argument("--both-unpinned-weight", type=float, default=5000.0)
    parser.add_argument("--bad-contact-gate-weight", type=float, default=0.1)
    parser.add_argument("--bad-contact-gate-margin", type=float, default=0.05)
    parser.add_argument("--contact-on-threshold", type=float, default=0.8)
    parser.add_argument("--target-loss", type=float, default=1e-6)
    parser.add_argument("--max-train-seconds", type=float, default=0.0)
    parser.add_argument("--save-last-every-epochs", type=int, default=10)
    parser.add_argument("--print-every-epochs", type=int, default=10)
    parser.add_argument("--comparison-output-path", default="training/runs/model_comparisons/model_comparison.html")
    parser.add_argument("--comparison-device", default="cpu")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
