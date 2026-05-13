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


def ae_score(
    prior,
    mean,
    std,
    features: torch.Tensor,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    x = (features - mean) / std
    recon = prior(x)
    if loss_type == "huber":
        error = F.huber_loss(recon, x, reduction="none", delta=huber_delta)
    else:
        error = F.mse_loss(recon, x, reduction="none")
    return error.mean(dim=-1).mean()


def run_batch_ae(
    model: torch.nn.Module,
    prior: torch.nn.Module,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
    clips: list[tl.MotionClip],
    batch: list[torch.Tensor],
    cfg: tl.TrainConfig,
    rollout_k: int,
    device: torch.device,
    compute_diagnostics: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    clip_indices, starts = batch
    total_loss = torch.zeros((), device=device)
    groups = {}
    for row, ci in enumerate(clip_indices.tolist()):
        groups.setdefault(ci, []).append(row)
    scores: list[torch.Tensor] = []
    motion_sizes: list[torch.Tensor] = []
    joint_rmses: list[torch.Tensor] = []
    ee_rmses: list[torch.Tensor] = []
    output_mses: list[torch.Tensor] = []

    for ci, rows in groups.items():
        clip = clips[ci]
        row_t = torch.tensor(rows, dtype=torch.long)
        start = starts[row_t].long().to(device)
        prev_idx = start - 1
        cur_idx = start
        prev_pose = tl.get_pose_from_clip(clip, prev_idx, device)
        cur_pose = tl.get_pose_from_clip(clip, cur_idx, device)
        prev_pose, cur_pose = tl.maybe_apply_initial_offsets(
            clip, prev_idx, cur_idx, prev_pose, cur_pose, cfg, device
        )
        group_loss = torch.zeros((), device=device)
        for step in range(rollout_k):
            inp = tl.build_input(clip, prev_idx, cur_idx, prev_pose, cur_pose, cfg, device)
            raw_out = tl.predict_next_raw(model, inp, cur_pose, cfg)
            pred_pose, raw_pose = tl.output_to_pose(raw_out, clip)
            target_idx = cur_idx + 1
            tensors = clip.tensors(device)
            root_pos, root_rot, _yaw, _heading = tl.root_state(clip, target_idx, cfg, device)
            pred_global_pos, _, pred_canon = tl.fk_from_pose(clip, root_pos, root_rot, pred_pose, device)
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
            score = ae_score(prior, prior_mean, prior_std, features, cfg.ae_score_loss, cfg.ae_huber_delta)
            step_loss = cfg.ae_loss_weight * score
            term_mask = torch.zeros(cur_idx.shape[0], dtype=torch.bool, device=device)
            if cfg.enable_contact_physics_losses:
                target_pose = tl.get_pose_from_clip(clip, target_idx, device)
                physics_cfg = copy.copy(cfg)
                physics_cfg.alpha0_pelvis_location = 0.0
                physics_cfg.alpha1_pelvis_rotation = 0.0
                physics_cfg.alpha2_pose_rotation = 0.0
                physics_cfg.alpha3_pose_6d_aux = 0.0
                physics_cfg.alpha4_end_effector_location = 0.0
                physics_cfg.alpha5_end_effector_rotation = 0.0
                physics_cfg.alpha6_full_body_location = 0.0
                physics_loss, _physics_parts, _physics_next_pose, term_mask = tl.compute_losses(
                    clip,
                    prev_pose,
                    cur_pose,
                    pred_pose,
                    raw_pose,
                    target_pose,
                    prev_idx,
                    cur_idx,
                    target_idx,
                    physics_cfg,
                    device,
                )
                step_loss = step_loss + physics_loss
            group_loss = group_loss + step_loss / rollout_k
            scores.append(score.detach())
            motion_sizes.append((next_pose["canon_pos"] - cur_pose["canon_pos"]).square().mean().sqrt().detach())
            if compute_diagnostics:
                target_pose = tl.get_pose_from_clip(clip, target_idx, device)
                target_global_pos, _target_global_rot = tl.global_from_clip(clip, target_idx, cfg, device)
                joint_rmses.append((pred_global_pos - target_global_pos).square().sum(dim=-1).mean().sqrt().detach())
                ee_idx = tensors["end_effectors"]
                ee_rmses.append(
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
                output_mses.append(
                    F.mse_loss(
                        tl.pose_target_output(next_pose),
                        tl.pose_target_output(target_pose),
                    )
                    .detach()
                )
            if cfg.enable_early_termination and cfg.restart_on_termination and step + 1 < rollout_k and bool(term_mask.any()):
                remaining_steps = rollout_k - step - 1
                max_start = max(1, clip.cyclic_period - 1 if cfg.cyclic_animation else clip.T - 1 - remaining_steps)
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
                continue
            prev_pose = cur_pose
            cur_pose = next_pose
            prev_idx = cur_idx
            cur_idx = target_idx
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
    }


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
    cfg.device = args.device
    cfg.cyclic_animation = args.cyclic_animation
    cfg.training_loop = args.training_loop
    cfg.agent_sampling = args.agent_sampling
    cfg.enable_contact_physics_losses = not args.no_contact_physics_losses
    cfg.enable_early_termination = args.enable_early_termination
    cfg.restart_on_termination = not args.no_restart_on_termination
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
    tl.set_seed(cfg.seed)
    device = torch.device(cfg.device)
    tl.apply_cuda_performance_settings(cfg, device)
    profiler = tl.TimingProfiler(args.profile_timing, device, args.profile_sync_cuda)

    folder = tl.resolve_path(args.folder_path)
    with profiler.section("setup/load_npz_and_prior"):
        clips = tl.load_clips(folder, cfg)
        prior, prior_ckpt = load_prior(tl.resolve_path(args.prior_checkpoint), device)
        prior_mean = prior_ckpt["mean"].to(device)
        prior_std = prior_ckpt["std"].to(device)

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
    run_dir = tl.resolve_path(cfg.output_dir) / cfg.run_name
    ckpt_dir = run_dir / "checkpoints"
    writer = SummaryWriter(run_dir / "tb")
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
        "npz_folder": str(folder),
        "body_names": clips[0].body_names,
        "parents_body": clips[0].parents_body.tolist(),
        "pelvis_index": clips[0].pelvis,
        "non_pelvis_indices": clips[0].non_pelvis,
        "end_effector_indices": clips[0].end_effectors,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "ae_prior_checkpoint": str(tl.resolve_path(args.prior_checkpoint)),
        "loss_type": "pure_transition_ae_prior",
        "ae_score_loss": args.ae_score_loss,
        "ae_huber_delta": args.ae_huber_delta,
        "compile_enabled": compile_enabled,
    }

    schedule = tuple(
        k for k in cfg.rollout_schedule if k <= min((clip.cyclic_period if cfg.cyclic_animation else clip.T - 2) for clip in clips)
    ) or (1,)
    rollout_idx = 0
    rollout_k = schedule[rollout_idx]
    agent_rng = random.Random(cfg.seed + 4817)
    agent_coverage_order: list[tuple[int, int]] = []
    agent_coverage_cursor = 0

    def make_loader(max_rollout: int) -> tuple[tl.MotionIndexDataset, DataLoader]:
        dataset = tl.MotionIndexDataset(clips, cfg, "train", max_rollout)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        return dataset, loader

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
        ci = agent_rng.randrange(len(clips)) if clip_index is None else int(clip_index)
        max_start = max(1, clips[ci].cyclic_period - 1 if cfg.cyclic_animation else clips[ci].T - max_rollout - 1)
        return ci, agent_rng.randint(1, max_start)

    def agent_batch(dataset: tl.MotionIndexDataset, max_rollout: int) -> tuple[torch.Tensor, torch.Tensor]:
        clip_ids = []
        starts = []
        fixed_clip = None
        if args.agent_batch_clips == 1 and cfg.agent_sampling == "random" and len(clips) > 1:
            fixed_clip = agent_rng.randrange(len(clips))
        for _ in range(cfg.batch_size):
            if cfg.agent_sampling == "coverage":
                ci, start = coverage_agent_start(dataset)
            else:
                ci, start = random_agent_start(max_rollout, fixed_clip)
            clip_ids.append(ci)
            starts.append(start)
        return torch.tensor(clip_ids, dtype=torch.long), torch.tensor(starts, dtype=torch.long)

    dataset, loader = make_loader(rollout_k)
    print(
        f"pure_ae run={cfg.run_name} prior={args.prior_checkpoint} K={rollout_k} "
        f"samples={len(dataset)} loop={cfg.training_loop}",
        flush=True,
    )
    best = math.inf
    stalls = 0
    stage_start_epoch = 1
    start_time = time.perf_counter()
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
            train_iter = [agent_batch(dataset, rollout_k) for _ in range(max(1, args.agent_batches_per_epoch))]
        else:
            train_iter = loader
        for batch in train_iter:
            with profiler.section("train/forward_loss"):
                loss, scalars = run_batch_ae(
                    model,
                    prior,
                    prior_mean,
                    prior_std,
                    clips,
                    batch,
                    cfg,
                    rollout_k,
                    device,
                    compute_diagnostics=compute_diagnostics,
                )
            with profiler.section("train/zero_grad"):
                opt.zero_grad(set_to_none=True)
            with profiler.section("train/backward"):
                loss.backward()
            with profiler.section("train/clip_grad"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            with profiler.section("train/optimizer_step"):
                opt.step()
            parts.append(scalars)
        train_total = float(np.mean([p["total"] for p in parts]))
        train_score = float(np.mean([p["ae_score"] for p in parts]))
        motion_rms = float(np.mean([p["canon_step_rms"] for p in parts]))
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
        writer.add_scalar("motion/canon_step_rms", motion_rms, epoch)
        if compute_diagnostics:
            writer.add_scalar("accuracy/joint_rmse_m", joint_rmse, epoch)
            writer.add_scalar("accuracy/end_effector_rmse_m", ee_rmse, epoch)
            writer.add_scalar("accuracy/output_mse", output_mse, epoch)
        writer.add_scalar(f"selection/{args.best_metric}", selection_metric, epoch)
        writer.add_scalar("curriculum/rollout_k", rollout_k, epoch)
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
        stage_epochs = epoch - stage_start_epoch + 1
        can_advance = (
            cfg.curriculum_max_epochs_per_stage > 0
            and stage_epochs >= cfg.curriculum_max_epochs_per_stage
        ) or (
            cfg.curriculum_stall_patience_epochs > 0
            and stage_epochs >= cfg.curriculum_min_epochs
            and stalls >= cfg.curriculum_stall_patience_epochs
        )
        print(
            f"epoch={epoch:04d} K={rollout_k:02d} ae={train_total:.6g} best_{args.best_metric}={best:.6g} "
            f"joint_rmse={joint_rmse:.6g} ee_rmse={ee_rmse:.6g} motion_rms={motion_rms:.6g} "
            f"stalls={stalls} elapsed_s={time.perf_counter() - start_time:.1f}",
            flush=True,
        )
        if can_advance and rollout_idx < len(schedule) - 1:
            rollout_idx += 1
            rollout_k = schedule[rollout_idx]
            dataset, loader = make_loader(rollout_k)
            reset_agent_coverage_order(dataset)
            best = math.inf
            stalls = 0
            stage_start_epoch = epoch + 1
            print(f"advanced rollout_k={rollout_k} samples={len(dataset)}", flush=True)
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
    parser.add_argument("--prior-checkpoint", required=True)
    parser.add_argument("--run-name", default="locomotion_pure_ae")
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
        "--agent-batch-clips",
        type=int,
        default=1,
        help="With random agent sampling, 1 means each batch samples starts from one randomly chosen clip.",
    )
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--rollout-schedule", default="1")
    parser.add_argument("--curriculum-max-epochs-per-stage", type=int, default=120)
    parser.add_argument("--curriculum-stall-patience-epochs", type=int, default=60)
    parser.add_argument("--curriculum-min-epochs", type=int, default=30)
    parser.add_argument("--curriculum-min-delta", type=float, default=1e-6)
    parser.add_argument("--max-train-seconds", type=float, default=0.0)
    parser.add_argument("--resume-checkpoint", default=None)
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
    train(parser.parse_args())


if __name__ == "__main__":
    main()
