from __future__ import annotations

import argparse
import math
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


def ae_score(prior, mean, std, features: torch.Tensor) -> torch.Tensor:
    x = (features - mean) / std
    recon = prior(x)
    return F.mse_loss(recon, x, reduction="none").mean(dim=-1).mean()


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
) -> tuple[torch.Tensor, dict[str, float]]:
    clip_indices, starts = batch
    total_loss = torch.zeros((), device=device)
    groups = {}
    for row, ci in enumerate(clip_indices.tolist()):
        groups.setdefault(ci, []).append(row)
    scores = []
    motion_sizes = []
    joint_rmses = []
    ee_rmses = []
    output_mses = []

    for ci, rows in groups.items():
        clip = clips[ci]
        row_t = torch.tensor(rows, dtype=torch.long)
        start = starts[row_t].long()
        prev_idx = start - 1
        cur_idx = start
        prev_pose = tl.get_pose_from_clip(clip, prev_idx, device)
        cur_pose = tl.get_pose_from_clip(clip, cur_idx, device)
        group_loss = torch.zeros((), device=device)
        for _ in range(rollout_k):
            inp = tl.build_input(clip, prev_idx, cur_idx, prev_pose, cur_pose, cfg, device)
            raw_out = tl.predict_next_raw(model, inp, cur_pose, cfg)
            pred_pose, _ = tl.output_to_pose(raw_out, clip)
            target_idx = cur_idx + 1
            tensors = clip.tensors(device)
            root_pos = tensors["root_pos"].index_select(0, target_idx.to(device))
            root_rot = tensors["root_rot"].index_select(0, target_idx.to(device))
            pred_global_pos, _, pred_canon = tl.fk_from_pose(clip, root_pos, root_rot, pred_pose, device)
            next_pose = {
                "pelvis_pos": pred_pose["pelvis_pos"],
                "pelvis_rot6": pred_pose["pelvis_rot6"],
                "nonpelvis_rot6": pred_pose["nonpelvis_rot6"],
                "canon_pos": pred_canon,
            }
            features = tae.transition_feature_from_next_pose(
                clip, prev_idx, cur_idx, prev_pose, cur_pose, next_pose, cfg, device
            )
            score = ae_score(prior, prior_mean, prior_std, features)
            group_loss = group_loss + score / rollout_k
            scores.append(float(score.detach().cpu()))
            motion_sizes.append(float((next_pose["canon_pos"] - cur_pose["canon_pos"]).square().mean().sqrt().detach().cpu()))
            target_pose = tl.get_pose_from_clip(clip, target_idx, device)
            target_global_pos = tensors["global_pos"].index_select(0, target_idx.to(device))
            joint_rmses.append(float((pred_global_pos - target_global_pos).square().sum(dim=-1).mean().sqrt().detach().cpu()))
            ee_idx = tensors["end_effectors"]
            ee_rmses.append(
                float(
                    (
                        pred_global_pos.index_select(1, ee_idx)
                        - target_global_pos.index_select(1, ee_idx)
                    )
                    .square()
                    .sum(dim=-1)
                    .mean()
                    .sqrt()
                    .detach()
                    .cpu()
                )
            )
            output_mses.append(
                float(
                    F.mse_loss(
                        tl.pose_target_output(next_pose),
                        tl.pose_target_output(target_pose),
                    )
                    .detach()
                    .cpu()
                )
            )
            prev_pose = cur_pose
            cur_pose = next_pose
            prev_idx = cur_idx
            cur_idx = target_idx
        total_loss = total_loss + group_loss
    total_loss = total_loss / max(1, len(groups))
    return total_loss, {
        "total": float(total_loss.detach().cpu()),
        "ae_score": float(np.mean(scores)) if scores else 0.0,
        "canon_step_rms": float(np.mean(motion_sizes)) if motion_sizes else 0.0,
        "joint_rmse": float(np.mean(joint_rmses)) if joint_rmses else 0.0,
        "ee_rmse": float(np.mean(ee_rmses)) if ee_rmses else 0.0,
        "output_mse": float(np.mean(output_mses)) if output_mses else 0.0,
    }


def train(args: argparse.Namespace) -> None:
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
    cfg.use_torch_compile = False
    cfg.predict_residual = args.predict_residual
    cfg.zero_init_output = args.zero_init_output
    cfg.run_name = args.run_name
    cfg.device = args.device
    tl.set_seed(cfg.seed)
    device = torch.device(cfg.device)
    if cfg.allow_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    folder = tl.resolve_path(args.folder_path)
    clips = tl.load_clips(folder, cfg)
    prior, prior_ckpt = load_prior(tl.resolve_path(args.prior_checkpoint), device)
    prior_mean = prior_ckpt["mean"].to(device)
    prior_std = prior_ckpt["std"].to(device)

    input_dim, output_dim = tl.make_batch_dims(clips[0], cfg)
    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if args.resume_checkpoint:
        resume_path = tl.resolve_path(args.resume_checkpoint)
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume["model"])
        print(f"resumed model weights from {resume_path}", flush=True)
    run_dir = tl.resolve_path(cfg.output_dir) / cfg.run_name
    ckpt_dir = run_dir / "checkpoints"
    writer = SummaryWriter(run_dir / "tb")
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
    }

    schedule = tuple(k for k in cfg.rollout_schedule if k <= min(clip.T - 2 for clip in clips)) or (1,)
    rollout_idx = 0
    rollout_k = schedule[rollout_idx]
    loader = DataLoader(
        tl.MotionIndexDataset(clips, cfg, "train", rollout_k),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    print(f"pure_ae run={cfg.run_name} prior={args.prior_checkpoint} K={rollout_k} samples={len(loader.dataset)}", flush=True)
    best = math.inf
    stalls = 0
    stage_start_epoch = 1
    start_time = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        parts = []
        model.train()
        for batch in loader:
            loss, scalars = run_batch_ae(model, prior, prior_mean, prior_std, clips, batch, cfg, rollout_k, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            parts.append(scalars)
        train_total = float(np.mean([p["total"] for p in parts]))
        train_score = float(np.mean([p["ae_score"] for p in parts]))
        motion_rms = float(np.mean([p["canon_step_rms"] for p in parts]))
        joint_rmse = float(np.mean([p["joint_rmse"] for p in parts]))
        ee_rmse = float(np.mean([p["ee_rmse"] for p in parts]))
        output_mse = float(np.mean([p["output_mse"] for p in parts]))
        selection_metric = {
            "ae_score": train_total,
            "joint_rmse": joint_rmse,
            "ee_rmse": ee_rmse,
            "output_mse": output_mse,
        }[args.best_metric]
        writer.add_scalar("loss/train_total", train_total, epoch)
        writer.add_scalar("loss/ae_score", train_score, epoch)
        writer.add_scalar("motion/canon_step_rms", motion_rms, epoch)
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
        if args.save_live_every_epochs > 0 and epoch % args.save_live_every_epochs == 0:
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
            loader = DataLoader(
                tl.MotionIndexDataset(clips, cfg, "train", rollout_k),
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            best = math.inf
            stalls = 0
            stage_start_epoch = epoch + 1
            print(f"advanced rollout_k={rollout_k} samples={len(loader.dataset)}", flush=True)
        if args.max_train_seconds > 0 and time.perf_counter() - start_time >= args.max_train_seconds:
            print(f"max_train_seconds reached {args.max_train_seconds}", flush=True)
            break
    last_path = ckpt_dir / "checkpoint_last.pt"
    tl.save_checkpoint(last_path, model, opt, epoch, best, rollout_k, cfg, metadata)
    refresh_live_viewer(args, last_path)
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-path", default="data/npz_final")
    parser.add_argument("--prior-checkpoint", required=True)
    parser.add_argument("--run-name", default="locomotion_pure_ae")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--rollout-schedule", default="1")
    parser.add_argument("--curriculum-max-epochs-per-stage", type=int, default=120)
    parser.add_argument("--curriculum-stall-patience-epochs", type=int, default=60)
    parser.add_argument("--curriculum-min-epochs", type=int, default=30)
    parser.add_argument("--curriculum-min-delta", type=float, default=1e-6)
    parser.add_argument("--max-train-seconds", type=float, default=0.0)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--best-metric", choices=("ae_score", "joint_rmse", "ee_rmse", "output_mse"), default="joint_rmse")
    parser.add_argument("--save-live-every-epochs", type=int, default=20)
    parser.add_argument("--live-viewer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--live-npz-path", default="data/npz_final/testcasc.npz")
    parser.add_argument("--live-output-path", default="training/runs/model_comparisons/model_comparison.html")
    parser.add_argument("--predict-residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--zero-init-output", action=argparse.BooleanOptionalAction, default=True)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
