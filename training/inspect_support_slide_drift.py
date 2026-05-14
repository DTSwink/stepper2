from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

import contact_physics as cp
import train_locomotion as tl
from inspect_foot_sliding import load_controller, rollout_autoreg, short_name


def is_idle(clip: tl.MotionClip) -> bool:
    return "idle" in Path(clip.path).stem.lower()


def clip_metrics(model, clip: tl.MotionClip, cfg: tl.TrainConfig, device: torch.device, frame_count: int) -> dict[str, float | str | int]:
    pred_pos, pred_rot = rollout_autoreg(model, clip, cfg, device, frame_count)
    gt_pos, _gt_rot = tl.global_from_clip(clip, torch.arange(frame_count, device=device), cfg, device)
    root_pos, _root_rot, _yaw, _heading = tl.root_state(clip, torch.arange(frame_count, device=device), cfg, device)

    foot_indices = tuple(int(x) for x in clip.foot_indices_tensor.tolist())
    toe_indices = tuple(int(x) for x in clip.toe_indices_tensor.tolist())
    speeds = cp.foot_slide_speeds(
        pred_pos[:-1],
        pred_rot[:-1],
        pred_pos[1:],
        pred_rot[1:],
        foot_indices,
        toe_indices,
        clip.fps,
    ).detach().cpu().numpy()
    support = speeds.mean(axis=-1) if is_idle(clip) else speeds.min(axis=-1)

    pred_np = pred_pos.detach().cpu().numpy()
    gt_np = gt_pos.detach().cpu().numpy()
    root_np = root_pos.detach().cpu().numpy()
    body_ids = list(foot_indices) + list(toe_indices)
    root_delta = root_np[-1, [0, 2]] - root_np[0, [0, 2]]
    root_len = float(np.linalg.norm(root_delta))
    if root_len > 1e-4:
        forward = root_delta / root_len
        side = np.array([-forward[1], forward[0]])
        pred_vel = (pred_np[1:, body_ids, :][:, :, [0, 2]] - pred_np[:-1, body_ids, :][:, :, [0, 2]]) * float(clip.fps)
        gt_vel = (gt_np[1:, body_ids, :][:, :, [0, 2]] - gt_np[:-1, body_ids, :][:, :, [0, 2]]) * float(clip.fps)
        pred_side = np.abs(pred_vel @ side).mean(axis=1)
        pred_forward = np.abs(pred_vel @ forward).mean(axis=1)
        gt_side = np.abs(gt_vel @ side).mean(axis=1)
        gt_forward = np.abs(gt_vel @ forward).mean(axis=1)
        lat = pred_side / (pred_side + pred_forward + 1e-8)
        gt_lat = gt_side / (gt_side + gt_forward + 1e-8)
        lateral_excess = lat - gt_lat
    else:
        lateral_excess = np.zeros_like(support)

    window = min(15, support.shape[0])
    first = support[:window]
    last = support[-window:]
    first_lat = lateral_excess[:window]
    last_lat = lateral_excess[-window:]
    return {
        "clip": short_name(clip.path),
        "frames": frame_count,
        "support_mean": float(np.mean(support)),
        "support_p95": float(np.percentile(support, 95)),
        "support_max": float(np.max(support)),
        "support_first_mean": float(np.mean(first)),
        "support_last_mean": float(np.mean(last)),
        "support_first_p95": float(np.percentile(first, 95)),
        "support_last_p95": float(np.percentile(last, 95)),
        "support_mean_drift": float(np.mean(last) - np.mean(first)),
        "lateral_excess_mean": float(np.mean(lateral_excess)),
        "lateral_excess_first": float(np.mean(first_lat)),
        "lateral_excess_last": float(np.mean(last_lat)),
        "lateral_excess_drift": float(np.mean(last_lat) - np.mean(first_lat)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronological support-slide and lateral-drift monitor.")
    parser.add_argument("--folder-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cyclic-animation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip-filter", default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    base_cfg = tl.TrainConfig()
    base_cfg.cyclic_animation = args.cyclic_animation
    initial_clips = tl.load_clips(tl.resolve_path(args.folder_path), base_cfg)
    cfg = tl.TrainConfig()
    cfg.cyclic_animation = args.cyclic_animation
    model = load_controller(tl.resolve_path(args.checkpoint_path), cfg, initial_clips[0], device)
    cfg.cyclic_animation = args.cyclic_animation
    clips = tl.load_clips(tl.resolve_path(args.folder_path), cfg)
    if args.clip_filter:
        clips = [clip for clip in clips if args.clip_filter.lower() in Path(clip.path).stem.lower()]
    rows = []
    for clip in clips:
        frame_count = int(clip.cyclic_period + 1 if cfg.cyclic_animation else clip.T)
        rows.append(clip_metrics(model, clip, cfg, device, frame_count))

    if args.output_csv:
        out = tl.resolve_path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out}")

    print("clip frames mean p95 first_mean last_mean drift lat_first lat_last lat_drift")
    for row in rows:
        print(
            f"{row['clip']:>5s} {int(row['frames']):4d} "
            f"{row['support_mean']:.4f} {row['support_p95']:.4f} "
            f"{row['support_first_mean']:.4f} {row['support_last_mean']:.4f} {row['support_mean_drift']:+.4f} "
            f"{row['lateral_excess_first']:+.4f} {row['lateral_excess_last']:+.4f} {row['lateral_excess_drift']:+.4f}"
        )
    mean_support = float(np.mean([float(row["support_mean"]) for row in rows]))
    mean_p95 = float(np.mean([float(row["support_p95"]) for row in rows]))
    mean_drift = float(np.mean([float(row["support_mean_drift"]) for row in rows]))
    mean_lat_drift = float(np.mean([float(row["lateral_excess_drift"]) for row in rows]))
    print(f"MEAN support_mean={mean_support:.4f} support_p95={mean_p95:.4f} support_drift={mean_drift:+.4f} lat_drift={mean_lat_drift:+.4f}")


if __name__ == "__main__":
    main()
