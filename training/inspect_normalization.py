from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import train_locomotion as tl


def summarize(name: str, value: torch.Tensor) -> dict:
    flat = value.detach().float().reshape(-1)
    return {
        "name": name,
        "shape": list(value.shape),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "abs_max": float(flat.abs().max()),
        "p50_abs": float(flat.abs().quantile(0.50)),
        "p95_abs": float(flat.abs().quantile(0.95)),
        "p99_abs": float(flat.abs().quantile(0.99)),
    }


def inspect_clip(npz_path: Path, cfg: tl.TrainConfig, sample_all_frames: bool) -> dict:
    clip = tl.MotionClip(npz_path, cfg)
    device = torch.device("cpu")
    max_t = clip.T - cfg.future_window - 1
    if max_t < 1:
        raise ValueError(f"Clip too short for future window {cfg.future_window}: {npz_path}")
    indices = torch.arange(1, max_t + 1, dtype=torch.long)
    if not sample_all_frames and len(indices) > 512:
        indices = indices[torch.linspace(0, len(indices) - 1, 512).long()]

    prev = indices - 1
    cur = indices
    prev_pose = tl.get_pose_from_clip(clip, prev, device)
    cur_pose = tl.get_pose_from_clip(clip, cur, device)

    current = tl.body_pose_vector(cur_pose)
    previous = tl.body_pose_vector(prev_pose)
    pelvis_vel = (cur_pose["pelvis_pos"] - prev_pose["pelvis_pos"]) / cfg.pose_delta_scale_final
    joint_vel = (cur_pose["canon_pos"] - prev_pose["canon_pos"]).reshape(cur.shape[0], -1) / cfg.pose_delta_scale_final
    root_feat = tl.root_delta_feature(clip, prev, cur, cfg, device)
    future_feat = tl.future_root_features(clip, cur, cfg, device).reshape(cur.shape[0], cfg.future_window, 4)
    full_input = tl.build_input(clip, prev, cur, prev_pose, cur_pose, cfg, device)

    root_pos_delta = clip.root_pos[cur] - clip.root_pos[prev]
    root_delta_m_per_frame = torch.linalg.norm(root_pos_delta, dim=-1)
    root_delta_m_per_sec = root_delta_m_per_frame * cfg.fps

    return {
        "npz": str(npz_path),
        "frames": clip.T,
        "fps": clip.fps,
        "position_unit_scale": cfg.position_unit_scale,
        "future_window": cfg.future_window,
        "body_bones": clip.J,
        "body_names_first": clip.body_names[:20],
        "root_motion_m_per_sec": summarize("root_motion_m_per_sec", root_delta_m_per_sec),
        "features": [
            summarize("current_pose", current),
            summarize("previous_pose", previous),
            summarize("pelvis_velocity_norm", pelvis_vel),
            summarize("joint_canonical_velocity_norm", joint_vel),
            summarize("current_root_delta_norm_dx_dz_dyaw", root_feat),
            summarize("future_root_norm_dx_dz_cos_sin", future_feat),
            summarize("full_model_input", full_input),
        ],
        "current_root_delta_columns": {
            "dx_norm": summarize("dx_norm", root_feat[:, 0]),
            "dz_norm": summarize("dz_norm", root_feat[:, 1]),
            "dyaw_norm": summarize("dyaw_norm", root_feat[:, 2]),
        },
        "future_root_columns": {
            "dx_norm": summarize("future_dx_norm", future_feat[:, :, 0]),
            "dz_norm": summarize("future_dz_norm", future_feat[:, :, 1]),
            "cos_delta_yaw": summarize("future_cos_delta_yaw", future_feat[:, :, 2]),
            "sin_delta_yaw": summarize("future_sin_delta_yaw", future_feat[:, :, 3]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect normalized training inputs for NPZ motion clips.")
    parser.add_argument("npz", type=Path)
    parser.add_argument("--future-window-seconds", type=float, default=1.0)
    parser.add_argument("--position-unit-scale", type=float, default=0.01)
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    cfg = tl.TrainConfig()
    cfg.future_window_seconds = args.future_window_seconds
    cfg.position_unit_scale = args.position_unit_scale
    report = inspect_clip(args.npz.resolve(), cfg, args.all_frames)
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
