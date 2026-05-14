from __future__ import annotations

import argparse
import csv
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

import contact_physics as cp
import train_locomotion as tl


def apply_config_dict(cfg: tl.TrainConfig, values: dict) -> None:
    valid = {field.name for field in fields(tl.TrainConfig)}
    for key, value in values.items():
        if key not in valid:
            continue
        current = getattr(cfg, key)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(cfg, key, value)


def short_name(path: str | Path) -> str:
    stem = Path(path).stem
    return (
        stem.replace("M_Neutral_", "")
        .replace("Walk_Loop_", "")
        .replace("Stand_Idle_Loop", "Idle")
    )


def load_controller(path: Path, cfg: tl.TrainConfig, clip: tl.MotionClip, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    apply_config_dict(cfg, checkpoint.get("config", {}))
    cfg.device = str(device)
    cfg.use_torch_compile = False
    input_dim, output_dim = tl.make_batch_dims(clip, cfg)
    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def rollout_autoreg(
    model: torch.nn.Module,
    clip: tl.MotionClip,
    cfg: tl.TrainConfig,
    device: torch.device,
    frame_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_pos = torch.zeros((frame_count, clip.J, 3), dtype=torch.float32, device=device)
    pred_rot = torch.zeros((frame_count, clip.J, 3, 3), dtype=torch.float32, device=device)
    gt_pos = clip.global_pos[:frame_count].to(device)
    gt_rot = clip.global_rot[:frame_count].to(device)
    pred_pos[:2] = gt_pos[:2]
    pred_rot[:2] = gt_rot[:2]

    prev_idx = torch.tensor([0], dtype=torch.long, device=device)
    cur_idx = torch.tensor([1], dtype=torch.long, device=device)
    prev_pose = tl.get_pose_from_clip(clip, prev_idx, device)
    cur_pose = tl.get_pose_from_clip(clip, cur_idx, device)
    prev_pose, cur_pose = tl.maybe_apply_initial_offsets(clip, prev_idx, cur_idx, prev_pose, cur_pose, cfg, device)

    for target in range(2, frame_count):
        target_idx = torch.tensor([target], dtype=torch.long, device=device)
        root_pos, root_rot, _yaw, _heading = tl.root_state(clip, target_idx, cfg, device)
        inp = tl.build_input(clip, prev_idx, cur_idx, prev_pose, cur_pose, cfg, device)
        raw_out = tl.predict_next_raw(model, inp, cur_pose, cfg)
        pred_pose, _raw_pose = tl.output_to_pose(raw_out, clip)
        global_pos, global_rot, canon_pos = tl.fk_from_pose(clip, root_pos, root_rot, pred_pose, device)
        pred_pos[target] = global_pos[0]
        pred_rot[target] = global_rot[0]
        prev_pose = cur_pose
        cur_pose = {
            "pelvis_pos": pred_pose["pelvis_pos"],
            "pelvis_rot6": pred_pose["pelvis_rot6"],
            "nonpelvis_rot6": pred_pose["nonpelvis_rot6"],
            "canon_pos": canon_pos,
            "contacts": pred_pose["contacts"],
        }
        prev_idx = cur_idx
        cur_idx = target_idx
    return pred_pos, pred_rot


@torch.no_grad()
def slide_metrics(
    positions: torch.Tensor,
    rotations: torch.Tensor,
    contacts: torch.Tensor,
    clip: tl.MotionClip,
    fps: float,
    near_height_threshold: float,
) -> dict[str, float]:
    prev_pos = positions[:-1]
    cur_pos = positions[1:]
    prev_rot = rotations[:-1]
    cur_rot = rotations[1:]
    speeds = cp.foot_slide_speeds(
        prev_pos,
        prev_rot,
        cur_pos,
        cur_rot,
        tuple(int(x) for x in clip.foot_indices_tensor.tolist()),
        tuple(int(x) for x in clip.toe_indices_tensor.tolist()),
        fps,
    )
    heights, _points = cp.foot_lowest_heights_and_points(
        cur_pos,
        cur_rot,
        tuple(int(x) for x in clip.foot_indices_tensor.tolist()),
        tuple(int(x) for x in clip.toe_indices_tensor.tolist()),
    )
    source_contact = contacts[1 : 1 + speeds.shape[0]].to(speeds.device) > 0.5
    near_ground = heights <= near_height_threshold
    either_mask = torch.logical_or(source_contact, near_ground)

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        if not bool(mask.any()):
            return 0.0
        return float(values[mask].mean().cpu())

    def masked_p95(values: torch.Tensor, mask: torch.Tensor) -> float:
        if not bool(mask.any()):
            return 0.0
        return float(torch.quantile(values[mask], 0.95).cpu())

    def masked_max(values: torch.Tensor, mask: torch.Tensor) -> float:
        if not bool(mask.any()):
            return 0.0
        return float(values[mask].max().cpu())

    metrics: dict[str, float] = {
        "all_mean_mps": float(speeds.mean().cpu()),
        "all_p95_mps": float(torch.quantile(speeds, 0.95).cpu()),
        "all_max_mps": float(speeds.max().cpu()),
        "contact_mean_mps": masked_mean(speeds, source_contact),
        "contact_p95_mps": masked_p95(speeds, source_contact),
        "contact_max_mps": masked_max(speeds, source_contact),
        "near_ground_mean_mps": masked_mean(speeds, near_ground),
        "near_ground_p95_mps": masked_p95(speeds, near_ground),
        "near_ground_max_mps": masked_max(speeds, near_ground),
        "contact_or_near_mean_mps": masked_mean(speeds, either_mask),
        "contact_or_near_p95_mps": masked_p95(speeds, either_mask),
        "contact_or_near_max_mps": masked_max(speeds, either_mask),
        "min_height_m": float(heights.min().cpu()),
        "mean_height_m": float(heights.mean().cpu()),
        "contact_frames": int(source_contact.any(dim=-1).sum().cpu()),
        "near_ground_frames": int(near_ground.any(dim=-1).sum().cpu()),
    }
    for side, label in enumerate(("L", "R")):
        side_contact = source_contact[:, side]
        side_near = near_ground[:, side]
        side_either = torch.logical_or(side_contact, side_near)
        metrics[f"{label}_contact_mean_mps"] = masked_mean(speeds[:, side], side_contact)
        metrics[f"{label}_contact_p95_mps"] = masked_p95(speeds[:, side], side_contact)
        metrics[f"{label}_near_mean_mps"] = masked_mean(speeds[:, side], side_near)
        metrics[f"{label}_near_p95_mps"] = masked_p95(speeds[:, side], side_near)
        metrics[f"{label}_contact_or_near_p95_mps"] = masked_p95(speeds[:, side], side_either)
        metrics[f"{label}_min_height_m"] = float(heights[:, side].min().cpu())
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only geometric foot sliding monitor for model rollouts.")
    parser.add_argument("--folder-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--label", default="model")
    parser.add_argument("--output-csv", default="training/runs/foot_sliding_report.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cyclic-animation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--near-height-threshold", type=float, default=0.025)
    args = parser.parse_args()

    device = torch.device(args.device)
    base_cfg = tl.TrainConfig()
    base_cfg.cyclic_animation = args.cyclic_animation
    folder = tl.resolve_path(args.folder_path)
    initial_clips = tl.load_clips(folder, base_cfg)
    cfg = tl.TrainConfig()
    cfg.cyclic_animation = args.cyclic_animation
    model = load_controller(tl.resolve_path(args.checkpoint_path), cfg, initial_clips[0], device)
    clips = tl.load_clips(folder, cfg)

    rows: list[dict[str, object]] = []
    for clip in clips:
        limit = clip.cyclic_period if cfg.cyclic_animation else clip.T
        if args.max_frames > 0:
            limit = min(limit, args.max_frames)
        frame_count = max(3, int(limit))
        pred_pos, pred_rot = rollout_autoreg(model, clip, cfg, device, frame_count)
        gt_pos = clip.global_pos[:frame_count].to(device)
        gt_rot = clip.global_rot[:frame_count].to(device)
        contacts = clip.contacts[:frame_count].to(device)
        gt = slide_metrics(gt_pos, gt_rot, contacts, clip, clip.fps, args.near_height_threshold)
        pred = slide_metrics(pred_pos, pred_rot, contacts, clip, clip.fps, args.near_height_threshold)
        row: dict[str, object] = {
            "label": args.label,
            "clip": short_name(clip.path),
            "frames": frame_count,
        }
        for key, value in gt.items():
            row[f"gt_{key}"] = value
        for key, value in pred.items():
            row[f"pred_{key}"] = value
        row["delta_contact_p95_mps"] = pred["contact_p95_mps"] - gt["contact_p95_mps"]
        row["delta_near_ground_p95_mps"] = pred["near_ground_p95_mps"] - gt["near_ground_p95_mps"]
        row["delta_contact_or_near_p95_mps"] = pred["contact_or_near_p95_mps"] - gt["contact_or_near_p95_mps"]
        rows.append(row)

    output = tl.resolve_path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {output}")
    print("clip pred_contact_p95 gt_contact_p95 delta pred_near_p95 gt_near_p95 delta min_h")
    for row in rows:
        print(
            f"{row['clip']:>5s} "
            f"{float(row['pred_contact_p95_mps']):.4f} {float(row['gt_contact_p95_mps']):.4f} "
            f"{float(row['delta_contact_p95_mps']):+.4f} "
            f"{float(row['pred_near_ground_p95_mps']):.4f} {float(row['gt_near_ground_p95_mps']):.4f} "
            f"{float(row['delta_near_ground_p95_mps']):+.4f} "
            f"{float(row['pred_min_height_m']):+.4f}"
        )
    pred_contact = np.asarray([float(row["pred_contact_p95_mps"]) for row in rows], dtype=np.float64)
    gt_contact = np.asarray([float(row["gt_contact_p95_mps"]) for row in rows], dtype=np.float64)
    pred_near = np.asarray([float(row["pred_near_ground_p95_mps"]) for row in rows], dtype=np.float64)
    gt_near = np.asarray([float(row["gt_near_ground_p95_mps"]) for row in rows], dtype=np.float64)
    print(
        f"summary contact_p95 pred_mean={pred_contact.mean():.4f} gt_mean={gt_contact.mean():.4f} "
        f"delta={pred_contact.mean() - gt_contact.mean():+.4f}"
    )
    print(
        f"summary near_ground_p95 pred_mean={pred_near.mean():.4f} gt_mean={gt_near.mean():.4f} "
        f"delta={pred_near.mean() - gt_near.mean():+.4f}"
    )


if __name__ == "__main__":
    main()
