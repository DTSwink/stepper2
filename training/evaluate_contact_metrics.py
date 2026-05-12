from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

import contact_physics as cp
import train_locomotion as tl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def apply_config_dict(cfg: tl.TrainConfig, values: dict) -> None:
    valid = {field.name for field in fields(tl.TrainConfig)}
    for key, value in values.items():
        if key not in valid:
            continue
        current = getattr(cfg, key)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(cfg, key, value)


def load_model(checkpoint_path: Path, npz_path: Path, device: torch.device) -> tuple[dict, tl.TrainConfig, tl.MotionClip, torch.nn.Module]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = tl.TrainConfig()
    apply_config_dict(cfg, checkpoint.get("config", {}))
    clip = tl.MotionClip(npz_path, cfg)
    input_dim, output_dim = tl.make_batch_dims(clip, cfg)
    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return checkpoint, cfg, clip, model


def pose_global(clip: tl.MotionClip, idx: torch.Tensor, pose: dict[str, torch.Tensor], device: torch.device):
    root_pos = clip.root_pos[idx].to(device)
    root_rot = clip.root_rot[idx].to(device)
    return tl.fk_from_pose(clip, root_pos, root_rot, pose, device)


@torch.no_grad()
def collect_metrics(
    checkpoint_path: Path,
    npz_path: Path,
    device: torch.device,
    autoregressive: bool,
) -> dict[str, float]:
    checkpoint, cfg, clip, model = load_model(checkpoint_path, npz_path, device)
    geom = cp.DEFAULT_GEOMETRY
    contact_probs = []
    target_contacts = []
    heights = []
    speeds = []
    errors = []

    if autoregressive:
        prev_idx = torch.tensor([0], dtype=torch.long)
        cur_idx = torch.tensor([1], dtype=torch.long)
        prev_pose = tl.get_pose_from_clip(clip, prev_idx, device)
        cur_pose = tl.get_pose_from_clip(clip, cur_idx, device)

    for target in range(2, clip.T):
        target_idx = torch.tensor([target], dtype=torch.long)
        if autoregressive:
            tf_prev_idx = prev_idx
            tf_cur_idx = cur_idx
            tf_prev_pose = prev_pose
            tf_cur_pose = cur_pose
        else:
            tf_prev_idx = torch.tensor([target - 2], dtype=torch.long)
            tf_cur_idx = torch.tensor([target - 1], dtype=torch.long)
            tf_prev_pose = tl.get_pose_from_clip(clip, tf_prev_idx, device)
            tf_cur_pose = tl.get_pose_from_clip(clip, tf_cur_idx, device)

        inp = tl.build_input(clip, tf_prev_idx, tf_cur_idx, tf_prev_pose, tf_cur_pose, cfg, device)
        raw = tl.predict_next_raw(model, inp, tf_cur_pose, cfg)
        pred_pose, _raw_pose = tl.output_to_pose(raw, clip)
        pred_pos, pred_rot, pred_canon = pose_global(clip, target_idx, pred_pose, device)
        target_pos = clip.global_pos[target_idx].to(device)
        foot_h, _points = cp.foot_lowest_heights_and_points(
            pred_pos, pred_rot, tuple(clip.foot_indices), tuple(clip.toe_indices), geom
        )
        cur_pos, cur_rot, _canon = pose_global(clip, tf_cur_idx, tf_cur_pose, device)
        foot_s = cp.foot_slide_speeds(
            cur_pos, cur_rot, pred_pos, pred_rot, tuple(clip.foot_indices), tuple(clip.toe_indices), clip.fps, geom
        )
        contact_probs.append(pred_pose["contacts"].cpu().numpy()[0])
        target_contacts.append(clip.contacts[target].cpu().numpy())
        heights.append(foot_h.cpu().numpy()[0])
        speeds.append(foot_s.cpu().numpy()[0])
        errors.append(torch.linalg.norm(pred_pos - target_pos, dim=-1).mean().item())

        if autoregressive:
            prev_pose = cur_pose
            cur_pose = {
                "pelvis_pos": pred_pose["pelvis_pos"],
                "pelvis_rot6": pred_pose["pelvis_rot6"],
                "nonpelvis_rot6": pred_pose["nonpelvis_rot6"],
                "canon_pos": pred_canon,
                "contacts": pred_pose["contacts"],
            }
            prev_idx = cur_idx
            cur_idx = target_idx

    contact_probs_np = np.asarray(contact_probs)
    target_contacts_np = np.asarray(target_contacts)
    heights_np = np.asarray(heights)
    speeds_np = np.asarray(speeds)
    errors_np = np.asarray(errors)
    pred_on = contact_probs_np > 0.5
    target_on = target_contacts_np > 0.5
    on_heights = np.where(pred_on, heights_np, np.nan)
    on_speeds = np.where(pred_on, speeds_np, np.nan)
    above = np.where(pred_on, heights_np > geom.height_threshold_m, False)
    penetrate = heights_np < 0.0
    both_off = np.logical_not(pred_on).all(axis=1)
    return {
        "epoch": float(checkpoint.get("epoch", -1)),
        "rollout_k": float(checkpoint.get("rollout_k", -1)),
        "best_val": float(checkpoint.get("best_val", np.nan)),
        "joint_error_avg_m": float(errors_np.mean()),
        "joint_error_end_m": float(errors_np[-1]),
        "joint_error_max_m": float(errors_np.max()),
        "contact_accuracy": float((pred_on == target_on).mean()),
        "contact_on_rate_l": float(pred_on[:, 0].mean()),
        "contact_on_rate_r": float(pred_on[:, 1].mean()),
        "both_contacts_off_rate": float(both_off.mean()),
        "contact_height_mean_m": float(np.nanmean(on_heights)),
        "contact_height_max_m": float(np.nanmax(on_heights)),
        "contact_height_above_threshold_rate": float(above.sum() / max(pred_on.sum(), 1)),
        "penetration_min_m": float(heights_np.min()),
        "penetration_frame_rate": float(penetrate.any(axis=1).mean()),
        "contact_slide_speed_mean_mps": float(np.nanmean(on_speeds)),
        "contact_slide_speed_max_mps": float(np.nanmax(on_speeds)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--npz-path", default="data/fbx/npz_final/testcasc.npz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher-forced", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    metrics = collect_metrics(
        resolve_path(args.checkpoint_path),
        resolve_path(args.npz_path),
        device,
        autoregressive=not args.teacher_forced,
    )
    mode = "teacher_forced" if args.teacher_forced else "autoregressive"
    print(f"mode={mode}")
    for key, value in metrics.items():
        print(f"{key}={value:.6g}")


if __name__ == "__main__":
    main()
