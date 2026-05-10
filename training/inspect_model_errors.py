from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_locomotion as tl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def apply_config(cfg: tl.TrainConfig, values: dict) -> None:
    valid = {field.name for field in fields(tl.TrainConfig)}
    for key, value in values.items():
        if key not in valid:
            continue
        current = getattr(cfg, key)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(cfg, key, value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print per-frame model-vs-ground-truth joint errors.")
    parser.add_argument("--npz-path", default="data/npz_cascadeur/testcasc.npz")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=("one_step", "autoregressive"), default="one_step")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    ckpt = torch.load(resolve_path(args.checkpoint_path), map_location="cpu", weights_only=False)
    cfg = tl.TrainConfig()
    apply_config(cfg, ckpt.get("config", {}))
    cfg.device = args.device
    device = torch.device(args.device)
    clip = tl.MotionClip(resolve_path(args.npz_path), cfg)
    model = tl.MLPController(*tl.make_batch_dims(clip, cfg), cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    watch_names = [name for name in ("foot_l", "ball_l", "foot_r", "ball_r", "pelvis", "hand_l", "hand_r") if name in clip.body_names]
    watch = [(name, clip.body_names.index(name)) for name in watch_names]
    rows = []
    with torch.no_grad():
        prev_pose_roll = tl.get_pose_from_clip(clip, torch.tensor([0], dtype=torch.long), device)
        cur_pose_roll = tl.get_pose_from_clip(clip, torch.tensor([1], dtype=torch.long), device)
        prev_idx_roll = torch.tensor([0], dtype=torch.long)
        cur_idx_roll = torch.tensor([1], dtype=torch.long)
        for frame in range(2, clip.T):
            target = torch.tensor([frame], dtype=torch.long)
            if args.mode == "autoregressive":
                prev = prev_idx_roll
                cur = cur_idx_roll
                prev_pose = prev_pose_roll
                cur_pose = cur_pose_roll
            else:
                prev = torch.tensor([frame - 2], dtype=torch.long)
                cur = torch.tensor([frame - 1], dtype=torch.long)
                prev_pose = tl.get_pose_from_clip(clip, prev, device)
                cur_pose = tl.get_pose_from_clip(clip, cur, device)
            inp = tl.build_input(
                clip,
                prev,
                cur,
                prev_pose,
                cur_pose,
                cfg,
                device,
            )
            raw_out = tl.predict_next_raw(model, inp, cur_pose, cfg)
            pred, raw = tl.output_to_pose(raw_out, clip)
            target_pose = tl.get_pose_from_clip(clip, target, device)
            total, parts, _ = tl.compute_losses(clip, pred, raw, target_pose, target, cfg, device)
            gp, _, _ = tl.fk_from_pose(clip, clip.root_pos[target].to(device), clip.root_rot[target].to(device), pred, device)
            tgt = clip.global_pos[target].to(device)
            err = torch.linalg.norm(gp - tgt, dim=-1)[0]
            watched = {name: float(err[index].cpu()) for name, index in watch}
            foot_worst = max(watched.get(name, 0.0) for name in ("foot_l", "ball_l", "foot_r", "ball_r"))
            rows.append(
                {
                    "frame": frame,
                    "mean": float(err.mean().cpu()),
                    "max": float(err.max().cpu()),
                    "foot_worst": foot_worst,
                    "watched": watched,
                    "total": float(total.cpu()),
                    "parts": {k: float(v.cpu()) for k, v in parts.items()},
                }
            )
            if args.mode == "autoregressive":
                _, _, canon_pos = tl.fk_from_pose(
                    clip,
                    clip.root_pos[target].to(device),
                    clip.root_rot[target].to(device),
                    pred,
                    device,
                )
                prev_pose_roll = cur_pose_roll
                cur_pose_roll = {
                    "pelvis_pos": pred["pelvis_pos"],
                    "pelvis_rot6": pred["pelvis_rot6"],
                    "nonpelvis_rot6": pred["nonpelvis_rot6"],
                    "canon_pos": canon_pos,
                }
                prev_idx_roll = cur_idx_roll
                cur_idx_roll = target

    print(f"mode={args.mode}")
    print(f"checkpoint={resolve_path(args.checkpoint_path)} epoch={ckpt.get('epoch')} best_val={ckpt.get('best_val')}")
    print(f"effectors={cfg.end_effector_bones} alpha4={cfg.alpha4_end_effector_location} alpha6={cfg.alpha6_full_body_location}")
    print("Worst foot frames:")
    for row in sorted(rows, key=lambda item: item["foot_worst"], reverse=True)[: args.top]:
        watched = " ".join(f"{k}={v:.4f}" for k, v in row["watched"].items())
        print(
            f"frame={row['frame']:02d} foot_worst={row['foot_worst']:.4f} "
            f"mean={row['mean']:.4f} max={row['max']:.4f} total={row['total']:.4f} "
            f"ee={row['parts']['end_effector_location']:.4f} full={row['parts']['full_body_location']:.4f} "
            f"{watched}"
        )


if __name__ == "__main__":
    main()
