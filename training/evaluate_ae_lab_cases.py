from __future__ import annotations

import argparse
import csv
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn.functional as F

import train_locomotion as tl
import transition_autoencoder as tae


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def apply_locomotion_config(cfg: tl.TrainConfig, values: dict) -> None:
    valid = {field.name for field in fields(tl.TrainConfig)}
    for key, value in values.items():
        if key in valid:
            setattr(cfg, key, value)


def short_name(path: str | Path) -> str:
    stem = Path(path).stem
    return (
        stem.replace("M_Neutral_", "")
        .replace("Walk_Loop_", "Loop")
        .replace("Walk_Circle_Strafe_", "Circle")
        .replace("Stand_Turn_", "Turn")
        .replace("Stand_Idle_Loop", "Idle")
    )


def load_prior(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = tae.ae_config_from_dict(ckpt["config"])
    model = tae.TransitionAutoencoder(int(ckpt["schema"]["total_dim"]), cfg, dict(ckpt["schema"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def score_rows(model: tae.TransitionAutoencoder, mean: torch.Tensor, std: torch.Tensor, raw: torch.Tensor, compat_weight: float) -> torch.Tensor:
    x = (raw - mean) / std
    recon = model(x)
    target = model.target(x)
    score = tae.reconstruction_loss_rows(model, recon, target, loss_type="mse")
    if compat_weight > 0.0 and model.has_compatibility_head():
        score = score + compat_weight * F.softplus(-model.compatibility_logits(x))
    return score


def summarize(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().cpu()
    return {
        "mean": float(values.mean()),
        "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def load_controller(path: Path, clip: tl.MotionClip, cfg: tl.TrainConfig, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    apply_locomotion_config(cfg, ckpt.get("config", {}))
    cfg.device = str(device)
    cfg.use_torch_compile = False
    input_dim, output_dim = tl.make_batch_dims(clip, cfg)
    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def rollout_features_for_prior(
    checkpoint_path: Path,
    clip_path: Path,
    prior_cfg: tl.TrainConfig,
    device: torch.device,
) -> torch.Tensor:
    model_cfg = tl.TrainConfig()
    model_cfg.device = str(device)
    model_cfg.use_torch_compile = False
    clip_for_model = tl.MotionClip(clip_path, model_cfg)
    model, model_cfg = load_controller(checkpoint_path, clip_for_model, model_cfg, device)
    clip = tl.MotionClip(clip_path, prior_cfg)
    prev_idx = torch.tensor([0], dtype=torch.long, device=device)
    cur_idx = torch.tensor([1], dtype=torch.long, device=device)
    prev_pose = tl.get_pose_from_clip(clip, prev_idx, device)
    cur_pose = tl.get_pose_from_clip(clip, cur_idx, device)
    features: list[torch.Tensor] = []
    for target in range(2, clip.T):
        target_idx = torch.tensor([target], dtype=torch.long, device=device)
        inp = tl.build_input(clip, prev_idx, cur_idx, prev_pose, cur_pose, model_cfg, device)
        raw_out = tl.predict_next_raw(model, inp, cur_pose, model_cfg)
        pred_pose, _raw_pose = tl.output_to_pose(raw_out, clip)
        root_pos, root_rot, _yaw, _heading = tl.root_state(clip, target_idx, prior_cfg, device)
        _global_pos, _global_rot, pred_canon = tl.fk_from_pose(clip, root_pos, root_rot, pred_pose, device)
        next_pose = {
            "pelvis_pos": pred_pose["pelvis_pos"],
            "pelvis_rot6": pred_pose["pelvis_rot6"],
            "nonpelvis_rot6": pred_pose["nonpelvis_rot6"],
            "canon_pos": pred_canon,
            "contacts": pred_pose["contacts"],
        }
        features.append(
            tae.transition_feature_from_next_pose(clip, prev_idx, cur_idx, prev_pose, cur_pose, next_pose, prior_cfg, device)
        )
        prev_pose = cur_pose
        cur_pose = next_pose
        prev_idx = cur_idx
        cur_idx = target_idx
    return torch.cat(features, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score AE priors on GT and adversarial lab perturbations.")
    parser.add_argument("--prior-checkpoint", action="append", required=True)
    parser.add_argument("--prior-label", action="append", default=[])
    parser.add_argument("--npz-path", action="append", required=True)
    parser.add_argument("--bad-checkpoint", action="append", default=[])
    parser.add_argument("--bad-label", action="append", default=[])
    parser.add_argument("--compatibility-score-weight", type=float, default=1.0)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []
    for prior_i, prior_text in enumerate(args.prior_checkpoint):
        prior_path = tl.resolve_path(prior_text)
        label = args.prior_label[prior_i] if prior_i < len(args.prior_label) else prior_path.parent.parent.name
        model, ckpt = load_prior(prior_path, device)
        cfg = tl.TrainConfig()
        apply_locomotion_config(cfg, ckpt.get("locomotion_config", {}))
        cfg.device = str(device)
        cfg.use_torch_compile = False
        mean = ckpt["mean"].to(device)
        std = ckpt["std"].to(device)
        schema = dict(ckpt["schema"])
        horizon = max(int(cfg.future_window), max(0, int(getattr(cfg, "root_lookahead_steps", 0))) + 1)
        for npz_text in args.npz_path:
            npz_path = tl.resolve_path(npz_text)
            clip = tl.MotionClip(npz_path, cfg)
            stop = max(2, clip.T - horizon - 1)
            idx = torch.arange(1, stop + 1, dtype=torch.long, device=device)
            gt = tae.clean_transition_features(clip, idx, cfg, device)
            cases = {
                "gt": gt,
                "statue": tae.make_statue_tier(gt, schema),
                "yaw_body_10deg": tae.make_yaw_body_tier(gt, schema, 10.0),
                "yaw_body_20deg": tae.make_yaw_body_tier(gt, schema, 20.0),
            }
            for bad_i, bad_text in enumerate(args.bad_checkpoint):
                bad_label = args.bad_label[bad_i] if bad_i < len(args.bad_label) else Path(bad_text).parent.parent.name
                cases[f"rollout_{bad_label}"] = rollout_features_for_prior(tl.resolve_path(bad_text), npz_path, cfg, device)
            for case_name, raw in cases.items():
                scores = score_rows(model, mean, std, raw, args.compatibility_score_weight)
                stats = summarize(scores)
                rows.append(
                    {
                        "prior": label,
                        "clip": short_name(npz_path),
                        "case": case_name,
                        **stats,
                    }
                )

    if args.output_csv:
        out = tl.resolve_path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out}")
    for row in rows:
        print(
            f"{row['prior']:>18s} {row['clip']:>20s} {row['case']:>20s} "
            f"mean={float(row['mean']):.6g} p95={float(row['p95']):.6g} max={float(row['max']):.6g}"
        )


if __name__ == "__main__":
    main()
