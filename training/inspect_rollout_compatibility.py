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


def load_controller(path: Path, clip: tl.MotionClip, cfg: tl.TrainConfig, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    apply_config_dict(cfg, ckpt.get("config", {}))
    cfg.device = str(device)
    cfg.use_torch_compile = False
    input_dim, output_dim = tl.make_batch_dims(clip, cfg)
    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def load_prior(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = tae.AEConfig(**ckpt["config"])
    model = tae.TransitionAutoencoder(int(ckpt["schema"]["total_dim"]), cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def prior_score(
    prior: tae.TransitionAutoencoder,
    mean: torch.Tensor,
    std: torch.Tensor,
    features: torch.Tensor,
    compatibility_score_weight: float,
) -> torch.Tensor:
    x = (features - mean) / std
    recon = F.mse_loss(prior(x), x, reduction="none").mean(dim=-1)
    if compatibility_score_weight > 0.0 and prior.has_compatibility_head():
        recon = recon + compatibility_score_weight * F.softplus(-prior.compatibility_logits(x))
    return recon


@torch.no_grad()
def rollout_transition_features(
    model: torch.nn.Module,
    clip: tl.MotionClip,
    cfg: tl.TrainConfig,
    device: torch.device,
    max_steps: int,
) -> torch.Tensor:
    steps = min(max_steps, max(1, (clip.cyclic_period if cfg.cyclic_animation else clip.T - 1) - 1))
    prev_idx = torch.tensor([0], dtype=torch.long, device=device)
    cur_idx = torch.tensor([1], dtype=torch.long, device=device)
    prev_pose = tl.get_pose_from_clip(clip, prev_idx, device)
    cur_pose = tl.get_pose_from_clip(clip, cur_idx, device)
    prev_pose, cur_pose = tl.maybe_apply_initial_offsets(clip, prev_idx, cur_idx, prev_pose, cur_pose, cfg, device)
    features = []
    for _step in range(steps):
        inp = tl.build_input(clip, prev_idx, cur_idx, prev_pose, cur_pose, cfg, device)
        raw_out = tl.predict_next_raw(model, inp, cur_pose, cfg)
        pred_pose, _raw_pose = tl.output_to_pose(raw_out, clip)
        target_idx = cur_idx + 1
        root_pos, root_rot, _yaw, _heading = tl.root_state(clip, target_idx, cfg, device)
        _global_pos, _global_rot, pred_canon = tl.fk_from_pose(clip, root_pos, root_rot, pred_pose, device)
        next_pose = {
            "pelvis_pos": pred_pose["pelvis_pos"],
            "pelvis_rot6": pred_pose["pelvis_rot6"],
            "nonpelvis_rot6": pred_pose["nonpelvis_rot6"],
            "canon_pos": pred_canon,
            "contacts": pred_pose["contacts"],
        }
        features.append(
            tae.transition_feature_from_next_pose(clip, prev_idx, cur_idx, prev_pose, cur_pose, next_pose, cfg, device)
        )
        prev_pose = cur_pose
        cur_pose = next_pose
        prev_idx = cur_idx
        cur_idx = target_idx
    return torch.cat(features, dim=0)


@torch.no_grad()
def candidate_root_slices(
    clips: list[tl.MotionClip],
    cfg: tl.TrainConfig,
    device: torch.device,
    schema: dict,
    steps: int,
) -> list[torch.Tensor]:
    idx = torch.arange(1, steps + 1, dtype=torch.long, device=device)
    root_start = schema["input_root_start"]
    root_end = schema["input_root_end"]
    roots = []
    for clip in clips:
        clean = tae.clean_transition_features(clip, idx, cfg, device)
        roots.append(clean[:, root_start:root_end])
    return roots


def write_rows(path: Path, rows: list[dict[str, object]], score_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "commanded",
        "best_match",
        "commanded_rank",
        "commanded_score",
        "best_score",
        "gap_commanded_minus_best",
        *score_fields,
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only root/body compatibility monitor for trained rollouts.")
    parser.add_argument("--folder-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--prior-checkpoint", required=True)
    parser.add_argument("--output-csv", default="training/runs/rollout_compatibility_report.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cyclic-animation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=111)
    parser.add_argument("--compatibility-score-weight", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = tl.TrainConfig()
    cfg.cyclic_animation = args.cyclic_animation
    folder = tl.resolve_path(args.folder_path)
    clips = tl.load_clips(folder, cfg)
    names = [short_name(clip.path) for clip in clips]
    model, _controller_ckpt = load_controller(tl.resolve_path(args.checkpoint_path), clips[0], cfg, device)
    clips = tl.load_clips(folder, cfg)
    prior, prior_ckpt = load_prior(tl.resolve_path(args.prior_checkpoint), device)
    mean = prior_ckpt["mean"].to(device)
    std = prior_ckpt["std"].to(device)
    schema = prior_ckpt["schema"]
    root_start = schema["input_root_start"]
    root_end = schema["input_root_end"]

    steps = min(args.max_steps, min((clip.cyclic_period if cfg.cyclic_animation else clip.T - 1) - 1 for clip in clips))
    roots = candidate_root_slices(clips, cfg, device, schema, steps)
    score_fields = [f"score_{name}" for name in names]
    rows: list[dict[str, object]] = []

    for commanded_index, clip in enumerate(clips):
        generated = rollout_transition_features(model, clip, cfg, device, steps)
        scores = []
        for root_slice in roots:
            paired = generated.clone()
            paired[:, root_start:root_end] = root_slice
            scores.append(float(prior_score(prior, mean, std, paired, args.compatibility_score_weight).mean().cpu()))
        order = sorted(range(len(scores)), key=lambda i: scores[i])
        commanded_rank = order.index(commanded_index) + 1
        best_index = order[0]
        row: dict[str, object] = {
            "commanded": names[commanded_index],
            "best_match": names[best_index],
            "commanded_rank": commanded_rank,
            "commanded_score": scores[commanded_index],
            "best_score": scores[best_index],
            "gap_commanded_minus_best": scores[commanded_index] - scores[best_index],
        }
        for name, score in zip(names, scores):
            row[f"score_{name}"] = score
        rows.append(row)

    write_rows(tl.resolve_path(args.output_csv), rows, score_fields)
    print(f"wrote {tl.resolve_path(args.output_csv)}")
    print("commanded -> best_match rank gap")
    for row in rows:
        print(
            f"{row['commanded']:>5s} -> {row['best_match']:<5s} "
            f"rank={row['commanded_rank']} gap={float(row['gap_commanded_minus_best']):.6g} "
            f"cmd={float(row['commanded_score']):.6g} best={float(row['best_score']):.6g}"
        )


if __name__ == "__main__":
    main()
