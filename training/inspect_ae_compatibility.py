from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

import train_locomotion as tl
import transition_autoencoder as tae


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def short_name(path: str | Path) -> str:
    stem = Path(path).stem
    return (
        stem.replace("M_Neutral_", "")
        .replace("Walk_Loop_", "")
        .replace("Stand_Idle_Loop", "Idle")
    )


@torch.no_grad()
def transition_scores(
    model: tae.TransitionAutoencoder,
    x_norm: torch.Tensor,
    compatibility_weight: float,
) -> torch.Tensor:
    recon = F.mse_loss(model(x_norm), x_norm, reduction="none").mean(dim=-1)
    if compatibility_weight > 0.0 and model.has_compatibility_head():
        recon = recon + compatibility_weight * F.softplus(-model.compatibility_logits(x_norm))
    return recon


def load_prior(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = tae.AEConfig(**ckpt["config"])
    model = tae.TransitionAutoencoder(int(ckpt["schema"]["total_dim"]), cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def build_clip_features(clips: list[tl.MotionClip], cfg: tl.TrainConfig, device: torch.device):
    features = []
    names = []
    for clip in clips:
        stop = clip.cyclic_period if cfg.cyclic_animation else clip.T - 1
        idx = torch.arange(1, stop, dtype=torch.long, device=device)
        features.append(tae.clean_transition_features(clip, idx, cfg, device))
        names.append(short_name(clip.path))
    return names, features


def write_matrix(path: Path, names: list[str], matrix: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["root/body", *names])
        for name, row in zip(names, matrix.detach().cpu().tolist()):
            writer.writerow([name, *row])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-path", required=True)
    parser.add_argument("--prior-checkpoint", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cyclic-animation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compatibility-score-weight", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = tl.TrainConfig()
    cfg.cyclic_animation = args.cyclic_animation
    clips = tl.load_clips(tl.resolve_path(args.folder_path), cfg)
    model, ckpt = load_prior(tl.resolve_path(args.prior_checkpoint), device)
    mean = ckpt["mean"].to(device)
    std = ckpt["std"].to(device)
    schema = ckpt["schema"]
    root_start = schema["input_root_start"]
    root_end = schema["input_root_end"]

    names, raw_features = build_clip_features(clips, cfg, device)
    xs = [(feat - mean) / std for feat in raw_features]
    common = min(x.shape[0] for x in xs)
    matrix = torch.empty((len(xs), len(xs)), device=device)
    recon_matrix = torch.empty_like(matrix)
    compat_matrix = torch.empty_like(matrix)

    with torch.no_grad():
        for i, root_x in enumerate(xs):
            root_slice = root_x[:common, root_start:root_end]
            for j, body_x in enumerate(xs):
                paired = body_x[:common].clone()
                paired[:, root_start:root_end] = root_slice
                recon = F.mse_loss(model(paired), paired, reduction="none").mean(dim=-1)
                if model.has_compatibility_head():
                    compat = F.softplus(-model.compatibility_logits(paired))
                else:
                    compat = torch.zeros_like(recon)
                recon_matrix[i, j] = recon.mean()
                compat_matrix[i, j] = compat.mean()
                matrix[i, j] = recon_matrix[i, j] + args.compatibility_score_weight * compat_matrix[i, j]

    print("rows=root, cols=body")
    print("names:", ", ".join(names))
    for label, mat in (("total", matrix), ("reconstruction", recon_matrix), ("compatibility", compat_matrix)):
        print(f"\n{label}:")
        for i, name in enumerate(names):
            values = " ".join(f"{float(mat[i, j]):7.4f}" for j in range(len(names)))
            print(f"{name:>5s} {values}")
        for i, name in enumerate(names):
            order = torch.argsort(mat[i]).tolist()
            print(
                f"{label} root={name} correct_rank={order.index(i) + 1} "
                f"best_body={names[order[0]]} diag={float(mat[i, i]):.6g} best={float(mat[i, order[0]]):.6g}"
            )

    if args.output_csv:
        base = tl.resolve_path(args.output_csv)
        write_matrix(base, names, matrix)
        write_matrix(base.with_name(base.stem + "_reconstruction.csv"), names, recon_matrix)
        write_matrix(base.with_name(base.stem + "_compatibility.csv"), names, compat_matrix)


if __name__ == "__main__":
    main()
