from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import train_locomotion as tl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AEConfig:
    folder_path: str = "data/npz_final"
    run_name: str = "transition_ae"
    output_dir: str = "training/runs"
    latent_dim: int = 64
    hidden_dim: int = 512
    num_hidden_layers: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    max_epochs: int = 2000
    seed: int = 1234
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cyclic_animation: bool = False
    input_noise_std: float = 0.0
    std_floor: float = 1e-4
    target_loss_reduction: float = 0.995
    stall_patience_epochs: int = 120
    min_delta: float = 1e-6
    tier_eval_every_epochs: int = 25


class TransitionAutoencoder(nn.Module):
    def __init__(self, dim: int, cfg: AEConfig):
        super().__init__()
        encoder: list[nn.Module] = []
        in_dim = dim
        for _ in range(cfg.num_hidden_layers):
            encoder += [nn.Linear(in_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.GELU()]
            in_dim = cfg.hidden_dim
        encoder += [nn.Linear(in_dim, cfg.latent_dim), nn.GELU()]
        decoder: list[nn.Module] = []
        in_dim = cfg.latent_dim
        for _ in range(cfg.num_hidden_layers):
            decoder += [nn.Linear(in_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.GELU()]
            in_dim = cfg.hidden_dim
        decoder += [nn.Linear(in_dim, dim)]
        self.net = nn.Sequential(*(encoder + decoder))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def transition_schema(clip: tl.MotionClip, cfg: tl.TrainConfig) -> dict[str, int]:
    pose_dim = 3 + 6 + clip.J * 3 + clip.Jn * 6 + 2
    velocity_dim = 3 + clip.J * 3
    input_dim, output_dim = tl.make_batch_dims(clip, cfg)
    next_canon_dim = clip.J * 3
    next_velocity_dim = 3 + clip.J * 3
    return {
        "pose_dim": pose_dim,
        "velocity_dim": velocity_dim,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "next_canon_dim": next_canon_dim,
        "next_velocity_dim": next_velocity_dim,
        "total_dim": input_dim + output_dim + next_canon_dim + next_velocity_dim,
        "input_root_start": pose_dim * 2 + velocity_dim,
        "input_root_end": input_dim,
        "next_output_start": input_dim,
        "next_canon_start": input_dim + output_dim,
        "next_velocity_start": input_dim + output_dim + next_canon_dim,
    }


def transition_feature_from_next_pose(
    clip: tl.MotionClip,
    prev_idx: torch.Tensor,
    cur_idx: torch.Tensor,
    prev_pose: dict[str, torch.Tensor],
    cur_pose: dict[str, torch.Tensor],
    next_pose: dict[str, torch.Tensor],
    cfg: tl.TrainConfig,
    device: torch.device,
) -> torch.Tensor:
    model_input = tl.build_input(clip, prev_idx, cur_idx, prev_pose, cur_pose, cfg, device)
    next_output = tl.pose_target_output(next_pose)
    pelvis_next_vel = (next_pose["pelvis_pos"] - cur_pose["pelvis_pos"]) / cfg.pose_delta_scale_final
    joint_next_vel = (next_pose["canon_pos"] - cur_pose["canon_pos"]).reshape(cur_idx.shape[0], -1)
    joint_next_vel = joint_next_vel / cfg.pose_delta_scale_final
    return torch.cat(
        (
            model_input,
            next_output,
            next_pose["canon_pos"].reshape(cur_idx.shape[0], -1),
            pelvis_next_vel,
            joint_next_vel,
        ),
        dim=-1,
    )


def clean_transition_features(
    clip: tl.MotionClip,
    cur_idx: torch.Tensor,
    cfg: tl.TrainConfig,
    device: torch.device,
) -> torch.Tensor:
    prev_idx = cur_idx - 1
    next_idx = cur_idx + 1
    prev_pose = tl.get_pose_from_clip(clip, prev_idx, device)
    cur_pose = tl.get_pose_from_clip(clip, cur_idx, device)
    next_pose = tl.get_pose_from_clip(clip, next_idx, device)
    return transition_feature_from_next_pose(clip, prev_idx, cur_idx, prev_pose, cur_pose, next_pose, cfg, device)


def collect_clean_features(clips: list[tl.MotionClip], cfg: tl.TrainConfig, device: torch.device) -> torch.Tensor:
    chunks = []
    for clip in clips:
        stop = clip.cyclic_period if cfg.cyclic_animation else clip.T - 1
        idx = torch.arange(1, stop, dtype=torch.long, device=device)
        chunks.append(clean_transition_features(clip, idx, cfg, device).detach().cpu())
    return torch.cat(chunks, dim=0)


def normalise(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def denormalise(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std + mean


def alteration_mask(schema: dict[str, int], device: torch.device) -> torch.Tensor:
    mask = torch.ones(schema["total_dim"], device=device)
    mask[schema["input_root_start"] : schema["input_root_end"]] = 0.0
    return mask


def make_statue_tier(x: torch.Tensor, schema: dict[str, int]) -> torch.Tensor:
    y = x.clone()
    pose_dim = schema["pose_dim"]
    input_dim = schema["input_dim"]
    output_start = schema["next_output_start"]
    canon_start = schema["next_canon_start"]
    vel_start = schema["next_velocity_start"]
    output_dim = schema["output_dim"]
    canon_dim = schema["next_canon_dim"]
    vel_dim = schema["next_velocity_dim"]

    current_pose = y[:, :pose_dim]
    current_output = torch.cat(
        (
            current_pose[:, :9],
            current_pose[:, 9 + canon_dim :],
        ),
        dim=-1,
    )
    y[:, output_start : output_start + output_dim] = current_output
    y[:, canon_start : canon_start + canon_dim] = current_pose[:, 9 : 9 + canon_dim]
    y[:, vel_start : vel_start + vel_dim] = 0.0
    return y


def make_tiers(x_norm: torch.Tensor, schema: dict[str, int]) -> dict[str, torch.Tensor]:
    mask = alteration_mask(schema, x_norm.device).unsqueeze(0)
    tier2 = x_norm + 0.05 * torch.randn_like(x_norm) * mask
    noisy_bad = x_norm + 0.75 * torch.randn_like(x_norm) * mask
    statue = make_statue_tier(x_norm, schema)
    tier3 = torch.where((torch.arange(x_norm.shape[0], device=x_norm.device)[:, None] % 2) == 0, statue, noisy_bad)
    tier4 = torch.randn_like(x_norm)
    tier4[:, schema["input_root_start"] : schema["input_root_end"]] = x_norm[
        :, schema["input_root_start"] : schema["input_root_end"]
    ]
    return {
        "tier1_clean": x_norm,
        "tier2_slight": tier2,
        "tier3_bad": tier3,
        "tier4_noise": tier4,
    }


@torch.no_grad()
def reconstruction_errors(model: nn.Module, x_norm: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
    values = []
    model.eval()
    for start in range(0, x_norm.shape[0], batch_size):
        batch = x_norm[start : start + batch_size]
        recon = model(batch)
        values.append(F.mse_loss(recon, batch, reduction="none").mean(dim=-1))
    return torch.cat(values, dim=0)


@torch.no_grad()
def tier_report(model: nn.Module, x_norm: torch.Tensor, schema: dict[str, int]) -> dict[str, float]:
    tiers = make_tiers(x_norm, schema)
    report: dict[str, float] = {}
    for name, values in tiers.items():
        err = reconstruction_errors(model, values)
        report[f"{name}_mean"] = float(err.mean().cpu())
        report[f"{name}_p95"] = float(torch.quantile(err, 0.95).cpu())
    return report


def save_tier_report(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train(args: argparse.Namespace) -> None:
    cfg = AEConfig()
    for field in cfg.__dataclass_fields__:
        value = getattr(args, field, None)
        if value is not None:
            setattr(cfg, field, value)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    locomotion_cfg = tl.TrainConfig()
    locomotion_cfg.cyclic_animation = cfg.cyclic_animation
    folder = tl.resolve_path(cfg.folder_path)
    clips = tl.load_clips(folder, locomotion_cfg)
    clean = collect_clean_features(clips, locomotion_cfg, device)
    mean = clean.mean(dim=0)
    std = clean.std(dim=0).clamp_min(cfg.std_floor)
    x_norm = normalise(clean, mean, std).to(device)
    mean = mean.to(device)
    std = std.to(device)
    schema = transition_schema(clips[0], locomotion_cfg)

    run_dir = tl.resolve_path(cfg.output_dir) / cfg.run_name
    ckpt_dir = run_dir / "checkpoints"
    writer = SummaryWriter(run_dir / "tb")
    model = TransitionAutoencoder(schema["total_dim"], cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    print(
        f"transition_ae samples={x_norm.shape[0]} dim={x_norm.shape[1]} latent={cfg.latent_dim} "
        f"device={device} folder={folder}",
        flush=True,
    )
    best = math.inf
    baseline = None
    target = None
    stalls = 0
    rows: list[dict[str, float]] = []
    indices = torch.arange(x_norm.shape[0], device=device)
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        perm = indices[torch.randperm(indices.numel(), device=device)]
        losses = []
        for start in range(0, perm.numel(), cfg.batch_size):
            batch = x_norm.index_select(0, perm[start : start + cfg.batch_size])
            noisy = batch
            if cfg.input_noise_std > 0.0:
                noisy = batch + cfg.input_noise_std * torch.randn_like(batch)
            recon = model(noisy)
            loss = F.huber_loss(recon, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(losses))
        if baseline is None:
            baseline = train_loss
            target = baseline * (1.0 - cfg.target_loss_reduction)
        improved = train_loss < best - cfg.min_delta
        stalls = 0 if improved else stalls + 1
        if train_loss < best:
            best = train_loss
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(cfg),
                    "locomotion_config": asdict(locomotion_cfg),
                    "schema": schema,
                    "mean": mean.detach().cpu(),
                    "std": std.detach().cpu(),
                    "metadata": {
                        "npz_folder": str(folder),
                        "body_names": clips[0].body_names,
                        "parents_body": clips[0].parents_body.tolist(),
                    },
                    "epoch": epoch,
                    "best": best,
                },
                ckpt_dir / "checkpoint_best.pt",
            )
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/best", best, epoch)
        if epoch == 1 or epoch % cfg.tier_eval_every_epochs == 0:
            report = tier_report(model, x_norm, schema)
            row = {"epoch": epoch, "train_loss": train_loss, "best": best, **report}
            rows.append(row)
            save_tier_report(run_dir / "tier_report.csv", rows)
            for key, value in report.items():
                writer.add_scalar(f"tiers/{key}", value, epoch)
            print(
                "epoch={:04d} loss={:.6g} best={:.6g} tiers clean={:.6g} slight={:.6g} bad={:.6g} noise={:.6g}".format(
                    epoch,
                    train_loss,
                    best,
                    report["tier1_clean_mean"],
                    report["tier2_slight_mean"],
                    report["tier3_bad_mean"],
                    report["tier4_noise_mean"],
                ),
                flush=True,
            )
        elif epoch % 10 == 0:
            print(f"epoch={epoch:04d} loss={train_loss:.6g} best={best:.6g} stalls={stalls}", flush=True)
        if target is not None and train_loss <= target:
            print(f"target reached epoch={epoch} loss={train_loss:.6g} target={target:.6g}", flush=True)
            break
        if cfg.stall_patience_epochs > 0 and stalls >= cfg.stall_patience_epochs:
            print(f"stopped on stall epoch={epoch} best={best:.6g}", flush=True)
            break
    final_report = tier_report(model, x_norm, schema)
    rows.append({"epoch": epoch, "train_loss": train_loss, "best": best, **final_report})
    save_tier_report(run_dir / "tier_report.csv", rows)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    for name, field in AEConfig.__dataclass_fields__.items():
        default = field.default
        arg = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(arg, action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument(arg, type=type(default), default=None)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
