from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

try:
    from .bootstrap import PROJECT_ROOT, ensure_paths
    from .naming import checkpoint_path, ik_run_id
    from . import ik_core as tl
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    from naming import checkpoint_path, ik_run_id
    import ik_core as tl

ensure_paths()


DEFAULT_WALK_F = PROJECT_ROOT / "ue5" / "animations_omni_only_full" / "npz_final" / "M_Neutral_Walk_Loop_F.npz"
RUNS_DIR = PROJECT_ROOT / "training" / "runs"

LATENT_DIM = 32
HIDDEN_DIM = 512
NUM_HIDDEN_LAYERS = 2
BATCH_SIZE = 512
TRAIN_STEPS = 12000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
STD_FLOOR = 1e-4
VAL_FRACTION = 0.1
SEED = 1234
LOG_EVERY = 250


@dataclass
class SimpleAEConfig:
    latent_dim: int = LATENT_DIM
    hidden_dim: int = HIDDEN_DIM
    num_hidden_layers: int = NUM_HIDDEN_LAYERS
    batch_size: int = BATCH_SIZE
    train_steps: int = TRAIN_STEPS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    std_floor: float = STD_FLOOR
    val_fraction: float = VAL_FRACTION
    seed: int = SEED
    pose_representation: str = tl.IK_POSE_REPRESENTATION
    feature: str = "agent_input_plus_target_output"


class SimpleAutoencoder(nn.Module):
    def __init__(self, dim: int, cfg: SimpleAEConfig):
        super().__init__()
        encoder: list[nn.Module] = []
        in_dim = int(dim)
        for _ in range(int(cfg.num_hidden_layers)):
            encoder.extend((nn.Linear(in_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.GELU()))
            in_dim = int(cfg.hidden_dim)
        encoder.extend((nn.Linear(in_dim, cfg.latent_dim), nn.GELU()))

        decoder: list[nn.Module] = []
        in_dim = int(cfg.latent_dim)
        for _ in range(int(cfg.num_hidden_layers)):
            decoder.extend((nn.Linear(in_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.GELU()))
            in_dim = int(cfg.hidden_dim)
        decoder.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*(encoder + decoder))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_locomotion_cfg(device: torch.device) -> tl.TrainConfig:
    cfg = tl.TrainConfig()
    cfg.pose_representation = tl.IK_POSE_REPRESENTATION
    cfg.cyclic_animation = True
    cfg.predict_residual = True
    cfg.zero_init_output = True
    cfg.live_viewer = False
    cfg.visual_reporter = False
    cfg.update_comparison_on_exit = False
    cfg.use_torch_compile = False
    cfg.device = str(device)
    return cfg


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def npz_paths_from_text(path_text: str) -> list[Path]:
    paths: list[Path] = []
    for raw_part in str(path_text or "").split(";"):
        part = raw_part.strip()
        if not part:
            continue
        path = resolve_path(part)
        if path.is_dir():
            found = sorted(path.glob("*.npz"))
            if not found:
                raise FileNotFoundError(f"No .npz files found in {path}")
            paths.extend(found)
        else:
            if not path.exists():
                raise FileNotFoundError(f"NPZ path does not exist: {path}")
            if path.suffix.lower() != ".npz":
                raise ValueError(f"Expected .npz file, got: {path}")
            paths.append(path)
    return paths


def resolve_clip_specs(
    npz_text: str | None,
    periodic_text: str | None,
    nonperiodic_text: str | None,
) -> list[tuple[Path, bool]]:
    specs: list[tuple[Path, bool]] = []
    for path in npz_paths_from_text(periodic_text or ""):
        specs.append((path, True))
    for path in npz_paths_from_text(nonperiodic_text or ""):
        specs.append((path, False))
    if specs:
        return specs
    if npz_text:
        return [(path, True) for path in npz_paths_from_text(npz_text)]
    if not DEFAULT_WALK_F.exists():
        raise FileNotFoundError(f"Default walk-forward NPZ not found: {DEFAULT_WALK_F}")
    return [(DEFAULT_WALK_F.resolve(), True)]


def load_clips(specs: list[tuple[Path, bool]], cfg: tl.TrainConfig) -> list[tl.MotionClip]:
    clips = [tl.MotionClip(path, cfg, cyclic_animation=cyclic) for path, cyclic in specs]
    first_names = clips[0].body_names
    first_parents = clips[0].parents_body_list
    for clip in clips[1:]:
        if clip.body_names != first_names or clip.parents_body_list != first_parents:
            raise ValueError(f"Skeleton mismatch: {clip.path} vs {clips[0].path}")
    return clips


def valid_current_indices(clip: tl.MotionClip, cfg: tl.TrainConfig, device: torch.device) -> torch.Tensor:
    if clip.cyclic_animation:
        max_cur = int(clip.cyclic_period) - 1
    else:
        max_cur = int(clip.T) - int(cfg.future_window) - 1
    if max_cur < 1:
        return torch.empty((0,), dtype=torch.long, device=device)
    return torch.arange(1, max_cur + 1, dtype=torch.long, device=device)


def feature_schema(clip: tl.MotionClip, cfg: tl.TrainConfig) -> dict[str, object]:
    input_dim, output_dim = tl.make_batch_dims(clip, cfg)
    pose_dim = int(tl.pose_target_output(tl.get_pose_from_clip(clip, torch.tensor([1]), torch.device("cpu"))).shape[-1])
    velocity_dim = input_dim - pose_dim * 2 - 3 - int(cfg.future_window) * 4
    input_root_start = pose_dim * 2 + velocity_dim
    input_root_end = input_dim
    return {
        "feature": "agent_input_plus_target_output",
        "total_dim": int(input_dim + output_dim),
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
        "pose_dim": int(pose_dim),
        "velocity_dim": int(velocity_dim),
        "input_root_start": int(input_root_start),
        "input_root_end": int(input_root_end),
        "target_output_start": int(input_dim),
        "target_output_end": int(input_dim + output_dim),
        "body_names": list(clip.body_names),
        "pose_representation": clip.pose_representation,
    }


@torch.no_grad()
def collect_agent_features(
    clips: list[tl.MotionClip],
    locomotion_cfg: tl.TrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    chunks: list[torch.Tensor] = []
    clip_chunks: list[torch.Tensor] = []
    idx_chunks: list[torch.Tensor] = []
    schema = feature_schema(clips[0], locomotion_cfg)
    for clip_id, clip in enumerate(clips):
        cur_idx = valid_current_indices(clip, locomotion_cfg, device)
        if cur_idx.numel() == 0:
            continue
        prev_idx = cur_idx - 1
        target_idx = cur_idx + 1
        prev_pose = tl.get_pose_from_clip(clip, prev_idx, device)
        cur_pose = tl.get_pose_from_clip(clip, cur_idx, device)
        target_pose = tl.get_pose_from_clip(clip, target_idx, device)
        agent_input = tl.build_input(clip, prev_idx, cur_idx, prev_pose, cur_pose, locomotion_cfg, device)
        agent_target = tl.pose_target_output(target_pose)
        chunks.append(torch.cat((agent_input, agent_target), dim=-1).detach().cpu())
        clip_chunks.append(torch.full((cur_idx.numel(),), clip_id, dtype=torch.long))
        idx_chunks.append(cur_idx.detach().cpu())
    if not chunks:
        raise ValueError("No valid AE feature rows found.")
    return torch.cat(chunks), torch.cat(clip_chunks), torch.cat(idx_chunks), schema


def split_rows(row_count: int, val_fraction: float, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(int(row_count), generator=generator)
    val_count = max(1, int(round(row_count * float(val_fraction)))) if row_count > 1 else 0
    val = order[:val_count].to(device)
    train = order[val_count:].to(device)
    if train.numel() == 0:
        train = val
    return train, val


def normalize_features(features: torch.Tensor, std_floor: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(float(std_floor))
    return (features - mean) / std, mean, std


def batch_indices(rows: torch.Tensor, batch_size: int) -> torch.Tensor:
    if rows.numel() <= int(batch_size):
        return rows.index_select(0, torch.randperm(rows.numel(), device=rows.device))
    choice = torch.randint(0, rows.numel(), (int(batch_size),), device=rows.device)
    return rows.index_select(0, choice)


def lr_for_step(step: int, total_steps: int, base_lr: float) -> float:
    progress = float(max(0, step - 1)) / float(max(1, total_steps - 1))
    if progress >= 0.9:
        return float(base_lr) * 0.1
    if progress >= 0.7:
        return float(base_lr) * 0.3
    return float(base_lr)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def row_mse(model: nn.Module, x: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for start in range(0, x.shape[0], int(batch_size)):
        part = x[start : start + int(batch_size)]
        rows.append((model(part) - part).square().mean(dim=-1))
    return torch.cat(rows, dim=0)


def alteration_mask(schema: dict[str, object], device: torch.device) -> torch.Tensor:
    mask = torch.ones((int(schema["total_dim"]),), dtype=torch.float32, device=device)
    mask[int(schema["input_root_start"]) : int(schema["input_root_end"])] = 0.0
    return mask


def make_statue_tier(x: torch.Tensor, schema: dict[str, object]) -> torch.Tensor:
    y = x.clone()
    pose_dim = int(schema["pose_dim"])
    output_dim = int(schema["output_dim"])
    output_start = int(schema["target_output_start"])
    y[:, output_start : output_start + output_dim] = x[:, :pose_dim]
    return y


def make_bad_tiers(x: torch.Tensor, schema: dict[str, object]) -> dict[str, torch.Tensor]:
    mask = alteration_mask(schema, x.device).reshape(1, -1)
    perm = torch.randperm(x.shape[0], device=x.device)
    shuffled = x.clone()
    out_start = int(schema["target_output_start"])
    out_end = int(schema["target_output_end"])
    shuffled[:, out_start:out_end] = x.index_select(0, perm)[:, out_start:out_end]
    noise = torch.randn_like(x)
    noise[:, int(schema["input_root_start"]) : int(schema["input_root_end"])] = x[
        :, int(schema["input_root_start"]) : int(schema["input_root_end"])
    ]
    return {
        "tier1_clean": x,
        "tier2_slight": x + 0.05 * torch.randn_like(x) * mask,
        "tier3_bad_statue": make_statue_tier(x, schema),
        "tier3_bad_shuffle_output": shuffled,
        "tier4_noise": noise,
    }


@torch.no_grad()
def diagnostic_report(
    model: nn.Module,
    x: torch.Tensor,
    schema: dict[str, object],
) -> dict[str, float]:
    model.eval()
    tiers = make_bad_tiers(x, schema)
    means: dict[str, float] = {}
    for name, tier in tiers.items():
        values = row_mse(model, tier)
        means[f"{name}_mean"] = float(values.mean().detach().cpu())
        means[f"{name}_p95"] = float(torch.quantile(values, 0.95).detach().cpu())
    clean = max(means["tier1_clean_mean"], 1e-12)
    for key in list(means):
        if key.endswith("_mean") and key != "tier1_clean_mean":
            means[f"{key[:-5]}_over_clean"] = means[key] / clean
    model.train()
    return means


def save_diagnostic_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[-1].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_payload(
    model: SimpleAutoencoder,
    optimizer: torch.optim.Optimizer,
    cfg: SimpleAEConfig,
    locomotion_cfg: tl.TrainConfig,
    schema: dict[str, object],
    mean: torch.Tensor,
    std: torch.Tensor,
    step: int,
    best: float,
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "kind": "simple_agent_io_autoencoder",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(cfg),
        "locomotion_config": asdict(locomotion_cfg),
        "schema": schema,
        "mean": mean.detach().cpu(),
        "std": std.detach().cpu(),
        "step": int(step),
        "best": float(best),
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple IK agent-input plus output autoencoder.")
    parser.add_argument("--npz", default=None, help="NPZ file/folder or semicolon-separated NPZ list.")
    parser.add_argument("--periodic-folder", default=None, help="Periodic NPZ folder/list.")
    parser.add_argument("--nonperiodic-folder", default=None, help="Nonperiodic NPZ folder/list.")
    parser.add_argument("--run-label", default="simple_ae")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(SEED)

    cfg = SimpleAEConfig()
    locomotion_cfg = make_locomotion_cfg(device)
    specs = resolve_clip_specs(args.npz, args.periodic_folder, args.nonperiodic_folder)
    clips = load_clips(specs, locomotion_cfg)
    raw_features, clip_ids, cur_indices, schema = collect_agent_features(clips, locomotion_cfg, device)
    x_cpu, mean_cpu, std_cpu = normalize_features(raw_features, cfg.std_floor)
    x = x_cpu.to(device)
    mean = mean_cpu.to(device)
    std = std_cpu.to(device)
    train_rows, val_rows = split_rows(x.shape[0], cfg.val_fraction, cfg.seed, device)

    model = SimpleAutoencoder(x.shape[1], cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    run_id = ik_run_id(args.run_label)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"), flush_secs=1)
    metadata = {
        "npz_paths": [str(path) for path, _cyclic in specs],
        "npz_folders": [{"path": str(path.parent), "cyclic": bool(cyclic)} for path, cyclic in specs],
        "row_count": int(x.shape[0]),
        "train_rows": int(train_rows.numel()),
        "val_rows": int(val_rows.numel()),
        "clip_count": int(len(clips)),
        "tensorboard_logdir": str(run_dir / "tb"),
    }
    config_payload = {
        "config": asdict(cfg),
        "locomotion_config": asdict(locomotion_cfg),
        "schema": schema,
        "metadata": metadata,
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    writer.add_text("config/json", f"```json\n{json.dumps(config_payload, indent=2)}\n```", 0)

    print(
        f"simple_ae run={run_id} rows={x.shape[0]} dim={x.shape[1]} "
        f"train={train_rows.numel()} val={val_rows.numel()} tensorboard_logdir={run_dir / 'tb'}",
        flush=True,
    )

    best = float("inf")
    diagnostic_rows: list[dict[str, float | int]] = []
    start_time = time.perf_counter()
    for step in range(1, cfg.train_steps + 1):
        lr = lr_for_step(step, cfg.train_steps, cfg.learning_rate)
        set_lr(optimizer, lr)
        rows = batch_indices(train_rows, cfg.batch_size)
        batch = x.index_select(0, rows)
        recon = model(batch)
        loss = F.mse_loss(recon, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % LOG_EVERY == 0 or step == cfg.train_steps:
            with torch.no_grad():
                model.eval()
                train_score = float(row_mse(model, x.index_select(0, train_rows)).mean().detach().cpu())
                val_score = (
                    float(row_mse(model, x.index_select(0, val_rows)).mean().detach().cpu())
                    if val_rows.numel() > 0
                    else train_score
                )
                report = diagnostic_report(model, x.index_select(0, val_rows if val_rows.numel() > 0 else train_rows), schema)
                model.train()
            if val_score < best:
                best = val_score
                path = checkpoint_path(run_dir, run_id, "best")
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    checkpoint_payload(model, optimizer, cfg, locomotion_cfg, schema, mean, std, step, best, metadata),
                    path,
                )
            elapsed = time.perf_counter() - start_time
            writer.add_scalar("loss/train_recon", train_score, step)
            writer.add_scalar("loss/val_recon", val_score, step)
            writer.add_scalar("loss/best", best, step)
            writer.add_scalar("time/elapsed_s", elapsed, step)
            for key, value in report.items():
                writer.add_scalar(f"synthetic/{key}", value, step)
            row = {"step": step, "train_recon": train_score, "val_recon": val_score, "best": best, **report}
            diagnostic_rows.append(row)
            save_diagnostic_csv(run_dir / "synthetic_diagnostics.csv", diagnostic_rows)
            print(
                f"step={step:05d} train={train_score:.6g} val={val_score:.6g} best={best:.6g} "
                f"bad_statue_x={report['tier3_bad_statue_over_clean']:.2f} "
                f"bad_shuffle_x={report['tier3_bad_shuffle_output_over_clean']:.2f} "
                f"noise_x={report['tier4_noise_over_clean']:.2f} lr={lr:.3g} elapsed_s={elapsed:.1f}",
                flush=True,
            )

    last = checkpoint_path(run_dir, run_id, "last")
    torch.save(checkpoint_payload(model, optimizer, cfg, locomotion_cfg, schema, mean, std, cfg.train_steps, best, metadata), last)
    writer.close()
    print(f"saved {last}", flush=True)


if __name__ == "__main__":
    main()
