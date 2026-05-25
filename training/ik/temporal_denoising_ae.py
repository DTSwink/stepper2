from __future__ import annotations

import argparse
import json
import math
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
    from . import excess_envelope as env
    from . import ik_core as tl
    from . import train_full_ae_envelope as full_train
    from . import train_simple_ae_controller as ctl
    from . import train_simple_autoencoder as simple_ae
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    from naming import checkpoint_path, ik_run_id
    import excess_envelope as env
    import ik_core as tl
    import train_full_ae_envelope as full_train
    import train_simple_ae_controller as ctl
    import train_simple_autoencoder as simple_ae

ensure_paths()


RUNS_DIR = PROJECT_ROOT / "training" / "runs"
PERIODIC_DIR = PROJECT_ROOT / "ue5" / "animations_omni_only_full" / "npz_final"
NONPERIODIC_DIR = PROJECT_ROOT / "ue5" / "animations_transitions_only_full_trimmed" / "npz_final"
FULL_SIMPLE_AE = (
    RUNS_DIR
    / "20260522_073652_ik_full_vanilla_ae_all"
    / "checkpoints"
    / "20260522_073652_ik_full_vanilla_ae_all_best.pt"
)
BASELINE_114624 = (
    RUNS_DIR
    / "20260523_114624_ik_full_k64_no_noise_from_045443_clear_gpu"
    / "checkpoints"
    / "20260523_114624_ik_full_k64_no_noise_from_045443_clear_gpu_latest.pt"
)

MINI_PERIODIC = (
    PERIODIC_DIR / "M_Neutral_Stand_Idle_Loop.npz",
    PERIODIC_DIR / "M_Neutral_Walk_Loop_F.npz",
)
MINI_NONPERIODIC = (
    NONPERIODIC_DIR / "M_Neutral_Stand_Turn_045_L.npz",
    NONPERIODIC_DIR / "M_Neutral_Stand_Turn_045_R.npz",
    NONPERIODIC_DIR / "M_Neutral_Walk_Circle_Strafe_L.npz",
    NONPERIODIC_DIR / "M_Neutral_Walk_Circle_Strafe_R.npz",
    NONPERIODIC_DIR / "M_Neutral_Walk_Diamond_BL_F_Lfoot.npz",
)
DIAMOND_NAME = "M_Neutral_Walk_Diamond_BL_F_Lfoot.npz"

WINDOW_FRAMES = 8
AE_BATCH = 1024
AE_STEPS = 5000
AE_LR = 1e-4
AE_LATENT = 160
AE_HIDDEN = 512
AE_LAYERS = 2
NOISE_AMOUNT = 0.25
DAMP_PROB = 0.35
LOG_EVERY = 100
SNAPSHOT_EVERY = 500
CONTROLLER_BATCH = 512
CONTROLLER_STEPS = 1800
CONTROLLER_LR = 2e-5
CONTROLLER_K = 32
CONTROLLER_LOG_EVERY = 50
CONTROLLER_SNAPSHOT_EVERY = 250
LOSS_SCALE = 500.0
AMP_BATCH = 256
AMP_ROLLOUT_K = 64
AMP_STEPS = 2000
AMP_CONTROLLER_LR = 1.0e-5
AMP_CRITIC_LR = 1.0e-4
AMP_CRITIC_WARMUP_STEPS = 50
AMP_LOG_EVERY = 25
AMP_EVAL_EVERY = 100
AMP_SNAPSHOT_EVERY = 250
AMP_FOOL_WEIGHT = 1.0
AMP_SUPERVISED_TARGET_LOSS = 0.5
AMP_SUPERVISED_ESTIMATE_BATCHES = 8


@dataclass
class TemporalAEConfig:
    variant: str
    frames: int = WINDOW_FRAMES
    latent_dim: int = AE_LATENT
    hidden_dim: int = AE_HIDDEN
    num_hidden_layers: int = AE_LAYERS
    batch_size: int = AE_BATCH
    train_steps: int = AE_STEPS
    learning_rate: float = AE_LR
    noise_amount: float = NOISE_AMOUNT
    damp_prob: float = DAMP_PROB
    seed: int = 1234


class TemporalDenoisingAE(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, cfg: TemporalAEConfig):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(int(cfg.num_hidden_layers)):
            layers.extend((nn.Linear(in_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.GELU()))
            in_dim = int(cfg.hidden_dim)
        layers.extend((nn.Linear(in_dim, cfg.latent_dim), nn.GELU()))
        in_dim = int(cfg.latent_dim)
        for _ in range(int(cfg.num_hidden_layers)):
            layers.extend((nn.Linear(in_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.GELU()))
            in_dim = int(cfg.hidden_dim)
        layers.append(nn.Linear(in_dim, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class AMPConfig:
    frames: int = WINDOW_FRAMES
    batch_size: int = AMP_BATCH
    rollout_k: int = AMP_ROLLOUT_K
    steps: int = AMP_STEPS
    controller_lr: float = AMP_CONTROLLER_LR
    critic_lr: float = AMP_CRITIC_LR
    critic_warmup_steps: int = AMP_CRITIC_WARMUP_STEPS
    amp_fool_weight: float = AMP_FOOL_WEIGHT
    hidden_dim: int = 512
    num_hidden_layers: int = 3
    log_every: int = AMP_LOG_EVERY
    eval_every: int = AMP_EVAL_EVERY
    snapshot_every: int = AMP_SNAPSHOT_EVERY
    supervised_target_loss: float = AMP_SUPERVISED_TARGET_LOSS
    supervised_estimate_batches: int = AMP_SUPERVISED_ESTIMATE_BATCHES
    seed: int = 1234


class TemporalAMPCritic(nn.Module):
    def __init__(self, input_dim: int, cfg: AMPConfig):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(int(cfg.num_hidden_layers)):
            layers.extend((nn.Linear(in_dim, int(cfg.hidden_dim)), nn.LayerNorm(int(cfg.hidden_dim)), nn.GELU()))
            in_dim = int(cfg.hidden_dim)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def device_and_seed(seed: int = 1234) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(int(seed))
    return device


def specs_from_lists(periodic: tuple[Path, ...], nonperiodic: tuple[Path, ...]) -> list[tuple[Path, bool]]:
    specs = [(path.resolve(), True) for path in periodic] + [(path.resolve(), False) for path in nonperiodic]
    for path, _cyclic in specs:
        if not path.exists():
            raise FileNotFoundError(path)
    return specs


def full_specs() -> list[tuple[Path, bool]]:
    return ctl.resolve_clip_specs(None, str(PERIODIC_DIR), str(NONPERIODIC_DIR))


def mini_specs() -> list[tuple[Path, bool]]:
    return specs_from_lists(MINI_PERIODIC, MINI_NONPERIODIC)


def load_store(specs: list[tuple[Path, bool]], device: torch.device) -> tuple[tl.TrainConfig, ctl.SimpleClipStore]:
    ae_ckpt = torch.load(FULL_SIMPLE_AE, map_location="cpu", weights_only=False)
    cfg = ctl.make_cfg(device, ae_ckpt)
    clips = ctl.load_clips(specs, cfg)
    return cfg, ctl.SimpleClipStore(clips, cfg, device)


def output_dim_from_store(store: ctl.SimpleClipStore) -> int:
    _input_dim, output_dim = tl.make_batch_dims(store.prototype, store.cfg)
    return int(output_dim)


def root_dim_from_store(store: ctl.SimpleClipStore) -> int:
    return int(store.input_root_features.shape[-1])


def temporal_schema(store: ctl.SimpleClipStore, cfg: TemporalAEConfig) -> dict[str, object]:
    if cfg.variant == "root8_body8":
        input_body_frames = int(cfg.frames)
        target_body_frames = int(cfg.frames)
    elif cfg.variant == "root8_body1":
        input_body_frames = 1
        target_body_frames = 1
    elif cfg.variant == "root8_body1_to_body8":
        input_body_frames = 1
        target_body_frames = int(cfg.frames)
    else:
        raise ValueError(f"Unknown temporal AE variant: {cfg.variant}")
    root_dim = root_dim_from_store(store)
    body_dim = output_dim_from_store(store)
    return {
        "kind": "temporal_denoising_controller_ae",
        "variant": cfg.variant,
        "frames": int(cfg.frames),
        "root_dim": int(root_dim),
        "body_dim": int(body_dim),
        "body_frames": int(input_body_frames),
        "input_body_frames": int(input_body_frames),
        "target_body_frames": int(target_body_frames),
        "input_body_offset": 0 if cfg.variant == "root8_body1_to_body8" else 1,
        "target_body_offset": 1,
        "input_dim": int(cfg.frames) * int(root_dim) + int(input_body_frames) * int(body_dim),
        "target_dim": int(target_body_frames) * int(body_dim),
        "pose_representation": tl.IK_POSE_REPRESENTATION,
        "output_reference_root": tl.OUTPUT_REFERENCE_ROOT,
        "output_prediction_mode": tl.normalized_output_prediction_mode(),
        "state_reference_root": tl.STATE_REFERENCE_ROOT,
        "body_names": list(store.prototype.body_names),
    }


def build_temporal_rows(
    store: ctl.SimpleClipStore,
    cfg: TemporalAEConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    frames = int(cfg.frames)
    schema = temporal_schema(store, cfg)
    root_chunks: list[torch.Tensor] = []
    input_body_chunks: list[torch.Tensor] = []
    target_body_chunks: list[torch.Tensor] = []
    clip_chunks: list[torch.Tensor] = []
    idx_chunks: list[torch.Tensor] = []
    offsets = torch.arange(frames, dtype=torch.long, device=store.device)
    for clip_id, clip in enumerate(store.clips):
        if clip.cyclic_animation:
            max_start = ctl.max_training_start_for_clip(clip, store.cfg)
        else:
            max_start = ctl.max_training_start_for_clip(clip, store.cfg) - (frames - 1)
        if max_start < 1:
            continue
        starts = torch.arange(1, max_start + 1, dtype=torch.long, device=store.device)
        clip_ids = torch.full((starts.numel(),), int(clip_id), dtype=torch.long, device=store.device)
        row_idx = starts[:, None] + offsets[None, :]
        flat_clip = clip_ids[:, None].expand_as(row_idx).reshape(-1)
        flat_idx = row_idx.reshape(-1)
        roots = store.get_input_root_features(flat_clip, flat_idx).reshape(starts.numel(), -1)
        bodies8 = ctl.transition_target_output(store, flat_clip, flat_idx).reshape(starts.numel(), frames, -1)
        if cfg.variant == "root8_body8":
            input_body = bodies8.reshape(starts.numel(), -1)
            target_body = input_body
        elif cfg.variant == "root8_body1":
            input_body = bodies8[:, 0, :]
            target_body = input_body
        elif cfg.variant == "root8_body1_to_body8":
            input_body = store.get_target_output(clip_ids, starts)
            target_body = bodies8.reshape(starts.numel(), -1)
        else:
            raise ValueError(f"Unknown temporal AE variant: {cfg.variant}")
        root_chunks.append(roots.detach())
        input_body_chunks.append(input_body.detach())
        target_body_chunks.append(target_body.detach())
        clip_chunks.append(clip_ids.detach())
        idx_chunks.append(starts.detach())
    if not root_chunks:
        raise ValueError("No valid temporal AE rows.")
    roots = torch.cat(root_chunks, dim=0)
    input_bodies = torch.cat(input_body_chunks, dim=0)
    target_bodies = torch.cat(target_body_chunks, dim=0)
    return roots, input_bodies, target_bodies, torch.cat(clip_chunks), torch.cat(idx_chunks), schema


def normalize_pair(
    roots: torch.Tensor,
    input_bodies: torch.Tensor,
    target_bodies: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    clean_input = torch.cat((roots, input_bodies), dim=-1)
    input_mean = clean_input.mean(dim=0)
    input_std = clean_input.std(dim=0, unbiased=False).clamp_min(1e-4)
    target_mean = target_bodies.mean(dim=0)
    target_std = target_bodies.std(dim=0, unbiased=False).clamp_min(1e-4)
    return input_mean, input_std, target_mean, target_std


def corrupt_body_batch(store: ctl.SimpleClipStore, body: torch.Tensor, cfg: TemporalAEConfig) -> torch.Tensor:
    body_dim = output_dim_from_store(store)
    frames = int(body.shape[-1] // body_dim)
    view = body.reshape(body.shape[0], frames, body_dim)
    noisy = view
    if frames > 1 and float(cfg.damp_prob) > 0.0:
        apply = torch.rand((body.shape[0], 1, 1), device=body.device) < float(cfg.damp_prob)
        scale = torch.rand((body.shape[0], 1, 1), device=body.device) * 0.45
        first = view[:, :1, :]
        damped = first + (view - first) * scale
        noisy = torch.where(apply, damped, noisy)
    flat = noisy.reshape(-1, body_dim)
    flat = ctl.add_pose_noise_to_vector(store, flat, float(cfg.noise_amount))
    return flat.reshape(body.shape[0], frames * body_dim)


def train_temporal_ae_variant(variant: str, run_label: str, steps: int) -> Path:
    device = device_and_seed()
    cfg = TemporalAEConfig(variant=variant, train_steps=int(steps))
    _locomotion_cfg, store = load_store(full_specs(), device)
    roots, input_bodies, target_bodies, clip_ids, cur_idx, schema = build_temporal_rows(store, cfg)
    input_mean, input_std, target_mean, target_std = normalize_pair(roots, input_bodies, target_bodies)
    row_count = int(roots.shape[0])
    generator = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
    order = torch.randperm(row_count, generator=generator, device="cpu").to(device)
    val_count = max(1, int(round(row_count * 0.05)))
    val_rows = order[:val_count]
    train_rows = order[val_count:]

    model = TemporalDenoisingAE(int(schema["input_dim"]), int(schema["target_dim"]), cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=0.0, fused=(device.type == "cuda"))
    run_id = ik_run_id(run_label)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"), flush_secs=1)
    metadata = {
        "npz_paths": [str(path) for path, _cyclic in full_specs()],
        "clip_count": len(store.clips),
        "row_count": row_count,
        "train_rows": int(train_rows.numel()),
        "val_rows": int(val_rows.numel()),
        "tensorboard_logdir": str(run_dir / "tb"),
        "corruption": {
            "pose_noise_amount": float(cfg.noise_amount),
            "damp_prob": float(cfg.damp_prob) if int(schema["input_body_frames"]) > 1 else 0.0,
        },
    }
    payload = {"config": asdict(cfg), "schema": schema, "metadata": metadata}
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    writer.add_text("config/json", f"```json\n{json.dumps(payload, indent=2)}\n```", 0)
    writer.add_scalar("run/started", 1.0, 0)
    writer.flush()
    ctl.refresh_tensorboard_async()
    print(f"temporal_ae variant={variant} run={run_id} rows={row_count} dim={schema['input_dim']} target={schema['target_dim']}", flush=True)

    best = math.inf
    best_path = ckpt_dir / f"{run_id}_best.pt"
    latest_path = ckpt_dir / f"{run_id}_latest.pt"

    def save_ae_checkpoint(path: Path, step: int, best_value: float) -> None:
        torch.save(
            {
                "kind": "temporal_denoising_controller_ae",
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": asdict(cfg),
                "schema": schema,
                "input_mean": input_mean.detach().cpu(),
                "input_std": input_std.detach().cpu(),
                "target_mean": target_mean.detach().cpu(),
                "target_std": target_std.detach().cpu(),
                "step": int(step),
                "best": float(best_value),
                "metadata": metadata,
            },
            path,
        )

    save_ae_checkpoint(ckpt_dir / f"{run_id}_init.pt", 0, best)
    start = time.perf_counter()
    for step in range(1, int(cfg.train_steps) + 1):
        rows = train_rows.index_select(0, torch.randint(0, train_rows.numel(), (int(cfg.batch_size),), device=device))
        clean_input_body = input_bodies.index_select(0, rows)
        clean_target_body = target_bodies.index_select(0, rows)
        noisy_input_body = corrupt_body_batch(store, clean_input_body, cfg)
        batch_in = torch.cat((roots.index_select(0, rows), noisy_input_body), dim=-1)
        x = (batch_in - input_mean) / input_std
        y = (clean_target_body - target_mean) / target_std
        pred = model(x)
        loss = F.mse_loss(pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % LOG_EVERY == 0 or step == int(cfg.train_steps):
            with torch.no_grad():
                model.eval()
                val_input_body = input_bodies.index_select(0, val_rows)
                val_target_body = target_bodies.index_select(0, val_rows)
                val_noisy = corrupt_body_batch(store, val_input_body, cfg)
                val_x = (torch.cat((roots.index_select(0, val_rows), val_noisy), dim=-1) - input_mean) / input_std
                val_y = (val_target_body - target_mean) / target_std
                val_loss = float(F.mse_loss(model(val_x), val_y).detach().cpu())
                clean_x = (torch.cat((roots.index_select(0, val_rows), val_input_body), dim=-1) - input_mean) / input_std
                clean_loss = float(F.mse_loss(model(clean_x), val_y).detach().cpu())
                model.train()
            writer.add_scalar("loss/train", float(loss.detach().cpu()), step)
            writer.add_scalar("loss/val_noisy", val_loss, step)
            writer.add_scalar("loss/val_clean", clean_loss, step)
            writer.add_scalar("time/elapsed_s", time.perf_counter() - start, step)
            writer.flush()
            print(f"{variant}: step={step}/{cfg.train_steps} train={float(loss.detach().cpu()):.6g} val_noisy={val_loss:.6g} val_clean={clean_loss:.6g}", flush=True)
            if val_loss < best:
                best = val_loss
                save_ae_checkpoint(best_path, step, best)
            save_ae_checkpoint(latest_path, step, best)
            if step % SNAPSHOT_EVERY == 0 or step == int(cfg.train_steps):
                save_ae_checkpoint(ckpt_dir / f"{run_id}_step{step:06d}.pt", step, best)
    last_path = ckpt_dir / f"{run_id}_last.pt"
    save_ae_checkpoint(last_path, int(cfg.train_steps), best)
    writer.close()
    return best_path if best_path.exists() else last_path


def load_temporal_ae(path: Path, device: torch.device) -> tuple[TemporalDenoisingAE, dict[str, object], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("kind") != "temporal_denoising_controller_ae":
        raise ValueError(f"Not a temporal denoising AE checkpoint: {path}")
    cfg = TemporalAEConfig(**ckpt["config"])
    schema = dict(ckpt["schema"])
    model = TemporalDenoisingAE(int(schema["input_dim"]), int(schema["target_dim"]), cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return (
        model,
        schema,
        ckpt["input_mean"].to(device=device, dtype=torch.float32),
        ckpt["input_std"].to(device=device, dtype=torch.float32).clamp_min(1e-8),
        ckpt["target_mean"].to(device=device, dtype=torch.float32),
        ckpt["target_std"].to(device=device, dtype=torch.float32).clamp_min(1e-8),
        ckpt,
    )


def root_window_with_offsets(
    store: ctl.SimpleClipStore,
    clip_ids: torch.Tensor,
    cur_idx: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    idx = cur_idx[:, None] + offsets[None, :]
    flat_clip = clip_ids[:, None].expand_as(idx).reshape(-1)
    flat_idx = idx.reshape(-1)
    return store.get_input_root_features(flat_clip, flat_idx).reshape(cur_idx.numel(), int(offsets.numel()) * store.input_root_features.shape[-1])


def root_window(store: ctl.SimpleClipStore, clip_ids: torch.Tensor, cur_idx: torch.Tensor, frames: int) -> torch.Tensor:
    offsets = torch.arange(int(frames), dtype=torch.long, device=store.device)
    return root_window_with_offsets(store, clip_ids, cur_idx, offsets)


def initial_body_context(store: ctl.SimpleClipStore, clip_ids: torch.Tensor, cur_idx: torch.Tensor, frames: int) -> torch.Tensor:
    if int(frames) <= 1:
        return torch.empty((clip_ids.numel(), 0, output_dim_from_store(store)), dtype=torch.float32, device=store.device)
    offsets = torch.arange(int(frames) - 1, dtype=torch.long, device=store.device)
    idx = cur_idx[:, None] - (int(frames) - 1) + offsets[None, :]
    flat_clip = clip_ids[:, None].expand_as(idx).reshape(-1)
    flat_idx = idx.reshape(-1)
    return ctl.transition_target_output(store, flat_clip, flat_idx).reshape(cur_idx.numel(), int(frames) - 1, -1)


def temporal_score_rows(
    ae: TemporalDenoisingAE,
    schema: dict[str, object],
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    roots: torch.Tensor,
    input_body: torch.Tensor,
    target_body: torch.Tensor,
) -> torch.Tensor:
    x = (torch.cat((roots, input_body), dim=-1) - input_mean) / input_std
    pred = ae(x)
    y = (target_body - target_mean) / target_std
    return (pred - y).square().mean(dim=-1)


def set_module_requires_grad(module: nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def amp_feature(roots: torch.Tensor, current_body: torch.Tensor, future_body: torch.Tensor) -> torch.Tensor:
    return torch.cat((roots, current_body, future_body), dim=-1)


def build_amp_real_rows(
    store: ctl.SimpleClipStore,
    cfg: AMPConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    frames = int(cfg.frames)
    offsets = torch.arange(frames, dtype=torch.long, device=store.device)
    root_chunks: list[torch.Tensor] = []
    current_chunks: list[torch.Tensor] = []
    future_chunks: list[torch.Tensor] = []
    feature_chunks: list[torch.Tensor] = []
    clip_chunks: list[torch.Tensor] = []
    start_chunks: list[torch.Tensor] = []
    for clip_id, clip in enumerate(store.clips):
        if clip.cyclic_animation:
            max_start = ctl.max_training_start_for_clip(clip, store.cfg)
        else:
            max_start = ctl.max_training_start_for_clip(clip, store.cfg) - (frames - 1)
        if max_start < 1:
            continue
        starts = torch.arange(1, max_start + 1, dtype=torch.long, device=store.device)
        clip_ids = torch.full((starts.numel(),), int(clip_id), dtype=torch.long, device=store.device)
        row_idx = starts[:, None] + offsets[None, :]
        flat_clip = clip_ids[:, None].expand_as(row_idx).reshape(-1)
        flat_idx = row_idx.reshape(-1)
        roots = store.get_input_root_features(flat_clip, flat_idx).reshape(starts.numel(), -1)
        current_body = store.get_target_output(clip_ids, starts)
        future_body = ctl.transition_target_output(store, flat_clip, flat_idx).reshape(starts.numel(), -1)
        feature = amp_feature(roots, current_body, future_body)
        root_chunks.append(roots.detach())
        current_chunks.append(current_body.detach())
        future_chunks.append(future_body.detach())
        feature_chunks.append(feature.detach())
        clip_chunks.append(clip_ids.detach())
        start_chunks.append(starts.detach())
    if not feature_chunks:
        raise ValueError("No valid AMP windows.")
    body_dim = output_dim_from_store(store)
    root_dim = root_dim_from_store(store)
    schema = {
        "kind": "temporal_amp_critic",
        "frames": frames,
        "root_dim": int(root_dim),
        "body_dim": int(body_dim),
        "input_dim": int(frames * root_dim + (frames + 1) * body_dim),
        "feature_layout": "roots[8] + current_body[1] + future_body[8]",
        "pose_representation": tl.IK_POSE_REPRESENTATION,
        "output_reference_root": tl.OUTPUT_REFERENCE_ROOT,
        "output_prediction_mode": tl.normalized_output_prediction_mode(),
        "state_reference_root": tl.STATE_REFERENCE_ROOT,
        "body_names": list(store.prototype.body_names),
    }
    return (
        torch.cat(feature_chunks, dim=0),
        torch.cat(root_chunks, dim=0),
        torch.cat(current_chunks, dim=0),
        torch.cat(future_chunks, dim=0),
        torch.cat(clip_chunks, dim=0),
        torch.cat(start_chunks, dim=0),
        schema,
    )


def normalize_amp_features(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return features.mean(dim=0), features.std(dim=0, unbiased=False).clamp_min(1e-4)


def critic_logits(
    critic: TemporalAMPCritic,
    features: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return critic((features - mean) / std)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def sample_amp_real_features(real_features: torch.Tensor, count: int) -> torch.Tensor:
    rows = torch.randint(0, int(real_features.shape[0]), (max(1, int(count)),), dtype=torch.long, device=real_features.device)
    return real_features.index_select(0, rows)


def collect_amp_fake_features(
    model: torch.nn.Module,
    store: ctl.SimpleClipStore,
    cfg: AMPConfig,
    start_pool: ctl.StartPool,
    simple_ae_model: simple_ae.SimpleAutoencoder | None = None,
    simple_ae_mean: torch.Tensor | None = None,
    simple_ae_std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frames = int(cfg.frames)
    rollout_k = int(cfg.rollout_k)
    batch_size = max(1, int(cfg.batch_size))
    root_offsets = torch.arange(frames, dtype=torch.long, device=store.device)
    effective_k = torch.full((batch_size,), rollout_k, dtype=torch.long, device=store.device)
    rows = torch.randint(0, start_pool.row_count, (batch_size,), dtype=torch.long, device=store.device)
    clip_ids = start_pool.clip_ids.index_select(0, rows)
    starts = start_pool.starts.index_select(0, rows)
    cur_idx = starts
    prev_vec, prev_pelvis, prev_payload = ctl.target_state(store, clip_ids, cur_idx - 1)
    cur_vec, cur_pelvis, cur_payload = ctl.target_state(store, clip_ids, cur_idx)
    pred_history: list[torch.Tensor] = []
    state_history: list[torch.Tensor] = []
    idx_history: list[torch.Tensor] = []
    feature_chunks: list[torch.Tensor] = []
    weight_chunks: list[torch.Tensor] = []
    simple_ae_loss = torch.zeros((), dtype=torch.float32, device=store.device)
    row_weight = (1.0 / effective_k.float()) / float(batch_size)
    predicted_age = torch.zeros_like(effective_k)
    max_start_by_row = temporal_max_training_start_for_clip_ids(store, clip_ids, frames)
    cyclic_by_row = store.cyclic.index_select(0, clip_ids)
    reset_starts_by_step = None
    if rollout_k > 1:
        span = max_start_by_row.clamp_min(1).float().reshape(1, -1)
        reset_starts_by_step = (
            torch.floor(torch.rand((rollout_k - 1, batch_size), dtype=torch.float32, device=store.device) * span).long() + 1
        ).clamp_min(1)
        reset_starts_by_step = torch.minimum(reset_starts_by_step, max_start_by_row.reshape(1, -1))

    for step in range(rollout_k):
        state_history.append(cur_vec)
        inp = ctl.build_controller_input(store, clip_ids, cur_idx, prev_vec, cur_vec, prev_pelvis, cur_pelvis, prev_payload, cur_payload)
        raw = ctl.model_forward(model, inp, cur_vec, store.cfg)
        pred_vec = ctl.clean_output_vector(raw, store)
        active = effective_k > step
        if simple_ae_model is not None and simple_ae_mean is not None and simple_ae_std is not None:
            simple_ae_loss = simple_ae_loss + (
                ctl.ae_score_rows(simple_ae_model, simple_ae_mean, simple_ae_std, inp, pred_vec) * row_weight * active.float()
            ).sum()
        predicted_age = torch.where(active, predicted_age + 1, predicted_age)
        pred_history.append(pred_vec)
        idx_history.append(cur_idx)
        if len(pred_history) >= frames:
            roots = root_window_with_offsets(store, clip_ids, idx_history[-frames], root_offsets)
            current_body = state_history[-frames]
            future_body = torch.cat(pred_history[-frames:], dim=-1)
            feature_chunks.append(amp_feature(roots, current_body, future_body))
            weight_chunks.append((active & (predicted_age >= frames)).float())
        if step + 1 >= rollout_k:
            break
        continuing = effective_k > (step + 1)
        reset = continuing & (~cyclic_by_row) & (cur_idx >= max_start_by_row)
        advance = continuing & (~reset)
        assert reset_starts_by_step is not None
        reset_starts = reset_starts_by_step[step]
        reset_prev_vec, reset_prev_pelvis, reset_prev_payload = ctl.target_state(store, clip_ids, reset_starts - 1)
        reset_cur_vec, reset_cur_pelvis, reset_cur_payload = ctl.target_state(store, clip_ids, reset_starts)
        next_vec, next_pelvis, next_payload = ctl.advance_transition_state(store, clip_ids, cur_idx, pred_vec)
        reset_mask = reset[:, None]
        advance_mask = advance[:, None]
        prev_vec = torch.where(reset_mask, reset_prev_vec, torch.where(advance_mask, cur_vec, prev_vec))
        prev_pelvis = torch.where(reset_mask, reset_prev_pelvis, torch.where(advance_mask, cur_pelvis, prev_pelvis))
        prev_payload = torch.where(reset_mask, reset_prev_payload, torch.where(advance_mask, cur_payload, prev_payload))
        cur_vec = torch.where(reset_mask, reset_cur_vec, torch.where(advance_mask, next_vec, cur_vec))
        cur_pelvis = torch.where(reset_mask, reset_cur_pelvis, torch.where(advance_mask, next_pelvis, cur_pelvis))
        cur_payload = torch.where(reset_mask, reset_cur_payload, torch.where(advance_mask, next_payload, cur_payload))
        cur_idx = torch.where(reset, reset_starts, torch.where(continuing, cur_idx + 1, cur_idx))
        predicted_age = torch.where(reset, torch.zeros_like(predicted_age), predicted_age)
    if not feature_chunks:
        raise ValueError("AMP rollout produced no scoreable windows; rollout_k must be >= frames.")
    return torch.cat(feature_chunks, dim=0), torch.cat(weight_chunks, dim=0), simple_ae_loss


def amp_controller_loss_and_features(
    model: torch.nn.Module,
    critic: TemporalAMPCritic,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    store: ctl.SimpleClipStore,
    cfg: AMPConfig,
    start_pool: ctl.StartPool,
    simple_ae_model: simple_ae.SimpleAutoencoder,
    simple_ae_mean: torch.Tensor,
    simple_ae_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    set_module_requires_grad(critic, False)
    fake_features, weights, simple_ae_loss = collect_amp_fake_features(
        model, store, cfg, start_pool, simple_ae_model, simple_ae_mean, simple_ae_std
    )
    logits = critic_logits(critic, fake_features, feature_mean, feature_std)
    loss = weighted_mean(F.softplus(-logits), weights)
    stats = {
        "fake_logit": weighted_mean(logits.detach(), weights),
        "valid_windows": weights.sum().detach(),
    }
    set_module_requires_grad(critic, True)
    return loss, simple_ae_loss, fake_features, weights, stats


def amp_critic_loss(
    critic: TemporalAMPCritic,
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    fake_weights: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    real_batch = sample_amp_real_features(real_features, int(fake_features.shape[0]))
    real_logits = critic_logits(critic, real_batch, feature_mean, feature_std)
    fake_logits = critic_logits(critic, fake_features.detach(), feature_mean, feature_std)
    real_loss = F.softplus(-real_logits).mean()
    fake_loss = weighted_mean(F.softplus(fake_logits), fake_weights.detach())
    loss = real_loss + fake_loss
    stats = {
        "real_logit": real_logits.detach().mean(),
        "fake_logit": weighted_mean(fake_logits.detach(), fake_weights.detach()),
        "real_loss": real_loss.detach(),
        "fake_loss": fake_loss.detach(),
    }
    return loss, stats


def sample_rows_from_start_pool(pool: ctl.StartPool, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.randint(0, int(pool.row_count), (max(1, int(count)),), dtype=torch.long, device=pool.starts.device)
    return pool.clip_ids.index_select(0, rows), pool.starts.index_select(0, rows)


def supervised_k1_transition_loss(
    model: torch.nn.Module,
    store: ctl.SimpleClipStore,
    batch_size: int,
    start_pool: ctl.StartPool,
) -> torch.Tensor:
    clip_ids, cur_idx = sample_rows_from_start_pool(start_pool, int(batch_size))
    prev_vec, prev_pelvis, prev_payload = ctl.target_state(store, clip_ids, cur_idx - 1)
    cur_vec, cur_pelvis, cur_payload = ctl.target_state(store, clip_ids, cur_idx)
    inp = ctl.build_controller_input(store, clip_ids, cur_idx, prev_vec, cur_vec, prev_pelvis, cur_pelvis, prev_payload, cur_payload)
    raw = ctl.model_forward(model, inp, cur_vec, store.cfg)
    pred_vec = ctl.clean_output_vector(raw, store)
    target = ctl.transition_target_output(store, clip_ids, cur_idx)
    return F.mse_loss(pred_vec, target)


@torch.no_grad()
def estimate_supervised_k1_loss(
    model: torch.nn.Module,
    store: ctl.SimpleClipStore,
    batch_size: int,
    start_pool: ctl.StartPool,
    batches: int,
) -> float:
    losses: list[float] = []
    was_training = model.training
    model.eval()
    for _ in range(max(1, int(batches))):
        loss = supervised_k1_transition_loss(model, store, batch_size, start_pool)
        losses.append(float(loss.detach().cpu()))
    model.train(was_training)
    return sum(losses) / float(max(1, len(losses)))


def temporal_rollout_loss(
    model: torch.nn.Module,
    ae: TemporalDenoisingAE,
    schema: dict[str, object],
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    store: ctl.SimpleClipStore,
    rollout_k: int,
    batch_size: int,
    start_pool: ctl.StartPool,
) -> torch.Tensor:
    frames = int(schema["frames"])
    input_body_frames = int(schema.get("input_body_frames", schema.get("body_frames", 1)))
    target_body_frames = int(schema.get("target_body_frames", schema.get("body_frames", 1)))
    root_offsets = torch.arange(frames, dtype=torch.long, device=store.device)
    effective_k = ctl.sample_effective_rollout_k(batch_size, int(rollout_k), store.device)
    start_pools = {int(k): start_pool for k in ctl.rollout_values_for(int(rollout_k))}
    clip_ids, starts = ctl.sample_rollout_rows(start_pools, effective_k)
    cur_idx = starts
    prev_vec, prev_pelvis, prev_payload = ctl.target_state(store, clip_ids, cur_idx - 1)
    cur_vec, cur_pelvis, cur_payload = ctl.target_state(store, clip_ids, cur_idx)
    if target_body_frames == frames:
        row_weight = (1.0 / (effective_k.float() - float(frames) + 1.0).clamp_min(1.0)) / float(batch_size)
    else:
        row_weight = (1.0 / effective_k.float()) / float(batch_size)
    score_roots: list[torch.Tensor] = []
    score_inputs: list[torch.Tensor] = []
    score_targets: list[torch.Tensor] = []
    score_weights: list[torch.Tensor] = []
    pred_history: list[torch.Tensor] = []
    state_history: list[torch.Tensor] = []
    idx_history: list[torch.Tensor] = []
    predicted_age = torch.zeros_like(effective_k)
    max_start_by_row = temporal_max_training_start_for_clip_ids(store, clip_ids, frames)
    cyclic_by_row = store.cyclic.index_select(0, clip_ids)
    reset_starts_by_step = None
    if int(rollout_k) > 1:
        span = max_start_by_row.clamp_min(1).float().reshape(1, -1)
        reset_starts_by_step = (torch.floor(torch.rand((int(rollout_k) - 1, int(batch_size)), dtype=torch.float32, device=store.device) * span).long() + 1).clamp_min(1)
        reset_starts_by_step = torch.minimum(reset_starts_by_step, max_start_by_row.reshape(1, -1))
    for step in range(int(rollout_k)):
        state_history.append(cur_vec)
        inp = ctl.build_controller_input(store, clip_ids, cur_idx, prev_vec, cur_vec, prev_pelvis, cur_pelvis, prev_payload, cur_payload)
        raw = ctl.model_forward(model, inp, cur_vec, store.cfg)
        pred_vec = ctl.clean_output_vector(raw, store)
        active = effective_k > step
        predicted_age = torch.where(active, predicted_age + 1, predicted_age)
        pred_history.append(pred_vec)
        idx_history.append(cur_idx)
        should_score = False
        if target_body_frames == 1:
            roots = root_window_with_offsets(store, clip_ids, cur_idx, root_offsets)
            input_body = pred_vec
            target_body = pred_vec
            score_active = active
            should_score = True
        else:
            if len(pred_history) >= frames:
                roots = root_window_with_offsets(store, clip_ids, idx_history[-frames], root_offsets)
                target_body = torch.cat(pred_history[-frames:], dim=-1)
                if input_body_frames == 1:
                    input_body = state_history[-frames]
                elif input_body_frames == frames:
                    input_body = target_body
                else:
                    raise ValueError(f"Unsupported input_body_frames={input_body_frames}")
                score_active = active & (predicted_age >= frames)
                should_score = True
        if should_score:
            score_roots.append(roots)
            score_inputs.append(input_body)
            score_targets.append(target_body)
            score_weights.append(row_weight * score_active.float())
        if step + 1 >= int(rollout_k):
            break
        continuing = effective_k > (step + 1)
        reset = continuing & (~cyclic_by_row) & (cur_idx >= max_start_by_row)
        advance = continuing & (~reset)
        assert reset_starts_by_step is not None
        reset_starts = reset_starts_by_step[step]
        reset_prev_vec, reset_prev_pelvis, reset_prev_payload = ctl.target_state(store, clip_ids, reset_starts - 1)
        reset_cur_vec, reset_cur_pelvis, reset_cur_payload = ctl.target_state(store, clip_ids, reset_starts)
        next_vec, next_pelvis, next_payload = ctl.advance_transition_state(store, clip_ids, cur_idx, pred_vec)
        reset_mask = reset[:, None]
        advance_mask = advance[:, None]
        prev_vec = torch.where(reset_mask, reset_prev_vec, torch.where(advance_mask, cur_vec, prev_vec))
        prev_pelvis = torch.where(reset_mask, reset_prev_pelvis, torch.where(advance_mask, cur_pelvis, prev_pelvis))
        prev_payload = torch.where(reset_mask, reset_prev_payload, torch.where(advance_mask, cur_payload, prev_payload))
        cur_vec = torch.where(reset_mask, reset_cur_vec, torch.where(advance_mask, next_vec, cur_vec))
        cur_pelvis = torch.where(reset_mask, reset_cur_pelvis, torch.where(advance_mask, next_pelvis, cur_pelvis))
        cur_payload = torch.where(reset_mask, reset_cur_payload, torch.where(advance_mask, next_payload, cur_payload))
        cur_idx = torch.where(reset, reset_starts, torch.where(continuing, cur_idx + 1, cur_idx))
        predicted_age = torch.where(reset, torch.zeros_like(predicted_age), predicted_age)
    if not score_roots:
        return pred_history[-1].sum() * 0.0
    roots_all = torch.cat(score_roots, dim=0)
    inputs_all = torch.cat(score_inputs, dim=0)
    targets_all = torch.cat(score_targets, dim=0)
    weights_all = torch.cat(score_weights, dim=0)
    total = torch.zeros((), dtype=torch.float32, device=store.device)
    score_chunk = 8192
    for start in range(0, int(roots_all.shape[0]), score_chunk):
        end = min(start + score_chunk, int(roots_all.shape[0]))
        total = total + (
            temporal_score_rows(
                ae,
                schema,
                input_mean,
                input_std,
                target_mean,
                target_std,
                roots_all[start:end],
                inputs_all[start:end],
                targets_all[start:end],
            )
            * weights_all[start:end]
        ).sum()
    return total


def temporal_max_training_start_for_clip_ids(store: ctl.SimpleClipStore, clip_ids: torch.Tensor, frames: int) -> torch.Tensor:
    max_start = ctl.max_training_start_for_clip_ids(store, clip_ids)
    cyclic = store.cyclic.index_select(0, clip_ids)
    return torch.where(cyclic, max_start, (max_start - (int(frames) - 1)).clamp_min(1))


def temporal_start_pool(store: ctl.SimpleClipStore, frames: int) -> ctl.StartPool:
    chunks_c: list[torch.Tensor] = []
    chunks_s: list[torch.Tensor] = []
    for clip_id, clip in enumerate(store.clips):
        if clip.cyclic_animation:
            max_start = ctl.max_training_start_for_clip(clip, store.cfg)
        else:
            max_start = ctl.max_training_start_for_clip(clip, store.cfg) - (int(frames) - 1)
        if max_start < 1:
            continue
        starts = torch.arange(1, max_start + 1, dtype=torch.long, device=store.device)
        chunks_s.append(starts)
        chunks_c.append(torch.full_like(starts, int(clip_id)))
    if not chunks_s:
        raise ValueError("No valid temporal controller starts.")
    return ctl.StartPool(torch.cat(chunks_c), torch.cat(chunks_s))


def load_controller_from_checkpoint(path: Path, store: ctl.SimpleClipStore) -> torch.nn.Module:
    input_dim, output_dim = tl.make_batch_dims(store.prototype, store.cfg)
    model = tl.MLPController(input_dim, output_dim, store.cfg).to(store.device)
    state = torch.load(path, map_location=store.device, weights_only=False)
    model.load_state_dict(state["model"])
    return model


@torch.no_grad()
def diamond_envelope_metric(
    model: torch.nn.Module,
    store: ctl.SimpleClipStore,
    envelope: dict[str, object],
    clip_name: str = DIAMOND_NAME,
) -> dict[str, float]:
    clip_id = next((i for i, clip in enumerate(store.clips) if clip.path.name == clip_name), None)
    if clip_id is None:
        raise ValueError(f"{clip_name} not in store")
    clip = store.clips[int(clip_id)]
    max_cur = ctl.max_training_start_for_clip(clip, store.cfg)
    cur_idx = torch.arange(1, max_cur + 1, dtype=torch.long, device=store.device)
    clip_ids = torch.full_like(cur_idx, int(clip_id))
    prev_vec, prev_pelvis, prev_payload = ctl.target_state(store, clip_ids[:1], cur_idx[:1] - 1)
    cur_vec, cur_pelvis, cur_payload = ctl.target_state(store, clip_ids[:1], cur_idx[:1])
    pred_vecs: list[torch.Tensor] = [cur_vec.squeeze(0)]
    lin_excess: list[torch.Tensor] = []
    ang_excess: list[torch.Tensor] = []
    lin_raw: list[torch.Tensor] = []
    lin_bound: list[torch.Tensor] = []
    model.eval()
    for idx in cur_idx:
        cids = torch.tensor([int(clip_id)], dtype=torch.long, device=store.device)
        i = idx.reshape(1)
        inp = ctl.build_controller_input(store, cids, i, prev_vec, cur_vec, prev_pelvis, cur_pelvis, prev_payload, cur_payload)
        raw = ctl.model_forward(model, inp, cur_vec, store.cfg)
        pred = ctl.clean_output_vector(raw, store)
        cur_root_pos, cur_root_rot, _a, _b = store.root_state(cids, i)
        output_root_pos, output_root_rot = ctl.transition_output_root_state(store, cids, i)
        cur_foot_pos, cur_foot_rot = env.ik_foot_toe_state_from_vec(store, cur_root_pos, cur_root_rot, cur_vec)
        next_foot_pos, next_foot_rot = env.ik_foot_toe_state_from_vec(store, output_root_pos, output_root_rot, pred)
        linear, angular, bound_l, bound_a = env.envelope_values_ik_state_rows(
            store, envelope, cur_foot_pos, cur_foot_rot, next_foot_pos, next_foot_rot, cids, i
        )
        lin_excess.append(F.relu(linear - bound_l))
        ang_excess.append(F.relu(angular - bound_a))
        lin_raw.append(linear)
        lin_bound.append(bound_l)
        prev_vec, prev_pelvis, prev_payload = cur_vec, cur_pelvis, cur_payload
        cur_vec, cur_pelvis, cur_payload = ctl.advance_transition_state(store, cids, i, pred)
        pred_vecs.append(cur_vec.squeeze(0))
    lin = torch.cat(lin_excess)
    ang = torch.cat(ang_excess)
    raw = torch.cat(lin_raw)
    bound = torch.cat(lin_bound)
    return {
        "diamond_linear_excess_mean": float(lin.mean().detach().cpu()),
        "diamond_linear_excess_p95": float(torch.quantile(lin, 0.95).detach().cpu()),
        "diamond_linear_excess_max": float(lin.max().detach().cpu()),
        "diamond_angular_excess_mean": float(ang.mean().detach().cpu()),
        "diamond_angular_excess_p95": float(torch.quantile(ang, 0.95).detach().cpu()),
        "diamond_angular_excess_max": float(ang.max().detach().cpu()),
        "diamond_linear_selected_mps_mean": float(raw.mean().detach().cpu()),
        "diamond_linear_bound_mps_mean": float(bound.mean().detach().cpu()),
        "diamond_frames": float(cur_idx.numel()),
    }


def clip_id_by_name(store: ctl.SimpleClipStore, clip_name: str) -> int:
    clip_id = next((i for i, clip in enumerate(store.clips) if clip.path.name == clip_name), None)
    if clip_id is None:
        raise ValueError(f"{clip_name} not in store")
    return int(clip_id)


@torch.no_grad()
def rollout_clip_vectors(
    model: torch.nn.Module,
    store: ctl.SimpleClipStore,
    clip_name: str,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    clip_id = clip_id_by_name(store, clip_name)
    clip = store.clips[clip_id]
    max_cur = ctl.max_training_start_for_clip(clip, store.cfg)
    cids = torch.tensor([clip_id], dtype=torch.long, device=store.device)
    cur_idx = torch.tensor([1], dtype=torch.long, device=store.device)
    prev_vec, prev_pelvis, prev_payload = ctl.target_state(store, cids, cur_idx - 1)
    cur_vec, cur_pelvis, cur_payload = ctl.target_state(store, cids, cur_idx)
    pred_vecs = [cur_vec.squeeze(0)]
    model.eval()
    for idx_value in range(1, max_cur + 1):
        idx = torch.tensor([idx_value], dtype=torch.long, device=store.device)
        inp = ctl.build_controller_input(store, cids, idx, prev_vec, cur_vec, prev_pelvis, cur_pelvis, prev_payload, cur_payload)
        raw = ctl.model_forward(model, inp, cur_vec, store.cfg)
        pred = ctl.clean_output_vector(raw, store)
        prev_vec, prev_pelvis, prev_payload = cur_vec, cur_pelvis, cur_payload
        cur_vec, cur_pelvis, cur_payload = ctl.advance_transition_state(store, cids, idx, pred)
        pred_vecs.append(cur_vec.squeeze(0))
    frame_idx = torch.arange(1, max_cur + 2, dtype=torch.long, device=store.device)
    return clip_id, torch.stack(pred_vecs, dim=0), frame_idx


@torch.no_grad()
def gt_clip_vectors(
    store: ctl.SimpleClipStore,
    clip_name: str,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    clip_id = clip_id_by_name(store, clip_name)
    clip = store.clips[clip_id]
    max_cur = ctl.max_training_start_for_clip(clip, store.cfg)
    frame_idx = torch.arange(1, max_cur + 2, dtype=torch.long, device=store.device)
    clip_ids = torch.full_like(frame_idx, clip_id)
    return clip_id, store.get_target_output(clip_ids, frame_idx), frame_idx


@torch.no_grad()
def fk_vectors_for_clip(
    store: ctl.SimpleClipStore,
    clip_id: int,
    vecs: torch.Tensor,
    frame_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    clip_ids = torch.full((int(vecs.shape[0]),), int(clip_id), dtype=torch.long, device=store.device)
    root_pos, root_rot, _yaw, _heading = store.root_state(clip_ids, frame_idx)
    pose, _raw = tl.output_to_pose(vecs, store.clips[int(clip_id)])
    return tl.fk_from_pose(store.clips[int(clip_id)], root_pos, root_rot, pose, store.device)[:2]


@torch.no_grad()
def clip_reenactment_metric(
    model: torch.nn.Module,
    store: ctl.SimpleClipStore,
    clip_name: str,
    tail_start: int,
) -> dict[str, float]:
    clip_id, pred_vecs, frame_idx = rollout_clip_vectors(model, store, clip_name)
    _gt_id, gt_vecs, _gt_idx = gt_clip_vectors(store, clip_name)
    pred_pos, pred_rot = fk_vectors_for_clip(store, clip_id, pred_vecs, frame_idx)
    gt_pos, gt_rot = fk_vectors_for_clip(store, clip_id, gt_vecs, frame_idx)
    loc_err = torch.linalg.norm(pred_pos - gt_pos, dim=-1).mean(dim=-1)
    rot_err = tl.geodesic_angles(pred_rot.reshape(-1, 3, 3), gt_rot.reshape(-1, 3, 3)).reshape(pred_rot.shape[0], pred_rot.shape[1]).mean(dim=-1)
    pred_vel = torch.linalg.norm(pred_pos[1:] - pred_pos[:-1], dim=-1).mean(dim=-1)
    gt_vel = torch.linalg.norm(gt_pos[1:] - gt_pos[:-1], dim=-1).mean(dim=-1)
    vel_diff = (pred_vel - gt_vel).abs()
    vec_delta = torch.linalg.norm(pred_vecs[1:, 3:] - pred_vecs[:-1, 3:], dim=-1)
    gt_vec_delta = torch.linalg.norm(gt_vecs[1:, 3:] - gt_vecs[:-1, 3:], dim=-1)
    tail_mask = frame_idx >= int(tail_start)
    tail_vel_mask = frame_idx[:-1] >= int(tail_start)
    return {
        "loc_err_m": float(loc_err.mean().detach().cpu()),
        "loc_err_tail_m": float(loc_err[tail_mask].mean().detach().cpu()),
        "rot_err_rad": float(rot_err.mean().detach().cpu()),
        "rot_err_tail_rad": float(rot_err[tail_mask].mean().detach().cpu()),
        "velocity_diff_mps": float(vel_diff.mean().detach().cpu()),
        "velocity_diff_tail_mps": float(vel_diff[tail_vel_mask].mean().detach().cpu()),
        "motion_ratio_tail": float((vec_delta[tail_vel_mask].mean() / gt_vec_delta[tail_vel_mask].mean().clamp_min(1e-8)).detach().cpu()),
    }


@torch.no_grad()
def amp_eval_metrics(model: torch.nn.Module, store: ctl.SimpleClipStore) -> dict[str, float]:
    result: dict[str, float] = {}
    probes = (
        ("circleR", "M_Neutral_Walk_Circle_Strafe_R.npz", 48),
        ("diamond", DIAMOND_NAME, 28),
    )
    for prefix, clip_name, tail_start in probes:
        if any(clip.path.name == clip_name for clip in store.clips):
            metric = clip_reenactment_metric(model, store, clip_name, tail_start)
            for key, value in metric.items():
                result[f"{prefix}_{key}"] = value
    if any(clip.path.name == DIAMOND_NAME for clip in store.clips):
        try:
            envelope = env.load_or_build_excess_envelope(store)
            diamond = diamond_envelope_metric(model, store, envelope)
            result["diamond_linear_excess_mean"] = diamond["diamond_linear_excess_mean"]
            result["diamond_angular_excess_mean"] = diamond["diamond_angular_excess_mean"]
        except Exception as exc:
            result["diamond_envelope_error"] = float("nan")
            print(f"diamond envelope eval skipped: {exc}", flush=True)
    return result


def amp_eval_score(metrics: dict[str, float]) -> float:
    keys = (
        "circleR_loc_err_tail_m",
        "circleR_rot_err_tail_rad",
        "diamond_loc_err_tail_m",
        "diamond_rot_err_tail_rad",
        "diamond_linear_excess_mean",
    )
    total = 0.0
    count = 0
    for key in keys:
        value = metrics.get(key)
        if value is None or math.isnan(float(value)):
            continue
        total += float(value)
        count += 1
    return total / float(max(1, count))


def specs_for_amp_dataset(dataset: str) -> list[tuple[Path, bool]]:
    dataset = str(dataset).lower().strip()
    if dataset == "mini":
        return mini_specs()
    if dataset == "full":
        return full_specs()
    raise ValueError(f"Unknown AMP dataset {dataset!r}")


def save_amp_checkpoint(
    path: Path,
    model: torch.nn.Module,
    critic: TemporalAMPCritic,
    controller_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    step: int,
    best: float,
    cfg: AMPConfig,
    locomotion_cfg: tl.TrainConfig,
    schema: dict[str, object],
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "temporal_amp_controller",
            "model": model.state_dict(),
            "critic": critic.state_dict(),
            "controller_optimizer": controller_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "step": int(step),
            "best": float(best),
            "amp_config": asdict(cfg),
            "locomotion_config": asdict(locomotion_cfg),
            "schema": schema,
            "feature_mean": feature_mean.detach().cpu(),
            "feature_std": feature_std.detach().cpu(),
            "metadata": metadata,
        },
        path,
    )


def train_amp_controller(
    dataset: str,
    run_label: str,
    steps: int,
    init_checkpoint: Path,
    rollout_k: int,
    batch_size: int,
    critic_warmup_steps: int,
) -> Path:
    device = device_and_seed()
    locomotion_cfg, store = load_store(specs_for_amp_dataset(dataset), device)
    cfg = AMPConfig(
        rollout_k=int(rollout_k),
        batch_size=max(1, int(batch_size)),
        steps=max(1, int(steps)),
        critic_warmup_steps=max(0, int(critic_warmup_steps)),
    )
    if int(cfg.rollout_k) < int(cfg.frames):
        raise ValueError(f"AMP rollout_k={cfg.rollout_k} must be at least frames={cfg.frames}")
    real_features, _roots, _current, _future, _clip_ids, _starts, schema = build_amp_real_rows(store, cfg)
    feature_mean, feature_std = normalize_amp_features(real_features)
    simple_ae_model, simple_ae_mean, simple_ae_std, _simple_ae_ckpt = ctl.load_simple_ae(FULL_SIMPLE_AE, device)
    model = load_controller_from_checkpoint(init_checkpoint, store)
    critic = TemporalAMPCritic(int(schema["input_dim"]), cfg).to(device)
    controller_optimizer = ctl.make_adamw(model.parameters(), float(cfg.controller_lr), device)
    critic_optimizer = ctl.make_adamw(critic.parameters(), float(cfg.critic_lr), device)
    pool = temporal_start_pool(store, int(cfg.frames))
    cfg.batch_size = min(int(cfg.batch_size), int(pool.row_count))
    supervised_raw_mean = estimate_supervised_k1_loss(
        model,
        store,
        int(cfg.batch_size),
        pool,
        int(cfg.supervised_estimate_batches),
    )
    supervised_weight = (
        float(cfg.supervised_target_loss) / max(float(supervised_raw_mean), 1e-12)
        if float(cfg.supervised_target_loss) > 0.0
        else 0.0
    )

    run_id = ik_run_id(run_label)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    metadata: dict[str, object] = {
        "dataset": str(dataset),
        "npz_paths": [str(path) for path, _cyclic in specs_for_amp_dataset(dataset)],
        "init_checkpoint": str(init_checkpoint),
        "tensorboard_logdir": str(run_dir / "tb"),
        "loss": "simple_ae_plus_conditional_amp_online_critic",
        "simple_ae_checkpoint": str(FULL_SIMPLE_AE),
        "amp_rule": "real GT windows are positive; current generated windows become negatives immediately",
        "amp_fool_weight": float(cfg.amp_fool_weight),
        "row_count": int(real_features.shape[0]),
        "batch_size": int(cfg.batch_size),
        "rollout_k": int(cfg.rollout_k),
        "supervised_k1_raw_mean_at_init": float(supervised_raw_mean),
        "supervised_k1_weight": float(supervised_weight),
        "supervised_k1_target_loss": float(cfg.supervised_target_loss),
    }
    config_payload = {"important": metadata, "amp_config": asdict(cfg), "schema": schema, "locomotion_config": asdict(locomotion_cfg)}
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    (run_dir / "config_readable.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    writer = SummaryWriter(log_dir=str(run_dir / "tb"), flush_secs=1)
    writer.add_text("config/json", f"```json\n{json.dumps(config_payload, indent=2)}\n```", 0)
    writer.flush()
    ctl.refresh_tensorboard_async()

    best = math.inf
    last_controller_loss = math.inf
    init_path = ckpt_dir / f"{run_id}_init.pt"
    latest_path = ckpt_dir / f"{run_id}_latest.pt"
    best_path = ckpt_dir / f"{run_id}_best.pt"
    save_amp_checkpoint(
        init_path,
        model,
        critic,
        controller_optimizer,
        critic_optimizer,
        0,
        best,
        cfg,
        locomotion_cfg,
        schema,
        feature_mean,
        feature_std,
        metadata,
    )
    print(f"amp_controller run={run_id} dataset={dataset} rows={real_features.shape[0]} batch={cfg.batch_size} K={cfg.rollout_k}", flush=True)

    start_time = time.perf_counter()
    for warmup_step in range(1, int(cfg.critic_warmup_steps) + 1):
        with torch.no_grad():
            fake_features, fake_weights, _simple_ae_loss = collect_amp_fake_features(model, store, cfg, pool)
        critic_loss, critic_stats = amp_critic_loss(critic, real_features, fake_features, fake_weights, feature_mean, feature_std)
        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_optimizer.step()
        if warmup_step == 1 or warmup_step % max(1, int(cfg.log_every)) == 0 or warmup_step == int(cfg.critic_warmup_steps):
            print(
                f"warmup={warmup_step}/{cfg.critic_warmup_steps} critic={float(critic_loss.detach().cpu()):.6g} "
                f"real_logit={float(critic_stats['real_logit'].cpu()):.4f} fake_logit={float(critic_stats['fake_logit'].cpu()):.4f}",
                flush=True,
            )

    best_metric = math.inf
    for step in range(1, int(cfg.steps) + 1):
        model.train()
        critic.train()
        controller_optimizer.zero_grad(set_to_none=True)
        amp_loss, simple_ae_loss, fake_features, fake_weights, controller_stats = amp_controller_loss_and_features(
            model,
            critic,
            feature_mean,
            feature_std,
            store,
            cfg,
            pool,
            simple_ae_model,
            simple_ae_mean,
            simple_ae_std,
        )
        supervised_loss = supervised_k1_transition_loss(model, store, int(cfg.batch_size), pool)
        weighted_simple_ae_loss = float(LOSS_SCALE) * simple_ae_loss
        weighted_amp_loss = float(cfg.amp_fool_weight) * amp_loss
        weighted_supervised_loss = float(supervised_weight) * supervised_loss
        controller_loss = weighted_simple_ae_loss + weighted_amp_loss + weighted_supervised_loss
        controller_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        controller_optimizer.step()
        last_controller_loss = float(controller_loss.detach().cpu())

        critic_loss, critic_stats = amp_critic_loss(critic, real_features, fake_features, fake_weights, feature_mean, feature_std)
        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_optimizer.step()

        should_log = step == 1 or step % int(cfg.log_every) == 0 or step == int(cfg.steps)
        should_eval = step == 1 or step % int(cfg.eval_every) == 0 or step == int(cfg.steps)
        metrics: dict[str, float] = {}
        if should_eval:
            model.eval()
            metrics = amp_eval_metrics(model, store)
            metric_score = amp_eval_score(metrics)
            if metric_score < best_metric:
                best_metric = metric_score
                best = metric_score
                save_amp_checkpoint(
                    best_path,
                    model,
                    critic,
                    controller_optimizer,
                    critic_optimizer,
                    step,
                    best,
                    cfg,
                    locomotion_cfg,
                    schema,
                    feature_mean,
                    feature_std,
                    metadata,
                )
            (run_dir / "latest_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        if should_log:
            elapsed = time.perf_counter() - start_time
            critic_loss_f = float(critic_loss.detach().cpu())
            real_logit_f = float(critic_stats["real_logit"].detach().cpu())
            fake_logit_f = float(critic_stats["fake_logit"].detach().cpu())
            valid_windows_f = float(controller_stats["valid_windows"].detach().cpu())
            writer.add_scalar("loss/amp_controller", last_controller_loss, step)
            writer.add_scalar("loss/simple_ae", float(simple_ae_loss.detach().cpu()), step)
            writer.add_scalar("loss/simple_ae_weighted", float(weighted_simple_ae_loss.detach().cpu()), step)
            writer.add_scalar("loss/amp_fool", float(amp_loss.detach().cpu()), step)
            writer.add_scalar("loss/amp_fool_weighted", float(weighted_amp_loss.detach().cpu()), step)
            writer.add_scalar("loss/supervised_k1_raw", float(supervised_loss.detach().cpu()), step)
            writer.add_scalar("loss/supervised_k1_weighted", float(weighted_supervised_loss.detach().cpu()), step)
            writer.add_scalar("loss/amp_critic", critic_loss_f, step)
            writer.add_scalar("logit/real", real_logit_f, step)
            writer.add_scalar("logit/fake", fake_logit_f, step)
            writer.add_scalar("amp/valid_windows", valid_windows_f, step)
            writer.add_scalar("time/elapsed_s", elapsed, step)
            for key, value in metrics.items():
                if not math.isnan(float(value)):
                    writer.add_scalar(f"eval/{key}", float(value), step)
            writer.flush()
            save_amp_checkpoint(
                latest_path,
                model,
                critic,
                controller_optimizer,
                critic_optimizer,
                step,
                best,
                cfg,
                locomotion_cfg,
                schema,
                feature_mean,
                feature_std,
                metadata,
            )
            if step % int(cfg.snapshot_every) == 0 or step == int(cfg.steps):
                save_amp_checkpoint(
                    ckpt_dir / f"{run_id}_step{step:06d}.pt",
                    model,
                    critic,
                    controller_optimizer,
                    critic_optimizer,
                    step,
                    best,
                    cfg,
                    locomotion_cfg,
                    schema,
                    feature_mean,
                    feature_std,
                    metadata,
                )
            metric_text = ""
            if metrics:
                metric_text = (
                    f" circleR_loc_tail={metrics.get('circleR_loc_err_tail_m', float('nan')):.4f}"
                    f" circleR_motion={metrics.get('circleR_motion_ratio_tail', float('nan')):.3f}"
                    f" diamond_loc_tail={metrics.get('diamond_loc_err_tail_m', float('nan')):.4f}"
                    f" diamond_motion={metrics.get('diamond_motion_ratio_tail', float('nan')):.3f}"
                )
            print(
                f"step={step}/{cfg.steps} ctrl={last_controller_loss:.6g} critic={critic_loss_f:.6g} "
                f"ae_w={float(weighted_simple_ae_loss.detach().cpu()):.6g} amp_w={float(weighted_amp_loss.detach().cpu()):.6g} "
                f"sup_w={float(weighted_supervised_loss.detach().cpu()):.6g} "
                f"real_logit={real_logit_f:.3f} fake_logit={fake_logit_f:.3f} windows={valid_windows_f:.0f}"
                f"{metric_text} elapsed_s={elapsed:.1f}",
                flush=True,
            )

    last_path = ckpt_dir / f"{run_id}_last.pt"
    save_amp_checkpoint(
        last_path,
        model,
        critic,
        controller_optimizer,
        critic_optimizer,
        int(cfg.steps),
        best,
        cfg,
        locomotion_cfg,
        schema,
        feature_mean,
        feature_std,
        metadata,
    )
    writer.close()
    print(f"amp_controller done checkpoint={last_path}", flush=True)
    return last_path


def amp_smoke_tests() -> dict[str, float]:
    device = device_and_seed()
    _locomotion_cfg, store = load_store(mini_specs(), device)
    cfg = AMPConfig(batch_size=8, rollout_k=8, steps=1, critic_warmup_steps=0)
    real_features, _roots, _current, _future, _clip_ids, _starts, schema = build_amp_real_rows(store, cfg)
    feature_mean, feature_std = normalize_amp_features(real_features)
    model = load_controller_from_checkpoint(BASELINE_114624, store)
    critic = TemporalAMPCritic(int(schema["input_dim"]), cfg).to(device)
    simple_ae_model, simple_ae_mean, simple_ae_std, _simple_ae_ckpt = ctl.load_simple_ae(FULL_SIMPLE_AE, device)
    pool = temporal_start_pool(store, int(cfg.frames))
    amp_loss, simple_ae_loss, fake_features, fake_weights, _controller_stats = amp_controller_loss_and_features(
        model,
        critic,
        feature_mean,
        feature_std,
        store,
        cfg,
        pool,
        simple_ae_model,
        simple_ae_mean,
        simple_ae_std,
    )
    controller_loss = float(LOSS_SCALE) * simple_ae_loss + float(cfg.amp_fool_weight) * amp_loss
    model.zero_grad(set_to_none=True)
    controller_loss.backward()
    controller_grad = torch.stack(
        [p.grad.detach().norm() for p in model.parameters() if p.grad is not None]
    ).sum()
    critic_loss, _critic_stats = amp_critic_loss(critic, real_features, fake_features.detach(), fake_weights.detach(), feature_mean, feature_std)
    critic.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_grad = torch.stack(
        [p.grad.detach().norm() for p in critic.parameters() if p.grad is not None]
    ).sum()
    result = {
        "real_rows": float(real_features.shape[0]),
        "feature_dim": float(real_features.shape[1]),
        "fake_rows": float(fake_features.shape[0]),
        "fake_valid_windows": float(fake_weights.sum().detach().cpu()),
        "controller_loss": float(controller_loss.detach().cpu()),
        "critic_loss": float(critic_loss.detach().cpu()),
        "controller_grad_norm_sum": float(controller_grad.detach().cpu()),
        "critic_grad_norm_sum": float(critic_grad.detach().cpu()),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError(f"AMP smoke test produced non-finite values: {result}")
    if result["fake_valid_windows"] <= 0.0 or result["controller_grad_norm_sum"] <= 0.0 or result["critic_grad_norm_sum"] <= 0.0:
        raise RuntimeError(f"AMP smoke test failed gradient/window checks: {result}")
    print(json.dumps(result, indent=2), flush=True)
    return result


def train_controller_with_temporal_ae(ae_path: Path, run_label: str, steps: int) -> Path:
    device = device_and_seed()
    _cfg, store = load_store(mini_specs(), device)
    ae, schema, input_mean, input_std, target_mean, target_std, _ckpt = load_temporal_ae(ae_path, device)
    model = load_controller_from_checkpoint(BASELINE_114624, store)
    optimizer = ctl.make_adamw(model.parameters(), CONTROLLER_LR, device)
    run_id = ik_run_id(run_label)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"), flush_secs=1)
    metadata = {
        "temporal_ae_checkpoint": str(ae_path),
        "init_checkpoint": str(BASELINE_114624),
        "npz_paths": [str(path) for path, _cyclic in mini_specs()],
        "rollout_k": CONTROLLER_K,
        "loss": "temporal_denoising_ae_only",
        "tensorboard_logdir": str(run_dir / "tb"),
    }
    (run_dir / "config.json").write_text(json.dumps({"schema": schema, "metadata": metadata}, indent=2), encoding="utf-8")
    writer.add_text("config/json", f"```json\n{json.dumps({'schema': schema, 'metadata': metadata}, indent=2)}\n```", 0)
    writer.flush()
    ctl.refresh_tensorboard_async()
    envelope = env.load_or_build_excess_envelope(store)
    before = diamond_envelope_metric(model, store, envelope)
    for key, value in before.items():
        writer.add_scalar(f"diamond_before/{key}", value, 0)
    print(f"controller_temporal run={run_id} ae={ae_path.name} before={before}", flush=True)
    pool = temporal_start_pool(store, int(schema["frames"]))
    batch = min(CONTROLLER_BATCH, int(pool.row_count))
    start = time.perf_counter()
    last_loss = math.inf
    best_loss = math.inf
    init_path = checkpoint_path(run_dir, run_id, "init")
    latest_path = checkpoint_path(run_dir, run_id, "latest")
    best_path = checkpoint_path(run_dir, run_id, "best")

    def save_controller_checkpoint(path: Path, step: int, loss_value: float) -> None:
        torch.save(
            tl.checkpoint_payload(model, optimizer, int(step), float(loss_value), CONTROLLER_K, store.cfg, metadata),
            path,
        )

    save_controller_checkpoint(init_path, 0, last_loss)
    for step in range(1, int(steps) + 1):
        loss = temporal_rollout_loss(
            model,
            ae,
            schema,
            input_mean,
            input_std,
            target_mean,
            target_std,
            store,
            CONTROLLER_K,
            batch,
            pool,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if step == 1 or step % CONTROLLER_LOG_EVERY == 0 or step == int(steps):
            writer.add_scalar("loss/temporal_ae_score", last_loss * LOSS_SCALE, step)
            writer.add_scalar("time/elapsed_s", time.perf_counter() - start, step)
            writer.flush()
            print(f"{run_id}: step={step}/{steps} loss={last_loss:.6g}", flush=True)
            if last_loss < best_loss:
                best_loss = last_loss
                save_controller_checkpoint(best_path, step, best_loss)
            save_controller_checkpoint(latest_path, step, last_loss)
            if step % CONTROLLER_SNAPSHOT_EVERY == 0 or step == int(steps):
                save_controller_checkpoint(checkpoint_path(run_dir, run_id, f"step{step:06d}"), step, last_loss)
    after = diamond_envelope_metric(model, store, envelope)
    for key, value in after.items():
        writer.add_scalar(f"diamond_after/{key}", value, int(steps))
    ckpt = checkpoint_path(run_dir, run_id, "last")
    save_controller_checkpoint(ckpt, int(steps), last_loss)
    save_controller_checkpoint(latest_path, int(steps), last_loss)
    (run_dir / "metrics.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    writer.close()
    print(f"controller_temporal done run={run_id} after={after} checkpoint={ckpt}", flush=True)
    return ckpt


def metric_only(checkpoint: Path) -> dict[str, float]:
    device = device_and_seed()
    _cfg, store = load_store(mini_specs(), device)
    model = load_controller_from_checkpoint(checkpoint, store)
    envelope = env.load_or_build_excess_envelope(store)
    metric = diamond_envelope_metric(model, store, envelope)
    print(json.dumps(metric, indent=2), flush=True)
    return metric


def main() -> None:
    parser = argparse.ArgumentParser(description="Root-conditioned temporal denoising AE experiments for IK controllers.")
    parser.add_argument("--phase", choices=("metric", "train-ae", "train-controller", "all", "smoke-amp", "train-amp"), default="all")
    parser.add_argument(
        "--variant",
        choices=("root8_body8", "root8_body1", "root8_body1_to_body8", "both", "all3"),
        default="both",
    )
    parser.add_argument("--ae-checkpoint", default="")
    parser.add_argument("--run-label", default="")
    parser.add_argument("--ae-steps", type=int, default=AE_STEPS)
    parser.add_argument("--controller-steps", type=int, default=CONTROLLER_STEPS)
    parser.add_argument("--checkpoint", default=str(BASELINE_114624))
    parser.add_argument("--dataset", choices=("mini", "full"), default="mini")
    parser.add_argument("--steps", type=int, default=AMP_STEPS)
    parser.add_argument("--rollout-k", type=int, default=AMP_ROLLOUT_K)
    parser.add_argument("--batch-size", type=int, default=AMP_BATCH)
    parser.add_argument("--critic-warmup-steps", type=int, default=AMP_CRITIC_WARMUP_STEPS)
    args = parser.parse_args()

    if args.phase == "smoke-amp":
        amp_smoke_tests()
        return

    if args.phase == "train-amp":
        label = args.run_label or f"amp_{args.dataset}_online_critic"
        train_amp_controller(
            args.dataset,
            label,
            int(args.steps),
            Path(args.checkpoint).resolve(),
            int(args.rollout_k),
            int(args.batch_size),
            int(args.critic_warmup_steps),
        )
        return

    if args.phase == "metric":
        metric_only(Path(args.checkpoint).resolve())
        return

    if args.variant == "both":
        variants = ("root8_body8", "root8_body1")
    elif args.variant == "all3":
        variants = ("root8_body8", "root8_body1", "root8_body1_to_body8")
    else:
        variants = (args.variant,)
    ae_paths: dict[str, Path] = {}
    if args.phase in {"train-ae", "all"}:
        for variant in variants:
            label = args.run_label or f"temporal_denoise_{variant}_full"
            ae_paths[variant] = train_temporal_ae_variant(variant, label, int(args.ae_steps))
    if args.phase == "train-ae":
        return

    if args.phase == "train-controller":
        if not args.ae_checkpoint:
            raise ValueError("--ae-checkpoint is required for train-controller")
        variant = args.variant if args.variant != "both" else "temporal"
        label = args.run_label or f"temporal_controller_{variant}"
        train_controller_with_temporal_ae(Path(args.ae_checkpoint).resolve(), label, int(args.controller_steps))
        return

    for variant, ae_path in ae_paths.items():
        train_controller_with_temporal_ae(
            ae_path,
            args.run_label or f"temporal_ctrl_{variant}_mini_from114624",
            int(args.controller_steps),
        )


if __name__ == "__main__":
    main()
