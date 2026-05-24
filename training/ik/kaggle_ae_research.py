from __future__ import annotations

import argparse
import csv
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
    from . import ik_core as tl
    from . import train_simple_autoencoder as sae
    from . import train_simple_ae_controller as ctl
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    from naming import checkpoint_path, ik_run_id
    import ik_core as tl
    import train_simple_autoencoder as sae
    import train_simple_ae_controller as ctl

ensure_paths()


RUNS_DIR = PROJECT_ROOT / "training" / "runs"
PERIODIC_DIR = PROJECT_ROOT / "ue5" / "animations_omni_only_full" / "npz_final"
NONPERIODIC_DIR = PROJECT_ROOT / "ue5" / "animations_transitions_only_full_trimmed" / "npz_final"

DEFAULT_PERIODIC = (
    "M_Neutral_Stand_Idle_Loop.npz",
    "M_Neutral_Walk_Loop_F.npz",
)
DEFAULT_NONPERIODIC = (
    "M_Neutral_Stand_Turn_045_L.npz",
    "M_Neutral_Stand_Turn_045_R.npz",
    "M_Neutral_Walk_Circle_Strafe_L.npz",
    "M_Neutral_Walk_Circle_Strafe_R.npz",
    "M_Neutral_Walk_Hourglass_FL_RL_Rfoot.npz",
)
DEFAULT_CONTROLLER_PERIODIC = ()
DEFAULT_CONTROLLER_NONPERIODIC = ("M_Neutral_Walk_Diamond_BL_F_Lfoot.npz",)

OUTPUT_OPT_STEPS = 80
OUTPUT_OPT_LR = 0.05
EVAL_ROWS = 1024
REPORT_EVERY = 100
CONTROLLER_REPORT_EVERY = 250
CONTROLLER_STEPS = 3000
CONTROLLER_BATCH = 1024
CONTROLLER_LR = 1e-4
CONTROLLER_SCHEDULE = (1, 2, 8, 16)
BASELINE_VARIANT = "current_simple_ae"


@dataclass(frozen=True)
class Variant:
    name: str
    latent_dim: int = 32
    hidden_dim: int = 512
    layers: int = 2
    steps: int = 4000
    lr: float = 1e-3
    weight_decay: float = 1e-5
    noise_std: float = 0.0
    output_noise_std: float = 0.0
    shuffle_output_prob: float = 0.0
    output_weight: float = 1.0
    topk_frac: float = 1.0


class ResearchAE(nn.Module):
    def __init__(self, dim: int, variant: Variant):
        super().__init__()
        modules: list[nn.Module] = []
        in_dim = int(dim)
        for _ in range(int(variant.layers)):
            modules.extend((nn.Linear(in_dim, variant.hidden_dim), nn.LayerNorm(variant.hidden_dim), nn.GELU()))
            in_dim = int(variant.hidden_dim)
        modules.extend((nn.Linear(in_dim, variant.latent_dim), nn.GELU()))
        in_dim = int(variant.latent_dim)
        for _ in range(int(variant.layers)):
            modules.extend((nn.Linear(in_dim, variant.hidden_dim), nn.LayerNorm(variant.hidden_dim), nn.GELU()))
            in_dim = int(variant.hidden_dim)
        modules.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def resolve_subset(periodic_text: str, nonperiodic_text: str) -> list[tuple[Path, bool]]:
    specs: list[tuple[Path, bool]] = []
    periodic_names = tuple(name.strip() for name in periodic_text.split(";") if name.strip()) or DEFAULT_PERIODIC
    nonperiodic_names = tuple(name.strip() for name in nonperiodic_text.split(";") if name.strip()) or DEFAULT_NONPERIODIC
    if len(periodic_names) == 1 and periodic_names[0].lower() in {"__all__", "all", "*"}:
        periodic_names = tuple(path.name for path in sorted(PERIODIC_DIR.glob("*.npz")))
    if len(nonperiodic_names) == 1 and nonperiodic_names[0].lower() in {"__all__", "all", "*"}:
        nonperiodic_names = tuple(path.name for path in sorted(NONPERIODIC_DIR.glob("*.npz")))
    for name in periodic_names:
        path = PERIODIC_DIR / name
        if not path.exists():
            raise FileNotFoundError(path)
        specs.append((path, True))
    for name in nonperiodic_names:
        path = NONPERIODIC_DIR / name
        if not path.exists():
            raise FileNotFoundError(path)
        specs.append((path, False))
    return specs


def names_from_text(text: str, defaults: tuple[str, ...]) -> tuple[str, ...]:
    names = tuple(name.strip() for name in str(text or "").split(";") if name.strip())
    if not names:
        return defaults
    if len(names) == 1 and names[0].lower() in {"__empty__", "empty", "none", "null"}:
        return ()
    return names


def resolve_named_subset(
    periodic_text: str,
    nonperiodic_text: str,
    periodic_defaults: tuple[str, ...],
    nonperiodic_defaults: tuple[str, ...],
) -> list[tuple[Path, bool]]:
    specs: list[tuple[Path, bool]] = []
    periodic_names = names_from_text(periodic_text, periodic_defaults)
    nonperiodic_names = names_from_text(nonperiodic_text, nonperiodic_defaults)
    if len(periodic_names) == 1 and periodic_names[0].lower() in {"__all__", "all", "*"}:
        periodic_names = tuple(path.name for path in sorted(PERIODIC_DIR.glob("*.npz")))
    if len(nonperiodic_names) == 1 and nonperiodic_names[0].lower() in {"__all__", "all", "*"}:
        nonperiodic_names = tuple(path.name for path in sorted(NONPERIODIC_DIR.glob("*.npz")))
    for name in periodic_names:
        path = PERIODIC_DIR / name
        if not path.exists():
            raise FileNotFoundError(path)
        specs.append((path, True))
    for name in nonperiodic_names:
        path = NONPERIODIC_DIR / name
        if not path.exists():
            raise FileNotFoundError(path)
        specs.append((path, False))
    if not specs:
        raise ValueError("Resolved empty clip subset.")
    return specs


def variants(train_steps: int, window_frames: int = 1) -> list[Variant]:
    base_steps = int(train_steps)
    prefix = "" if int(window_frames) == 1 else f"temporal{int(window_frames)}_"
    simple_name = BASELINE_VARIANT if int(window_frames) == 1 else f"{prefix}simple_recipe"
    return [
        Variant(
            simple_name,
            latent_dim=int(sae.LATENT_DIM),
            hidden_dim=int(sae.HIDDEN_DIM),
            layers=int(sae.NUM_HIDDEN_LAYERS),
            steps=base_steps,
            lr=float(sae.LEARNING_RATE),
            weight_decay=float(sae.WEIGHT_DECAY),
        ),
        Variant(f"{prefix}tight_ld16", latent_dim=16, steps=base_steps),
        Variant(f"{prefix}wide_ld64", latent_dim=64, steps=base_steps),
        Variant(f"{prefix}wide_ld128", latent_dim=128, steps=base_steps),
        Variant(f"{prefix}small_hidden_ld32", latent_dim=32, hidden_dim=256, steps=base_steps),
        Variant(f"{prefix}denoise_light", latent_dim=32, steps=base_steps, noise_std=0.03, output_noise_std=0.03),
        Variant(f"{prefix}denoise_medium", latent_dim=32, steps=base_steps, noise_std=0.06, output_noise_std=0.08),
        Variant(f"{prefix}denoise_output_only", latent_dim=32, steps=base_steps, output_noise_std=0.10),
        Variant(f"{prefix}denoise_shuffle10", latent_dim=32, steps=base_steps, output_noise_std=0.06, shuffle_output_prob=0.10),
        Variant(f"{prefix}output_weight2", latent_dim=32, steps=base_steps, output_weight=2.0),
        Variant(f"{prefix}output_weight4", latent_dim=32, steps=base_steps, output_weight=4.0),
        Variant(f"{prefix}topk25", latent_dim=32, steps=base_steps, topk_frac=0.25),
        Variant(f"{prefix}denoise_topk25", latent_dim=32, steps=base_steps, noise_std=0.04, output_noise_std=0.08, topk_frac=0.25),
    ]


def select_variants(all_variants: list[Variant], names_text: str, max_variants: int) -> list[Variant]:
    selected = all_variants
    names = [name.strip() for name in str(names_text or "").split(";") if name.strip()]
    if names:
        by_name = {variant.name: variant for variant in all_variants}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ValueError(f"Unknown variant name(s): {missing}; available={list(by_name)}")
        selected = [by_name[name] for name in names]
    if int(max_variants) > 0:
        selected = selected[: int(max_variants)]
    return selected


def window_frames(schema: dict[str, object]) -> int:
    return max(1, int(schema.get("window_frames", 1)))


def base_total_dim(schema: dict[str, object]) -> int:
    return int(schema.get("base_total_dim", schema["total_dim"]))


def frame_offsets(schema: dict[str, object]) -> list[int]:
    stride = base_total_dim(schema)
    return [frame * stride for frame in range(window_frames(schema))]


def output_mask(schema: dict[str, object], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.zeros((int(schema["total_dim"]),), dtype=dtype, device=device)
    out_start = int(schema["target_output_start"])
    out_end = int(schema["target_output_end"])
    for offset in frame_offsets(schema):
        mask[offset + out_start : offset + out_end] = 1.0
    return mask


def weighted_dim_mask(schema: dict[str, object], device: torch.device, dtype: torch.dtype, output_weight: float) -> torch.Tensor:
    weights = torch.ones((int(schema["total_dim"]),), dtype=dtype, device=device)
    if float(output_weight) != 1.0:
        out_start = int(schema["target_output_start"])
        out_end = int(schema["target_output_end"])
        for offset in frame_offsets(schema):
            weights[offset + out_start : offset + out_end] = float(output_weight)
    return weights


def temporal_schema(base_schema: dict[str, object], frames: int) -> dict[str, object]:
    frames = max(1, int(frames))
    if frames == 1:
        out = dict(base_schema)
        out["window_frames"] = 1
        out["base_total_dim"] = int(base_schema["total_dim"])
        return out
    out = dict(base_schema)
    out["window_frames"] = frames
    out["base_total_dim"] = int(base_schema["total_dim"])
    out["total_dim"] = int(base_schema["total_dim"]) * frames
    out["feature"] = f"{frames}x_controller_input_plus_target_output"
    return out


def set_requires_grad(module: nn.Module, value: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(value)


def make_corrupted_input(x: torch.Tensor, schema: dict[str, object], variant: Variant) -> torch.Tensor:
    if variant.noise_std <= 0.0 and variant.output_noise_std <= 0.0 and variant.shuffle_output_prob <= 0.0:
        return x
    y = x.clone()
    input_root_start = int(schema["input_root_start"])
    input_root_end = int(schema["input_root_end"])
    output_start = int(schema["target_output_start"])
    output_end = int(schema["target_output_end"])
    if variant.noise_std > 0.0:
        noise = torch.randn_like(y) * float(variant.noise_std)
        for offset in frame_offsets(schema):
            noise[:, offset + input_root_start : offset + input_root_end] = 0.0
            noise[:, offset + output_start : offset + output_end] = 0.0
        y = y + noise
    if variant.output_noise_std > 0.0:
        for offset in frame_offsets(schema):
            sl = slice(offset + output_start, offset + output_end)
            y[:, sl] = y[:, sl] + torch.randn_like(y[:, sl]) * float(variant.output_noise_std)
    if variant.shuffle_output_prob > 0.0:
        mask = torch.rand((x.shape[0],), device=x.device) < float(variant.shuffle_output_prob)
        if bool(mask.any()):
            perm = torch.randperm(x.shape[0], device=x.device)
            shuffled = x.index_select(0, perm)
            for offset in frame_offsets(schema):
                sl = slice(offset + output_start, offset + output_end)
                y[mask, sl] = shuffled[mask, sl]
    return y


def variant_loss(recon: torch.Tensor, target: torch.Tensor, schema: dict[str, object], variant: Variant) -> torch.Tensor:
    err = (recon - target).square()
    if variant.output_weight != 1.0:
        weights = weighted_dim_mask(schema, target.device, target.dtype, float(variant.output_weight))
        err = err * weights.reshape(1, -1)
    if variant.topk_frac < 1.0:
        k = max(1, int(math.ceil(err.shape[-1] * float(variant.topk_frac))))
        return err.topk(k, dim=-1).values.mean()
    return err.mean()


def row_output_mse(model: nn.Module, x: torch.Tensor, schema: dict[str, object], batch_size: int = 8192) -> torch.Tensor:
    mask = output_mask(schema, x.device, x.dtype).reshape(1, -1)
    denom = mask.sum().clamp_min(1.0)
    out: list[torch.Tensor] = []
    for start in range(0, x.shape[0], int(batch_size)):
        part = x[start : start + int(batch_size)]
        recon = model(part)
        out.append(((recon - part).square() * mask).sum(dim=-1) / denom)
    return torch.cat(out, dim=0)


def ae_score_rows(model: nn.Module, feature: torch.Tensor, schema: dict[str, object]) -> torch.Tensor:
    mask = output_mask(schema, feature.device, feature.dtype).reshape(1, -1)
    denom = mask.sum().clamp_min(1.0)
    recon = model(feature)
    return ((recon - feature).square() * mask).sum(dim=-1) / denom


def build_temporal_features(
    raw: torch.Tensor,
    clip_ids: torch.Tensor,
    cur_indices: torch.Tensor,
    base_schema: dict[str, object],
    frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    frames = max(1, int(frames))
    schema = temporal_schema(base_schema, frames)
    if frames == 1:
        return raw, clip_ids, cur_indices, schema

    chunks: list[torch.Tensor] = []
    clip_chunks: list[torch.Tensor] = []
    idx_chunks: list[torch.Tensor] = []
    for clip_id in clip_ids.unique(sorted=True).tolist():
        rows = (clip_ids == int(clip_id)).nonzero(as_tuple=False).flatten()
        order = torch.argsort(cur_indices.index_select(0, rows))
        rows = rows.index_select(0, order)
        idx = cur_indices.index_select(0, rows)
        if rows.numel() < frames:
            continue
        for end in range(frames - 1, int(rows.numel())):
            idx_window = idx[end - frames + 1 : end + 1]
            if not bool(torch.all(idx_window[1:] - idx_window[:-1] == 1)):
                continue
            row_window = rows[end - frames + 1 : end + 1]
            chunks.append(raw.index_select(0, row_window).reshape(1, -1))
            clip_chunks.append(torch.tensor([int(clip_id)], dtype=torch.long))
            idx_chunks.append(idx[end].reshape(1).to(dtype=torch.long))
    if not chunks:
        raise ValueError(f"No valid {frames}-frame AE windows found.")
    return torch.cat(chunks, dim=0), torch.cat(clip_chunks, dim=0), torch.cat(idx_chunks, dim=0), schema


def gt_transition_feature(store: ctl.SimpleClipStore, clip_ids: torch.Tensor, cur_idx: torch.Tensor) -> torch.Tensor:
    prev_vec, prev_pelvis, prev_payload = ctl.target_state(store, clip_ids, cur_idx - 1)
    cur_vec, cur_pelvis, cur_payload = ctl.target_state(store, clip_ids, cur_idx)
    inp = ctl.build_controller_input(
        store, clip_ids, cur_idx, prev_vec, cur_vec, prev_pelvis, cur_pelvis, prev_payload, cur_payload
    )
    target_vec = store.get_target_output(clip_ids, cur_idx + 1)
    return torch.cat((inp, target_vec), dim=-1)


def initial_temporal_context(
    store: ctl.SimpleClipStore,
    clip_ids: torch.Tensor,
    cur_idx: torch.Tensor,
    frames: int,
) -> torch.Tensor:
    frames = max(1, int(frames))
    if frames <= 1:
        return torch.empty((clip_ids.shape[0], 0, 0), dtype=torch.float32, device=store.device)
    rows = [gt_transition_feature(store, clip_ids, cur_idx + offset) for offset in range(-(frames - 1), 0)]
    return torch.stack(rows, dim=1)


def build_temporal_start_pool(store: ctl.SimpleClipStore, rollout_k: int, frames: int) -> ctl.StartPool:
    min_start = max(1, int(frames))
    clip_chunks: list[torch.Tensor] = []
    start_chunks: list[torch.Tensor] = []
    for clip_id, clip in enumerate(store.clips):
        max_start = ctl.max_start_for_clip(clip, store.cfg, rollout_k)
        if max_start < min_start:
            continue
        starts = torch.arange(min_start, max_start + 1, dtype=torch.long, device=store.device)
        clip_chunks.append(torch.full_like(starts, int(clip_id)))
        start_chunks.append(starts)
    if not start_chunks:
        raise ValueError(f"No valid {frames}-frame rollout starts found for K={rollout_k}")
    return ctl.StartPool(torch.cat(clip_chunks, dim=0), torch.cat(start_chunks, dim=0))


def build_temporal_start_pools(store: ctl.SimpleClipStore, rollout_values: tuple[int, ...], frames: int) -> dict[int, ctl.StartPool]:
    return {int(k): build_temporal_start_pool(store, int(k), int(frames)) for k in rollout_values}


def temporal_validation_ae_score(
    model: torch.nn.Module,
    ae: nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    store: ctl.SimpleClipStore,
    rollout_k: int,
    pool: ctl.StartPool,
    schema: dict[str, object],
    max_rows: int,
) -> float:
    with torch.no_grad():
        frames = window_frames(schema)
        clip_ids, starts = ctl.validation_rows(pool, max_rows)
        cur_idx = starts
        prev_vec, prev_pelvis, prev_payload = ctl.target_state(store, clip_ids, cur_idx - 1)
        cur_vec, cur_pelvis, cur_payload = ctl.target_state(store, clip_ids, cur_idx)
        context = initial_temporal_context(store, clip_ids, cur_idx, frames)
        scores: list[torch.Tensor] = []
        for step in range(max(1, int(rollout_k))):
            inp = ctl.build_controller_input(
                store, clip_ids, cur_idx, prev_vec, cur_vec, prev_pelvis, cur_pelvis, prev_payload, cur_payload
            )
            raw = ctl.model_forward(model, inp, cur_vec, store.cfg)
            pred_vec = ctl.clean_output_vector(raw, store)
            raw_feature = torch.cat((inp, pred_vec), dim=-1)
            if frames > 1:
                window_raw = torch.cat((context, raw_feature[:, None, :]), dim=1).reshape(raw_feature.shape[0], -1)
            else:
                window_raw = raw_feature
            x = (window_raw - mean) / std
            scores.append(ae_score_rows(ae, x, schema))
            if step + 1 >= int(rollout_k):
                break
            if frames > 1:
                context = torch.cat((context[:, 1:, :], raw_feature[:, None, :]), dim=1)
            prev_vec = cur_vec
            prev_pelvis = cur_pelvis
            prev_payload = cur_payload
            cur_vec, cur_pelvis, cur_payload = ctl.predicted_state_from_vector(pred_vec, store)
            cur_idx = cur_idx + 1
        return float(torch.cat(scores).mean().detach().cpu()) if scores else 0.0


def temporal_ae_rollout_loss(
    model: torch.nn.Module,
    ae: nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    store: ctl.SimpleClipStore,
    rollout_k: int,
    batch_size: int,
    start_pools: dict[int, ctl.StartPool],
    schema: dict[str, object],
) -> torch.Tensor:
    frames = window_frames(schema)
    max_k = max(1, int(rollout_k))
    original_batch_size = max(1, int(batch_size))
    effective_k = ctl.sample_effective_rollout_k(original_batch_size, max_k, store.device)
    clip_ids, starts = ctl.sample_rollout_rows(start_pools, effective_k)
    cur_idx = starts
    prev_vec, prev_pelvis, prev_payload = ctl.target_state(store, clip_ids, cur_idx - 1)
    cur_vec, cur_pelvis, cur_payload = ctl.target_state(store, clip_ids, cur_idx)
    context = initial_temporal_context(store, clip_ids, cur_idx, frames)
    row_weight = (1.0 / effective_k.float()) / float(original_batch_size)
    total_loss = torch.zeros((), dtype=torch.float32, device=store.device)

    for step in range(max_k):
        inp = ctl.build_controller_input(
            store, clip_ids, cur_idx, prev_vec, cur_vec, prev_pelvis, cur_pelvis, prev_payload, cur_payload
        )
        raw = ctl.model_forward(model, inp, cur_vec, store.cfg)
        pred_vec = ctl.clean_output_vector(raw, store)
        raw_feature = torch.cat((inp, pred_vec), dim=-1)
        if frames > 1:
            window_raw = torch.cat((context, raw_feature[:, None, :]), dim=1).reshape(raw_feature.shape[0], -1)
        else:
            window_raw = raw_feature
        x = (window_raw - mean) / std
        total_loss = total_loss + (ae_score_rows(ae, x, schema) * row_weight).sum()
        if step + 1 >= max_k:
            break

        continuing = effective_k > (step + 1)
        rows = continuing.nonzero(as_tuple=False).flatten()
        if rows.numel() == 0:
            break
        if frames > 1:
            context = torch.cat((context[:, 1:, :], raw_feature[:, None, :]), dim=1).index_select(0, rows)
        pred_vec = pred_vec.index_select(0, rows)
        clip_ids = clip_ids.index_select(0, rows)
        prev_vec = cur_vec.index_select(0, rows)
        prev_pelvis = cur_pelvis.index_select(0, rows)
        prev_payload = cur_payload.index_select(0, rows)
        cur_vec, cur_pelvis, cur_payload = ctl.predicted_state_from_vector(pred_vec, store)
        cur_idx = cur_idx.index_select(0, rows) + 1
        effective_k = effective_k.index_select(0, rows)
        row_weight = row_weight.index_select(0, rows)
    return total_loss


def sample_same_clip_training_starts_min(store: ctl.SimpleClipStore, clip_ids: torch.Tensor, min_start: int) -> torch.Tensor:
    min_start = max(1, int(min_start))
    max_start = ctl.max_training_start_for_clip_ids(store, clip_ids)
    if min_start <= 1:
        return ctl.sample_same_clip_training_starts(store, clip_ids)
    span = (max_start - min_start + 1).clamp_min(1)
    noise = torch.rand(span.shape, dtype=torch.float32, device=store.device)
    starts = torch.floor(noise * span.float()).long() + min_start
    return torch.minimum(starts, max_start).clamp_min(1)


def build_training_start_pool_min(store: ctl.SimpleClipStore, frames: int) -> ctl.StartPool:
    min_start = max(1, int(frames))
    clip_chunks: list[torch.Tensor] = []
    start_chunks: list[torch.Tensor] = []
    for clip_id, clip in enumerate(store.clips):
        max_start = ctl.max_training_start_for_clip(clip, store.cfg)
        if max_start < min_start:
            continue
        starts = torch.arange(min_start, max_start + 1, dtype=torch.long, device=store.device)
        clip_chunks.append(torch.full_like(starts, int(clip_id)))
        start_chunks.append(starts)
    if not start_chunks:
        raise ValueError(f"No valid controller training starts found for frames={frames}")
    return ctl.StartPool(torch.cat(clip_chunks, dim=0), torch.cat(start_chunks, dim=0))


def build_controller_training_start_pools(
    store: ctl.SimpleClipStore,
    rollout_values: tuple[int, ...],
    frames: int,
) -> dict[int, ctl.StartPool]:
    pool = build_training_start_pool_min(store, frames)
    return {int(k): pool for k in rollout_values}


def controller_ae_rollout_loss(
    model: torch.nn.Module,
    ae: nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    store: ctl.SimpleClipStore,
    rollout_k: int,
    batch_size: int,
    start_pools: dict[int, ctl.StartPool],
    schema: dict[str, object],
    pose_noise_amount: float,
) -> torch.Tensor:
    frames = window_frames(schema)
    max_k = max(1, int(rollout_k))
    original_batch_size = max(1, int(batch_size))
    effective_k = ctl.sample_effective_rollout_k(original_batch_size, max_k, store.device)
    clip_ids, starts = ctl.sample_rollout_rows(start_pools, effective_k)
    cur_idx = starts
    prev_vec, prev_pelvis, prev_payload = ctl.target_state(store, clip_ids, cur_idx - 1)
    cur_vec, cur_pelvis, cur_payload = ctl.target_state(store, clip_ids, cur_idx)
    if float(pose_noise_amount) > 0.0:
        cur_vec = ctl.add_pose_noise_to_vector(store, cur_vec, float(pose_noise_amount))
        cur_vec, cur_pelvis, cur_payload = ctl.predicted_state_from_vector(cur_vec, store)
    context = initial_temporal_context(store, clip_ids, cur_idx, frames)
    row_weight = (1.0 / effective_k.float()) / float(original_batch_size)
    total_loss = torch.zeros((), dtype=torch.float32, device=store.device)

    for step in range(max_k):
        inp = ctl.build_controller_input(
            store, clip_ids, cur_idx, prev_vec, cur_vec, prev_pelvis, cur_pelvis, prev_payload, cur_payload
        )
        raw = ctl.model_forward(model, inp, cur_vec, store.cfg)
        pred_vec = ctl.clean_output_vector(raw, store)
        raw_feature = torch.cat((inp, pred_vec), dim=-1)
        if frames > 1:
            window_raw = torch.cat((context, raw_feature[:, None, :]), dim=1).reshape(raw_feature.shape[0], -1)
        else:
            window_raw = raw_feature
        x = (window_raw - mean) / std
        total_loss = total_loss + (ae_score_rows(ae, x, schema) * row_weight).sum()
        if step + 1 >= max_k:
            break

        continuing = effective_k > (step + 1)
        reset = ctl.training_reset_rows(store, clip_ids, cur_idx, continuing)
        advance = continuing & (~reset)
        reset_starts = sample_same_clip_training_starts_min(store, clip_ids, frames)
        reset_prev_vec, reset_prev_pelvis, reset_prev_payload = ctl.target_state(store, clip_ids, reset_starts - 1)
        reset_cur_vec, reset_cur_pelvis, reset_cur_payload = ctl.target_state(store, clip_ids, reset_starts)
        if float(pose_noise_amount) > 0.0:
            reset_cur_vec = ctl.add_pose_noise_to_vector(store, reset_cur_vec, float(pose_noise_amount))
            reset_cur_vec, reset_cur_pelvis, reset_cur_payload = ctl.predicted_state_from_vector(reset_cur_vec, store)
        next_vec, next_pelvis, next_payload = ctl.predicted_state_from_vector(pred_vec, store)

        reset_mask = reset[:, None]
        advance_mask = advance[:, None]
        prev_vec = torch.where(reset_mask, reset_prev_vec, torch.where(advance_mask, cur_vec, prev_vec))
        prev_pelvis = torch.where(reset_mask, reset_prev_pelvis, torch.where(advance_mask, cur_pelvis, prev_pelvis))
        prev_payload = torch.where(reset_mask, reset_prev_payload, torch.where(advance_mask, cur_payload, prev_payload))
        cur_vec = torch.where(reset_mask, reset_cur_vec, torch.where(advance_mask, next_vec, cur_vec))
        cur_pelvis = torch.where(reset_mask, reset_cur_pelvis, torch.where(advance_mask, next_pelvis, cur_pelvis))
        cur_payload = torch.where(reset_mask, reset_cur_payload, torch.where(advance_mask, next_payload, cur_payload))
        cur_idx = torch.where(reset, reset_starts, torch.where(continuing, cur_idx + 1, cur_idx))
        if frames > 1:
            next_context = torch.cat((context[:, 1:, :], raw_feature[:, None, :]), dim=1)
            reset_context = initial_temporal_context(store, clip_ids, reset_starts, frames)
            reset_mask_ctx = reset[:, None, None]
            advance_mask_ctx = advance[:, None, None]
            context = torch.where(reset_mask_ctx, reset_context, torch.where(advance_mask_ctx, next_context, context))
    return total_loss


def train_scratch_controller_k16(
    ae_model: nn.Module,
    clips: list[tl.MotionClip],
    cfg: tl.TrainConfig,
    mean: torch.Tensor,
    std: torch.Tensor,
    schema: dict[str, object],
    device: torch.device,
    steps: int,
    label: str = "",
    pose_noise_amount: float = 0.0,
) -> dict[str, float]:
    store = ctl.SimpleClipStore(clips, cfg, device)
    input_dim, output_dim = tl.make_batch_dims(store.prototype, cfg)
    controller = tl.MLPController(input_dim, output_dim, cfg).to(device)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=CONTROLLER_LR, weight_decay=0.0)
    set_requires_grad(ae_model, False)
    ae_model.eval()
    controller.train()
    schedule = tuple(int(k) for k in CONTROLLER_SCHEDULE)
    base = int(steps) // len(schedule)
    extra = int(steps) % len(schedule)
    step = 0
    last_loss = float("inf")
    frames = window_frames(schema)
    if float(pose_noise_amount) > 0.0:
        print(f"{label} controller_pose_noise={float(pose_noise_amount):.3f}", flush=True)
    for stage_idx, stage_k in enumerate(schedule):
        stage_steps = base + (1 if stage_idx < extra else 0)
        if stage_steps <= 0:
            continue
        print(f"{label} controller_stage K={stage_k} steps={stage_steps} start", flush=True)
        start_pools = build_controller_training_start_pools(store, (stage_k,), frames)
        batch = min(int(CONTROLLER_BATCH), int(start_pools[stage_k].row_count))
        for _ in range(stage_steps):
            step += 1
            loss = controller_ae_rollout_loss(
                controller,
                ae_model,
                mean,
                std,
                store,
                stage_k,
                batch,
                start_pools,
                schema,
                float(pose_noise_amount),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            if step == 1 or step % CONTROLLER_REPORT_EVERY == 0 or step == int(steps):
                print(
                    f"{label} controller_step={step:05d}/{int(steps)} K={stage_k} "
                    f"loss={last_loss:.6g}",
                    flush=True,
                )
        print(f"{label} controller_stage K={stage_k} done step={step} last_loss={last_loss:.6g}", flush=True)
    pool1 = ctl.build_start_pool(store, 1)
    pool16 = ctl.build_start_pool(store, 16)
    print(f"{label} controller_eval K1/K16 global metrics start", flush=True)
    k1 = rollout_global_velocity_metrics(controller, store, 1, pool1, min(EVAL_ROWS, pool1.row_count))
    k16 = rollout_global_velocity_metrics(controller, store, 16, pool16, min(EVAL_ROWS, pool16.row_count))
    if frames > 1:
        ae_pool16 = build_temporal_start_pool(store, 16, frames)
        ae16 = temporal_validation_ae_score(controller, ae_model, mean, std, store, 16, ae_pool16, schema, min(EVAL_ROWS, ae_pool16.row_count))
    else:
        ae16 = ctl.validation_ae_score(controller, ae_model, mean, std, store, 16, pool16)
    print(
        f"{label} controller_eval done k16_pos={k16['k16_global_pos_m']:.6g} "
        f"k16_rot={k16['k16_global_rot_rad']:.6g} k16_vel={k16['k16_global_vel_mps']:.6g} ae16={float(ae16):.6g}",
        flush=True,
    )
    return {
        "scratch_controller_train_steps": float(step),
        "scratch_controller_final_train_loss": float(last_loss),
        "scratch_controller_k16_ae_score": float(ae16),
        "scratch_controller_ae_window_frames": float(frames),
        "scratch_controller_pose_noise": float(pose_noise_amount),
        **{f"scratch_controller_{key}": value for key, value in k1.items()},
        **{f"scratch_controller_{key}": value for key, value in k16.items()},
    }


def optimize_output_fixed_input(
    model: nn.Module,
    x: torch.Tensor,
    schema: dict[str, object],
    init: str,
) -> tuple[torch.Tensor, float]:
    input_dim = int(schema["input_dim"])
    output_dim = int(schema["output_dim"])
    pose_dim = int(schema["pose_dim"])
    controller_input = x[:, :input_dim]
    target = x[:, input_dim:]
    if init == "current":
        value = controller_input[:, :output_dim].clone()
    elif init == "noisy_gt":
        value = target + torch.randn_like(target) * 0.10
    else:
        raise ValueError(init)
    if value.shape[-1] != output_dim:
        value = controller_input[:, :pose_dim].clone()
    value = value.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([value], lr=OUTPUT_OPT_LR)
    for _ in range(OUTPUT_OPT_STEPS):
        feature = torch.cat((controller_input, value), dim=-1)
        recon = model(feature)
        loss = (recon[:, input_dim:] - value).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    mse = float((value.detach() - target).square().mean().detach().cpu())
    return value.detach(), mse


@torch.no_grad()
def output_global_metrics(
    out_norm: torch.Tensor,
    target_norm: torch.Tensor,
    row_clip_ids: torch.Tensor,
    row_cur_indices: torch.Tensor,
    clips: list[tl.MotionClip],
    cfg: tl.TrainConfig,
    mean: torch.Tensor,
    std: torch.Tensor,
    schema: dict[str, object],
    device: torch.device,
) -> dict[str, float]:
    input_dim = int(schema["input_dim"])
    mean_out = mean[input_dim:].to(device)
    std_out = std[input_dim:].to(device)
    pred_out = out_norm * std_out.reshape(1, -1) + mean_out.reshape(1, -1)
    target_out = target_norm * std_out.reshape(1, -1) + mean_out.reshape(1, -1)
    pos_errs: list[torch.Tensor] = []
    rot_errs: list[torch.Tensor] = []
    for clip_id in row_clip_ids.unique().tolist():
        rows = (row_clip_ids == int(clip_id)).nonzero(as_tuple=False).flatten()
        clip = clips[int(clip_id)]
        idx = row_cur_indices.index_select(0, rows).to(device) + 1
        root_pos, root_rot, _yaw, _heading = tl.root_state(clip, idx, cfg, device)
        pred_pose, _ = tl.output_to_pose(pred_out.index_select(0, rows), clip)
        target_pose, _ = tl.output_to_pose(target_out.index_select(0, rows), clip)
        pred_global, pred_rot, _pred_canon = tl.fk_from_pose(clip, root_pos, root_rot, pred_pose, device)
        target_global, target_rot, _target_canon = tl.fk_from_pose(clip, root_pos, root_rot, target_pose, device)
        pos_errs.append((pred_global - target_global).norm(dim=-1).mean(dim=-1))
        rel = pred_rot.transpose(-1, -2) @ target_rot
        trace = rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        angle = torch.acos(torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0))
        rot_errs.append(angle.mean(dim=-1))
    pos = torch.cat(pos_errs)
    rot = torch.cat(rot_errs)
    return {
        "global_pos_m": float(pos.mean().detach().cpu()),
        "global_pos_p95_m": float(torch.quantile(pos, 0.95).detach().cpu()),
        "global_rot_rad": float(rot.mean().detach().cpu()),
        "global_rot_p95_rad": float(torch.quantile(rot, 0.95).detach().cpu()),
    }


@torch.no_grad()
def rollout_global_velocity_metrics(
    model: torch.nn.Module,
    store: ctl.SimpleClipStore,
    rollout_k: int,
    pool: ctl.StartPool,
    max_rows: int,
) -> dict[str, float]:
    clip_ids, starts = ctl.validation_rows(pool, max_rows)
    cur_idx = starts
    prev_vec, prev_pelvis, prev_payload = ctl.target_state(store, clip_ids, cur_idx - 1)
    cur_vec, cur_pelvis, cur_payload = ctl.target_state(store, clip_ids, cur_idx)
    pred_prev_global: torch.Tensor | None = None
    target_prev_global: torch.Tensor | None = None
    pos_errs: list[torch.Tensor] = []
    rot_errs: list[torch.Tensor] = []
    vel_errs: list[torch.Tensor] = []
    pred_speed: list[torch.Tensor] = []
    target_speed: list[torch.Tensor] = []
    for step in range(max(1, int(rollout_k))):
        inp = ctl.build_controller_input(
            store, clip_ids, cur_idx, prev_vec, cur_vec, prev_pelvis, cur_pelvis, prev_payload, cur_payload
        )
        raw = ctl.model_forward(model, inp, cur_vec, store.cfg)
        pred_vec = ctl.clean_output_vector(raw, store)
        pred_pose, _ = tl.output_to_pose(pred_vec, store.prototype)
        target_idx = cur_idx + 1
        root_pos, root_rot, _yaw, _heading = store.root_state(clip_ids, target_idx)
        pred_global = ctl.fk_positions_by_clip(store, clip_ids, root_pos, root_rot, pred_pose)
        pred_rot = torch.empty((clip_ids.shape[0], store.J, 3, 3), dtype=root_pos.dtype, device=store.device)
        target_global = torch.empty_like(pred_global)
        target_rot = torch.empty_like(pred_rot)
        target_pose = store.get_pose(clip_ids, target_idx)
        for clip_id in clip_ids.unique().tolist():
            rows = (clip_ids == int(clip_id)).nonzero(as_tuple=False).flatten()
            p_pos, p_rot, _p_canon = tl.fk_from_pose(
                store.clips[int(clip_id)],
                root_pos.index_select(0, rows),
                root_rot.index_select(0, rows),
                ctl.pose_rows(pred_pose, rows),
                store.device,
            )
            t_pos, t_rot, _t_canon = tl.fk_from_pose(
                store.clips[int(clip_id)],
                root_pos.index_select(0, rows),
                root_rot.index_select(0, rows),
                ctl.pose_rows(target_pose, rows),
                store.device,
            )
            pred_global[rows] = p_pos
            pred_rot[rows] = p_rot
            target_global[rows] = t_pos
            target_rot[rows] = t_rot
        pos_errs.append((pred_global - target_global).norm(dim=-1).mean(dim=-1))
        rel = pred_rot.transpose(-1, -2) @ target_rot
        trace = rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        rot_errs.append(torch.acos(torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)).mean(dim=-1))
        if pred_prev_global is not None and target_prev_global is not None:
            pred_vel = (pred_global - pred_prev_global) * float(store.prototype.fps)
            target_vel = (target_global - target_prev_global) * float(store.prototype.fps)
            vel_errs.append((pred_vel - target_vel).norm(dim=-1).mean(dim=-1))
            pred_speed.append(pred_vel.norm(dim=-1).mean(dim=-1))
            target_speed.append(target_vel.norm(dim=-1).mean(dim=-1))
        pred_prev_global = pred_global
        target_prev_global = target_global
        if step + 1 >= int(rollout_k):
            break
        prev_vec = cur_vec
        prev_pelvis = cur_pelvis
        prev_payload = cur_payload
        cur_vec, cur_pelvis, cur_payload = ctl.predicted_state_from_vector(pred_vec, store)
        cur_idx = target_idx
    pos = torch.cat(pos_errs)
    rot = torch.cat(rot_errs)
    vel = torch.cat(vel_errs) if vel_errs else torch.zeros_like(pos)
    pred_motion = torch.cat(pred_speed) if pred_speed else torch.zeros_like(pos)
    target_motion = torch.cat(target_speed) if target_speed else torch.zeros_like(pos)
    return {
        f"k{int(rollout_k)}_global_pos_m": float(pos.mean().detach().cpu()),
        f"k{int(rollout_k)}_global_pos_p95_m": float(torch.quantile(pos, 0.95).detach().cpu()),
        f"k{int(rollout_k)}_global_rot_rad": float(rot.mean().detach().cpu()),
        f"k{int(rollout_k)}_global_rot_p95_rad": float(torch.quantile(rot, 0.95).detach().cpu()),
        f"k{int(rollout_k)}_global_vel_mps": float(vel.mean().detach().cpu()),
        f"k{int(rollout_k)}_global_vel_p95_mps": float(torch.quantile(vel, 0.95).detach().cpu()),
        f"k{int(rollout_k)}_pred_motion_mps": float(pred_motion.mean().detach().cpu()),
        f"k{int(rollout_k)}_target_motion_mps": float(target_motion.mean().detach().cpu()),
        f"k{int(rollout_k)}_motion_ratio": float(
            (pred_motion.mean() / target_motion.mean().clamp_min(1e-8)).detach().cpu()
        ),
    }


@torch.no_grad()
def synthetic_metrics(model: nn.Module, x: torch.Tensor, schema: dict[str, object]) -> dict[str, float]:
    model.eval()
    device = x.device
    dtype = x.dtype
    out_mask = output_mask(schema, device, dtype).reshape(1, -1)
    root_mask = torch.zeros_like(out_mask)
    input_root_start = int(schema["input_root_start"])
    input_root_end = int(schema["input_root_end"])
    output_start = int(schema["target_output_start"])
    output_end = int(schema["target_output_end"])
    pose_dim = int(schema["pose_dim"])
    output_dim = int(schema["output_dim"])
    for offset in frame_offsets(schema):
        root_mask[:, offset + input_root_start : offset + input_root_end] = 1.0

    slight = x + torch.randn_like(x) * 0.05 * (1.0 - root_mask) * (1.0 - out_mask)
    perm = torch.randperm(x.shape[0], device=device)
    shuffled = x.clone()
    shuffled_src = x.index_select(0, perm)
    statue = x.clone()
    for offset in frame_offsets(schema):
        out_slice = slice(offset + output_start, offset + output_start + output_dim)
        shuffled[:, out_slice] = shuffled_src[:, out_slice]
        statue[:, out_slice] = x[:, offset : offset + pose_dim]
    noise = torch.randn_like(x)
    noise = torch.where(root_mask.bool(), x, noise)
    tiers = {
        "tier1_clean": x,
        "tier2_slight": slight,
        "tier3_bad_statue": statue,
        "tier3_bad_shuffle_output": shuffled,
        "tier4_noise": noise,
    }
    means: dict[str, float] = {}
    for name, tier in tiers.items():
        values = row_output_mse(model, tier, schema)
        means[f"{name}_mean"] = float(values.mean().detach().cpu())
        means[f"{name}_p95"] = float(torch.quantile(values, 0.95).detach().cpu())
    clean = max(means["tier1_clean_mean"], 1e-12)
    for key in list(means):
        if key.endswith("_mean") and key != "tier1_clean_mean":
            means[f"{key[:-5]}_over_clean"] = means[key] / clean
    model.train()
    return means


def evaluate_variant(
    model: nn.Module,
    x: torch.Tensor,
    eval_rows: torch.Tensor,
    clip_ids: torch.Tensor,
    cur_indices: torch.Tensor,
    clips: list[tl.MotionClip],
    cfg: tl.TrainConfig,
    mean: torch.Tensor,
    std: torch.Tensor,
    schema: dict[str, object],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    probe = x.index_select(0, eval_rows)
    probe_clip = clip_ids.index_select(0, eval_rows).to(device)
    probe_idx = cur_indices.index_select(0, eval_rows).to(device)
    if window_frames(schema) > 1:
        clean_out = row_output_mse(model, probe, schema)
        metrics = {
            "clean_output_mse": float(clean_out.mean().detach().cpu()),
            "current_before_output_mse": float("nan"),
            "current_after_output_mse": float("nan"),
            "current_improvement_ratio": float("nan"),
            "current_after_global_pos_m": float("nan"),
            "current_after_global_rot_rad": float("nan"),
            "noisy_before_output_mse": float("nan"),
            "noisy_after_output_mse": float("nan"),
            "noisy_improvement_ratio": float("nan"),
            "noisy_after_global_pos_m": float("nan"),
            "noisy_after_global_rot_rad": float("nan"),
        }
        metrics.update(synthetic_metrics(model, probe, schema))
        return metrics
    input_dim = int(schema["input_dim"])
    target = probe[:, input_dim:]
    current_init = probe[:, : int(schema["output_dim"])]
    noisy_init = target + torch.randn_like(target) * 0.10
    clean_out = row_output_mse(model, probe, schema)
    current_before_mse = float((current_init - target).square().mean().detach().cpu())
    noisy_before_mse = float((noisy_init - target).square().mean().detach().cpu())
    opt_current, current_after_mse = optimize_output_fixed_input(model, probe, schema, "current")
    opt_noisy, noisy_after_mse = optimize_output_fixed_input(model, probe, schema, "noisy_gt")
    current_global = output_global_metrics(opt_current, target, probe_clip, probe_idx, clips, cfg, mean, std, schema, device)
    noisy_global = output_global_metrics(opt_noisy, target, probe_clip, probe_idx, clips, cfg, mean, std, schema, device)
    metrics = {
        "clean_output_mse": float(clean_out.mean().detach().cpu()),
        "current_before_output_mse": current_before_mse,
        "current_after_output_mse": current_after_mse,
        "current_improvement_ratio": current_before_mse / max(current_after_mse, 1e-12),
        "current_after_global_pos_m": current_global["global_pos_m"],
        "current_after_global_rot_rad": current_global["global_rot_rad"],
        "noisy_before_output_mse": noisy_before_mse,
        "noisy_after_output_mse": noisy_after_mse,
        "noisy_improvement_ratio": noisy_before_mse / max(noisy_after_mse, 1e-12),
        "noisy_after_global_pos_m": noisy_global["global_pos_m"],
        "noisy_after_global_rot_rad": noisy_global["global_rot_rad"],
    }
    metrics.update(synthetic_metrics(model, probe, schema))
    return metrics


def save_table(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, object]], metadata: dict[str, object]) -> None:
    baseline = next((row for row in rows if row.get("variant") == BASELINE_VARIANT), rows[0] if rows else None)
    baseline_pos = float(baseline.get("scratch_controller_k16_global_pos_m", 0.0)) if baseline else 0.0
    baseline_rot = float(baseline.get("scratch_controller_k16_global_rot_rad", 0.0)) if baseline else 0.0
    baseline_vel = float(baseline.get("scratch_controller_k16_global_vel_mps", 0.0)) if baseline else 0.0
    ranked = sorted(
        rows,
        key=lambda r: (
            float(r.get("scratch_controller_k16_global_pos_m", r["current_after_global_pos_m"])),
            float(r.get("scratch_controller_k16_global_rot_rad", r["current_after_global_rot_rad"])),
            float(r.get("scratch_controller_k16_global_vel_mps", 0.0)),
        ),
    )
    lines = [
        "# IK AE Research Report",
        "",
        str(metadata.get("rules", "No real rollout negatives; no contact labels.")),
        f"Reference baseline: `{BASELINE_VARIANT}` is the current one-frame simple AE recipe, trained from scratch on this subset with the same row budget as the sweep.",
        "",
        "## Dataset",
        "",
        "AE training clips:",
    ]
    for item in metadata["clips"]:  # type: ignore[index]
        lines.append(f"- {item}")
    lines.extend(["", "Controller probe clips:"])
    for item in metadata.get("controller_clips", []):  # type: ignore[assignment]
        lines.append(f"- {item}")
    lines.extend(["", "## Ranking", ""])
    lines.append(
        "| rank | variant | K16 pos m | d pos vs simple | K16 rot rad | d rot | K16 vel m/s | d vel | motion ratio | K1 pos m | K16 AE | clean out mse | statue/clean |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for rank, row in enumerate(ranked, 1):
        pos = float(row.get("scratch_controller_k16_global_pos_m", 0.0))
        rot = float(row.get("scratch_controller_k16_global_rot_rad", 0.0))
        vel = float(row.get("scratch_controller_k16_global_vel_mps", 0.0))
        lines.append(
            f"| {rank} | {row['variant']} | {pos:.6f} | {pos - baseline_pos:+.6f} | "
            f"{rot:.6f} | {rot - baseline_rot:+.6f} | "
            f"{vel:.6f} | {vel - baseline_vel:+.6f} | "
            f"{float(row.get('scratch_controller_k16_motion_ratio', 0.0)):.3f} | "
            f"{float(row.get('scratch_controller_k1_global_pos_m', 0.0)):.6f} | "
            f"{float(row.get('scratch_controller_k16_ae_score', 0.0)):.6g} | "
            f"{float(row['clean_output_mse']):.6g} | {float(row.get('tier3_bad_statue_over_clean', 0.0)):.2f} |"
        )
    best = ranked[0]
    lines.extend(
        [
            "",
            "## Short Take",
            "",
            f"Best by fresh scratch controller K=16 global position error: `{best['variant']}`.",
            f"Current simple AE baseline K16: pos={baseline_pos:.6f} m, rot={baseline_rot:.6f} rad, vel={baseline_vel:.6f} m/s.",
            "Prefer a winner that improves K16 global position, rotation, and velocity without losing synthetic bad-case separation.",
            "",
            "Full numeric results are in `ae_research_results.csv` and `ae_research_results.json`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_variant_suite(
    variants_to_run: list[Variant],
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    schema: dict[str, object],
    train_rows: torch.Tensor,
    val_rows: torch.Tensor,
    eval_rows: torch.Tensor,
    clip_ids: torch.Tensor,
    cur_indices: torch.Tensor,
    clips: list[tl.MotionClip],
    controller_clips: list[tl.MotionClip],
    cfg: tl.TrainConfig,
    device: torch.device,
    writer: SummaryWriter,
    run_dir: Path,
    run_id: str,
    metadata: dict[str, object],
    rows: list[dict[str, object]],
    start_time: float,
    controller_steps: int,
    controller_pose_noise: float,
) -> None:
    for local_idx, variant in enumerate(variants_to_run, 1):
        variant_idx = len(rows) + 1
        model = ResearchAE(x.shape[1], variant).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(variant.lr), weight_decay=float(variant.weight_decay))
        best = float("inf")
        print(
            f"variant {variant_idx} local={local_idx}/{len(variants_to_run)} window={window_frames(schema)} "
            f"{variant.name} {asdict(variant)}",
            flush=True,
        )
        for step in range(1, int(variant.steps) + 1):
            lr = sae.lr_for_step(step, int(variant.steps), float(variant.lr))
            sae.set_lr(optimizer, lr)
            batch_rows = sae.batch_indices(train_rows, 512)
            clean = x.index_select(0, batch_rows)
            corrupt = make_corrupted_input(clean, schema, variant)
            recon = model(corrupt)
            loss = variant_loss(recon, clean, schema, variant)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == 1 or step % REPORT_EVERY == 0 or step == int(variant.steps):
                with torch.no_grad():
                    model.eval()
                    val = x.index_select(0, val_rows)
                    val_score = float(row_output_mse(model, val, schema).mean().detach().cpu())
                    model.train()
                best = min(best, val_score)
                writer.add_scalar(f"variant/{variant.name}/val_output_mse", val_score, step)
                writer.add_scalar(f"variant/{variant.name}/loss", float(loss.detach().cpu()), step)
                print(
                    f"{variant.name} step={step:05d} loss={float(loss.detach().cpu()):.6g} "
                    f"val_out={val_score:.6g} best={best:.6g} elapsed_s={time.perf_counter() - start_time:.1f}",
                    flush=True,
                )
        metrics = evaluate_variant(model, x, eval_rows, clip_ids, cur_indices, clips, cfg, mean, std, schema, device)
        metrics.update(
            train_scratch_controller_k16(
                model,
                controller_clips,
                cfg,
                mean,
                std,
                schema,
                device,
                int(controller_steps),
                variant.name,
                float(controller_pose_noise),
            )
        )
        row = {"variant": variant.name, "window_frames": window_frames(schema), **asdict(variant), **metrics}
        rows.append(row)
        for key, value in metrics.items():
            if isinstance(value, (float, int)) and math.isfinite(float(value)):
                writer.add_scalar(f"final/{key}", float(value), variant_idx)
        torch.save(
            {
                "kind": "ik_ae_research_variant",
                "variant": asdict(variant),
                "window_frames": window_frames(schema),
                "model": model.state_dict(),
                "schema": schema,
                "locomotion_config": asdict(cfg),
                "mean": mean.detach().cpu(),
                "std": std.detach().cpu(),
                "metadata": metadata,
                "metrics": metrics,
            },
            checkpoint_path(run_dir, run_id, f"{variant.name}_last"),
        )
        save_table(run_dir / "ae_research_results.csv", rows)
        (run_dir / "ae_research_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        write_report(run_dir / "REPORT.md", rows, metadata)
        print(
            f"RESULT {variant.name}: k16_pos={metrics['scratch_controller_k16_global_pos_m']:.6g} "
            f"k16_rot={metrics['scratch_controller_k16_global_rot_rad']:.6g} "
            f"k16_vel={metrics['scratch_controller_k16_global_vel_mps']:.6g} "
            f"k16_motion_ratio={metrics.get('scratch_controller_k16_motion_ratio', float('nan')):.4g} "
            f"clean_out={metrics['clean_output_mse']:.6g}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle-only sweep for IK AE variants.")
    parser.add_argument("--run-label", default="kaggle_ae_research")
    parser.add_argument("--periodic-names", default=";".join(DEFAULT_PERIODIC))
    parser.add_argument("--nonperiodic-names", default=";".join(DEFAULT_NONPERIODIC))
    parser.add_argument("--controller-periodic-names", default=";".join(DEFAULT_CONTROLLER_PERIODIC))
    parser.add_argument("--controller-nonperiodic-names", default=";".join(DEFAULT_CONTROLLER_NONPERIODIC))
    parser.add_argument("--controller-pose-noise", type=float, default=0.0)
    parser.add_argument("--train-steps", type=int, default=4000)
    parser.add_argument("--controller-steps", type=int, default=CONTROLLER_STEPS)
    parser.add_argument("--max-variants", type=int, default=0)
    parser.add_argument("--variant-names", default="")
    parser.add_argument("--eval-rows", type=int, default=EVAL_ROWS)
    parser.add_argument("--window-frames", type=int, default=1)
    parser.add_argument("--skip-one-frame-baseline", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(1234)

    cfg = sae.make_locomotion_cfg(device)
    specs = resolve_subset(args.periodic_names, args.nonperiodic_names)
    clips = sae.load_clips(specs, cfg)
    controller_specs = resolve_named_subset(
        args.controller_periodic_names,
        args.controller_nonperiodic_names,
        DEFAULT_CONTROLLER_PERIODIC,
        DEFAULT_CONTROLLER_NONPERIODIC,
    )
    controller_clips = sae.load_clips(controller_specs, cfg)
    raw, clip_ids_cpu, cur_indices_cpu, base_schema = sae.collect_controller_features(clips, cfg, device)
    single_schema = temporal_schema(base_schema, 1)
    raw_window, clip_ids_window_cpu, cur_indices_window_cpu, schema = build_temporal_features(
        raw, clip_ids_cpu, cur_indices_cpu, base_schema, int(args.window_frames)
    )
    x_cpu, mean_cpu, std_cpu = sae.normalize_features(raw_window, sae.STD_FLOOR)
    x = x_cpu.to(device)
    mean = mean_cpu.to(device)
    std = std_cpu.to(device)
    clip_ids = clip_ids_window_cpu.to(device)
    cur_indices = cur_indices_window_cpu.to(device)
    train_rows, val_rows = sae.split_rows(x.shape[0], 0.1, 1234, device)
    eval_count = min(int(args.eval_rows), int(x.shape[0]))
    eval_rows = torch.randperm(x.shape[0], device=device)[:eval_count]

    run_id = ik_run_id(args.run_label)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"), flush_secs=1)
    all_variants = select_variants(variants(args.train_steps, window_frames(schema)), args.variant_names, int(args.max_variants))
    metadata = {
        "clips": [f"{path.name} cyclic={cyclic}" for path, cyclic in specs],
        "controller_clips": [f"{path.name} cyclic={cyclic}" for path, cyclic in controller_specs],
        "controller_pose_noise": float(args.controller_pose_noise),
        "row_count": int(x.shape[0]),
        "single_frame_row_count": int(raw.shape[0]),
        "window_frames": window_frames(schema),
        "eval_rows": int(eval_count),
        "variant_count": int(len(all_variants)),
        "includes_one_frame_baseline": bool(window_frames(schema) > 1 and not args.skip_one_frame_baseline),
        "rules": (
            f"AE sees {window_frames(schema)} consecutive transition row(s) from the AE dataset; "
            f"probe controller trains only on controller_clips with pose noise {float(args.controller_pose_noise):.3f}; "
            "no real rollout negatives; each probe controller starts from scratch and trains up to K16"
        ),
        "schema": schema,
        "single_frame_schema": single_schema,
    }
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"ae_research run={run_id} rows={x.shape[0]} dim={x.shape[1]} variants={len(all_variants)}", flush=True)

    rows: list[dict[str, object]] = []
    start_time = time.perf_counter()
    if window_frames(schema) > 1 and not args.skip_one_frame_baseline:
        base_x_cpu, base_mean_cpu, base_std_cpu = sae.normalize_features(raw, sae.STD_FLOOR)
        base_x = base_x_cpu.to(device)
        base_mean = base_mean_cpu.to(device)
        base_std = base_std_cpu.to(device)
        base_clip_ids = clip_ids_cpu.to(device)
        base_cur_indices = cur_indices_cpu.to(device)
        base_train_rows, base_val_rows = sae.split_rows(base_x.shape[0], 0.1, 1234, device)
        base_eval_rows = torch.randperm(base_x.shape[0], device=device)[: min(int(args.eval_rows), int(base_x.shape[0]))]
        print("running one-frame current_simple_ae baseline before temporal variants", flush=True)
        train_variant_suite(
            variants(args.train_steps, 1)[:1],
            base_x,
            base_mean,
            base_std,
            single_schema,
            base_train_rows,
            base_val_rows,
            base_eval_rows,
            base_clip_ids,
            base_cur_indices,
            clips,
            controller_clips,
            cfg,
            device,
            writer,
            run_dir,
            run_id,
            metadata,
            rows,
            start_time,
            int(args.controller_steps),
            float(args.controller_pose_noise),
        )
    train_variant_suite(
        all_variants,
        x,
        mean,
        std,
        schema,
        train_rows,
        val_rows,
        eval_rows,
        clip_ids,
        cur_indices,
        clips,
        controller_clips,
        cfg,
        device,
        writer,
        run_dir,
        run_id,
        metadata,
        rows,
        start_time,
        int(args.controller_steps),
        float(args.controller_pose_noise),
    )
    writer.close()
    write_report(run_dir / "REPORT.md", rows, metadata)
    print(f"report={run_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
