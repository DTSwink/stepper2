from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

try:
    from . import ik_core as tl
    from . import transition_autoencoder as tae
except ImportError:
    import ik_core as tl
    import transition_autoencoder as tae


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_BASE_PRIOR = (
    PROJECT_ROOT
    / "training"
    / "runs"
    / "20260516_022731_ae_poseaware_hybrid_canonbasis_refresh"
    / "checkpoints"
    / "checkpoint_best.pt"
)
DEFAULT_COMPARE_MODEL = (
    PROJECT_ROOT
    / "training"
    / "runs"
    / "20260517_031133_hybrid_footmotionae_k220_only_synth050_slide0_yaw0_firstframe_from_k32"
    / "checkpoints"
    / "checkpoint_best.pt"
)
DEFAULT_COMPARE_FOLDER = (
    PROJECT_ROOT
    / "ue5"
    / "animations_transitions_only_full_trimmed_turn_in_place"
    / "npz_final"
)


@dataclass
class WindowAEConfig:
    base_prior_checkpoint: str = str(DEFAULT_BASE_PRIOR)
    model_checkpoint: str = str(DEFAULT_COMPARE_MODEL)
    compare_folder_path: str = str(DEFAULT_COMPARE_FOLDER)
    compare_glob: str = "M_Neutral_Stand_Turn_090_*.npz"
    run_name: str = "window_transition_ae"
    date_prefix_run_name: bool = True
    output_dir: str = "training/runs"
    window_sizes: str = "8,16,32"
    latent_dim: int = 128
    hidden_dim: int = 512
    num_hidden_layers: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    max_epochs: int = 240
    stall_patience_epochs: int = 45
    min_delta: float = 1e-6
    input_noise_std: float = 0.0
    input_noise_mask: str = "none"
    seed: int = 1234
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    feature_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    collect_batch_size: int = 2048
    num_workers: int = 0
    max_train_windows: int = 0
    train_stride: int = 1
    compare_max_frames: int = 0
    feature_cache_path: str = "training/runs/cache/window_transition_features.pt"
    conditional_root: bool = False
    anchor_first_root: bool = False


class WindowAutoencoder(nn.Module):
    def __init__(self, dim: int, cfg: WindowAEConfig):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = dim
        for _ in range(cfg.num_hidden_layers):
            layers += [nn.Linear(in_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.GELU()]
            in_dim = cfg.hidden_dim
        layers += [nn.Linear(in_dim, cfg.latent_dim), nn.GELU()]
        in_dim = cfg.latent_dim
        for _ in range(cfg.num_hidden_layers):
            layers += [nn.Linear(in_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.GELU()]
            in_dim = cfg.hidden_dim
        layers += [nn.Linear(in_dim, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def target(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ConditionalWindowPredictor(nn.Module):
    def __init__(
        self,
        full_dim: int,
        condition_indices: torch.Tensor,
        target_indices: torch.Tensor,
        cfg: WindowAEConfig,
    ):
        super().__init__()
        self.full_dim = int(full_dim)
        self.register_buffer("condition_indices", condition_indices.long())
        self.register_buffer("target_indices", target_indices.long())
        layers: list[nn.Module] = []
        in_dim = int(condition_indices.numel())
        for _ in range(cfg.num_hidden_layers):
            layers += [nn.Linear(in_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.GELU()]
            in_dim = cfg.hidden_dim
        layers += [nn.Linear(in_dim, cfg.latent_dim), nn.GELU()]
        in_dim = cfg.latent_dim
        for _ in range(cfg.num_hidden_layers):
            layers += [nn.Linear(in_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.GELU()]
            in_dim = cfg.hidden_dim
        layers += [nn.Linear(in_dim, int(target_indices.numel()))]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.index_select(-1, self.condition_indices))

    def target(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(-1, self.target_indices)


class WindowDataset(Dataset):
    def __init__(self, sequences: list[torch.Tensor], window_size: int, stride: int = 1, max_windows: int = 0, seed: int = 0):
        self.sequences = sequences
        self.window_size = int(window_size)
        stride = max(1, int(stride))
        starts: list[tuple[int, int]] = []
        for ci, seq in enumerate(sequences):
            max_start = int(seq.shape[0]) - self.window_size
            if max_start < 0:
                continue
            starts.extend((ci, start) for start in range(0, max_start + 1, stride))
        if max_windows > 0 and len(starts) > max_windows:
            rng = random.Random(seed + self.window_size)
            starts = rng.sample(starts, max_windows)
        if not starts:
            raise ValueError(f"No feature windows available for W={self.window_size}")
        self.starts = starts

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> torch.Tensor:
        ci, start = self.starts[index]
        return self.sequences[ci][start : start + self.window_size].reshape(-1)


class TensorWindowDataset(Dataset):
    def __init__(self, windows: torch.Tensor):
        if windows.ndim != 2 or windows.shape[0] <= 0:
            raise ValueError("TensorWindowDataset needs a non-empty [N, D] tensor.")
        self.windows = windows.contiguous()

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.windows[index]


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_config_dict(cfg: tl.TrainConfig, values: dict) -> None:
    valid = cfg.__dataclass_fields__
    for key, value in values.items():
        if key not in valid:
            continue
        current = getattr(cfg, key)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(cfg, key, value)


def load_base_prior(path: Path, device: torch.device) -> tuple[tae.TransitionAutoencoder, tl.TrainConfig, dict, torch.Tensor, torch.Tensor, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ae_cfg = tae.AEConfig()
    for key, value in ckpt.get("config", {}).items():
        if hasattr(ae_cfg, key):
            setattr(ae_cfg, key, value)
    schema = dict(ckpt["schema"])
    model = tae.TransitionAutoencoder(schema["total_dim"], ae_cfg, schema).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    allowed_missing = {"reconstruction_weights"}
    if unexpected or any(key not in allowed_missing for key in missing):
        raise RuntimeError(f"Could not load base prior {path}: missing={missing}, unexpected={unexpected}")
    model.eval()

    locomotion_cfg = tl.TrainConfig()
    apply_config_dict(locomotion_cfg, ckpt.get("locomotion_config", {}))
    locomotion_cfg.include_transition_foot_motion = bool(
        ckpt.get("locomotion_config", {}).get("include_transition_foot_motion", schema.get("transition_foot_motion_dim", 0) > 0)
    )
    locomotion_cfg.foot_slide_scale_mps = float(ckpt.get("locomotion_config", {}).get("foot_slide_scale_mps", 1.0))
    locomotion_cfg.transition_yaw_scale_radps = float(ckpt.get("locomotion_config", {}).get("transition_yaw_scale_radps", 10.0))

    mean = ckpt["mean"].float().to(device)
    std = ckpt["std"].float().to(device)
    return model, locomotion_cfg, schema, mean, std, ckpt


def prior_clip_specs(prior_ckpt: dict, fallback_folder: str = "") -> list[tuple[Path, bool | None]]:
    specs = []
    for item in prior_ckpt.get("metadata", {}).get("npz_folders", []):
        specs.append((resolve_path(item["path"]), bool(item.get("cyclic", False))))
    if specs:
        return specs
    return tl.clip_specs_from_folders(fallback_folder or "data/npz_final", None, None)


def feature_cache_payload_matches(payload: dict, specs: list[tuple[Path, bool | None]], base_prior: Path, schema_total: int) -> bool:
    meta = payload.get("metadata", {})
    expected = [{"path": str(path), "cyclic": cyclic} for path, cyclic in specs]
    return (
        meta.get("base_prior_checkpoint") == str(base_prior)
        and meta.get("schema_total_dim") == int(schema_total)
        and meta.get("npz_folders") == expected
    )


@torch.no_grad()
def collect_normalized_sequences(
    specs: list[tuple[Path, bool | None]],
    cfg: tl.TrainConfig,
    mean_cpu: torch.Tensor,
    std_cpu: torch.Tensor,
    device: torch.device,
    collect_batch_size: int,
) -> tuple[list[torch.Tensor], list[dict]]:
    clips = tl.load_clips_from_specs(specs, cfg)
    mean = mean_cpu.to(device)
    std = std_cpu.to(device)
    sequences: list[torch.Tensor] = []
    clip_meta: list[dict] = []
    for clip in clips:
        stop = clip.cyclic_period if clip.cyclic_animation else clip.T - cfg.future_window
        if stop <= 1:
            continue
        rows = []
        for start in range(1, stop, collect_batch_size):
            end = min(stop, start + collect_batch_size)
            idx = torch.arange(start, end, dtype=torch.long, device=device)
            raw = tae.clean_transition_features(clip, idx, cfg, device)
            rows.append(((raw - mean) / std).detach().cpu())
        seq = torch.cat(rows, dim=0)
        if seq.shape[0] > 0:
            sequences.append(seq.contiguous())
            clip_meta.append(
                {
                    "path": str(clip.path),
                    "name": clip.path.stem,
                    "cyclic": bool(clip.cyclic_animation),
                    "feature_rows": int(seq.shape[0]),
                }
            )
    if not sequences:
        raise ValueError("No normalized transition feature sequences were collected.")
    return sequences, clip_meta


def load_or_collect_sequences(
    cfg: WindowAEConfig,
    specs: list[tuple[Path, bool | None]],
    locomotion_cfg: tl.TrainConfig,
    mean_cpu: torch.Tensor,
    std_cpu: torch.Tensor,
    schema_total: int,
    base_prior: Path,
) -> tuple[list[torch.Tensor], list[dict]]:
    cache_path = resolve_path(cfg.feature_cache_path)
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if feature_cache_payload_matches(payload, specs, base_prior, schema_total):
            print(f"loaded window AE feature cache {cache_path}", flush=True)
            return payload["sequences"], payload["clip_meta"]
    feature_device = torch.device(cfg.feature_device)
    tl.apply_cuda_performance_settings(locomotion_cfg, feature_device)
    print(f"collecting window AE features on {feature_device} from {len(specs)} folder specs", flush=True)
    sequences, clip_meta = collect_normalized_sequences(
        specs,
        locomotion_cfg,
        mean_cpu,
        std_cpu,
        feature_device,
        cfg.collect_batch_size,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sequences": sequences,
            "clip_meta": clip_meta,
            "metadata": {
                "base_prior_checkpoint": str(base_prior),
                "schema_total_dim": int(schema_total),
                "npz_folders": [{"path": str(path), "cyclic": cyclic} for path, cyclic in specs],
            },
        },
        cache_path,
    )
    print(f"saved window AE feature cache {cache_path}", flush=True)
    return sequences, clip_meta


def rotate_vectors_between_headings(values: torch.Tensor, from_yaw: torch.Tensor, anchor_yaw: torch.Tensor) -> torch.Tensor:
    from_heading = tl.yaw_to_row_matrix(from_yaw)
    anchor_heading = tl.yaw_to_row_matrix(anchor_yaw)
    rot = from_heading.transpose(-1, -2) @ anchor_heading
    return torch.matmul(values, rot)


def anchor_local_positions(
    values: torch.Tensor,
    frame_pos: torch.Tensor,
    frame_yaw: torch.Tensor,
    anchor_pos: torch.Tensor,
    anchor_yaw: torch.Tensor,
) -> torch.Tensor:
    anchor_heading = tl.yaw_to_row_matrix(anchor_yaw)
    root_offset = torch.matmul((frame_pos - anchor_pos).unsqueeze(1), anchor_heading).squeeze(1)
    local = rotate_vectors_between_headings(values, frame_yaw, anchor_yaw)
    return local + root_offset[:, None, :]


@torch.no_grad()
def transform_transition_feature_to_anchor(
    raw: torch.Tensor,
    clip: tl.MotionClip,
    prev_idx: torch.Tensor,
    cur_idx: torch.Tensor,
    next_idx: torch.Tensor,
    anchor_idx: torch.Tensor,
    cfg: tl.TrainConfig,
    schema: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    out = raw.clone()
    b = raw.shape[0]
    j_count = int(schema["next_canon_dim"]) // 3
    pose_dim = int(schema["pose_dim"])
    velocity_dim = int(schema["velocity_dim"])
    input_dim = int(schema["input_dim"])
    output_dim = int(schema["output_dim"])
    next_canon_start = int(schema["next_canon_start"])
    next_velocity_start = int(schema["next_velocity_start"])

    anchor_pos, _anchor_rot, anchor_yaw, anchor_heading = tl.root_state(clip, anchor_idx, cfg, device)

    def root_info(idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos, _rot, yaw, _heading = tl.root_state(clip, idx, cfg, device)
        return pos, yaw

    def anchor_pose_positions(base: int, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frame_pos, frame_yaw = root_info(idx)
        pelvis_local = raw[:, base : base + 3].reshape(b, 1, 3)
        pelvis_anchor = anchor_local_positions(pelvis_local, frame_pos, frame_yaw, anchor_pos, anchor_yaw).reshape(b, 3)
        canon_start = base + 9
        canon_end = canon_start + j_count * 3
        canon_local = raw[:, canon_start:canon_end].reshape(b, j_count, 3)
        canon_anchor = anchor_local_positions(canon_local, frame_pos, frame_yaw, anchor_pos, anchor_yaw)
        out[:, base : base + 3] = pelvis_anchor
        out[:, canon_start:canon_end] = canon_anchor.reshape(b, j_count * 3)
        return pelvis_anchor, canon_anchor

    prev_pelvis, prev_canon = anchor_pose_positions(pose_dim, prev_idx)
    cur_pelvis, cur_canon = anchor_pose_positions(0, cur_idx)

    velocity_start = pose_dim * 2
    out[:, velocity_start : velocity_start + 3] = (cur_pelvis - prev_pelvis) / cfg.pose_delta_scale_final
    out[:, velocity_start + 3 : velocity_start + velocity_dim] = (
        (cur_canon - prev_canon).reshape(b, j_count * 3) / cfg.pose_delta_scale_final
    )

    prev_root_pos, prev_root_yaw = root_info(prev_idx)
    cur_root_pos, cur_root_yaw = root_info(cur_idx)
    root_start = velocity_start + velocity_dim
    root_delta_anchor = torch.matmul((cur_root_pos - prev_root_pos).unsqueeze(1), anchor_heading).squeeze(1)
    out[:, root_start] = root_delta_anchor[:, 0] / cfg.max_speed_scale_final
    out[:, root_start + 1] = root_delta_anchor[:, 2] / cfg.max_speed_scale_final
    out[:, root_start + 2] = tl.wrap_angle(cur_root_yaw - prev_root_yaw) / cfg.max_turn_rate_scale_final

    future_start = root_start + 3
    for k in range(1, cfg.future_window + 1):
        fut_idx = cur_idx + k
        fut_pos, fut_yaw = root_info(fut_idx)
        fut_anchor = torch.matmul((fut_pos - anchor_pos).unsqueeze(1), anchor_heading).squeeze(1)
        horizon_frames = (fut_idx - anchor_idx).to(dtype=raw.dtype).clamp_min(1.0)
        scale = horizon_frames * cfg.max_speed_scale_final
        offset = future_start + (k - 1) * 4
        out[:, offset] = torch.clamp(fut_anchor[:, 0] / scale, -2.0, 2.0)
        out[:, offset + 1] = torch.clamp(fut_anchor[:, 2] / scale, -2.0, 2.0)
        dyaw = tl.wrap_angle(fut_yaw - anchor_yaw)
        out[:, offset + 2] = torch.cos(dyaw)
        out[:, offset + 3] = torch.sin(dyaw)

    next_output_start = input_dim
    next_root_pos, next_root_yaw = root_info(next_idx)
    next_pelvis_local = raw[:, next_output_start : next_output_start + 3].reshape(b, 1, 3)
    next_pelvis = anchor_local_positions(
        next_pelvis_local,
        next_root_pos,
        next_root_yaw,
        anchor_pos,
        anchor_yaw,
    ).reshape(b, 3)
    out[:, next_output_start : next_output_start + 3] = next_pelvis

    next_canon = raw[:, next_canon_start : next_canon_start + j_count * 3].reshape(b, j_count, 3)
    next_canon_anchor = anchor_local_positions(next_canon, next_root_pos, next_root_yaw, anchor_pos, anchor_yaw)
    out[:, next_canon_start : next_canon_start + j_count * 3] = next_canon_anchor.reshape(b, j_count * 3)
    out[:, next_velocity_start : next_velocity_start + 3] = (next_pelvis - cur_pelvis) / cfg.pose_delta_scale_final
    out[:, next_velocity_start + 3 : next_velocity_start + velocity_dim] = (
        (next_canon_anchor - cur_canon).reshape(b, j_count * 3) / cfg.pose_delta_scale_final
    )
    return out


@torch.no_grad()
def anchored_windows_from_poses(
    clip: tl.MotionClip,
    poses: list[dict[str, torch.Tensor]],
    cfg: tl.TrainConfig,
    schema: dict[str, int],
    mean: torch.Tensor,
    std: torch.Tensor,
    window_size: int,
    device: torch.device,
) -> torch.Tensor:
    max_cur = min(len(poses) - 2, clip.T - cfg.future_window - 1)
    if max_cur < window_size:
        return torch.empty((0, int(schema["total_dim"]) * int(window_size)), dtype=torch.float32)
    windows = []
    for start in range(1, max_cur - window_size + 2):
        rows = []
        anchor_idx = torch.tensor([start], dtype=torch.long, device=device)
        for cur in range(start, start + window_size):
            prev_idx = torch.tensor([cur - 1], dtype=torch.long, device=device)
            cur_idx = torch.tensor([cur], dtype=torch.long, device=device)
            next_idx = torch.tensor([cur + 1], dtype=torch.long, device=device)
            raw = tae.transition_feature_from_next_pose(
                clip,
                prev_idx,
                cur_idx,
                poses[cur - 1],
                poses[cur],
                poses[cur + 1],
                cfg,
                device,
            )
            anchored = transform_transition_feature_to_anchor(
                raw,
                clip,
                prev_idx,
                cur_idx,
                next_idx,
                anchor_idx,
                cfg,
                schema,
                device,
            )
            rows.append(((anchored - mean) / std).squeeze(0).detach().cpu())
        windows.append(torch.cat(rows, dim=0))
    if not windows:
        return torch.empty((0, int(schema["total_dim"]) * int(window_size)), dtype=torch.float32)
    return torch.stack(windows, dim=0).contiguous()


@torch.no_grad()
def anchored_windows_from_clean_clip(
    clip: tl.MotionClip,
    cfg: tl.TrainConfig,
    schema: dict[str, int],
    mean: torch.Tensor,
    std: torch.Tensor,
    window_size: int,
    device: torch.device,
) -> torch.Tensor:
    stop = clip.cyclic_period if clip.cyclic_animation else clip.T - cfg.future_window
    if stop <= 1:
        return torch.empty((0, int(schema["total_dim"]) * int(window_size)), dtype=torch.float32)
    cur_indices = torch.arange(1, stop, dtype=torch.long, device=device)
    if cur_indices.numel() < window_size:
        return torch.empty((0, int(schema["total_dim"]) * int(window_size)), dtype=torch.float32)
    raw_rows = tae.clean_transition_features(clip, cur_indices, cfg, device)
    n = int(raw_rows.shape[0]) - int(window_size) + 1
    anchor_idx = cur_indices[:n]
    chunks = []
    for offset in range(int(window_size)):
        cur_idx = cur_indices[offset : offset + n]
        prev_idx = cur_idx - 1
        next_idx = cur_idx + 1
        anchored = transform_transition_feature_to_anchor(
            raw_rows[offset : offset + n],
            clip,
            prev_idx,
            cur_idx,
            next_idx,
            anchor_idx,
            cfg,
            schema,
            device,
        )
        chunks.append(((anchored - mean) / std).detach().cpu())
    return torch.cat(chunks, dim=-1).contiguous()


@torch.no_grad()
def collect_anchored_window_dataset(
    specs: list[tuple[Path, bool | None]],
    cfg: tl.TrainConfig,
    schema: dict[str, int],
    mean: torch.Tensor,
    std: torch.Tensor,
    window_size: int,
    device: torch.device,
    max_windows: int,
    seed: int,
) -> tuple[TensorWindowDataset, list[dict]]:
    clips = tl.load_clips_from_specs(specs, cfg)
    chunks = []
    clip_meta: list[dict] = []
    for clip in clips:
        windows = anchored_windows_from_clean_clip(clip, cfg, schema, mean, std, window_size, device)
        if windows.numel() == 0:
            continue
        chunks.append(windows)
        clip_meta.append(
            {
                "path": str(clip.path),
                "name": clip.path.stem,
                "cyclic": bool(clip.cyclic_animation),
                "feature_rows": int(windows.shape[0]),
            }
        )
    if not chunks:
        raise ValueError(f"No anchored windows available for W={window_size}")
    all_windows = torch.cat(chunks, dim=0)
    if max_windows > 0 and all_windows.shape[0] > max_windows:
        generator = torch.Generator().manual_seed(seed + window_size)
        keep = torch.randperm(all_windows.shape[0], generator=generator)[:max_windows]
        all_windows = all_windows.index_select(0, keep)
    return TensorWindowDataset(all_windows), clip_meta


def window_indices(per_frame_indices: list[int], feature_dim: int, window_size: int) -> torch.Tensor:
    values: list[int] = []
    for step in range(int(window_size)):
        offset = step * int(feature_dim)
        values.extend(offset + int(index) for index in per_frame_indices)
    return torch.tensor(values, dtype=torch.long)


def conditional_root_indices(schema: dict[str, int], feature_dim: int, window_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    root = list(range(int(schema["input_root_start"]), int(schema["input_root_end"])))
    root_set = set(root)
    target = [index for index in range(int(feature_dim)) if index not in root_set]
    return window_indices(root, feature_dim, window_size), window_indices(target, feature_dim, window_size)


def model_target(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "target"):
        return model.target(x)
    return x


@torch.no_grad()
def ae_errors(model: nn.Module, x: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
    values = []
    model.eval()
    for start in range(0, x.shape[0], batch_size):
        batch = x[start : start + batch_size]
        recon = model(batch)
        values.append(F.mse_loss(recon, model_target(model, batch), reduction="none").mean(dim=-1))
    return torch.cat(values, dim=0)


def train_window_ae(
    cfg: WindowAEConfig,
    run_dir: Path,
    sequences: list[torch.Tensor],
    window_size: int,
    feature_dim: int,
    schema: dict[str, int],
    clip_meta: list[dict],
    base_prior: Path,
) -> Path:
    dataset = WindowDataset(
        sequences,
        window_size,
        stride=cfg.train_stride,
        max_windows=cfg.max_train_windows,
        seed=cfg.seed,
    )
    return train_window_ae_dataset(cfg, run_dir, dataset, window_size, feature_dim, schema, clip_meta, base_prior)


def train_window_ae_dataset(
    cfg: WindowAEConfig,
    run_dir: Path,
    dataset: Dataset,
    window_size: int,
    feature_dim: int,
    schema: dict[str, int],
    clip_meta: list[dict],
    base_prior: Path,
) -> Path:
    device = torch.device(cfg.device)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    condition_indices = None
    target_indices = None
    if cfg.conditional_root:
        condition_indices, target_indices = conditional_root_indices(schema, feature_dim, window_size)
        model = ConditionalWindowPredictor(
            feature_dim * window_size,
            condition_indices,
            target_indices,
            cfg,
        ).to(device)
    else:
        model = WindowAutoencoder(feature_dim * window_size, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    writer = SummaryWriter(run_dir / f"tb_w{window_size:02d}")
    ckpt_dir = run_dir / "checkpoints"
    best = math.inf
    stalls = 0
    start_time = time.perf_counter()
    per_frame_noise_mask = tae.input_noise_mask(schema, cfg.input_noise_mask, device)
    noise_mask = per_frame_noise_mask.repeat(int(window_size)).unsqueeze(0)
    print(
        f"window_ae W={window_size} windows={len(dataset)} dim={feature_dim * window_size} "
        f"conditional_root={cfg.conditional_root} anchor_first_root={cfg.anchor_first_root} "
        f"noise={cfg.input_noise_std:g}/{cfg.input_noise_mask} batch={cfg.batch_size} device={device}",
        flush=True,
    )
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            noisy = batch
            if cfg.input_noise_std > 0.0:
                noisy = batch + float(cfg.input_noise_std) * torch.randn_like(batch) * noise_mask
            recon = model(noisy)
            loss = F.huber_loss(recon, model_target(model, batch))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach().cpu())
            count += 1
        train_loss = loss_sum / max(1, count)
        writer.add_scalar("loss/train_huber", train_loss, epoch)
        improved = train_loss < best - cfg.min_delta
        stalls = 0 if improved else stalls + 1
        if train_loss < best:
            best = train_loss
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(cfg),
                    "window_size": int(window_size),
                    "feature_dim": int(feature_dim),
                    "schema": dict(schema),
                    "conditional_root": bool(cfg.conditional_root),
                    "anchor_first_root": bool(cfg.anchor_first_root),
                    "condition_indices": condition_indices.tolist() if condition_indices is not None else None,
                    "target_indices": target_indices.tolist() if target_indices is not None else None,
                    "base_prior_checkpoint": str(base_prior),
                    "clip_meta": clip_meta,
                    "epoch": int(epoch),
                    "best": float(best),
                },
                tl.ik_checkpoint_path(ckpt_dir / f"checkpoint_best_w{window_size:02d}.pt", cfg.run_name),
            )
        if epoch == 1 or epoch % 10 == 0 or improved:
            elapsed = time.perf_counter() - start_time
            print(
                f"W={window_size:02d} epoch={epoch:04d} train={train_loss:.6g} "
                f"best={best:.6g} stalls={stalls} elapsed_s={elapsed:.1f}",
                flush=True,
            )
        if cfg.stall_patience_epochs > 0 and stalls >= cfg.stall_patience_epochs:
            print(f"W={window_size:02d} stopped on stall epoch={epoch} best={best:.6g}", flush=True)
            break
    writer.close()
    return ckpt_dir / f"checkpoint_best_w{window_size:02d}.pt"


def make_windows_from_sequence(seq: torch.Tensor, window_size: int) -> torch.Tensor:
    if seq.shape[0] < window_size:
        return torch.empty((0, seq.shape[1] * window_size), dtype=seq.dtype)
    return torch.stack([seq[start : start + window_size].reshape(-1) for start in range(seq.shape[0] - window_size + 1)])


def load_window_model(path: Path, device: torch.device) -> tuple[WindowAutoencoder, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = WindowAEConfig()
    for key, value in ckpt.get("config", {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    full_dim = int(ckpt["feature_dim"]) * int(ckpt["window_size"])
    if ckpt.get("conditional_root", False):
        model = ConditionalWindowPredictor(
            full_dim,
            torch.tensor(ckpt["condition_indices"], dtype=torch.long),
            torch.tensor(ckpt["target_indices"], dtype=torch.long),
            cfg,
        ).to(device)
    else:
        model = WindowAutoencoder(full_dim, cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def load_controller(checkpoint_path: Path, clip: tl.MotionClip, device: torch.device) -> tuple[nn.Module, tl.TrainConfig, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = tl.TrainConfig()
    apply_config_dict(cfg, ckpt.get("config", {}))
    cfg.device = str(device)
    cfg.use_torch_compile = False
    input_dim, output_dim = tl.make_batch_dims(clip, cfg)
    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def rollout_pose_sequence(
    model: nn.Module,
    clip: tl.MotionClip,
    cfg: tl.TrainConfig,
    device: torch.device,
    max_frames: int,
) -> list[dict[str, torch.Tensor]]:
    frame_count = clip.T if max_frames <= 0 else min(clip.T, max(3, max_frames))
    poses: list[dict[str, torch.Tensor]] = []
    prev_idx = torch.tensor([0], dtype=torch.long, device=device)
    cur_idx = torch.tensor([1], dtype=torch.long, device=device)
    prev_pose = tl.get_pose_from_clip(clip, prev_idx, device)
    cur_pose = tl.get_pose_from_clip(clip, cur_idx, device)
    poses.append({key: value.detach().clone() for key, value in prev_pose.items()})
    poses.append({key: value.detach().clone() for key, value in cur_pose.items()})
    for target in range(2, frame_count):
        inp = tl.build_input(clip, prev_idx, cur_idx, prev_pose, cur_pose, cfg, device)
        raw_out = tl.predict_next_raw(model, inp, cur_pose, cfg)
        pred_pose, _ = tl.output_to_pose(raw_out, clip)
        root_pos, root_rot, _yaw, _heading = tl.root_state(clip, torch.tensor([target], dtype=torch.long, device=device), cfg, device)
        _global_pos, _global_rot, canon_pos = tl.fk_from_pose(clip, root_pos, root_rot, pred_pose, device)
        next_pose = tl.next_pose_from_prediction(pred_pose, canon_pos)
        poses.append({key: value.detach().clone() for key, value in next_pose.items()})
        prev_pose = cur_pose
        cur_pose = next_pose
        prev_idx = cur_idx
        cur_idx = torch.tensor([target], dtype=torch.long, device=device)
    return poses


@torch.no_grad()
def feature_sequence_from_poses(
    clip: tl.MotionClip,
    poses: list[dict[str, torch.Tensor]],
    cfg: tl.TrainConfig,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    rows = []
    max_cur = min(len(poses) - 2, clip.T - cfg.future_window - 1)
    for cur in range(1, max_cur + 1):
        prev_idx = torch.tensor([cur - 1], dtype=torch.long, device=device)
        cur_idx = torch.tensor([cur], dtype=torch.long, device=device)
        raw = tae.transition_feature_from_next_pose(
            clip,
            prev_idx,
            cur_idx,
            poses[cur - 1],
            poses[cur],
            poses[cur + 1],
            cfg,
            device,
        )
        rows.append(((raw - mean) / std).squeeze(0).detach().cpu())
    if not rows:
        return torch.empty((0, mean.shape[0]), dtype=torch.float32)
    return torch.stack(rows, dim=0)


def summarize_errors(errors: torch.Tensor) -> dict[str, float]:
    if errors.numel() == 0:
        return {"mean": float("nan"), "p95": float("nan"), "count": 0.0}
    return {
        "mean": float(errors.mean().cpu()),
        "p95": float(torch.quantile(errors.detach().cpu(), 0.95)),
        "count": float(errors.numel()),
    }


@torch.no_grad()
def compare_models(
    cfg: WindowAEConfig,
    run_dir: Path,
    window_ckpts: list[Path],
    base_prior_model: tae.TransitionAutoencoder,
    feature_cfg: tl.TrainConfig,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Path:
    device = torch.device(cfg.device)
    compare_folder = resolve_path(cfg.compare_folder_path)
    paths = sorted(compare_folder.glob(cfg.compare_glob))
    if not paths:
        raise FileNotFoundError(f"No comparison npz files matching {cfg.compare_glob!r} in {compare_folder}")
    rows: list[dict[str, str | float]] = []
    base_prior_model = base_prior_model.to(device).eval()
    mean = mean.to(device)
    std = std.to(device)

    window_models = []
    for path in window_ckpts:
        model, ckpt = load_window_model(path, device)
        window_models.append((int(ckpt["window_size"]), model, path, ckpt))

    for npz_path in paths:
        eval_clip = tl.MotionClip(npz_path, feature_cfg, cyclic_animation=False)
        controller, controller_cfg, controller_ckpt = load_controller(resolve_path(cfg.model_checkpoint), eval_clip, device)
        feature_cfg_for_rollout = tl.TrainConfig()
        apply_config_dict(feature_cfg_for_rollout, asdict(controller_cfg))
        feature_cfg_for_rollout.include_transition_foot_motion = bool(feature_cfg.include_transition_foot_motion)
        feature_cfg_for_rollout.foot_slide_scale_mps = float(feature_cfg.foot_slide_scale_mps)
        feature_cfg_for_rollout.transition_yaw_scale_radps = float(feature_cfg.transition_yaw_scale_radps)
        feature_cfg_for_rollout.root_lookahead_steps = int(getattr(feature_cfg, "root_lookahead_steps", 0))
        feature_cfg_for_rollout.use_torch_compile = False
        feature_cfg_for_rollout.device = str(device)

        gt_poses = [
            tl.get_pose_from_clip(eval_clip, torch.tensor([i], dtype=torch.long, device=device), device)
            for i in range(eval_clip.T if cfg.compare_max_frames <= 0 else min(eval_clip.T, cfg.compare_max_frames))
        ]
        pred_poses = rollout_pose_sequence(controller, eval_clip, controller_cfg, device, cfg.compare_max_frames)
        gt_seq = feature_sequence_from_poses(eval_clip, gt_poses, feature_cfg_for_rollout, mean, std, device)
        pred_seq = feature_sequence_from_poses(eval_clip, pred_poses, feature_cfg_for_rollout, mean, std, device)

        gt_1 = summarize_errors(ae_errors(base_prior_model, gt_seq.to(device)))
        pred_1 = summarize_errors(ae_errors(base_prior_model, pred_seq.to(device)))
        base_row = {
            "clip": npz_path.stem,
            "checkpoint": str(resolve_path(cfg.model_checkpoint)),
            "checkpoint_epoch": float(controller_ckpt.get("epoch", -1)),
            "score_type": "transition_ae_w01",
            "gt_mean": gt_1["mean"],
            "gt_p95": gt_1["p95"],
            "generated_mean": pred_1["mean"],
            "generated_p95": pred_1["p95"],
            "sample_count": pred_1["count"],
        }
        rows.append(base_row)

        for window_size, model, ckpt_path, window_ckpt in window_models:
            if bool(window_ckpt.get("anchor_first_root", False)):
                gt_windows = anchored_windows_from_poses(
                    eval_clip,
                    gt_poses,
                    feature_cfg_for_rollout,
                    dict(window_ckpt.get("schema", tae.transition_schema(eval_clip, feature_cfg_for_rollout))),
                    mean,
                    std,
                    window_size,
                    device,
                ).to(device)
                pred_windows = anchored_windows_from_poses(
                    eval_clip,
                    pred_poses,
                    feature_cfg_for_rollout,
                    dict(window_ckpt.get("schema", tae.transition_schema(eval_clip, feature_cfg_for_rollout))),
                    mean,
                    std,
                    window_size,
                    device,
                ).to(device)
            else:
                gt_windows = make_windows_from_sequence(gt_seq, window_size).to(device)
                pred_windows = make_windows_from_sequence(pred_seq, window_size).to(device)
            gt_w = summarize_errors(ae_errors(model, gt_windows))
            pred_w = summarize_errors(ae_errors(model, pred_windows))
            rows.append(
                {
                    "clip": npz_path.stem,
                    "checkpoint": str(resolve_path(cfg.model_checkpoint)),
                    "checkpoint_epoch": float(controller_ckpt.get("epoch", -1)),
                    "score_type": f"window_ae_w{window_size:02d}",
                    "gt_mean": gt_w["mean"],
                    "gt_p95": gt_w["p95"],
                    "generated_mean": pred_w["mean"],
                    "generated_p95": pred_w["p95"],
                    "sample_count": pred_w["count"],
                    "window_checkpoint": str(ckpt_path),
                }
            )

    summary: dict[str, dict[str, float]] = {}
    for score_type in sorted({str(row["score_type"]) for row in rows}):
        subset = [row for row in rows if row["score_type"] == score_type]
        summary[score_type] = {
            "gt_mean_over_clips": float(np.nanmean([float(row["gt_mean"]) for row in subset])),
            "generated_mean_over_clips": float(np.nanmean([float(row["generated_mean"]) for row in subset])),
            "generated_p95_over_clips": float(np.nanmean([float(row["generated_p95"]) for row in subset])),
        }

    out = run_dir / "neutral_stand_90_window_compare.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "neutral_stand_90_window_compare_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("neutral stand 90 comparison summary:", flush=True)
    for key, value in summary.items():
        print(
            f"  {key}: gt_mean={value['gt_mean_over_clips']:.6g} "
            f"generated_mean={value['generated_mean_over_clips']:.6g} "
            f"generated_p95={value['generated_p95_over_clips']:.6g}",
            flush=True,
        )
    print(f"wrote comparison {out}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train W-step transition autoencoders and compare against the 1-frame AE.")
    for name, field in WindowAEConfig.__dataclass_fields__.items():
        default = field.default
        arg = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(arg, action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument(arg, type=type(default), default=None)
    args = parser.parse_args()
    cfg = WindowAEConfig()
    for field in cfg.__dataclass_fields__:
        value = getattr(args, field, None)
        if value is not None:
            setattr(cfg, field, value)

    set_seed(cfg.seed)
    if cfg.date_prefix_run_name:
        cfg.run_name = tl.date_prefixed_run_name(cfg.run_name)
    run_dir = resolve_path(cfg.output_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    base_prior = resolve_path(cfg.base_prior_checkpoint)
    device = torch.device(cfg.device)
    tl.apply_cuda_performance_settings(tl.TrainConfig(device=cfg.device), device)
    base_model, locomotion_cfg, schema, mean, std, prior_ckpt = load_base_prior(base_prior, device)
    specs = prior_clip_specs(prior_ckpt)
    mean_cpu = mean.detach().cpu()
    std_cpu = std.detach().cpu()
    window_sizes = [int(part.strip()) for part in cfg.window_sizes.split(",") if part.strip()]
    window_ckpts = []
    if cfg.anchor_first_root:
        for window_size in window_sizes:
            dataset, clip_meta = collect_anchored_window_dataset(
                specs,
                locomotion_cfg,
                schema,
                mean.to(torch.device(cfg.feature_device)),
                std.to(torch.device(cfg.feature_device)),
                window_size,
                torch.device(cfg.feature_device),
                cfg.max_train_windows,
                cfg.seed,
            )
            window_ckpts.append(
                train_window_ae_dataset(
                    cfg,
                    run_dir,
                    dataset,
                    window_size,
                    int(schema["total_dim"]),
                    schema,
                    clip_meta,
                    base_prior,
                )
            )
    else:
        sequences, clip_meta = load_or_collect_sequences(cfg, specs, locomotion_cfg, mean_cpu, std_cpu, schema["total_dim"], base_prior)
        for window_size in window_sizes:
            window_ckpts.append(
                train_window_ae(
                    cfg,
                    run_dir,
                    sequences,
                    window_size,
                    int(schema["total_dim"]),
                    schema,
                    clip_meta,
                    base_prior,
                )
            )
    compare_models(cfg, run_dir, window_ckpts, base_model, locomotion_cfg, mean, std)
    print(f"window AE run complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
