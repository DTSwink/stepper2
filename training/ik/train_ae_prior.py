from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import contact_physics as cp
import excess_envelope as ee
try:
    from . import ik_core as tl
    from . import transition_autoencoder as tae
except ImportError:
    import ik_core as tl
    import transition_autoencoder as tae
import window_transition_autoencoder as wae


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def refresh_live_viewer(args: argparse.Namespace, checkpoint_path: Path) -> None:
    if not args.live_viewer:
        return
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "training" / "visualize_model.py"),
        "--npz-path",
        str(tl.resolve_path(args.live_npz_path)),
        "--checkpoint-path",
        str(checkpoint_path),
        "--output-path",
        str(tl.resolve_path(args.live_output_path)),
    ]
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        print(f"live viewer refresh skipped: {exc}", flush=True)


def load_prior(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "window_size" in ckpt:
        model, ckpt = wae.load_window_model(path, device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model, ckpt
    cfg = tae.ae_config_from_dict(ckpt["config"])
    model = tae.TransitionAutoencoder(int(ckpt["schema"]["total_dim"]), cfg, dict(ckpt["schema"])).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    allowed_missing = {"reconstruction_weights"}
    if unexpected or any(key not in allowed_missing for key in missing):
        raise RuntimeError(f"Could not load AE prior {path}: missing={missing}, unexpected={unexpected}")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, ckpt


def resolve_checkpoint_reference(path_text: str | Path) -> Path:
    path = Path(path_text)
    candidates = [path]
    text = str(path_text).replace("\\", "/")
    for marker in ("/stepper/", "stepper/"):
        if marker in text:
            rel = text.split(marker, 1)[1]
            candidates.append(PROJECT_ROOT / rel)
            break
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return tl.resolve_path(str(path_text))


def apply_locomotion_config_dict(cfg: tl.TrainConfig, values: dict) -> None:
    valid = set(tl.TrainConfig.__dataclass_fields__.keys())
    for key, value in values.items():
        if key in valid:
            setattr(cfg, key, value)


def load_prior_bundle(paths: list[Path], device: torch.device, weights: list[float] | None = None):
    bundle = []
    weights = weights or [1.0 for _path in paths]
    for prior_i, path in enumerate(paths):
        prior, ckpt = load_prior(path, device)
        weight = float(weights[prior_i])
        if "window_size" in ckpt:
            window_size = int(ckpt["window_size"])
            base_path = resolve_checkpoint_reference(ckpt["base_prior_checkpoint"])
            base_ckpt = torch.load(base_path, map_location=device, weights_only=False)
            locomotion_cfg = tl.TrainConfig()
            apply_locomotion_config_dict(locomotion_cfg, base_ckpt.get("locomotion_config", {}))
            bundle.append(
                {
                    "kind": "window",
                    "path": path,
                    "model": prior,
                    "mean": base_ckpt["mean"].to(device),
                    "std": base_ckpt["std"].to(device),
                    "locomotion_config": locomotion_cfg,
                    "window_size": window_size,
                    "feature_dim": int(ckpt["feature_dim"]),
                    "schema": dict(ckpt.get("schema", {})),
                    "anchor_first_root": bool(ckpt.get("anchor_first_root", False)),
                    "base_prior_checkpoint": base_path,
                    "weight": weight,
                    "label": f"prior_{prior_i}_window_w{window_size:02d}",
                }
            )
            continue
        locomotion_cfg = tl.TrainConfig()
        apply_locomotion_config_dict(locomotion_cfg, ckpt.get("locomotion_config", {}))
        bundle.append(
            {
                "kind": "transition",
                "path": path,
                "model": prior,
                "mean": ckpt["mean"].to(device),
                "std": ckpt["std"].to(device),
                "locomotion_config": locomotion_cfg,
                "weight": weight,
                "label": f"prior_{prior_i}_transition",
            }
        )
    return bundle


def ae_score(
    prior,
    mean,
    std,
    features: torch.Tensor,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
    compatibility_weight: float = 0.0,
) -> torch.Tensor:
    x = (features - mean) / std
    recon = prior(x)
    target = prior.target(x) if hasattr(prior, "target") else x
    score = tae.reconstruction_loss_rows(prior, recon, target, loss_type, huber_delta)
    if compatibility_weight > 0.0 and hasattr(prior, "has_compatibility_head") and prior.has_compatibility_head():
        compatibility = F.softplus(-prior.compatibility_logits(x))
        score = score + compatibility_weight * compatibility
    return score.mean()


def ae_score_rows(
    prior,
    mean,
    std,
    features: torch.Tensor,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
    compatibility_weight: float = 0.0,
) -> torch.Tensor:
    x = (features - mean) / std
    recon = prior(x)
    target = prior.target(x) if hasattr(prior, "target") else x
    score = tae.reconstruction_loss_rows(prior, recon, target, loss_type, huber_delta)
    if compatibility_weight > 0.0 and hasattr(prior, "has_compatibility_head") and prior.has_compatibility_head():
        compatibility = F.softplus(-prior.compatibility_logits(x))
        score = score + compatibility_weight * compatibility
    return score


def ae_score_normalized(
    prior,
    x: torch.Tensor,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
    compatibility_weight: float = 0.0,
) -> torch.Tensor:
    recon = prior(x)
    target = prior.target(x) if hasattr(prior, "target") else x
    score = tae.reconstruction_loss_rows(prior, recon, target, loss_type, huber_delta)
    if compatibility_weight > 0.0 and hasattr(prior, "has_compatibility_head") and prior.has_compatibility_head():
        compatibility = F.softplus(-prior.compatibility_logits(x))
        score = score + compatibility_weight * compatibility
    return score.mean()


def ae_score_normalized_rows(
    prior,
    x: torch.Tensor,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
    compatibility_weight: float = 0.0,
) -> torch.Tensor:
    recon = prior(x)
    target = prior.target(x) if hasattr(prior, "target") else x
    score = tae.reconstruction_loss_rows(prior, recon, target, loss_type, huber_delta)
    if compatibility_weight > 0.0 and hasattr(prior, "has_compatibility_head") and prior.has_compatibility_head():
        compatibility = F.softplus(-prior.compatibility_logits(x))
        score = score + compatibility_weight * compatibility
    return score


def reduce_ae_score_rows(score_rows: torch.Tensor, cfg: tl.TrainConfig) -> torch.Tensor:
    mean_score = score_rows.mean()
    top_fraction = float(getattr(cfg, "ae_row_top_fraction", 0.0))
    top_weight = float(getattr(cfg, "ae_row_top_weight", 0.0))
    if top_fraction <= 0.0 or top_weight <= 0.0 or score_rows.numel() <= 1:
        return mean_score
    k = max(1, min(score_rows.numel(), int(math.ceil(score_rows.numel() * top_fraction))))
    top_score = torch.topk(score_rows, k=k, largest=True, sorted=False).values.mean()
    return mean_score + top_weight * top_score


def prior_transition_cfg(prior_info: dict[str, object], fallback_cfg: tl.TrainConfig) -> tl.TrainConfig:
    cfg = prior_info.get("locomotion_config")
    return cfg if isinstance(cfg, tl.TrainConfig) else fallback_cfg


def transition_feature_root_lookahead_steps(prior_info: dict[str, object], fallback_cfg: tl.TrainConfig) -> int:
    return max(0, int(getattr(prior_transition_cfg(prior_info, fallback_cfg), "root_lookahead_steps", 0)))


def transition_feature_horizon(cfg: tl.TrainConfig) -> int:
    root_lookahead_steps = max(0, int(getattr(cfg, "root_lookahead_steps", 0)))
    return max(int(cfg.future_window), root_lookahead_steps + 1)


class AnyStartDataset(torch.utils.data.Dataset):
    def __init__(self, clips: list[tl.MotionClip], cfg: tl.TrainConfig, requested_k: int):
        self.items: list[tuple[int, int]] = []
        for ci, clip in enumerate(clips):
            max_start = clip_sample_start_max(clip, cfg, requested_k, cfg.agent_min_cohort_steps)
            if max_start < 1:
                continue
            starts = list(range(1, max_start + 1))
            random.Random(cfg.seed + ci).shuffle(starts)
            self.items.extend((ci, s) for s in starts)
        if not self.items:
            raise ValueError("No one-step starts found. Need clips with at least 3 frames.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[int, int]:
        return self.items[index]


def clip_future_safe_current_max(clip: tl.MotionClip, cfg: tl.TrainConfig) -> int:
    if clip.cyclic_animation:
        return int(clip.cyclic_period) - 1
    return int(clip.T) - transition_feature_horizon(cfg) - 1


def clip_any_start_max(clip: tl.MotionClip, cfg: tl.TrainConfig) -> int:
    if clip.cyclic_animation:
        return int(clip.cyclic_period) - 1
    return clip_future_safe_current_max(clip, cfg)


def clip_sample_start_max(
    clip: tl.MotionClip,
    cfg: tl.TrainConfig,
    requested_k: int,
    min_cohort_steps: int = 1,
) -> int:
    if clip.cyclic_animation:
        return int(clip.cyclic_period) - 1
    requested_k = max(1, int(requested_k))
    horizon = transition_feature_horizon(cfg)
    full_k_max_start = int(clip.T) - horizon - requested_k
    max_start = full_k_max_start if full_k_max_start >= 1 else clip_any_start_max(clip, cfg)
    min_cohort_steps = max(1, int(min_cohort_steps))
    if min_cohort_steps > 1:
        guaranteed_max_start = int(clip.T) - horizon - min_cohort_steps
        if guaranteed_max_start >= 1:
            max_start = min(max_start, guaranteed_max_start)
    return max_start


def clip_has_any_start(clip: tl.MotionClip, cfg: tl.TrainConfig) -> bool:
    return clip_any_start_max(clip, cfg) >= 1


def expected_active_rollout_steps(
    clip: tl.MotionClip,
    cfg: tl.TrainConfig,
    requested_k: int,
    min_cohort_steps: int = 1,
) -> float:
    requested_k = max(1, int(requested_k))
    if clip.cyclic_animation:
        return float(requested_k)
    max_start = clip_sample_start_max(clip, cfg, requested_k, min_cohort_steps)
    if max_start < 1:
        return 0.0
    total = 0
    for start in range(1, max_start + 1):
        total += max(1, min(requested_k, clip_future_safe_current_max(clip, cfg) - start + 1))
    return float(total) / float(max_start)


def clip_is_idle(clip: tl.MotionClip) -> bool:
    return "idle" in Path(clip.path).stem.lower()


def clip_is_turn_in_place(clip: tl.MotionClip) -> bool:
    stem = Path(clip.path).stem.lower()
    return "stand_turn" in stem and any(token in stem for token in ("045", "090", "135", "180"))

class ClipStore:
    """Dense multi-clip storage for Isaac-style per-agent random rollouts."""

    def __init__(
        self,
        clips: list[tl.MotionClip],
        cfg: tl.TrainConfig,
        device: torch.device,
        synthetic_clip_indices: set[int] | None = None,
    ):
        if not clips:
            raise ValueError("ClipStore needs at least one clip")
        first = clips[0]
        for clip in clips[1:]:
            if clip.body_names != first.body_names or clip.parents_body_list != first.parents_body_list:
                raise ValueError("row-mixed rollouts require all clips to share the same reduced skeleton")
        self.clips = clips
        self.cfg = cfg
        self.device = device
        synthetic_clip_indices = synthetic_clip_indices or set()
        self.prototype = first
        self.J = first.J
        self.Jn = first.Jn
        self.Jcore = first.Jcore
        self.Jmarkers = first.Jmarkers
        self.ik_payload_dim = int(getattr(first, "ik_payload_dim", 0))
        self.pelvis = first.pelvis
        self.nonpelvis_map = first.nonpelvis_map
        self.core_nonpelvis_map = first.core_nonpelvis_map
        self.parents_body_list = first.parents_body_list
        self.pose_representation = first.pose_representation
        self.ik_controlled_set = set(first.ik_controlled_set)
        self.ik_marker_names = list(first.ik_marker_names)
        self.ik_marker_indices = tuple(int(x) for x in first.ik_marker_indices)
        self.ik_marker_indices_tensor = torch.tensor(self.ik_marker_indices, dtype=torch.long, device=device)
        self.ik_marker_index_by_name = {name: i for i, name in enumerate(self.ik_marker_names)}
        self.ik_limb_specs = list(first.ik_limb_specs)
        self.ik_rest_axis = first.ik_rest_axis.to(device)
        self.ik_rest_pole = first.ik_rest_pole.to(device)
        self.ik_limb_lengths = first.ik_limb_lengths.to(device)
        self.ik_local_pole_axis = first.ik_local_pole_axis.to(device)
        self.ik_toe_offsets = first.ik_toe_offsets.to(device)
        self.ik_toe_axis = first.ik_toe_axis.to(device)
        self.ik_rest_axis_by_clip = torch.stack([clip.ik_rest_axis for clip in clips], dim=0).to(device)
        self.ik_rest_pole_by_clip = torch.stack([clip.ik_rest_pole for clip in clips], dim=0).to(device)
        self.ik_local_pole_axis_by_clip = torch.stack([clip.ik_local_pole_axis for clip in clips], dim=0).to(device)
        self.ik_toe_axis_by_clip = torch.stack([clip.ik_toe_axis for clip in clips], dim=0).to(device)
        self.foot_indices = tuple(int(x) for x in first.foot_indices_tensor.tolist())
        self.toe_indices = tuple(int(x) for x in first.toe_indices_tensor.tolist())
        self.fps = float(first.fps)
        self.up_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device).view(1, 1, 3)
        self.side_axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device).view(1, 1, 3)

        lengths = [clip.T for clip in clips]
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + int(length))
        self.frame_offsets = torch.tensor(offsets[:-1], dtype=torch.long, device=device)
        self.lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.periods = torch.tensor([clip.cyclic_period for clip in clips], dtype=torch.long, device=device)
        self.cyclic = torch.tensor([clip.cyclic_animation for clip in clips], dtype=torch.bool, device=device)
        self.synthetic = torch.tensor(
            [i in synthetic_clip_indices for i in range(len(clips))],
            dtype=torch.bool,
            device=device,
        )
        self.idle = torch.tensor([clip_is_idle(clip) for clip in clips], dtype=torch.bool, device=device)
        self.turn_in_place = torch.tensor([clip_is_turn_in_place(clip) for clip in clips], dtype=torch.bool, device=device)
        self.future_safe_max = torch.where(
            self.cyclic,
            self.periods - 1,
            self.lengths - transition_feature_horizon(cfg) - 1,
        )

        tensors = [clip.tensors(device) for clip in clips]
        self.root_pos = torch.cat([t["root_pos"] for t in tensors], dim=0)
        self.root_rot = torch.cat([t["root_rot"] for t in tensors], dim=0)
        self.pelvis_local_pos = torch.cat([t["pelvis_local_pos"] for t in tensors], dim=0)
        self.pelvis_rot6 = torch.cat([t["pelvis_rot6"] for t in tensors], dim=0)
        self.non_pelvis_rot6 = torch.cat([t["non_pelvis_rot6"] for t in tensors], dim=0)
        self.core_non_pelvis_rot6 = torch.cat([t["core_non_pelvis_rot6"] for t in tensors], dim=0)
        self.canonical_pos = torch.cat([t["canonical_pos"] for t in tensors], dim=0)
        self.ik_marker_pos = torch.cat([t["ik_marker_pos"] for t in tensors], dim=0)
        self.ik_payload = torch.cat([t["ik_payload"] for t in tensors], dim=0)
        self.target_output = self._build_target_output()
        self.local_offsets = torch.stack([t["local_offsets"] for t in tensors], dim=0)
        self.slide_bound_mps: torch.Tensor | None = None
        self.yaw_bound_radps: torch.Tensor | None = None
        self.excess_envelope_metadata: dict[str, float | int | str] = {}
        if cfg.excess_envelope_enabled and (
            cfg.slide_excess_loss_weight > 0.0 or cfg.yaw_excess_loss_weight > 0.0
        ):
            env_cfg = ee.ExcessEnvelopeConfig(
                margin=float(cfg.excess_envelope_margin),
                knn=int(cfg.excess_envelope_knn),
                cache_dir=str(cfg.excess_envelope_cache_dir),
            )
            real_indices = [i for i in range(len(clips)) if i not in synthetic_clip_indices]
            envelope = ee.load_or_build_excess_envelope(
                clips,
                cfg,
                device,
                synthetic_clip_indices=synthetic_clip_indices,
                real_clip_indices=real_indices,
                env_cfg=env_cfg,
            )
            self.slide_bound_mps = envelope["slide_bound_mps"].to(device)  # type: ignore[union-attr]
            self.yaw_bound_radps = envelope["yaw_excess_bound_radps"].to(device)  # type: ignore[union-attr]
            self.excess_envelope_metadata = envelope["metadata"]  # type: ignore[assignment]
            print(
                "excess envelope "
                f"cache_hit={self.excess_envelope_metadata.get('cache_hit')} "
                f"real_transitions={self.excess_envelope_metadata.get('source_real_transitions')} "
                f"targets={self.excess_envelope_metadata.get('target_transitions')} "
                f"max_real_slide={self.excess_envelope_metadata.get('max_real_slide_mps'):.6g} "
                f"max_real_yaw_excess={self.excess_envelope_metadata.get('max_real_yaw_excess_radps'):.6g}",
                flush=True,
            )

        self.root0_pos = torch.stack([t["root_pos"][0] for t in tensors], dim=0)
        self.root0_rot = torch.stack([t["root_rot"][0] for t in tensors], dim=0)
        self.root0_inv = self.root0_rot.transpose(-1, -2)
        self.end_pos = torch.stack([t["root_pos"][clip.cyclic_period] for clip, t in zip(clips, tensors)], dim=0)
        self.end_rot = torch.stack([t["root_rot"][clip.cyclic_period] for clip, t in zip(clips, tensors)], dim=0)
        self.cycle_pos = torch.matmul(
            (self.end_pos - self.root0_pos).unsqueeze(1),
            self.root0_inv,
        ).squeeze(1)
        self.cycle_rot = self.end_rot @ self.root0_inv
        self.input_root_features = self._build_input_root_features()

        self.stage_clip_indices = torch.arange(len(clips), dtype=torch.long, device=device)
        self.stage_clip_probs = torch.full((len(clips),), 1.0 / len(clips), dtype=torch.float32, device=device)
        self.real_stage_clip_indices = self.stage_clip_indices
        self.real_stage_clip_probs = self.stage_clip_probs
        self.synthetic_stage_clip_indices = torch.empty((0,), dtype=torch.long, device=device)
        self.synthetic_stage_clip_probs = torch.empty((0,), dtype=torch.float32, device=device)

    def _set_pool(self, indices: list[int], weights: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
        if not indices:
            return (
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.float32, device=self.device),
            )
        idx = torch.tensor(indices, dtype=torch.long, device=self.device)
        w = torch.tensor(weights, dtype=torch.float32, device=self.device).clamp_min(0.0)
        if float(w.sum().detach().cpu()) <= 0.0:
            w = torch.ones_like(w)
        return idx, w / w.sum()

    def update_stage_sampling(
        self,
        indices: list[int],
        weights: list[float],
        synthetic_indices: list[int] | None = None,
        synthetic_weights: list[float] | None = None,
    ) -> None:
        if not indices:
            raise ValueError("stage sampling needs at least one real clip")
        self.real_stage_clip_indices, self.real_stage_clip_probs = self._set_pool(indices, weights)
        self.synthetic_stage_clip_indices, self.synthetic_stage_clip_probs = self._set_pool(
            synthetic_indices or [],
            synthetic_weights or [],
        )
        self.stage_clip_indices = self.real_stage_clip_indices
        self.stage_clip_probs = self.real_stage_clip_probs

    def _build_target_output(self) -> torch.Tensor:
        b = self.pelvis_local_pos.shape[0]
        if tl.uses_ik_markers(self.pose_representation):
            return torch.cat(
                (
                    self.pelvis_local_pos,
                    self.pelvis_rot6,
                    self.core_non_pelvis_rot6.reshape(b, -1),
                    self.ik_payload,
                ),
                dim=-1,
            )
        return torch.cat(
            (
                self.pelvis_local_pos,
                self.pelvis_rot6,
                self.non_pelvis_rot6.reshape(b, -1),
            ),
            dim=-1,
        )

    def _build_input_root_features(self) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        future_steps = int(self.cfg.future_window)
        feature_dim = 3 + future_steps * 4
        for clip_id, clip in enumerate(self.clips):
            features = torch.zeros((int(clip.T), feature_dim), dtype=torch.float32, device=self.device)
            if int(clip.T) <= 1:
                chunks.append(features)
                continue

            if clip.cyclic_animation:
                period = max(1, int(clip.cyclic_period))
                rows = torch.arange(period, dtype=torch.long, device=self.device)
                cur_idx = rows.clone()
                cur_idx[0] = period
            else:
                max_cur = int(clip.T) - future_steps - 1
                if max_cur < 1:
                    chunks.append(features)
                    continue
                rows = torch.arange(1, max_cur + 1, dtype=torch.long, device=self.device)
                cur_idx = rows

            clip_ids = torch.full((cur_idx.numel(),), clip_id, dtype=torch.long, device=self.device)
            prev_idx = cur_idx - 1
            prev_pos, _prev_rot, prev_yaw, prev_heading = self.root_state(clip_ids, prev_idx)
            cur_pos, _cur_rot, cur_yaw, cur_heading = self.root_state(clip_ids, cur_idx)
            delta_local = torch.matmul((cur_pos - prev_pos).unsqueeze(1), prev_heading).squeeze(1)
            root_feat = torch.stack(
                (
                    delta_local[:, 0] / self.cfg.max_speed_scale_final,
                    delta_local[:, 2] / self.cfg.max_speed_scale_final,
                    tl.wrap_angle(cur_yaw - prev_yaw) / self.cfg.max_turn_rate_scale_final,
                ),
                dim=-1,
            )
            future_offsets = torch.arange(1, future_steps + 1, device=self.device, dtype=cur_idx.dtype)
            flat_clip_ids = clip_ids.reshape(-1, 1).expand(-1, future_steps).reshape(-1)
            flat_idx = (cur_idx.reshape(-1, 1) + future_offsets.reshape(1, future_steps)).reshape(-1)
            fut_pos, _fut_rot, fut_yaw, _fut_heading = self.root_state(flat_clip_ids, flat_idx)
            fut_pos = fut_pos.reshape(cur_idx.numel(), future_steps, 3)
            fut_yaw = fut_yaw.reshape(cur_idx.numel(), future_steps)
            fut_local = torch.matmul((fut_pos - cur_pos[:, None, :]).unsqueeze(-2), cur_heading[:, None, :, :]).squeeze(-2)
            scale = future_offsets.to(dtype=fut_local.dtype).reshape(1, future_steps) * self.cfg.max_speed_scale_final
            dyaw = tl.wrap_angle(fut_yaw - cur_yaw[:, None])
            future_feat = torch.stack(
                (
                    torch.clamp(fut_local[:, :, 0] / scale, -2.0, 2.0),
                    torch.clamp(fut_local[:, :, 2] / scale, -2.0, 2.0),
                    torch.cos(dyaw),
                    torch.sin(dyaw),
                ),
                dim=-1,
            ).reshape(cur_idx.numel(), future_steps * 4)
            features[rows] = torch.cat((root_feat, future_feat), dim=-1)
            chunks.append(features)
        return torch.cat(chunks, dim=0)

    def _sample_from_pool(
        self,
        count: int,
        requested_k: int,
        indices: torch.Tensor,
        probs: torch.Tensor,
        require_full: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if int(count) <= 0:
            return (
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.long, device=self.device),
            )
        if indices.numel() == 0:
            raise ValueError("cannot sample starts from an empty clip pool")
        if require_full:
            cyclic = self.cyclic.index_select(0, indices)
            lengths = self.lengths.index_select(0, indices)
            full_ok = torch.logical_or(
                cyclic,
                lengths - transition_feature_horizon(self.cfg) - max(1, int(requested_k)) >= 1,
            )
            if not bool(full_ok.any()):
                raise ValueError(f"no clips in this pool cover a full K={requested_k} rollout")
            indices = indices[full_ok]
            probs = probs[full_ok]
            probs = probs / probs.sum().clamp_min(1e-12)
        choices = torch.multinomial(probs, int(count), replacement=True)
        clip_ids = indices.index_select(0, choices)
        max_start = self.clip_sample_start_max(clip_ids, requested_k)
        if int(self.cfg.agent_fixed_start_frame) > 0:
            starts = torch.minimum(
                torch.full_like(max_start, int(self.cfg.agent_fixed_start_frame)),
                max_start,
            ).clamp_min(1)
        else:
            starts = (torch.rand(int(count), device=self.device) * max_start.float()).floor().long() + 1
        return clip_ids, starts

    def sample_starts(
        self,
        count: int,
        requested_k: int,
        synthetic: bool = False,
        require_full: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if synthetic:
            return self._sample_from_pool(
                count,
                requested_k,
                self.synthetic_stage_clip_indices,
                self.synthetic_stage_clip_probs,
                require_full=require_full,
            )
        return self._sample_from_pool(
            count,
            requested_k,
            self.real_stage_clip_indices,
            self.real_stage_clip_probs,
            require_full=require_full,
        )

    def sample_starts_for_existing_groups(
        self,
        synthetic_mask: torch.Tensor,
        requested_k: int,
        require_full: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        synthetic_mask = synthetic_mask.to(self.device).bool()
        count = int(synthetic_mask.numel())
        clip_ids = torch.empty((count,), dtype=torch.long, device=self.device)
        starts = torch.empty((count,), dtype=torch.long, device=self.device)
        for want_synthetic in (False, True):
            rows = (synthetic_mask == want_synthetic).nonzero(as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            new_clip_ids, new_starts = self.sample_starts(
                int(rows.numel()),
                requested_k,
                synthetic=want_synthetic,
                require_full=require_full,
            )
            clip_ids[rows] = new_clip_ids
            starts[rows] = new_starts
        return clip_ids, starts

    def clip_sample_start_max(self, clip_ids: torch.Tensor, requested_k: int) -> torch.Tensor:
        clip_ids = clip_ids.to(self.device)
        cyclic = self.cyclic.index_select(0, clip_ids)
        period_max = self.periods.index_select(0, clip_ids) - 1
        lengths = self.lengths.index_select(0, clip_ids)
        horizon = transition_feature_horizon(self.cfg)
        any_start_max = lengths - horizon - 1
        requested_k = max(1, int(requested_k))
        full_k_max = lengths - horizon - requested_k
        max_start = torch.where(full_k_max >= 1, full_k_max, any_start_max)
        min_steps = max(1, int(self.cfg.agent_min_cohort_steps))
        guaranteed = lengths - horizon - min_steps
        max_start = torch.where(guaranteed >= 1, torch.minimum(max_start, guaranteed), max_start)
        return torch.where(cyclic, period_max, max_start).clamp_min(1)

    def frame_index(self, clip_ids: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        clip_ids = clip_ids.to(self.device).long()
        idx = idx.to(self.device).long()
        periods = self.periods.index_select(0, clip_ids).clamp_min(1)
        cyclic = self.cyclic.index_select(0, clip_ids)
        logical = torch.where(cyclic, torch.remainder(idx, periods), idx)
        return self.frame_offsets.index_select(0, clip_ids) + logical

    def get_pose(self, clip_ids: torch.Tensor, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        frame = self.frame_index(clip_ids, idx)
        pose = {
            "pelvis_pos": self.pelvis_local_pos.index_select(0, frame),
            "pelvis_rot6": self.pelvis_rot6.index_select(0, frame),
            "nonpelvis_rot6": self.non_pelvis_rot6.index_select(0, frame),
            "canon_pos": self.canonical_pos.index_select(0, frame),
        }
        if tl.uses_ik_markers(self.pose_representation):
            pose["core_nonpelvis_rot6"] = self.core_non_pelvis_rot6.index_select(0, frame)
            pose["ik_payload"] = self.ik_payload.index_select(0, frame)
        return pose

    def get_target_output(self, clip_ids: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return self.target_output.index_select(0, self.frame_index(clip_ids, idx))

    def get_input_root_features(self, clip_ids: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return self.input_root_features.index_select(0, self.frame_index(clip_ids, idx))

    def root_state(self, clip_ids: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        clip_ids = clip_ids.to(self.device).long()
        idx = idx.to(self.device).long()
        frame = self.frame_index(clip_ids, idx)
        base_pos = self.root_pos.index_select(0, frame)
        base_rot = self.root_rot.index_select(0, frame)
        root0_pos = self.root0_pos.index_select(0, clip_ids)
        root0_rot = self.root0_rot.index_select(0, clip_ids)
        root0_inv = self.root0_inv.index_select(0, clip_ids)
        periods = self.periods.index_select(0, clip_ids).clamp_min(1)
        cyclic = self.cyclic.index_select(0, clip_ids)
        cycles = torch.where(cyclic, torch.div(idx, periods, rounding_mode="floor"), torch.zeros_like(idx))

        rel_pos = torch.matmul((base_pos - root0_pos).unsqueeze(1), root0_inv).squeeze(1)
        rel_rot = base_rot @ root0_inv
        cycle_pos = self.cycle_pos.index_select(0, clip_ids)
        cycle_rot = self.cycle_rot.index_select(0, clip_ids)
        max_cycles = int(cycles.max().detach().cpu()) if cycles.numel() else 0
        for cycle in range(max_cycles):
            mask = (cycles > cycle).reshape(-1, 1)
            next_pos = torch.matmul(rel_pos.unsqueeze(1), cycle_rot).squeeze(1) + cycle_pos
            next_rot = rel_rot @ cycle_rot
            rel_pos = torch.where(mask, next_pos, rel_pos)
            rel_rot = torch.where(mask.unsqueeze(-1), next_rot, rel_rot)

        pos = torch.matmul(rel_pos.unsqueeze(1), root0_rot).squeeze(1) + root0_pos
        rot = rel_rot @ root0_rot
        yaw = tl.heading_yaw_from_root(rot)
        heading = tl.yaw_to_row_matrix(yaw)
        return pos, rot, yaw, heading

    def fk_from_pose(
        self,
        clip_ids: torch.Tensor,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        pose: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = root_pos.shape[0]
        pelvis_rot = tl.rotation_6d_to_matrix(pose["pelvis_rot6"])
        clip_ids_long = clip_ids.to(self.device).long()
        offsets = self.local_offsets.index_select(0, clip_ids_long).clone()
        offsets[:, self.pelvis] = pose["pelvis_pos"]
        if "ik_payload" in pose:
            core_rot = tl.rotation_6d_to_matrix(pose["core_nonpelvis_rot6"])
            identity = torch.eye(3, dtype=root_rot.dtype, device=self.device).expand(b, 3, 3)
            global_pos_list: list[torch.Tensor] = []
            global_rot_list: list[torch.Tensor] = []
            for j in range(self.J):
                if j == self.pelvis:
                    local_rot_j = pelvis_rot
                elif j in self.core_nonpelvis_map:
                    local_rot_j = core_rot[:, self.core_nonpelvis_map[j]]
                else:
                    local_rot_j = identity
                parent = self.parents_body_list[j]
                if parent < 0:
                    rot_j = local_rot_j @ root_rot
                    pos_j = torch.matmul(offsets[:, j].unsqueeze(1), root_rot).squeeze(1) + root_pos
                else:
                    parent_rot = global_rot_list[parent]
                    parent_pos = global_pos_list[parent]
                    rot_j = local_rot_j @ parent_rot
                    pos_j = torch.matmul(offsets[:, j].unsqueeze(1), parent_rot).squeeze(1) + parent_pos
                global_rot_list.append(rot_j)
                global_pos_list.append(pos_j)

            up_hint = torch.matmul(
                self.up_axis.to(dtype=root_rot.dtype),
                root_rot,
            ).squeeze(1)
            payload = pose["ik_payload"]
            cursor = 0
            changed_bones: set[int] = set()
            for limb_i, spec in enumerate(self.ik_limb_specs):
                start = int(spec["start"])
                mid = int(spec["mid"])
                end = int(spec["end"])
                end_root = payload[:, cursor : cursor + 3]
                cursor += 3
                end_rot_root = tl.rotation_6d_to_matrix(payload[:, cursor : cursor + 6])
                cursor += 6
                pole_float = payload[:, cursor]
                cursor += 1

                base = global_pos_list[start]
                base_root = torch.matmul((base - root_pos).unsqueeze(1), root_rot.transpose(-1, -2)).squeeze(1)
                l1 = torch.linalg.norm(offsets[:, mid], dim=-1)
                l2 = torch.linalg.norm(offsets[:, end], dim=-1)
                delta = end_root - base_root
                axis = tl.normalize(delta)
                d = torch.linalg.norm(delta, dim=-1, keepdim=True)
                min_d = (l1 - l2).abs().unsqueeze(-1) + 1e-5
                max_d = (l1 + l2).unsqueeze(-1) - 1e-5
                d_clamped = d.clamp_min(1e-8).clamp(min=min_d, max=max_d)
                end_root = base_root + axis * d_clamped
                rest_axis = self.ik_rest_axis_by_clip.index_select(0, clip_ids_long)[:, limb_i].to(dtype=root_rot.dtype)
                rest_pole = self.ik_rest_pole_by_clip.index_select(0, clip_ids_long)[:, limb_i].to(dtype=root_rot.dtype)
                natural_pole = tl.project_to_plane(tl.swing_only_transport(rest_axis, axis, rest_pole), axis)
                pole_root = tl.rotate_around_axis(natural_pole, axis, pole_float * tl.IK_POLE_ALPHA)
                mid_root = tl.solve_two_bone_with_pole(
                    base_root,
                    end_root,
                    l1,
                    l2,
                    rest_axis,
                    rest_pole,
                    pole_float,
                )
                solved_mid = tl.root_relative_to_world(mid_root.unsqueeze(1), root_pos, root_rot).squeeze(1)
                solved_end = tl.root_relative_to_world(end_root.unsqueeze(1), root_pos, root_rot).squeeze(1)
                end_rot_world = tl.root_relative_rot_to_world(end_rot_root, root_rot)
                world_pole = torch.matmul(pole_root.unsqueeze(1), root_rot).squeeze(1)
                global_pos_list[mid] = solved_mid
                global_pos_list[end] = solved_end
                local_pole_axis = self.ik_local_pole_axis_by_clip.index_select(0, clip_ids_long)[:, limb_i].to(dtype=root_rot.dtype)
                mid_rot_world = tl.rotation_from_axis_and_pole(
                    offsets[:, end],
                    solved_end - solved_mid,
                    local_pole_axis[:, 1],
                    world_pole,
                )
                start_rot_world = tl.rotation_from_axis_and_pole(
                    offsets[:, mid],
                    solved_mid - base,
                    local_pole_axis[:, 0],
                    world_pole,
                )
                global_rot_list[start] = start_rot_world
                global_rot_list[mid] = mid_rot_world
                global_rot_list[end] = end_rot_world
                changed_bones.update((start, mid, end))
                toe = spec.get("toe")
                if toe is not None:
                    toe_i = int(toe)
                    toe_float = payload[:, cursor]
                    cursor += 1
                    toe_offset = offsets[:, toe_i]
                    toe_pos_root = end_root + torch.matmul(toe_offset.unsqueeze(1), end_rot_root).squeeze(1)
                    toe_axis = self.ik_toe_axis_by_clip.index_select(0, clip_ids_long)[:, limb_i].to(dtype=root_rot.dtype)
                    toe_hinge = tl.axis_angle_to_row_matrix(toe_axis, toe_float * tl.IK_TOE_ALPHA)
                    toe_rot_root = toe_hinge @ end_rot_root
                    global_pos_list[toe_i] = tl.root_relative_to_world(toe_pos_root.unsqueeze(1), root_pos, root_rot).squeeze(1)
                    global_rot_list[toe_i] = tl.root_relative_rot_to_world(toe_rot_root, root_rot)
                    changed_bones.add(toe_i)

            dirty_bones = set(changed_bones)
            for j in range(self.J):
                parent = self.parents_body_list[j]
                if parent < 0 or parent not in dirty_bones or j in self.ik_controlled_set:
                    continue
                if j == self.pelvis:
                    local_rot_j = pelvis_rot
                elif j in self.core_nonpelvis_map:
                    local_rot_j = core_rot[:, self.core_nonpelvis_map[j]]
                else:
                    local_rot_j = identity
                parent_rot = global_rot_list[parent]
                parent_pos = global_pos_list[parent]
                global_rot_list[j] = local_rot_j @ parent_rot
                global_pos_list[j] = torch.matmul(offsets[:, j].unsqueeze(1), parent_rot).squeeze(1) + parent_pos
                dirty_bones.add(j)

            global_pos = torch.stack(global_pos_list, dim=1)
            global_rot = torch.stack(global_rot_list, dim=1)
            root_yaw = tl.heading_yaw_from_root(root_rot)
            heading = tl.yaw_to_row_matrix(root_yaw)
            canon = torch.einsum("bjc,bcd->bjd", global_pos - root_pos[:, None, :], heading)
            return global_pos, global_rot, canon

        nonpelvis_rot = tl.rotation_6d_to_matrix(pose["nonpelvis_rot6"])

        global_pos_list: list[torch.Tensor] = []
        global_rot_list: list[torch.Tensor] = []
        for j in range(self.J):
            local_rot_j = pelvis_rot if j == self.pelvis else nonpelvis_rot[:, self.nonpelvis_map[j]]
            parent = self.parents_body_list[j]
            if parent < 0:
                rot_j = local_rot_j @ root_rot
                pos_j = torch.matmul(offsets[:, j].unsqueeze(1), root_rot).squeeze(1) + root_pos
            else:
                parent_rot = global_rot_list[parent]
                parent_pos = global_pos_list[parent]
                rot_j = local_rot_j @ parent_rot
                pos_j = torch.matmul(offsets[:, j].unsqueeze(1), parent_rot).squeeze(1) + parent_pos
            global_rot_list.append(rot_j)
            global_pos_list.append(pos_j)

        global_pos = torch.stack(global_pos_list, dim=1)
        global_rot = torch.stack(global_rot_list, dim=1)
        root_yaw = tl.heading_yaw_from_root(root_rot)
        heading = tl.yaw_to_row_matrix(root_yaw)
        canon = torch.einsum("bjc,bcd->bjd", global_pos - root_pos[:, None, :], heading)
        return global_pos, global_rot, canon

    def fk_positions_from_pose(
        self,
        clip_ids: torch.Tensor,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        pose: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "ik_payload" in pose:
            global_pos, _global_rot, canon = self.fk_from_pose(clip_ids, root_pos, root_rot, pose)
            return global_pos, canon
        if "ik_marker_pos" not in pose:
            global_pos, _global_rot, canon = self.fk_from_pose(clip_ids, root_pos, root_rot, pose)
            return global_pos, canon
        raise ValueError("endpoint-only ik_marker_pos poses are obsolete; use ik_payload")
        b = root_pos.shape[0]
        clip_ids = clip_ids.to(self.device).long()
        pelvis_rot = tl.rotation_6d_to_matrix(pose["pelvis_rot6"])
        local_offsets = self.local_offsets.index_select(0, clip_ids)
        offsets = local_offsets.clone()
        offsets[:, self.pelvis] = pose["pelvis_pos"]
        core_rot = tl.rotation_6d_to_matrix(pose["core_nonpelvis_rot6"])
        identity = torch.eye(3, dtype=root_rot.dtype, device=self.device).expand(b, 3, 3)
        global_pos_list: list[torch.Tensor] = []
        global_rot_list: list[torch.Tensor] = []
        for j in range(self.J):
            if j == self.pelvis:
                local_rot_j = pelvis_rot
            elif j in self.core_nonpelvis_map:
                local_rot_j = core_rot[:, self.core_nonpelvis_map[j]]
            else:
                local_rot_j = identity
            parent = self.parents_body_list[j]
            if parent < 0:
                rot_j = local_rot_j @ root_rot
                pos_j = torch.matmul(offsets[:, j].unsqueeze(1), root_rot).squeeze(1) + root_pos
            else:
                parent_rot = global_rot_list[parent]
                parent_pos = global_pos_list[parent]
                rot_j = local_rot_j @ parent_rot
                pos_j = torch.matmul(offsets[:, j].unsqueeze(1), parent_rot).squeeze(1) + parent_pos
            global_rot_list.append(rot_j)
            global_pos_list.append(pos_j)

        marker_world = tl.root_relative_to_world(pose["ik_marker_pos"], root_pos, root_rot)
        marker_by_name = {name: marker_world[:, marker_i] for name, marker_i in self.ik_marker_index_by_name.items()}
        up_hint = torch.matmul(
            self.up_axis.to(dtype=root_rot.dtype),
            root_rot,
        ).squeeze(1)
        side_hint = torch.matmul(
            self.side_axis.to(dtype=root_rot.dtype),
            root_rot,
        ).squeeze(1)
        for spec in self.ik_limb_specs:
            start = int(spec["start"])
            mid = int(spec["mid"])
            end = int(spec["end"])
            end_name = self.prototype.body_names[end]
            target = marker_by_name.get(end_name, global_pos_list[end])
            base = global_pos_list[start]
            preferred_mid = global_pos_list[mid]
            upper_len = torch.linalg.norm(local_offsets[:, mid], dim=-1)
            lower_len = torch.linalg.norm(local_offsets[:, end], dim=-1)
            pole = side_hint if spec["kind"] == "arm" else up_hint
            if str(spec["side"]) == "r":
                pole = -pole if spec["kind"] == "arm" else pole
            solved_mid, solved_end = tl.solve_two_bone_positions(
                base,
                preferred_mid,
                target,
                upper_len,
                lower_len,
                pole,
            )
            global_pos_list[mid] = solved_mid
            global_pos_list[end] = solved_end
            toe = spec.get("toe")
            if toe is not None:
                toe_i = int(toe)
                toe_name = self.prototype.body_names[toe_i]
                toe_target = marker_by_name.get(toe_name, global_pos_list[toe_i])
                foot_len = torch.linalg.norm(local_offsets[:, toe_i], dim=-1, keepdim=True).clamp_min(1e-6)
                toe_delta = toe_target - solved_end
                toe_dist = torch.linalg.norm(toe_delta, dim=-1, keepdim=True)
                fallback = torch.matmul(local_offsets[:, toe_i].unsqueeze(1), global_rot_list[end]).squeeze(1)
                toe_dir = torch.where(toe_dist > 1e-8, tl.normalize(toe_delta), tl.normalize(fallback))
                global_pos_list[toe_i] = solved_end + toe_dir * foot_len

        global_pos = torch.stack(global_pos_list, dim=1)
        root_local = tl.root_relative_positions(global_pos, root_pos, root_rot)
        if self.ik_marker_indices:
            pose["ik_marker_pos"] = root_local.index_select(1, self.ik_marker_indices_tensor)
        root_yaw = tl.heading_yaw_from_root(root_rot)
        heading = tl.yaw_to_row_matrix(root_yaw)
        canon = torch.einsum("bjc,bcd->bjd", global_pos - root_pos[:, None, :], heading)
        return global_pos, canon


def store_build_input(
    store: ClipStore,
    clip_ids: torch.Tensor,
    prev_idx: torch.Tensor,
    cur_idx: torch.Tensor,
    prev_pose: dict[str, torch.Tensor],
    cur_pose: dict[str, torch.Tensor],
    cfg: tl.TrainConfig,
) -> torch.Tensor:
    current = tl.body_pose_vector(cur_pose)
    previous = tl.body_pose_vector(prev_pose)
    pelvis_vel = (cur_pose["pelvis_pos"] - prev_pose["pelvis_pos"]) / cfg.pose_delta_scale_final
    if "ik_payload" in cur_pose:
        joint_vel = (
            cur_pose["ik_payload"] - prev_pose["ik_payload"]
        ).reshape(cur_idx.shape[0], -1) / cfg.pose_delta_scale_final
    else:
        joint_vel = (cur_pose["canon_pos"] - prev_pose["canon_pos"]).reshape(cur_idx.shape[0], -1) / cfg.pose_delta_scale_final
    root_features = store.get_input_root_features(clip_ids, cur_idx)
    return torch.cat((current, previous, pelvis_vel, joint_vel, root_features), dim=-1)


def store_root_lookahead_features(
    store: ClipStore,
    clip_ids: torch.Tensor,
    cur_idx: torch.Tensor,
    cfg: tl.TrainConfig,
) -> torch.Tensor:
    steps = max(0, int(getattr(cfg, "root_lookahead_steps", 0)))
    if steps <= 0:
        return torch.empty((cur_idx.shape[0], 0), dtype=torch.float32, device=cur_idx.device)
    b = cur_idx.shape[0]
    offsets = torch.arange(1, steps + 2, device=cur_idx.device, dtype=cur_idx.dtype)
    flat_clip_ids = clip_ids.reshape(b, 1).expand(b, steps + 1).reshape(-1)
    flat_idx = (cur_idx.reshape(b, 1) + offsets.reshape(1, steps + 1)).reshape(-1)
    pos, _rot, yaw, heading = store.root_state(flat_clip_ids, flat_idx)
    pos = pos.reshape(b, steps + 1, 3)
    yaw = yaw.reshape(b, steps + 1)
    heading = heading.reshape(b, steps + 1, 3, 3)
    delta_local = torch.matmul((pos[:, 1:] - pos[:, :-1]).unsqueeze(-2), heading[:, :-1]).squeeze(-2)
    features = torch.stack(
        (
            delta_local[:, :, 0] / cfg.max_speed_scale_final,
            delta_local[:, :, 2] / cfg.max_speed_scale_final,
            tl.wrap_angle(yaw[:, 1:] - yaw[:, :-1]) / cfg.max_turn_rate_scale_final,
        ),
        dim=-1,
    )
    return features.reshape(b, steps * 3)


def store_transition_feature_from_next_pose(
    store: ClipStore,
    clip_ids: torch.Tensor,
    prev_idx: torch.Tensor,
    cur_idx: torch.Tensor,
    prev_pose: dict[str, torch.Tensor],
    cur_pose: dict[str, torch.Tensor],
    next_pose: dict[str, torch.Tensor],
    cfg: tl.TrainConfig,
) -> torch.Tensor:
    model_input = store_build_input(store, clip_ids, prev_idx, cur_idx, prev_pose, cur_pose, cfg)
    next_output = tl.pose_target_output(next_pose)
    pelvis_next_vel = (next_pose["pelvis_pos"] - cur_pose["pelvis_pos"]) / cfg.pose_delta_scale_final
    if "ik_payload" in next_pose:
        joint_next_vel = (next_pose["ik_payload"] - cur_pose["ik_payload"]).reshape(cur_idx.shape[0], -1)
    else:
        joint_next_vel = (next_pose["canon_pos"] - cur_pose["canon_pos"]).reshape(cur_idx.shape[0], -1)
    joint_next_vel = joint_next_vel / cfg.pose_delta_scale_final
    parts = [
        model_input,
        next_output,
        next_pose["canon_pos"].reshape(cur_idx.shape[0], -1),
        pelvis_next_vel,
        joint_next_vel,
    ]
    if getattr(cfg, "include_transition_foot_motion", False):
        cur_root_pos, cur_root_rot, _cur_yaw, _cur_heading = store.root_state(clip_ids, cur_idx)
        next_root_pos, next_root_rot, _next_yaw, _next_heading = store.root_state(clip_ids, cur_idx + 1)
        cur_pos, cur_rot, _cur_canon = store.fk_from_pose(clip_ids, cur_root_pos, cur_root_rot, cur_pose)
        next_pos, next_rot, _next_canon = store.fk_from_pose(clip_ids, next_root_pos, next_root_rot, next_pose)
        slide = cp.foot_slide_speeds(
            cur_pos,
            cur_rot,
            next_pos,
            next_rot,
            store.foot_indices,
            store.toe_indices,
            store.fps,
        )
        yaw = cp.foot_vertical_yaw_speeds(
            cur_pos,
            cur_rot,
            next_pos,
            next_rot,
            store.foot_indices,
            store.toe_indices,
            store.fps,
        )
        slide_scale = max(float(getattr(cfg, "foot_slide_scale_mps", 1.0)), 1e-6)
        yaw_scale = max(float(getattr(cfg, "transition_yaw_scale_radps", 10.0)), 1e-6)
        parts.append(torch.cat((slide / slide_scale, yaw / yaw_scale), dim=-1))
    root_lookahead_steps = max(0, int(getattr(cfg, "root_lookahead_steps", 0)))
    if root_lookahead_steps > 0:
        parts.append(store_root_lookahead_features(store, clip_ids, cur_idx, cfg))
    return torch.cat(parts, dim=-1)


def store_transform_transition_feature_to_anchor(
    raw: torch.Tensor,
    store: ClipStore,
    clip_ids: torch.Tensor,
    prev_idx: torch.Tensor,
    cur_idx: torch.Tensor,
    next_idx: torch.Tensor,
    anchor_clip_ids: torch.Tensor,
    anchor_idx: torch.Tensor,
    cfg: tl.TrainConfig,
    schema: dict[str, int],
) -> torch.Tensor:
    out = raw.clone()
    b = raw.shape[0]
    j_count = int(schema["next_canon_dim"]) // 3
    pose_dim = int(schema["pose_dim"])
    velocity_dim = int(schema["velocity_dim"])
    input_dim = int(schema["input_dim"])
    next_canon_start = int(schema["next_canon_start"])
    next_velocity_start = int(schema["next_velocity_start"])

    anchor_pos, _anchor_rot, anchor_yaw, anchor_heading = store.root_state(anchor_clip_ids, anchor_idx)

    def root_info(row_clip_ids: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos, _rot, yaw, _heading = store.root_state(row_clip_ids, idx)
        return pos, yaw

    def anchor_pose_positions(base: int, row_clip_ids: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frame_pos, frame_yaw = root_info(row_clip_ids, idx)
        pelvis_local = raw[:, base : base + 3].reshape(b, 1, 3)
        pelvis_anchor = wae.anchor_local_positions(pelvis_local, frame_pos, frame_yaw, anchor_pos, anchor_yaw).reshape(b, 3)
        canon_start = base + 9
        canon_end = canon_start + j_count * 3
        canon_local = raw[:, canon_start:canon_end].reshape(b, j_count, 3)
        canon_anchor = wae.anchor_local_positions(canon_local, frame_pos, frame_yaw, anchor_pos, anchor_yaw)
        out[:, base : base + 3] = pelvis_anchor
        out[:, canon_start:canon_end] = canon_anchor.reshape(b, j_count * 3)
        return pelvis_anchor, canon_anchor

    prev_pelvis, prev_canon = anchor_pose_positions(pose_dim, clip_ids, prev_idx)
    cur_pelvis, cur_canon = anchor_pose_positions(0, clip_ids, cur_idx)

    velocity_start = pose_dim * 2
    out[:, velocity_start : velocity_start + 3] = (cur_pelvis - prev_pelvis) / cfg.pose_delta_scale_final
    out[:, velocity_start + 3 : velocity_start + velocity_dim] = (
        (cur_canon - prev_canon).reshape(b, j_count * 3) / cfg.pose_delta_scale_final
    )

    prev_root_pos, prev_root_yaw = root_info(clip_ids, prev_idx)
    cur_root_pos, cur_root_yaw = root_info(clip_ids, cur_idx)
    root_start = velocity_start + velocity_dim
    root_delta_anchor = torch.matmul((cur_root_pos - prev_root_pos).unsqueeze(1), anchor_heading).squeeze(1)
    out[:, root_start] = root_delta_anchor[:, 0] / cfg.max_speed_scale_final
    out[:, root_start + 1] = root_delta_anchor[:, 2] / cfg.max_speed_scale_final
    out[:, root_start + 2] = tl.wrap_angle(cur_root_yaw - prev_root_yaw) / cfg.max_turn_rate_scale_final

    future_start = root_start + 3
    for k in range(1, cfg.future_window + 1):
        fut_idx = cur_idx + k
        fut_pos, fut_yaw = root_info(clip_ids, fut_idx)
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
    next_root_pos, next_root_yaw = root_info(clip_ids, next_idx)
    next_pelvis_local = raw[:, next_output_start : next_output_start + 3].reshape(b, 1, 3)
    next_pelvis = wae.anchor_local_positions(
        next_pelvis_local,
        next_root_pos,
        next_root_yaw,
        anchor_pos,
        anchor_yaw,
    ).reshape(b, 3)
    out[:, next_output_start : next_output_start + 3] = next_pelvis

    next_canon = raw[:, next_canon_start : next_canon_start + j_count * 3].reshape(b, j_count, 3)
    next_canon_anchor = wae.anchor_local_positions(next_canon, next_root_pos, next_root_yaw, anchor_pos, anchor_yaw)
    out[:, next_canon_start : next_canon_start + j_count * 3] = next_canon_anchor.reshape(b, j_count * 3)
    out[:, next_velocity_start : next_velocity_start + 3] = (next_pelvis - cur_pelvis) / cfg.pose_delta_scale_final
    out[:, next_velocity_start + 3 : next_velocity_start + velocity_dim] = (
        (next_canon_anchor - cur_canon).reshape(b, j_count * 3) / cfg.pose_delta_scale_final
    )
    return out


def store_slide_speed(
    store: ClipStore,
    cur_pos: torch.Tensor,
    cur_rot: torch.Tensor,
    next_pos: torch.Tensor,
    next_rot: torch.Tensor,
    clip_ids: torch.Tensor,
) -> torch.Tensor:
    speeds = cp.foot_slide_speeds(
        cur_pos,
        cur_rot,
        next_pos,
        next_rot,
        store.foot_indices,
        store.toe_indices,
        store.fps,
    )
    selected, _planted, _heights = cp.planted_foot_values(
        speeds,
        cur_pos,
        cur_rot,
        store.foot_indices,
        store.toe_indices,
    )
    return selected


def store_slide_excess_loss(
    store: ClipStore,
    cur_pos: torch.Tensor,
    cur_rot: torch.Tensor,
    next_pos: torch.Tensor,
    next_rot: torch.Tensor,
    clip_ids: torch.Tensor,
    cur_idx: torch.Tensor,
    special_tolerance_divisor: float = 1.0,
) -> torch.Tensor:
    return store_slide_excess_loss_rows(
        store,
        cur_pos,
        cur_rot,
        next_pos,
        next_rot,
        clip_ids,
        cur_idx,
        special_tolerance_divisor,
    ).mean()


def store_slide_excess_loss_rows(
    store: ClipStore,
    cur_pos: torch.Tensor,
    cur_rot: torch.Tensor,
    next_pos: torch.Tensor,
    next_rot: torch.Tensor,
    clip_ids: torch.Tensor,
    cur_idx: torch.Tensor,
    special_tolerance_divisor: float = 1.0,
) -> torch.Tensor:
    slide_speed = store_slide_speed(
        store,
        cur_pos,
        cur_rot,
        next_pos,
        next_rot,
        clip_ids,
    )
    if store.slide_bound_mps is None:
        raise RuntimeError("slide-excess loss requires slide_bound_mps")
    bounds = store.slide_bound_mps.index_select(0, store.frame_index(clip_ids, cur_idx))
    divisor = max(1.0, float(special_tolerance_divisor))
    if divisor > 1.0:
        rows = clip_ids.to(store.device).long()
        special = store.turn_in_place.index_select(0, rows)
        bounds = torch.where(special, bounds / divisor, bounds)
    return F.relu(slide_speed - bounds)


def store_vertical_yaw_excess_loss(
    store: ClipStore,
    cur_pos: torch.Tensor,
    cur_rot: torch.Tensor,
    next_pos: torch.Tensor,
    next_rot: torch.Tensor,
    clip_ids: torch.Tensor,
    cur_idx: torch.Tensor,
    scale_radps: float,
) -> torch.Tensor:
    return store_vertical_yaw_excess_loss_rows(
        store,
        cur_pos,
        cur_rot,
        next_pos,
        next_rot,
        clip_ids,
        cur_idx,
        scale_radps,
    ).mean()


def store_vertical_yaw_excess_loss_rows(
    store: ClipStore,
    cur_pos: torch.Tensor,
    cur_rot: torch.Tensor,
    next_pos: torch.Tensor,
    next_rot: torch.Tensor,
    clip_ids: torch.Tensor,
    cur_idx: torch.Tensor,
    scale_radps: float,
) -> torch.Tensor:
    speeds = cp.foot_vertical_yaw_speeds(
        cur_pos,
        cur_rot,
        next_pos,
        next_rot,
        store.foot_indices,
        store.toe_indices,
        store.fps,
    )
    selected, _planted, _heights = cp.planted_foot_values(
        speeds,
        cur_pos,
        cur_rot,
        store.foot_indices,
        store.toe_indices,
    )
    if store.yaw_bound_radps is None:
        bounds = torch.zeros_like(selected)
    else:
        bounds = store.yaw_bound_radps.index_select(0, store.frame_index(clip_ids, cur_idx))
    scale = max(float(scale_radps), 1e-6)
    return F.relu(selected - bounds) / scale


@torch.no_grad()
def estimate_forward_yaw_excess_scale(
    model: torch.nn.Module,
    store: ClipStore,
    cfg: tl.TrainConfig,
    device: torch.device,
) -> float:
    """Scale yaw-excess so the current forward-walk checkpoint has max loss 1."""
    if store.yaw_bound_radps is None:
        return 1.0
    candidates = []
    for ci, clip in enumerate(store.clips):
        if bool(store.synthetic[ci].detach().cpu()):
            continue
        stem = Path(clip.path).stem.lower()
        score = 0
        if "walk" in stem:
            score += 1
        if "loop_f" in stem or stem.endswith("_f") or "forward" in stem:
            score += 4
        if "stand" in stem or "idle" in stem or "turn" in stem:
            score -= 2
        candidates.append((score, ci))
    if not candidates:
        return 1.0
    _score, clip_id = max(candidates, key=lambda item: item[0])
    clip = store.clips[clip_id]
    max_cur = int(clip.cyclic_period) - 1 if clip.cyclic_animation else int(clip.T) - transition_feature_horizon(cfg) - 1
    max_cur = max(1, max_cur)
    clip_ids = torch.tensor([clip_id], dtype=torch.long, device=device)
    prev_idx = torch.tensor([0], dtype=torch.long, device=device)
    cur_idx = torch.tensor([1], dtype=torch.long, device=device)
    prev_pose = store.get_pose(clip_ids, prev_idx)
    cur_pose = store.get_pose(clip_ids, cur_idx)
    max_excess = torch.zeros((), device=device)
    was_training = model.training
    model.eval()
    for _step in range(max_cur):
        inp = store_build_input(store, clip_ids, prev_idx, cur_idx, prev_pose, cur_pose, cfg)
        raw_out = tl.predict_next_raw(model, inp, cur_pose, cfg)
        pred_pose, _raw_pose = tl.output_to_pose(raw_out, store.prototype)
        target_idx = cur_idx + 1
        root_pos, root_rot, _yaw, _heading = store.root_state(clip_ids, target_idx)
        pred_global_pos, pred_global_rot, pred_canon = store.fk_from_pose(clip_ids, root_pos, root_rot, pred_pose)
        cur_root_pos, cur_root_rot, _cur_yaw, _cur_heading = store.root_state(clip_ids, cur_idx)
        cur_global_pos, cur_global_rot, _cur_canon = store.fk_from_pose(clip_ids, cur_root_pos, cur_root_rot, cur_pose)
        speeds = cp.foot_vertical_yaw_speeds(
            cur_global_pos,
            cur_global_rot,
            pred_global_pos,
            pred_global_rot,
            store.foot_indices,
            store.toe_indices,
            store.fps,
        )
        speeds, _planted, _heights = cp.planted_foot_values(
            speeds,
            cur_global_pos,
            cur_global_rot,
            store.foot_indices,
            store.toe_indices,
        )
        bounds = store.yaw_bound_radps.index_select(0, store.frame_index(clip_ids, cur_idx))
        max_excess = torch.maximum(max_excess, F.relu(speeds - bounds).amax())
        next_pose = tl.next_pose_from_prediction(pred_pose, pred_canon)
        prev_pose = cur_pose
        cur_pose = next_pose
        prev_idx = cur_idx
        cur_idx = target_idx
    model.train(was_training)
    scale = float(max_excess.detach().cpu())
    return max(scale, 1e-6)


def run_batch_ae_truncated(
    model: torch.nn.Module,
    priors: list[dict[str, object]],
    clips: list[tl.MotionClip],
    batch: list[torch.Tensor],
    cfg: tl.TrainConfig,
    rollout_k: int,
    device: torch.device,
    compute_diagnostics: bool = True,
    compatibility_score_weight: float = 0.0,
    reset_sampler=None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if any(prior_info.get("kind") == "window" for prior_info in priors):
        raise NotImplementedError("Window AE priors currently require row-mixed rollout.")
    clip_indices, starts = batch[0], batch[1]
    init_clip_indices = batch[2] if len(batch) >= 4 else None
    init_starts = batch[3] if len(batch) >= 4 else None
    total_loss = torch.zeros((), device=device)
    groups = {}
    for row, ci in enumerate(clip_indices.tolist()):
        groups.setdefault(ci, []).append(row)
    scores: list[torch.Tensor] = []
    motion_sizes: list[torch.Tensor] = []
    joint_rmses: list[torch.Tensor] = []
    ee_rmses: list[torch.Tensor] = []
    output_mses: list[torch.Tensor] = []
    slide_excess_losses: list[torch.Tensor] = []
    active_step_counts: list[torch.Tensor] = []

    def select_pose_rows(pose: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return {key: value[mask] for key, value in pose.items()}

    def assign_pose_rows(pose: dict[str, torch.Tensor], row_indices: torch.Tensor, source: dict[str, torch.Tensor]) -> None:
        for key in pose:
            pose[key][row_indices] = source[key]

    def gather_initial_poses(row_t: torch.Tensor, fallback_clip: tl.MotionClip, fallback_start: torch.Tensor):
        if init_clip_indices is None or init_starts is None:
            prev_idx_local = fallback_start - 1
            cur_idx_local = fallback_start
            prev_pose_local = tl.get_pose_from_clip(fallback_clip, prev_idx_local, device)
            cur_pose_local = tl.get_pose_from_clip(fallback_clip, cur_idx_local, device)
            return tl.maybe_apply_initial_offsets(
                fallback_clip,
                prev_idx_local,
                cur_idx_local,
                prev_pose_local,
                cur_pose_local,
                cfg,
                device,
            )

        local_init_clip_indices = init_clip_indices[row_t]
        local_init_starts = init_starts[row_t]
        prev_pose_acc = None
        cur_pose_acc = None
        for init_ci in sorted(set(local_init_clip_indices.tolist())):
            local_rows = (local_init_clip_indices == init_ci).nonzero(as_tuple=False).flatten()
            local_rows_dev = local_rows.to(device)
            local_start = local_init_starts[local_rows].long().to(device)
            local_prev_idx = local_start - 1
            local_cur_idx = local_start
            init_clip = clips[int(init_ci)]
            prev_pose_local = tl.get_pose_from_clip(init_clip, local_prev_idx, device)
            cur_pose_local = tl.get_pose_from_clip(init_clip, local_cur_idx, device)
            prev_pose_local, cur_pose_local = tl.maybe_apply_initial_offsets(
                init_clip,
                local_prev_idx,
                local_cur_idx,
                prev_pose_local,
                cur_pose_local,
                cfg,
                device,
            )
            if prev_pose_acc is None:
                prev_pose_acc = {
                    key: torch.empty(
                        (row_t.numel(), *value.shape[1:]),
                        dtype=value.dtype,
                        device=device,
                    )
                    for key, value in prev_pose_local.items()
                }
                cur_pose_acc = {
                    key: torch.empty(
                        (row_t.numel(), *value.shape[1:]),
                        dtype=value.dtype,
                        device=device,
                    )
                    for key, value in cur_pose_local.items()
                }
            for key in prev_pose_acc:
                prev_pose_acc[key][local_rows_dev] = prev_pose_local[key]
                cur_pose_acc[key][local_rows_dev] = cur_pose_local[key]
        return prev_pose_acc, cur_pose_acc

    for ci, rows in groups.items():
        clip = clips[ci]
        row_t = torch.tensor(rows, dtype=torch.long)
        start = starts[row_t].long().to(device)
        prev_idx = start - 1
        cur_idx = start
        prev_pose, cur_pose = gather_initial_poses(row_t, clip, start)
        group_loss = torch.zeros((), device=device)
        cur_root_pos, cur_root_rot, _cur_yaw, _cur_heading = tl.root_state(clip, cur_idx, cfg, device)
        cur_global_pos, cur_global_rot, _cur_canon = tl.fk_from_pose(
            clip, cur_root_pos, cur_root_rot, cur_pose, device
        )
        if reset_sampler is not None:
            effective_lengths = torch.full_like(cur_idx, int(rollout_k))
        elif clip.cyclic_animation:
            effective_lengths = torch.full_like(cur_idx, int(rollout_k))
        else:
            effective_lengths = torch.clamp(clip_future_safe_current_max(clip, cfg) - cur_idx + 1, min=1, max=int(rollout_k))
        group_steps = int(effective_lengths.max().item())
        active_step_counts.append(effective_lengths.float().sum().detach())
        for step in range(group_steps):
            if reset_sampler is not None and step > 0 and (not clip.cyclic_animation):
                expired = cur_idx > clip_future_safe_current_max(clip, cfg)
                if bool(expired.any()):
                    expired_rows = expired.nonzero(as_tuple=False).flatten()
                    remaining_steps = int(rollout_k) - step
                    max_start = clip_sample_start_max(clip, cfg, remaining_steps, cfg.agent_min_cohort_steps)
                    reset_start = torch.randint(
                        1,
                        max_start + 1,
                        (int(expired_rows.numel()),),
                        dtype=cur_idx.dtype,
                        device=device,
                    )
                    reset_prev_idx = reset_start - 1
                    reset_cur_idx = reset_start
                    reset_prev_pose = tl.get_pose_from_clip(clip, reset_prev_idx, device)
                    reset_cur_pose = tl.get_pose_from_clip(clip, reset_cur_idx, device)
                    reset_prev_pose, reset_cur_pose = tl.maybe_apply_initial_offsets(
                        clip,
                        reset_prev_idx,
                        reset_cur_idx,
                        reset_prev_pose,
                        reset_cur_pose,
                        cfg,
                        device,
                    )
                    prev_idx[expired_rows] = reset_prev_idx
                    cur_idx[expired_rows] = reset_cur_idx
                    assign_pose_rows(prev_pose, expired_rows, reset_prev_pose)
                    assign_pose_rows(cur_pose, expired_rows, reset_cur_pose)
                    cur_root_pos, cur_root_rot, _cur_yaw, _cur_heading = tl.root_state(clip, cur_idx, cfg, device)
                    cur_global_pos, cur_global_rot, _cur_canon = tl.fk_from_pose(
                        clip, cur_root_pos, cur_root_rot, cur_pose, device
                    )
            active = step < effective_lengths
            if not bool(active.any()):
                continue
            inp = tl.build_input(clip, prev_idx, cur_idx, prev_pose, cur_pose, cfg, device)
            raw_out = tl.predict_next_raw(model, inp, cur_pose, cfg)
            pred_pose, raw_pose = tl.output_to_pose(raw_out, clip)
            target_idx = torch.where(active, cur_idx + 1, cur_idx)
            tensors = clip.tensors(device)
            root_pos, root_rot, _yaw, _heading = tl.root_state(clip, target_idx, cfg, device)
            pred_global_pos, pred_global_rot, pred_canon = tl.fk_from_pose(clip, root_pos, root_rot, pred_pose, device)
            next_pose = tl.next_pose_from_prediction(pred_pose, pred_canon)
            prior_scores = [
                (
                    ae_score(
                    prior_info["model"],
                    prior_info["mean"],
                    prior_info["std"],
                    tae.transition_feature_from_next_pose(
                        clip,
                        prev_idx,
                        cur_idx,
                        prev_pose,
                        cur_pose,
                        next_pose,
                        prior_transition_cfg(prior_info, cfg),
                        device,
                    )[active],
                    cfg.ae_score_loss,
                    cfg.ae_huber_delta,
                    compatibility_score_weight,
                    ),
                    float(prior_info.get("weight", 1.0)),
                )
                for prior_info in priors
            ]
            score = (
                torch.stack([raw_score * weight for raw_score, weight in prior_scores]).sum()
                if prior_scores
                else torch.zeros((), device=device)
            )
            step_loss = cfg.ae_loss_weight * score
            if cfg.slide_excess_loss_weight > 0.0:
                raise RuntimeError(
                    "excess-envelope slide-excess loss requires row-mixed rollout; "
                    "the old scalar fallback has been removed"
                )
            term_mask = torch.zeros(cur_idx.shape[0], dtype=torch.bool, device=device)
            if cfg.enable_contact_physics_losses:
                target_pose = tl.get_pose_from_clip(clip, target_idx[active], device)
                physics_cfg = copy.copy(cfg)
                physics_cfg.alpha0_pelvis_location = 0.0
                physics_cfg.alpha1_pelvis_rotation = 0.0
                physics_cfg.alpha2_pose_rotation = 0.0
                physics_cfg.alpha3_pose_6d_aux = 0.0
                physics_cfg.alpha4_end_effector_location = 0.0
                physics_cfg.alpha5_end_effector_rotation = 0.0
                physics_cfg.alpha6_full_body_location = 0.0
                physics_loss, _physics_parts, _physics_next_pose, active_term_mask = tl.compute_losses(
                    clip,
                    select_pose_rows(prev_pose, active),
                    select_pose_rows(cur_pose, active),
                    select_pose_rows(pred_pose, active),
                    select_pose_rows(raw_pose, active),
                    target_pose,
                    prev_idx[active],
                    cur_idx[active],
                    target_idx[active],
                    physics_cfg,
                    device,
                )
                term_mask[active] = active_term_mask
                step_loss = step_loss + physics_loss
            group_loss = group_loss + step_loss / rollout_k
            scores.append(score.detach())
            motion_sizes.append(
                (next_pose["canon_pos"][active].detach() - cur_pose["canon_pos"][active].detach())
                .square()
                .mean()
                .sqrt()
            )
            if compute_diagnostics:
                target_pose = tl.get_pose_from_clip(clip, target_idx[active], device)
                target_global_pos, _target_global_rot = tl.global_from_clip(clip, target_idx[active], cfg, device)
                joint_rmses.append(
                    (pred_global_pos[active].detach() - target_global_pos.detach())
                    .square()
                    .sum(dim=-1)
                    .mean()
                    .sqrt()
                )
                ee_idx = tensors["end_effectors"]
                ee_rmses.append(
                    (
                        pred_global_pos[active].index_select(1, ee_idx)
                        - target_global_pos.index_select(1, ee_idx)
                    )
                    .square()
                    .sum(dim=-1)
                    .mean()
                    .sqrt()
                    .detach()
                )
                output_mses.append(
                    F.mse_loss(
                        tl.pose_target_output(select_pose_rows(next_pose, active)),
                        tl.pose_target_output(target_pose),
                    )
                    .detach()
                )
            if cfg.enable_early_termination and cfg.restart_on_termination and step + 1 < rollout_k and bool(term_mask.any()):
                remaining_steps = rollout_k - step - 1
                max_start = tl.clip_rollout_max_start(clip, remaining_steps, cfg)
                if max_start < 1:
                    term_mask = torch.zeros_like(term_mask)
                    continue
                restart_start = torch.randint(1, max_start + 1, cur_idx.shape, device=device)
                restart_prev_idx = restart_start - 1
                restart_cur_idx = restart_start
                restart_prev_pose = tl.get_pose_from_clip(clip, restart_prev_idx, device)
                restart_cur_pose = tl.get_pose_from_clip(clip, restart_cur_idx, device)
                restart_prev_pose, restart_cur_pose = tl.maybe_apply_initial_offsets(
                    clip,
                    restart_prev_idx,
                    restart_cur_idx,
                    restart_prev_pose,
                    restart_cur_pose,
                    cfg,
                    device,
                )
                prev_pose = tl.blend_pose_by_mask(cur_pose, restart_prev_pose, term_mask)
                cur_pose = tl.blend_pose_by_mask(next_pose, restart_cur_pose, term_mask)
                prev_idx = torch.where(term_mask, restart_prev_idx, cur_idx)
                cur_idx = torch.where(term_mask, restart_cur_idx, target_idx)
                cur_root_pos, cur_root_rot, _cur_yaw, _cur_heading = tl.root_state(clip, cur_idx, cfg, device)
                cur_global_pos, cur_global_rot, _cur_canon = tl.fk_from_pose(
                    clip, cur_root_pos, cur_root_rot, cur_pose, device
                )
                continue
            prev_pose = tl.blend_pose_by_mask(prev_pose, cur_pose, active)
            cur_pose = tl.blend_pose_by_mask(cur_pose, next_pose, active)
            prev_idx = torch.where(active, cur_idx, prev_idx)
            cur_idx = torch.where(active, target_idx, cur_idx)
            cur_global_pos = torch.where(active.reshape(-1, 1, 1), pred_global_pos, cur_global_pos)
            cur_global_rot = torch.where(active.reshape(-1, 1, 1, 1), pred_global_rot, cur_global_rot)
        total_loss = total_loss + group_loss
    total_loss = total_loss / max(1, len(groups))
    def mean_metric(values: list[torch.Tensor]) -> float:
        if not values:
            return 0.0
        return float(torch.stack(values).mean().cpu())

    return total_loss, {
        "total": float(total_loss.detach().cpu()),
        "ae_score": mean_metric(scores),
        "canon_step_rms": mean_metric(motion_sizes),
        "joint_rmse": mean_metric(joint_rmses),
        "ee_rmse": mean_metric(ee_rmses),
        "output_mse": mean_metric(output_mses),
        "slide_excess": mean_metric(slide_excess_losses),
        "active_fraction": (
            float(torch.stack(active_step_counts).sum().item()) / max(1.0, float(rollout_k * len(clip_indices)))
            if active_step_counts
            else 0.0
        ),
    }


def run_batch_ae_resetting(
    model: torch.nn.Module,
    priors: list[dict[str, object]],
    clips: list[tl.MotionClip],
    batch: list[torch.Tensor],
    cfg: tl.TrainConfig,
    rollout_k: int,
    device: torch.device,
    reset_sampler,
    compute_diagnostics: bool = True,
    compatibility_score_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if any(prior_info.get("kind") == "window" for prior_info in priors):
        raise NotImplementedError("Window AE priors currently require row-mixed rollout.")
    clip_indices, starts = batch[0].long().cpu(), batch[1].long().cpu()
    init_clip_indices = batch[2].long().cpu() if len(batch) >= 4 else None
    init_starts = batch[3].long().cpu() if len(batch) >= 4 else None
    batch_size = int(clip_indices.shape[0])
    cur_clip_indices = clip_indices.clone()
    prev_idx_cpu = starts - 1
    cur_idx_cpu = starts.clone()
    total_loss = torch.zeros((), device=device)

    score_values: list[torch.Tensor] = []
    score_counts: list[int] = []
    motion_values: list[torch.Tensor] = []
    motion_counts: list[int] = []
    joint_values: list[torch.Tensor] = []
    joint_counts: list[int] = []
    ee_values: list[torch.Tensor] = []
    ee_counts: list[int] = []
    output_values: list[torch.Tensor] = []
    output_counts: list[int] = []
    slide_excess_values: list[torch.Tensor] = []
    slide_excess_counts: list[int] = []

    def select_pose_rows(pose: dict[str, torch.Tensor], rows_dev: torch.Tensor) -> dict[str, torch.Tensor]:
        return {key: value.index_select(0, rows_dev) for key, value in pose.items()}

    def assign_pose_rows(target: dict[str, torch.Tensor], rows_dev: torch.Tensor, source: dict[str, torch.Tensor]) -> None:
        for key in target:
            target[key][rows_dev] = source[key]

    def rows_by_clip(clip_ids: torch.Tensor) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = {}
        for row, ci in enumerate(clip_ids.tolist()):
            groups.setdefault(int(ci), []).append(row)
        return groups

    def gather_pose_pair(clip_ids: torch.Tensor, starts_cpu: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        prev_pose_acc = None
        cur_pose_acc = None
        for ci, rows in rows_by_clip(clip_ids).items():
            rows_cpu = torch.tensor(rows, dtype=torch.long)
            rows_dev = rows_cpu.to(device)
            local_start = starts_cpu.index_select(0, rows_cpu).to(device)
            local_prev_idx = local_start - 1
            local_cur_idx = local_start
            clip = clips[ci]
            prev_pose_local = tl.get_pose_from_clip(clip, local_prev_idx, device)
            cur_pose_local = tl.get_pose_from_clip(clip, local_cur_idx, device)
            prev_pose_local, cur_pose_local = tl.maybe_apply_initial_offsets(
                clip,
                local_prev_idx,
                local_cur_idx,
                prev_pose_local,
                cur_pose_local,
                cfg,
                device,
            )
            if prev_pose_acc is None:
                prev_pose_acc = {
                    key: torch.empty((clip_ids.numel(), *value.shape[1:]), dtype=value.dtype, device=device)
                    for key, value in prev_pose_local.items()
                }
                cur_pose_acc = {
                    key: torch.empty((clip_ids.numel(), *value.shape[1:]), dtype=value.dtype, device=device)
                    for key, value in cur_pose_local.items()
                }
            assign_pose_rows(prev_pose_acc, rows_dev, prev_pose_local)
            assign_pose_rows(cur_pose_acc, rows_dev, cur_pose_local)
        assert prev_pose_acc is not None and cur_pose_acc is not None
        return prev_pose_acc, cur_pose_acc

    if init_clip_indices is None or init_starts is None:
        prev_pose, cur_pose = gather_pose_pair(cur_clip_indices, starts)
    else:
        prev_pose, cur_pose = gather_pose_pair(init_clip_indices, init_starts)

    def reset_rows(rows: list[int], remaining_steps: int) -> None:
        nonlocal cur_clip_indices, prev_idx_cpu, cur_idx_cpu, prev_pose, cur_pose
        if not rows or remaining_steps <= 0:
            return
        rows_cpu = torch.tensor(rows, dtype=torch.long)
        rows_dev = rows_cpu.to(device)
        new_clip_ids, new_starts = reset_sampler(len(rows), max(1, int(remaining_steps)))
        new_clip_ids = new_clip_ids.long().cpu()
        new_starts = new_starts.long().cpu()
        new_prev_pose, new_cur_pose = gather_pose_pair(new_clip_ids, new_starts)
        cur_clip_indices[rows_cpu] = new_clip_ids
        prev_idx_cpu[rows_cpu] = new_starts - 1
        cur_idx_cpu[rows_cpu] = new_starts
        assign_pose_rows(prev_pose, rows_dev, new_prev_pose)
        assign_pose_rows(cur_pose, rows_dev, new_cur_pose)

    def weighted_metric(values: list[torch.Tensor], counts: list[int]) -> float:
        if not values:
            return 0.0
        denom = max(1, int(sum(counts)))
        return float((torch.stack(values).sum() / denom).cpu())

    for step in range(int(rollout_k)):
        if step > 0:
            expired_rows = [
                row
                for row, ci in enumerate(cur_clip_indices.tolist())
                if (not clips[int(ci)].cyclic_animation)
                and int(cur_idx_cpu[row]) > clip_future_safe_current_max(clips[int(ci)], cfg)
            ]
            reset_rows(expired_rows, int(rollout_k) - step)

        step_loss_sum = torch.zeros((), device=device)
        step_count = 0
        for ci, rows in rows_by_clip(cur_clip_indices).items():
            clip = clips[ci]
            rows_cpu = torch.tensor(rows, dtype=torch.long)
            rows_dev = rows_cpu.to(device)
            n_rows = int(rows_cpu.numel())
            local_prev_idx = prev_idx_cpu.index_select(0, rows_cpu).to(device)
            local_cur_idx = cur_idx_cpu.index_select(0, rows_cpu).to(device)
            local_target_idx = local_cur_idx + 1
            local_prev_pose = select_pose_rows(prev_pose, rows_dev)
            local_cur_pose = select_pose_rows(cur_pose, rows_dev)

            inp = tl.build_input(clip, local_prev_idx, local_cur_idx, local_prev_pose, local_cur_pose, cfg, device)
            raw_out = tl.predict_next_raw(model, inp, local_cur_pose, cfg)
            pred_pose, raw_pose = tl.output_to_pose(raw_out, clip)
            root_pos, root_rot, _yaw, _heading = tl.root_state(clip, local_target_idx, cfg, device)
            pred_global_pos, pred_global_rot, pred_canon = tl.fk_from_pose(clip, root_pos, root_rot, pred_pose, device)
            next_pose = tl.next_pose_from_prediction(pred_pose, pred_canon)
            prior_scores = [
                ae_score(
                    prior_info["model"],
                    prior_info["mean"],
                    prior_info["std"],
                    tae.transition_feature_from_next_pose(
                        clip,
                        local_prev_idx,
                        local_cur_idx,
                        local_prev_pose,
                        local_cur_pose,
                        next_pose,
                        prior_transition_cfg(prior_info, cfg),
                        device,
                    ),
                    cfg.ae_score_loss,
                    cfg.ae_huber_delta,
                    compatibility_score_weight,
                )
                for prior_info in priors
            ]
            score = torch.stack(prior_scores).mean()
            step_loss = cfg.ae_loss_weight * score
            if cfg.slide_excess_loss_weight > 0.0:
                raise RuntimeError(
                    "excess-envelope slide-excess loss requires row-mixed rollout; "
                    "the old scalar fallback has been removed"
                )

            if cfg.enable_contact_physics_losses:
                target_pose = tl.get_pose_from_clip(clip, local_target_idx, device)
                physics_cfg = copy.copy(cfg)
                physics_cfg.alpha0_pelvis_location = 0.0
                physics_cfg.alpha1_pelvis_rotation = 0.0
                physics_cfg.alpha2_pose_rotation = 0.0
                physics_cfg.alpha3_pose_6d_aux = 0.0
                physics_cfg.alpha4_end_effector_location = 0.0
                physics_cfg.alpha5_end_effector_rotation = 0.0
                physics_cfg.alpha6_full_body_location = 0.0
                physics_loss, _physics_parts, _physics_next_pose, _term_mask = tl.compute_losses(
                    clip,
                    local_prev_pose,
                    local_cur_pose,
                    pred_pose,
                    raw_pose,
                    target_pose,
                    local_prev_idx,
                    local_cur_idx,
                    local_target_idx,
                    physics_cfg,
                    device,
                )
                step_loss = step_loss + physics_loss

            step_loss_sum = step_loss_sum + step_loss * n_rows
            step_count += n_rows
            score_values.append(score.detach() * n_rows)
            score_counts.append(n_rows)
            motion_values.append(
                (next_pose["canon_pos"].detach() - local_cur_pose["canon_pos"].detach())
                .square()
                .mean()
                .sqrt()
                * n_rows
            )
            motion_counts.append(n_rows)
            if compute_diagnostics:
                tensors = clip.tensors(device)
                target_pose = tl.get_pose_from_clip(clip, local_target_idx, device)
                target_global_pos, _target_global_rot = tl.global_from_clip(clip, local_target_idx, cfg, device)
                joint_values.append(
                    (pred_global_pos.detach() - target_global_pos.detach())
                    .square()
                    .sum(dim=-1)
                    .mean()
                    .sqrt()
                    * n_rows
                )
                joint_counts.append(n_rows)
                ee_idx = tensors["end_effectors"]
                ee_values.append(
                    (
                        pred_global_pos.index_select(1, ee_idx) - target_global_pos.index_select(1, ee_idx)
                    )
                    .square()
                    .sum(dim=-1)
                    .mean()
                    .sqrt()
                    .detach()
                    * n_rows
                )
                ee_counts.append(n_rows)
                output_values.append(
                    F.mse_loss(tl.pose_target_output(next_pose), tl.pose_target_output(target_pose)).detach() * n_rows
                )
                output_counts.append(n_rows)

            assign_pose_rows(prev_pose, rows_dev, local_cur_pose)
            assign_pose_rows(cur_pose, rows_dev, next_pose)
            prev_idx_cpu[rows_cpu] = local_cur_idx.detach().cpu()
            cur_idx_cpu[rows_cpu] = local_target_idx.detach().cpu()

        if step_count > 0:
            total_loss = total_loss + (step_loss_sum / step_count) / max(1, int(rollout_k))

    return total_loss, {
        "total": float(total_loss.detach().cpu()),
        "ae_score": weighted_metric(score_values, score_counts),
        "canon_step_rms": weighted_metric(motion_values, motion_counts),
        "joint_rmse": weighted_metric(joint_values, joint_counts),
        "ee_rmse": weighted_metric(ee_values, ee_counts),
        "output_mse": weighted_metric(output_values, output_counts),
        "slide_excess": weighted_metric(slide_excess_values, slide_excess_counts),
        "active_fraction": 1.0,
    }


def run_batch_ae_store(
    model: torch.nn.Module,
    priors: list[dict[str, object]],
    store: ClipStore,
    batch: list[torch.Tensor],
    cfg: tl.TrainConfig,
    rollout_k: int,
    device: torch.device,
    compute_diagnostics: bool = True,
    compatibility_score_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    clip_ids = batch[0].long().to(device)
    initial_clip_ids = clip_ids.detach().clone()
    starts = batch[1].long().to(device)
    init_clip_ids = None
    init_starts = None
    episode_lengths = None
    if len(batch) == 3:
        episode_lengths = batch[2].long().to(device).clamp(1, int(rollout_k))
    elif len(batch) >= 4:
        init_clip_ids = batch[2].long().to(device)
        init_starts = batch[3].long().to(device)
        if len(batch) >= 5:
            episode_lengths = batch[4].long().to(device).clamp(1, int(rollout_k))
    prev_idx = starts - 1
    cur_idx = starts.clone()
    if init_clip_ids is None or init_starts is None:
        prev_pose = store.get_pose(clip_ids, prev_idx)
        cur_pose = store.get_pose(clip_ids, cur_idx)
    else:
        prev_pose = store.get_pose(init_clip_ids, init_starts - 1)
        cur_pose = store.get_pose(init_clip_ids, init_starts)

    total_loss = torch.zeros((), device=device)
    row_loss_accum = torch.zeros_like(clip_ids, dtype=torch.float32, device=device)
    score_values: list[torch.Tensor] = []
    motion_values: list[torch.Tensor] = []
    joint_values: list[torch.Tensor] = []
    ee_values: list[torch.Tensor] = []
    output_values: list[torch.Tensor] = []
    slide_excess_values: list[torch.Tensor] = []
    yaw_excess_values: list[torch.Tensor] = []
    motion_floor_values: list[torch.Tensor] = []
    prior_raw_values: dict[str, list[torch.Tensor]] = {
        str(prior_info.get("label", f"prior_{i}")): []
        for i, prior_info in enumerate(priors)
    }
    prior_weighted_values: dict[str, list[torch.Tensor]] = {
        str(prior_info.get("label", f"prior_{i}")): []
        for i, prior_info in enumerate(priors)
    }
    transition_priors = [prior_info for prior_info in priors if prior_info.get("kind") != "window"]
    window_priors = [prior_info for prior_info in priors if prior_info.get("kind") == "window"]
    window_buffers: list[list[torch.Tensor]] = [[] for _prior_info in window_priors]
    window_clip_buffers: list[list[torch.Tensor]] = [[] for _prior_info in window_priors]
    window_prev_idx_buffers: list[list[torch.Tensor]] = [[] for _prior_info in window_priors]
    window_cur_idx_buffers: list[list[torch.Tensor]] = [[] for _prior_info in window_priors]
    window_next_idx_buffers: list[list[torch.Tensor]] = [[] for _prior_info in window_priors]
    window_valid_counts = [
        torch.zeros_like(clip_ids, dtype=torch.long, device=device)
        for _prior_info in window_priors
    ]
    if transition_priors:
        ae_denominator = max(1, int(rollout_k))
    elif window_priors:
        min_window = min(int(prior_info["window_size"]) for prior_info in window_priors)
        ae_denominator = max(1, int(rollout_k) - min_window + 1)
    else:
        ae_denominator = max(1, int(rollout_k))

    def reset_rows(mask: torch.Tensor, requested_steps: int | torch.Tensor) -> None:
        nonlocal clip_ids, prev_idx, cur_idx, prev_pose, cur_pose
        if not bool(mask.any()):
            return
        rows = mask.nonzero(as_tuple=False).flatten()
        if isinstance(requested_steps, torch.Tensor):
            row_requested = requested_steps.index_select(0, rows).clamp_min(1)
        else:
            row_requested = torch.full((rows.numel(),), max(1, int(requested_steps)), dtype=torch.long, device=device)
        for requested in torch.unique(row_requested).tolist():
            requested_int = max(1, int(requested))
            sub = rows[row_requested == requested_int]
            if sub.numel() == 0:
                continue
            row_synthetic = store.synthetic.index_select(0, clip_ids.index_select(0, sub))
            new_clip_ids, new_starts = store.sample_starts_for_existing_groups(
                row_synthetic,
                requested_int,
                require_full=episode_lengths is not None,
            )
            new_prev_pose = store.get_pose(new_clip_ids, new_starts - 1)
            new_cur_pose = store.get_pose(new_clip_ids, new_starts)
            clip_ids[sub] = new_clip_ids
            prev_idx[sub] = new_starts - 1
            cur_idx[sub] = new_starts
            for key in prev_pose:
                replaced_prev = prev_pose[key].clone()
                replaced_cur = cur_pose[key].clone()
                replaced_prev[sub] = new_prev_pose[key]
                replaced_cur[sub] = new_cur_pose[key]
                prev_pose[key] = replaced_prev
                cur_pose[key] = replaced_cur
        for valid_counts in window_valid_counts:
            valid_counts[rows] = 0

    for step in range(int(rollout_k)):
        if episode_lengths is not None and step > 0:
            boundary = torch.remainder(torch.full_like(episode_lengths, step), episode_lengths) == 0
            if bool(boundary.any()):
                remaining = torch.full_like(episode_lengths, int(rollout_k) - step)
                reset_rows(boundary, torch.minimum(episode_lengths, remaining).clamp_min(1))
        if step > 0:
            safe_max = store.future_safe_max.index_select(0, clip_ids)
            expired = torch.logical_and(~store.cyclic.index_select(0, clip_ids), cur_idx > safe_max)
            reset_rows(expired, int(rollout_k) - step)

        inp = store_build_input(store, clip_ids, prev_idx, cur_idx, prev_pose, cur_pose, cfg)
        raw_out = tl.predict_next_raw(model, inp, cur_pose, cfg)
        pred_pose, raw_pose = tl.output_to_pose(raw_out, store.prototype)
        target_idx = cur_idx + 1
        root_pos, root_rot, _yaw, _heading = store.root_state(clip_ids, target_idx)
        needs_pred_global_rot = (
            cfg.slide_excess_loss_weight > 0.0
            or cfg.yaw_excess_loss_weight > 0.0
            or compute_diagnostics
        )
        if needs_pred_global_rot:
            pred_global_pos, pred_global_rot, pred_canon = store.fk_from_pose(clip_ids, root_pos, root_rot, pred_pose)
        else:
            pred_global_pos, pred_canon = store.fk_positions_from_pose(clip_ids, root_pos, root_rot, pred_pose)
            pred_global_rot = None
        next_pose = tl.next_pose_from_prediction(pred_pose, pred_canon)
        prior_scores: list[tuple[torch.Tensor, float, str]] = []
        for prior_info in transition_priors:
            raw_score_rows = ae_score_rows(
                prior_info["model"],
                prior_info["mean"],
                prior_info["std"],
                store_transition_feature_from_next_pose(
                    store,
                    clip_ids,
                    prev_idx,
                    cur_idx,
                    prev_pose,
                    cur_pose,
                    next_pose,
                    prior_transition_cfg(prior_info, cfg),
                ),
                cfg.ae_score_loss,
                cfg.ae_huber_delta,
                compatibility_score_weight,
            )
            prior_scores.append(
                (
                    raw_score_rows,
                    float(prior_info.get("weight", 1.0)),
                    str(prior_info.get("label", "transition")),
                )
            )
        for prior_i, prior_info in enumerate(window_priors):
            features = store_transition_feature_from_next_pose(
                store,
                clip_ids,
                prev_idx,
                cur_idx,
                prev_pose,
                cur_pose,
                next_pose,
                prior_transition_cfg(prior_info, cfg),
            )
            window_size = int(prior_info["window_size"])
            if bool(prior_info.get("anchor_first_root", False)):
                window_buffers[prior_i].append(features)
                window_clip_buffers[prior_i].append(clip_ids.detach().clone())
                window_prev_idx_buffers[prior_i].append(prev_idx.detach().clone())
                window_cur_idx_buffers[prior_i].append(cur_idx.detach().clone())
                window_next_idx_buffers[prior_i].append((cur_idx + 1).detach().clone())
            else:
                x = (features - prior_info["mean"]) / prior_info["std"]
                window_buffers[prior_i].append(x)
            if len(window_buffers[prior_i]) > window_size:
                window_buffers[prior_i] = window_buffers[prior_i][-window_size:]
                window_clip_buffers[prior_i] = window_clip_buffers[prior_i][-window_size:]
                window_prev_idx_buffers[prior_i] = window_prev_idx_buffers[prior_i][-window_size:]
                window_cur_idx_buffers[prior_i] = window_cur_idx_buffers[prior_i][-window_size:]
                window_next_idx_buffers[prior_i] = window_next_idx_buffers[prior_i][-window_size:]
            window_valid_counts[prior_i] = window_valid_counts[prior_i] + 1
            if len(window_buffers[prior_i]) >= window_size:
                ready = window_valid_counts[prior_i] >= window_size
                if bool(ready.any()):
                    if bool(prior_info.get("anchor_first_root", False)):
                        schema = dict(prior_info.get("schema", {}))
                        if not schema:
                            raise ValueError(f"Anchored window prior {prior_info.get('path')} is missing schema metadata.")
                        anchor_clip_ids = window_clip_buffers[prior_i][-window_size][ready]
                        anchor_idx = window_cur_idx_buffers[prior_i][-window_size][ready]
                        parts = []
                        for raw_part, clip_part, prev_part, cur_part, next_part in zip(
                            window_buffers[prior_i][-window_size:],
                            window_clip_buffers[prior_i][-window_size:],
                            window_prev_idx_buffers[prior_i][-window_size:],
                            window_cur_idx_buffers[prior_i][-window_size:],
                            window_next_idx_buffers[prior_i][-window_size:],
                        ):
                            anchored = store_transform_transition_feature_to_anchor(
                                raw_part[ready],
                                store,
                                clip_part[ready],
                                prev_part[ready],
                                cur_part[ready],
                                next_part[ready],
                                anchor_clip_ids,
                                anchor_idx,
                                prior_transition_cfg(prior_info, cfg),
                                schema,
                            )
                            parts.append((anchored - prior_info["mean"]) / prior_info["std"])
                        window_x = torch.cat(parts, dim=-1)
                    else:
                        window_x = torch.cat([part[ready] for part in window_buffers[prior_i][-window_size:]], dim=-1)
                    ready_score_rows = ae_score_normalized_rows(
                        prior_info["model"],
                        window_x,
                        cfg.ae_score_loss,
                        cfg.ae_huber_delta,
                        compatibility_score_weight,
                    )
                    raw_score_rows = torch.zeros_like(clip_ids, dtype=torch.float32, device=device)
                    raw_score_rows[ready] = ready_score_rows
                    prior_scores.append(
                        (
                            raw_score_rows,
                            float(prior_info.get("weight", 1.0)),
                            str(prior_info.get("label", f"window_w{window_size:02d}")),
                        )
                    )
        score_rows = (
            torch.stack([raw_score_rows * weight for raw_score_rows, weight, _label in prior_scores], dim=0).sum(dim=0)
            if prior_scores
            else torch.zeros_like(clip_ids, dtype=torch.float32, device=device)
        )
        score = reduce_ae_score_rows(score_rows, cfg)
        if prior_scores:
            total_loss = total_loss + cfg.ae_loss_weight * score / ae_denominator
            row_loss_accum = row_loss_accum + (cfg.ae_loss_weight * score_rows / ae_denominator).detach()
            for raw_score_rows, weight, label in prior_scores:
                raw_score = raw_score_rows.mean()
                prior_raw_values.setdefault(label, []).append(raw_score.detach())
                prior_weighted_values.setdefault(label, []).append((raw_score * weight).detach())
        step_loss = torch.zeros((), device=device)
        slide_excess_loss = torch.zeros((), device=device)
        needs_current_global = cfg.slide_excess_loss_weight > 0.0 or cfg.yaw_excess_loss_weight > 0.0
        cur_global_pos: torch.Tensor | None = None
        cur_global_rot: torch.Tensor | None = None
        if needs_current_global:
            cur_root_pos, cur_root_rot, _cur_yaw, _cur_heading = store.root_state(clip_ids, cur_idx)
            cur_global_pos, cur_global_rot, _cur_canon = store.fk_from_pose(
                clip_ids,
                cur_root_pos,
                cur_root_rot,
                cur_pose,
            )
        if cfg.slide_excess_loss_weight > 0.0:
            assert cur_global_pos is not None and cur_global_rot is not None
            assert pred_global_rot is not None
            slide_excess_rows = store_slide_excess_loss_rows(
                store,
                cur_global_pos,
                cur_global_rot,
                pred_global_pos,
                pred_global_rot,
                clip_ids,
                cur_idx,
                cfg.turn_slide_bound_divisor,
            )
            slide_excess_loss = slide_excess_rows.mean()
            step_loss = step_loss + cfg.slide_excess_loss_weight * slide_excess_loss
            row_loss_accum = row_loss_accum + (
                cfg.slide_excess_loss_weight * slide_excess_rows / max(1, int(rollout_k))
            ).detach()
        yaw_excess_loss = torch.zeros((), device=device)
        if cfg.yaw_excess_loss_weight > 0.0:
            assert cur_global_pos is not None and cur_global_rot is not None
            assert pred_global_rot is not None
            yaw_excess_rows = store_vertical_yaw_excess_loss_rows(
                store,
                cur_global_pos,
                cur_global_rot,
                pred_global_pos,
                pred_global_rot,
                clip_ids,
                cur_idx,
                cfg.yaw_excess_scale_radps,
            )
            yaw_excess_loss = yaw_excess_rows.mean()
            step_loss = step_loss + cfg.yaw_excess_loss_weight * yaw_excess_loss
            row_loss_accum = row_loss_accum + (
                cfg.yaw_excess_loss_weight * yaw_excess_rows / max(1, int(rollout_k))
            ).detach()
        motion_floor_loss = torch.zeros((), device=device)
        if cfg.motion_floor_loss_weight > 0.0:
            motion_schema = tae.transition_schema(store.prototype, cfg)
            source_cur_pose_for_motion = store.get_pose(clip_ids, cur_idx)
            source_prev_pose_for_motion = store.get_pose(clip_ids, prev_idx)
            target_pose_for_motion = store.get_pose(clip_ids, target_idx)
            generated_features_for_motion = store_transition_feature_from_next_pose(
                store,
                clip_ids,
                prev_idx,
                cur_idx,
                prev_pose,
                cur_pose,
                next_pose,
                cfg,
            )
            target_features_for_motion = store_transition_feature_from_next_pose(
                store,
                clip_ids,
                prev_idx,
                cur_idx,
                source_prev_pose_for_motion,
                source_cur_pose_for_motion,
                target_pose_for_motion,
                cfg,
            )
            motion_start = int(motion_schema["next_velocity_start"])
            motion_end = int(motion_schema["transition_foot_motion_start"])
            generated_motion = generated_features_for_motion[:, motion_start:motion_end].square().mean(dim=-1).sqrt()
            target_motion = target_features_for_motion[:, motion_start:motion_end].square().mean(dim=-1).sqrt().detach()
            motion_floor_loss = F.relu(cfg.motion_floor_margin * target_motion - generated_motion).mean()
            step_loss = step_loss + cfg.motion_floor_loss_weight * motion_floor_loss

        if cfg.enable_contact_physics_losses:
            raise NotImplementedError("AE rollouts currently require --no-contact-physics-losses")

        total_loss = total_loss + step_loss / max(1, int(rollout_k))
        if prior_scores:
            score_values.append(score.detach())
        slide_excess_values.append(slide_excess_loss.detach())
        yaw_excess_values.append(yaw_excess_loss.detach())
        motion_floor_values.append(motion_floor_loss.detach())
        motion_values.append(
            (next_pose["canon_pos"].detach() - cur_pose["canon_pos"].detach())
            .square()
            .mean()
            .sqrt()
        )
        if compute_diagnostics:
            target_pose = store.get_pose(clip_ids, target_idx)
            target_root_pos, target_root_rot, _target_yaw, _target_heading = store.root_state(clip_ids, target_idx)
            target_global_pos, _target_global_rot, _target_canon = store.fk_from_pose(
                clip_ids,
                target_root_pos,
                target_root_rot,
                target_pose,
            )
            joint_values.append(
                (pred_global_pos.detach() - target_global_pos.detach())
                .square()
                .sum(dim=-1)
                .mean()
                .sqrt()
            )
            ee_idx = store.prototype.end_effectors_tensor.to(device)
            ee_values.append(
                (
                    pred_global_pos.index_select(1, ee_idx)
                    - target_global_pos.index_select(1, ee_idx)
                )
                .square()
                .sum(dim=-1)
                .mean()
                .sqrt()
                .detach()
            )
            output_values.append(
                F.mse_loss(tl.pose_target_output(next_pose), tl.pose_target_output(target_pose)).detach()
            )

        prev_pose = cur_pose
        cur_pose = next_pose
        prev_idx = cur_idx
        cur_idx = target_idx

    def mean_metric(values: list[torch.Tensor]) -> float:
        if not values:
            return 0.0
        return float(torch.stack(values).mean().cpu())

    return total_loss, {
        "total": float(total_loss.detach().cpu()),
        "ae_score": mean_metric(score_values),
        "canon_step_rms": mean_metric(motion_values),
        "joint_rmse": mean_metric(joint_values),
        "ee_rmse": mean_metric(ee_values),
        "output_mse": mean_metric(output_values),
        "slide_excess": mean_metric(slide_excess_values),
        "yaw_excess": mean_metric(yaw_excess_values),
        "motion_floor": mean_metric(motion_floor_values),
        "active_fraction": 1.0,
        "__clip_ids": initial_clip_ids.detach().cpu(),
        "__row_loss": row_loss_accum.detach().cpu(),
        **{f"ae_raw/{label}": mean_metric(values) for label, values in prior_raw_values.items()},
        **{f"ae_weighted/{label}": mean_metric(values) for label, values in prior_weighted_values.items()},
    }


def run_batch_ae(
    model: torch.nn.Module,
    priors: list[dict[str, object]],
    clips: list[tl.MotionClip],
    batch: list[torch.Tensor],
    cfg: tl.TrainConfig,
    rollout_k: int,
    device: torch.device,
    compute_diagnostics: bool = True,
    compatibility_score_weight: float = 0.0,
    reset_sampler=None,
    store: ClipStore | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if store is not None:
        return run_batch_ae_store(
            model,
            priors,
            store,
            batch,
            cfg,
            rollout_k,
            device,
            compute_diagnostics=compute_diagnostics,
            compatibility_score_weight=compatibility_score_weight,
        )
    return run_batch_ae_truncated(
        model,
        priors,
        clips,
        batch,
        cfg,
        rollout_k,
        device,
        compute_diagnostics=compute_diagnostics,
        compatibility_score_weight=compatibility_score_weight,
        reset_sampler=reset_sampler,
    )


def train(args: argparse.Namespace) -> None:
    process_start_time = time.perf_counter()
    cfg = tl.TrainConfig()
    cfg.hidden_dim = args.hidden_dim
    cfg.num_hidden_layers = args.num_hidden_layers
    cfg.learning_rate = args.learning_rate
    cfg.batch_size = args.batch_size
    cfg.max_epochs = args.max_epochs
    cfg.rollout_schedule = tl.parse_rollout_schedule(args.rollout_schedule)
    cfg.curriculum_max_epochs_per_stage = args.curriculum_max_epochs_per_stage
    cfg.curriculum_stall_patience_epochs = args.curriculum_stall_patience_epochs
    cfg.curriculum_min_delta = args.curriculum_min_delta
    cfg.curriculum_min_epochs = args.curriculum_min_epochs
    cfg.val_fraction = 0.0
    cfg.disable_validation = True
    cfg.use_torch_compile = bool(args.compile) and not args.no_compile
    cfg.torch_compile_mode = args.compile_mode
    cfg.predict_residual = args.predict_residual
    cfg.zero_init_output = args.zero_init_output
    cfg.run_name = args.run_name
    if args.date_prefix_run_name:
        cfg.run_name = tl.date_prefixed_run_name(cfg.run_name)
    cfg.device = args.device
    cfg.cyclic_animation = args.cyclic_animation
    cfg.pose_representation = "ik_markers"
    cfg.agent_min_cohort_steps = max(1, int(args.agent_min_cohort_steps))
    cfg.gradient_accumulation_batches = max(1, int(args.gradient_accumulation_batches))
    cfg.periodic_sampling_weight = max(0.0, float(args.periodic_sampling_weight))
    cfg.nonperiodic_sampling_weight = max(0.0, float(args.nonperiodic_sampling_weight))
    cfg.synthetic_agent_fraction = min(1.0, max(0.0, float(args.synthetic_agent_fraction)))
    cfg.init_pose_sampling = args.init_pose_sampling
    cfg.agent_fixed_start_frame = max(0, int(args.agent_fixed_start_frame))
    cfg.enable_contact_physics_losses = False
    cfg.enable_early_termination = args.enable_early_termination
    cfg.restart_on_termination = not args.no_restart_on_termination
    cfg.reset_exhausted_agents = bool(args.reset_exhausted_agents)
    cfg.freefall_body_height_offset_m = args.freefall_body_height_offset_m
    cfg.freefall_initial_offset_history = max(1, int(args.freefall_initial_offset_history))
    cfg.freefall_initial_contacts_off = bool(args.freefall_initial_contacts_off)
    cfg.alpha7_contact_label = args.alpha7_contact_label
    cfg.alpha8_foot_penetration = args.alpha8_foot_penetration
    cfg.alpha9_foot_sliding = args.alpha9_foot_sliding
    cfg.alpha10_freefall = args.alpha10_freefall
    cfg.alpha11_contact_height = args.alpha11_contact_height
    cfg.alpha12_termination = args.alpha12_termination
    cfg.ae_loss_weight = args.ae_loss_weight
    cfg.ae_score_loss = args.ae_score_loss
    cfg.ae_huber_delta = args.ae_huber_delta
    cfg.ae_row_top_fraction = min(1.0, max(0.0, float(args.ae_row_top_fraction)))
    cfg.ae_row_top_weight = max(0.0, float(args.ae_row_top_weight))
    cfg.slide_excess_loss_weight = args.slide_excess_loss_weight
    cfg.turn_slide_bound_divisor = max(1.0, float(args.turn_slide_bound_divisor))
    cfg.excess_envelope_enabled = bool(args.excess_envelope)
    cfg.excess_envelope_knn = max(1, int(args.excess_envelope_knn))
    cfg.excess_envelope_margin = float(args.excess_envelope_margin)
    cfg.excess_envelope_cache_dir = str(args.excess_envelope_cache_dir)
    cfg.yaw_excess_loss_weight = float(args.yaw_excess_loss_weight)
    cfg.yaw_excess_scale_radps = max(0.0, float(args.yaw_excess_scale_radps))
    cfg.yaw_excess_scale_checkpoint = str(args.yaw_excess_scale_checkpoint or "")
    cfg.motion_floor_loss_weight = max(0.0, float(args.motion_floor_loss_weight))
    cfg.motion_floor_margin = max(0.0, float(args.motion_floor_margin))
    cfg.timed_checkpoint_interval_minutes = max(0.0, float(args.timed_checkpoint_interval_minutes))
    if cfg.slide_excess_loss_weight > 0.0:
        if not cfg.excess_envelope_enabled:
            raise ValueError("--slide-excess-loss-weight requires --excess-envelope")
    tl.set_seed(cfg.seed)
    device = torch.device(cfg.device)
    tl.apply_cuda_performance_settings(cfg, device)
    profiler = tl.TimingProfiler(args.profile_timing, device, args.profile_sync_cuda)

    real_clip_specs = tl.clip_specs_from_folders(args.folder_path, args.periodic_folder_path, args.nonperiodic_folder_path)
    synthetic_clip_specs = [(folder, False) for folder in tl.parse_path_list(args.synthetic_folder_path)]
    real_clip_count = sum(
        len(list(tl.npz_folder_from_path(folder).glob("*.npz")))
        for folder, _cyclic in real_clip_specs
    )
    synthetic_clip_count = sum(
        len(list(tl.npz_folder_from_path(folder).glob("*.npz")))
        for folder, _cyclic in synthetic_clip_specs
    )
    if cfg.synthetic_agent_fraction > 0.0 and synthetic_clip_count <= 0:
        raise ValueError("--synthetic-agent-fraction > 0 requires --synthetic-folder-path with .npz files")
    clip_specs = real_clip_specs + synthetic_clip_specs
    synthetic_clip_indices = set(range(real_clip_count, real_clip_count + synthetic_clip_count))
    real_clip_indices = [i for i in range(real_clip_count)]
    with profiler.section("setup/load_npz_and_prior"):
        clips = tl.load_clips_from_specs(clip_specs, cfg)
        if real_clip_count <= 0:
            raise ValueError("At least one real/non-synthetic clip is required")
        prior_paths = [tl.resolve_path(args.prior_checkpoint)]
        prior_paths.extend(tl.resolve_path(path) for path in args.extra_prior_checkpoint)
        if len(args.extra_prior_weight) > len(args.extra_prior_checkpoint):
            raise ValueError("--extra-prior-weight was provided more times than --extra-prior-checkpoint")
        prior_weights = [float(args.prior_weight)]
        prior_weights.extend(float(weight) for weight in args.extra_prior_weight)
        while len(prior_weights) < len(prior_paths):
            prior_weights.append(1.0)
        active_prior_pairs = [(path, weight) for path, weight in zip(prior_paths, prior_weights) if abs(float(weight)) > 0.0]
        if not active_prior_pairs:
            raise ValueError("At least one AE prior weight must be non-zero")
        prior_paths = [path for path, _weight in active_prior_pairs]
        prior_weights = [float(weight) for _path, weight in active_prior_pairs]
        priors = load_prior_bundle(prior_paths, device, prior_weights)
        print(
            "AE priors: "
            + ", ".join(
                f"{prior_info.get('label')} weight={float(prior_info.get('weight', 1.0)):.6g} "
                f"path={prior_info.get('path')}"
                for prior_info in priors
            ),
            flush=True,
        )
        max_prior_root_lookahead = max(
            [max(0, int(getattr(prior_transition_cfg(prior_info, cfg), "root_lookahead_steps", 0))) for prior_info in priors]
            or [0]
        )
        if max_prior_root_lookahead > max(0, int(getattr(cfg, "root_lookahead_steps", 0))):
            cfg.root_lookahead_steps = max_prior_root_lookahead
            print(
                f"transition feature horizon uses root_lookahead_steps={cfg.root_lookahead_steps} from AE prior",
                flush=True,
            )
    with profiler.section("setup/model_optimizer_compile"):
        input_dim, output_dim = tl.make_batch_dims(clips[0], cfg)
        model = tl.MLPController(input_dim, output_dim, cfg).to(device)
        model, compile_enabled = tl.maybe_compile_model(model, input_dim, cfg, device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if args.resume_checkpoint:
        resume_path = tl.resolve_path(args.resume_checkpoint)
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        tl.unwrap_compiled_model(model).load_state_dict(resume["model"])
        print(f"resumed model weights from {resume_path}", flush=True)
        if args.resume_optimizer:
            if "optimizer" not in resume:
                raise KeyError(f"checkpoint has no optimizer state: {resume_path}")
            opt.load_state_dict(resume["optimizer"])
            print(f"resumed optimizer state from {resume_path}", flush=True)
    run_dir = tl.resolve_path(cfg.output_dir) / cfg.run_name
    ckpt_dir = run_dir / "checkpoints"
    writer = SummaryWriter(run_dir / "tb")
    print(f"pure_ae run_dir={run_dir}", flush=True)
    print(f"pure_ae best_checkpoint={ckpt_dir / 'checkpoint_best.pt'}", flush=True)
    schedule = tuple(max(1, int(k)) for k in cfg.rollout_schedule) or (1,)
    if args.visual_reporter:
        print("visual reporter disabled: use the standalone model viewer instead", flush=True)
    metadata = {
        "npz_folders": [
            {"path": str(tl.npz_folder_from_path(path)), "cyclic": cyclic}
            for path, cyclic in real_clip_specs
        ],
        "synthetic_npz_folders": [
            {"path": str(tl.npz_folder_from_path(path)), "cyclic": cyclic}
            for path, cyclic in synthetic_clip_specs
        ],
        "real_clip_count": int(real_clip_count),
        "synthetic_folder_path": str(args.synthetic_folder_path or ""),
        "synthetic_clip_count": int(synthetic_clip_count),
        "synthetic_agent_fraction": float(cfg.synthetic_agent_fraction),
        "synthetic_policy": (
            "synthetic clips are controller rollout roots only; AE prior training must not include them"
            if synthetic_clip_count
            else "none"
        ),
        "body_names": clips[0].body_names,
        "parents_body": clips[0].parents_body.tolist(),
        "pelvis_index": clips[0].pelvis,
        "non_pelvis_indices": clips[0].non_pelvis,
        "end_effector_indices": clips[0].end_effectors,
        "pose_representation": cfg.pose_representation,
        "ik_marker_names": clips[0].ik_marker_names,
        "ik_marker_indices": clips[0].ik_marker_indices,
        "core_non_pelvis_indices": clips[0].core_non_pelvis,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "ae_prior_checkpoint": str(prior_paths[0]),
        "ae_prior_checkpoints": [str(path) for path in prior_paths],
        "ae_prior_weights": prior_weights,
        "real_clip_paths": [str(c.path) for c in clips[:real_clip_count]],
        "synthetic_clip_paths": [str(c.path) for c in clips[real_clip_count:]],
        "loss_type": "pure_transition_ae_prior",
        "ae_score_loss": args.ae_score_loss,
        "ae_huber_delta": args.ae_huber_delta,
        "slide_excess_loss_weight": cfg.slide_excess_loss_weight,
        "turn_slide_bound_divisor": cfg.turn_slide_bound_divisor,
        "excess_envelope_enabled": cfg.excess_envelope_enabled,
        "excess_envelope_knn": cfg.excess_envelope_knn,
        "excess_envelope_margin": cfg.excess_envelope_margin,
        "excess_envelope_cache_dir": cfg.excess_envelope_cache_dir,
        "yaw_excess_loss_weight": cfg.yaw_excess_loss_weight,
        "yaw_excess_scale_radps": cfg.yaw_excess_scale_radps,
        "yaw_excess_scale_checkpoint": cfg.yaw_excess_scale_checkpoint,
        "motion_floor_loss_weight": cfg.motion_floor_loss_weight,
        "motion_floor_margin": cfg.motion_floor_margin,
        "init_pose_sampling": cfg.init_pose_sampling,
        "agent_min_cohort_steps": cfg.agent_min_cohort_steps,
        "gradient_accumulation_batches": cfg.gradient_accumulation_batches,
        "periodic_sampling_weight": cfg.periodic_sampling_weight,
        "nonperiodic_sampling_weight": cfg.nonperiodic_sampling_weight,
        "rollout_truncation_policy": (
            "nonperiodic rows reset independently to fresh starts in the same clip before target or future-root frames are missing; periodic rows use cyclic indexing"
            if cfg.reset_exhausted_agents
            else "nonperiodic rows contribute until their future-root window would pass clip end; periodic rows use full requested K"
        ),
        "agent_clip_sampling_policy": (
            (
                "per-row random clips: periodic group total weight "
                f"{cfg.periodic_sampling_weight:g}, nonperiodic group total weight {cfg.nonperiodic_sampling_weight:g}, "
                f"synthetic fixed agent fraction {cfg.synthetic_agent_fraction:g}"
            )
            if cfg.reset_exhausted_agents
            else "weighted by inverse expected active rollout steps to reduce truncation imbalance"
        ),
        "rollout_data_path": "gpu_resident_per_row",
        "row_mixed_rollout_forced": True,
        "final_stage_random_rollout": bool(args.final_stage_random_rollout),
        "final_stage_random_rollout_choices": list(schedule),
        "mixed_rollout_cohorts": bool(args.mixed_rollout_cohorts),
        "mixed_rollout_cohort_choices": (
            [int(k) for k in tl.parse_rollout_schedule(args.mixed_rollout_cohort_schedule)]
            if args.mixed_rollout_cohort_schedule
            else list(schedule)
        ),
        "mixed_rollout_cohort_weights": [
            float(part)
            for part in args.mixed_rollout_cohort_weights.replace(";", ",").split(",")
            if part.strip()
        ],
        "adaptive_clip_sampling": bool(args.adaptive_clip_sampling),
        "adaptive_clip_score_k": int(args.adaptive_clip_score_k),
        "adaptive_clip_score_ema": float(args.adaptive_clip_score_ema),
        "adaptive_clip_score_floor": float(args.adaptive_clip_score_floor),
        "adaptive_clip_score_batch_size": int(args.adaptive_clip_score_batch_size),
        "adaptive_clip_leaderboard_top_n": int(args.adaptive_clip_leaderboard_top_n),
        "adaptive_clip_leaderboard_every_epochs": int(args.adaptive_clip_leaderboard_every_epochs),
        "compile_enabled": compile_enabled,
    }

    rollout_idx = 0
    if args.initial_rollout_k is not None:
        initial_k = max(1, int(args.initial_rollout_k))
        if initial_k not in schedule:
            raise ValueError(f"--initial-rollout-k {initial_k} is not present in --rollout-schedule {schedule}")
        rollout_idx = schedule.index(initial_k)
    rollout_k = schedule[rollout_idx]
    with profiler.section("setup/resident_clips"):
        store = ClipStore(clips, cfg, device, synthetic_clip_indices=synthetic_clip_indices)
        if store.excess_envelope_metadata:
            metadata["excess_envelope"] = dict(store.excess_envelope_metadata)
        if cfg.yaw_excess_loss_weight > 0.0 and cfg.yaw_excess_scale_radps <= 0.0:
            scale = estimate_forward_yaw_excess_scale(model, store, cfg, device)
            cfg.yaw_excess_scale_radps = scale
            metadata["yaw_excess_scale_radps"] = float(scale)
            metadata["yaw_excess_auto_scale_radps"] = float(scale)
            print(
                f"auto yaw_excess_scale_radps={scale:.6g} "
                "(max forward-walk autoreg excess on the current checkpoint)",
                flush=True,
            )
    agent_rng = random.Random(cfg.seed + 4817)
    agent_coverage_order: list[tuple[int, int]] = []
    agent_coverage_cursor = 0
    stage_clip_indices: list[int] = []
    stage_clip_weights: list[float] = []
    synthetic_stage_clip_indices: list[int] = []
    synthetic_stage_clip_weights: list[float] = []
    adaptive_scores = torch.zeros((len(clips),), dtype=torch.float32, device=device)
    adaptive_seen = torch.zeros((len(clips),), dtype=torch.bool, device=device)
    adaptive_counts = torch.zeros((len(clips),), dtype=torch.long, device=device)

    def make_loader(max_rollout: int) -> tuple[tl.MotionIndexDataset, DataLoader]:
        dataset = AnyStartDataset(clips, cfg, max_rollout)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        return dataset, loader

    def refresh_stage_sampling(max_rollout: int) -> None:
        nonlocal stage_clip_indices, stage_clip_weights, synthetic_stage_clip_indices, synthetic_stage_clip_weights
        eligible_indices = [ci for ci, clip in enumerate(clips) if clip_has_any_start(clip, cfg)]
        stage_clip_indices = [ci for ci in eligible_indices if ci not in synthetic_clip_indices]
        synthetic_stage_clip_indices = [ci for ci in eligible_indices if ci in synthetic_clip_indices]
        if not stage_clip_indices:
            raise ValueError("No clips cover one-step training")
        if cfg.synthetic_agent_fraction > 0.0 and not synthetic_stage_clip_indices:
            raise ValueError("Synthetic agent fraction requested, but no synthetic clips cover one-step training")
        periodic_indices = [ci for ci in stage_clip_indices if clips[ci].cyclic_animation]
        nonperiodic_indices = [ci for ci in stage_clip_indices if not clips[ci].cyclic_animation]
        group_counts = {
            True: max(1, len(periodic_indices)),
            False: max(1, len(nonperiodic_indices)),
        }
        group_weights = {
            True: cfg.periodic_sampling_weight,
            False: cfg.nonperiodic_sampling_weight,
        }
        if cfg.reset_exhausted_agents:
            base_stage_clip_weights = [
                group_weights[clips[ci].cyclic_animation] / group_counts[clips[ci].cyclic_animation]
                for ci in stage_clip_indices
            ]
        else:
            expected_steps = [
                expected_active_rollout_steps(clips[ci], cfg, max_rollout, cfg.agent_min_cohort_steps)
                for ci in stage_clip_indices
            ]
            base_stage_clip_weights = [
                group_weights[clips[ci].cyclic_animation]
                / group_counts[clips[ci].cyclic_animation]
                / max(1e-6, steps)
                for ci, steps in zip(stage_clip_indices, expected_steps)
            ]
        use_adaptive = False
        if args.adaptive_clip_sampling:
            stage_tensor = torch.tensor(stage_clip_indices, dtype=torch.long, device=device)
            use_adaptive = bool(adaptive_seen.index_select(0, stage_tensor).all().detach().cpu())
        if use_adaptive:
            floor = max(0.0, float(args.adaptive_clip_score_floor))
            stage_clip_weights = [
                max(floor, float(adaptive_scores[int(ci)].detach().cpu()))
                for ci in stage_clip_indices
            ]
        else:
            stage_clip_weights = base_stage_clip_weights
        if not any(weight > 0.0 for weight in stage_clip_weights):
            stage_clip_weights = [1.0 for _ in stage_clip_indices]
        synthetic_stage_clip_weights = [1.0 for _ in synthetic_stage_clip_indices]
        if store is not None:
            store.update_stage_sampling(
                stage_clip_indices,
                stage_clip_weights,
                synthetic_stage_clip_indices,
                synthetic_stage_clip_weights,
            )

    def adaptive_score_rollout_k(clip_index: int, requested_k: int) -> int:
        clip = clips[int(clip_index)]
        requested_k = max(1, int(requested_k))
        if clip.cyclic_animation:
            return requested_k
        return max(1, min(requested_k, int(clip.T) - transition_feature_horizon(cfg) - 1))

    def grade_adaptive_clips(max_rollout: int, picked_clip_ids: list[int]) -> dict[str, float]:
        if not args.adaptive_clip_sampling:
            return {}
        if store is None:
            raise ValueError("--adaptive-clip-sampling requires row-mixed rollout")
        if not stage_clip_indices:
            return {}

        stage_set = set(int(ci) for ci in stage_clip_indices)
        unseen = [
            int(ci)
            for ci in stage_clip_indices
            if not bool(adaptive_seen[int(ci)].detach().cpu())
        ]
        if unseen:
            candidates = unseen
        else:
            candidates = sorted({int(ci) for ci in picked_clip_ids if int(ci) in stage_set})
        if not candidates:
            return {}

        requested_k = int(args.adaptive_clip_score_k) if int(args.adaptive_clip_score_k) > 0 else int(max_rollout)
        batch_size = max(1, int(args.adaptive_clip_score_batch_size))
        was_training = model.training
        observed: dict[int, list[float]] = {}
        model.eval()
        with torch.no_grad():
            by_k: dict[int, list[int]] = {}
            for ci in candidates:
                k_eff = adaptive_score_rollout_k(ci, requested_k)
                by_k.setdefault(k_eff, []).append(ci)
            for k_eff, group in sorted(by_k.items()):
                for start_i in range(0, len(group), batch_size):
                    chunk = group[start_i : start_i + batch_size]
                    starts = []
                    for ci in chunk:
                        max_start = clip_sample_start_max(clips[ci], cfg, k_eff, cfg.agent_min_cohort_steps)
                        if cfg.agent_fixed_start_frame > 0:
                            starts.append(min(int(cfg.agent_fixed_start_frame), max_start))
                        else:
                            starts.append(agent_rng.randint(1, max_start))
                    score_batch = [
                        torch.tensor(chunk, dtype=torch.long),
                        torch.tensor(starts, dtype=torch.long),
                    ]
                    _loss, scalars = run_batch_ae_store(
                        model,
                        priors,
                        store,
                        score_batch,
                        cfg,
                        k_eff,
                        device,
                        compute_diagnostics=False,
                        compatibility_score_weight=args.compatibility_score_weight,
                    )
                    row_clip_ids = scalars.get("__clip_ids")
                    row_losses = scalars.get("__row_loss")
                    if not isinstance(row_clip_ids, torch.Tensor) or not isinstance(row_losses, torch.Tensor):
                        raise RuntimeError("adaptive scoring expected row-wise clip ids and losses")
                    for row_ci, row_loss in zip(row_clip_ids.tolist(), row_losses.tolist()):
                        observed.setdefault(int(row_ci), []).append(float(row_loss))
        if was_training:
            model.train()

        ema = min(1.0, max(0.0, float(args.adaptive_clip_score_ema)))
        for ci, values in observed.items():
            value = float(sum(values) / max(1, len(values)))
            if bool(adaptive_seen[ci].detach().cpu()):
                adaptive_scores[ci] = (1.0 - ema) * adaptive_scores[ci] + ema * value
            else:
                adaptive_scores[ci] = value
                adaptive_seen[ci] = True
            adaptive_counts[ci] += 1

        refresh_stage_sampling(max_rollout)
        stage_tensor = torch.tensor(stage_clip_indices, dtype=torch.long, device=device)
        stage_seen = adaptive_seen.index_select(0, stage_tensor)
        seen_fraction = float(stage_seen.float().mean().detach().cpu())
        seen_scores = adaptive_scores.index_select(0, stage_tensor)[stage_seen]
        score_mean = float(seen_scores.mean().detach().cpu()) if seen_scores.numel() else 0.0
        score_max = float(seen_scores.max().detach().cpu()) if seen_scores.numel() else 0.0
        score_min = float(seen_scores.min().detach().cpu()) if seen_scores.numel() else 0.0
        if observed:
            print(
                f"adaptive_clip_sampling graded={len(observed)} seen={seen_fraction:.3f} "
                f"active={1 if seen_fraction >= 1.0 else 0} score_mean={score_mean:.6g} "
                f"score_min={score_min:.6g} score_max={score_max:.6g}",
                flush=True,
            )
        return {
            "graded": float(len(observed)),
            "seen_fraction": seen_fraction,
            "active": 1.0 if seen_fraction >= 1.0 else 0.0,
            "score_mean": score_mean,
            "score_min": score_min,
            "score_max": score_max,
        }

    def adaptive_clip_leaderboard_rows() -> list[dict[str, object]]:
        if not args.adaptive_clip_sampling or not stage_clip_indices:
            return []
        weights = [max(0.0, float(w)) for w in stage_clip_weights]
        weight_sum = sum(weights)
        rows: list[dict[str, object]] = []
        for ci, weight in zip(stage_clip_indices, weights):
            score = float(adaptive_scores[int(ci)].detach().cpu())
            seen = bool(adaptive_seen[int(ci)].detach().cpu())
            rows.append(
                {
                    "rank": 0,
                    "clip_index": int(ci),
                    "clip_name": Path(clips[int(ci)].path).stem,
                    "clip_path": str(clips[int(ci)].path),
                    "score": score,
                    "sample_probability": (weight / weight_sum) if weight_sum > 0.0 else 0.0,
                    "grade_count": int(adaptive_counts[int(ci)].detach().cpu()),
                    "seen": int(seen),
                    "cyclic": int(bool(clips[int(ci)].cyclic_animation)),
                    "frames": int(clips[int(ci)].T),
                }
            )
        rows.sort(key=lambda row: (float(row["score"]), float(row["sample_probability"])), reverse=True)
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        return rows

    def write_adaptive_clip_leaderboard(epoch: int) -> None:
        if not args.adaptive_clip_sampling:
            return
        every = max(1, int(args.adaptive_clip_leaderboard_every_epochs))
        if epoch % every != 0:
            return
        rows = adaptive_clip_leaderboard_rows()
        if not rows:
            return
        board_dir = run_dir / "adaptive_clip_leaderboard"
        board_dir.mkdir(parents=True, exist_ok=True)
        latest_csv = board_dir / "latest.csv"
        latest_json = board_dir / "latest_top.json"
        fields = [
            "rank",
            "clip_index",
            "clip_name",
            "score",
            "sample_probability",
            "grade_count",
            "seen",
            "cyclic",
            "frames",
            "clip_path",
        ]
        with latest_csv.open("w", newline="", encoding="utf-8") as f:
            writer_csv = csv.DictWriter(f, fieldnames=fields)
            writer_csv.writeheader()
            writer_csv.writerows(rows)
        top_n = max(1, int(args.adaptive_clip_leaderboard_top_n))
        top_rows = rows[:top_n]
        latest_json.write_text(
            json.dumps({"epoch": int(epoch), "top": top_rows}, indent=2),
            encoding="utf-8",
        )
        history_csv = board_dir / "history_top.csv"
        write_header = not history_csv.exists()
        with history_csv.open("a", newline="", encoding="utf-8") as f:
            writer_csv = csv.DictWriter(f, fieldnames=["epoch", *fields])
            if write_header:
                writer_csv.writeheader()
            for row in top_rows:
                writer_csv.writerow({"epoch": int(epoch), **row})
        writer.add_text(
            "adaptive/leaderboard_top",
            "\n".join(
                f"{int(row['rank']):02d}. {row['clip_name']} "
                f"score={float(row['score']):.6g} "
                f"p={float(row['sample_probability']):.3f} "
                f"n={int(row['grade_count'])}"
                for row in top_rows[: min(10, top_n)]
            ),
            epoch,
        )
        top_line = "; ".join(
            f"{row['clip_name']}={float(row['score']):.4g}/p{float(row['sample_probability']):.2f}"
            for row in top_rows[: min(5, top_n)]
        )
        if top_line:
            print(f"adaptive_clip_leaderboard epoch={epoch} top: {top_line}", flush=True)

    def reset_agent_coverage_order(dataset: tl.MotionIndexDataset) -> None:
        nonlocal agent_coverage_order, agent_coverage_cursor
        agent_coverage_order = list(dataset.items)
        agent_rng.shuffle(agent_coverage_order)
        agent_coverage_cursor = 0

    def coverage_agent_start(dataset: tl.MotionIndexDataset) -> tuple[int, int]:
        nonlocal agent_coverage_cursor
        if not agent_coverage_order or agent_coverage_cursor >= len(agent_coverage_order):
            reset_agent_coverage_order(dataset)
        item = agent_coverage_order[agent_coverage_cursor]
        agent_coverage_cursor += 1
        return item

    def random_agent_start(max_rollout: int, clip_index: int | None = None) -> tuple[int, int]:
        if clip_index is None:
            ci = agent_rng.choices(stage_clip_indices, weights=stage_clip_weights, k=1)[0]
        else:
            ci = int(clip_index)
        max_start = clip_sample_start_max(clips[ci], cfg, max_rollout, cfg.agent_min_cohort_steps)
        if cfg.agent_fixed_start_frame > 0:
            return ci, min(cfg.agent_fixed_start_frame, max_start)
        return ci, agent_rng.randint(1, max_start)

    def random_initial_pose_start() -> tuple[int, int]:
        ci = agent_rng.randrange(len(clips))
        max_start = max(1, clips[ci].cyclic_period - 1 if clips[ci].cyclic_animation else clips[ci].T - 1)
        if cfg.agent_fixed_start_frame > 0:
            return ci, min(cfg.agent_fixed_start_frame, max_start)
        return ci, agent_rng.randint(1, max_start)

    def random_agent_reset_batch(count: int, max_rollout: int) -> tuple[torch.Tensor, torch.Tensor]:
        ci = agent_rng.choices(stage_clip_indices, weights=stage_clip_weights, k=1)[0]
        max_start = clip_sample_start_max(clips[ci], cfg, max_rollout, cfg.agent_min_cohort_steps)
        clip_ids = [ci] * int(count)
        starts = []
        for _ in range(int(count)):
            if cfg.agent_fixed_start_frame > 0:
                starts.append(min(cfg.agent_fixed_start_frame, max_start))
            else:
                starts.append(agent_rng.randint(1, max_start))
        return torch.tensor(clip_ids, dtype=torch.long), torch.tensor(starts, dtype=torch.long)

    def mixed_rollout_choices(max_rollout: int) -> list[int]:
        if not args.mixed_rollout_cohorts:
            return []
        raw = (
            tl.parse_rollout_schedule(args.mixed_rollout_cohort_schedule)
            if args.mixed_rollout_cohort_schedule
            else schedule
        )
        choices = sorted({int(k) for k in raw if 1 <= int(k) <= int(max_rollout)})
        if not choices:
            raise ValueError(f"--mixed-rollout-cohorts has no choices <= current K={max_rollout}")
        return choices

    def mixed_rollout_weights(choices: list[int]) -> list[float]:
        if not args.mixed_rollout_cohort_weights:
            return [1.0 for _choice in choices]
        raw = [
            float(part)
            for part in args.mixed_rollout_cohort_weights.replace(";", ",").split(",")
            if part.strip()
        ]
        full_choices = (
            tl.parse_rollout_schedule(args.mixed_rollout_cohort_schedule)
            if args.mixed_rollout_cohort_schedule
            else schedule
        )
        if len(raw) != len(full_choices):
            raise ValueError(
                "--mixed-rollout-cohort-weights must have the same number of entries as "
                "--mixed-rollout-cohort-schedule"
            )
        by_choice = {int(choice): max(0.0, float(weight)) for choice, weight in zip(full_choices, raw)}
        weights = [by_choice[int(choice)] for choice in choices]
        if sum(weights) <= 0.0:
            raise ValueError("--mixed-rollout-cohort-weights must contain at least one positive value")
        return weights

    def make_mixed_rollout_lengths(count: int, max_rollout: int) -> torch.Tensor | None:
        choices = mixed_rollout_choices(max_rollout)
        if not choices:
            return None
        weights = mixed_rollout_weights(choices)
        weight_sum = sum(weights)
        expected = [int(count) * weight / weight_sum for weight in weights]
        counts = [int(math.floor(value)) for value in expected]
        leftover = int(count) - sum(counts)
        order = sorted(
            range(len(choices)),
            key=lambda i: (expected[i] - counts[i], weights[i]),
            reverse=True,
        )
        for i in order[:leftover]:
            counts[i] += 1
        lengths: list[int] = []
        for choice, choice_count in zip(choices, counts):
            lengths.extend([choice] * choice_count)
        agent_rng.shuffle(lengths)
        return torch.tensor(lengths, dtype=torch.long)

    def agent_batch(
        dataset: tl.MotionIndexDataset,
        max_rollout: int,
        cohort_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        clip_ids: list[int] = []
        starts: list[int] = []
        init_clip_ids: list[int] = []
        init_starts: list[int] = []
        if store is not None:
            lengths_dev = (
                cohort_lengths.to(device).long().clamp(1, int(max_rollout))
                if cohort_lengths is not None
                else torch.full((cfg.batch_size,), int(max_rollout), dtype=torch.long, device=device)
            )
            synthetic_count = int(round(cfg.batch_size * cfg.synthetic_agent_fraction))
            synthetic_count = min(cfg.batch_size, max(0, synthetic_count))
            real_count = cfg.batch_size - synthetic_count
            synthetic_rows = torch.zeros((cfg.batch_size,), dtype=torch.bool, device=device)
            if synthetic_count > 0:
                synthetic_rows[torch.randperm(cfg.batch_size, device=device)[:synthetic_count]] = True
            all_clip_ids = torch.empty((cfg.batch_size,), dtype=torch.long, device=device)
            all_starts = torch.empty((cfg.batch_size,), dtype=torch.long, device=device)
            for want_synthetic in (False, True):
                group_rows = (synthetic_rows == want_synthetic).nonzero(as_tuple=False).flatten()
                if group_rows.numel() == 0:
                    continue
                group_lengths = lengths_dev.index_select(0, group_rows)
                for requested in torch.unique(group_lengths).tolist():
                    requested_int = max(1, int(requested))
                    sub_rows = group_rows[group_lengths == requested_int]
                    new_clip_ids, new_starts = store.sample_starts(
                        int(sub_rows.numel()),
                        requested_int,
                        synthetic=want_synthetic,
                        require_full=cohort_lengths is not None,
                    )
                    all_clip_ids[sub_rows] = new_clip_ids
                    all_starts[sub_rows] = new_starts
            order = torch.randperm(cfg.batch_size, device=device)
            all_clip_ids = all_clip_ids.index_select(0, order).detach().cpu()
            all_starts = all_starts.index_select(0, order).detach().cpu()
            synthetic_rows = synthetic_rows.index_select(0, order).detach().cpu()
            lengths_cpu = lengths_dev.index_select(0, order).detach().cpu()
            if cfg.init_pose_sampling == "random_dataset" or bool(synthetic_rows.any()):
                init_clip_ids_auto = all_clip_ids.clone()
                init_starts_auto = all_starts.clone()
                random_init_pairs = [random_initial_pose_start() for _ in range(int(synthetic_rows.sum().item()))]
                if random_init_pairs:
                    init_clip_ids_auto[synthetic_rows] = torch.tensor(
                        [ci for ci, _start in random_init_pairs],
                        dtype=torch.long,
                    )
                    init_starts_auto[synthetic_rows] = torch.tensor(
                        [start for _ci, start in random_init_pairs],
                        dtype=torch.long,
                    )
                if cfg.init_pose_sampling == "random_dataset":
                    random_real_pairs = [random_initial_pose_start() for _ in range(int((~synthetic_rows).sum().item()))]
                    if random_real_pairs:
                        init_clip_ids_auto[~synthetic_rows] = torch.tensor(
                            [ci for ci, _start in random_real_pairs],
                            dtype=torch.long,
                        )
                        init_starts_auto[~synthetic_rows] = torch.tensor(
                            [start for _ci, start in random_real_pairs],
                            dtype=torch.long,
                        )
                init_clip_tensor = init_clip_ids_auto
                init_start_tensor = init_starts_auto
                if cohort_lengths is not None:
                    return all_clip_ids, all_starts, init_clip_tensor, init_start_tensor, lengths_cpu
                return all_clip_ids, all_starts, init_clip_tensor, init_start_tensor
            if cohort_lengths is not None:
                return all_clip_ids, all_starts, lengths_cpu
            return all_clip_ids, all_starts

        for _ in range(cfg.batch_size):
            ci, start = random_agent_start(max_rollout)
            clip_ids.append(ci)
            starts.append(start)
            if cfg.init_pose_sampling == "random_dataset":
                init_ci, init_start = random_initial_pose_start()
                init_clip_ids.append(init_ci)
                init_starts.append(init_start)
        if cfg.init_pose_sampling == "random_dataset":
            result = (
                torch.tensor(clip_ids, dtype=torch.long),
                torch.tensor(starts, dtype=torch.long),
                torch.tensor(init_clip_ids, dtype=torch.long),
                torch.tensor(init_starts, dtype=torch.long),
            )
            if cohort_lengths is not None:
                return (*result, cohort_lengths.cpu())
            return result
        result = (torch.tensor(clip_ids, dtype=torch.long), torch.tensor(starts, dtype=torch.long))
        if cohort_lengths is not None:
            return (*result, cohort_lengths.cpu())
        return result

    def sample_effective_rollout_k() -> int:
        if not args.final_stage_random_rollout or rollout_idx != len(schedule) - 1:
            return int(rollout_k)
        return int(schedule[agent_rng.randrange(len(schedule))])

    refresh_stage_sampling(rollout_k)
    dataset, loader = make_loader(rollout_k)
    print(
        f"pure_ae run={cfg.run_name} priors={len(priors)} primary_prior={args.prior_checkpoint} K={rollout_k} "
        f"samples={len(dataset)} loop=agents data=resident",
        flush=True,
    )
    if cfg.agent_fixed_start_frame > 0:
        print(f"agent_fixed_start_frame={cfg.agent_fixed_start_frame}", flush=True)
    if cfg.synthetic_agent_fraction > 0.0:
        synthetic_per_batch = int(round(cfg.batch_size * cfg.synthetic_agent_fraction))
        real_per_batch = cfg.batch_size - synthetic_per_batch
        print(
            f"synthetic agent sampler: real_per_batch={real_per_batch} "
            f"synthetic_per_batch={synthetic_per_batch} "
            f"fraction={synthetic_per_batch / max(1, cfg.batch_size):.6f} "
            f"synthetic_folder={args.synthetic_folder_path}",
            flush=True,
        )
    best = math.inf
    stalls = 0
    stage_start_epoch = 1
    stage_batches = 0
    start_time = time.perf_counter()
    timed_interval_seconds = 60.0 * max(0.0, float(cfg.timed_checkpoint_interval_minutes))
    next_timed_checkpoint_at = start_time + timed_interval_seconds if timed_interval_seconds > 0.0 else math.inf
    last_diagnostics = {
        "joint_rmse": 0.0,
        "ee_rmse": 0.0,
        "output_mse": 0.0,
    }
    for epoch in range(1, cfg.max_epochs + 1):
        parts = []
        model.train()
        compute_diagnostics = (
            args.best_metric != "ae_score"
            or args.diagnostic_metrics_every_epochs == 1
            or (
                args.diagnostic_metrics_every_epochs > 1
                and (epoch == 1 or epoch % args.diagnostic_metrics_every_epochs == 0)
            )
        )
        train_iter = []
        rollout_iter = []
        rollout_report_iter = []
        for _ in range(max(1, args.agent_batches_per_epoch) * cfg.gradient_accumulation_batches):
            effective_k = int(rollout_k) if args.mixed_rollout_cohorts else sample_effective_rollout_k()
            cohort_lengths = make_mixed_rollout_lengths(cfg.batch_size, effective_k)
            train_iter.append(agent_batch(dataset, effective_k, cohort_lengths))
            rollout_iter.append(effective_k)
            if cohort_lengths is not None:
                rollout_report_iter.append(float(cohort_lengths.float().mean().item()))
            else:
                rollout_report_iter.append(float(effective_k))
        picked_clip_ids = sorted(
            {
                int(ci)
                for batch in train_iter
                for ci in batch[0].detach().cpu().tolist()
            }
        )
        stage_batches += len(train_iter)
        for accum_start in range(0, len(train_iter), cfg.gradient_accumulation_batches):
            accum_batches = train_iter[accum_start : accum_start + cfg.gradient_accumulation_batches]
            accum_rollout = rollout_iter[accum_start : accum_start + cfg.gradient_accumulation_batches]
            with profiler.section("train/zero_grad"):
                opt.zero_grad(set_to_none=True)
            for batch, effective_k in zip(accum_batches, accum_rollout):
                with profiler.section("train/forward_loss"):
                    loss, scalars = run_batch_ae(
                        model,
                        priors,
                        clips,
                        batch,
                        cfg,
                        effective_k,
                        device,
                        compute_diagnostics=compute_diagnostics,
                        compatibility_score_weight=args.compatibility_score_weight,
                        reset_sampler=random_agent_reset_batch if cfg.reset_exhausted_agents else None,
                        store=store,
                    )
                with profiler.section("train/backward"):
                    (loss / max(1, len(accum_batches))).backward()
                parts.append(scalars)
            with profiler.section("train/clip_grad"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            with profiler.section("train/optimizer_step"):
                opt.step()
        adaptive_stats = grade_adaptive_clips(int(rollout_k), picked_clip_ids)
        train_total = float(np.mean([p["total"] for p in parts]))
        train_score = float(np.mean([p["ae_score"] for p in parts]))
        motion_rms = float(np.mean([p["canon_step_rms"] for p in parts]))
        slide_excess_loss = float(np.mean([p["slide_excess"] for p in parts]))
        yaw_excess_loss = float(np.mean([p.get("yaw_excess", 0.0) for p in parts]))
        motion_floor_loss = float(np.mean([p.get("motion_floor", 0.0) for p in parts]))
        ae_raw_by_prior = {
            key.removeprefix("ae_raw/"): float(np.mean([p.get(key, 0.0) for p in parts]))
            for key in sorted({key for p in parts for key in p.keys() if key.startswith("ae_raw/")})
        }
        ae_weighted_by_prior = {
            key.removeprefix("ae_weighted/"): float(np.mean([p.get(key, 0.0) for p in parts]))
            for key in sorted({key for p in parts for key in p.keys() if key.startswith("ae_weighted/")})
        }
        active_fraction = float(np.mean([p["active_fraction"] for p in parts]))
        effective_rollout_mean = float(np.mean(rollout_report_iter)) if rollout_report_iter else float(rollout_k)
        effective_rollout_max = float(np.max(rollout_iter)) if rollout_iter else float(rollout_k)
        if compute_diagnostics:
            last_diagnostics["joint_rmse"] = float(np.mean([p["joint_rmse"] for p in parts]))
            last_diagnostics["ee_rmse"] = float(np.mean([p["ee_rmse"] for p in parts]))
            last_diagnostics["output_mse"] = float(np.mean([p["output_mse"] for p in parts]))
        joint_rmse = last_diagnostics["joint_rmse"]
        ee_rmse = last_diagnostics["ee_rmse"]
        output_mse = last_diagnostics["output_mse"]
        selection_metric = {
            "ae_score": train_total,
            "joint_rmse": joint_rmse,
            "ee_rmse": ee_rmse,
            "output_mse": output_mse,
        }[args.best_metric]
        weighted_ae_score = cfg.ae_loss_weight * train_score
        weighted_slide_excess = cfg.slide_excess_loss_weight * slide_excess_loss
        weighted_yaw_excess = cfg.yaw_excess_loss_weight * yaw_excess_loss
        weighted_motion_floor = cfg.motion_floor_loss_weight * motion_floor_loss
        writer.add_scalar("loss/train_total", train_total, epoch)
        writer.add_scalar("loss/ae_score", weighted_ae_score, epoch)
        writer.add_scalar("loss/weighted_ae_score", weighted_ae_score, epoch)
        writer.add_scalar("monitor/raw_ae_score", train_score, epoch)
        for label, value in ae_weighted_by_prior.items():
            writer.add_scalar(f"loss/ae_weighted/{label}", cfg.ae_loss_weight * value, epoch)
        for label, value in ae_raw_by_prior.items():
            writer.add_scalar(f"monitor/ae_raw/{label}", value, epoch)
        writer.add_scalar("monitor/raw_slide_excess", slide_excess_loss, epoch)
        writer.add_scalar(
            "loss/slide_excess",
            weighted_slide_excess,
            epoch,
        )
        writer.add_scalar(
            "loss/weighted_slide_excess",
            weighted_slide_excess,
            epoch,
        )
        if cfg.yaw_excess_loss_weight > 0.0:
            writer.add_scalar("loss/yaw_excess", weighted_yaw_excess, epoch)
            writer.add_scalar("loss/weighted_yaw_excess", weighted_yaw_excess, epoch)
            writer.add_scalar("monitor/raw_yaw_excess", yaw_excess_loss, epoch)
            writer.add_scalar("monitor/yaw_excess_scale_radps", cfg.yaw_excess_scale_radps, epoch)
        if cfg.motion_floor_loss_weight > 0.0:
            writer.add_scalar("loss/motion_floor", weighted_motion_floor, epoch)
            writer.add_scalar("loss/weighted_motion_floor", weighted_motion_floor, epoch)
            writer.add_scalar("monitor/raw_motion_floor", motion_floor_loss, epoch)
            writer.add_scalar("monitor/motion_floor_margin", cfg.motion_floor_margin, epoch)
        writer.add_scalar("curriculum/active_fraction", active_fraction, epoch)
        writer.add_scalar("optim/gradient_accumulation_batches", cfg.gradient_accumulation_batches, epoch)
        if cfg.excess_envelope_enabled and store is not None and store.slide_bound_mps is not None:
            writer.add_scalar(
                "monitor/slide_excess_bound_mean_mps",
                float(store.slide_bound_mps.mean().detach().cpu()),
                epoch,
            )
        writer.add_scalar("motion/canon_step_rms", motion_rms, epoch)
        if compute_diagnostics:
            writer.add_scalar("accuracy/joint_rmse_m", joint_rmse, epoch)
            writer.add_scalar("accuracy/end_effector_rmse_m", ee_rmse, epoch)
            writer.add_scalar("accuracy/output_mse", output_mse, epoch)
        writer.add_scalar(f"selection/{args.best_metric}", selection_metric, epoch)
        writer.add_scalar("curriculum/rollout_k", rollout_k, epoch)
        writer.add_scalar("curriculum/effective_rollout_k_mean", effective_rollout_mean, epoch)
        writer.add_scalar("curriculum/effective_rollout_k_max", effective_rollout_max, epoch)
        eligible_clip_count = max(1, sum(clip_has_any_start(clip, cfg) for clip in clips))
        real_eligible_clip_count = max(1, sum(clip_has_any_start(clips[ci], cfg) for ci in real_clip_indices))
        synthetic_eligible_clip_count = sum(clip_has_any_start(clips[ci], cfg) for ci in synthetic_clip_indices)
        min_stage_batches = (
            math.ceil(float(args.curriculum_min_eligible_clip_visits) * eligible_clip_count)
            if args.curriculum_min_eligible_clip_visits > 0.0
            else 0
        )
        writer.add_scalar("curriculum/stage_batches", stage_batches, epoch)
        writer.add_scalar("curriculum/eligible_clips", eligible_clip_count, epoch)
        writer.add_scalar("curriculum/real_eligible_clips", real_eligible_clip_count, epoch)
        writer.add_scalar("curriculum/synthetic_eligible_clips", synthetic_eligible_clip_count, epoch)
        writer.add_scalar("curriculum/synthetic_agent_fraction", cfg.synthetic_agent_fraction, epoch)
        if adaptive_stats:
            writer.add_scalar("adaptive/graded_clips", adaptive_stats["graded"], epoch)
            writer.add_scalar("adaptive/seen_fraction", adaptive_stats["seen_fraction"], epoch)
            writer.add_scalar("adaptive/active", adaptive_stats["active"], epoch)
            writer.add_scalar("adaptive/score_mean", adaptive_stats["score_mean"], epoch)
            writer.add_scalar("adaptive/score_min", adaptive_stats["score_min"], epoch)
            writer.add_scalar("adaptive/score_max", adaptive_stats["score_max"], epoch)
            write_adaptive_clip_leaderboard(epoch)
        if min_stage_batches > 0:
            writer.add_scalar("curriculum/min_stage_batches", min_stage_batches, epoch)
        improved = selection_metric < best - cfg.curriculum_min_delta
        stalls = 0 if improved else stalls + 1
        if selection_metric < best:
            best = selection_metric
            tl.save_checkpoint(ckpt_dir / "checkpoint_best.pt", model, opt, epoch, best, rollout_k, cfg, metadata)
            tl.save_checkpoint(ckpt_dir / f"checkpoint_best_k{rollout_k:02d}.pt", model, opt, epoch, best, rollout_k, cfg, metadata)
        save_live_period = args.save_live_every_epochs
        if save_live_period > 0 and epoch % save_live_period == 0:
            last_path = ckpt_dir / "checkpoint_last.pt"
            tl.save_checkpoint(last_path, model, opt, epoch, best, rollout_k, cfg, metadata)
            refresh_live_viewer(args, last_path)
        now_perf = time.perf_counter()
        if now_perf >= next_timed_checkpoint_at:
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            tl.save_checkpoint(
                ckpt_dir / f"checkpoint_time_{stamp}_epoch_{epoch:06d}.pt",
                model,
                opt,
                epoch,
                best,
                rollout_k,
                cfg,
                metadata,
            )
            while next_timed_checkpoint_at <= now_perf:
                next_timed_checkpoint_at += timed_interval_seconds
        stage_epochs = epoch - stage_start_epoch + 1
        min_stage_reached = stage_epochs >= cfg.curriculum_min_epochs and stage_batches >= min_stage_batches
        threshold_reached = (
            args.curriculum_threshold is not None
            and min_stage_reached
            and train_total <= float(args.curriculum_threshold)
        )
        can_advance = threshold_reached or (
            cfg.curriculum_max_epochs_per_stage > 0
            and stage_epochs >= cfg.curriculum_max_epochs_per_stage
            and stage_batches >= min_stage_batches
        ) or (
            cfg.curriculum_stall_patience_epochs > 0
            and min_stage_reached
            and stalls >= cfg.curriculum_stall_patience_epochs
        )
        print(
            f"epoch={epoch:04d} K={rollout_k:02d} ae={train_total:.6g} best_{args.best_metric}={best:.6g} "
            f"ae_score={train_score:.6g} slide_excess={slide_excess_loss:.6g} yaw_excess={yaw_excess_loss:.6g} "
            f"motionfloor={motion_floor_loss:.6g} joint_rmse={joint_rmse:.6g} ee_rmse={ee_rmse:.6g} motion_rms={motion_rms:.6g} "
            f"active={active_fraction:.3f} stalls={stalls} stage_batches={stage_batches}/{min_stage_batches} "
            f"eligible_clips={eligible_clip_count} real_clips={real_eligible_clip_count} "
            f"synthetic_clips={synthetic_eligible_clip_count} synthetic_frac={cfg.synthetic_agent_fraction:.3f} "
            f"adaptive_seen={adaptive_stats.get('seen_fraction', 0.0) if adaptive_stats else 0.0:.3f} "
            f"adaptive_active={adaptive_stats.get('active', 0.0) if adaptive_stats else 0.0:.0f} "
            f"effective_k={effective_rollout_mean:.1f} "
            f"elapsed_s={time.perf_counter() - start_time:.1f}",
            flush=True,
        )
        if can_advance and rollout_idx < len(schedule) - 1:
            rollout_idx += 1
            rollout_k = schedule[rollout_idx]
            refresh_stage_sampling(rollout_k)
            dataset, loader = make_loader(rollout_k)
            reset_agent_coverage_order(dataset)
            best = math.inf
            stalls = 0
            stage_start_epoch = epoch + 1
            stage_batches = 0
            print(f"advanced rollout_k={rollout_k} samples={len(dataset)}", flush=True)
        elif can_advance and args.stop_on_final_stall:
            print(
                f"stopped on final stall epoch={epoch} K={rollout_k} "
                f"best_{args.best_metric}={best:.6g}",
                flush=True,
            )
            break
        if args.max_train_seconds > 0 and time.perf_counter() - start_time >= args.max_train_seconds:
            print(f"max_train_seconds reached {args.max_train_seconds}", flush=True)
            break
    last_path = ckpt_dir / "checkpoint_last.pt"
    tl.save_checkpoint(last_path, model, opt, epoch, best, rollout_k, cfg, metadata)
    refresh_live_viewer(args, last_path)
    writer.close()
    profiler.write_csv(run_dir / "timing_profile.csv", time.perf_counter() - process_start_time)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-path", default="data/npz_final")
    parser.add_argument(
        "--periodic-folder-path",
        default=None,
        help="Semicolon-separated motion folders that should use cyclic root/pose indexing.",
    )
    parser.add_argument(
        "--nonperiodic-folder-path",
        default=None,
        help="Semicolon-separated motion folders that use non-cyclic indexing; exhausted rows can reset to fresh starts.",
    )
    parser.add_argument(
        "--synthetic-folder-path",
        default=None,
        help=(
            "Semicolon-separated synthetic NPZ folders used only as controller rollout roots. "
            "These clips are excluded from AE prior training and random init pose sampling."
        ),
    )
    parser.add_argument("--prior-checkpoint", required=True)
    parser.add_argument(
        "--prior-weight",
        type=float,
        default=1.0,
        help="Weight for --prior-checkpoint inside the AE prior loss.",
    )
    parser.add_argument(
        "--extra-prior-checkpoint",
        action="append",
        default=[],
        help="Additional frozen AE prior checkpoint.",
    )
    parser.add_argument(
        "--extra-prior-weight",
        action="append",
        type=float,
        default=[],
        help="Weight for each --extra-prior-checkpoint, in the same order. Defaults to 1.0 when omitted.",
    )
    parser.add_argument("--run-name", default="locomotion_pure_ae")
    parser.add_argument("--date-prefix-run-name", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cyclic-animation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--agent-batches-per-epoch", type=int, default=1)
    parser.add_argument(
        "--agent-min-cohort-steps",
        type=int,
        default=8,
        help="For non-periodic one-clip agent cohorts, avoid starts that would force a reset before this many frames when possible.",
    )
    parser.add_argument(
        "--gradient-accumulation-batches",
        type=int,
        default=1,
        help="Average this many one-clip cohorts before one optimizer step.",
    )
    parser.add_argument(
        "--periodic-sampling-weight",
        type=float,
        default=1.0,
        help="Total sampling weight assigned to cyclic/periodic clips as a group.",
    )
    parser.add_argument(
        "--nonperiodic-sampling-weight",
        type=float,
        default=1.0,
        help="Total sampling weight assigned to non-cyclic transition clips as a group.",
    )
    parser.add_argument(
        "--synthetic-agent-fraction",
        type=float,
        default=0.0,
        help="Fixed fraction of each row-mixed random-agent batch sampled from --synthetic-folder-path.",
    )
    parser.add_argument(
        "--init-pose-sampling",
        choices=("same_clip", "random_dataset"),
        default="same_clip",
        help="Initial body pose source. random_dataset keeps the sampled root clip but initializes the pose from any clip.",
    )
    parser.add_argument(
        "--agent-fixed-start-frame",
        type=int,
        default=0,
        help=(
            "If >0, force every sampled agent start/reset to this frame index, "
            "clamped to each clip's valid range. Use 1 for the first trainable frame."
        ),
    )
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--rollout-schedule", default="1")
    parser.add_argument(
        "--initial-rollout-k",
        type=int,
        default=None,
        help="Start a resumed curriculum at this scheduled K while keeping the full schedule for final-stage sampling.",
    )
    parser.add_argument("--curriculum-max-epochs-per-stage", type=int, default=120)
    parser.add_argument("--curriculum-stall-patience-epochs", type=int, default=60)
    parser.add_argument("--curriculum-min-epochs", type=int, default=30)
    parser.add_argument("--curriculum-min-delta", type=float, default=1e-6)
    parser.add_argument(
        "--curriculum-min-eligible-clip-visits",
        type=float,
        default=0.0,
        help="Minimum one-clip agent batches per eligible clip before rollout K can advance.",
    )
    parser.add_argument(
        "--curriculum-threshold",
        type=float,
        default=None,
        help="Optional train-total threshold for advancing rollout K after curriculum-min-epochs.",
    )
    parser.add_argument(
        "--final-stage-random-rollout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="At the last scheduled K, sample each agent microbatch rollout length from the rollout schedule rungs.",
    )
    parser.add_argument(
        "--mixed-rollout-cohorts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Inside each row-mixed batch, split rows across rollout-length cohorts and reset each row when "
            "its cohort episode ends. The outer GPU rollout still runs at the current K."
        ),
    )
    parser.add_argument(
        "--mixed-rollout-cohort-schedule",
        default="",
        help="Comma-separated cohort lengths, for example 2,4,8,16,32. Empty means use --rollout-schedule entries <= current K.",
    )
    parser.add_argument(
        "--mixed-rollout-cohort-weights",
        default="",
        help=(
            "Comma-separated weights/percentages for --mixed-rollout-cohort-schedule. "
            "They are normalized, so 5,10,15 and 1,2,3 are equivalent."
        ),
    )
    parser.add_argument(
        "--adaptive-clip-sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After every real clip has been graded once, sample real clips in proportion to their "
            "latest rollout loss estimate so hard motions are replayed more often."
        ),
    )
    parser.add_argument(
        "--adaptive-clip-score-k",
        type=int,
        default=0,
        help="Rollout K used for clip grading. 0 means the current scheduled K.",
    )
    parser.add_argument(
        "--adaptive-clip-score-ema",
        type=float,
        default=0.35,
        help="EMA update rate for clip difficulty scores after the initial grade.",
    )
    parser.add_argument(
        "--adaptive-clip-score-floor",
        type=float,
        default=1e-6,
        help="Minimum positive sampling weight for already-graded clips.",
    )
    parser.add_argument(
        "--adaptive-clip-score-batch-size",
        type=int,
        default=64,
        help="No-grad microbatch size for adaptive clip grading.",
    )
    parser.add_argument(
        "--adaptive-clip-leaderboard-top-n",
        type=int,
        default=25,
        help="Number of hardest clips to keep in the adaptive leaderboard JSON/history and TensorBoard text.",
    )
    parser.add_argument(
        "--adaptive-clip-leaderboard-every-epochs",
        type=int,
        default=1,
        help="Write the adaptive clip leaderboard every N epochs.",
    )
    parser.add_argument("--stop-on-final-stall", action="store_true")
    parser.add_argument("--max-train-seconds", type=float, default=0.0)
    parser.add_argument("--timed-checkpoint-interval-minutes", type=float, default=30.0)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Try torch.compile after a forward/backward probe.")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile.")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--best-metric", choices=("ae_score", "joint_rmse", "ee_rmse", "output_mse"), default="ae_score")
    parser.add_argument(
        "--diagnostic-metrics-every-epochs",
        type=int,
        default=10,
        help="Compute visual/GT diagnostic RMSE every N epochs for AE-prior runs. Set 1 for every epoch, 0 to disable unless best-metric needs it.",
    )
    parser.add_argument("--ae-score-loss", choices=("mse", "huber"), default="mse")
    parser.add_argument("--ae-huber-delta", type=float, default=1.0)
    parser.add_argument(
        "--ae-row-top-fraction",
        type=float,
        default=0.0,
        help="Optional fraction of worst AE-scored rows to add to the mean AE loss. 0 keeps the old pure mean.",
    )
    parser.add_argument(
        "--ae-row-top-weight",
        type=float,
        default=0.0,
        help="Weight for the worst-row AE term used with --ae-row-top-fraction.",
    )
    parser.add_argument("--save-live-every-epochs", type=int, default=20)
    parser.add_argument("--live-viewer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--live-npz-path", default="data/npz_final/testcasc.npz")
    parser.add_argument("--live-output-path", default="training/runs/model_comparisons/model_comparison.html")
    parser.add_argument("--visual-reporter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--visual-report-save-every-epochs", type=int, default=20)
    parser.add_argument("--visual-report-interval-seconds", type=float, default=60.0)
    parser.add_argument("--visual-report-device", default="cpu")
    parser.add_argument("--visual-report-max-frames", type=int, default=180)
    parser.add_argument("--profile-timing", action="store_true", help="Write timing_profile.csv in the run directory.")
    parser.add_argument(
        "--profile-sync-cuda",
        action="store_true",
        help="Synchronize CUDA around timed sections for stricter timing. Slower, but more precise.",
    )
    parser.add_argument("--predict-residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--zero-init-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--contact-physics-losses",
        dest="contact_physics_losses",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-contact-physics-losses",
        dest="contact_physics_losses",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--enable-early-termination", action="store_true")
    parser.add_argument("--no-restart-on-termination", action="store_true")
    parser.add_argument(
        "--reset-exhausted-agents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For long AE rollouts, reset non-periodic rows before target/future-root frames would pass clip end.",
    )
    parser.add_argument("--freefall-body-height-offset-m", type=float, default=0.0)
    parser.add_argument("--freefall-initial-offset-history", type=int, choices=(1, 2), default=1)
    parser.add_argument("--freefall-initial-contacts-off", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--alpha7-contact-label", type=float, default=5.0)
    parser.add_argument("--alpha8-foot-penetration", type=float, default=1700.0)
    parser.add_argument("--alpha9-foot-sliding", type=float, default=1.0)
    parser.add_argument("--alpha10-freefall", type=float, default=1700.0)
    parser.add_argument("--alpha11-contact-height", type=float, default=245.0)
    parser.add_argument("--alpha12-termination", type=float, default=0.07)
    parser.add_argument("--ae-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--slide-excess-loss-weight",
        type=float,
        default=0.0,
        help="Excess-envelope generated-geometry slide-excess loss. Requires row-mixed rollout.",
    )
    parser.add_argument(
        "--turn-slide-bound-divisor",
        type=float,
        default=1.0,
        help=(
            "Divide the slide-excess tolerance by this value for stand-turn clips. "
            "Slide-excess uses the foot with the lowest custom collider point."
        ),
    )
    parser.add_argument(
        "--excess-envelope",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use cached root-conditioned excess envelopes for slide-excess and yaw-excess losses. "
            "Synthetic clips get cached bounds but do not define them."
        ),
    )
    parser.add_argument(
        "--excess-envelope-knn",
        type=int,
        default=32,
        help="Number of nearest real root-motion situations used to build the zero-loss excess envelope.",
    )
    parser.add_argument(
        "--excess-envelope-margin",
        type=float,
        default=1.05,
        help="Safety multiplier applied to root-conditioned ground-truth excess bounds.",
    )
    parser.add_argument(
        "--excess-envelope-cache-dir",
        default="training/runs/cache/excess_envelopes",
        help="Directory for cached per-frame excess envelopes.",
    )
    parser.add_argument(
        "--yaw-excess-loss-weight",
        type=float,
        default=0.0,
        help="Extra loss on vertical-axis foot/toe angular speed above the cached root-conditioned excess envelope.",
    )
    parser.add_argument(
        "--yaw-excess-scale-radps",
        type=float,
        default=0.0,
        help=(
            "Scale for yaw-excess. If <=0, it is set so the current "
            "checkpoint's forward-walk autoreg max excess has loss 1."
        ),
    )
    parser.add_argument(
        "--yaw-excess-scale-checkpoint",
        default="",
        help="Metadata-only note for the checkpoint used to choose yaw-excess scaling; model weights come from --resume-checkpoint.",
    )
    parser.add_argument(
        "--motion-floor-loss-weight",
        type=float,
        default=0.0,
        help="One-sided loss that prevents generated local body motion from dropping below the source transition's motion RMS.",
    )
    parser.add_argument(
        "--motion-floor-margin",
        type=float,
        default=0.9,
        help="Fraction of source transition local body motion RMS required before the motion-floor loss becomes zero.",
    )
    parser.add_argument(
        "--compatibility-score-weight",
        type=float,
        default=0.0,
        help="Extra model-training penalty from a compatible transition prior head, if the AE checkpoint has one.",
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
