from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

try:
    from .bootstrap import PROJECT_ROOT, ensure_paths
    from .naming import checkpoint_path, ik_run_id
    from . import checkpoint_runtime as ckpt_runtime
    from . import excess_envelope as env
    from . import ik_core as tl
    from .rl_loss import RL_TERM_NAMES, RLLossConfig, compute_rl_loss, foot_world_positions
    from .train_simple_autoencoder import SimpleAEConfig, SimpleAutoencoder
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    from naming import checkpoint_path, ik_run_id
    import checkpoint_runtime as ckpt_runtime
    import excess_envelope as env
    import ik_core as tl
    from rl_loss import RL_TERM_NAMES, RLLossConfig, compute_rl_loss, foot_world_positions
    from train_simple_autoencoder import SimpleAEConfig, SimpleAutoencoder

ensure_paths()


DEFAULT_WALK_F = PROJECT_ROOT / "ue5" / "animations_omni_only_full" / "npz_final" / "M_Neutral_Walk_Loop_F.npz"
RUNS_DIR = PROJECT_ROOT / "training" / "runs"
DEFAULT_AE_GLOB = "*_ik_simple_ae_*"
SIMPLE_CONTROLLER_AE_KIND = "simple_controller_io_autoencoder"
LEGACY_SIMPLE_AE_KIND = "simple_" + "ag" + "ent_io_autoencoder"
ENVELOPE_TERM_NAMES = ("linear_slide_weighted", "angular_slide_weighted")
IDENTITY_TERM_NAME = "identity_output"
IDENTITY_MAXABS_TERM_NAME = "identity_output_maxabs"
IDENTITY_WORLD_POS_TERM_NAME = "identity_world_pos"
IDENTITY_PELVIS_POS_TERM_NAME = "identity_pelvis_pos"

BATCH_SIZE = 4096
SMALL_CUDA_K64_BATCH_SIZE = 3328
ROLLOUT_SCHEDULE = (1, 2, 8, 16, 32, 64)
ROLLOUT_STAGE_STEPS = (3000, 1000, 1500, 1500, 2500, 2500)
ROLLOUT_K = 64
MIXED_ROLLOUT_AT_MAX = True
LEARNING_RATE = 1e-4
STAGE_LEARNING_RATES = {
    1: 1e-4,
    2: 8e-5,
    8: 5e-5,
    16: 2e-5,
    32: 7.5e-6,
    64: 7.5e-6,
}
LOG_EVERY = 250
VALIDATION_ROWS = 256
RUN_FK_DIAGNOSTIC = False
AE_SCORE_OUTPUT_ONLY = True
NAN_METRIC = float("nan")
POSE_NOISE_POS_SIGMA_M_AT_1 = 0.12
POSE_NOISE_ROT_SIGMA_DEG_AT_1 = 25.0
POSE_NOISE_SCALAR_SIGMA_AT_1 = 1.0
RL_GRAD_CLIP_NORM = 1.0
ZERO_LOSS_STOP_THRESHOLD = 0.0


def refresh_tensorboard_async() -> None:
    script = PROJECT_ROOT / "training" / "ik" / "launch_tensorboard_latest.ps1"
    if not script.exists():
        return
    kwargs = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
    }
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        subprocess.Popen(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            cwd=str(PROJECT_ROOT),
            **kwargs,
        )
    except Exception as exc:
        print(f"tensorboard refresh skipped: {exc}", flush=True)


@dataclass(frozen=True)
class StartPool:
    clip_ids: torch.Tensor
    starts: torch.Tensor

    @property
    def row_count(self) -> int:
        return int(self.starts.numel())


@dataclass(frozen=True)
class ControllerLossResult:
    total: torch.Tensor
    terms: dict[str, torch.Tensor]


def detach_loss_result(result: ControllerLossResult) -> ControllerLossResult:
    return ControllerLossResult(
        total=result.total.detach(),
        terms={key: value.detach() for key, value in result.terms.items()},
    )


def maybe_clip_rl_gradients(model: torch.nn.Module, rl_cfg: RLLossConfig) -> None:
    if rl_cfg.enabled:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(RL_GRAD_CLIP_NORM))


def envelope_loss_enabled(linear_weight: float, angular_weight: float) -> bool:
    return float(linear_weight) != 0.0 or float(angular_weight) != 0.0


def wake_on_zero_loss(run_id: str, step: int, loss: float) -> None:
    print(f"\a\a\aZERO_LOSS_STOP run={run_id} step={step} loss={loss:.6g}", flush=True)
    try:
        import winsound

        for _ in range(3):
            winsound.Beep(880, 350)
            winsound.Beep(1175, 350)
    except Exception:
        pass


class SimpleClipStore:
    def __init__(self, clips: list[tl.MotionClip], cfg: tl.TrainConfig, device: torch.device):
        if not clips:
            raise ValueError("SimpleClipStore needs at least one clip")
        first = clips[0]
        for clip in clips[1:]:
            if clip.body_names != first.body_names or clip.parents_body_list != first.parents_body_list:
                raise ValueError(f"Skeleton mismatch: {clip.path} vs {first.path}")

        self.clips = clips
        self.cfg = cfg
        self.device = device
        self.prototype = first
        self.J = int(first.J)
        self.Jcore = int(first.Jcore)
        self.ik_payload_dim = int(getattr(first, "ik_payload_dim", 0))
        self.pelvis = int(first.pelvis)
        prototype_tensors = first.tensors(device)
        self.local_offsets = prototype_tensors["local_offsets"]
        self.ik_limb_lengths = prototype_tensors["ik_limb_lengths"]
        self.ik_rest_axis = prototype_tensors["ik_rest_axis"]
        self.ik_ee_pole_ref = prototype_tensors["ik_ee_pole_ref"]
        self.ik_pole_alpha = prototype_tensors["ik_pole_alpha"]
        self.ik_base_start_indices = tuple(int(spec["start"]) for spec in first.ik_limb_specs)
        needed_bones: set[int] = set()
        for start in self.ik_base_start_indices:
            bone = int(start)
            while bone >= 0:
                needed_bones.add(bone)
                bone = int(first.parents_body_list[bone])
        self.ik_base_eval_order = tuple(bone for bone in range(first.J) if bone in needed_bones)
        fast_base_names = (
            "spine_01",
            "spine_02",
            "spine_03",
            "spine_04",
            "spine_05",
            "clavicle_l",
            "clavicle_r",
            "upperarm_l",
            "upperarm_r",
            "thigh_l",
            "thigh_r",
        )
        self.ik_fast_base_indices = (
            {name: first.body_names.index(name) for name in fast_base_names}
            if all(name in first.body_names for name in fast_base_names)
            else None
        )

        tensors = [clip.tensors(device) for clip in clips]
        lengths = [int(clip.T) for clip in clips]
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)

        self.frame_offsets = torch.tensor(offsets[:-1], dtype=torch.long, device=device)
        self.lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.periods = torch.tensor([int(clip.cyclic_period) for clip in clips], dtype=torch.long, device=device)
        self.cyclic = torch.tensor([bool(clip.cyclic_animation) for clip in clips], dtype=torch.bool, device=device)

        self.root_pos = torch.cat([t["root_pos"] for t in tensors], dim=0)
        self.root_rot = torch.cat([t["root_rot"] for t in tensors], dim=0)
        self.pelvis_local_pos = torch.cat([t["pelvis_local_pos"] for t in tensors], dim=0)
        self.pelvis_rot6 = torch.cat([t["pelvis_rot6"] for t in tensors], dim=0)
        self.non_pelvis_rot6 = torch.cat([t["non_pelvis_rot6"] for t in tensors], dim=0)
        self.core_non_pelvis_rot6 = torch.cat([t["core_non_pelvis_rot6"] for t in tensors], dim=0)
        self.canonical_pos = torch.cat([t["canonical_pos"] for t in tensors], dim=0)
        self.ik_payload = torch.cat([t["ik_payload"] for t in tensors], dim=0)

        self.root0_pos = torch.stack([t["root_pos"][0] for t in tensors], dim=0)
        self.root0_rot = torch.stack([t["root_rot"][0] for t in tensors], dim=0)
        self.root0_inv = self.root0_rot.transpose(-1, -2)
        self.end_pos = torch.stack([t["root_pos"][clip.cyclic_period] for clip, t in zip(clips, tensors)], dim=0)
        self.end_rot = torch.stack([t["root_rot"][clip.cyclic_period] for clip, t in zip(clips, tensors)], dim=0)
        self.cycle_pos = torch.matmul((self.end_pos - self.root0_pos).unsqueeze(1), self.root0_inv).squeeze(1)
        self.cycle_rot = self.end_rot @ self.root0_inv

        self.target_output = self._build_target_output()
        self.input_root_features = self._build_input_root_features()

    def _build_target_output(self) -> torch.Tensor:
        b = self.pelvis_local_pos.shape[0]
        return torch.cat(
            (
                self.pelvis_local_pos,
                self.pelvis_rot6,
                self.core_non_pelvis_rot6.reshape(b, -1),
                self.ik_payload,
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
                if int(clip.T) <= 1:
                    chunks.append(features)
                    continue
                rows = torch.arange(1, int(clip.T), dtype=torch.long, device=self.device)
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
            if not clip.cyclic_animation:
                flat_idx = flat_idx.clamp(max=int(clip.T) - 1)
            fut_pos, _fut_rot, fut_yaw, _fut_heading = self.root_state(flat_clip_ids, flat_idx)
            fut_pos = fut_pos.reshape(cur_idx.numel(), future_steps, 3)
            fut_yaw = fut_yaw.reshape(cur_idx.numel(), future_steps)
            fut_local = torch.matmul((fut_pos - cur_pos[:, None, :]).unsqueeze(-2), cur_heading[:, None]).squeeze(-2)
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

    def frame_index(self, clip_ids: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        clip_ids = clip_ids.to(self.device).long()
        idx = idx.to(self.device).long()
        periods = self.periods.index_select(0, clip_ids).clamp_min(1)
        cyclic = self.cyclic.index_select(0, clip_ids)
        logical = torch.where(cyclic, torch.remainder(idx, periods), idx)
        return self.frame_offsets.index_select(0, clip_ids) + logical

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
        acc_pos = torch.zeros_like(cycle_pos)
        identity = torch.eye(3, dtype=cycle_rot.dtype, device=self.device).expand_as(cycle_rot)
        acc_rot = identity
        base_pos = cycle_pos
        base_rot = cycle_rot
        for bit in range(16):
            use = torch.remainder(torch.div(cycles, 1 << bit, rounding_mode="floor"), 2).bool()
            use_pos = use.reshape(-1, 1)
            use_rot = use.reshape(-1, 1, 1)
            next_acc_pos = torch.matmul(acc_pos.unsqueeze(1), base_rot).squeeze(1) + base_pos
            next_acc_rot = acc_rot @ base_rot
            acc_pos = torch.where(use_pos, next_acc_pos, acc_pos)
            acc_rot = torch.where(use_rot, next_acc_rot, acc_rot)
            base_pos = torch.matmul(base_pos.unsqueeze(1), base_rot).squeeze(1) + base_pos
            base_rot = base_rot @ base_rot
        rel_pos = torch.matmul(rel_pos.unsqueeze(1), acc_rot).squeeze(1) + acc_pos
        rel_rot = rel_rot @ acc_rot

        pos = torch.matmul(rel_pos.unsqueeze(1), root0_rot).squeeze(1) + root0_pos
        rot = rel_rot @ root0_rot
        yaw = tl.heading_yaw_from_root(rot)
        heading = tl.yaw_to_row_matrix(yaw)
        return pos, rot, yaw, heading

    def get_pose(self, clip_ids: torch.Tensor, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        frame = self.frame_index(clip_ids, idx)
        return {
            "pelvis_pos": self.pelvis_local_pos.index_select(0, frame),
            "pelvis_rot6": self.pelvis_rot6.index_select(0, frame),
            "nonpelvis_rot6": self.non_pelvis_rot6.index_select(0, frame),
            "canon_pos": self.canonical_pos.index_select(0, frame),
            "core_nonpelvis_rot6": self.core_non_pelvis_rot6.index_select(0, frame),
            "ik_payload": self.ik_payload.index_select(0, frame),
        }

    def get_target_output(self, clip_ids: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return self.target_output.index_select(0, self.frame_index(clip_ids, idx))

    def get_input_root_features(self, clip_ids: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return self.input_root_features.index_select(0, self.frame_index(clip_ids, idx))


def apply_config_dict(cfg: tl.TrainConfig, values: dict) -> None:
    valid = {field.name for field in fields(tl.TrainConfig)}
    for key, value in values.items():
        if key not in valid:
            continue
        current = getattr(cfg, key)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(cfg, key, value)


def make_cfg(device: torch.device, ae_ckpt: dict) -> tl.TrainConfig:
    cfg = tl.TrainConfig()
    apply_config_dict(cfg, ae_ckpt.get("locomotion_config", {}))
    cfg.pose_representation = tl.IK_POSE_REPRESENTATION
    cfg.predict_residual = tl.output_prediction_uses_residual()
    cfg.zero_init_output = tl.output_prediction_uses_residual()
    cfg.hidden_dim = 512
    cfg.num_hidden_layers = 2
    cfg.learning_rate = LEARNING_RATE
    cfg.batch_size = BATCH_SIZE
    cfg.live_viewer = False
    cfg.visual_reporter = False
    cfg.update_comparison_on_exit = False
    cfg.use_torch_compile = False
    cfg.device = str(device)
    return cfg


def initialize_controller_output_from_ae_mean(
    model: torch.nn.Module,
    mean: torch.Tensor,
    input_dim: int,
    output_dim: int,
) -> str:
    output_layer = getattr(model, "net", [None])[-1]
    if not isinstance(output_layer, torch.nn.Linear):
        raise TypeError("Expected MLPController final layer to be torch.nn.Linear.")
    target_mean = mean[int(input_dim) : int(input_dim) + int(output_dim)].to(
        device=output_layer.bias.device,
        dtype=output_layer.bias.dtype,
    )
    if int(target_mean.numel()) != int(output_dim):
        raise ValueError(f"AE feature mean output dim {target_mean.numel()} does not match controller output dim {output_dim}.")
    with torch.no_grad():
        output_layer.weight.zero_()
        output_layer.bias.copy_(target_mean)
    return "ae_feature_output_mean_bias_zero_weight"


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


def latest_simple_ae_checkpoint() -> Path:
    candidates: list[Path] = []
    for run_dir in RUNS_DIR.glob(DEFAULT_AE_GLOB):
        ckpt_dir = run_dir / "checkpoints"
        if ckpt_dir.exists():
            candidates.extend(sorted(ckpt_dir.glob("*_best.pt"), key=lambda p: p.stat().st_mtime, reverse=True))
    if not candidates:
        raise FileNotFoundError(f"No simple AE checkpoints found under {RUNS_DIR}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0].resolve()


def load_simple_ae(path: Path, device: torch.device) -> tuple[SimpleAutoencoder, torch.Tensor, torch.Tensor, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("kind") not in {SIMPLE_CONTROLLER_AE_KIND, LEGACY_SIMPLE_AE_KIND}:
        raise ValueError(f"Not a simple controller IO AE checkpoint: {path}")
    ae_cfg = SimpleAEConfig(**ckpt["config"])
    schema = dict(ckpt["schema"])
    if schema.get("output_reference_root") != tl.OUTPUT_REFERENCE_ROOT:
        raise ValueError(
            f"Simple AE checkpoint uses output_reference_root={schema.get('output_reference_root')!r}; "
            f"expected {tl.OUTPUT_REFERENCE_ROOT!r}: {path}"
        )
    checkpoint_prediction = str(schema.get("output_prediction_mode", tl.OUTPUT_PREDICTION_MODE_ABSOLUTE)).lower().strip()
    if checkpoint_prediction != tl.normalized_output_prediction_mode():
        raise ValueError(
            f"Simple AE checkpoint uses output_prediction_mode={checkpoint_prediction!r}; "
            f"expected {tl.normalized_output_prediction_mode()!r}: {path}"
        )
    if schema.get("ik_schema_version") != tl.IK_SCHEMA_VERSION or schema.get("ik_pole_reference") != tl.IK_POLE_REFERENCE:
        raise ValueError(
            "Simple AE checkpoint uses a legacy IK pole schema; retrain the AE with "
            f"ik_schema_version={tl.IK_SCHEMA_VERSION} and ik_pole_reference={tl.IK_POLE_REFERENCE!r}: {path}"
        )
    model = SimpleAutoencoder(int(schema["total_dim"]), ae_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    mean = ckpt["mean"].to(device=device, dtype=torch.float32)
    std = ckpt["std"].to(device=device, dtype=torch.float32).clamp_min(1e-8)
    return model, mean, std, ckpt


def transition_feature_horizon(cfg: tl.TrainConfig) -> int:
    root_lookahead_steps = max(0, int(getattr(cfg, "root_lookahead_steps", 0)))
    return max(int(cfg.future_window), root_lookahead_steps + 1)


def max_start_for_clip(clip: tl.MotionClip, cfg: tl.TrainConfig, rollout_k: int) -> int:
    if clip.cyclic_animation:
        return int(clip.cyclic_period) - 1
    return int(clip.T) - transition_feature_horizon(cfg) - max(1, int(rollout_k))


def max_training_start_for_clip(clip: tl.MotionClip, cfg: tl.TrainConfig) -> int:
    if clip.cyclic_animation:
        return int(clip.cyclic_period) - 1
    return int(clip.T) - transition_feature_horizon(cfg) - 1


def build_start_pool(store: SimpleClipStore, rollout_k: int) -> StartPool:
    clip_chunks: list[torch.Tensor] = []
    start_chunks: list[torch.Tensor] = []
    for clip_id, clip in enumerate(store.clips):
        max_start = max_start_for_clip(clip, store.cfg, rollout_k)
        if max_start < 1:
            continue
        starts = torch.arange(1, max_start + 1, dtype=torch.long, device=store.device)
        clip_chunks.append(torch.full_like(starts, int(clip_id)))
        start_chunks.append(starts)
    if not start_chunks:
        raise ValueError(f"No valid full-window rollout starts found for K={rollout_k}")
    return StartPool(torch.cat(clip_chunks, dim=0), torch.cat(start_chunks, dim=0))


def build_training_start_pool(store: SimpleClipStore, rollout_k: int) -> StartPool:
    clip_chunks: list[torch.Tensor] = []
    start_chunks: list[torch.Tensor] = []
    for clip_id, clip in enumerate(store.clips):
        max_start = max_training_start_for_clip(clip, store.cfg)
        if max_start < 1:
            continue
        starts = torch.arange(1, max_start + 1, dtype=torch.long, device=store.device)
        clip_chunks.append(torch.full_like(starts, int(clip_id)))
        start_chunks.append(starts)
    if not start_chunks:
        raise ValueError(f"No valid rollout starts found for K={rollout_k}")
    return StartPool(torch.cat(clip_chunks, dim=0), torch.cat(start_chunks, dim=0))


def rollout_values_for(max_k: int) -> tuple[int, ...]:
    values: list[int] = []
    k = max(1, int(max_k))
    while k > 1:
        values.append(k)
        k = max(1, k // 2)
    values.append(1)
    return tuple(dict.fromkeys(values))


def mixed_rollout_enabled(rollout_k: int) -> bool:
    return bool(MIXED_ROLLOUT_AT_MAX) and int(rollout_k) >= int(ROLLOUT_K)


def build_start_pools(store: SimpleClipStore, rollout_values: tuple[int, ...]) -> dict[int, StartPool]:
    return {int(k): build_start_pool(store, int(k)) for k in rollout_values}


def build_training_start_pools(store: SimpleClipStore, rollout_values: tuple[int, ...]) -> dict[int, StartPool]:
    return {int(k): build_training_start_pool(store, int(k)) for k in rollout_values}


def sample_from_pool(pool: StartPool, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    count = max(1, int(count))
    if count == pool.row_count:
        order = torch.randperm(pool.row_count, device=pool.starts.device)
        return pool.clip_ids.index_select(0, order), pool.starts.index_select(0, order)
    rows = torch.randint(0, pool.row_count, (count,), device=pool.starts.device)
    return pool.clip_ids.index_select(0, rows), pool.starts.index_select(0, rows)


def sample_effective_rollout_k(batch_size: int, rollout_k: int, device: torch.device) -> torch.Tensor:
    batch_size = max(1, int(batch_size))
    rollout_k = max(1, int(rollout_k))
    if not mixed_rollout_enabled(rollout_k):
        return torch.full((batch_size,), rollout_k, dtype=torch.long, device=device)
    values = rollout_values_for(rollout_k)
    remaining = batch_size
    chunks: list[torch.Tensor] = []
    for value in values[:-1]:
        count = remaining // 2
        if count:
            chunks.append(torch.full((count,), int(value), dtype=torch.long, device=device))
        remaining -= count
    chunks.append(torch.full((remaining,), int(values[-1]), dtype=torch.long, device=device))
    effective_k = torch.cat(chunks, dim=0)
    return effective_k.index_select(0, torch.randperm(batch_size, device=device))


def max_training_start_for_clip_ids(store: SimpleClipStore, clip_ids: torch.Tensor) -> torch.Tensor:
    clip_ids = clip_ids.to(store.device).long()
    cyclic = store.cyclic.index_select(0, clip_ids)
    periods = store.periods.index_select(0, clip_ids).clamp_min(1)
    lengths = store.lengths.index_select(0, clip_ids)
    horizon = int(transition_feature_horizon(store.cfg))
    noncyclic_max = (lengths - horizon - 1).clamp_min(1)
    return torch.where(cyclic, periods - 1, noncyclic_max).clamp_min(1)


def sample_same_clip_training_starts(store: SimpleClipStore, clip_ids: torch.Tensor) -> torch.Tensor:
    max_start = max_training_start_for_clip_ids(store, clip_ids)
    noise = torch.rand(max_start.shape, dtype=torch.float32, device=store.device)
    return (torch.floor(noise * max_start.float()).long() + 1).clamp_min(1)


def sample_same_clip_training_starts_by_step(
    store: SimpleClipStore,
    clip_ids: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    steps = max(1, int(steps))
    max_start = max_training_start_for_clip_ids(store, clip_ids).float().unsqueeze(0)
    noise = torch.rand((steps, int(clip_ids.numel())), dtype=torch.float32, device=store.device)
    return (torch.floor(noise * max_start).long() + 1).clamp_min(1)


def training_reset_rows(
    store: SimpleClipStore,
    clip_ids: torch.Tensor,
    cur_idx: torch.Tensor,
    continuing: torch.Tensor,
) -> torch.Tensor:
    cyclic = store.cyclic.index_select(0, clip_ids)
    max_start = max_training_start_for_clip_ids(store, clip_ids)
    return continuing & (~cyclic) & (cur_idx >= max_start)


def rollout_stat_summary(batch_size: int, rollout_k: int) -> dict[str, float]:
    values = rollout_values_for(rollout_k) if mixed_rollout_enabled(rollout_k) else (max(1, int(rollout_k)),)
    remaining = max(1, int(batch_size))
    counts: list[int] = []
    for _value in values[:-1]:
        count = remaining // 2
        counts.append(count)
        remaining -= count
    counts.append(remaining)
    total = float(sum(counts))
    return {
        "effective_k_mean": sum(float(k) * float(c) for k, c in zip(values, counts)) / total,
        "effective_k_max": float(max(values)),
    }


def sample_rollout_rows(start_pools: dict[int, StartPool], effective_k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    clip_ids = torch.empty_like(effective_k)
    starts = torch.empty_like(effective_k)
    for rollout_k, pool in start_pools.items():
        rows = (effective_k == int(rollout_k)).nonzero(as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        row_clip_ids, row_starts = sample_from_pool(pool, int(rows.numel()))
        clip_ids[rows] = row_clip_ids
        starts[rows] = row_starts
    return clip_ids, starts


def stage_for_step(step: int) -> tuple[int, int, int, int]:
    start = 1
    for stage_idx, (rollout_k, stage_steps) in enumerate(zip(ROLLOUT_SCHEDULE, ROLLOUT_STAGE_STEPS)):
        end = start + int(stage_steps) - 1
        if int(step) <= end:
            return stage_idx, int(rollout_k), start, end
        start = end + 1
    return len(ROLLOUT_SCHEDULE) - 1, int(ROLLOUT_SCHEDULE[-1]), start, start


def stage_learning_rate(rollout_k: int) -> float:
    return float(STAGE_LEARNING_RATES.get(int(rollout_k), LEARNING_RATE))


def default_stage_learning_rates_for_schedule(schedule: tuple[int, ...]) -> dict[int, float]:
    if max(int(k) for k in schedule) <= 32:
        return {
            1: 1e-4,
            2: 1e-4,
            8: 1e-4,
            16: 5e-5,
            32: 2e-5,
        }
    return dict(STAGE_LEARNING_RATES)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def make_adamw(params, lr: float, device: torch.device, capturable: bool = False) -> torch.optim.Optimizer:
    kwargs = {"lr": lr, "weight_decay": 0.0}
    if device.type == "cuda":
        kwargs["fused"] = True
        if capturable:
            kwargs["capturable"] = True
    try:
        return torch.optim.AdamW(params, **kwargs)
    except (RuntimeError, TypeError):
        kwargs.pop("fused", None)
        kwargs.pop("capturable", None)
        return torch.optim.AdamW(params, **kwargs)


def batch_size_for_stage(store: SimpleClipStore, rollout_k: int, row_count: int) -> int:
    cap = int(BATCH_SIZE)
    if store.device.type == "cuda" and int(rollout_k) >= 64:
        total_gb = torch.cuda.get_device_properties(store.device).total_memory / float(1024**3)
        if total_gb <= 10.0:
            cap = min(cap, int(SMALL_CUDA_K64_BATCH_SIZE))
    return max(1, min(int(row_count), cap))


def payload_slice(store: SimpleClipStore) -> slice:
    start = 3 + 6 + store.Jcore * 6
    return slice(start, start + store.ik_payload_dim)


def vector_position_slices(store: SimpleClipStore) -> list[slice]:
    slices = [slice(0, 3)]
    payload_start = payload_slice(store).start
    for spec in tl.IK_PAYLOAD_SLICES:
        pos_slice = spec["pos"]
        assert isinstance(pos_slice, slice)
        slices.append(slice(payload_start + pos_slice.start, payload_start + pos_slice.stop))
    return slices


def vector_rot6_slices(store: SimpleClipStore) -> list[slice]:
    slices = [slice(3, 9)]
    cursor = 9
    for _ in range(store.Jcore):
        slices.append(slice(cursor, cursor + 6))
        cursor += 6
    payload_start = cursor
    for spec in tl.IK_PAYLOAD_SLICES:
        rot_slice = spec["rot6"]
        assert isinstance(rot_slice, slice)
        slices.append(slice(payload_start + rot_slice.start, payload_start + rot_slice.stop))
    return slices


def vector_scalar_slices(store: SimpleClipStore) -> list[slice]:
    slices: list[slice] = []
    payload_start = payload_slice(store).start
    for spec in tl.IK_PAYLOAD_SLICES:
        pole_slice = spec["pole"]
        toe_slice = spec["toe_float"]
        assert isinstance(pole_slice, slice)
        slices.append(slice(payload_start + pole_slice.start, payload_start + pole_slice.stop))
        if toe_slice is not None:
            assert isinstance(toe_slice, slice)
            slices.append(slice(payload_start + toe_slice.start, payload_start + toe_slice.stop))
    return slices


def add_pose_noise_to_vector(store: SimpleClipStore, vec: torch.Tensor, amount: float) -> torch.Tensor:
    amount = float(amount)
    if amount <= 0.0:
        return vec
    out = vec.clone()
    pos_sigma = amount * float(POSE_NOISE_POS_SIGMA_M_AT_1)
    rot_sigma = amount * float(POSE_NOISE_ROT_SIGMA_DEG_AT_1) * math.pi / 180.0
    scalar_sigma = amount * float(POSE_NOISE_SCALAR_SIGMA_AT_1)
    for sl in vector_position_slices(store):
        out[:, sl] = out[:, sl] + torch.randn_like(out[:, sl]) * pos_sigma
    if scalar_sigma > 0.0:
        for sl in vector_scalar_slices(store):
            out[:, sl] = out[:, sl] + torch.randn_like(out[:, sl]) * scalar_sigma
    if rot_sigma > 0.0:
        for sl in vector_rot6_slices(store):
            base_rot = tl.rotation_6d_to_matrix(vec[:, sl])
            axis = torch.randn((vec.shape[0], 3), dtype=vec.dtype, device=vec.device)
            angle = torch.randn((vec.shape[0],), dtype=vec.dtype, device=vec.device) * rot_sigma
            delta = tl.axis_angle_to_row_matrix(axis, angle)
            out[:, sl] = tl.rotmat_to_6d(delta @ base_rot)
    return clean_output_vector(out, store)


def target_state(store: SimpleClipStore, clip_ids: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vec = store.get_target_output(clip_ids, idx)
    return vec, vec[:, :3], vec[:, payload_slice(store)]


def rebase_output_vector_root(
    store: SimpleClipStore,
    vec: torch.Tensor,
    from_root_pos: torch.Tensor,
    from_root_rot: torch.Tensor,
    to_root_pos: torch.Tensor,
    to_root_rot: torch.Tensor,
) -> torch.Tensor:
    rebased = tl.rebase_output_vector_root(
        store.prototype,
        vec,
        from_root_pos,
        from_root_rot,
        to_root_pos,
        to_root_rot,
    )
    return clean_output_vector(rebased, store)


def transition_output_root_state(
    store: SimpleClipStore,
    clip_ids: torch.Tensor,
    cur_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_idx = cur_idx if tl.output_reference_uses_current_root() else cur_idx + 1
    root_pos, root_rot, _yaw, _heading = store.root_state(clip_ids, output_idx)
    return root_pos, root_rot


def current_state_as_transition_output(
    store: SimpleClipStore,
    clip_ids: torch.Tensor,
    cur_idx: torch.Tensor,
    cur_vec: torch.Tensor,
) -> torch.Tensor:
    if tl.output_reference_uses_current_root():
        return cur_vec
    cur_root_pos, cur_root_rot, _cur_yaw, _cur_heading = store.root_state(clip_ids, cur_idx)
    next_root_pos, next_root_rot, _next_yaw, _next_heading = store.root_state(clip_ids, cur_idx + 1)
    return rebase_output_vector_root(store, cur_vec, cur_root_pos, cur_root_rot, next_root_pos, next_root_rot)


def transition_target_output(store: SimpleClipStore, clip_ids: torch.Tensor, cur_idx: torch.Tensor) -> torch.Tensor:
    target_idx = cur_idx + 1
    target = store.get_target_output(clip_ids, target_idx)
    if tl.output_reference_uses_future_root():
        return target
    cur_root_pos, cur_root_rot, _cur_yaw, _cur_heading = store.root_state(clip_ids, cur_idx)
    target_root_pos, target_root_rot, _target_yaw, _target_heading = store.root_state(clip_ids, target_idx)
    return rebase_output_vector_root(store, target, target_root_pos, target_root_rot, cur_root_pos, cur_root_rot)


def advance_transition_output(
    store: SimpleClipStore,
    clip_ids: torch.Tensor,
    cur_idx: torch.Tensor,
    transition_vec: torch.Tensor,
) -> torch.Tensor:
    if tl.output_reference_uses_future_root():
        return clean_output_vector(transition_vec, store)
    next_idx = cur_idx + 1
    cur_root_pos, cur_root_rot, _cur_yaw, _cur_heading = store.root_state(clip_ids, cur_idx)
    next_root_pos, next_root_rot, _next_yaw, _next_heading = store.root_state(clip_ids, next_idx)
    return rebase_output_vector_root(store, transition_vec, cur_root_pos, cur_root_rot, next_root_pos, next_root_rot)


def advance_transition_state(
    store: SimpleClipStore,
    clip_ids: torch.Tensor,
    cur_idx: torch.Tensor,
    transition_vec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state_vec = advance_transition_output(store, clip_ids, cur_idx, transition_vec)
    return predicted_state_from_vector(state_vec, store)


def fast_clean_6d(d6: torch.Tensor) -> torch.Tensor:
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-8)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1, eps=1e-8)
    return torch.cat((b1, b2), dim=-1)


def fast_clean_ik_payload(payload: torch.Tensor) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for spec in tl.IK_PAYLOAD_SLICES:
        pos_slice = spec["pos"]
        rot_slice = spec["rot6"]
        pole_slice = spec["pole"]
        toe_slice = spec["toe_float"]
        assert isinstance(pos_slice, slice)
        assert isinstance(rot_slice, slice)
        assert isinstance(pole_slice, slice)
        parts.append(payload[:, pos_slice])
        parts.append(fast_clean_6d(payload[:, rot_slice]))
        parts.append(payload[:, pole_slice].clamp(-1.0, 1.0))
        if toe_slice is not None:
            assert isinstance(toe_slice, slice)
            parts.append(payload[:, toe_slice].clamp(-1.0, 1.0))
    return torch.cat(parts, dim=-1)


def _ik_base_positions_root(
    store: SimpleClipStore,
    pelvis_pos: torch.Tensor,
    pelvis_rot: torch.Tensor,
    core_rot6: torch.Tensor,
) -> torch.Tensor:
    clip = store.prototype
    b = int(pelvis_pos.shape[0])
    dtype = pelvis_pos.dtype
    offsets = store.local_offsets.to(dtype=dtype)
    fast_indices = store.ik_fast_base_indices
    if fast_indices is not None:
        def core_rot_for_name(bone_name: str) -> torch.Tensor:
            bone = int(fast_indices[bone_name])
            slot = int(clip.core_nonpelvis_map[bone])
            return tl.rotation_6d_to_matrix(core_rot6[:, slot])

        def child_pos(parent_pos: torch.Tensor, parent_rot: torch.Tensor, bone_name: str) -> torch.Tensor:
            offset = offsets[int(fast_indices[bone_name])].reshape(1, 3).expand(b, 3)
            return torch.matmul(offset.unsqueeze(1), parent_rot).squeeze(1) + parent_pos

        def child_rot(parent_rot: torch.Tensor, bone_name: str) -> torch.Tensor:
            return core_rot_for_name(bone_name) @ parent_rot

        pos = pelvis_pos
        rot = pelvis_rot
        for spine_name in ("spine_01", "spine_02", "spine_03", "spine_04", "spine_05"):
            pos = child_pos(pos, rot, spine_name)
            rot = child_rot(rot, spine_name)
        spine_pos = pos
        spine_rot = rot
        clav_l_pos = child_pos(spine_pos, spine_rot, "clavicle_l")
        clav_l_rot = child_rot(spine_rot, "clavicle_l")
        clav_r_pos = child_pos(spine_pos, spine_rot, "clavicle_r")
        clav_r_rot = child_rot(spine_rot, "clavicle_r")
        base_by_bone = {
            int(fast_indices["upperarm_l"]): child_pos(clav_l_pos, clav_l_rot, "upperarm_l"),
            int(fast_indices["upperarm_r"]): child_pos(clav_r_pos, clav_r_rot, "upperarm_r"),
            int(fast_indices["thigh_l"]): child_pos(pelvis_pos, pelvis_rot, "thigh_l"),
            int(fast_indices["thigh_r"]): child_pos(pelvis_pos, pelvis_rot, "thigh_r"),
        }
        return torch.stack([base_by_bone[int(start)] for start in store.ik_base_start_indices], dim=1)

    core_rot = tl.rotation_6d_to_matrix(core_rot6.reshape(-1, 6)).reshape(b, store.Jcore, 3, 3)
    identity = torch.eye(3, dtype=dtype, device=store.device).expand(b, 3, 3)
    pos_root: list[torch.Tensor | None] = [None] * int(clip.J)
    rot_root: list[torch.Tensor | None] = [None] * int(clip.J)
    for j in store.ik_base_eval_order:
        if j == int(clip.pelvis):
            local_pos = pelvis_pos
            local_rot = pelvis_rot
        else:
            local_pos = offsets[j].reshape(1, 3).expand(b, 3)
            local_rot = core_rot[:, int(clip.core_nonpelvis_map[j])] if j in clip.core_nonpelvis_map else identity
        parent = int(clip.parents_body_list[j])
        if parent < 0:
            pos_j = local_pos
            rot_j = local_rot
        else:
            parent_pos = pos_root[parent]
            parent_rot = rot_root[parent]
            assert parent_pos is not None and parent_rot is not None
            pos_j = torch.matmul(local_pos.unsqueeze(1), parent_rot).squeeze(1) + parent_pos
            rot_j = local_rot @ parent_rot
        pos_root[j] = pos_j
        rot_root[j] = rot_j
    starts = store.ik_base_start_indices
    return torch.stack([pos_root[start] for start in starts], dim=1)  # type: ignore[index]


def clamp_clean_ik_payload(
    payload: torch.Tensor,
    store: SimpleClipStore,
    pelvis_pos: torch.Tensor,
    pelvis_rot: torch.Tensor,
    core_rot6: torch.Tensor,
) -> torch.Tensor:
    cleaned = fast_clean_ik_payload(payload)
    if not store.prototype.ik_limb_specs:
        return cleaned
    base_root = _ik_base_positions_root(store, pelvis_pos, pelvis_rot, core_rot6)
    lengths = store.ik_limb_lengths.to(dtype=cleaned.dtype)
    rest_axis = store.ik_rest_axis.to(dtype=cleaned.dtype)
    end_parts: list[torch.Tensor] = []
    rot_parts: list[torch.Tensor] = []
    pole_parts: list[torch.Tensor] = []
    toe_parts: list[torch.Tensor | None] = []
    for spec in tl.IK_PAYLOAD_SLICES:
        pos_slice = spec["pos"]
        rot_slice = spec["rot6"]
        pole_slice = spec["pole"]
        toe_slice = spec["toe_float"]
        assert isinstance(pos_slice, slice)
        assert isinstance(rot_slice, slice)
        assert isinstance(pole_slice, slice)
        end_parts.append(cleaned[:, pos_slice])
        rot_parts.append(cleaned[:, rot_slice])
        pole_parts.append(cleaned[:, pole_slice])
        if toe_slice is not None:
            assert isinstance(toe_slice, slice)
            toe_parts.append(cleaned[:, toe_slice])
        else:
            toe_parts.append(None)
    end_root = torch.stack(end_parts, dim=1)
    delta = end_root - base_root
    d = torch.linalg.norm(delta, dim=-1, keepdim=True)
    fallback_axis = rest_axis.reshape(1, -1, 3).expand_as(delta)
    axis = torch.where(d > 1e-8, tl.normalize(delta), tl.normalize(fallback_axis))
    l1 = lengths[:, 0].reshape(1, -1, 1)
    l2 = lengths[:, 1].reshape(1, -1, 1)
    min_d = torch.abs(l1 - l2) + 1e-5
    max_d = l1 + l2 - 1e-5
    clamped_pos = base_root + axis * d.clamp_min(1e-8).clamp(min=min_d, max=max_d)
    parts: list[torch.Tensor] = []
    for limb_i, toe in enumerate(toe_parts):
        parts.append(clamped_pos[:, limb_i])
        parts.append(rot_parts[limb_i])
        parts.append(pole_parts[limb_i])
        if toe is not None:
            parts.append(toe)
    return torch.cat(parts, dim=-1)


def clean_output_vector(raw: torch.Tensor, store: SimpleClipStore) -> torch.Tensor:
    b = raw.shape[0]
    cursor = 0
    pelvis_pos = raw[:, cursor : cursor + 3]
    cursor += 3
    pelvis_rot6 = fast_clean_6d(raw[:, cursor : cursor + 6])
    cursor += 6
    core_dim = store.Jcore * 6
    core_rot6 = fast_clean_6d(raw[:, cursor : cursor + core_dim].reshape(-1, 6)).reshape(b, store.Jcore, 6)
    cursor += core_dim
    # IK end-effector position clamping is intentionally disabled here:
    # clamping projected the foot back onto the reachable sphere of its leg
    # base, which forcefully dragged a "planted" foot along with the pelvis
    # whenever the pelvis moved forward, defeating any foot-pin RL term. We
    # still clean 6D rotations and pole/toe scalars in `fast_clean_ik_payload`
    # so the rest of the pose layout stays valid.
    payload = fast_clean_ik_payload(raw[:, cursor : cursor + store.ik_payload_dim])
    return torch.cat((pelvis_pos, pelvis_rot6, core_rot6.reshape(b, -1), payload), dim=-1)


def predicted_state_from_raw(raw: torch.Tensor, store: SimpleClipStore) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vec = clean_output_vector(raw, store)
    return vec, vec[:, :3], vec[:, payload_slice(store)]


def predicted_state_from_vector(vec: torch.Tensor, store: SimpleClipStore) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return vec, vec[:, :3], vec[:, payload_slice(store)]


def model_forward(model: torch.nn.Module, inp: torch.Tensor, cur_vec: torch.Tensor, cfg: tl.TrainConfig) -> torch.Tensor:
    raw = model(inp)
    if cfg.predict_residual or tl.output_prediction_uses_residual():
        return cur_vec + raw
    return raw


def build_controller_input(
    store: SimpleClipStore,
    clip_ids: torch.Tensor,
    cur_idx: torch.Tensor,
    prev_vec: torch.Tensor,
    cur_vec: torch.Tensor,
    prev_pelvis: torch.Tensor,
    cur_pelvis: torch.Tensor,
    prev_payload: torch.Tensor,
    cur_payload: torch.Tensor,
) -> torch.Tensor:
    pelvis_vel = (cur_pelvis - prev_pelvis) / store.cfg.pose_delta_scale_final
    payload_vel = (cur_payload - prev_payload).reshape(cur_idx.shape[0], -1) / store.cfg.pose_delta_scale_final
    root_features = store.get_input_root_features(clip_ids, cur_idx)
    return torch.cat((cur_vec, prev_vec, pelvis_vel, payload_vel, root_features), dim=-1)


def ae_score_rows(
    ae: SimpleAutoencoder,
    mean: torch.Tensor,
    std: torch.Tensor,
    controller_input: torch.Tensor,
    predicted_output: torch.Tensor,
) -> torch.Tensor:
    input_dim = int(controller_input.shape[-1])
    feature = torch.cat((controller_input, predicted_output), dim=-1)
    x = (feature - mean) / std
    recon = ae(x)
    if AE_SCORE_OUTPUT_ONLY:
        return (recon[:, input_dim:] - x[:, input_dim:]).square().mean(dim=-1)
    return (recon - x).square().mean(dim=-1)


def identity_world_position_rows(
    store: SimpleClipStore,
    clip_ids: torch.Tensor,
    cur_idx: torch.Tensor,
    pred_vec: torch.Tensor,
    cur_vec: torch.Tensor,
) -> torch.Tensor:
    cur_root_pos, cur_root_rot, _cur_yaw, _cur_heading = store.root_state(clip_ids, cur_idx)
    pred_root_pos, pred_root_rot = transition_output_root_state(store, clip_ids, cur_idx)
    pred_pose, _pred_raw_pose = tl.output_to_pose(pred_vec, store.prototype)
    cur_pose, _cur_raw_pose = tl.output_to_pose(cur_vec, store.prototype)
    pred_pos, _pred_rot, _pred_canon = tl.fk_from_pose(
        store.prototype, pred_root_pos, pred_root_rot, pred_pose, store.device
    )
    cur_pos, _cur_rot, _cur_canon = tl.fk_from_pose(store.prototype, cur_root_pos, cur_root_rot, cur_pose, store.device)
    return (pred_pos - cur_pos).square().sum(dim=-1).mean(dim=-1)


def pure_ae_rollout_loss(
    model: torch.nn.Module,
    ae: SimpleAutoencoder,
    mean: torch.Tensor,
    std: torch.Tensor,
    store: SimpleClipStore,
    rollout_k: int,
    batch_size: int,
    start_pools: dict[int, StartPool],
    rl_cfg: RLLossConfig,
    ae_loss_weight: float,
    envelope: dict[str, torch.Tensor | dict[str, float | int | str]] | None = None,
    linear_slide_weight: float = 0.0,
    angular_slide_weight: float = 0.0,
    identity_loss_weight: float = 0.0,
    identity_maxabs_weight: float = 0.0,
    identity_world_pos_weight: float = 0.0,
    identity_pelvis_pos_weight: float = 0.0,
) -> ControllerLossResult:
    max_k = max(1, int(rollout_k))
    original_batch_size = max(1, int(batch_size))
    effective_k = sample_effective_rollout_k(original_batch_size, max_k, store.device)
    clip_ids, starts = sample_rollout_rows(start_pools, effective_k)
    cur_idx = starts
    prev_vec, prev_pelvis, prev_payload = target_state(store, clip_ids, cur_idx - 1)
    cur_vec, cur_pelvis, cur_payload = target_state(store, clip_ids, cur_idx)
    row_weight = (1.0 / effective_k.float()) / float(original_batch_size)
    total_loss = torch.zeros((), dtype=torch.float32, device=store.device)
    ae_total = torch.zeros_like(total_loss)
    identity_total = torch.zeros_like(total_loss)
    identity_maxabs_total = torch.zeros_like(total_loss)
    identity_world_pos_total = torch.zeros_like(total_loss)
    identity_pelvis_pos_total = torch.zeros_like(total_loss)
    rl_terms = {name: torch.zeros_like(total_loss) for name in RL_TERM_NAMES}
    envelope_terms = {name: torch.zeros_like(total_loss) for name in ENVELOPE_TERM_NAMES}
    has_envelope = envelope is not None and (float(linear_slide_weight) != 0.0 or float(angular_slide_weight) != 0.0)

    for step in range(max_k):
        inp = build_controller_input(
            store, clip_ids, cur_idx, prev_vec, cur_vec, prev_pelvis, cur_pelvis, prev_payload, cur_payload
        )
        raw = model_forward(model, inp, cur_vec, store.cfg)
        pred_vec = clean_output_vector(raw, store)
        cur_output_vec = current_state_as_transition_output(store, clip_ids, cur_idx, cur_vec)
        active = torch.ones_like(row_weight, dtype=torch.bool)
        if float(ae_loss_weight) != 0.0:
            ae_loss = (ae_score_rows(ae, mean, std, inp, pred_vec) * row_weight).sum() * float(ae_loss_weight)
        else:
            ae_loss = torch.zeros_like(total_loss)
        if float(identity_loss_weight) != 0.0:
            identity_loss = (
                (pred_vec - cur_output_vec).square().mean(dim=-1) * row_weight
            ).sum() * float(identity_loss_weight)
        else:
            identity_loss = torch.zeros_like(total_loss)
        if float(identity_maxabs_weight) != 0.0:
            identity_maxabs_loss = (
                (pred_vec - cur_output_vec).abs().amax(dim=-1) * row_weight
            ).sum() * float(identity_maxabs_weight)
        else:
            identity_maxabs_loss = torch.zeros_like(total_loss)
        if float(identity_world_pos_weight) != 0.0:
            identity_world_pos_loss = (
                identity_world_position_rows(store, clip_ids, cur_idx, pred_vec, cur_vec) * row_weight
            ).sum() * float(identity_world_pos_weight)
        else:
            identity_world_pos_loss = torch.zeros_like(total_loss)
        if float(identity_pelvis_pos_weight) != 0.0:
            identity_pelvis_pos_loss = (
                (pred_vec[:, :3] - cur_output_vec[:, :3]).square().sum(dim=-1) * row_weight
            ).sum() * float(identity_pelvis_pos_weight)
        else:
            identity_pelvis_pos_loss = torch.zeros_like(total_loss)
        if rl_cfg.world_foot_enabled:
            cur_root_pos_for_rl, cur_root_rot_for_rl, _yaw_for_rl, _heading_for_rl = store.root_state(clip_ids, cur_idx)
        else:
            cur_root_pos_for_rl = None
            cur_root_rot_for_rl = None
        rl_loss = compute_rl_loss(
            pred_vec,
            cur_output_vec,
            row_weight,
            active,
            rl_cfg,
            root_pos=cur_root_pos_for_rl,
            root_rot=cur_root_rot_for_rl,
        )
        ae_total = ae_total + ae_loss
        identity_total = identity_total + identity_loss
        identity_maxabs_total = identity_maxabs_total + identity_maxabs_loss
        identity_world_pos_total = identity_world_pos_total + identity_world_pos_loss
        identity_pelvis_pos_total = identity_pelvis_pos_total + identity_pelvis_pos_loss
        for key, value in rl_loss.terms.items():
            rl_terms[key] = rl_terms.get(key, torch.zeros_like(total_loss)) + value
        envelope_loss = torch.zeros_like(total_loss)
        if has_envelope:
            cur_root_pos, cur_root_rot, _cur_yaw, _cur_heading = store.root_state(clip_ids, cur_idx)
            next_root_pos, next_root_rot = transition_output_root_state(store, clip_ids, cur_idx)
            cur_foot_pos, cur_foot_rot = env.ik_foot_toe_state_from_vec(store, cur_root_pos, cur_root_rot, cur_vec)
            next_foot_pos, next_foot_rot = env.ik_foot_toe_state_from_vec(store, next_root_pos, next_root_rot, pred_vec)
            linear_rows, angular_rows = env.envelope_excess_ik_state_rows(
                store,
                envelope,  # type: ignore[arg-type]
                cur_foot_pos,
                cur_foot_rot,
                next_foot_pos,
                next_foot_rot,
                clip_ids,
                cur_idx,
            )
            linear_loss = (linear_rows * row_weight).sum() * float(linear_slide_weight)
            angular_loss = (angular_rows * row_weight).sum() * float(angular_slide_weight)
            envelope_terms["linear_slide_weighted"] = envelope_terms["linear_slide_weighted"] + linear_loss
            envelope_terms["angular_slide_weighted"] = envelope_terms["angular_slide_weighted"] + angular_loss
            envelope_loss = linear_loss + angular_loss
        total_loss = (
            total_loss
            + ae_loss
            + identity_loss
            + identity_maxabs_loss
            + identity_world_pos_loss
            + identity_pelvis_pos_loss
            + rl_loss.total
            + envelope_loss
        )
        if step + 1 >= max_k:
            break

        continuing = effective_k > (step + 1)
        rows = continuing.nonzero(as_tuple=False).flatten()
        if rows.numel() == 0:
            break
        clip_ids = clip_ids.index_select(0, rows)
        reset = training_reset_rows(store, clip_ids, cur_idx.index_select(0, rows), torch.ones_like(rows, dtype=torch.bool))
        selected_cur_idx = cur_idx.index_select(0, rows)
        next_vec, next_pelvis, next_payload = advance_transition_state(
            store,
            clip_ids,
            selected_cur_idx,
            pred_vec.index_select(0, rows),
        )
        next_idx = cur_idx.index_select(0, rows) + 1
        reset_starts = sample_same_clip_training_starts(store, clip_ids)
        reset_prev_vec, reset_prev_pelvis, reset_prev_payload = target_state(store, clip_ids, reset_starts - 1)
        reset_cur_vec, reset_cur_pelvis, reset_cur_payload = target_state(store, clip_ids, reset_starts)
        reset_mask = reset[:, None]
        prev_vec = torch.where(reset_mask, reset_prev_vec, cur_vec.index_select(0, rows))
        prev_pelvis = torch.where(reset_mask, reset_prev_pelvis, cur_pelvis.index_select(0, rows))
        prev_payload = torch.where(reset_mask, reset_prev_payload, cur_payload.index_select(0, rows))
        cur_vec = torch.where(reset_mask, reset_cur_vec, next_vec)
        cur_pelvis = torch.where(reset_mask, reset_cur_pelvis, next_pelvis)
        cur_payload = torch.where(reset_mask, reset_cur_payload, next_payload)
        cur_idx = torch.where(reset, reset_starts, next_idx)
        effective_k = effective_k.index_select(0, rows)
        row_weight = row_weight.index_select(0, rows)
    return ControllerLossResult(
        total=total_loss,
        terms={
            "ae_score": ae_total,
            IDENTITY_TERM_NAME: identity_total,
            IDENTITY_MAXABS_TERM_NAME: identity_maxabs_total,
            IDENTITY_WORLD_POS_TERM_NAME: identity_world_pos_total,
            IDENTITY_PELVIS_POS_TERM_NAME: identity_pelvis_pos_total,
            **rl_terms,
            **envelope_terms,
        },
    )


def pure_ae_rollout_loss_static(
    model: torch.nn.Module,
    ae: SimpleAutoencoder,
    mean: torch.Tensor,
    std: torch.Tensor,
    store: SimpleClipStore,
    rollout_k: int,
    batch_size: int,
    effective_k: torch.Tensor,
    clip_ids: torch.Tensor,
    starts: torch.Tensor,
    reset_starts_by_step: torch.Tensor | None = None,
    rl_cfg: RLLossConfig | None = None,
    ae_loss_weight: float = 1.0,
    envelope: dict[str, torch.Tensor | dict[str, float | int | str]] | None = None,
    linear_slide_weight: float = 0.0,
    angular_slide_weight: float = 0.0,
    identity_loss_weight: float = 0.0,
    identity_maxabs_weight: float = 0.0,
    identity_world_pos_weight: float = 0.0,
    identity_pelvis_pos_weight: float = 0.0,
) -> ControllerLossResult:
    rl_cfg = rl_cfg or RLLossConfig()
    max_k = max(1, int(rollout_k))
    cur_idx = starts
    prev_vec, prev_pelvis, prev_payload = target_state(store, clip_ids, cur_idx - 1)
    cur_vec, cur_pelvis, cur_payload = target_state(store, clip_ids, cur_idx)
    row_weight = (1.0 / effective_k.float()) / float(max(1, int(batch_size)))
    total_loss = torch.zeros((), dtype=torch.float32, device=store.device)
    ae_total = torch.zeros_like(total_loss)
    identity_total = torch.zeros_like(total_loss)
    identity_maxabs_total = torch.zeros_like(total_loss)
    identity_world_pos_total = torch.zeros_like(total_loss)
    identity_pelvis_pos_total = torch.zeros_like(total_loss)
    rl_terms = {name: torch.zeros_like(total_loss) for name in RL_TERM_NAMES}
    envelope_terms = {name: torch.zeros_like(total_loss) for name in ENVELOPE_TERM_NAMES}
    has_envelope = envelope is not None and (float(linear_slide_weight) != 0.0 or float(angular_slide_weight) != 0.0)

    for step in range(max_k):
        inp = build_controller_input(
            store, clip_ids, cur_idx, prev_vec, cur_vec, prev_pelvis, cur_pelvis, prev_payload, cur_payload
        )
        raw = model_forward(model, inp, cur_vec, store.cfg)
        pred_vec = clean_output_vector(raw, store)
        cur_output_vec = current_state_as_transition_output(store, clip_ids, cur_idx, cur_vec)
        active = effective_k > step
        if float(ae_loss_weight) != 0.0:
            ae_loss = (ae_score_rows(ae, mean, std, inp, pred_vec) * row_weight * active.float()).sum() * float(ae_loss_weight)
        else:
            ae_loss = torch.zeros_like(total_loss)
        if float(identity_loss_weight) != 0.0:
            identity_loss = (
                (pred_vec - cur_output_vec).square().mean(dim=-1) * row_weight * active.float()
            ).sum() * float(identity_loss_weight)
        else:
            identity_loss = torch.zeros_like(total_loss)
        if float(identity_maxabs_weight) != 0.0:
            identity_maxabs_loss = (
                (pred_vec - cur_output_vec).abs().amax(dim=-1) * row_weight * active.float()
            ).sum() * float(identity_maxabs_weight)
        else:
            identity_maxabs_loss = torch.zeros_like(total_loss)
        if float(identity_world_pos_weight) != 0.0:
            active_f = active.float()
            identity_world_pos_loss = (
                identity_world_position_rows(store, clip_ids, cur_idx, pred_vec, cur_vec) * row_weight * active_f
            ).sum() * float(identity_world_pos_weight)
        else:
            identity_world_pos_loss = torch.zeros_like(total_loss)
        if float(identity_pelvis_pos_weight) != 0.0:
            active_f = active.float()
            identity_pelvis_pos_loss = (
                (pred_vec[:, :3] - cur_output_vec[:, :3]).square().sum(dim=-1) * row_weight * active_f
            ).sum() * float(identity_pelvis_pos_weight)
        else:
            identity_pelvis_pos_loss = torch.zeros_like(total_loss)
        if rl_cfg.world_foot_enabled:
            cur_root_pos_for_rl, cur_root_rot_for_rl, _yaw_for_rl, _heading_for_rl = store.root_state(clip_ids, cur_idx)
        else:
            cur_root_pos_for_rl = None
            cur_root_rot_for_rl = None
        rl_loss = compute_rl_loss(
            pred_vec,
            cur_output_vec,
            row_weight * active.to(dtype=row_weight.dtype),
            torch.ones_like(active, dtype=torch.bool),
            rl_cfg,
            root_pos=cur_root_pos_for_rl,
            root_rot=cur_root_rot_for_rl,
        )
        ae_total = ae_total + ae_loss
        identity_total = identity_total + identity_loss
        identity_maxabs_total = identity_maxabs_total + identity_maxabs_loss
        identity_world_pos_total = identity_world_pos_total + identity_world_pos_loss
        identity_pelvis_pos_total = identity_pelvis_pos_total + identity_pelvis_pos_loss
        for key, value in rl_loss.terms.items():
            rl_terms[key] = rl_terms.get(key, torch.zeros_like(total_loss)) + value
        envelope_loss = torch.zeros_like(total_loss)
        if has_envelope:
            active_f = active.float()
            cur_root_pos, cur_root_rot, _cur_yaw, _cur_heading = store.root_state(clip_ids, cur_idx)
            next_root_pos, next_root_rot = transition_output_root_state(store, clip_ids, cur_idx)
            cur_foot_pos, cur_foot_rot = env.ik_foot_toe_state_from_vec(store, cur_root_pos, cur_root_rot, cur_vec)
            next_foot_pos, next_foot_rot = env.ik_foot_toe_state_from_vec(store, next_root_pos, next_root_rot, pred_vec)
            linear_rows, angular_rows = env.envelope_excess_ik_state_rows(
                store,
                envelope,  # type: ignore[arg-type]
                cur_foot_pos,
                cur_foot_rot,
                next_foot_pos,
                next_foot_rot,
                clip_ids,
                cur_idx,
            )
            linear_loss = (linear_rows * row_weight * active_f).sum() * float(linear_slide_weight)
            angular_loss = (angular_rows * row_weight * active_f).sum() * float(angular_slide_weight)
            envelope_terms["linear_slide_weighted"] = envelope_terms["linear_slide_weighted"] + linear_loss
            envelope_terms["angular_slide_weighted"] = envelope_terms["angular_slide_weighted"] + angular_loss
            envelope_loss = linear_loss + angular_loss
        total_loss = (
            total_loss
            + ae_loss
            + identity_loss
            + identity_maxabs_loss
            + identity_world_pos_loss
            + identity_pelvis_pos_loss
            + rl_loss.total
            + envelope_loss
        )
        if step + 1 >= max_k:
            break

        continuing = effective_k > (step + 1)
        next_vec, next_pelvis, next_payload = advance_transition_state(store, clip_ids, cur_idx, pred_vec)
        reset = training_reset_rows(store, clip_ids, cur_idx, continuing)
        advance = continuing & (~reset)
        reset_starts = (
            reset_starts_by_step[step]
            if reset_starts_by_step is not None
            else sample_same_clip_training_starts(store, clip_ids)
        )
        reset_prev_vec, reset_prev_pelvis, reset_prev_payload = target_state(store, clip_ids, reset_starts - 1)
        reset_cur_vec, reset_cur_pelvis, reset_cur_payload = target_state(store, clip_ids, reset_starts)
        reset_mask = reset[:, None]
        advance_mask = advance[:, None]
        prev_vec = torch.where(reset_mask, reset_prev_vec, torch.where(advance_mask, cur_vec, prev_vec))
        prev_pelvis = torch.where(reset_mask, reset_prev_pelvis, torch.where(advance_mask, cur_pelvis, prev_pelvis))
        prev_payload = torch.where(reset_mask, reset_prev_payload, torch.where(advance_mask, cur_payload, prev_payload))
        cur_vec = torch.where(reset_mask, reset_cur_vec, torch.where(advance_mask, next_vec, cur_vec))
        cur_pelvis = torch.where(reset_mask, reset_cur_pelvis, torch.where(advance_mask, next_pelvis, cur_pelvis))
        cur_payload = torch.where(reset_mask, reset_cur_payload, torch.where(advance_mask, next_payload, cur_payload))
        cur_idx = torch.where(reset, reset_starts, torch.where(continuing, cur_idx + 1, cur_idx))
    return ControllerLossResult(
        total=total_loss,
        terms={
            "ae_score": ae_total,
            IDENTITY_TERM_NAME: identity_total,
            IDENTITY_MAXABS_TERM_NAME: identity_maxabs_total,
            IDENTITY_WORLD_POS_TERM_NAME: identity_world_pos_total,
            IDENTITY_PELVIS_POS_TERM_NAME: identity_pelvis_pos_total,
            **rl_terms,
            **envelope_terms,
        },
    )


class CudaGraphPureAEStep:
    kind = "cuda_graph_static_masked"

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        ae: SimpleAutoencoder,
        mean: torch.Tensor,
        std: torch.Tensor,
        store: SimpleClipStore,
        rollout_k: int,
        batch_size: int,
        start_pools: dict[int, StartPool],
        rl_cfg: RLLossConfig,
        ae_loss_weight: float,
        envelope: dict[str, torch.Tensor | dict[str, float | int | str]] | None,
        linear_slide_weight: float,
        angular_slide_weight: float,
        identity_loss_weight: float,
        identity_maxabs_weight: float,
        identity_world_pos_weight: float,
        identity_pelvis_pos_weight: float,
    ):
        if store.device.type != "cuda":
            raise RuntimeError("CudaGraphPureAEStep requires CUDA")
        self.model = model
        self.optimizer = optimizer
        self.ae = ae
        self.mean = mean
        self.std = std
        self.store = store
        self.rollout_k = int(rollout_k)
        self.batch_size = int(batch_size)
        self.start_pools = start_pools
        self.rl_cfg = rl_cfg
        self.ae_loss_weight = float(ae_loss_weight)
        self.envelope = envelope
        self.linear_slide_weight = float(linear_slide_weight)
        self.angular_slide_weight = float(angular_slide_weight)
        self.identity_loss_weight = float(identity_loss_weight)
        self.identity_maxabs_weight = float(identity_maxabs_weight)
        self.identity_world_pos_weight = float(identity_world_pos_weight)
        self.identity_pelvis_pos_weight = float(identity_pelvis_pos_weight)
        self.effective_k = torch.empty((self.batch_size,), dtype=torch.long, device=store.device)
        self.clip_ids = torch.empty_like(self.effective_k)
        self.starts = torch.empty_like(self.effective_k)
        self.reset_starts_by_step = torch.empty(
            (max(1, int(self.rollout_k)), self.batch_size), dtype=torch.long, device=store.device
        )
        self.loss = torch.zeros((), dtype=torch.float32, device=store.device)
        self.term_names = (
            "ae_score",
            IDENTITY_TERM_NAME,
            IDENTITY_MAXABS_TERM_NAME,
            IDENTITY_WORLD_POS_TERM_NAME,
            IDENTITY_PELVIS_POS_TERM_NAME,
            *RL_TERM_NAMES,
            *ENVELOPE_TERM_NAMES,
        )
        self.term_tensors = {name: torch.zeros_like(self.loss) for name in self.term_names}
        self.graph = torch.cuda.CUDAGraph()
        self._capture()

    def _sample_into_static_buffers(self) -> None:
        effective_k = sample_effective_rollout_k(self.batch_size, self.rollout_k, self.store.device)
        clip_ids, starts = sample_rollout_rows(self.start_pools, effective_k)
        self.effective_k.copy_(effective_k)
        self.clip_ids.copy_(clip_ids)
        self.starts.copy_(starts)
        self.reset_starts_by_step.copy_(
            sample_same_clip_training_starts_by_step(self.store, clip_ids, self.reset_starts_by_step.shape[0])
        )

    def _loss(self) -> ControllerLossResult:
        return pure_ae_rollout_loss_static(
            self.model,
            self.ae,
            self.mean,
            self.std,
            self.store,
            self.rollout_k,
            self.batch_size,
            self.effective_k,
            self.clip_ids,
            self.starts,
            self.reset_starts_by_step,
            self.rl_cfg,
            self.ae_loss_weight,
            self.envelope,
            self.linear_slide_weight,
            self.angular_slide_weight,
            self.identity_loss_weight,
            self.identity_maxabs_weight,
            self.identity_world_pos_weight,
            self.identity_pelvis_pos_weight,
        )

    def _capture(self) -> None:
        self._sample_into_static_buffers()
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(2):
                self._sample_into_static_buffers()
                self.optimizer.zero_grad(set_to_none=False)
                loss_result = self._loss()
                loss_result.total.backward()
                maybe_clip_rl_gradients(self.model, self.rl_cfg)
                self.optimizer.step()
                del loss_result
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()
        self.optimizer.zero_grad(set_to_none=False)
        with torch.cuda.graph(self.graph):
            self.optimizer.zero_grad(set_to_none=False)
            loss_result = self._loss()
            self.loss = loss_result.total
            for name in self.term_names:
                self.term_tensors[name] = loss_result.terms[name]
            self.loss.backward()
            maybe_clip_rl_gradients(self.model, self.rl_cfg)
            self.optimizer.step()

    def step(self) -> ControllerLossResult:
        self._sample_into_static_buffers()
        self.graph.replay()
        return ControllerLossResult(
            total=self.loss.detach(),
            terms={name: self.term_tensors[name].detach() for name in self.term_names},
        )


class EagerPureAEStep:
    kind = "eager"

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        ae: SimpleAutoencoder,
        mean: torch.Tensor,
        std: torch.Tensor,
        store: SimpleClipStore,
        rollout_k: int,
        batch_size: int,
        start_pools: dict[int, StartPool],
        rl_cfg: RLLossConfig,
        ae_loss_weight: float,
        envelope: dict[str, torch.Tensor | dict[str, float | int | str]] | None,
        linear_slide_weight: float,
        angular_slide_weight: float,
        identity_loss_weight: float,
        identity_maxabs_weight: float,
        identity_world_pos_weight: float,
        identity_pelvis_pos_weight: float,
    ):
        self.model = model
        self.optimizer = optimizer
        self.ae = ae
        self.mean = mean
        self.std = std
        self.store = store
        self.rollout_k = int(rollout_k)
        self.batch_size = int(batch_size)
        self.start_pools = start_pools
        self.rl_cfg = rl_cfg
        self.ae_loss_weight = float(ae_loss_weight)
        self.envelope = envelope
        self.linear_slide_weight = float(linear_slide_weight)
        self.angular_slide_weight = float(angular_slide_weight)
        self.identity_loss_weight = float(identity_loss_weight)
        self.identity_maxabs_weight = float(identity_maxabs_weight)
        self.identity_world_pos_weight = float(identity_world_pos_weight)
        self.identity_pelvis_pos_weight = float(identity_pelvis_pos_weight)

    def step(self) -> ControllerLossResult:
        loss_result = pure_ae_rollout_loss(
            self.model,
            self.ae,
            self.mean,
            self.std,
            self.store,
            self.rollout_k,
            self.batch_size,
            self.start_pools,
            self.rl_cfg,
            self.ae_loss_weight,
            self.envelope,
            self.linear_slide_weight,
            self.angular_slide_weight,
            self.identity_loss_weight,
            self.identity_maxabs_weight,
            self.identity_world_pos_weight,
            self.identity_pelvis_pos_weight,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss_result.total.backward()
        maybe_clip_rl_gradients(self.model, self.rl_cfg)
        self.optimizer.step()
        return detach_loss_result(loss_result)


def make_pure_ae_stepper(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ae: SimpleAutoencoder,
    mean: torch.Tensor,
    std: torch.Tensor,
    store: SimpleClipStore,
    rollout_k: int,
    batch_size: int,
    start_pools: dict[int, StartPool],
    rl_cfg: RLLossConfig,
    ae_loss_weight: float,
    envelope: dict[str, torch.Tensor | dict[str, float | int | str]] | None = None,
    linear_slide_weight: float = 0.0,
    angular_slide_weight: float = 0.0,
    identity_loss_weight: float = 0.0,
    identity_maxabs_weight: float = 0.0,
    identity_world_pos_weight: float = 0.0,
    identity_pelvis_pos_weight: float = 0.0,
) -> CudaGraphPureAEStep | EagerPureAEStep:
    # The FK-based identity terms currently allocate small constants inside FK;
    # CUDA graph capture forbids that, so keep this diagnostic loss on eager.
    # The world-foot RL terms (foot pin / no-hover / foot floor) need
    # store.root_state per step and call into torch ops that have not been
    # graph-tested; keep them on eager too.
    if (
        store.device.type == "cuda"
        and float(identity_world_pos_weight) == 0.0
        and not rl_cfg.world_foot_enabled
    ):
        return CudaGraphPureAEStep(
            model,
            optimizer,
            ae,
            mean,
            std,
            store,
            rollout_k,
            batch_size,
            start_pools,
            rl_cfg,
            ae_loss_weight,
            envelope,
            linear_slide_weight,
            angular_slide_weight,
            identity_loss_weight,
            identity_maxabs_weight,
            identity_world_pos_weight,
            identity_pelvis_pos_weight,
        )
    return EagerPureAEStep(
        model,
        optimizer,
        ae,
        mean,
        std,
        store,
        rollout_k,
        batch_size,
        start_pools,
        rl_cfg,
        ae_loss_weight,
        envelope,
        linear_slide_weight,
        angular_slide_weight,
        identity_loss_weight,
        identity_maxabs_weight,
        identity_world_pos_weight,
        identity_pelvis_pos_weight,
    )


def validation_rows(pool: StartPool, max_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    if pool.row_count <= max_rows:
        return pool.clip_ids, pool.starts
    rows = torch.linspace(0, pool.row_count - 1, steps=max_rows, device=pool.starts.device).round().long().unique()
    return pool.clip_ids.index_select(0, rows), pool.starts.index_select(0, rows)


@torch.no_grad()
def validation_ae_score(
    model: torch.nn.Module,
    ae: SimpleAutoencoder,
    mean: torch.Tensor,
    std: torch.Tensor,
    store: SimpleClipStore,
    rollout_k: int,
    pool: StartPool,
) -> float:
    clip_ids, starts = validation_rows(pool, VALIDATION_ROWS)
    cur_idx = starts
    prev_vec, prev_pelvis, prev_payload = target_state(store, clip_ids, cur_idx - 1)
    cur_vec, cur_pelvis, cur_payload = target_state(store, clip_ids, cur_idx)
    total = 0.0
    count = 0
    for step in range(max(1, int(rollout_k))):
        inp = build_controller_input(
            store, clip_ids, cur_idx, prev_vec, cur_vec, prev_pelvis, cur_pelvis, prev_payload, cur_payload
        )
        raw = model_forward(model, inp, cur_vec, store.cfg)
        pred_vec = clean_output_vector(raw, store)
        score = ae_score_rows(ae, mean, std, inp, pred_vec)
        total += float(score.sum().detach().cpu())
        count += int(score.numel())
        if step + 1 >= int(rollout_k):
            break
        prev_vec = cur_vec
        prev_pelvis = cur_pelvis
        prev_payload = cur_payload
        cur_vec, cur_pelvis, cur_payload = advance_transition_state(store, clip_ids, cur_idx, pred_vec)
        cur_idx = cur_idx + 1
    return total / float(max(1, count))


def pose_rows(pose: dict[str, torch.Tensor], rows: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value.index_select(0, rows) for key, value in pose.items()}


@torch.no_grad()
def fk_positions_by_clip(
    store: SimpleClipStore,
    clip_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    pose: dict[str, torch.Tensor],
) -> torch.Tensor:
    out = torch.empty((clip_ids.shape[0], store.J, 3), dtype=root_pos.dtype, device=store.device)
    for clip_id in clip_ids.unique().tolist():
        rows = (clip_ids == int(clip_id)).nonzero(as_tuple=False).flatten()
        pos, _rot, _canon = tl.fk_from_pose(
            store.clips[int(clip_id)],
            root_pos.index_select(0, rows),
            root_rot.index_select(0, rows),
            pose_rows(pose, rows),
            store.device,
        )
        out[rows] = pos
    return out


@torch.no_grad()
def rollout_joint_error(
    model: torch.nn.Module,
    store: SimpleClipStore,
    rollout_k: int,
    pool: StartPool,
) -> tuple[float, float]:
    clip_ids, starts = validation_rows(pool, VALIDATION_ROWS)
    cur_idx = starts
    prev_vec, prev_pelvis, prev_payload = target_state(store, clip_ids, cur_idx - 1)
    cur_vec, cur_pelvis, cur_payload = target_state(store, clip_ids, cur_idx)
    total_error = 0.0
    total_frames = 0
    max_error = 0.0
    for step in range(max(1, int(rollout_k))):
        inp = build_controller_input(
            store, clip_ids, cur_idx, prev_vec, cur_vec, prev_pelvis, cur_pelvis, prev_payload, cur_payload
        )
        raw = model_forward(model, inp, cur_vec, store.cfg)
        pred_vec = clean_output_vector(raw, store)
        pred_pose, _raw_pose = tl.output_to_pose(pred_vec, store.prototype)
        target_idx = cur_idx + 1
        target_root_pos, target_root_rot, _target_yaw, _target_heading = store.root_state(clip_ids, target_idx)
        pred_root_pos, pred_root_rot = transition_output_root_state(store, clip_ids, cur_idx)
        pred_global = fk_positions_by_clip(store, clip_ids, pred_root_pos, pred_root_rot, pred_pose)
        target_global = fk_positions_by_clip(
            store, clip_ids, target_root_pos, target_root_rot, store.get_pose(clip_ids, target_idx)
        )
        per_frame = (pred_global - target_global).norm(dim=-1).mean(dim=-1)
        total_error += float(per_frame.sum().detach().cpu())
        total_frames += int(per_frame.numel())
        max_error = max(max_error, float(per_frame.max().detach().cpu()))
        if step + 1 >= int(rollout_k):
            break
        prev_vec = cur_vec
        prev_pelvis = cur_pelvis
        prev_payload = cur_payload
        cur_vec, cur_pelvis, cur_payload = advance_transition_state(store, clip_ids, cur_idx, pred_vec)
        cur_idx = target_idx
    return total_error / float(max(1, total_frames)), max_error


def save_controller_checkpoint(
    run_dir: Path,
    run_id: str,
    tag: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best: float,
    rollout_k: int,
    cfg: tl.TrainConfig,
    metadata: dict,
) -> Path:
    path = checkpoint_path(run_dir, run_id, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tl.checkpoint_payload(model, optimizer, step, best, rollout_k, cfg, metadata), path)
    return path


def load_controller_init_checkpoint(model: torch.nn.Module, path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt_runtime.require_current_ik_controller_checkpoint(ckpt, path)
    policy = ckpt_runtime.checkpoint_policy(ckpt)
    checkpoint_root = str(policy.get("output_reference_root", "")).strip().lower()
    if checkpoint_root != tl.OUTPUT_REFERENCE_ROOT:
        raise ValueError(
            f"Controller init checkpoint uses output_reference_root={checkpoint_root!r}; "
            f"expected {tl.OUTPUT_REFERENCE_ROOT!r}: {path}"
        )
    checkpoint_prediction = str(policy.get("output_prediction_mode", tl.OUTPUT_PREDICTION_MODE_ABSOLUTE)).strip().lower()
    if checkpoint_prediction != tl.normalized_output_prediction_mode():
        raise ValueError(
            f"Controller init checkpoint uses output_prediction_mode={checkpoint_prediction!r}; "
            f"expected {tl.normalized_output_prediction_mode()!r}: {path}"
        )
    model.load_state_dict(ckpt["model"], strict=True)
    return ckpt


def main() -> None:
    global LOG_EVERY, MIXED_ROLLOUT_AT_MAX, ROLLOUT_K, ROLLOUT_SCHEDULE, ROLLOUT_STAGE_STEPS, STAGE_LEARNING_RATES

    parser = argparse.ArgumentParser(description="Train IK controller with simple AE and optional separated RL losses.")
    parser.add_argument("--npz", default=None)
    parser.add_argument("--periodic-folder", default=None)
    parser.add_argument("--nonperiodic-folder", default=None)
    parser.add_argument("--ae-checkpoint", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--run-label", default="walkF_simple_ae_controller")
    parser.add_argument("--ae-loss-weight", type=float, default=1.0)
    parser.add_argument("--rollout-schedule", type=int, nargs="+", default=None)
    parser.add_argument("--rollout-stage-steps", type=int, nargs="+", default=None)
    parser.add_argument("--rollout-k", type=int, default=None)
    parser.add_argument("--mixed-rollout-at-max", dest="mixed_rollout_at_max", action="store_true", default=None)
    parser.add_argument("--no-mixed-rollout-at-max", dest="mixed_rollout_at_max", action="store_false")
    parser.add_argument("--stage-learning-rate", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--identity-output-loss-weight", type=float, default=0.0)
    parser.add_argument("--identity-output-maxabs-loss-weight", type=float, default=0.0)
    parser.add_argument("--identity-world-pos-loss-weight", type=float, default=0.0)
    parser.add_argument("--identity-pelvis-pos-loss-weight", type=float, default=0.0)
    parser.add_argument("--linear-slide-loss-weight", type=float, default=0.0)
    parser.add_argument("--angular-slide-loss-weight", type=float, default=0.0)
    parser.add_argument("--envelope-margin", type=float, default=env.ExcessEnvelopeConfig().margin)
    parser.add_argument("--envelope-knn", type=int, default=env.ExcessEnvelopeConfig().knn)
    parser.add_argument("--pelvis-root-horizontal-loss-weight", type=float, default=0.0)
    parser.add_argument("--pelvis-root-horizontal-limit-m", type=float, default=RLLossConfig().pelvis_root_horizontal_limit_m)
    parser.add_argument("--pelvis-root-rotation-loss-weight", type=float, default=0.0)
    parser.add_argument("--pelvis-root-rotation-limit-deg", type=float, default=RLLossConfig().pelvis_root_rotation_limit_deg)
    parser.add_argument("--end-effector-location-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--end-effector-location-limit-m",
        type=float,
        nargs=4,
        default=RLLossConfig().end_effector_location_limit_m,
        metavar=("HAND_L", "HAND_R", "FOOT_L", "FOOT_R"),
    )
    parser.add_argument("--end-effector-rotation-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--end-effector-rotation-limit-deg",
        type=float,
        nargs=4,
        default=RLLossConfig().end_effector_rotation_limit_deg,
        metavar=("HAND_L", "HAND_R", "FOOT_L", "FOOT_R"),
    )
    parser.add_argument("--end-effector-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--end-effector-velocity-limit-mps", type=float, default=RLLossConfig().end_effector_velocity_limit_mps)
    parser.add_argument("--end-effector-angular-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--end-effector-angular-velocity-limit-deg-s",
        type=float,
        default=RLLossConfig().end_effector_angular_velocity_limit_deg_s,
    )
    parser.add_argument("--pelvis-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--pelvis-velocity-limit-mps", type=float, default=RLLossConfig().pelvis_velocity_limit_mps)
    parser.add_argument("--pelvis-angular-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--pelvis-angular-velocity-limit-deg-s", type=float, default=RLLossConfig().pelvis_angular_velocity_limit_deg_s)
    parser.add_argument("--core-angular-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--core-angular-velocity-limit-deg-s", type=float, default=RLLossConfig().core_angular_velocity_limit_deg_s)
    parser.add_argument("--foot-pin-loss-weight", type=float, default=0.0,
                        help="Weight for soft-contact-masked foot world-velocity penalty (pinned foot must not slide).")
    parser.add_argument("--foot-pin-height-threshold-m", type=float, default=RLLossConfig().foot_pin_height_threshold_m,
                        help="World-y of the foot bone at which soft contact = 0.5.")
    parser.add_argument("--foot-pin-height-temp-m", type=float, default=RLLossConfig().foot_pin_height_temp_m,
                        help="Sigmoid temperature (meters) for the soft contact mask.")
    parser.add_argument("--foot-pin-speed-floor-mps", type=float, default=RLLossConfig().foot_pin_speed_floor_mps,
                        help="Foot horizontal world speed (m/s) below which slide is not penalised.")
    parser.add_argument("--no-hover-loss-weight", type=float, default=0.0,
                        help="Weight for the both-feet-off-the-ground hover penalty.")
    parser.add_argument("--no-hover-height-threshold-m", type=float, default=RLLossConfig().no_hover_height_threshold_m,
                        help="World-y where the per-foot contact-band sigmoid is at 0.5 for the no-hover term.")
    parser.add_argument("--no-hover-height-temp-m", type=float, default=RLLossConfig().no_hover_height_temp_m,
                        help="Sigmoid temperature (meters) for the no-hover soft mask.")
    parser.add_argument("--foot-floor-loss-weight", type=float, default=0.0,
                        help="Weight for penalising foot world-y dipping below the floor plane.")
    parser.add_argument("--foot-floor-y-m", type=float, default=RLLossConfig().foot_floor_y_m,
                        help="World-y of the floor plane (default 0).")
    parser.add_argument("--foot-ceiling-loss-weight", type=float, default=0.0,
                        help="Weight for penalising foot world-y above the ceiling height (caps swing height).")
    parser.add_argument("--foot-ceiling-y-m", type=float, default=RLLossConfig().foot_ceiling_y_m,
                        help="World-y of the foot ceiling (default 0.15m). Feet rising above this are penalised.")
    parser.add_argument("--pelvis-height-loss-weight", type=float, default=0.0,
                        help="Weight for penalising deviation of pelvis-local Z (root-local up axis) from the target height.")
    parser.add_argument("--pelvis-height-target-m", type=float, default=RLLossConfig().pelvis_height_target_m,
                        help="Target pelvis-local Z (m) for the pelvis-height constraint. Default is walkF mean (~0.886).")
    parser.add_argument("--pelvis-height-tolerance-m", type=float, default=RLLossConfig().pelvis_height_tolerance_m,
                        help="Tolerance band around the pelvis-height target before the penalty engages.")
    args = parser.parse_args()

    if args.rollout_schedule is not None:
        ROLLOUT_SCHEDULE = tuple(max(1, int(k)) for k in args.rollout_schedule)
        if args.rollout_k is None:
            ROLLOUT_K = int(ROLLOUT_SCHEDULE[-1])
    if args.rollout_stage_steps is not None:
        ROLLOUT_STAGE_STEPS = tuple(max(1, int(n)) for n in args.rollout_stage_steps)
    if args.rollout_k is not None:
        ROLLOUT_K = max(1, int(args.rollout_k))
    if len(ROLLOUT_SCHEDULE) != len(ROLLOUT_STAGE_STEPS):
        raise ValueError(
            f"rollout schedule length {len(ROLLOUT_SCHEDULE)} must match stage steps length {len(ROLLOUT_STAGE_STEPS)}"
        )
    if int(ROLLOUT_K) not in {int(k) for k in ROLLOUT_SCHEDULE}:
        raise ValueError(f"rollout_k {ROLLOUT_K} must appear in rollout_schedule {ROLLOUT_SCHEDULE}")
    if args.mixed_rollout_at_max is not None:
        MIXED_ROLLOUT_AT_MAX = bool(args.mixed_rollout_at_max)
    if args.stage_learning_rate is not None:
        STAGE_LEARNING_RATES = {int(k): float(args.stage_learning_rate) for k in ROLLOUT_SCHEDULE}
    elif args.rollout_schedule is not None:
        STAGE_LEARNING_RATES = default_stage_learning_rates_for_schedule(ROLLOUT_SCHEDULE)
    if args.log_every is not None:
        LOG_EVERY = max(1, int(args.log_every))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    ae_loss_weight = float(args.ae_loss_weight)
    identity_loss_weight = float(args.identity_output_loss_weight)
    identity_maxabs_weight = float(args.identity_output_maxabs_loss_weight)
    identity_world_pos_weight = float(args.identity_world_pos_loss_weight)
    identity_pelvis_pos_weight = float(args.identity_pelvis_pos_loss_weight)
    ae_path: Path | None = None
    ae_ckpt: dict = {}
    if ae_loss_weight != 0.0:
        ae_path = resolve_path(args.ae_checkpoint) if args.ae_checkpoint else latest_simple_ae_checkpoint()
        ae, mean, std, ae_ckpt = load_simple_ae(ae_path, device)
    else:
        ae = None
        mean = torch.empty(0, dtype=torch.float32, device=device)
        std = torch.empty(0, dtype=torch.float32, device=device)
    cfg = make_cfg(device, ae_ckpt)
    cfg.ae_loss_weight = ae_loss_weight
    tl.set_seed(int(cfg.seed))
    rl_cfg = RLLossConfig(
        pelvis_root_horizontal_weight=float(args.pelvis_root_horizontal_loss_weight),
        pelvis_root_horizontal_limit_m=float(args.pelvis_root_horizontal_limit_m),
        pelvis_root_rotation_weight=float(args.pelvis_root_rotation_loss_weight),
        pelvis_root_rotation_limit_deg=float(args.pelvis_root_rotation_limit_deg),
        end_effector_location_weight=float(args.end_effector_location_loss_weight),
        end_effector_location_limit_m=tuple(float(v) for v in args.end_effector_location_limit_m),
        end_effector_rotation_weight=float(args.end_effector_rotation_loss_weight),
        end_effector_rotation_limit_deg=tuple(float(v) for v in args.end_effector_rotation_limit_deg),
        end_effector_velocity_weight=float(args.end_effector_velocity_loss_weight),
        end_effector_velocity_limit_mps=float(args.end_effector_velocity_limit_mps),
        end_effector_angular_velocity_weight=float(args.end_effector_angular_velocity_loss_weight),
        end_effector_angular_velocity_limit_deg_s=float(args.end_effector_angular_velocity_limit_deg_s),
        pelvis_velocity_weight=float(args.pelvis_velocity_loss_weight),
        pelvis_velocity_limit_mps=float(args.pelvis_velocity_limit_mps),
        pelvis_angular_velocity_weight=float(args.pelvis_angular_velocity_loss_weight),
        pelvis_angular_velocity_limit_deg_s=float(args.pelvis_angular_velocity_limit_deg_s),
        core_angular_velocity_weight=float(args.core_angular_velocity_loss_weight),
        core_angular_velocity_limit_deg_s=float(args.core_angular_velocity_limit_deg_s),
        foot_pin_weight=float(args.foot_pin_loss_weight),
        foot_pin_height_threshold_m=float(args.foot_pin_height_threshold_m),
        foot_pin_height_temp_m=float(args.foot_pin_height_temp_m),
        foot_pin_speed_floor_mps=float(args.foot_pin_speed_floor_mps),
        no_hover_weight=float(args.no_hover_loss_weight),
        no_hover_height_threshold_m=float(args.no_hover_height_threshold_m),
        no_hover_height_temp_m=float(args.no_hover_height_temp_m),
        foot_floor_weight=float(args.foot_floor_loss_weight),
        foot_floor_y_m=float(args.foot_floor_y_m),
        foot_ceiling_weight=float(args.foot_ceiling_loss_weight),
        foot_ceiling_y_m=float(args.foot_ceiling_y_m),
        pelvis_height_weight=float(args.pelvis_height_loss_weight),
        pelvis_height_target_m=float(args.pelvis_height_target_m),
        pelvis_height_tolerance_m=float(args.pelvis_height_tolerance_m),
        fps=float(cfg.fps),
    )
    identity_enabled = (
        identity_loss_weight != 0.0
        or identity_maxabs_weight != 0.0
        or identity_world_pos_weight != 0.0
        or identity_pelvis_pos_weight != 0.0
    )
    specs = resolve_clip_specs(args.npz, args.periodic_folder, args.nonperiodic_folder)
    clips = load_clips(specs, cfg)
    input_dim, output_dim = tl.make_batch_dims(clips[0], cfg)
    if ae is not None:
        expected_dim = int(ae_ckpt["schema"]["total_dim"])
        if input_dim + output_dim != expected_dim:
            raise ValueError(f"AE dim {expected_dim} does not match controller feature dim {input_dim + output_dim}")

    store = SimpleClipStore(clips, cfg, device)
    envelope = None
    linear_slide_weight = float(args.linear_slide_loss_weight)
    angular_slide_weight = float(args.angular_slide_loss_weight)
    if envelope_loss_enabled(linear_slide_weight, angular_slide_weight):
        envelope = env.load_or_build_excess_envelope(
            store,
            env.ExcessEnvelopeConfig(margin=float(args.envelope_margin), knn=int(args.envelope_knn)),
        )
        sanity = env.groundtruth_sanity(store, envelope)
        if max(float(value) for value in sanity.values()) > 1e-6:
            raise RuntimeError(f"GT exceeds foot-slide envelope unexpectedly: {sanity}")
    init_checkpoint_path = resolve_path(args.init_checkpoint) if args.init_checkpoint else None
    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    output_initialization = "default"
    init_ckpt: dict | None = None
    if init_checkpoint_path is not None:
        init_ckpt = load_controller_init_checkpoint(model, init_checkpoint_path)
        output_initialization = f"loaded_controller_checkpoint:{init_checkpoint_path}"
    elif ae is not None and ae_loss_weight != 0.0 and not tl.output_prediction_uses_residual():
        output_initialization = initialize_controller_output_from_ae_mean(model, mean, input_dim, output_dim)
    elif tl.output_prediction_uses_residual():
        output_initialization = "residual_zero_weight_zero_bias"
    optimizer = make_adamw(model.parameters(), LEARNING_RATE, device, capturable=bool(device.type == "cuda"))

    stage_cache: dict[int, dict[str, object]] = {}
    for stage_k in ROLLOUT_SCHEDULE:
        rollout_values = rollout_values_for(stage_k) if mixed_rollout_enabled(stage_k) else (int(stage_k),)
        start_pools = build_training_start_pools(store, rollout_values)
        max_pool = start_pools[int(stage_k)]
        batch_size = batch_size_for_stage(store, int(stage_k), max_pool.row_count)
        stage_cache[int(stage_k)] = {
            "rollout_values": rollout_values,
            "start_pools": start_pools,
            "max_pool": max_pool,
            "row_count": max_pool.row_count,
            "batch_size": batch_size,
            "rollout_stats": rollout_stat_summary(batch_size, int(stage_k)),
            "pool_rows": {str(int(k)): int(pool.row_count) for k, pool in start_pools.items()},
        }

    run_id = ik_run_id(args.run_label)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    final_cache = stage_cache[int(ROLLOUT_K)]
    metadata = {
        "npz_paths": [str(path) for path, _cyclic in specs],
        "npz_folders": [{"path": str(path.parent), "cyclic": bool(cyclic)} for path, cyclic in specs],
        "simple_ae_checkpoint": str(ae_path) if ae_path is not None else None,
        "init_checkpoint": str(init_checkpoint_path) if init_checkpoint_path is not None else None,
        "tensorboard_logdir": str(run_dir / "tb"),
        "policy": {
            "loss": (
                "identity_statue"
                if ae_loss_weight == 0.0 and identity_enabled and not rl_cfg.enabled
                else (
                    "rl_only"
                    if (ae_loss_weight == 0.0 and rl_cfg.enabled)
                    else (
                        "simple_ae_output_reconstruction_with_optional_envelope"
                        if envelope is not None
                        else "simple_ae_output_reconstruction"
                    )
                )
            ),
            "ae_loss_weight": float(ae_loss_weight),
            "identity_output_loss_weight": float(identity_loss_weight),
            "identity_output_maxabs_loss_weight": float(identity_maxabs_weight),
            "identity_world_pos_loss_weight": float(identity_world_pos_weight),
            "identity_pelvis_pos_loss_weight": float(identity_pelvis_pos_weight),
            "ae_score_output_only": bool(AE_SCORE_OUTPUT_ONLY),
            "ae_scores_raw_output": True,
            "rl_loss_enabled": bool(rl_cfg.enabled),
            "rl_loss": rl_cfg.to_dict(),
            "envelope_loss_enabled": bool(envelope is not None),
            "envelope_loss": {
                "linear_slide_loss_weight": float(linear_slide_weight),
                "angular_slide_loss_weight": float(angular_slide_weight),
                "margin": float(args.envelope_margin),
                "knn": int(args.envelope_knn),
                "metadata": envelope["metadata"] if envelope is not None else None,
            },
            "rl_grad_clip_norm": float(RL_GRAD_CLIP_NORM) if rl_cfg.enabled else 0.0,
            "zero_loss_stop_threshold": float(ZERO_LOSS_STOP_THRESHOLD),
            "predict_residual": bool(cfg.predict_residual),
            "zero_init_output": bool(cfg.zero_init_output),
            "output_initialization": output_initialization,
            "output_reference_root": tl.OUTPUT_REFERENCE_ROOT,
            "output_prediction_mode": tl.normalized_output_prediction_mode(),
            "state_reference_root": tl.STATE_REFERENCE_ROOT,
            "ik_schema_version": tl.IK_SCHEMA_VERSION,
            "ik_pole_reference": tl.IK_POLE_REFERENCE,
            "ik_leg_pole_alpha_deg": math.degrees(tl.IK_LEG_POLE_ALPHA),
            "ik_arm_pole_alpha_deg": math.degrees(tl.IK_ARM_POLE_ALPHA),
            "mixed_rollout_at_max": bool(MIXED_ROLLOUT_AT_MAX),
            "pose_representation": tl.IK_POSE_REPRESENTATION,
            "test_set": False,
            "checkpoint_selection": "latest_stage_last",
            "init_checkpoint_epoch": int(init_ckpt.get("epoch", 0)) if init_ckpt is not None else 0,
            "init_checkpoint_rollout_k": int(init_ckpt.get("rollout_k", 0)) if init_ckpt is not None else 0,
        },
        "rollout_schedule": [int(k) for k in ROLLOUT_SCHEDULE],
        "rollout_stage_steps": [int(n) for n in ROLLOUT_STAGE_STEPS],
        "rollout_k": int(ROLLOUT_K),
        "rollout_mode": "mixed_geometric_at_max" if MIXED_ROLLOUT_AT_MAX else "fixed_per_stage",
        "log_every": int(LOG_EVERY),
        "row_count": int(final_cache["row_count"]),
        "batch_size": int(final_cache["batch_size"]),
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
        "start_pool_rows": {str(k): v["pool_rows"] for k, v in stage_cache.items()},
        "stage_learning_rates": {str(k): float(stage_learning_rate(k)) for k in ROLLOUT_SCHEDULE},
    }
    config_payload = {"config": asdict(cfg), "metadata": metadata}
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    writer = SummaryWriter(log_dir=str(run_dir / "tb"), flush_secs=1)
    writer.add_text("config/json", f"```json\n{json.dumps(config_payload, indent=2)}\n```", 0)
    writer.flush()
    refresh_tensorboard_async()

    print(
        f"simple_ae_controller run={run_id} "
        f"ae={ae_path if ae_path is not None else 'disabled'} tensorboard_logdir={run_dir / 'tb'}",
        flush=True,
    )
    last_loss = float("inf")
    best_loss = float("inf")
    init_path = save_controller_checkpoint(run_dir, run_id, "init", model, optimizer, 0, last_loss, 0, cfg, metadata)
    print(f"saved initial checkpoint {init_path}", flush=True)

    start = time.perf_counter()
    total_steps = sum(int(x) for x in ROLLOUT_STAGE_STEPS)
    step = 0
    stop_training = False
    last_stage_k = int(ROLLOUT_SCHEDULE[0]) if ROLLOUT_SCHEDULE else int(ROLLOUT_K)
    for stage_idx, stage_k in enumerate(ROLLOUT_SCHEDULE):
        last_stage_k = int(stage_k)
        stage_steps = int(ROLLOUT_STAGE_STEPS[stage_idx])
        lr = stage_learning_rate(int(stage_k))
        if stage_idx > 0 and bool(cfg.lr_reset_on_rollout_advance):
            optimizer = make_adamw(model.parameters(), lr, device, capturable=bool(device.type == "cuda"))
        else:
            set_optimizer_lr(optimizer, lr)
        cache = stage_cache[int(stage_k)]
        model.train()
        stepper = make_pure_ae_stepper(
            model,
            optimizer,
            ae,
            mean,
            std,
            store,
            int(stage_k),
            int(cache["batch_size"]),
            cache["start_pools"],
            rl_cfg,
            ae_loss_weight,
            envelope,
            linear_slide_weight,
            angular_slide_weight,
            identity_loss_weight,
            identity_maxabs_weight,
            identity_world_pos_weight,
            identity_pelvis_pos_weight,
        )
        print(
            f"stage={stage_idx + 1}/{len(ROLLOUT_SCHEDULE)} K={stage_k} "
            f"steps={stage_steps} batch={int(cache['batch_size'])} lr={lr:.3g} stepper={stepper.kind}",
            flush=True,
        )
        for stage_step in range(1, stage_steps + 1):
            step += 1
            loss_result = stepper.step()
            last_loss = float(loss_result.total.detach().cpu())
            loss_terms = {key: float(value.detach().cpu()) for key, value in loss_result.terms.items()}
            zero_loss_stop = math.isfinite(last_loss) and last_loss <= float(ZERO_LOSS_STOP_THRESHOLD)
            if stage_step == 1 or stage_step % LOG_EVERY == 0 or stage_step == stage_steps or zero_loss_stop:
                model.eval()
                mean_err = NAN_METRIC
                max_err = NAN_METRIC
                if RUN_FK_DIAGNOSTIC:
                    mean_err, max_err = rollout_joint_error(model, store, int(stage_k), cache["max_pool"])
                latest = save_controller_checkpoint(
                    run_dir, run_id, "latest", model, optimizer, step, last_loss, int(stage_k), cfg, metadata
                )
                if last_loss < best_loss:
                    best_loss = last_loss
                    save_controller_checkpoint(
                        run_dir, run_id, "best", model, optimizer, step, best_loss, int(stage_k), cfg, metadata
                    )
                elapsed = time.perf_counter() - start
                stats = cache["rollout_stats"]
                gt_text = (
                    f"gt_mean_m={mean_err:.6f} gt_max_m={max_err:.6f}"
                    if RUN_FK_DIAGNOSTIC
                    else "gt_diag=off"
                )
                ae_text = "disabled" if ae_loss_weight == 0.0 else f"{loss_terms.get('ae_score', NAN_METRIC):.6g}"
                identity_text = ""
                if identity_enabled:
                    identity_parts = []
                    if identity_loss_weight != 0.0:
                        identity_parts.append(f"identity_output={loss_terms.get(IDENTITY_TERM_NAME, NAN_METRIC):.6g}")
                    if identity_maxabs_weight != 0.0:
                        identity_parts.append(
                            f"identity_output_maxabs={loss_terms.get(IDENTITY_MAXABS_TERM_NAME, NAN_METRIC):.6g}"
                        )
                    if identity_world_pos_weight != 0.0:
                        identity_parts.append(
                            f"identity_world_pos={loss_terms.get(IDENTITY_WORLD_POS_TERM_NAME, NAN_METRIC):.6g}"
                        )
                    if identity_pelvis_pos_weight != 0.0:
                        identity_parts.append(
                            f"identity_pelvis_pos={loss_terms.get(IDENTITY_PELVIS_POS_TERM_NAME, NAN_METRIC):.6g}"
                        )
                    identity_text = " " + " ".join(identity_parts)
                rl_text = " ".join(f"{name}={loss_terms.get(name, NAN_METRIC):.6g}" for name in rl_cfg.enabled_terms())
                slide_text = ""
                if envelope is not None:
                    slide_text = (
                        f"linear_slide_weighted={loss_terms.get('linear_slide_weighted', NAN_METRIC):.6g} "
                        f"angular_slide_weighted={loss_terms.get('angular_slide_weighted', NAN_METRIC):.6g} "
                    )
                print(
                    f"step={step:05d} K={stage_k} train_loss={last_loss:.6g} "
                    f"ae={ae_text}{identity_text} "
                    f"{rl_text} "
                    f"{slide_text}"
                    f"{gt_text} effK_mean={stats['effective_k_mean']:.2f} lr={lr:.3g} elapsed_s={elapsed:.1f}",
                    flush=True,
                )
                writer.add_scalar("loss/train_total", last_loss, step)
                if ae_loss_weight != 0.0:
                    writer.add_scalar("loss/ae_score", loss_terms.get("ae_score", NAN_METRIC), step)
                if identity_loss_weight != 0.0:
                    writer.add_scalar("loss/identity_output", loss_terms.get(IDENTITY_TERM_NAME, NAN_METRIC), step)
                if identity_maxabs_weight != 0.0:
                    writer.add_scalar(
                        "loss/identity_output_maxabs", loss_terms.get(IDENTITY_MAXABS_TERM_NAME, NAN_METRIC), step
                    )
                if identity_world_pos_weight != 0.0:
                    writer.add_scalar(
                        "loss/identity_world_pos", loss_terms.get(IDENTITY_WORLD_POS_TERM_NAME, NAN_METRIC), step
                    )
                if identity_pelvis_pos_weight != 0.0:
                    writer.add_scalar(
                        "loss/identity_pelvis_pos", loss_terms.get(IDENTITY_PELVIS_POS_TERM_NAME, NAN_METRIC), step
                    )
                for name in rl_cfg.enabled_terms():
                    writer.add_scalar(f"loss/{name}", loss_terms.get(name, NAN_METRIC), step)
                if envelope is not None:
                    writer.add_scalar("loss/linear_slide_weighted", loss_terms.get("linear_slide_weighted", NAN_METRIC), step)
                    writer.add_scalar("loss/angular_slide_weighted", loss_terms.get("angular_slide_weighted", NAN_METRIC), step)
                if RUN_FK_DIAGNOSTIC:
                    writer.add_scalar("eval/rollout_mean_m", mean_err, step)
                    writer.add_scalar("eval/rollout_max_m", max_err, step)
                writer.add_scalar("curriculum/rollout_k", int(stage_k), step)
                writer.add_scalar("curriculum/effective_rollout_k_mean", stats["effective_k_mean"], step)
                writer.add_scalar("curriculum/effective_rollout_k_max", stats["effective_k_max"], step)
                writer.add_scalar("time/elapsed_s", elapsed, step)
                writer.flush()
                print(f"checkpoint_latest={latest}", flush=True)
                model.train()
            if zero_loss_stop:
                stop_training = True
                wake_on_zero_loss(run_id, step, last_loss)
                break
        stage_path = save_controller_checkpoint(
            run_dir, run_id, f"stage_K{int(stage_k)}", model, optimizer, step, last_loss, int(stage_k), cfg, metadata
        )
        print(f"saved stage checkpoint {stage_path}", flush=True)
        del stepper
        if stop_training:
            break

    last = save_controller_checkpoint(run_dir, run_id, "last", model, optimizer, step, last_loss, last_stage_k, cfg, metadata)
    writer.close()
    print(f"saved {last}", flush=True)


if __name__ == "__main__":
    main()
