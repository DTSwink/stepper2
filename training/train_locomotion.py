from __future__ import annotations

# Put your NPZ folder path here. Relative paths are resolved from the stepper
# project root, not from the current shell directory.
folder_path = "data/npz_final"

import argparse
import math
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    fps: int = 30
    # Cascadeur/Unreal FBX data is usually centimeters. Training in meters keeps
    # MAX_SPEED_SCALE=5.0 interpretable as 5 m/s. Set to 1.0 for raw FBX units.
    position_unit_scale: float = 0.01
    max_speed_scale: float = 5.0
    max_turn_rate_per_sec_scale: float = math.radians(720.0)
    pose_delta_scale: float = 2.0
    future_window_seconds: float = 0.25

    hidden_dim: int = 512
    num_hidden_layers: int = 2
    activation: str = "GELU"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 64
    max_epochs: int = 2000
    val_fraction: float = 0.1
    seed: int = 1234
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    allow_tf32: bool = True
    use_torch_compile: bool = True
    torch_compile_mode: str = "default"
    show_progress: bool = False
    save_last_every_epochs: int = 5
    save_best_every_epochs: int = 0
    writer_flush_every_epochs: int = 5
    predict_residual: bool = True
    zero_init_output: bool = True
    target_loss_reduction: float = 0.98
    stop_at_target_loss_reduction: bool = False
    max_train_seconds: float = 0.0
    profile_timing: bool = False
    profile_sync_cuda: bool = False
    disable_validation: bool = False

    rollout_schedule: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    curriculum_threshold: float = 1e-3
    curriculum_min_epochs: int = 0
    curriculum_max_epochs_per_stage: int = 0
    curriculum_patience_epochs: int = 5
    curriculum_stall_patience_epochs: int = 0
    curriculum_min_delta: float = 1e-5
    stop_on_final_stall: bool = False

    alpha0_pelvis_location: float = 1.0
    alpha1_pelvis_rotation: float = 1.0
    alpha2_pose_rotation: float = 1.0
    alpha3_pose_6d_aux: float = 0.1
    alpha4_end_effector_location: float = 10.0
    alpha5_end_effector_rotation: float = 0.5
    alpha6_full_body_location: float = 1.0

    end_effector_bones: tuple[str, ...] = ("foot_l", "ball_l", "foot_r", "ball_r")
    exclude_bone_prefixes: tuple[str, ...] = ("ik_", "weapon_")
    exclude_bone_names: tuple[str, ...] = ("root", "attach")

    checkpoint_every_epochs: int = 500
    run_name: str = "locomotion_mlp"
    output_dir: str = "training/runs"

    @property
    def max_speed_scale_final(self) -> float:
        return self.max_speed_scale / self.fps

    @property
    def max_turn_rate_scale_final(self) -> float:
        return self.max_turn_rate_per_sec_scale / self.fps

    @property
    def pose_delta_scale_final(self) -> float:
        return self.pose_delta_scale / self.fps

    @property
    def future_window(self) -> int:
        return max(1, int(round(self.future_window_seconds * self.fps)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)


def rotmat_to_6d(rot: torch.Tensor) -> torch.Tensor:
    # Row-vector convention: store the first two basis rows.
    return rot[..., :2, :].reshape(*rot.shape[:-2], 6)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    # Row-vector Gram-Schmidt. This matches rotmat_to_6d above.
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = normalize(a1)
    b2 = normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def clean_6d(d6: torch.Tensor) -> torch.Tensor:
    return rotmat_to_6d(rotation_6d_to_matrix(d6))


def geodesic_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Row-vector relative rotation: R_delta = pred * target^-1 = pred * target^T.
    delta = pred @ target.transpose(-1, -2)
    trace = delta.diagonal(dim1=-1, dim2=-2).sum(dim=-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos).mean()


def yaw_to_row_matrix(yaw: torch.Tensor) -> torch.Tensor:
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    row0 = torch.stack((c, z, s), dim=-1)
    row1 = torch.stack((z, o, z), dim=-1)
    row2 = torch.stack((-s, z, c), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def heading_yaw_from_root(root_rot: torch.Tensor) -> torch.Tensor:
    # Local +Z row as forward, projected onto the XZ ground plane.
    forward = root_rot[..., 2, :]
    return torch.atan2(forward[..., 0], forward[..., 2])


def should_keep_bone(name: str, cfg: TrainConfig) -> bool:
    if name in cfg.exclude_bone_names:
        return False
    return not any(name.startswith(prefix) for prefix in cfg.exclude_bone_prefixes)


def axis_up_axis(arrays: np.lib.npyio.NpzFile) -> int:
    if "axis_up_axis" not in arrays.files:
        return 2
    return int(arrays["axis_up_axis"])


def canonicalize_positions(pos: torch.Tensor, up_axis: int) -> torch.Tensor:
    if up_axis == 3:
        return pos[..., [1, 2, 0]]
    return pos


def canonicalize_rotations(rot: torch.Tensor, up_axis: int) -> torch.Tensor:
    if up_axis != 3:
        return rot
    # Row-vector convention, p_c = p_s P, R_c = P^-1 R_s P.
    perm = torch.tensor([1, 2, 0], dtype=torch.long, device=rot.device)
    inv = torch.tensor([2, 0, 1], dtype=torch.long, device=rot.device)
    return rot.index_select(-2, inv).index_select(-1, perm)


class MotionClip:
    def __init__(self, path: Path, cfg: TrainConfig):
        arrays = np.load(path)
        self.path = path
        self.source_up_axis = axis_up_axis(arrays)
        self.bone_names = [str(x) for x in arrays["bone_names"]]
        self.parents_full = arrays["parents"].astype(np.int64)
        self.fps = float(arrays["fps"])
        if abs(self.fps - cfg.fps) > 1e-3:
            raise ValueError(f"{path} is {self.fps} FPS, config expects {cfg.fps} FPS")

        self.keep_full = [i for i, n in enumerate(self.bone_names) if should_keep_bone(n, cfg)]
        self.body_names = [self.bone_names[i] for i in self.keep_full]
        if "pelvis" not in self.body_names:
            raise ValueError(f"{path} does not contain a kept pelvis bone")
        self.pelvis = self.body_names.index("pelvis")
        self.non_pelvis = [i for i, n in enumerate(self.body_names) if n != "pelvis"]
        self.end_effectors = []
        for name in cfg.end_effector_bones:
            if name not in self.body_names:
                raise ValueError(f"{path} is missing end effector bone {name!r}")
            self.end_effectors.append(self.body_names.index(name))
        self.end_effectors_tensor = torch.tensor(self.end_effectors, dtype=torch.long)

        full_to_body = {full_i: body_i for body_i, full_i in enumerate(self.keep_full)}
        parents_body = []
        for full_i in self.keep_full:
            parent = int(self.parents_full[full_i])
            parents_body.append(full_to_body.get(parent, -1))
        self.parents_body = torch.tensor(parents_body, dtype=torch.long)

        global_pos_full = canonicalize_positions(
            torch.tensor(arrays["global_joint_pos"], dtype=torch.float32) * cfg.position_unit_scale,
            self.source_up_axis,
        )
        global_rot_full = canonicalize_rotations(
            torch.tensor(arrays["global_matrix"][:, :, :3, :3], dtype=torch.float32),
            self.source_up_axis,
        )
        local_rot_full = canonicalize_rotations(
            torch.tensor(arrays["local_matrix"][:, :, :3, :3], dtype=torch.float32),
            self.source_up_axis,
        )
        lcl_translation_full = canonicalize_positions(
            torch.tensor(arrays["fbx_lcl_translation"], dtype=torch.float32) * cfg.position_unit_scale,
            self.source_up_axis,
        )
        default_lcl_translation_full = (
            torch.tensor(arrays["default_lcl_translation"], dtype=torch.float32) * cfg.position_unit_scale
        )
        default_lcl_translation_full = canonicalize_positions(default_lcl_translation_full, self.source_up_axis)

        root_index = self.bone_names.index("root")
        self.root_pos = global_pos_full[:, root_index]
        self.root_rot = global_rot_full[:, root_index]
        self.root_yaw = heading_yaw_from_root(self.root_rot)
        self.root_heading_rot = yaw_to_row_matrix(self.root_yaw)

        keep = torch.tensor(self.keep_full, dtype=torch.long)
        self.global_pos = global_pos_full.index_select(1, keep)
        self.global_rot = global_rot_full.index_select(1, keep)
        self.local_rot = local_rot_full.index_select(1, keep)
        self.local_rot6 = rotmat_to_6d(self.local_rot)
        self.local_offsets = default_lcl_translation_full.index_select(0, keep)
        self.pelvis_local_pos = lcl_translation_full[:, self.keep_full[self.pelvis]]
        self.pelvis_rot6 = self.local_rot6[:, self.pelvis]
        self.non_pelvis_rot6 = self.local_rot6[:, self.non_pelvis]

        root_delta = self.global_pos - self.root_pos[:, None, :]
        self.canonical_pos = torch.einsum("tjc,tdc->tjd", root_delta, self.root_heading_rot)

        self.T = int(global_pos_full.shape[0])
        self.J = len(self.body_names)
        self.Jn = len(self.non_pelvis)
        self.nonpelvis_map = {bone_index: i for i, bone_index in enumerate(self.non_pelvis)}
        self.parents_body_list = [int(parent) for parent in self.parents_body.tolist()]
        self._device_cache: dict[str, dict[str, torch.Tensor]] = {}

    def pose_at(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "pelvis_pos": self.pelvis_local_pos[idx],
            "pelvis_rot6": self.pelvis_rot6[idx],
            "nonpelvis_rot6": self.non_pelvis_rot6[idx],
            "canon_pos": self.canonical_pos[idx],
        }

    def tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        key = str(device)
        cached = self._device_cache.get(key)
        if cached is None:
            cached = {
                "root_pos": self.root_pos.to(device),
                "root_rot": self.root_rot.to(device),
                "root_yaw": self.root_yaw.to(device),
                "root_heading_rot": self.root_heading_rot.to(device),
                "global_pos": self.global_pos.to(device),
                "global_rot": self.global_rot.to(device),
                "local_offsets": self.local_offsets.to(device),
                "pelvis_local_pos": self.pelvis_local_pos.to(device),
                "pelvis_rot6": self.pelvis_rot6.to(device),
                "non_pelvis_rot6": self.non_pelvis_rot6.to(device),
                "canonical_pos": self.canonical_pos.to(device),
                "end_effectors": self.end_effectors_tensor.to(device),
            }
            self._device_cache[key] = cached
        return cached


class MotionIndexDataset(Dataset):
    def __init__(self, clips: list[MotionClip], cfg: TrainConfig, split: str, max_rollout: int):
        self.items: list[tuple[int, int]] = []
        for ci, clip in enumerate(clips):
            max_start = clip.T - max_rollout - 1
            if max_start < 1:
                continue
            starts = list(range(1, max_start + 1))
            random.Random(cfg.seed + ci).shuffle(starts)
            if cfg.val_fraction <= 0.0:
                chosen = starts
            else:
                val_count = max(1, int(round(len(starts) * cfg.val_fraction))) if len(starts) > 1 else 0
                chosen = starts[:val_count] if split == "val" else starts[val_count:]
            self.items.extend((ci, s) for s in chosen)
        if not self.items:
            raise ValueError(
                f"No {split} samples. Need longer clips or smaller future_window_seconds/max rollout."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[int, int]:
        return self.items[index]


class MLPController(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, cfg: TrainConfig):
        super().__init__()
        act_cls = getattr(nn, cfg.activation)
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(cfg.num_hidden_layers):
            layers.append(nn.Linear(in_dim, cfg.hidden_dim))
            layers.append(nn.LayerNorm(cfg.hidden_dim))
            layers.append(act_cls())
            in_dim = cfg.hidden_dim
        output = nn.Linear(in_dim, output_dim)
        if cfg.zero_init_output:
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def body_pose_vector(pose: dict[str, torch.Tensor]) -> torch.Tensor:
    b = pose["pelvis_pos"].shape[0]
    return torch.cat(
        (
            pose["pelvis_pos"],
            pose["pelvis_rot6"],
            pose["canon_pos"].reshape(b, -1),
            pose["nonpelvis_rot6"].reshape(b, -1),
        ),
        dim=-1,
    )


def output_to_pose(raw: torch.Tensor, clip: MotionClip) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    b = raw.shape[0]
    cursor = 0
    pelvis_pos = raw[:, cursor : cursor + 3]
    cursor += 3
    pelvis_rot6_raw = raw[:, cursor : cursor + 6]
    cursor += 6
    nonpelvis_rot6_raw = raw[:, cursor:].reshape(b, clip.Jn, 6)

    pelvis_rot6 = clean_6d(pelvis_rot6_raw)
    nonpelvis_rot6 = clean_6d(nonpelvis_rot6_raw.reshape(-1, 6)).reshape(b, clip.Jn, 6)
    clean_pose = {
        "pelvis_pos": pelvis_pos,
        "pelvis_rot6": pelvis_rot6,
        "nonpelvis_rot6": nonpelvis_rot6,
    }
    raw_pose = {
        "pelvis_rot6": pelvis_rot6_raw,
        "nonpelvis_rot6": nonpelvis_rot6_raw,
    }
    return clean_pose, raw_pose


def pose_target_output(pose: dict[str, torch.Tensor]) -> torch.Tensor:
    b = pose["pelvis_pos"].shape[0]
    return torch.cat(
        (
            pose["pelvis_pos"],
            pose["pelvis_rot6"],
            pose["nonpelvis_rot6"].reshape(b, -1),
        ),
        dim=-1,
    )


def predict_next_raw(
    model: nn.Module,
    inp: torch.Tensor,
    cur_pose: dict[str, torch.Tensor],
    cfg: TrainConfig,
) -> torch.Tensor:
    raw = model(inp)
    if cfg.predict_residual:
        raw = pose_target_output(cur_pose) + raw
    return raw


def fk_from_pose(
    clip: MotionClip,
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    pose: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b = root_pos.shape[0]
    pelvis_rot = rotation_6d_to_matrix(pose["pelvis_rot6"])
    nonpelvis_rot = rotation_6d_to_matrix(pose["nonpelvis_rot6"])
    tensors = clip.tensors(device)

    offsets = tensors["local_offsets"].unsqueeze(0).expand(b, -1, -1).clone()
    offsets[:, clip.pelvis] = pose["pelvis_pos"]

    global_pos_list: list[torch.Tensor] = []
    global_rot_list: list[torch.Tensor] = []
    for j in range(clip.J):
        local_rot_j = pelvis_rot if j == clip.pelvis else nonpelvis_rot[:, clip.nonpelvis_map[j]]
        parent = clip.parents_body_list[j]
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
    root_yaw = heading_yaw_from_root(root_rot)
    heading = yaw_to_row_matrix(root_yaw)
    canon = torch.einsum("bjc,bdc->bjd", global_pos - root_pos[:, None, :], heading)
    return global_pos, global_rot, canon


def root_delta_feature(clip: MotionClip, prev_idx: torch.Tensor, cur_idx: torch.Tensor, cfg: TrainConfig, device) -> torch.Tensor:
    tensors = clip.tensors(device)
    prev_idx = prev_idx.to(device)
    cur_idx = cur_idx.to(device)
    prev_pos = tensors["root_pos"].index_select(0, prev_idx)
    cur_pos = tensors["root_pos"].index_select(0, cur_idx)
    prev_heading = tensors["root_heading_rot"].index_select(0, prev_idx)
    delta_local = torch.matmul((cur_pos - prev_pos).unsqueeze(1), prev_heading.transpose(-1, -2)).squeeze(1)
    dx = delta_local[:, 0] / cfg.max_speed_scale_final
    dz = delta_local[:, 2] / cfg.max_speed_scale_final
    yaw_delta = wrap_angle(tensors["root_yaw"].index_select(0, cur_idx) - tensors["root_yaw"].index_select(0, prev_idx))
    dyaw = yaw_delta / cfg.max_turn_rate_scale_final
    return torch.stack((dx, dz, dyaw), dim=-1)


def future_root_features(clip: MotionClip, cur_idx: torch.Tensor, cfg: TrainConfig, device) -> torch.Tensor:
    feats = []
    tensors = clip.tensors(device)
    cur_idx = cur_idx.to(device)
    cur_pos = tensors["root_pos"].index_select(0, cur_idx)
    cur_heading = tensors["root_heading_rot"].index_select(0, cur_idx)
    cur_yaw = tensors["root_yaw"].index_select(0, cur_idx)
    for k in range(1, cfg.future_window + 1):
        fut_idx = torch.clamp(cur_idx + k, max=clip.T - 1)
        fut_pos = tensors["root_pos"].index_select(0, fut_idx)
        fut_local = torch.matmul((fut_pos - cur_pos).unsqueeze(1), cur_heading.transpose(-1, -2)).squeeze(1)
        scale_k = k * cfg.max_speed_scale_final
        dx = torch.clamp(fut_local[:, 0] / scale_k, -2.0, 2.0)
        dz = torch.clamp(fut_local[:, 2] / scale_k, -2.0, 2.0)
        dyaw = wrap_angle(tensors["root_yaw"].index_select(0, fut_idx) - cur_yaw)
        feats.append(torch.stack((dx, dz, torch.cos(dyaw), torch.sin(dyaw)), dim=-1))
    return torch.cat(feats, dim=-1)


def build_input(
    clip: MotionClip,
    prev_idx: torch.Tensor,
    cur_idx: torch.Tensor,
    prev_pose: dict[str, torch.Tensor],
    cur_pose: dict[str, torch.Tensor],
    cfg: TrainConfig,
    device: torch.device,
) -> torch.Tensor:
    current = body_pose_vector(cur_pose)
    previous = body_pose_vector(prev_pose)
    pelvis_vel = (cur_pose["pelvis_pos"] - prev_pose["pelvis_pos"]) / cfg.pose_delta_scale_final
    joint_vel = (cur_pose["canon_pos"] - prev_pose["canon_pos"]).reshape(cur_idx.shape[0], -1) / cfg.pose_delta_scale_final
    root_feat = root_delta_feature(clip, prev_idx, cur_idx, cfg, device)
    future_feat = future_root_features(clip, cur_idx, cfg, device)
    return torch.cat((current, previous, pelvis_vel, joint_vel, root_feat, future_feat), dim=-1)


def get_pose_from_clip(clip: MotionClip, idx: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    tensors = clip.tensors(device)
    idx = idx.to(device)
    return {
        "pelvis_pos": tensors["pelvis_local_pos"].index_select(0, idx),
        "pelvis_rot6": tensors["pelvis_rot6"].index_select(0, idx),
        "nonpelvis_rot6": tensors["non_pelvis_rot6"].index_select(0, idx),
        "canon_pos": tensors["canonical_pos"].index_select(0, idx),
    }


def compute_losses(
    clip: MotionClip,
    pred_pose: dict[str, torch.Tensor],
    raw_pose: dict[str, torch.Tensor],
    target_pose: dict[str, torch.Tensor],
    target_idx: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    b = target_idx.shape[0]
    tensors = clip.tensors(device)
    target_idx_device = target_idx.to(device)
    root_pos = tensors["root_pos"].index_select(0, target_idx_device)
    root_rot = tensors["root_rot"].index_select(0, target_idx_device)
    pred_global_pos, pred_global_rot, pred_canon = fk_from_pose(clip, root_pos, root_rot, pred_pose, device)

    target_global_pos = tensors["global_pos"].index_select(0, target_idx_device)
    target_global_rot = tensors["global_rot"].index_select(0, target_idx_device)

    pelvis_loc = F.huber_loss(pred_pose["pelvis_pos"], target_pose["pelvis_pos"])
    pelvis_rot = geodesic_loss(
        rotation_6d_to_matrix(pred_pose["pelvis_rot6"]),
        rotation_6d_to_matrix(target_pose["pelvis_rot6"]),
    )
    pose_rot = geodesic_loss(
        rotation_6d_to_matrix(pred_pose["nonpelvis_rot6"].reshape(-1, 6)),
        rotation_6d_to_matrix(target_pose["nonpelvis_rot6"].reshape(-1, 6)),
    )
    pose_aux = F.mse_loss(pred_pose["nonpelvis_rot6"], target_pose["nonpelvis_rot6"]) + F.mse_loss(
        pred_pose["pelvis_rot6"], target_pose["pelvis_rot6"]
    )
    ee_idx = tensors["end_effectors"]
    ee_delta = pred_global_pos.index_select(1, ee_idx) - target_global_pos.index_select(1, ee_idx)
    ee_loc = (ee_delta.square().sum(dim=-1)).mean()
    ee_rot = geodesic_loss(
        pred_global_rot.index_select(1, ee_idx).reshape(-1, 3, 3),
        target_global_rot.index_select(1, ee_idx).reshape(-1, 3, 3),
    )
    full_body_loc = pred_global_pos.sub(target_global_pos).square().sum(dim=-1).mean()
    total = (
        cfg.alpha0_pelvis_location * pelvis_loc
        + cfg.alpha1_pelvis_rotation * pelvis_rot
        + cfg.alpha2_pose_rotation * pose_rot
        + cfg.alpha3_pose_6d_aux * pose_aux
        + cfg.alpha4_end_effector_location * ee_loc
        + cfg.alpha5_end_effector_rotation * ee_rot
        + cfg.alpha6_full_body_location * full_body_loc
    )
    losses = {
        "pelvis_location": pelvis_loc.detach(),
        "pelvis_rotation": pelvis_rot.detach(),
        "pose_rotation": pose_rot.detach(),
        "pose_6d_aux": pose_aux.detach(),
        "end_effector_location": ee_loc.detach(),
        "end_effector_rotation": ee_rot.detach(),
        "full_body_location": full_body_loc.detach(),
    }
    next_pose = {
        "pelvis_pos": pred_pose["pelvis_pos"],
        "pelvis_rot6": pred_pose["pelvis_rot6"],
        "nonpelvis_rot6": pred_pose["nonpelvis_rot6"],
        "canon_pos": pred_canon,
    }
    return total, losses, next_pose


def run_batch(
    model: nn.Module,
    clips: list[MotionClip],
    batch: list[torch.Tensor],
    cfg: TrainConfig,
    rollout_k: int,
    device: torch.device,
    train: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    clip_indices, starts = batch
    # Group by clip so variable skeleton metadata remains simple and explicit.
    total_loss = torch.zeros((), device=device)
    accum: dict[str, torch.Tensor] = {}
    groups = {}
    for row, ci in enumerate(clip_indices.tolist()):
        groups.setdefault(ci, []).append(row)

    group_count = 0
    for ci, rows in groups.items():
        clip = clips[ci]
        row_t = torch.tensor(rows, dtype=torch.long)
        start = starts[row_t].long()
        prev_idx = start - 1
        cur_idx = start
        prev_pose = get_pose_from_clip(clip, prev_idx, device)
        cur_pose = get_pose_from_clip(clip, cur_idx, device)

        group_loss = torch.zeros((), device=device)
        for step in range(rollout_k):
            inp = build_input(clip, prev_idx, cur_idx, prev_pose, cur_pose, cfg, device)
            raw_out = predict_next_raw(model, inp, cur_pose, cfg)
            pred_pose, raw_pose = output_to_pose(raw_out, clip)
            target_idx = cur_idx + 1
            target_pose = get_pose_from_clip(clip, target_idx, device)
            step_loss, parts, next_pose = compute_losses(
                clip, pred_pose, raw_pose, target_pose, target_idx, cfg, device
            )
            group_loss = group_loss + step_loss / rollout_k
            for key, value in parts.items():
                accum[key] = accum.get(key, torch.zeros((), device=device)) + value / rollout_k
            prev_pose = cur_pose
            cur_pose = next_pose
            prev_idx = cur_idx
            cur_idx = target_idx

        total_loss = total_loss + group_loss
        group_count += 1

    total_loss = total_loss / max(1, group_count)
    scalars = {k: float(v.detach().cpu() / max(1, group_count)) for k, v in accum.items()}
    scalars["total"] = float(total_loss.detach().cpu())
    return total_loss, scalars


def load_clips(folder: Path, cfg: TrainConfig) -> list[MotionClip]:
    paths = sorted(folder.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {folder}")
    clips = [MotionClip(path, cfg) for path in paths]
    first_names = clips[0].body_names
    for clip in clips[1:]:
        if clip.body_names != first_names:
            raise ValueError(f"Skeleton mismatch: {clip.path} does not match {clips[0].path}")
    return clips


def make_batch_dims(clip: MotionClip, cfg: TrainConfig) -> tuple[int, int]:
    pose_dim = 3 + 6 + clip.J * 3 + clip.Jn * 6
    velocity_dim = 3 + clip.J * 3
    input_dim = pose_dim * 2 + velocity_dim + 3 + cfg.future_window * 4
    output_dim = 3 + 6 + clip.Jn * 6
    return input_dim, output_dim


def unwrap_compiled_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def parse_rollout_schedule(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("rollout schedule cannot be empty")
    if any(value < 1 for value in values):
        raise ValueError("rollout schedule values must be >= 1")
    return values


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_val: float, rollout_k: int, cfg, metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, optimizer, epoch, best_val, rollout_k, cfg, metadata), path)


def checkpoint_payload(model, optimizer, epoch: int, best_val: float, rollout_k: int, cfg, metadata) -> dict:
    return {
        "epoch": epoch,
        "best_val": best_val,
        "rollout_k": rollout_k,
        "config": asdict(cfg),
        "metadata": metadata,
        "model": unwrap_compiled_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
    }


def clone_checkpoint_payload(payload: dict) -> dict:
    cloned = dict(payload)
    cloned["model"] = {k: v.detach().cpu().clone() for k, v in payload["model"].items()}
    cloned["optimizer"] = payload["optimizer"]
    return cloned


def save_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


class TimingProfiler:
    def __init__(self, enabled: bool, device: torch.device | None = None, sync_cuda: bool = False) -> None:
        self.enabled = enabled
        self.device = device
        self.sync_cuda = sync_cuda
        self.seconds: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def _sync(self) -> None:
        if (
            self.sync_cuda
            and self.device is not None
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize(self.device)

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.seconds[name] = self.seconds.get(name, 0.0) + (time.perf_counter() - start)
            self.counts[name] = self.counts.get(name, 0) + 1

    def write_csv(self, path: Path, total_seconds: float) -> None:
        if not self.enabled:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        known = sum(self.seconds.values())
        rows = [("section", "seconds", "percent", "count")]
        for key, seconds in sorted(self.seconds.items(), key=lambda item: item[1], reverse=True):
            percent = 0.0 if total_seconds <= 0.0 else seconds * 100.0 / total_seconds
            rows.append((key, f"{seconds:.6f}", f"{percent:.3f}", str(self.counts.get(key, 0))))
        overhead = max(0.0, total_seconds - known)
        rows.append(("unprofiled_overhead", f"{overhead:.6f}", f"{(overhead * 100.0 / total_seconds) if total_seconds > 0.0 else 0.0:.3f}", ""))
        rows.append(("total_wall", f"{total_seconds:.6f}", "100.000", ""))
        path.write_text("\n".join(",".join(row) for row in rows) + "\n", encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    process_start_time = time.perf_counter()
    cfg = TrainConfig()
    cfg.max_epochs = args.max_epochs if args.max_epochs is not None else cfg.max_epochs
    cfg.batch_size = args.batch_size if args.batch_size is not None else cfg.batch_size
    cfg.future_window_seconds = (
        args.future_window_seconds if args.future_window_seconds is not None else cfg.future_window_seconds
    )
    if args.device is not None:
        cfg.device = args.device
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.val_fraction is not None:
        cfg.val_fraction = args.val_fraction
    if args.predict_residual is not None:
        cfg.predict_residual = args.predict_residual
    if args.zero_init_output is not None:
        cfg.zero_init_output = args.zero_init_output
    if args.rollout_schedule is not None:
        cfg.rollout_schedule = parse_rollout_schedule(args.rollout_schedule)
    if args.curriculum_threshold is not None:
        cfg.curriculum_threshold = args.curriculum_threshold
    if args.curriculum_min_epochs is not None:
        cfg.curriculum_min_epochs = args.curriculum_min_epochs
    if args.curriculum_max_epochs_per_stage is not None:
        cfg.curriculum_max_epochs_per_stage = args.curriculum_max_epochs_per_stage
    if args.curriculum_patience_epochs is not None:
        cfg.curriculum_patience_epochs = args.curriculum_patience_epochs
    if args.curriculum_stall_patience_epochs is not None:
        cfg.curriculum_stall_patience_epochs = args.curriculum_stall_patience_epochs
    if args.curriculum_min_delta is not None:
        cfg.curriculum_min_delta = args.curriculum_min_delta
    cfg.stop_on_final_stall = not args.no_stop_on_final_stall
    if args.alpha6_full_body_location is not None:
        cfg.alpha6_full_body_location = args.alpha6_full_body_location
    if args.alpha4_end_effector_location is not None:
        cfg.alpha4_end_effector_location = args.alpha4_end_effector_location
    if args.alpha0_pelvis_location is not None:
        cfg.alpha0_pelvis_location = args.alpha0_pelvis_location
    if args.alpha1_pelvis_rotation is not None:
        cfg.alpha1_pelvis_rotation = args.alpha1_pelvis_rotation
    if args.alpha2_pose_rotation is not None:
        cfg.alpha2_pose_rotation = args.alpha2_pose_rotation
    if args.alpha3_pose_6d_aux is not None:
        cfg.alpha3_pose_6d_aux = args.alpha3_pose_6d_aux
    if args.alpha5_end_effector_rotation is not None:
        cfg.alpha5_end_effector_rotation = args.alpha5_end_effector_rotation
    if args.hidden_dim is not None:
        cfg.hidden_dim = args.hidden_dim
    if args.num_hidden_layers is not None:
        cfg.num_hidden_layers = args.num_hidden_layers
    if args.save_last_every_epochs is not None:
        cfg.save_last_every_epochs = args.save_last_every_epochs
    if args.save_best_every_epochs is not None:
        cfg.save_best_every_epochs = args.save_best_every_epochs
    if args.writer_flush_every_epochs is not None:
        cfg.writer_flush_every_epochs = args.writer_flush_every_epochs
    if args.run_name is not None:
        cfg.run_name = args.run_name
    cfg.use_torch_compile = not args.no_compile
    if args.compile_mode is not None:
        cfg.torch_compile_mode = args.compile_mode
    cfg.show_progress = args.progress
    if args.target_loss_reduction is not None:
        cfg.target_loss_reduction = args.target_loss_reduction
    cfg.stop_at_target_loss_reduction = args.stop_at_target_loss_reduction
    if args.max_train_seconds is not None:
        cfg.max_train_seconds = args.max_train_seconds
    cfg.profile_timing = args.profile_timing
    cfg.profile_sync_cuda = args.profile_sync_cuda
    cfg.disable_validation = args.no_validation
    set_seed(cfg.seed)

    npz_folder = resolve_path(args.folder_path or folder_path)
    device = torch.device(cfg.device)
    profiler = TimingProfiler(cfg.profile_timing, device, cfg.profile_sync_cuda)
    with profiler.section("setup/load_npz_and_precompute"):
        clips = load_clips(npz_folder, cfg)
    max_possible = min(clip.T - 2 for clip in clips)
    schedule = tuple(k for k in cfg.rollout_schedule if k <= max_possible)
    if not schedule:
        schedule = (1,)
    def make_loaders(max_rollout: int) -> tuple[DataLoader, DataLoader | None]:
        train_ds = MotionIndexDataset(clips, cfg, "train", max_rollout)
        loader_kwargs = {
            "batch_size": cfg.batch_size,
            "num_workers": cfg.num_workers,
            "pin_memory": device.type == "cuda",
            "persistent_workers": cfg.num_workers > 0,
        }
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        val_loader = None
        if not cfg.disable_validation:
            val_ds = MotionIndexDataset(clips, cfg, "val", max_rollout)
            val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
        return train_loader, val_loader

    with profiler.section("setup/model_optimizer_compile"):
        input_dim, output_dim = make_batch_dims(clips[0], cfg)
        if cfg.allow_tf32 and device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        model = MLPController(input_dim, output_dim, cfg).to(device)
        compile_enabled = False
        if cfg.use_torch_compile:
            if hasattr(torch, "compile"):
                try:
                    compiled_model = torch.compile(model, mode=cfg.torch_compile_mode)
                    with torch.no_grad():
                        compiled_model(torch.zeros(1, input_dim, device=device))
                    model = compiled_model
                    compile_enabled = True
                except Exception as exc:
                    print(f"torch.compile disabled: {exc}")
            else:
                print("torch.compile disabled: this PyTorch build does not expose torch.compile")
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if args.resume_checkpoint is not None:
        with profiler.section("setup/load_resume_checkpoint"):
            resume_path = resolve_path(args.resume_checkpoint)
            resume = torch.load(resume_path, map_location=device, weights_only=False)
            unwrap_compiled_model(model).load_state_dict(resume["model"])
        print(f"resumed model weights from {resume_path}")

    run_dir = resolve_path(cfg.output_dir) / cfg.run_name
    ckpt_dir = run_dir / "checkpoints"
    with profiler.section("setup/tensorboard_writer"):
        writer = SummaryWriter(run_dir / "tb")
    metadata = {
        "npz_folder": str(npz_folder),
        "body_names": clips[0].body_names,
        "parents_body": clips[0].parents_body.tolist(),
        "pelvis_index": clips[0].pelvis,
        "non_pelvis_indices": clips[0].non_pelvis,
        "end_effector_indices": clips[0].end_effectors,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "compile_enabled": compile_enabled,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    rollout_idx = 0
    rollout_k = schedule[rollout_idx]
    with profiler.section("setup/build_dataloaders"):
        train_loader, val_loader = make_loaders(rollout_k)
    val_sample_text = "disabled" if val_loader is None else str(len(val_loader.dataset))
    print(f"rollout_k={rollout_k} train_samples={len(train_loader.dataset)} val_samples={val_sample_text}")
    stage_start_epoch = 1
    stable_epochs = 0
    stall_epochs = 0
    best_val = float("inf")
    baseline_val = None
    target_val = None
    start_time = time.perf_counter()
    target_reached_epoch = None
    target_reached_seconds = None
    pending_best_payload = None

    def flush_pending_best() -> None:
        nonlocal pending_best_payload
        if pending_best_payload is None:
            return
        best_k = int(pending_best_payload["rollout_k"])
        with profiler.section("checkpoint/write_best"):
            save_payload(ckpt_dir / "checkpoint_best.pt", pending_best_payload)
            save_payload(ckpt_dir / f"checkpoint_best_k{best_k:02d}.pt", pending_best_payload)
        pending_best_payload = None

    for epoch in range(1, cfg.max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_parts = []
        pbar = tqdm(train_loader, desc=f"epoch {epoch} train K={rollout_k}", leave=False, disable=not cfg.show_progress)
        for batch in pbar:
            with profiler.section("train/zero_grad"):
                optimizer.zero_grad(set_to_none=True)
            with profiler.section("train/forward_loss"):
                loss, scalars = run_batch(model, clips, batch, cfg, rollout_k, device, train=True)
            with profiler.section("train/backward"):
                loss.backward()
            with profiler.section("train/clip_grad"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            with profiler.section("train/optimizer_step"):
                optimizer.step()
            train_parts.append(scalars)
            if cfg.show_progress:
                pbar.set_postfix(loss=f"{scalars['total']:.4f}")

        model.eval()
        val_parts = []
        if val_loader is None:
            val_parts = train_parts
        else:
            with torch.no_grad():
                for batch in val_loader:
                    with profiler.section("validation/forward_loss"):
                        _, scalars = run_batch(model, clips, batch, cfg, rollout_k, device, train=False)
                    val_parts.append(scalars)

        def mean_scalar(parts: list[dict[str, float]], key: str) -> float:
            return float(np.mean([p[key] for p in parts])) if parts else 0.0

        train_total = mean_scalar(train_parts, "total")
        val_total = mean_scalar(val_parts, "total")
        elapsed_seconds = time.perf_counter() - start_time
        epoch_seconds = time.perf_counter() - epoch_start
        if baseline_val is None:
            baseline_val = val_total
            target_val = baseline_val * (1.0 - cfg.target_loss_reduction)
        reduction = 0.0 if baseline_val <= 0.0 else 1.0 - (val_total / baseline_val)
        with profiler.section("logging/tensorboard_scalars"):
            writer.add_scalar("loss/train_total", train_total, epoch)
            writer.add_scalar("loss/validation_total", val_total, epoch)
            for key in (
                "pelvis_location",
                "pelvis_rotation",
                "pose_rotation",
                "pose_6d_aux",
                "end_effector_location",
                "end_effector_rotation",
                "full_body_location",
            ):
                writer.add_scalar(f"loss/train_{key}", mean_scalar(train_parts, key), epoch)
                writer.add_scalar(f"loss/validation_{key}", mean_scalar(val_parts, key), epoch)
            writer.add_scalar("curriculum/rollout_k", rollout_k, epoch)
            writer.add_scalar("optim/learning_rate", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("timing/epoch_seconds", epoch_seconds, epoch)
            writer.add_scalar("timing/elapsed_seconds", elapsed_seconds, epoch)
            writer.add_scalar("timing/validation_loss_reduction", reduction, epoch)
            if cfg.writer_flush_every_epochs > 0 and epoch % cfg.writer_flush_every_epochs == 0:
                writer.flush()

        if cfg.save_last_every_epochs > 0 and epoch % cfg.save_last_every_epochs == 0:
            with profiler.section("checkpoint/write_last_periodic"):
                save_checkpoint(ckpt_dir / "checkpoint_last.pt", model, optimizer, epoch, best_val, rollout_k, cfg, metadata)
        improved_for_stall = val_total < best_val - cfg.curriculum_min_delta
        if val_total < best_val:
            best_val = val_total
            with profiler.section("checkpoint/build_best_payload"):
                payload = checkpoint_payload(model, optimizer, epoch, best_val, rollout_k, cfg, metadata)
                pending_best_payload = clone_checkpoint_payload(payload)
            if cfg.save_best_every_epochs > 0 and epoch % cfg.save_best_every_epochs == 0:
                flush_pending_best()
        stall_epochs = 0 if improved_for_stall else stall_epochs + 1
        if epoch % cfg.checkpoint_every_epochs == 0:
            with profiler.section("checkpoint/write_numbered"):
                save_checkpoint(
                    ckpt_dir / f"checkpoint_epoch_{epoch:06d}.pt",
                    model,
                    optimizer,
                    epoch,
                    best_val,
                    rollout_k,
                    cfg,
                    metadata,
                )

        stage_epochs = epoch - stage_start_epoch + 1
        can_advance_by_loss = (
            val_total <= cfg.curriculum_threshold
            and stage_epochs >= cfg.curriculum_min_epochs
        )
        can_advance_by_epoch_cap = (
            cfg.curriculum_max_epochs_per_stage > 0
            and stage_epochs >= cfg.curriculum_max_epochs_per_stage
        )
        can_advance_by_stall = (
            cfg.curriculum_stall_patience_epochs > 0
            and stage_epochs >= cfg.curriculum_min_epochs
            and stall_epochs >= cfg.curriculum_stall_patience_epochs
        )
        if can_advance_by_loss:
            stable_epochs += 1
        else:
            stable_epochs = 0
        should_advance = (
            stable_epochs >= cfg.curriculum_patience_epochs
            or can_advance_by_epoch_cap
            or can_advance_by_stall
        )
        was_final_stage = rollout_idx == len(schedule) - 1
        if should_advance and rollout_idx < len(schedule) - 1:
            flush_pending_best()
            reason = "loss" if stable_epochs >= cfg.curriculum_patience_epochs else "epoch_cap"
            if can_advance_by_stall:
                reason = "stall"
            rollout_idx += 1
            rollout_k = schedule[rollout_idx]
            with profiler.section("curriculum/rebuild_dataloaders"):
                train_loader, val_loader = make_loaders(rollout_k)
            best_val = float("inf")
            stage_start_epoch = epoch + 1
            val_sample_text = "disabled" if val_loader is None else str(len(val_loader.dataset))
            print(
                f"advanced rollout_k={rollout_k} reason={reason} "
                f"train_samples={len(train_loader.dataset)} val_samples={val_sample_text}",
                flush=True,
            )
            stable_epochs = 0
            stall_epochs = 0

        print(
            f"epoch={epoch:04d} K={rollout_k:02d} train={train_total:.6f} "
            f"val={val_total:.6f} best={best_val:.6f} "
            f"reduction={reduction * 100.0:.2f}% stall={stall_epochs} "
            f"epoch_s={epoch_seconds:.2f} elapsed_s={elapsed_seconds:.2f}",
            flush=True,
        )
        if cfg.stop_on_final_stall and was_final_stage and can_advance_by_stall:
            print(
                f"final rollout_k={rollout_k} stopped on validation stall "
                f"after {stall_epochs} epochs without improvement >= {cfg.curriculum_min_delta:g}",
                flush=True,
            )
            break
        if target_reached_epoch is None and target_val is not None and val_total <= target_val:
            target_reached_epoch = epoch
            target_reached_seconds = elapsed_seconds
            print(
                f"target_loss_reduction={cfg.target_loss_reduction * 100.0:.2f}% "
                f"reached at epoch={epoch} elapsed_s={elapsed_seconds:.2f} "
                f"baseline_val={baseline_val:.6f} target_val={target_val:.6f} val={val_total:.6f}",
                flush=True,
            )
            if cfg.stop_at_target_loss_reduction:
                break
        if cfg.max_train_seconds > 0.0 and elapsed_seconds >= cfg.max_train_seconds:
            print(
                f"max_train_seconds={cfg.max_train_seconds:.2f} reached at epoch={epoch} "
                f"elapsed_s={elapsed_seconds:.2f}",
                flush=True,
            )
            break

    with profiler.section("logging/tensorboard_close"):
        writer.close()
    flush_pending_best()
    with profiler.section("checkpoint/write_last_final"):
        save_checkpoint(ckpt_dir / "checkpoint_last.pt", model, optimizer, epoch, best_val, rollout_k, cfg, metadata)
    total_seconds = time.perf_counter() - start_time
    profiler.write_csv(run_dir / "timing_profile.csv", time.perf_counter() - process_start_time)
    if target_reached_epoch is None and baseline_val is not None and target_val is not None:
        print(
            f"target_loss_reduction={cfg.target_loss_reduction * 100.0:.2f}% not reached "
            f"after {epoch} epochs elapsed_s={total_seconds:.2f} "
            f"baseline_val={baseline_val:.6f} target_val={target_val:.6f} best_val={best_val:.6f}"
        )
    elif target_reached_epoch is not None:
        print(
            f"timing_summary target_epoch={target_reached_epoch} "
            f"target_elapsed_s={target_reached_seconds:.2f} total_elapsed_s={total_seconds:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a kinematic locomotion imitator from NPZ motion clips.")
    parser.add_argument("--folder-path", default=None, help="Override top-level folder_path.")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--future-window-seconds", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--predict-residual", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--zero-init-output", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rollout-schedule", default=None, help="Comma-separated rollout K values, e.g. 1,2,4,8,16,32.")
    parser.add_argument("--curriculum-threshold", type=float, default=None)
    parser.add_argument("--curriculum-min-epochs", type=int, default=None)
    parser.add_argument("--curriculum-max-epochs-per-stage", type=int, default=None)
    parser.add_argument("--curriculum-patience-epochs", type=int, default=None)
    parser.add_argument("--curriculum-stall-patience-epochs", type=int, default=None)
    parser.add_argument("--curriculum-min-delta", type=float, default=None)
    parser.add_argument("--no-stop-on-final-stall", action="store_true")
    parser.add_argument("--alpha0-pelvis-location", type=float, default=None)
    parser.add_argument("--alpha1-pelvis-rotation", type=float, default=None)
    parser.add_argument("--alpha2-pose-rotation", type=float, default=None)
    parser.add_argument("--alpha3-pose-6d-aux", type=float, default=None)
    parser.add_argument("--alpha4-end-effector-location", type=float, default=None)
    parser.add_argument("--alpha5-end-effector-rotation", type=float, default=None)
    parser.add_argument("--alpha6-full-body-location", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-hidden-layers", type=int, default=None)
    parser.add_argument("--save-last-every-epochs", type=int, default=None)
    parser.add_argument("--save-best-every-epochs", type=int, default=None)
    parser.add_argument("--writer-flush-every-epochs", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile.")
    parser.add_argument("--compile-mode", default=None, help="torch.compile mode, for example default or reduce-overhead.")
    parser.add_argument("--progress", action="store_true", help="Show per-epoch tqdm progress bars.")
    parser.add_argument("--target-loss-reduction", type=float, default=None)
    parser.add_argument("--stop-at-target-loss-reduction", action="store_true")
    parser.add_argument("--max-train-seconds", type=float, default=None)
    parser.add_argument("--profile-timing", action="store_true", help="Write timing_profile.csv in the run directory.")
    parser.add_argument(
        "--profile-sync-cuda",
        action="store_true",
        help="Synchronize CUDA around timed sections for stricter timing. Slower, but more precise.",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip validation passes and use train loss for curriculum/checkpoint selection.",
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
