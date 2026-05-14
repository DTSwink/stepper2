from __future__ import annotations

import math
import json
import sys
import threading
import time
import tkinter as tk
import ctypes
from dataclasses import dataclass, field, replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageTk
from OpenGL import GL, GLU
from pyopengltk import OpenGLFrame

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[0]
FBX_PIPELINE_DIR = PROJECT_ROOT / "fbx_npz_pipeline"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(FBX_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(FBX_PIPELINE_DIR))

import train_locomotion as tl
from foot_contact import DEFAULT_CONFIG as FOOT_CONTACT_CONFIG
from foot_contact import (
    FootContactConfig,
    box_lowest_point_signed_height,
    foot_lowest_ground_height,
    foot_sole_slide_distance,
    foot_toe_box_specs,
    is_foot_contact,
)
from visualize_model import apply_config_dict, load_model


DEFAULT_NPZ_DIR = PROJECT_ROOT / "data" / "npz_final"
DEFAULT_ANIMATION_LIBRARY_DIR = PROJECT_ROOT / "ue5" / "animations_omni_only" / "npz_final"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "training" / "runs"
APP_ICON = PROJECT_ROOT / "training" / "assets" / "stepper_model_viewer.ico"
SETTINGS_PATH = PROJECT_ROOT / "training" / "model_viewer_settings.json"

BG = "#101216"
PANEL = "#1a1d22"
PANEL_2 = "#23272f"
TEXT = "#edf1f7"
MUTED = "#9da6b5"
LINE = "#313846"
FLOOR = "#303846"
SHADOW = "#090b0f"
SKY = "#000000"
RENDER_SCALE = 1.5
GT_BLUE = "#72a7ff"
PRED_ORANGE = "#ff9b6a"
ACTOR_COLORS = ("#72a7ff", "#ff9b6a", "#55d6a7", "#e6df83", "#c58cff", "#ff6f91")
LIGHT_DIR = np.asarray([-0.45, 0.85, -0.30], dtype=np.float32)
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
FLAT_PITCH = 0.35 / 0.55
NEUTRAL_PITCH = 0.48
MIN_CAMERA_DISTANCE = 0.01
CAMERA_TRANSITION_SECONDS = 0.22
DEFAULT_FOOT_LENGTH = 0.175
DEFAULT_FOOT_WIDTH = 0.12
DEFAULT_FOOT_HEIGHT = 0.051
DEFAULT_TOE_LENGTH = 0.048
DEFAULT_TOE_WIDTH = 0.12
DEFAULT_TOE_HEIGHT = 0.049
DEFAULT_HAND_LENGTH = 0.09
DEFAULT_HAND_WIDTH = 0.068
DEFAULT_HAND_HEIGHT = 0.032
TARGET_RENDER_FPS = 60.0
TARGET_RENDER_INTERVAL_MS = max(1, int(round(1000.0 / TARGET_RENDER_FPS)))
TARGET_PLAYBACK_FPS = 120.0
MAX_PLAYBACK_SUBSTEPS = 3
XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE = 7849
XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE = 8689
XINPUT_THUMB_MAX = 32767.0


class XInputGamepad(ctypes.Structure):
    _fields_ = [
        ("wButtons", ctypes.c_ushort),
        ("bLeftTrigger", ctypes.c_ubyte),
        ("bRightTrigger", ctypes.c_ubyte),
        ("sThumbLX", ctypes.c_short),
        ("sThumbLY", ctypes.c_short),
        ("sThumbRX", ctypes.c_short),
        ("sThumbRY", ctypes.c_short),
    ]


class XInputState(ctypes.Structure):
    _fields_ = [("dwPacketNumber", ctypes.c_uint), ("Gamepad", XInputGamepad)]


def load_xinput() -> object | None:
    if not sys.platform.startswith("win"):
        return None
    for dll_name in ("xinput1_4", "xinput1_3", "xinput9_1_0"):
        try:
            return ctypes.windll.LoadLibrary(dll_name)
        except OSError:
            continue
    return None


def apply_stick_deadzone(x: int, y: int, deadzone: float) -> np.ndarray:
    raw = np.asarray([float(x), float(y)], dtype=np.float32)
    magnitude = float(np.linalg.norm(raw))
    if magnitude <= deadzone:
        return np.zeros(2, dtype=np.float32)
    scaled = (magnitude - deadzone) / max(1.0, XINPUT_THUMB_MAX - deadzone)
    return (raw / magnitude * min(1.0, scaled)).astype(np.float32)


@dataclass
class ControllerSample:
    connected: bool = False
    move: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    look: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    triggers: float = 0.0


@dataclass
class RootMotionSettings:
    max_speed_mps: float
    acceleration_response: float
    deceleration_response: float
    turn_rate_dps: float
    left_deadzone: float
    right_deadzone: float
    trajectory_seconds: float
    trajectory_step_seconds: float


@dataclass
class RootMotionState:
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    yaw: float = 0.0


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def vec3(values: tuple[float, float, float] | list[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def add3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)


def sub3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)


def mul3(a: np.ndarray, s: float) -> np.ndarray:
    return np.asarray(a, dtype=np.float32) * float(s)


def dot3(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def yaw_rotation_matrix_np(yaw: float) -> np.ndarray:
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    return np.asarray([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=np.float32)


def angle_wrap_np(angle: float) -> float:
    return (float(angle) + math.pi) % math.tau - math.pi


def integrate_controller_root_motion(
    state: RootMotionState,
    move_input: np.ndarray,
    look_input: np.ndarray,
    dt: float,
    settings: RootMotionSettings,
    movement_yaw: float | None = None,
) -> RootMotionState:
    """Controller root motion: stick intent drives the character root, pose stays layered above it."""

    dt = max(0.0, min(0.1, float(dt)))
    move = np.asarray(move_input, dtype=np.float32)
    look = np.asarray(look_input, dtype=np.float32)
    move_mag = min(1.0, float(np.linalg.norm(move)))
    if move_mag > 1e-6:
        move_dir = move / move_mag
    else:
        move_dir = np.zeros(2, dtype=np.float32)
    target_velocity = move_dir * (float(settings.max_speed_mps) * move_mag)
    response = float(settings.acceleration_response if np.linalg.norm(target_velocity) > np.linalg.norm(state.velocity) else settings.deceleration_response)
    blend = 1.0 - math.exp(-max(0.01, response) * dt)
    velocity = (state.velocity + (target_velocity - state.velocity) * blend).astype(np.float32)
    yaw_input = float(look[0]) if look.size > 0 else 0.0
    yaw = float(state.yaw) + yaw_input * math.radians(float(settings.turn_rate_dps)) * dt
    move_yaw = yaw if movement_yaw is None else float(movement_yaw)
    world_delta = np.asarray([velocity[0], 0.0, velocity[1]], dtype=np.float32) @ yaw_rotation_matrix_np(move_yaw) * dt
    return RootMotionState(
        offset=(state.offset + world_delta).astype(np.float32),
        velocity=velocity,
        yaw=yaw,
    )


def actor_basename(path: Path) -> str:
    if path.suffix.lower() in {".pt", ".pth"} and path.parent.name == "checkpoints":
        return path.parent.parent.name
    return path.stem


def newest_best_checkpoint() -> Path | None:
    if not DEFAULT_RUNS_DIR.exists():
        return None
    candidates = list(DEFAULT_RUNS_DIR.glob("*/checkpoints/checkpoint_best*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def newest_npz() -> Path | None:
    if not DEFAULT_NPZ_DIR.exists():
        return None
    candidates = list(DEFAULT_NPZ_DIR.glob("*.npz"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def usable_work_area(root: tk.Tk) -> tuple[int, int, int, int]:
    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes

            rect = wintypes.RECT()
            if ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(rect), 0):
                return int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)
        except Exception:
            pass
    return 0, 0, int(root.winfo_screenwidth()), int(root.winfo_screenheight())


def enable_precise_timers() -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes

        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass


def disable_precise_timers() -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes

        ctypes.windll.winmm.timeEndPeriod(1)
    except Exception:
        pass


def next_power_of_two(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def load_app_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_app_settings(settings: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def blend_hex(color: str, target: str, amount: float) -> str:
    amount = max(0.0, min(1.0, amount))
    c = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
    t = tuple(int(target[i : i + 2], 16) for i in (1, 3, 5))
    mixed = tuple(int(c[i] * (1.0 - amount) + t[i] * amount) for i in range(3))
    return f"#{mixed[0]:02x}{mixed[1]:02x}{mixed[2]:02x}"


@dataclass
class Actor:
    actor_id: int
    kind: str
    name: str
    color: str
    visible: bool = True
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    npz_path: Path | None = None
    checkpoint_path: Path | None = None
    source_npz_path: Path | None = None
    cfg: tl.TrainConfig | None = None
    clip: tl.MotionClip | None = None
    checkpoint: dict | None = None
    model: torch.nn.Module | None = None
    device: torch.device | None = None
    clip_tensors: dict[str, torch.Tensor] | None = None
    source_contacts: np.ndarray | None = None
    initial_pelvis_offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    generated_pos: list[np.ndarray] = field(default_factory=list)
    generated_rot: list[np.ndarray] = field(default_factory=list)
    generated_contacts: list[np.ndarray] = field(default_factory=list)
    controller_root_offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    controller_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    controller_root_anchor: np.ndarray | None = None
    controller_yaw: float = 0.0
    controller_move_yaw: float = 0.0
    controller_prev_root_pos: np.ndarray | None = None
    controller_cur_root_pos: np.ndarray | None = None
    controller_prev_root_yaw: float | None = None
    controller_cur_root_yaw: float | None = None
    controller_future_root_pos: np.ndarray | None = None
    controller_future_root_yaw: np.ndarray | None = None
    controller_active: bool = False
    controller_move_input: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    controller_turn_input: float = 0.0
    controller_settings_snapshot: RootMotionSettings | None = None
    prev_idx: torch.Tensor | None = None
    cur_idx: torch.Tensor | None = None
    prev_pose: dict[str, torch.Tensor] | None = None
    cur_pose: dict[str, torch.Tensor] | None = None
    generation_lock: object = field(default_factory=threading.Lock, repr=False)
    generation_target: int = 0
    generation_thread: threading.Thread | None = field(default=None, repr=False)
    status: str = "ready"

    @property
    def frame_count(self) -> int:
        if self.kind == "model" and self.controller_active:
            generated_count = min(len(self.generated_pos), len(self.generated_rot))
            return max(int(self.clip.T) if self.clip is not None else 1, generated_count, int(self.generation_target) + 1)
        return int(self.clip.T) if self.clip is not None else 1

    @property
    def bone_names(self) -> list[str]:
        return self.clip.body_names if self.clip is not None else []

    @property
    def parents(self) -> list[int]:
        return self.clip.parents_body_list if self.clip is not None else []

    @property
    def root_index(self) -> int:
        return int(self.clip.pelvis) if self.clip is not None else 0

    def has_initial_pelvis_offset(self) -> bool:
        return bool(np.linalg.norm(self.initial_pelvis_offset) > 1e-7)

    def apply_initial_pelvis_offset_to_pose(
        self, pose: dict[str, torch.Tensor], idx: torch.Tensor, device: torch.device
    ) -> dict[str, torch.Tensor]:
        if not self.has_initial_pelvis_offset() or self.clip is None:
            return pose
        out = tl.clone_pose(pose)
        tensors = self.clip_tensors if self.clip_tensors is not None else self.clip.tensors(device)
        root_rot = tensors["root_rot"].index_select(0, idx.to(device))
        world_offset = torch.tensor(self.initial_pelvis_offset, dtype=out["pelvis_pos"].dtype, device=device).view(1, 3)
        local_offset = torch.matmul(world_offset.unsqueeze(1), root_rot.transpose(-1, -2)).squeeze(1)
        out["pelvis_pos"] = out["pelvis_pos"] + local_offset
        if "contacts" in out:
            out["contacts"] = torch.zeros_like(out["contacts"])
        return out

    def refresh_seed_global_pose(self, index: int, pose: dict[str, torch.Tensor], device: torch.device) -> None:
        if self.clip is None:
            return
        tensors = self.clip_tensors if self.clip_tensors is not None else self.clip.tensors(device)
        idx = torch.tensor([index], dtype=torch.long, device=device)
        root_pos = tensors["root_pos"].index_select(0, idx)
        root_rot = tensors["root_rot"].index_select(0, idx)
        global_pos, global_rot, _canon = tl.fk_from_pose(self.clip, root_pos, root_rot, pose, device)
        while len(self.generated_pos) <= index:
            self.generated_pos.append(global_pos[0].detach().cpu().numpy().astype(np.float32))
            self.generated_rot.append(global_rot[0].detach().cpu().numpy().astype(np.float32))
        self.generated_pos[index] = global_pos[0].detach().cpu().numpy().astype(np.float32)
        self.generated_rot[index] = global_rot[0].detach().cpu().numpy().astype(np.float32)
        while len(self.generated_contacts) <= index:
            self.generated_contacts.append(np.zeros(2, dtype=np.float32))
        self.generated_contacts[index] = self.pose_contacts_numpy(pose)

    def pose_contacts_numpy(self, pose: dict[str, torch.Tensor]) -> np.ndarray:
        contacts = pose.get("contacts")
        if contacts is None:
            return np.zeros(2, dtype=np.float32)
        values = contacts.detach().cpu().numpy().reshape(-1)
        out = np.zeros(2, dtype=np.float32)
        out[: min(2, values.shape[0])] = values[:2].astype(np.float32)
        return out

    def load_npz(self, path: Path, cfg: tl.TrainConfig | None = None) -> None:
        self.npz_path = path
        self.source_npz_path = path if self.kind == "model" else None
        self.cfg = cfg or tl.TrainConfig()
        self.cfg.use_torch_compile = False
        self.clip = tl.MotionClip(path, self.cfg)
        self.clip_tensors = None
        self.source_contacts = self.load_source_contacts(path)
        self.status = f"{self.clip.T} frames, {self.clip.J} bones"

    def load_source_contacts(self, path: Path) -> np.ndarray | None:
        try:
            with np.load(path, allow_pickle=True) as arrays:
                if "contacts" not in arrays.files:
                    return None
                contacts = np.asarray(arrays["contacts"], dtype=np.bool_)
        except (OSError, ValueError):
            return None
        if contacts.ndim != 2 or contacts.shape[1] < 2:
            return None
        return contacts[:, :2].copy()

    def load_checkpoint(self, checkpoint_path: Path, source_npz_path: Path, device: torch.device) -> None:
        self.checkpoint_path = checkpoint_path
        self.source_npz_path = source_npz_path
        self.device = device
        self.checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.cfg = tl.TrainConfig()
        apply_config_dict(self.cfg, self.checkpoint.get("config", {}))
        self.cfg.device = str(device)
        self.cfg.use_torch_compile = False
        self.clip = tl.MotionClip(source_npz_path, self.cfg)
        self.source_contacts = None
        try:
            self.model = load_model(self.checkpoint, self.clip, self.cfg, device)
        except RuntimeError as exc:
            input_dim, output_dim = tl.make_batch_dims(self.clip, self.cfg)
            meta = self.checkpoint.get("metadata", {})
            ckpt_dims = f"{meta.get('input_dim', '?')} -> {meta.get('output_dim', '?')}"
            raise ValueError(
                "Checkpoint does not match this source NPZ skeleton/config. "
                f"Source expects {input_dim} -> {output_dim}, checkpoint metadata is {ckpt_dims}."
            ) from exc
        self.model.eval()
        self.clip_tensors = self.clip.tensors(device)
        self.reset_generation()
        self.status = f"on-demand model, source {source_npz_path.name}"

    def reset_generation(self) -> None:
        with self.generation_lock:
            self.generated_pos.clear()
            self.generated_rot.clear()
            self.generated_contacts.clear()
            self.controller_active = False
            self.controller_move_input = np.zeros(2, dtype=np.float32)
            self.controller_turn_input = 0.0
            self.clear_controller_root_plan()
            if self.clip is None:
                return
            first_count = min(2, self.clip.T)
            for i in range(first_count):
                self.generated_pos.append(self.clip.global_pos[i].detach().cpu().numpy().astype(np.float32))
                self.generated_rot.append(self.clip.global_rot[i].detach().cpu().numpy().astype(np.float32))
                if hasattr(self.clip, "contacts"):
                    self.generated_contacts.append(self.clip.contacts[i].detach().cpu().numpy().reshape(-1)[:2].astype(np.float32))
                else:
                    self.generated_contacts.append(np.zeros(2, dtype=np.float32))
            self.generation_target = max(0, first_count - 1)
            if self.kind == "model" and self.model is not None and self.clip.T >= 2:
                device = self.device or torch.device("cpu")
                self.prev_idx = torch.tensor([0], dtype=torch.long)
                self.cur_idx = torch.tensor([1], dtype=torch.long)
                self.prev_pose = tl.get_pose_from_clip(self.clip, self.prev_idx, device)
                self.cur_pose = tl.get_pose_from_clip(self.clip, self.cur_idx, device)
                if self.cfg is not None:
                    self.prev_pose, self.cur_pose = tl.maybe_apply_initial_offsets(
                        self.clip,
                        self.prev_idx,
                        self.cur_idx,
                        self.prev_pose,
                        self.cur_pose,
                        self.cfg,
                        device,
                    )
                    if self.has_initial_pelvis_offset():
                        self.prev_pose = self.apply_initial_pelvis_offset_to_pose(self.prev_pose, self.prev_idx, device)
                        self.cur_pose = self.apply_initial_pelvis_offset_to_pose(self.cur_pose, self.cur_idx, device)
                        self.refresh_seed_global_pose(0, self.prev_pose, device)
                        self.refresh_seed_global_pose(1, self.cur_pose, device)
                    elif self.cfg.freefall_body_height_offset_m != 0.0:
                        tensors = self.clip_tensors if self.clip_tensors is not None else self.clip.tensors(device)
                        prev_root_pos = tensors["root_pos"].index_select(0, self.prev_idx.to(device))
                        prev_root_rot = tensors["root_rot"].index_select(0, self.prev_idx.to(device))
                        cur_root_pos = tensors["root_pos"].index_select(0, self.cur_idx.to(device))
                        cur_root_rot = tensors["root_rot"].index_select(0, self.cur_idx.to(device))
                        prev_pos, prev_rot, _ = tl.fk_from_pose(self.clip, prev_root_pos, prev_root_rot, self.prev_pose, device)
                        cur_pos, cur_rot, _ = tl.fk_from_pose(self.clip, cur_root_pos, cur_root_rot, self.cur_pose, device)
                        if len(self.generated_pos) > 0:
                            self.generated_pos[0] = prev_pos[0].detach().cpu().numpy().astype(np.float32)
                            self.generated_rot[0] = prev_rot[0].detach().cpu().numpy().astype(np.float32)
                        if len(self.generated_pos) > 1:
                            self.generated_pos[1] = cur_pos[0].detach().cpu().numpy().astype(np.float32)
                            self.generated_rot[1] = cur_rot[0].detach().cpu().numpy().astype(np.float32)

    def pose_for_frame(self, frame: int, generate: bool = True) -> tuple[np.ndarray, np.ndarray] | None:
        if self.clip is None:
            return None
        requested_frame = max(0, int(frame))
        frame = requested_frame if self.kind == "model" and self.controller_active else min(requested_frame, self.clip.T - 1)
        if self.kind == "npz":
            return (
                self.clip.global_pos[frame].detach().cpu().numpy().astype(np.float32),
                self.clip.global_rot[frame].detach().cpu().numpy().astype(np.float32),
            )
        if self.model is None:
            return None
        if generate:
            self.generate_to(frame)
        elif frame >= len(self.generated_pos):
            if not self.generated_pos:
                return None
            frame = len(self.generated_pos) - 1
        generated_count = min(len(self.generated_pos), len(self.generated_rot))
        if generated_count <= 0:
            return None
        frame = min(frame, generated_count - 1)
        return self.generated_pos[frame], self.generated_rot[frame]

    def start_async_generation(self, frame: int) -> None:
        if self.kind != "model" or self.model is None or self.clip is None:
            return
        if self.controller_active:
            frame = max(0, int(frame))
        else:
            frame = max(0, min(int(frame), self.clip.T - 1))
        self.generation_target = max(self.generation_target, frame)
        thread = self.generation_thread
        if thread is not None and thread.is_alive():
            return
        self.generation_thread = threading.Thread(target=self._generation_worker, daemon=True)
        self.generation_thread.start()

    def _generation_worker(self) -> None:
        while True:
            target = int(self.generation_target)
            self.generate_to(target)
            if int(self.generation_target) <= target:
                return

    @torch.no_grad()
    def generate_to(self, frame: int) -> None:
        with self.generation_lock:
            self._generate_to_locked(frame)

    def _generate_to_locked(self, frame: int) -> None:
        if (
            self.kind != "model"
            or self.model is None
            or self.clip is None
            or self.cfg is None
            or self.device is None
            or self.prev_idx is None
            or self.cur_idx is None
            or self.prev_pose is None
            or self.cur_pose is None
        ):
            return
        tensors = self.clip_tensors if self.clip_tensors is not None else self.clip.tensors(self.device)
        while len(self.generated_pos) <= frame and (self.controller_active or len(self.generated_pos) < self.clip.T):
            target = len(self.generated_pos)
            target_idx = torch.tensor([target], dtype=torch.long)
            source_target_idx = torch.tensor([min(target, self.clip.T - 1)], dtype=torch.long, device=self.device)
            root_pos = tensors["root_pos"].index_select(0, source_target_idx)
            root_rot = tensors["root_rot"].index_select(0, source_target_idx)
            if self.controller_active:
                inp = self.build_controller_input()
            else:
                inp = tl.build_input(
                    self.clip,
                    self.prev_idx,
                    self.cur_idx,
                    self.prev_pose,
                    self.cur_pose,
                    self.cfg,
                    self.device,
                )
            raw_out = tl.predict_next_raw(self.model, inp, self.cur_pose, self.cfg)
            pred_pose, _ = tl.output_to_pose(raw_out, self.clip)
            global_pos, global_rot, canon_pos = tl.fk_from_pose(
                self.clip, root_pos, root_rot, pred_pose, self.device
            )
            self.generated_pos.append(global_pos[0].detach().cpu().numpy().astype(np.float32))
            self.generated_rot.append(global_rot[0].detach().cpu().numpy().astype(np.float32))
            self.generated_contacts.append(self.pose_contacts_numpy(pred_pose))
            self.prev_pose = self.cur_pose
            self.cur_pose = {
                "pelvis_pos": pred_pose["pelvis_pos"],
                "pelvis_rot6": pred_pose["pelvis_rot6"],
                "nonpelvis_rot6": pred_pose["nonpelvis_rot6"],
                "canon_pos": canon_pos,
                "contacts": pred_pose["contacts"],
            }
            self.prev_idx = self.cur_idx
            self.cur_idx = target_idx

    def build_controller_input(self) -> torch.Tensor:
        assert self.clip is not None and self.cfg is not None and self.device is not None
        assert self.prev_pose is not None and self.cur_pose is not None and self.cur_idx is not None
        current = tl.body_pose_vector(self.cur_pose, self.cfg.use_contact_state, self.cfg.zero_contact_state)
        previous = tl.body_pose_vector(self.prev_pose, self.cfg.use_contact_state, self.cfg.zero_contact_state)
        pelvis_vel = (self.cur_pose["pelvis_pos"] - self.prev_pose["pelvis_pos"]) / self.cfg.pose_delta_scale_final
        joint_vel = (self.cur_pose["canon_pos"] - self.prev_pose["canon_pos"]).reshape(self.cur_idx.shape[0], -1) / self.cfg.pose_delta_scale_final
        root_feat = self.controller_root_delta_feature()
        future_feat = self.controller_future_root_features()
        return torch.cat((current, previous, pelvis_vel, joint_vel, root_feat, future_feat), dim=-1)

    def clear_controller_root_plan(self) -> None:
        self.controller_prev_root_pos = None
        self.controller_cur_root_pos = None
        self.controller_prev_root_yaw = None
        self.controller_cur_root_yaw = None
        self.controller_future_root_pos = None
        self.controller_future_root_yaw = None

    def set_controller_root_plan(
        self,
        prev_pos: np.ndarray,
        cur_pos: np.ndarray,
        prev_yaw: float,
        cur_yaw: float,
        future_pos: np.ndarray,
        future_yaw: np.ndarray,
    ) -> None:
        self.controller_prev_root_pos = np.asarray(prev_pos, dtype=np.float32).reshape(3)
        self.controller_cur_root_pos = np.asarray(cur_pos, dtype=np.float32).reshape(3)
        self.controller_prev_root_yaw = float(prev_yaw)
        self.controller_cur_root_yaw = float(cur_yaw)
        self.controller_future_root_pos = np.asarray(future_pos, dtype=np.float32).reshape(-1, 3)
        self.controller_future_root_yaw = np.asarray(future_yaw, dtype=np.float32).reshape(-1)

    def controller_root_delta_feature(self) -> torch.Tensor:
        assert self.cfg is not None and self.device is not None and self.clip is not None
        if (
            self.controller_prev_root_pos is not None
            and self.controller_cur_root_pos is not None
            and self.controller_prev_root_yaw is not None
            and self.controller_cur_root_yaw is not None
        ):
            delta = (self.controller_cur_root_pos - self.controller_prev_root_pos) @ yaw_rotation_matrix_np(self.controller_prev_root_yaw)
            yaw_delta = angle_wrap_np(float(self.controller_cur_root_yaw) - float(self.controller_prev_root_yaw))
            values = np.asarray(
                [
                    float(delta[0]) / max(1e-6, float(self.cfg.max_speed_scale_final)),
                    float(delta[2]) / max(1e-6, float(self.cfg.max_speed_scale_final)),
                    yaw_delta / max(1e-6, float(self.cfg.max_turn_rate_scale_final)),
                ],
                dtype=np.float32,
            )
            return torch.tensor(values, dtype=torch.float32, device=self.device).view(1, 3)
        fps = max(1.0, float(self.clip.fps))
        dt = 1.0 / fps
        world_delta = np.asarray([self.controller_velocity[0], 0.0, self.controller_velocity[1]], dtype=np.float32)
        world_delta = world_delta @ yaw_rotation_matrix_np(self.controller_move_yaw) * dt
        local_delta = world_delta @ yaw_rotation_matrix_np(self.controller_yaw).T
        settings = self.controller_settings_snapshot or RootMotionSettings(1.8, 9.0, 11.0, 240.0, 0.24, 0.26, 1.0, 0.16)
        yaw_delta = float(self.controller_turn_input) * math.radians(float(settings.turn_rate_dps)) * dt
        values = np.asarray(
            [
                local_delta[0] / max(1e-6, float(self.cfg.max_speed_scale_final)),
                local_delta[2] / max(1e-6, float(self.cfg.max_speed_scale_final)),
                yaw_delta / max(1e-6, float(self.cfg.max_turn_rate_scale_final)),
            ],
            dtype=np.float32,
        )
        return torch.tensor(values, dtype=torch.float32, device=self.device).view(1, 3)

    def controller_future_root_features(self) -> torch.Tensor:
        assert self.cfg is not None and self.device is not None and self.clip is not None
        if (
            self.controller_cur_root_pos is not None
            and self.controller_cur_root_yaw is not None
            and self.controller_future_root_pos is not None
            and self.controller_future_root_yaw is not None
        ):
            feats: list[float] = []
            cur_pos = self.controller_cur_root_pos
            cur_yaw = float(self.controller_cur_root_yaw)
            heading = yaw_rotation_matrix_np(cur_yaw)
            future_count = int(self.cfg.future_window)
            for k in range(1, future_count + 1):
                idx = min(k - 1, len(self.controller_future_root_pos) - 1)
                fut_pos = self.controller_future_root_pos[idx]
                fut_local = (fut_pos - cur_pos) @ heading
                scale_k = max(1e-6, float(k) * float(self.cfg.max_speed_scale_final))
                dx = max(-2.0, min(2.0, float(fut_local[0]) / scale_k))
                dz = max(-2.0, min(2.0, float(fut_local[2]) / scale_k))
                dyaw = angle_wrap_np(float(self.controller_future_root_yaw[idx]) - cur_yaw)
                feats.extend((dx, dz, math.cos(dyaw), math.sin(dyaw)))
            return torch.tensor(feats, dtype=torch.float32, device=self.device).view(1, -1)
        settings = self.controller_settings_snapshot or RootMotionSettings(1.8, 9.0, 11.0, 240.0, 0.24, 0.26, 1.0, 0.16)
        fps = max(1.0, float(self.clip.fps))
        dt = 1.0 / fps
        state = RootMotionState(
            np.zeros(3, dtype=np.float32),
            self.controller_velocity.astype(np.float32, copy=True),
            float(self.controller_yaw),
        )
        feats: list[float] = []
        turn = np.asarray([self.controller_turn_input, 0.0], dtype=np.float32)
        root_heading_inv = yaw_rotation_matrix_np(self.controller_yaw).T
        for k in range(1, int(self.cfg.future_window) + 1):
            state = integrate_controller_root_motion(
                state, self.controller_move_input, turn, dt, settings, movement_yaw=self.controller_move_yaw
            )
            scale_k = max(1e-6, float(k) * float(self.cfg.max_speed_scale_final))
            local_offset = state.offset @ root_heading_inv
            dx = max(-2.0, min(2.0, float(local_offset[0]) / scale_k))
            dz = max(-2.0, min(2.0, float(local_offset[2]) / scale_k))
            dyaw = float(state.yaw) - float(self.controller_yaw)
            feats.extend((dx, dz, math.cos(dyaw), math.sin(dyaw)))
        return torch.tensor(feats, dtype=torch.float32, device=self.device).view(1, -1)


class MotionGLFrame(OpenGLFrame):
    def __init__(self, *args, app: "ModelViewerApp", **kwargs):
        self.app = app
        super().__init__(*args, **kwargs)

    def initgl(self) -> None:
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glShadeModel(GL.GL_SMOOTH)
        GL.glEnable(GL.GL_LINE_SMOOTH)
        GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
        GL.glEnable(GL.GL_NORMALIZE)
        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, (0.22, 0.22, 0.22, 1.0))
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, (0.92, 0.92, 0.88, 1.0))
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)
        self.app.init_gl_resources()

    def redraw(self) -> None:
        self.app.render_gl()


class ModelViewerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        enable_precise_timers()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.precise_timers_enabled = True
        self.withdraw()
        self.title("Stepper Model Viewer")
        self.set_app_icon()
        self.minsize(920, 500)
        self.configure(bg=BG)

        self.actors: list[Actor] = []
        self.next_actor_id = 1
        self.selected_actor_id: int | None = None
        self.playing = False
        self.frame = 0.0
        self.last_tick = time.perf_counter()
        self.next_tick_time = self.last_tick + (1.0 / TARGET_RENDER_FPS)
        self.playback_accumulator = 0.0
        self.center = np.zeros(3, dtype=np.float32)
        self.extent = 1.0
        self.app_settings = load_app_settings()
        camera_settings = self.app_settings.get("camera", {}) if isinstance(self.app_settings.get("camera", {}), dict) else {}
        self.yaw = float(camera_settings.get("yaw", -0.75))
        self.pitch = float(camera_settings.get("pitch", -0.18))
        start_distance = max(MIN_CAMERA_DISTANCE, float(camera_settings.get("distance", self.extent * 2.9 / 1.45)))
        self.pending_startup_camera_distance = start_distance if "distance" in camera_settings else None
        self.zoom = max(1e-6, self.extent * 2.9 / start_distance)
        self.pan_x = 0.0
        self.pan_y = 0.0
        target = camera_settings.get("target", [0.0, 0.0, 0.0])
        self.pending_startup_camera_target: np.ndarray | None = None
        try:
            self.camera_pan = np.asarray(target, dtype=np.float32)
            if self.camera_pan.shape != (3,):
                self.camera_pan = np.zeros(3, dtype=np.float32)
            elif "target" in camera_settings:
                self.pending_startup_camera_target = self.camera_pan.copy()
        except (TypeError, ValueError):
            self.camera_pan = np.zeros(3, dtype=np.float32)
        self.camera_target_locked = bool(camera_settings.get("locked", False))
        self.penetration_mode = False
        self.view_axis: str | None = None
        self.camera_transition: dict[str, object] | None = None
        self.timeline_last_frame: int | None = None
        self.timeline_last_max_frame: int | None = None
        self.pan_drag_start: tuple[int, int] | None = None
        self.drag_start: tuple[int, int] | None = None
        self.pointer_down_start: tuple[int, int] | None = None
        self.pending_gizmo_clear_on_release = False
        self.drag_pan = False
        self.gizmo_visible = False
        self.gizmo_drag: str | None = None
        self.gizmo_last_mouse: tuple[int, int] | None = None
        self.gizmo_hits: dict[str, tuple[float, float, float, float]] = {}
        self.follow_cam_actor_id: int | None = None
        self.programmatic_tree_selection = False
        self.tree_select_source: str | None = None
        self.gl_modelview: np.ndarray | None = None
        self.gl_projection: np.ndarray | None = None
        self.gl_viewport: np.ndarray | None = None
        self.loading_inspector = False
        self.updating_timeline = False
        self.timeline_dragging = False
        self.render_image: Image.Image | None = None
        self.render_draw: ImageDraw.ImageDraw | None = None
        self.render_photo: ImageTk.PhotoImage | None = None
        self.render_font: ImageFont.ImageFont | None = None
        self.gl_lists: dict[str, int] = {}
        self.shadow_lists: dict[int, tuple[int, int, int]] = {}
        self.shadow_generation = 0
        self.animation_browser_paths: list[Path] = []
        self.fps_last_time = time.perf_counter()
        self.fps_frame_count = 0
        self.display_fps = 0.0
        self.overlay_texture_id = 0
        self.overlay_texture_text = ""
        self.overlay_texture_size = (0, 0)
        self.overlay_texture_uv = (1.0, 1.0)
        self.contact_overlay_texture_id = 0
        self.contact_overlay_texture_text = ""
        self.contact_overlay_texture_size = (0, 0)
        self.contact_overlay_texture_uv = (1.0, 1.0)
        self.xinput = load_xinput()
        self.controller_sample = ControllerSample()
        self.synthetic_controller_phase = 0.0

        self.speed_var = tk.DoubleVar(value=1.0)
        self.scale_var = tk.DoubleVar(value=1.0)
        self.frame_var = tk.IntVar(value=0)
        self.volumes_var = tk.BooleanVar(value=True)
        self.labels_var = tk.BooleanVar(value=False)
        self.show_foot_height_var = tk.BooleanVar(value=False)
        self.show_foot_contact_var = tk.BooleanVar(value=False)
        self.foot_contact_from_source_var = tk.BooleanVar(value=False)
        self.controller_detected_on_startup = self.poll_controller().connected
        self.controller_enabled_var = tk.BooleanVar(value=self.controller_detected_on_startup)
        self.show_trajectory_var = tk.BooleanVar(value=self.controller_detected_on_startup)
        self.device_var = tk.StringVar(value="cpu")
        self.name_var = tk.StringVar(value="")
        self.visible_var = tk.BooleanVar(value=True)
        self.offset_vars = [tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0)]
        self.pelvis_offset_vars = [tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0)]
        self.status_var = tk.StringVar(value="Open an NPZ or checkpoint to begin.")
        current_target = self.camera_target()
        self.settings_yaw_var = tk.DoubleVar(value=self.yaw)
        self.settings_pitch_var = tk.DoubleVar(value=self.pitch)
        self.settings_distance_var = tk.DoubleVar(value=self.camera_distance())
        self.settings_target_vars = [
            tk.DoubleVar(value=float(current_target[0])),
            tk.DoubleVar(value=float(current_target[1])),
            tk.DoubleVar(value=float(current_target[2])),
        ]
        self.settings_locked_var = tk.BooleanVar(value=self.camera_target_locked)
        collider_settings = self.app_settings.get("colliders", {}) if isinstance(self.app_settings.get("colliders", {}), dict) else {}
        foot_length_setting = float(collider_settings.get("foot_length", DEFAULT_FOOT_LENGTH))
        if foot_length_setting <= 1e-6:
            foot_length_setting = DEFAULT_FOOT_LENGTH
        self.foot_length_var = tk.DoubleVar(value=foot_length_setting)
        self.foot_width_var = tk.DoubleVar(value=float(collider_settings.get("foot_width", DEFAULT_FOOT_WIDTH)))
        self.foot_height_var = tk.DoubleVar(value=float(collider_settings.get("foot_height", DEFAULT_FOOT_HEIGHT)))
        self.toe_length_var = tk.DoubleVar(value=float(collider_settings.get("toe_length", DEFAULT_TOE_LENGTH)))
        self.toe_width_var = tk.DoubleVar(value=float(collider_settings.get("toe_width", DEFAULT_TOE_WIDTH)))
        self.toe_height_var = tk.DoubleVar(value=float(collider_settings.get("toe_height", DEFAULT_TOE_HEIGHT)))
        self.hand_length_var = tk.DoubleVar(value=float(collider_settings.get("hand_length", DEFAULT_HAND_LENGTH)))
        self.hand_width_var = tk.DoubleVar(value=float(collider_settings.get("hand_width", DEFAULT_HAND_WIDTH)))
        self.hand_height_var = tk.DoubleVar(value=float(collider_settings.get("hand_height", DEFAULT_HAND_HEIGHT)))
        controller_settings = self.app_settings.get("controller", {}) if isinstance(self.app_settings.get("controller", {}), dict) else {}
        self.controller_max_speed_var = tk.DoubleVar(value=float(controller_settings.get("max_speed_mps", 1.8)))
        self.controller_accel_response_var = tk.DoubleVar(value=float(controller_settings.get("acceleration_response", 9.0)))
        self.controller_decel_response_var = tk.DoubleVar(value=float(controller_settings.get("deceleration_response", 11.0)))
        self.controller_turn_rate_var = tk.DoubleVar(value=float(controller_settings.get("turn_rate_dps", 240.0)))
        self.controller_left_deadzone_var = tk.DoubleVar(value=float(controller_settings.get("left_deadzone", 0.24)))
        self.controller_right_deadzone_var = tk.DoubleVar(value=float(controller_settings.get("right_deadzone", 0.26)))
        self.trajectory_seconds_var = tk.DoubleVar(value=float(controller_settings.get("trajectory_seconds", 1.0)))
        self.trajectory_step_var = tk.DoubleVar(value=float(controller_settings.get("trajectory_step_seconds", 0.16)))
        startup_settings = self.app_settings.get("startup", {}) if isinstance(self.app_settings.get("startup", {}), dict) else {}
        startup_npz = str(startup_settings.get("source_npz_path", "")).strip()
        if not startup_npz:
            latest_npz = newest_npz()
            startup_npz = str(latest_npz) if latest_npz is not None else ""
        self.startup_npz_path_var = tk.StringVar(value=startup_npz)
        animation_settings = (
            self.app_settings.get("animation_browser", {})
            if isinstance(self.app_settings.get("animation_browser", {}), dict)
            else {}
        )
        animation_folder = str(animation_settings.get("folder_path", "")).strip()
        if not animation_folder:
            animation_folder = str(DEFAULT_ANIMATION_LIBRARY_DIR)
        self.animation_folder_var = tk.StringVar(value=animation_folder)
        self.settings_status_var = tk.StringVar(value="")

        self._configure_style()
        self._build_ui()
        self.apply_startup_geometry()
        self.deiconify()
        self.after(250, self.keep_inside_work_area)
        self.after(80, self.refresh_animation_browser)
        self.after(120, self.auto_load_latest)
        self.schedule_next_tick()

    def on_close(self) -> None:
        self.destroy()

    def destroy(self) -> None:
        if getattr(self, "precise_timers_enabled", False):
            disable_precise_timers()
            self.precise_timers_enabled = False
        super().destroy()

    def set_app_icon(self) -> None:
        if not APP_ICON.exists():
            return
        try:
            self.iconbitmap(default=str(APP_ICON))
        except tk.TclError:
            pass

    def apply_startup_geometry(self) -> None:
        self.update_idletasks()
        left, top, right, bottom = usable_work_area(self)
        work_w = max(920, right - left)
        work_h = max(500, bottom - top)
        margin = min(max(24, int(round(self.winfo_fpixels("1c")))), max(0, (work_h - 500) // 2))
        frame_allowance = max(36, int(round(self.winfo_fpixels("1c"))) + 1)
        width = min(1360, max(920, work_w - 2 * margin))
        height = max(500, work_h - 2 * margin - frame_allowance)
        x = left + max(0, (work_w - width) // 2)
        y = top + margin
        self.maxsize(work_w, work_h)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def keep_inside_work_area(self) -> None:
        self.update_idletasks()
        left, top, right, bottom = usable_work_area(self)
        width = min(max(self.winfo_width(), 920), max(920, right - left))
        height = min(max(self.winfo_height(), 500), max(500, bottom - top))
        x = min(max(self.winfo_x(), left), max(left, right - width))
        y = min(max(self.winfo_y(), top), max(top, bottom - height))
        if (x, y, width, height) != (self.winfo_x(), self.winfo_y(), self.winfo_width(), self.winfo_height()):
            self.geometry(f"{width}x{height}+{x}+{y}")

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=PANEL, foreground=TEXT, fieldbackground=PANEL_2)
        style.configure("TFrame", background=PANEL)
        style.configure("Toolbar.TFrame", background=PANEL_2)
        style.configure("TLabel", background=PANEL, foreground=TEXT)
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED)
        style.configure("TButton", background=PANEL_2, foreground=TEXT, bordercolor=LINE)
        style.configure("TCheckbutton", background=PANEL, foreground=TEXT)
        style.configure("Treeview", background="#14171c", foreground=TEXT, fieldbackground="#14171c", rowheight=24)
        style.configure("Treeview.Heading", background=PANEL_2, foreground=TEXT)

    def _build_ui(self) -> None:
        root = ttk.Frame(self)
        root.pack(fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(root, style="Toolbar.TFrame", padding=(8, 6))
        toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(toolbar, text="Open File...", command=self.open_file).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="Play/Pause", command=self.toggle_play).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Stop", command=self.stop_playback).pack(side=tk.LEFT, padx=3)
        ttk.Label(toolbar, text="Speed", style="Muted.TLabel").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Spinbox(toolbar, from_=0.1, to=4.0, increment=0.1, textvariable=self.speed_var, width=5).pack(side=tk.LEFT)
        ttk.Label(toolbar, text="Scale", style="Muted.TLabel").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Spinbox(toolbar, from_=0.05, to=10.0, increment=0.05, textvariable=self.scale_var, width=5).pack(side=tk.LEFT)
        ttk.Label(toolbar, text="Device", style="Muted.TLabel").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Combobox(toolbar, values=("cpu", "cuda"), textvariable=self.device_var, width=6, state="readonly").pack(side=tk.LEFT)
        ttk.Checkbutton(toolbar, text="Colliders", variable=self.volumes_var, command=self.draw).pack(side=tk.LEFT, padx=(12, 4))
        ttk.Checkbutton(toolbar, text="Foot Height", variable=self.show_foot_height_var, command=self.draw).pack(side=tk.LEFT, padx=4)
        contact_controls = ttk.Frame(toolbar, style="Toolbar.TFrame")
        contact_controls.pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(contact_controls, text="Foot Contact", variable=self.show_foot_contact_var, command=self.draw).pack(
            anchor=tk.W
        )
        ttk.Checkbutton(contact_controls, text="From Source", variable=self.foot_contact_from_source_var, command=self.draw).pack(
            anchor=tk.W
        )
        controller_controls = ttk.Frame(toolbar, style="Toolbar.TFrame")
        controller_controls.pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(controller_controls, text="Controller", variable=self.controller_enabled_var, command=self.on_controller_toggle).pack(
            anchor=tk.W
        )
        ttk.Checkbutton(controller_controls, text="Show Trajectory", variable=self.show_trajectory_var, command=self.draw).pack(
            anchor=tk.W
        )
        ttk.Checkbutton(toolbar, text="Labels", variable=self.labels_var, command=self.draw).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Reset View", command=self.reset_camera).pack(side=tk.LEFT, padx=(12, 4))
        self.follow_cam_button = ttk.Button(toolbar, text="Follow Cam", command=self.follow_camera)
        self.follow_cam_button.pack(side=tk.LEFT, padx=(6, 4))
        ttk.Button(toolbar, text="Penetration Cam", command=self.penetration_camera).pack(side=tk.LEFT, padx=(6, 4))

        bottom = ttk.Frame(root, style="Toolbar.TFrame", padding=(8, 4))
        bottom.pack(side=tk.TOP, fill=tk.X)
        self.timeline_bar = bottom
        ttk.Label(bottom, text="Frame", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        self.timeline = tk.Canvas(bottom, height=26, bg=PANEL_2, highlightthickness=0, cursor="hand2")
        self.timeline.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.timeline.bind("<ButtonPress-1>", self.on_timeline_down)
        self.timeline.bind("<B1-Motion>", self.on_timeline_drag)
        self.timeline.bind("<ButtonRelease-1>", self.on_timeline_up)
        self.timeline.bind("<Configure>", lambda _event: self.draw_timeline())
        self.frame_label = ttk.Label(bottom, text="0 / 0", style="Muted.TLabel")
        self.frame_label.pack(side=tk.LEFT)

        main = ttk.Frame(root)
        main.pack(fill=tk.BOTH, expand=True)

        viewer = ttk.Frame(main)
        viewer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        viewer.columnconfigure(0, weight=1)
        viewer.rowconfigure(0, weight=1)

        self.canvas = MotionGLFrame(viewer, app=self)
        self.canvas.grid(row=0, column=0, sticky=tk.NSEW)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<ButtonPress-3>", self.on_pan_down)
        self.canvas.bind("<B3-Motion>", self.on_pan_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_pan_up)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Configure>", lambda _event: self.draw())
        self.axis_canvas = tk.Canvas(viewer, width=86, height=86, bg="#0b0e12", highlightthickness=1, highlightbackground=LINE, cursor="hand2")
        self.axis_canvas.place(relx=1.0, x=-14, y=14, anchor=tk.NE)
        self.axis_canvas.bind("<ButtonPress-1>", self.on_axis_gizmo_click)

        side = ttk.Frame(main, width=344, padding=(8, 6))
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)

        tabs = ttk.Notebook(side)
        tabs.pack(fill=tk.BOTH, expand=True)
        actors_tab = ttk.Frame(tabs, padding=(0, 0))
        settings_tab = ttk.Frame(tabs, padding=(8, 8))
        tabs.add(actors_tab, text="Actors")
        tabs.add(settings_tab, text="Settings")

        self.tree = ttk.Treeview(actors_tab, columns=("kind", "status"), show="tree headings", height=9)
        self.tree.heading("#0", text="Name")
        self.tree.heading("kind", text="Type")
        self.tree.heading("status", text="State")
        self.tree.column("#0", width=150)
        self.tree.column("kind", width=64, anchor=tk.CENTER)
        self.tree.column("status", width=98, anchor=tk.W)
        self.tree.pack(fill=tk.BOTH, expand=True, pady=(4, 6))
        self.tree.bind("<ButtonPress-1>", self.on_tree_button_press)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.tree.bind("<Double-1>", self.on_tree_double_click)
        actor_buttons = ttk.Frame(actors_tab)
        actor_buttons.pack(fill=tk.X, pady=(0, 7))
        ttk.Button(actor_buttons, text="Remove", command=self.remove_selected).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(actor_buttons, text="Duplicate", command=self.duplicate_selected).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        animation_browser = ttk.Frame(actors_tab)
        animation_browser.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(animation_browser, text="Animations").pack(anchor=tk.W)
        animation_list_frame = ttk.Frame(animation_browser)
        animation_list_frame.pack(fill=tk.X, pady=(3, 3))
        self.animation_listbox = tk.Listbox(
            animation_list_frame,
            height=7,
            bg="#14171c",
            fg=TEXT,
            selectbackground="#4f718e",
            selectforeground=TEXT,
            highlightthickness=1,
            highlightbackground=LINE,
            relief=tk.FLAT,
            exportselection=False,
        )
        animation_scroll = ttk.Scrollbar(animation_list_frame, orient=tk.VERTICAL, command=self.animation_listbox.yview)
        self.animation_listbox.configure(yscrollcommand=animation_scroll.set)
        self.animation_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        animation_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.animation_listbox.bind("<Double-1>", self.load_selected_animation)
        self.animation_listbox.bind("<Return>", self.load_selected_animation)
        animation_tools = ttk.Frame(animation_browser)
        animation_tools.pack(fill=tk.X)
        ttk.Button(animation_tools, text="Load", command=self.load_selected_animation).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(animation_tools, text="Folder...", command=self.browse_animation_folder).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0)
        )
        ttk.Button(animation_tools, text="Refresh", command=self.refresh_animation_browser).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0)
        )
        ttk.Label(animation_browser, textvariable=self.animation_folder_var, style="Muted.TLabel", wraplength=316).pack(
            anchor=tk.W, pady=(3, 0)
        )

        ttk.Separator(actors_tab).pack(fill=tk.X, pady=5)
        ttk.Label(actors_tab, text="Selected Actor").pack(anchor=tk.W)
        inspector = ttk.Frame(actors_tab)
        inspector.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(inspector, text="Name", style="Muted.TLabel").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(inspector, textvariable=self.name_var).grid(row=0, column=1, sticky=tk.EW, pady=2)
        ttk.Checkbutton(inspector, text="Visible", variable=self.visible_var, command=self.apply_inspector).grid(
            row=1, column=1, sticky=tk.W, pady=2
        )
        ttk.Label(inspector, text="Start", style="Muted.TLabel").grid(row=2, column=0, sticky=tk.W, pady=2)
        vector_row = ttk.Frame(inspector)
        vector_row.grid(row=2, column=1, sticky=tk.EW, pady=2)
        for index, label in enumerate(("X", "Y", "Z")):
            ttk.Label(vector_row, text=label, style="Muted.TLabel").pack(side=tk.LEFT, padx=(0 if index == 0 else 6, 2))
            ttk.Spinbox(
                vector_row,
                from_=-1000.0,
                to=1000.0,
                increment=0.1,
                textvariable=self.offset_vars[index],
                width=6,
                command=self.apply_inspector,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(inspector, text="Pelvis F0", style="Muted.TLabel").grid(row=3, column=0, sticky=tk.W, pady=2)
        pelvis_row = ttk.Frame(inspector)
        pelvis_row.grid(row=3, column=1, sticky=tk.EW, pady=2)
        for index, label in enumerate(("X", "Y", "Z")):
            ttk.Label(pelvis_row, text=label, style="Muted.TLabel").pack(side=tk.LEFT, padx=(0 if index == 0 else 6, 2))
            ttk.Spinbox(
                pelvis_row,
                from_=-10.0,
                to=10.0,
                increment=0.01,
                textvariable=self.pelvis_offset_vars[index],
                width=6,
                command=self.apply_inspector,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        inspector.columnconfigure(1, weight=1)
        self.source_button = ttk.Button(actors_tab, text="Set Model Source NPZ...", command=self.set_selected_source_npz)
        self.source_button.pack(fill=tk.X, pady=(4, 3))
        ttk.Label(actors_tab, textvariable=self.status_var, style="Muted.TLabel", wraplength=316).pack(anchor=tk.W, pady=(6, 0))

        self.build_settings_tab(settings_tab)

        self.bind_class("StepperSpacePlay", "<space>", self.on_space_play)
        self.install_space_play_bindtag(self)
        self.bind("<Left>", lambda _event: self.step_frame(-1))
        self.bind("<Right>", lambda _event: self.step_frame(1))
        self.bind("<Delete>", self.on_delete_selected)
        for var in (self.name_var, self.visible_var, *self.offset_vars, *self.pelvis_offset_vars):
            var.trace_add("write", lambda *_args: self.apply_inspector())
        self.scale_var.trace_add("write", lambda *_args: self.draw())
        for var in (
            self.foot_length_var,
            self.foot_width_var,
            self.foot_height_var,
            self.toe_length_var,
            self.toe_width_var,
            self.toe_height_var,
            self.hand_length_var,
            self.hand_width_var,
            self.hand_height_var,
        ):
            var.trace_add("write", lambda *_args: self.on_collider_setting_changed())

    def on_space_play(self, _event: tk.Event) -> str:
        self.toggle_play()
        return "break"

    def install_space_play_bindtag(self, widget: tk.Widget) -> None:
        tag = "StepperSpacePlay"
        bindtags = widget.bindtags()
        if tag not in bindtags:
            widget.bindtags((tag, *bindtags))
        for child in widget.winfo_children():
            self.install_space_play_bindtag(child)

    def build_settings_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        scroll_canvas = tk.Canvas(parent, bg=PANEL, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scroll_canvas.grid(row=0, column=0, sticky=tk.NSEW)
        scrollbar.grid(row=0, column=1, sticky=tk.NS)
        content = ttk.Frame(scroll_canvas)
        content_window = scroll_canvas.create_window((0, 0), window=content, anchor=tk.NW)

        def sync_scroll_region(_event: tk.Event | None = None) -> None:
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

        def sync_content_width(event: tk.Event) -> None:
            scroll_canvas.itemconfigure(content_window, width=event.width)

        def on_settings_wheel(event: tk.Event) -> str:
            scroll_canvas.yview_scroll(int(-event.delta / 120), "units")
            return "break"

        content.bind("<Configure>", sync_scroll_region)
        scroll_canvas.bind("<Configure>", sync_content_width)
        scroll_canvas.bind("<MouseWheel>", on_settings_wheel)
        content.bind("<MouseWheel>", on_settings_wheel)

        ttk.Label(content, text="Camera Start").pack(anchor=tk.W)
        form = ttk.Frame(content)
        form.pack(fill=tk.X, pady=(8, 6))
        rows = [
            ("Yaw", self.settings_yaw_var),
            ("Pitch", self.settings_pitch_var),
            ("Distance", self.settings_distance_var),
            ("Target X", self.settings_target_vars[0]),
            ("Target Y", self.settings_target_vars[1]),
            ("Target Z", self.settings_target_vars[2]),
        ]
        for row, (label, var) in enumerate(rows):
            ttk.Label(form, text=label, style="Muted.TLabel").grid(row=row, column=0, sticky=tk.W, pady=3)
            ttk.Spinbox(form, from_=-10000.0, to=10000.0, increment=0.05, textvariable=var, width=10).grid(
                row=row, column=1, sticky=tk.EW, pady=3
            )
        form.columnconfigure(1, weight=1)
        ttk.Checkbutton(content, text="Lock Target", variable=self.settings_locked_var).pack(anchor=tk.W, pady=(2, 8))
        ttk.Button(content, text="Use Current Camera", command=self.capture_current_camera_settings).pack(fill=tk.X, pady=(2, 6))

        ttk.Separator(content).pack(fill=tk.X, pady=(10, 8))
        ttk.Label(content, text="Startup Model Source").pack(anchor=tk.W)
        startup_form = ttk.Frame(content)
        startup_form.pack(fill=tk.X, pady=(8, 6))
        ttk.Label(startup_form, text="NPZ Path", style="Muted.TLabel").grid(row=0, column=0, sticky=tk.W, pady=3)
        ttk.Entry(startup_form, textvariable=self.startup_npz_path_var).grid(row=0, column=1, sticky=tk.EW, pady=3)
        startup_form.columnconfigure(1, weight=1)
        startup_buttons = ttk.Frame(content)
        startup_buttons.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(startup_buttons, text="Browse NPZ...", command=self.browse_startup_npz).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(startup_buttons, text="Use Latest", command=self.use_latest_startup_npz).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0)
        )

        ttk.Separator(content).pack(fill=tk.X, pady=(10, 8))
        ttk.Label(content, text="Controller Root Motion").pack(anchor=tk.W)
        controller_form = ttk.Frame(content)
        controller_form.pack(fill=tk.X, pady=(8, 6))
        controller_rows = [
            ("Max Speed", self.controller_max_speed_var, 0.05, 8.0, 0.05),
            ("Accel Response", self.controller_accel_response_var, 0.1, 40.0, 0.1),
            ("Decel Response", self.controller_decel_response_var, 0.1, 40.0, 0.1),
            ("Turn Rate", self.controller_turn_rate_var, 1.0, 720.0, 5.0),
            ("Left Deadzone", self.controller_left_deadzone_var, 0.0, 0.8, 0.01),
            ("Right Deadzone", self.controller_right_deadzone_var, 0.0, 0.8, 0.01),
            ("Trajectory Sec", self.trajectory_seconds_var, 0.1, 4.0, 0.05),
            ("Trajectory Step", self.trajectory_step_var, 0.04, 0.5, 0.01),
        ]
        for row, (label, var, start, end, step) in enumerate(controller_rows):
            ttk.Label(controller_form, text=label, style="Muted.TLabel").grid(row=row, column=0, sticky=tk.W, pady=3)
            ttk.Spinbox(controller_form, from_=start, to=end, increment=step, textvariable=var, width=10).grid(
                row=row, column=1, sticky=tk.EW, pady=3
            )
        controller_form.columnconfigure(1, weight=1)

        ttk.Separator(content).pack(fill=tk.X, pady=(10, 8))
        ttk.Label(content, text="Foot / Toe / Hand Colliders").pack(anchor=tk.W)
        collider_form = ttk.Frame(content)
        collider_form.pack(fill=tk.X, pady=(8, 6))
        collider_rows = [
            ("Foot Length", self.foot_length_var),
            ("Foot Width", self.foot_width_var),
            ("Foot Height", self.foot_height_var),
            ("Toe Length", self.toe_length_var),
            ("Toe Width", self.toe_width_var),
            ("Toe Height", self.toe_height_var),
            ("Hand Length", self.hand_length_var),
            ("Hand Width", self.hand_width_var),
            ("Hand Height", self.hand_height_var),
        ]
        for row, (label, var) in enumerate(collider_rows):
            ttk.Label(collider_form, text=label, style="Muted.TLabel").grid(row=row, column=0, sticky=tk.W, pady=3)
            ttk.Spinbox(collider_form, from_=0.0, to=1.0, increment=0.005, textvariable=var, width=10).grid(
                row=row, column=1, sticky=tk.EW, pady=3
            )
        collider_form.columnconfigure(1, weight=1)

        footer = ttk.Frame(parent)
        footer.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(8, 0))
        ttk.Button(footer, text="Save", command=self.save_settings).pack(fill=tk.X)
        ttk.Label(footer, textvariable=self.settings_status_var, style="Muted.TLabel", wraplength=300).pack(
            anchor=tk.W, pady=(6, 0)
        )

    def capture_current_camera_settings(self) -> None:
        target = self.camera_target()
        self.settings_yaw_var.set(float(self.yaw))
        self.settings_pitch_var.set(float(self.pitch))
        self.settings_distance_var.set(float(self.camera_distance()))
        for i, var in enumerate(self.settings_target_vars):
            var.set(float(target[i]))
        self.settings_locked_var.set(True)
        self.settings_status_var.set("Captured current camera.")

    def browse_startup_npz(self) -> None:
        path_text = filedialog.askopenfilename(
            title="Startup source NPZ",
            initialdir=str(DEFAULT_NPZ_DIR if DEFAULT_NPZ_DIR.exists() else PROJECT_ROOT),
            filetypes=(("NPZ motion", "*.npz"), ("All files", "*.*")),
        )
        if path_text:
            self.startup_npz_path_var.set(str(Path(path_text).resolve()))

    def use_latest_startup_npz(self) -> None:
        latest = newest_npz()
        if latest is None:
            self.settings_status_var.set("No NPZ found in npz_final.")
            return
        self.startup_npz_path_var.set(str(latest))
        self.settings_status_var.set("Startup NPZ set to latest npz_final file.")

    def startup_source_npz(self) -> Path | None:
        path_text = self.startup_npz_path_var.get().strip()
        if path_text:
            path = resolve_path(path_text)
            if path.exists():
                return path
        return newest_npz()

    def save_settings(self) -> None:
        try:
            target = [float(var.get()) for var in self.settings_target_vars]
            camera = {
                "yaw": float(self.settings_yaw_var.get()),
                "pitch": float(self.settings_pitch_var.get()),
                "distance": max(MIN_CAMERA_DISTANCE, float(self.settings_distance_var.get())),
                "target": target,
                "locked": bool(self.settings_locked_var.get()),
            }
            colliders = {
                "foot_length": max(0.0, float(self.foot_length_var.get())),
                "foot_width": max(0.001, float(self.foot_width_var.get())),
                "foot_height": max(0.001, float(self.foot_height_var.get())),
                "toe_length": max(0.001, float(self.toe_length_var.get())),
                "toe_width": max(0.001, float(self.toe_width_var.get())),
                "toe_height": max(0.001, float(self.toe_height_var.get())),
                "hand_length": max(0.001, float(self.hand_length_var.get())),
                "hand_width": max(0.001, float(self.hand_width_var.get())),
                "hand_height": max(0.001, float(self.hand_height_var.get())),
            }
            startup_path = self.startup_npz_path_var.get().strip()
            if startup_path:
                startup_path = str(resolve_path(startup_path))
            startup = {"source_npz_path": startup_path}
            animation_folder = self.animation_folder_var.get().strip()
            if animation_folder:
                animation_folder = str(resolve_path(animation_folder))
            animation_browser = {"folder_path": animation_folder}
            controller = {
                "max_speed_mps": max(0.05, float(self.controller_max_speed_var.get())),
                "acceleration_response": max(0.1, float(self.controller_accel_response_var.get())),
                "deceleration_response": max(0.1, float(self.controller_decel_response_var.get())),
                "turn_rate_dps": max(1.0, float(self.controller_turn_rate_var.get())),
                "left_deadzone": max(0.0, min(0.8, float(self.controller_left_deadzone_var.get()))),
                "right_deadzone": max(0.0, min(0.8, float(self.controller_right_deadzone_var.get()))),
                "trajectory_seconds": max(0.1, float(self.trajectory_seconds_var.get())),
                "trajectory_step_seconds": max(0.04, float(self.trajectory_step_var.get())),
            }
        except tk.TclError:
            self.settings_status_var.set("Invalid setting.")
            return
        self.app_settings["camera"] = camera
        self.app_settings["colliders"] = colliders
        self.app_settings["startup"] = startup
        self.app_settings["animation_browser"] = animation_browser
        self.app_settings["controller"] = controller
        try:
            write_app_settings(self.app_settings)
        except OSError as exc:
            self.settings_status_var.set(f"Save failed: {exc}")
            return
        self.settings_status_var.set(f"Saved to {SETTINGS_PATH.name}.")

    def invalidate_shadow_cache(self) -> None:
        self.shadow_generation += 1

    def on_collider_setting_changed(self) -> None:
        self.invalidate_shadow_cache()
        self.draw()

    def open_file(self) -> None:
        path_text = filedialog.askopenfilename(
            title="Open NPZ or checkpoint",
            initialdir=str(DEFAULT_RUNS_DIR if DEFAULT_RUNS_DIR.exists() else PROJECT_ROOT),
            filetypes=(
                ("Motion or checkpoint", "*.npz *.pt *.pth"),
                ("NPZ motion", "*.npz"),
                ("Torch checkpoint", "*.pt *.pth"),
                ("All files", "*.*"),
            ),
        )
        if not path_text:
            return
        path = Path(path_text)
        try:
            if path.suffix.lower() == ".npz":
                self.add_npz_actor(path)
            elif path.suffix.lower() in {".pt", ".pth"}:
                self.add_checkpoint_actor(path)
            else:
                messagebox.showerror("Unsupported file", "Open an .npz motion file or a .pt/.pth checkpoint.")
        except Exception as exc:
            messagebox.showerror("Open failed", str(exc))

    def animation_folder(self) -> Path:
        path_text = self.animation_folder_var.get().strip()
        if not path_text:
            return DEFAULT_ANIMATION_LIBRARY_DIR
        return resolve_path(path_text)

    def refresh_animation_browser(self) -> None:
        if not hasattr(self, "animation_listbox"):
            return
        folder = self.animation_folder()
        self.animation_browser_paths = []
        self.animation_listbox.delete(0, tk.END)
        if not folder.exists():
            self.animation_listbox.insert(tk.END, "(folder not found)")
            return
        try:
            paths = sorted(folder.glob("*.npz"), key=lambda item: item.name.lower())
        except OSError as exc:
            self.animation_listbox.insert(tk.END, f"(cannot read folder: {exc})")
            return
        self.animation_browser_paths = paths
        if not paths:
            self.animation_listbox.insert(tk.END, "(no npz files)")
            return
        for path in paths:
            self.animation_listbox.insert(tk.END, path.stem)

    def browse_animation_folder(self) -> None:
        initial = self.animation_folder()
        path_text = filedialog.askdirectory(
            title="Animation folder",
            initialdir=str(initial if initial.exists() else PROJECT_ROOT),
        )
        if not path_text:
            return
        self.animation_folder_var.set(str(Path(path_text).resolve()))
        self.refresh_animation_browser()

    def selected_animation_path(self) -> Path | None:
        if not hasattr(self, "animation_listbox"):
            return None
        selection = self.animation_listbox.curselection()
        if not selection:
            return None
        index = int(selection[0])
        if index < 0 or index >= len(self.animation_browser_paths):
            return None
        return self.animation_browser_paths[index]

    def load_selected_animation(self, _event: tk.Event | None = None) -> str | None:
        path = self.selected_animation_path()
        if path is None:
            return "break" if _event is not None else None
        try:
            actor = self.selected_actor()
            if actor is not None and actor.kind == "model":
                self.load_model_source(actor, path)
                self.refresh_tree(select_id=actor.actor_id)
                self.update_timeline()
                self.update_bounds()
                self.invalidate_shadow_cache()
                self.status_var.set(f"{actor.checkpoint_path.name if actor.checkpoint_path else 'checkpoint'}\n{path.name}\n{actor.status}")
                self.draw()
            else:
                self.add_npz_actor(path)
        except Exception as exc:
            messagebox.showerror("Animation load failed", str(exc))
        return "break" if _event is not None else None

    def auto_load_latest(self) -> None:
        if self.actors:
            return
        checkpoint = newest_best_checkpoint()
        source = self.startup_source_npz()
        if checkpoint is None or source is None:
            return
        try:
            actor = self.add_checkpoint_actor(checkpoint, source_npz_path=source)
            self.status_var.set(f"Auto-loaded\n{checkpoint.name}\n{source.name}")
        except Exception as exc:
            self.status_var.set(f"Auto-load skipped: {exc}")

    def add_npz_actor(self, path: Path) -> Actor:
        actor = Actor(
            actor_id=self.next_actor_id,
            kind="npz",
            name=actor_basename(path),
            color=GT_BLUE,
        )
        self.next_actor_id += 1
        actor.load_npz(path)
        self.actors.append(actor)
        self.refresh_tree(select_id=actor.actor_id)
        self.update_timeline()
        self.update_bounds()
        self.invalidate_shadow_cache()
        self.draw()
        return actor

    def add_checkpoint_actor(self, path: Path, source_npz_path: Path | None = None) -> Actor:
        actor = Actor(
            actor_id=self.next_actor_id,
            kind="model",
            name=actor_basename(path),
            color=PRED_ORANGE,
            checkpoint_path=path,
            status="needs source NPZ",
        )
        self.next_actor_id += 1
        self.actors.append(actor)
        source_path = source_npz_path or self.startup_source_npz()
        if source_path is None:
            source = self.selected_actor()
            if source is not None and source.kind == "npz" and source.npz_path is not None:
                source_path = source.npz_path
        if source_path is not None:
            self.load_model_source(actor, source_path)
        self.refresh_tree(select_id=actor.actor_id)
        self.update_timeline()
        self.update_bounds()
        self.invalidate_shadow_cache()
        self.draw()
        return actor

    def duplicate_selected(self) -> None:
        actor = self.selected_actor()
        if actor is None:
            return
        clone = Actor(
            actor_id=self.next_actor_id,
            kind=actor.kind,
            name=f"{actor.name} copy",
            color=actor.color,
            visible=actor.visible,
            offset=actor.offset + np.asarray([0.75, 0.0, 0.0], dtype=np.float32),
            initial_pelvis_offset=actor.initial_pelvis_offset.copy(),
            npz_path=actor.npz_path,
            checkpoint_path=actor.checkpoint_path,
            status=actor.status,
        )
        self.next_actor_id += 1
        try:
            if clone.kind == "npz" and clone.npz_path is not None:
                clone.load_npz(clone.npz_path)
            elif clone.kind == "model" and clone.checkpoint_path is not None and actor.source_npz_path is not None:
                self.load_model_source(clone, actor.source_npz_path)
        except Exception as exc:
            messagebox.showerror("Duplicate failed", str(exc))
            return
        self.actors.append(clone)
        self.refresh_tree(select_id=clone.actor_id)
        self.update_timeline()
        self.update_bounds()
        self.invalidate_shadow_cache()
        self.draw()

    def remove_selected(self) -> None:
        actor = self.selected_actor()
        if actor is None:
            return
        camera_target = self.camera_target().copy()
        camera_distance = self.camera_distance()
        yaw = self.yaw
        pitch = self.pitch
        view_axis = self.view_axis
        pan_y = self.pan_y
        target_locked = self.camera_target_locked
        penetration_mode = self.penetration_mode
        self.actors = [item for item in self.actors if item.actor_id != actor.actor_id]
        if self.follow_cam_actor_id == actor.actor_id:
            self.set_follow_cam_actor(None)
        self.selected_actor_id = None
        self.refresh_tree()
        self.update_timeline()
        self.update_bounds()
        self.yaw = yaw
        self.pitch = pitch
        self.view_axis = view_axis
        self.pan_y = pan_y
        self.camera_target_locked = target_locked
        self.penetration_mode = penetration_mode
        self.set_camera_target(camera_target)
        self.set_camera_distance(camera_distance)
        self.invalidate_shadow_cache()
        self.draw()

    def on_delete_selected(self, _event: tk.Event) -> str | None:
        focus = self.focus_get()
        if focus is not None:
            widget_class = str(focus.winfo_class())
            if widget_class in {"Entry", "TEntry", "Spinbox", "TSpinbox"}:
                return None
        if self.selected_actor() is None:
            return None
        self.remove_selected()
        return "break"

    def set_selected_source_npz(self) -> None:
        actor = self.selected_actor()
        if actor is None or actor.kind != "model":
            return
        path_text = filedialog.askopenfilename(
            title="Set source NPZ",
            initialdir=str(DEFAULT_NPZ_DIR if DEFAULT_NPZ_DIR.exists() else PROJECT_ROOT),
            filetypes=(("NPZ motion", "*.npz"), ("All files", "*.*")),
        )
        if not path_text:
            return
        try:
            self.load_model_source(actor, Path(path_text))
        except Exception as exc:
            messagebox.showerror("Source failed", str(exc))
            return
        self.refresh_tree(select_id=actor.actor_id)
        self.update_timeline()
        self.update_bounds()
        self.invalidate_shadow_cache()
        self.draw()

    def load_model_source(self, actor: Actor, source_npz_path: Path) -> None:
        if actor.checkpoint_path is None:
            raise ValueError("Model actor has no checkpoint path.")
        device_name = self.device_var.get()
        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
            self.device_var.set("cpu")
        actor.load_checkpoint(actor.checkpoint_path, source_npz_path, torch.device(device_name))

    def selected_actor(self) -> Actor | None:
        if self.selected_actor_id is None:
            return None
        for actor in self.actors:
            if actor.actor_id == self.selected_actor_id:
                return actor
        return None

    def refresh_tree(self, select_id: int | None = None) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for actor in self.actors:
            icon = "[x]" if actor.visible else "[ ]"
            state = actor.status
            self.tree.insert(
                "",
                tk.END,
                iid=str(actor.actor_id),
                text=f"{icon} {actor.name}",
                values=(actor.kind, state),
            )
        if select_id is not None:
            self.programmatic_tree_selection = True
            self.tree_select_source = "program"
            self.tree.selection_set(str(select_id))
            self.tree.focus(str(select_id))
            self.selected_actor_id = select_id
            self.load_inspector()
            self.gizmo_visible = False
            self.gizmo_drag = None
            self.after(500, self.finish_programmatic_tree_selection)

    def finish_programmatic_tree_selection(self) -> None:
        self.programmatic_tree_selection = False
        if self.tree_select_source == "program":
            self.tree_select_source = None

    def clear_tree_select_source(self, expected: str | None) -> None:
        if self.tree_select_source == expected:
            self.tree_select_source = None

    def clear_actor_selection(self, redraw: bool = True) -> None:
        self.selected_actor_id = None
        self.tree.selection_remove(self.tree.selection())
        self.tree.focus("")
        self.gizmo_visible = False
        self.gizmo_drag = None
        self.gizmo_hits.clear()
        self.load_inspector()
        if redraw:
            self.draw()

    def on_tree_button_press(self, event: tk.Event) -> None:
        if not self.tree.identify_row(event.y):
            self.tree_select_source = None
            self.programmatic_tree_selection = False
            self.clear_actor_selection()
            return
        self.tree_select_source = "user"
        self.programmatic_tree_selection = False

    def on_tree_select(self, _event: tk.Event) -> None:
        selection = self.tree.selection()
        self.selected_actor_id = int(selection[0]) if selection else None
        self.load_inspector()
        source = self.tree_select_source
        if source == "program" or self.programmatic_tree_selection:
            self.gizmo_visible = False
            self.gizmo_drag = None
        else:
            self.update_gizmo_visibility()
        if source in {"user", "viewport"}:
            self.after(150, lambda source=source: self.clear_tree_select_source(source))
        self.draw()

    def update_gizmo_visibility(self) -> None:
        actor = self.selected_actor()
        self.gizmo_visible = bool(actor is not None and actor.clip is not None and not self.playing)

    def on_tree_double_click(self, event: tk.Event) -> None:
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            self.selected_actor_id = int(item)
            self.load_inspector()
        actor = self.selected_actor()
        if actor is not None:
            self.focus_camera_on_actor(actor)

    def load_inspector(self) -> None:
        self.loading_inspector = True
        actor = self.selected_actor()
        if actor is None:
            self.name_var.set("")
            self.visible_var.set(True)
            for var in self.offset_vars:
                var.set(0.0)
            for var in self.pelvis_offset_vars:
                var.set(0.0)
            self.source_button.state(["disabled"])
            self.status_var.set("No actor selected.")
            self.loading_inspector = False
            return
        self.name_var.set(actor.name)
        self.visible_var.set(actor.visible)
        for i, var in enumerate(self.offset_vars):
            var.set(float(actor.offset[i]))
        for i, var in enumerate(self.pelvis_offset_vars):
            var.set(float(actor.initial_pelvis_offset[i] if actor.kind == "model" else 0.0))
        if actor.kind == "model":
            self.source_button.state(["!disabled"])
            checkpoint = actor.checkpoint_path.name if actor.checkpoint_path else "checkpoint"
            source = actor.source_npz_path.name if actor.source_npz_path else "no source NPZ"
            self.status_var.set(f"{checkpoint}\n{source}\n{actor.status}")
        else:
            self.source_button.state(["disabled"])
            self.status_var.set(f"{actor.npz_path}\n{actor.status}")
        self.loading_inspector = False

    def apply_inspector(self) -> None:
        if self.loading_inspector:
            return
        actor = self.selected_actor()
        if actor is None:
            return
        actor.name = self.name_var.get().strip() or actor.name
        actor.visible = bool(self.visible_var.get())
        try:
            actor.offset = np.asarray([var.get() for var in self.offset_vars], dtype=np.float32)
            pelvis_offset = np.asarray([var.get() for var in self.pelvis_offset_vars], dtype=np.float32)
        except tk.TclError:
            return
        if actor.kind == "model" and np.linalg.norm(actor.initial_pelvis_offset - pelvis_offset) > 1e-7:
            actor.initial_pelvis_offset = pelvis_offset
            actor.reset_generation()
            self.frame = 0.0
            self.update_timeline()
        if self.tree.exists(str(actor.actor_id)):
            icon = "[x]" if actor.visible else "[ ]"
            self.tree.item(str(actor.actor_id), text=f"{icon} {actor.name}", values=(actor.kind, actor.status))
        self.update_bounds()
        self.invalidate_shadow_cache()
        self.draw()

    def max_frames(self) -> int:
        counts = [actor.frame_count for actor in self.actors if actor.visible and actor.clip is not None]
        return max(counts) if counts else 1

    def update_timeline(self) -> None:
        max_frame = max(0, self.max_frames() - 1)
        if not self.playing:
            self.frame = min(self.frame, float(max_frame))
        frame_i = min(max_frame, int(self.frame))
        if self.timeline_last_frame == frame_i and self.timeline_last_max_frame == max_frame:
            return
        self.timeline_last_frame = frame_i
        self.timeline_last_max_frame = max_frame
        self.updating_timeline = True
        self.frame_var.set(frame_i)
        self.updating_timeline = False
        self.frame_label.configure(text=f"{frame_i} / {max_frame}")
        self.draw_timeline()

    def update_bounds(self) -> None:
        previous_target = self.camera_target() if hasattr(self, "camera_pan") else np.zeros(3, dtype=np.float32)
        points = []
        for actor in self.actors:
            if not actor.visible or actor.clip is None:
                continue
            clip_pos = actor.clip.global_pos.detach().cpu().numpy().reshape(-1, 3)
            points.append(clip_pos)
        if not points:
            self.center = np.zeros(3, dtype=np.float32)
            self.extent = 1.0
            self.set_camera_target(np.zeros(3, dtype=np.float32))
            return
        all_points = np.concatenate(points, axis=0)
        mn = all_points.min(axis=0)
        mx = all_points.max(axis=0)
        self.center = ((mn + mx) * 0.5).astype(np.float32)
        self.extent = float(max(np.max(mx - mn), 1.0))
        if self.pending_startup_camera_target is not None:
            self.set_camera_target(self.pending_startup_camera_target)
            self.pending_startup_camera_target = None
        elif self.camera_target_locked:
            self.set_camera_target(previous_target)
        else:
            self.set_camera_target(np.asarray([self.center[0], 0.0, self.center[2]], dtype=np.float32))
        if self.pending_startup_camera_distance is not None:
            self.set_camera_distance(self.pending_startup_camera_distance)
            self.pending_startup_camera_distance = None

    def toggle_play(self) -> None:
        self.playing = not self.playing
        if self.playing:
            self.gizmo_visible = False
            self.gizmo_drag = None
            self.playback_accumulator = 0.0
            if self.controller_enabled_var.get():
                actor = self.selected_controller_actor()
                if actor is not None:
                    actor.controller_active = True
                    actor.controller_settings_snapshot = self.controller_settings()
            else:
                for actor in self.actors:
                    if actor.kind == "model":
                        actor.start_async_generation(int(self.frame) + 8)
        self.last_tick = time.perf_counter()

    def advance_playback(self, step_seconds: float) -> None:
        fps = self.scene_fps()
        speed = max(0.01, float(self.speed_var.get()))
        self.frame += step_seconds * fps * speed
        frame_count = max(1, self.max_frames())
        if self.controller_enabled_var.get():
            return
        elif self.frame >= frame_count:
            self.frame = self.frame % float(frame_count)

    def stop_playback(self) -> None:
        self.playing = False
        self.frame = 0.0
        for actor in self.actors:
            if actor.kind == "model":
                actor.reset_generation()
        self.update_timeline()
        self.draw()

    def step_frame(self, delta: int) -> None:
        max_frame = max(0, self.max_frames() - 1)
        self.frame = float(max(0, min(max_frame, int(self.frame) + delta)))
        self.update_timeline()
        self.draw()

    def on_frame_scale(self, value: str) -> None:
        if self.updating_timeline:
            return
        self.frame = float(int(float(value)))
        self.frame_label.configure(text=f"{int(self.frame)} / {max(0, self.max_frames() - 1)}")
        self.draw_timeline()
        self.draw()

    def timeline_bounds(self) -> tuple[int, int, int]:
        width = max(1, self.timeline.winfo_width())
        y = max(15, self.timeline.winfo_height() // 2)
        return 12, max(13, width - 12), y

    def frame_from_timeline_x(self, x: float) -> int:
        max_frame = max(0, self.max_frames() - 1)
        if max_frame <= 0:
            return 0
        x0, x1, _ = self.timeline_bounds()
        t = (max(x0, min(x1, x)) - x0) / max(1, x1 - x0)
        return int(round(t * max_frame))

    def set_frame_from_timeline(self, x: float) -> None:
        self.frame = float(self.frame_from_timeline_x(x))
        self.frame_var.set(int(self.frame))
        self.frame_label.configure(text=f"{int(self.frame)} / {max(0, self.max_frames() - 1)}")
        self.draw_timeline()
        self.draw()

    def on_timeline_down(self, event: tk.Event) -> None:
        if self.max_frames() <= 1:
            return
        self.timeline_dragging = True
        self.set_frame_from_timeline(event.x)

    def on_timeline_drag(self, event: tk.Event) -> None:
        if not self.timeline_dragging:
            return
        self.set_frame_from_timeline(event.x)

    def on_timeline_up(self, event: tk.Event) -> None:
        if self.timeline_dragging:
            self.set_frame_from_timeline(event.x)
        self.timeline_dragging = False

    def draw_timeline(self) -> None:
        if not hasattr(self, "timeline"):
            return
        self.timeline.delete("all")
        width = max(1, self.timeline.winfo_width())
        height = max(1, self.timeline.winfo_height())
        max_frame = max(0, self.max_frames() - 1)
        x0, x1, y = self.timeline_bounds()
        self.timeline.create_rectangle(0, 0, width, height, fill=PANEL_2, outline="")
        self.timeline.create_line(x0, y, x1, y, fill=LINE, width=10, capstyle=tk.ROUND)
        if max_frame > 0:
            t = max(0.0, min(1.0, float(self.frame) / max_frame))
            pin_x = x0 + (x1 - x0) * t
            self.timeline.create_line(x0, y, pin_x, y, fill="#55d6a7", width=10, capstyle=tk.ROUND)
            self.timeline.create_oval(pin_x - 8, y - 8, pin_x + 8, y + 8, fill="#edf1f7", outline="#55d6a7", width=2)
            self.timeline.create_oval(pin_x - 3, y - 3, pin_x + 3, y + 3, fill="#55d6a7", outline="")
        else:
            self.timeline.create_text(width * 0.5, y, text="load motion", fill=MUTED)

    def reset_camera(self) -> None:
        self.cancel_camera_transition()
        self.set_follow_cam_actor(None)
        self.yaw = -0.75
        self.pitch = -0.18
        self.view_axis = None
        self.zoom = 1.45
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.camera_target_locked = False
        self.penetration_mode = False
        self.set_camera_target(np.asarray([self.center[0], 0.0, self.center[2]], dtype=np.float32))
        self.draw()

    def penetration_camera(self) -> None:
        self.cancel_camera_transition()
        target = self.camera_target().copy()
        if self.follow_cam_actor_id is not None:
            actor = next((item for item in self.actors if item.actor_id == self.follow_cam_actor_id), None)
            if actor is not None and actor.visible:
                root = self.actor_follow_root_world(actor)
                if root is not None:
                    target[0] = root[0]
                    target[2] = root[2]
        target[1] = 0.0
        self.set_camera_target(target)
        self.pitch = FLAT_PITCH
        self.view_axis = None
        self.pan_y = 0.0
        self.camera_target_locked = True
        self.penetration_mode = True
        self.draw()

    def follow_camera(self) -> None:
        if self.follow_cam_actor_id is not None:
            self.set_follow_cam_actor(None)
            self.draw()
            return
        actor = self.selected_controller_actor() or self.selected_actor()
        if actor is None:
            return
        root = self.actor_follow_root_world(actor)
        if root is None:
            return
        self.cancel_camera_transition()
        target = root.copy()
        target[1] = max(target[1], 0.85)
        self.set_camera_target(target)
        self.set_follow_cam_actor(actor.actor_id)
        actor_yaw = float(actor.controller_yaw) if actor.kind == "model" else 0.0
        self.yaw = actor_yaw + math.pi
        self.pitch = 0.22
        self.view_axis = None
        self.penetration_mode = False
        self.camera_target_locked = True
        self.set_camera_distance(3.2)
        self.draw()

    def set_follow_cam_actor(self, actor_id: int | None) -> None:
        self.follow_cam_actor_id = actor_id
        if hasattr(self, "follow_cam_button"):
            self.follow_cam_button.configure(text="Free Cam" if actor_id is not None else "Follow Cam")

    def update_follow_camera_target(self) -> bool:
        if self.follow_cam_actor_id is None:
            return False
        actor = next((item for item in self.actors if item.actor_id == self.follow_cam_actor_id), None)
        if actor is None or not actor.visible:
            self.set_follow_cam_actor(None)
            return False
        self.cancel_camera_transition()
        root = self.actor_follow_root_world(actor)
        if root is None:
            return False
        old_target = self.camera_target().copy()
        target = root.copy()
        if self.penetration_mode:
            target[1] = 0.0
        else:
            target[1] = max(target[1], 0.85)
        self.set_camera_target(target)
        self.camera_target_locked = True
        return bool(np.linalg.norm(target - old_target) > 1e-5)

    def camera_target(self) -> np.ndarray:
        return np.asarray([self.center[0], 0.0, self.center[2]], dtype=np.float32) + self.camera_pan

    def set_camera_target(self, target: np.ndarray) -> None:
        base = np.asarray([self.center[0], 0.0, self.center[2]], dtype=np.float32)
        self.camera_pan = (np.asarray(target, dtype=np.float32) - base).astype(np.float32)

    def camera_distance(self) -> float:
        return max(MIN_CAMERA_DISTANCE, self.extent * 2.9 / max(1e-6, self.zoom))

    def set_camera_distance(self, distance: float) -> None:
        distance = max(MIN_CAMERA_DISTANCE, float(distance))
        self.zoom = max(1e-6, min(1e6, self.extent * 2.9 / distance))

    def camera_view(
        self,
        yaw: float | None = None,
        pitch: float | None = None,
        distance: float | None = None,
        target: np.ndarray | None = None,
        view_axis: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        yaw = self.yaw if yaw is None else float(yaw)
        pitch = self.pitch if pitch is None else float(pitch)
        distance = self.camera_distance() if distance is None else max(MIN_CAMERA_DISTANCE, float(distance))
        target = self.camera_target() if target is None else np.asarray(target, dtype=np.float32)
        if view_axis == "+y":
            eye = target + np.asarray([0.0, distance, 0.0], dtype=np.float32)
            up = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
        elif view_axis == "-y":
            eye = target + np.asarray([0.0, -distance, 0.0], dtype=np.float32)
            up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            eye = target + np.asarray(
                [
                    math.sin(yaw) * distance,
                    distance * (0.35 - pitch * 0.55),
                    math.cos(yaw) * distance,
                ],
                dtype=np.float32,
            )
            up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        return eye.astype(np.float32), target.astype(np.float32), up.astype(np.float32)

    def slerp_unit(self, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        a = a / max(1e-6, float(np.linalg.norm(a)))
        b = b / max(1e-6, float(np.linalg.norm(b)))
        dot = max(-1.0, min(1.0, float(np.dot(a, b))))
        if dot > 0.9995:
            mixed = a * (1.0 - t) + b * t
            return mixed / max(1e-6, float(np.linalg.norm(mixed)))
        if dot < -0.9995:
            ortho = np.cross(a, np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
            if float(np.linalg.norm(ortho)) <= 1e-6:
                ortho = np.cross(a, np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
            ortho = ortho / max(1e-6, float(np.linalg.norm(ortho)))
            angle = math.pi * t
            return (a * math.cos(angle) + ortho * math.sin(angle)).astype(np.float32)
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        return (
            a * (math.sin((1.0 - t) * theta) / sin_theta)
            + b * (math.sin(t * theta) / sin_theta)
        ).astype(np.float32)

    def camera_up_from_hint(self, direction: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
        direction = np.asarray(direction, dtype=np.float32)
        direction = direction / max(1e-6, float(np.linalg.norm(direction)))
        forward = -direction
        up = np.asarray(up_hint, dtype=np.float32)
        up = up - forward * float(np.dot(up, forward))
        norm = float(np.linalg.norm(up))
        if norm <= 1e-6:
            fallback = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
            if abs(float(np.dot(fallback, forward))) > 0.96:
                fallback = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
            up = fallback - forward * float(np.dot(fallback, forward))
            norm = float(np.linalg.norm(up))
        return (up / max(1e-6, norm)).astype(np.float32)

    def camera_render_view(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.camera_transition is None:
            return self.camera_view(view_axis=self.view_axis)
        elapsed = time.perf_counter() - float(self.camera_transition["start_time"])
        duration = max(0.001, float(self.camera_transition["duration"]))
        t = max(0.0, min(1.0, elapsed / duration))
        eased = t * t * (3.0 - 2.0 * t)
        start_dir = np.asarray(self.camera_transition["start_dir"], dtype=np.float32)
        start_up = np.asarray(self.camera_transition["start_up"], dtype=np.float32)
        end_dir = np.asarray(self.camera_transition["end_dir"], dtype=np.float32)
        end_up = np.asarray(self.camera_transition["end_up"], dtype=np.float32)
        target = np.asarray(self.camera_transition["target"], dtype=np.float32)
        start_distance = float(self.camera_transition["start_distance"])
        end_distance = float(self.camera_transition["end_distance"])
        direction = self.slerp_unit(start_dir, end_dir, eased)
        up_hint = self.slerp_unit(start_up, end_up, eased)
        distance = start_distance * (1.0 - eased) + end_distance * eased
        eye = target + direction * distance
        up = self.camera_up_from_hint(direction, up_hint)
        if t >= 1.0:
            self.camera_transition = None
        return eye.astype(np.float32), target.astype(np.float32), up.astype(np.float32)

    def start_camera_transition(self, yaw: float, pitch: float, view_axis: str | None = None) -> None:
        start_eye, start_target, start_up = self.camera_render_view()
        target = self.camera_target().copy()
        start_vec = start_eye - start_target
        start_distance = max(MIN_CAMERA_DISTANCE, float(np.linalg.norm(start_vec)))
        self.set_camera_distance(start_distance)
        end_eye, _end_target, end_up = self.camera_view(
            yaw=yaw,
            pitch=pitch,
            distance=start_distance,
            target=target,
            view_axis=view_axis,
        )
        end_vec = end_eye - target
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.view_axis = view_axis
        self.penetration_mode = False
        self.camera_transition = {
            "start_time": time.perf_counter(),
            "duration": CAMERA_TRANSITION_SECONDS,
            "target": target,
            "start_dir": start_vec / start_distance,
            "start_distance": start_distance,
            "start_up": start_up,
            "end_dir": end_vec / start_distance,
            "end_distance": start_distance,
            "end_up": end_up,
        }

    def cancel_camera_transition(self) -> None:
        self.camera_transition = None

    def camera_right_vector(self) -> np.ndarray:
        eye, target, up = self.camera_render_view()
        forward = target - eye
        right = np.cross(forward, up)
        right[1] = 0.0
        norm = float(np.linalg.norm(right))
        if norm <= 1e-6:
            right = np.asarray([math.cos(self.yaw), 0.0, -math.sin(self.yaw)], dtype=np.float32)
            norm = float(np.linalg.norm(right))
        return right / max(norm, 1e-6)

    def camera_movement_yaw(self) -> float:
        eye, target, _up = self.camera_render_view()
        forward = target - eye
        forward[1] = 0.0
        norm = float(np.linalg.norm(forward))
        if norm <= 1e-6:
            return float(self.yaw)
        forward = forward / norm
        return math.atan2(float(forward[0]), float(forward[2]))

    def actor_anchor_root_world(self, actor: Actor) -> np.ndarray | None:
        if actor.clip is None:
            return None
        generated_count = min(len(actor.generated_pos), len(actor.generated_rot))
        if actor.kind == "model" and generated_count > 0:
            frame = max(0, min(generated_count - 1, int(self.frame)))
            positions = actor.generated_pos[frame]
        else:
            frame = max(0, min(actor.clip.T - 1, int(self.frame)))
            positions = actor.clip.global_pos[frame].detach().cpu().numpy().astype(np.float32)
        root = max(0, min(actor.root_index, len(positions) - 1))
        return (positions[root] + actor.offset).astype(np.float32)

    def actor_authored_root_world(self, actor: Actor, frame: int | None = None) -> np.ndarray | None:
        if actor.clip is None:
            return None
        frame_i = int(self.frame) if frame is None else int(frame)
        frame_i = max(0, min(actor.clip.T - 1, frame_i))
        root = actor.clip.root_pos[frame_i].detach().cpu().numpy().astype(np.float32)
        return (root + actor.offset).astype(np.float32)

    def actor_follow_root_world(self, actor: Actor) -> np.ndarray | None:
        if actor.kind == "model" and actor.controller_active and actor.controller_root_anchor is not None:
            return (actor.controller_root_anchor + actor.controller_root_offset + actor.offset).astype(np.float32)
        if actor.clip is None:
            return self.actor_root_world(actor)
        max_frame = max(0, actor.clip.T - 1)
        frame = max(0.0, min(float(self.frame), float(max_frame)))
        base = int(math.floor(frame))
        nxt = min(max_frame, base + 1)
        t = frame - base
        root_a = actor.clip.root_pos[base].detach().cpu().numpy().astype(np.float32)
        if nxt == base or t <= 1e-5:
            root = root_a
        else:
            root_b = actor.clip.root_pos[nxt].detach().cpu().numpy().astype(np.float32)
            root = (root_a * (1.0 - t) + root_b * t).astype(np.float32)
        return (root + actor.offset).astype(np.float32)

    def camera_pan_scale(self) -> float:
        distance = self.camera_distance()
        viewport = max(1, self.canvas.winfo_width(), self.canvas.winfo_height())
        return distance / viewport * 1.65

    def actor_pose_world(self, actor: Actor) -> tuple[np.ndarray, np.ndarray] | None:
        pose = actor.pose_for_frame(int(self.frame))
        if pose is None:
            return None
        positions, rotations = pose
        return self.apply_actor_world_transform(actor, positions, rotations)

    def apply_actor_world_transform(
        self, actor: Actor, positions: np.ndarray, rotations: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        transformed_pos = positions.astype(np.float32, copy=True)
        transformed_rot = rotations.astype(np.float32, copy=True)
        if actor.kind == "model" and (actor.controller_active or np.linalg.norm(actor.controller_root_offset) > 1e-6):
            pivot_index = max(0, min(actor.root_index, len(transformed_pos) - 1))
            authored_root = transformed_pos[pivot_index].copy()
            if actor.controller_active:
                if actor.controller_root_anchor is None:
                    actor.controller_root_anchor = authored_root.copy()
                pivot = actor.controller_root_anchor.astype(np.float32)
                transformed_pos = transformed_pos + (pivot - authored_root)[None, :]
            else:
                pivot = authored_root
            yaw_matrix = yaw_rotation_matrix_np(actor.controller_yaw)
            transformed_pos = ((transformed_pos - pivot[None, :]) @ yaw_matrix + pivot[None, :] + actor.controller_root_offset[None, :]).astype(np.float32)
            transformed_rot = np.matmul(transformed_rot, yaw_matrix).astype(np.float32)
        transformed_pos = transformed_pos + actor.offset[None, :]
        return transformed_pos, transformed_rot

    def blend_rotation_matrices(self, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        blended = a * (1.0 - t) + b * t
        try:
            u, _s, vh = np.linalg.svd(blended)
            rotations = u @ vh
            det = np.linalg.det(rotations)
            bad = det < 0.0
            if np.any(bad):
                u[bad, :, -1] *= -1.0
                rotations = u @ vh
            return rotations.astype(np.float32)
        except np.linalg.LinAlgError:
            return b.astype(np.float32) if t >= 0.5 else a.astype(np.float32)

    def actor_pose_for_display(self, actor: Actor) -> tuple[np.ndarray, np.ndarray] | None:
        if actor.clip is None:
            return None
        controller_live = bool(actor.kind == "model" and actor.controller_active)
        max_frame = max(0, actor.frame_count - 1)
        if controller_live:
            generated_count = min(len(actor.generated_pos), len(actor.generated_rot))
            if generated_count <= 0:
                return None
            if self.playing:
                frame = float(generated_count - 1)
            else:
                frame = max(0.0, min(float(self.frame), float(generated_count - 1)))
        else:
            frame = max(0.0, min(float(self.frame), float(max_frame)))
        base = int(math.floor(frame))
        nxt = base + 1 if controller_live else min(max_frame, base + 1)
        t = frame - base
        generate_sync = not (self.playing and actor.kind == "model")
        if not generate_sync:
            if controller_live:
                generated_count = min(len(actor.generated_pos), len(actor.generated_rot))
                actor.start_async_generation(max(nxt + 1, generated_count + 2))
            else:
                actor.start_async_generation(min(max_frame, nxt + 8))
                if nxt >= min(len(actor.generated_pos), len(actor.generated_rot)):
                    actor.generate_to(nxt)
        pose_a = actor.pose_for_frame(base, generate=generate_sync)
        if pose_a is None:
            return None
        if nxt == base or t <= 1e-4:
            positions, rotations = pose_a
            return self.apply_actor_world_transform(actor, positions, rotations)
        pose_b = actor.pose_for_frame(nxt, generate=generate_sync)
        if pose_b is None:
            positions, rotations = pose_a
            return self.apply_actor_world_transform(actor, positions, rotations)
        pos_a, rot_a = pose_a
        pos_b, rot_b = pose_b
        positions = (pos_a * (1.0 - t) + pos_b * t).astype(np.float32)
        rotations = rot_b if t >= 0.5 else rot_a
        return self.apply_actor_world_transform(actor, positions, rotations)

    def actor_root_world(self, actor: Actor) -> np.ndarray | None:
        if actor.kind == "model" and actor.controller_active and actor.controller_root_anchor is not None:
            return (actor.controller_root_anchor + actor.controller_root_offset + actor.offset).astype(np.float32)
        authored_root = self.actor_authored_root_world(actor)
        if authored_root is not None:
            return authored_root
        pose = self.actor_pose_world(actor)
        if pose is None:
            return None
        positions, _rotations = pose
        root = max(0, min(actor.root_index, len(positions) - 1))
        return positions[root].astype(np.float32)

    def focus_camera_on_actor(self, actor: Actor) -> None:
        self.cancel_camera_transition()
        pose = self.actor_pose_world(actor)
        if pose is None:
            return
        positions, _rotations = pose
        target = self.actor_root_world(actor)
        if target is None:
            root = max(0, min(actor.root_index, len(positions) - 1))
            target = positions[root].astype(np.float32)
        extent = float(max(np.max(positions.max(axis=0) - positions.min(axis=0)), 1.0))
        self.set_camera_target(target)
        self.camera_target_locked = True
        self.penetration_mode = False
        self.yaw = -0.75
        self.pitch = NEUTRAL_PITCH
        self.view_axis = None
        self.set_camera_distance(max(4.8, min(6.4, extent * 3.45)))
        self.draw()

    def schedule_next_tick(self) -> None:
        now = time.perf_counter()
        frame_interval = 1.0 / TARGET_RENDER_FPS
        if self.next_tick_time < now - frame_interval:
            self.next_tick_time = now + frame_interval
        delay_ms = max(1, int(round((self.next_tick_time - now) * 1000.0)))
        self.next_tick_time += frame_interval
        self.after(delay_ms, self._tick)

    def _tick(self) -> None:
        if self.state() == "iconic":
            self.last_tick = time.perf_counter()
            self.next_tick_time = self.last_tick + (1.0 / TARGET_RENDER_FPS)
            self.after(1000, self._tick)
            return
        now = time.perf_counter()
        dt = min(0.1, now - self.last_tick)
        self.last_tick = now
        drew = False
        controller_moved = self.update_controller_root_motion(dt)
        follow_moved = False
        if self.playing:
            self.playback_accumulator = min(0.08, self.playback_accumulator + dt)
            target_step = 1.0 / TARGET_PLAYBACK_FPS
            steps = 0
            max_substeps = 1 if self.controller_enabled_var.get() else MAX_PLAYBACK_SUBSTEPS
            while self.playback_accumulator >= target_step and steps < max_substeps:
                self.advance_playback(target_step)
                self.playback_accumulator -= target_step
                steps += 1
            if steps > 0:
                self.update_timeline()
                follow_moved = self.update_follow_camera_target()
                self.draw()
                drew = True
            if steps >= max_substeps:
                self.playback_accumulator = min(self.playback_accumulator, target_step)
        if not drew:
            follow_moved = self.update_follow_camera_target()
        if (self.camera_transition is not None or controller_moved or follow_moved) and not drew:
            self.draw()
        self.schedule_next_tick()

    def has_live_model_actor(self) -> bool:
        return any(actor.visible and actor.kind == "model" and actor.clip is not None for actor in self.actors)

    def scene_fps(self) -> float:
        for actor in self.actors:
            if actor.clip is not None:
                return float(actor.clip.fps)
        return 30.0

    def controller_settings(self) -> RootMotionSettings:
        try:
            return RootMotionSettings(
                max_speed_mps=max(0.05, float(self.controller_max_speed_var.get())),
                acceleration_response=max(0.1, float(self.controller_accel_response_var.get())),
                deceleration_response=max(0.1, float(self.controller_decel_response_var.get())),
                turn_rate_dps=max(1.0, float(self.controller_turn_rate_var.get())),
                left_deadzone=max(0.0, min(0.8, float(self.controller_left_deadzone_var.get()))),
                right_deadzone=max(0.0, min(0.8, float(self.controller_right_deadzone_var.get()))),
                trajectory_seconds=max(0.1, float(self.trajectory_seconds_var.get())),
                trajectory_step_seconds=max(0.04, float(self.trajectory_step_var.get())),
            )
        except tk.TclError:
            return RootMotionSettings(1.8, 9.0, 11.0, 240.0, 0.24, 0.26, 1.0, 0.16)

    def poll_controller(self) -> ControllerSample:
        if self.xinput is None:
            return ControllerSample(False)
        left_deadzone = getattr(self, "controller_left_deadzone_var", None)
        right_deadzone = getattr(self, "controller_right_deadzone_var", None)
        left_dz = float(left_deadzone.get()) * XINPUT_THUMB_MAX if left_deadzone is not None else XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE
        right_dz = float(right_deadzone.get()) * XINPUT_THUMB_MAX if right_deadzone is not None else XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE
        for user_index in range(4):
            state = XInputState()
            result = int(self.xinput.XInputGetState(ctypes.c_uint(user_index), ctypes.byref(state)))
            if result != 0:
                continue
            gamepad = state.Gamepad
            move = apply_stick_deadzone(gamepad.sThumbLX, gamepad.sThumbLY, left_dz)
            move[0] = -move[0]
            look = apply_stick_deadzone(gamepad.sThumbRX, gamepad.sThumbRY, right_dz)
            triggers = (float(gamepad.bLeftTrigger) - float(gamepad.bRightTrigger)) / 255.0
            return ControllerSample(True, move, look, triggers)
        return ControllerSample(False)

    def selected_controller_actor(self) -> Actor | None:
        actor = self.selected_actor()
        if actor is not None and actor.kind == "model" and actor.clip is not None:
            return actor
        for item in self.actors:
            if item.visible and item.kind == "model" and item.clip is not None:
                return item
        return None

    def update_controller_root_motion(self, dt: float) -> bool:
        if not self.controller_enabled_var.get():
            return False
        actor = self.selected_controller_actor()
        if actor is None:
            return False
        sample = self.poll_controller()
        self.controller_sample = sample
        if actor.controller_root_anchor is None:
            root = self.actor_anchor_root_world(actor)
            if root is not None:
                actor.controller_root_anchor = (root - actor.controller_root_offset).astype(np.float32)
        actor.controller_active = True
        actor.controller_move_input = sample.move.astype(np.float32, copy=True)
        actor.controller_turn_input = float(sample.triggers)
        actor.controller_move_yaw = self.camera_movement_yaw()
        actor.controller_settings_snapshot = self.controller_settings()
        state = RootMotionState(actor.controller_root_offset.copy(), actor.controller_velocity.copy(), float(actor.controller_yaw))
        root_turn = np.asarray([sample.triggers, 0.0], dtype=np.float32)
        anchor = actor.controller_root_anchor if actor.controller_root_anchor is not None else np.zeros(3, dtype=np.float32)
        prev_root_pos = (anchor + state.offset).astype(np.float32)
        prev_root_yaw = float(state.yaw)
        updated = integrate_controller_root_motion(
            state,
            sample.move,
            root_turn,
            dt,
            actor.controller_settings_snapshot,
            movement_yaw=actor.controller_move_yaw,
        )
        actor.controller_root_offset = updated.offset
        actor.controller_velocity = updated.velocity
        actor.controller_yaw = updated.yaw
        cur_root_pos = (anchor + updated.offset).astype(np.float32)
        cur_root_yaw = float(updated.yaw)
        future_positions: list[np.ndarray] = []
        future_yaws: list[float] = []
        if actor.clip is not None and actor.cfg is not None:
            future_state = RootMotionState(updated.offset.copy(), updated.velocity.copy(), float(updated.yaw))
            sim_dt = 1.0 / max(1.0, float(actor.clip.fps))
            for _ in range(int(actor.cfg.future_window)):
                future_state = integrate_controller_root_motion(
                    future_state,
                    sample.move,
                    root_turn,
                    sim_dt,
                    actor.controller_settings_snapshot,
                    movement_yaw=actor.controller_move_yaw,
                )
                future_positions.append((anchor + future_state.offset).astype(np.float32))
                future_yaws.append(float(future_state.yaw))
            actor.set_controller_root_plan(
                prev_root_pos,
                cur_root_pos,
                prev_root_yaw,
                cur_root_yaw,
                np.asarray(future_positions, dtype=np.float32),
                np.asarray(future_yaws, dtype=np.float32),
            )
        if self.playing:
            generated_count = min(len(actor.generated_pos), len(actor.generated_rot))
            actor.start_async_generation(generated_count + 2)
        if np.linalg.norm(sample.look) > 1e-5:
            self.view_axis = None
            self.yaw -= float(sample.look[0]) * 0.045
            self.pitch = max(-1.35, min(1.35, self.pitch - float(sample.look[1]) * 0.035))
            self.penetration_mode = False
        return bool(
            np.linalg.norm(sample.move) > 1e-5
            or abs(sample.triggers) > 1e-5
            or np.linalg.norm(sample.look) > 1e-5
            or np.linalg.norm(actor.controller_velocity) > 1e-5
        )

    def synthetic_controller_preview(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        move = np.asarray([math.sin(t) * 0.7, math.cos(t * 0.7) * 0.85], dtype=np.float32)
        look = np.asarray([math.sin(t * 0.43) * 0.45, 0.0], dtype=np.float32)
        return move, look

    def synthetic_controller_sample(self, t: float) -> ControllerSample:
        move, look = self.synthetic_controller_preview(t)
        return ControllerSample(True, move, look, math.sin(t * 0.31) * 0.35)

    def clear_controller_root_motion(self) -> None:
        for actor in self.actors:
            actor.controller_root_offset = np.zeros(3, dtype=np.float32)
            actor.controller_velocity = np.zeros(2, dtype=np.float32)
            actor.controller_root_anchor = None
            actor.controller_yaw = 0.0
            actor.controller_move_yaw = 0.0
            actor.controller_active = False
            actor.controller_move_input = np.zeros(2, dtype=np.float32)
            actor.controller_turn_input = 0.0
            actor.controller_settings_snapshot = None
            actor.clear_controller_root_plan()
            if actor.kind == "model":
                actor.reset_generation()
        self.controller_sample = ControllerSample()

    def on_controller_toggle(self) -> None:
        if not self.controller_enabled_var.get():
            self.clear_controller_root_motion()
        self.draw()

    def mark_pointer_dragged(self, x: int, y: int) -> None:
        if self.pointer_down_start is None:
            return
        start_x, start_y = self.pointer_down_start
        if abs(x - start_x) > 3 or abs(y - start_y) > 3:
            self.pending_gizmo_clear_on_release = False

    def clear_gizmo_on_click_release(self, x: int, y: int) -> None:
        if not self.pending_gizmo_clear_on_release:
            return
        self.mark_pointer_dragged(x, y)
        if self.pending_gizmo_clear_on_release:
            self.clear_actor_selection()
        self.pending_gizmo_clear_on_release = False

    def on_mouse_down(self, event: tk.Event) -> None:
        self.cancel_camera_transition()
        self.pointer_down_start = (event.x, event.y)
        self.pending_gizmo_clear_on_release = False
        hit = self.hit_gizmo(event.x, event.y)
        if hit is not None and not self.playing:
            self.gizmo_drag = hit
            self.gizmo_last_mouse = (event.x, event.y)
            return
        picked = self.pick_actor_at(event.x, event.y)
        if picked is not None and not self.playing:
            self.select_actor_from_viewport(picked.actor_id)
            return
        self.gizmo_drag = None
        self.pending_gizmo_clear_on_release = self.selected_actor_id is not None
        self.drag_start = (event.x, event.y)
        self.drag_pan = bool(event.state & 0x0001)

    def on_mouse_drag(self, event: tk.Event) -> None:
        if self.gizmo_drag is not None:
            self.drag_gizmo(event.x, event.y)
            return
        if self.drag_start is None:
            return
        self.mark_pointer_dragged(event.x, event.y)
        last_x, last_y = self.drag_start
        dx = event.x - last_x
        dy = event.y - last_y
        if self.drag_pan:
            scale = self.camera_pan_scale()
            right = self.camera_right_vector()
            self.camera_pan += (-right * dx * scale + np.asarray([0.0, dy * scale, 0.0], dtype=np.float32)).astype(np.float32)
            self.camera_target_locked = True
        else:
            self.view_axis = None
            self.yaw -= dx * 0.006
            self.pitch = max(-1.35, min(1.35, self.pitch - dy * 0.006))
            self.penetration_mode = False
        self.drag_start = (event.x, event.y)
        self.draw()

    def on_mouse_up(self, event: tk.Event) -> None:
        self.clear_gizmo_on_click_release(event.x, event.y)
        self.drag_start = None
        self.gizmo_drag = None
        self.gizmo_last_mouse = None
        self.pointer_down_start = None

    def on_pan_down(self, event: tk.Event) -> None:
        self.cancel_camera_transition()
        self.pointer_down_start = (event.x, event.y)
        self.pending_gizmo_clear_on_release = self.gizmo_visible
        self.pan_drag_start = (event.x, event.y)

    def on_pan_drag(self, event: tk.Event) -> None:
        if self.pan_drag_start is None:
            return
        self.mark_pointer_dragged(event.x, event.y)
        last_x, last_y = self.pan_drag_start
        dx = event.x - last_x
        dy = event.y - last_y
        scale = self.camera_pan_scale()
        right = self.camera_right_vector()
        self.camera_pan += (-right * dx * scale + np.asarray([0.0, dy * scale, 0.0], dtype=np.float32)).astype(np.float32)
        self.camera_target_locked = True
        self.pan_drag_start = (event.x, event.y)
        self.draw()

    def on_pan_up(self, event: tk.Event) -> None:
        self.clear_gizmo_on_click_release(event.x, event.y)
        self.pan_drag_start = None
        self.pointer_down_start = None

    def axis_gizmo_points(self) -> dict[str, tuple[int, int, str, str]]:
        return {
            "+x": (64, 43, "+X", "#ff6f6f"),
            "-x": (22, 43, "-X", "#a84646"),
            "+y": (43, 20, "+Y", "#6fdc8c"),
            "-y": (43, 66, "-Y", "#3f8c5a"),
            "+z": (64, 22, "+Z", "#72a7ff"),
            "-z": (22, 64, "-Z", "#4268a8"),
        }

    def draw_camera_axis_gizmo(self) -> None:
        if not hasattr(self, "axis_canvas"):
            return
        canvas = self.axis_canvas
        canvas.tk.call("raise", canvas._w)
        canvas.delete("all")
        canvas.create_rectangle(0, 0, 86, 86, fill="#090b0f", outline=LINE)
        center = (43, 43)
        points = self.axis_gizmo_points()
        for name in ("-x", "+x", "-y", "+y", "-z", "+z"):
            x, y, _label, color = points[name]
            canvas.create_line(center[0], center[1], x, y, fill=blend_hex(color, "#000000", 0.18), width=2)
        canvas.create_oval(36, 36, 50, 50, fill="#202733", outline="#8b96a8", width=1)
        for name, (x, y, label, color) in points.items():
            canvas.create_oval(x - 11, y - 11, x + 11, y + 11, fill=color, outline="#e4e9f2", width=1)
            canvas.create_text(x, y, text=label, fill="#05070a", font=("Segoe UI", 7, "bold"))

    def record_render_frame(self) -> None:
        self.fps_frame_count += 1
        now = time.perf_counter()
        elapsed = now - self.fps_last_time
        if elapsed >= 0.5:
            self.display_fps = self.fps_frame_count / max(1e-6, elapsed)
            self.fps_frame_count = 0
            self.fps_last_time = now

    def camera_overlay_text(self) -> str:
        target = self.camera_target()
        return (
            f"yaw {self.yaw:+.3f}  pitch {self.pitch:+.3f}  dist {self.camera_distance():.3f}\n"
            f"target {target[0]:+.3f} {target[1]:+.3f} {target[2]:+.3f}\n"
            f"fps {self.display_fps:5.1f}"
        )

    def on_axis_gizmo_click(self, event: tk.Event) -> None:
        nearest = None
        nearest_d2 = float("inf")
        for name, (x, y, _label, _color) in self.axis_gizmo_points().items():
            d2 = float((event.x - x) ** 2 + (event.y - y) ** 2)
            if d2 < nearest_d2:
                nearest = name
                nearest_d2 = d2
        if nearest is not None and nearest_d2 <= 18.0**2:
            self.snap_camera_axis(nearest)

    def snap_camera_axis(self, axis_name: str) -> None:
        if axis_name == "+x":
            self.start_camera_transition(math.pi * 0.5, FLAT_PITCH, None)
        elif axis_name == "-x":
            self.start_camera_transition(-math.pi * 0.5, FLAT_PITCH, None)
        elif axis_name == "+z":
            self.start_camera_transition(0.0, FLAT_PITCH, None)
        elif axis_name == "-z":
            self.start_camera_transition(math.pi, FLAT_PITCH, None)
        elif axis_name == "+y":
            self.start_camera_transition(self.yaw, self.pitch, "+y")
        elif axis_name == "-y":
            self.start_camera_transition(self.yaw, self.pitch, "-y")
        self.draw()

    def on_mouse_wheel(self, event: tk.Event) -> None:
        self.cancel_camera_transition()
        steps = event.delta / 120.0
        distance = self.camera_distance()
        if steps > 0:
            distance = MIN_CAMERA_DISTANCE + (distance - MIN_CAMERA_DISTANCE) * math.exp(-0.20 * steps)
        elif steps < 0:
            distance = distance * math.exp(-0.20 * steps)
        max_distance = max(20.0, self.extent * 20.0)
        self.set_camera_distance(min(max_distance, distance))
        self.draw()

    def project(self, p: np.ndarray) -> tuple[float, float, float]:
        w = max(1, self.canvas.winfo_width())
        h = max(1, self.canvas.winfo_height())
        scale = max(0.001, float(self.scale_var.get()))
        p0 = (np.asarray(p, dtype=np.float32) - self.camera_target()) * scale
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        x1 = p0[0] * cy - p0[2] * sy
        z1 = p0[0] * sy + p0[2] * cy
        y1 = p0[1] * cp - z1 * sp
        z2 = p0[1] * sp + z1 * cp
        s = min(w, h) * 0.72 * self.zoom / max(self.extent, 1e-3)
        return (w * 0.5 + self.pan_x + x1 * s, h * 0.48 + self.pan_y - y1 * s, z2)

    def screen_project(self, p: np.ndarray) -> tuple[float, float, float]:
        if self.gl_modelview is not None and self.gl_projection is not None and self.gl_viewport is not None:
            try:
                x, y, z = GLU.gluProject(
                    float(p[0]),
                    float(p[1]),
                    float(p[2]),
                    self.gl_modelview,
                    self.gl_projection,
                    self.gl_viewport,
                )
                return (float(x), float(self.gl_viewport[3] - y), float(z))
            except Exception:
                pass
        return self.project(p)

    def world_radius_px(self, radius_world: float) -> float:
        w = max(1, self.canvas.winfo_width())
        h = max(1, self.canvas.winfo_height())
        scale = max(0.001, float(self.scale_var.get()))
        return max(3.0, radius_world * scale * min(w, h) * 0.72 * self.zoom / max(self.extent, 1e-3))

    def begin_scene(self) -> None:
        self.canvas.configure(bg=SKY)
        self.canvas.delete("all")
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        scaled = (max(1, int(width * RENDER_SCALE)), max(1, int(height * RENDER_SCALE)))
        self.render_image = Image.new("RGBA", scaled, SKY)
        self.render_draw = ImageDraw.Draw(self.render_image, "RGBA")
        if self.render_font is None:
            try:
                self.render_font = ImageFont.truetype("segoeui.ttf", int(10 * RENDER_SCALE))
            except OSError:
                self.render_font = ImageFont.load_default()

    def finish_scene(self) -> None:
        if self.render_image is None:
            return
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        image = self.render_image.resize((width, height), Image.Resampling.LANCZOS)
        self.render_photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.render_photo)
        self.render_image = None
        self.render_draw = None

    def rgba(self, color: str, alpha: int = 255) -> tuple[int, int, int, int]:
        if color.startswith("#") and len(color) == 7:
            return (
                int(color[1:3], 16),
                int(color[3:5], 16),
                int(color[5:7], 16),
                max(0, min(255, int(alpha))),
            )
        return (255, 255, 255, max(0, min(255, int(alpha))))

    def scene_line(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
        fill: str,
        width: float = 1.0,
        round_caps: bool = False,
    ) -> None:
        if self.render_draw is None:
            return
        s = RENDER_SCALE
        xy = (a[0] * s, a[1] * s, b[0] * s, b[1] * s)
        draw_width = max(1, int(round(width * s)))
        self.render_draw.line(xy, fill=fill, width=draw_width)
        if round_caps:
            r = draw_width * 0.5
            for x, y in ((xy[0], xy[1]), (xy[2], xy[3])):
                self.render_draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)

    def scene_polyline(self, coords: list[float], fill: str, width: float = 1.0) -> None:
        if len(coords) < 4:
            return
        if self.render_draw is None:
            return
        s = RENDER_SCALE
        points = [(coords[i] * s, coords[i + 1] * s) for i in range(0, len(coords), 2)]
        draw_width = max(1, int(round(width * s)))
        self.render_draw.line(points, fill=fill, width=draw_width, joint="curve")
        r = draw_width * 0.5
        for x, y in (points[0], points[-1]):
            self.render_draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)

    def scene_ellipse(
        self,
        center: tuple[float, float],
        radius: float,
        fill: str,
        outline: str | None = None,
        width: float = 1.0,
    ) -> None:
        if self.render_draw is None:
            return
        s = RENDER_SCALE
        box = (
            (center[0] - radius) * s,
            (center[1] - radius) * s,
            (center[0] + radius) * s,
            (center[1] + radius) * s,
        )
        self.render_draw.ellipse(box, fill=fill, outline=outline, width=max(1, int(round(width * s))))

    def scene_polygon(
        self,
        coords: list[float],
        fill: str,
        outline: str | None = None,
        width: float = 1.0,
    ) -> None:
        if self.render_draw is None:
            return
        s = RENDER_SCALE
        points = [(coords[i] * s, coords[i + 1] * s) for i in range(0, len(coords), 2)]
        self.render_draw.polygon(points, fill=fill)
        if outline:
            self.render_draw.line(points + [points[0]], fill=outline, width=max(1, int(round(width * s))))

    def scene_text(self, xy: tuple[float, float], text: str, fill: str = MUTED) -> None:
        if self.render_draw is None:
            return
        self.render_draw.text((xy[0] * RENDER_SCALE, xy[1] * RENDER_SCALE), text, fill=fill, font=self.render_font)

    def selected_model_actor(self) -> Actor | None:
        actor = self.selected_actor()
        if actor is not None and actor.clip is not None:
            return actor
        return None

    def select_actor_from_viewport(self, actor_id: int) -> None:
        self.selected_actor_id = actor_id
        item = str(actor_id)
        if self.tree.exists(item):
            self.tree_select_source = "viewport"
            self.programmatic_tree_selection = False
            self.tree.selection_set(item)
            self.tree.focus(item)
            self.tree.see(item)
            self.after(150, lambda: self.clear_tree_select_source("viewport"))
        self.load_inspector()
        self.gizmo_visible = self.selected_model_actor() is not None
        self.gizmo_drag = None
        self.gizmo_last_mouse = None
        self.draw()

    def screen_segment_distance(self, x: float, y: float, a: np.ndarray, b: np.ndarray) -> float:
        point = np.asarray([x, y], dtype=np.float32)
        segment = b - a
        denom = float(np.dot(segment, segment))
        if denom <= 1e-6:
            return float(np.linalg.norm(point - a))
        t = max(0.0, min(1.0, float(np.dot(point - a, segment) / denom)))
        closest = a + segment * t
        return float(np.linalg.norm(point - closest))

    def projected_radius_px(self, center: np.ndarray, radius: float) -> float:
        base = np.asarray(self.screen_project(center)[:2], dtype=np.float32)
        if not np.all(np.isfinite(base)):
            return 10.0
        radius = max(0.001, float(radius))
        best = 0.0
        for delta in (
            np.asarray([radius, 0.0, 0.0], dtype=np.float32),
            np.asarray([0.0, radius, 0.0], dtype=np.float32),
            np.asarray([0.0, 0.0, radius], dtype=np.float32),
        ):
            projected = np.asarray(self.screen_project(center + delta)[:2], dtype=np.float32)
            if np.all(np.isfinite(projected)):
                best = max(best, float(np.linalg.norm(projected - base)))
        return max(6.0, min(80.0, best))

    def collider_segment_hit_score(
        self, x: float, y: float, a: np.ndarray, b: np.ndarray, radius: float
    ) -> float | None:
        pa_full = self.screen_project(a)
        pb_full = self.screen_project(b)
        pa = np.asarray(pa_full[:2], dtype=np.float32)
        pb = np.asarray(pb_full[:2], dtype=np.float32)
        if not np.all(np.isfinite(pa)) or not np.all(np.isfinite(pb)):
            return None
        distance = self.screen_segment_distance(x, y, pa, pb)
        center = (a + b) * 0.5
        threshold = max(9.0, self.projected_radius_px(center, radius) * 1.25 + 5.0)
        if distance > threshold:
            return None
        depth = max(0.0, min(1.0, (float(pa_full[2]) + float(pb_full[2])) * 0.5))
        return distance / threshold + depth * 0.05

    def collider_sphere_hit_score(self, x: float, y: float, center: np.ndarray, radius: float) -> float | None:
        projected_full = self.screen_project(center)
        projected = np.asarray(projected_full[:2], dtype=np.float32)
        if not np.all(np.isfinite(projected)):
            return None
        threshold = max(9.0, self.projected_radius_px(center, radius) + 5.0)
        distance = float(np.linalg.norm(np.asarray([x, y], dtype=np.float32) - projected))
        if distance > threshold:
            return None
        depth = max(0.0, min(1.0, float(projected_full[2])))
        return distance / threshold + depth * 0.05

    def convex_hull_2d(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        unique = sorted(set((round(float(x), 3), round(float(y), 3)) for x, y in points))
        if len(unique) <= 2:
            return unique

        def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower: list[tuple[float, float]] = []
        for point in unique:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
                lower.pop()
            lower.append(point)
        upper: list[tuple[float, float]] = []
        for point in reversed(unique):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
                upper.pop()
            upper.append(point)
        return lower[:-1] + upper[:-1]

    def point_in_polygon(self, x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
        inside = False
        count = len(polygon)
        if count < 3:
            return False
        j = count - 1
        for i in range(count):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / max(1e-6, yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def polygon_edge_distance(self, x: float, y: float, polygon: list[tuple[float, float]]) -> float:
        if not polygon:
            return float("inf")
        point = np.asarray([x, y], dtype=np.float32)
        best = float("inf")
        for i, a in enumerate(polygon):
            b = polygon[(i + 1) % len(polygon)]
            best = min(best, self.screen_segment_distance(point[0], point[1], np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)))
        return best

    def collider_box_hit_score(
        self,
        x: float,
        y: float,
        center: np.ndarray,
        axis_x: np.ndarray,
        axis_y: np.ndarray,
        axis_z: np.ndarray,
        dims: tuple[float, float, float],
    ) -> float | None:
        hx, hy, hz = dims[0] * 0.5, dims[1] * 0.5, dims[2] * 0.5
        points = []
        depths = []
        for sx in (-hx, hx):
            for sy in (-hy, hy):
                for sz in (-hz, hz):
                    world = add3(add3(add3(center, mul3(axis_x, sx)), mul3(axis_y, sy)), mul3(axis_z, sz))
                    projected = self.screen_project(world)
                    if not (math.isfinite(projected[0]) and math.isfinite(projected[1])):
                        continue
                    points.append((projected[0], projected[1]))
                    depths.append(projected[2])
        if not points:
            return None
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        pad = 5.0
        x0, x1 = min(xs) - pad, max(xs) + pad
        y0, y1 = min(ys) - pad, max(ys) + pad
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return None
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        hull = self.convex_hull_2d(points)
        if not self.point_in_polygon(x, y, hull) and self.polygon_edge_distance(x, y, hull) > 5.0:
            nx = abs(x - cx) / max(1.0, (x1 - x0) * 0.5)
            ny = abs(y - cy) / max(1.0, (y1 - y0) * 0.5)
            if nx * nx + ny * ny > 0.82:
                return None
        span = max(1.0, max(x1 - x0, y1 - y0) * 0.5)
        depth = max(0.0, min(1.0, float(sum(depths) / max(1, len(depths)))))
        return min(1.0, math.hypot(x - cx, y - cy) / span) + depth * 0.05

    def torso_box_hit_scores(
        self, x: float, y: float, positions: np.ndarray, name_to_index: dict[str, int]
    ) -> list[float]:
        frame = self.body_box_frame(positions, name_to_index)
        pelvis = name_to_index.get("pelvis")
        chest = name_to_index.get("spine_05")
        if frame is None or pelvis is None or chest is None:
            return []
        side, up, forward = frame
        base_width = self.hip_width_hint(positions, name_to_index)
        base = add3(positions[pelvis], mul3(up, 0.060))
        top = add3(positions[chest], mul3(up, -0.030))
        torso_height = max(0.30, min(0.48, float(np.linalg.norm(sub3(top, base)))))
        box_height = max(0.080, torso_height / 3.0)
        widths = (base_width * 0.86, base_width * 0.96, base_width * 1.06)
        depths = (base_width * 0.54, base_width * 0.57, base_width * 0.55)
        scores = []
        for index, (width, depth) in enumerate(zip(widths, depths)):
            center = add3(base, mul3(up, box_height * 0.5 + index * box_height))
            score = self.collider_box_hit_score(x, y, center, side, up, forward, (width, box_height, depth))
            if score is not None:
                scores.append(score)
        left = name_to_index.get("clavicle_l")
        right = name_to_index.get("clavicle_r")
        if left is not None and right is not None:
            left_pos = positions[left]
            right_pos = positions[right]
            shoulder_center = (left_pos + right_pos) * 0.5
            shoulder_width = max(base_width * 1.16, min(0.46, float(np.linalg.norm(sub3(left_pos, right_pos))) + 0.09))
            torso_top = add3(base, mul3(up, box_height * 3.0))
            connector_top = add3(shoulder_center, mul3(up, -0.018))
            connector_height = max(0.052, dot3(sub3(connector_top, torso_top), up) + 0.030)
            connector_center = add3(torso_top, mul3(up, connector_height * 0.5 - 0.006))
            for center, dims in (
                (connector_center, ((widths[-1] + shoulder_width) * 0.5, connector_height, base_width * 0.34)),
                (add3(shoulder_center, mul3(up, 0.006)), (shoulder_width, 0.048, base_width * 0.32)),
            ):
                score = self.collider_box_hit_score(x, y, center, side, up, forward, dims)
                if score is not None:
                    scores.append(score)
        return scores

    def actor_collider_hit_score(self, actor: Actor, x: float, y: float) -> float | None:
        if not actor.visible or actor.clip is None:
            return None
        pose = self.actor_pose_for_display(actor)
        if pose is None:
            return None
        positions, rotations = pose
        name_to_index = {name: i for i, name in enumerate(actor.bone_names)}
        scores: list[float] = []
        for a_name, b_name, radius_cm in self.capsule_specs():
            a = name_to_index.get(a_name)
            b = name_to_index.get(b_name)
            if a is None or b is None:
                continue
            score = self.collider_segment_hit_score(x, y, positions[a], positions[b], radius_cm * 0.01)
            if score is not None:
                scores.append(score)
        scores.extend(self.torso_box_hit_scores(x, y, positions, name_to_index))
        frame = self.body_box_frame(positions, name_to_index)
        if frame is not None:
            _side, up, _forward = frame
            pelvis = name_to_index.get("pelvis")
            if pelvis is not None:
                center = add3(positions[pelvis], mul3(up, 0.014))
                score = self.collider_sphere_hit_score(x, y, center, 0.058)
                if score is not None:
                    scores.append(score)
        for name, radius in (
            ("thigh_l", 0.066),
            ("thigh_r", 0.066),
            ("foot_l", 0.046),
            ("foot_r", 0.046),
            ("calf_l", 0.058),
            ("calf_r", 0.058),
            ("lowerarm_l", 0.041),
            ("lowerarm_r", 0.041),
        ):
            index = name_to_index.get(name)
            if index is None:
                continue
            score = self.collider_sphere_hit_score(x, y, positions[index], radius)
            if score is not None:
                scores.append(score)
        head = self.gl_head_sphere(positions, name_to_index)
        if head is not None:
            center, radius = head
            score = self.collider_sphere_hit_score(x, y, center, radius)
            if score is not None:
                scores.append(score)
        for ankle_name, toe_name in (("foot_l", "ball_l"), ("foot_r", "ball_r")):
            ankle = name_to_index.get(ankle_name)
            toe = name_to_index.get(toe_name)
            if ankle is None or toe is None:
                continue
            for center, axis_x, axis_y, axis_z, dims in self.foot_box_specs(positions, rotations, ankle, toe):
                score = self.collider_box_hit_score(x, y, center, axis_x, axis_y, axis_z, dims)
                if score is not None:
                    scores.append(score)
        for hand_name, mid_name, parent_name in (
            ("hand_l", "middle_03_l", "lowerarm_l"),
            ("hand_r", "middle_03_r", "lowerarm_r"),
        ):
            hand = name_to_index.get(hand_name)
            if hand is None:
                continue
            center, axis_x, axis_y, axis_z, dims = self.hand_box_spec(
                positions, rotations, hand, name_to_index.get(mid_name), name_to_index.get(parent_name)
            )
            score = self.collider_box_hit_score(x, y, center, axis_x, axis_y, axis_z, dims)
            if score is not None:
                scores.append(score)
        return min(scores) if scores else None

    def pick_actor_at(self, x: float, y: float) -> Actor | None:
        best_actor = None
        best_score = float("inf")
        for actor in self.actors:
            score = self.actor_collider_hit_score(actor, x, y)
            if score is not None and score < best_score:
                best_actor = actor
                best_score = score
        return best_actor

    def hit_gizmo(self, x: float, y: float) -> str | None:
        if not self.gizmo_visible:
            return None
        for name in ("x", "y", "z", "plane"):
            bounds = self.gizmo_hits.get(name)
            if bounds is None:
                continue
            x0, y0, x1, y1 = bounds
            if x0 <= x <= x1 and y0 <= y <= y1:
                return name
        return None

    def drag_gizmo(self, x: int, y: int) -> None:
        actor = self.selected_model_actor()
        if actor is None or self.gizmo_last_mouse is None or self.gizmo_drag is None:
            return
        last_x, last_y = self.gizmo_last_mouse
        delta = np.asarray([x - last_x, y - last_y], dtype=np.float32)
        origin = actor.offset.copy()
        screen_origin = np.asarray(self.screen_project(origin)[:2], dtype=np.float32)
        screen_x = np.asarray(self.screen_project(origin + np.asarray([1.0, 0.0, 0.0], dtype=np.float32))[:2], dtype=np.float32) - screen_origin
        screen_y = np.asarray(self.screen_project(origin + np.asarray([0.0, 1.0, 0.0], dtype=np.float32))[:2], dtype=np.float32) - screen_origin
        screen_z = np.asarray(self.screen_project(origin + np.asarray([0.0, 0.0, 1.0], dtype=np.float32))[:2], dtype=np.float32) - screen_origin
        axes = {"x": screen_x, "y": screen_y, "z": screen_z}
        if self.gizmo_drag in axes:
            axis_screen = axes[self.gizmo_drag]
            denom = float(np.dot(axis_screen, axis_screen))
            amount = 0.0 if denom <= 1e-6 else float(np.dot(delta, axis_screen) / denom)
            world = {
                "x": np.asarray([amount, 0.0, 0.0], dtype=np.float32),
                "y": np.asarray([0.0, amount, 0.0], dtype=np.float32),
                "z": np.asarray([0.0, 0.0, amount], dtype=np.float32),
            }[self.gizmo_drag]
        else:
            basis = np.column_stack((screen_x, screen_z))
            try:
                solution = np.linalg.lstsq(basis, delta, rcond=None)[0]
            except np.linalg.LinAlgError:
                solution = np.zeros(2, dtype=np.float32)
            world = np.asarray([solution[0], 0.0, solution[1]], dtype=np.float32)
        actor.offset = actor.offset + world.astype(np.float32)
        self.loading_inspector = True
        for i, var in enumerate(self.offset_vars):
            var.set(float(actor.offset[i]))
        self.loading_inspector = False
        self.gizmo_last_mouse = (x, y)
        self.update_bounds()
        self.draw()

    def draw(self) -> None:
        if isinstance(self.canvas, MotionGLFrame):
            self.frame_label.configure(text=f"{int(self.frame)} / {max(0, self.max_frames() - 1)}")
            if self.canvas.context_created and self.canvas.winfo_ismapped():
                self.canvas._display()
                self.record_render_frame()
            if hasattr(self, "timeline_bar"):
                self.timeline_bar.lift()
            self.draw_camera_axis_gizmo()
            return
        self.begin_scene()
        self.draw_floor_plane()
        actor_poses: list[tuple[Actor, np.ndarray, np.ndarray]] = []
        for actor in self.actors:
            if not actor.visible:
                continue
            pose = self.actor_pose_for_display(actor)
            if pose is None:
                self.draw_actor_placeholder(actor)
                continue
            positions, rotations = pose
            actor_poses.append((actor, positions, rotations))
        for actor, positions, _rotations in actor_poses:
            self.draw_actor_shadow(actor, positions)
        for actor, positions, rotations in actor_poses:
            self.draw_actor(actor, positions, rotations)
        self.draw_gizmo()
        self.finish_scene()
        self.record_render_frame()
        self.draw_camera_axis_gizmo()
        self.frame_label.configure(text=f"{int(self.frame)} / {max(0, self.max_frames() - 1)}")

    def init_gl_resources(self) -> None:
        if self.gl_lists:
            return
        self.gl_lists["capsule"] = self.create_unit_capsule_list(10)
        self.gl_lists["box"] = self.create_unit_box_list()
        self.gl_lists["sphere"] = self.create_unit_sphere_list()

    def create_unit_capsule_list(self, segments: int) -> int:
        list_id = GL.glGenLists(1)
        GL.glNewList(list_id, GL.GL_COMPILE)
        ring = []
        for i in range(segments):
            theta = math.tau * i / segments
            ring.append((math.cos(theta), math.sin(theta)))
        GL.glBegin(GL.GL_QUADS)
        for i in range(segments):
            y0, z0 = ring[i]
            y1, z1 = ring[(i + 1) % segments]
            GL.glNormal3f(0.0, y0, z0)
            GL.glVertex3f(0.0, y0, z0)
            GL.glVertex3f(1.0, y0, z0)
            GL.glNormal3f(0.0, y1, z1)
            GL.glVertex3f(1.0, y1, z1)
            GL.glVertex3f(0.0, y1, z1)
        GL.glEnd()
        for x, normal_x, order in ((0.0, -1.0, -1), (1.0, 1.0, 1)):
            GL.glBegin(GL.GL_TRIANGLE_FAN)
            GL.glNormal3f(normal_x, 0.0, 0.0)
            GL.glVertex3f(x, 0.0, 0.0)
            indices = range(segments, -1, -1) if order < 0 else range(segments + 1)
            for i in indices:
                y, z = ring[i % segments]
                GL.glVertex3f(x, y, z)
            GL.glEnd()
        GL.glEndList()
        return int(list_id)

    def create_unit_box_list(self) -> int:
        list_id = GL.glGenLists(1)
        GL.glNewList(list_id, GL.GL_COMPILE)
        face_specs = [
            ((1.0, 0.0, 0.0), [(0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5)]),
            ((-1.0, 0.0, 0.0), [(-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, -0.5)]),
            ((0.0, 1.0, 0.0), [(-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, -0.5)]),
            ((0.0, -1.0, 0.0), [(-0.5, -0.5, 0.5), (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5)]),
            ((0.0, 0.0, 1.0), [(-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)]),
            ((0.0, 0.0, -1.0), [(0.5, -0.5, -0.5), (-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5)]),
        ]
        GL.glBegin(GL.GL_QUADS)
        for normal, corners in face_specs:
            GL.glNormal3f(*normal)
            for corner in corners:
                GL.glVertex3f(*corner)
        GL.glEnd()
        GL.glEndList()
        return int(list_id)

    def create_unit_sphere_list(self) -> int:
        list_id = GL.glGenLists(1)
        GL.glNewList(list_id, GL.GL_COMPILE)
        quadric = GLU.gluNewQuadric()
        try:
            GLU.gluQuadricNormals(quadric, GLU.GLU_SMOOTH)
            GLU.gluSphere(quadric, 1.0, 14, 10)
        finally:
            GLU.gluDeleteQuadric(quadric)
        GL.glEndList()
        return int(list_id)

    def make_overlay_texture(
        self, text: str, texture_id: int, font_size: int = 14
    ) -> tuple[int, tuple[int, int], tuple[float, float]]:
        try:
            font = ImageFont.truetype("consola.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
        probe = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(probe)
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=1)
        text_w = max(1, int(math.ceil(bbox[2] - bbox[0])))
        text_h = max(1, int(math.ceil(bbox[3] - bbox[1])))
        tex_w = next_power_of_two(text_w)
        tex_h = next_power_of_two(text_h)
        image = Image.new("RGBA", (tex_w, tex_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.multiline_text((-bbox[0], -bbox[1]), text, font=font, fill=(255, 255, 255, 255), spacing=1)
        if not texture_id:
            texture_id = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA,
            tex_w,
            tex_h,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            image.tobytes(),
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return texture_id, (text_w, text_h), (text_w / tex_w, text_h / tex_h)

    def upload_overlay_texture(self, text: str) -> None:
        if text == self.overlay_texture_text and self.overlay_texture_id:
            return
        self.overlay_texture_id, self.overlay_texture_size, self.overlay_texture_uv = self.make_overlay_texture(
            text, self.overlay_texture_id, 14
        )
        self.overlay_texture_text = text

    def upload_contact_overlay_texture(self, text: str) -> None:
        if text == self.contact_overlay_texture_text and self.contact_overlay_texture_id:
            return
        (
            self.contact_overlay_texture_id,
            self.contact_overlay_texture_size,
            self.contact_overlay_texture_uv,
        ) = self.make_overlay_texture(text, self.contact_overlay_texture_id, 14)
        self.contact_overlay_texture_text = text

    def gl_draw_text_overlay(
        self,
        texture_id: int,
        texture_size: tuple[int, int],
        texture_uv: tuple[float, float],
        x0: float,
        y0: float,
        width: int,
        height: int,
    ) -> None:
        if not texture_id:
            return
        text_w, text_h = texture_size
        u, v = texture_uv
        x1, y1 = x0 + float(text_w), y0 + float(text_h)
        GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT | GL.GL_TEXTURE_BIT)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPushMatrix()
        GL.glLoadIdentity()
        GL.glOrtho(0.0, float(width), float(height), 0.0, -1.0, 1.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        GL.glLoadIdentity()
        GL.glColor4f(1.0, 1.0, 1.0, 1.0)
        GL.glBegin(GL.GL_QUADS)
        GL.glTexCoord2f(0.0, 0.0)
        GL.glVertex2f(x0, y0)
        GL.glTexCoord2f(u, 0.0)
        GL.glVertex2f(x1, y0)
        GL.glTexCoord2f(u, v)
        GL.glVertex2f(x1, y1)
        GL.glTexCoord2f(0.0, v)
        GL.glVertex2f(x0, y1)
        GL.glEnd()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopMatrix()
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPopMatrix()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glPopAttrib()

    def gl_draw_camera_overlay(self, width: int, height: int) -> None:
        text = self.camera_overlay_text()
        self.upload_overlay_texture(text)
        self.gl_draw_text_overlay(
            self.overlay_texture_id,
            self.overlay_texture_size,
            self.overlay_texture_uv,
            12.0,
            12.0,
            width,
            height,
        )

    def contact_metrics_text(self, actor_poses: list[tuple[Actor, np.ndarray, np.ndarray]]) -> str:
        if not (self.show_foot_height_var.get() or self.show_foot_contact_var.get()):
            return ""
        selected_id = self.selected_actor_id
        chosen: tuple[Actor, np.ndarray, np.ndarray] | None = None
        for item in actor_poses:
            if item[0].actor_id == selected_id:
                chosen = item
                break
        if chosen is None and actor_poses:
            chosen = actor_poses[0]
        if chosen is None:
            return ""
        actor, positions, rotations = chosen
        name_to_index = {name: i for i, name in enumerate(actor.bone_names)}
        sides = self.foot_contact_sides(name_to_index)
        if not sides:
            return ""
        config = self.foot_contact_config()
        frame = max(0, min(actor.frame_count - 1, int(round(float(self.frame)))))
        source_mode = bool(self.foot_contact_from_source_var.get())
        lines = [f"{actor.name} contact {'source' if source_mode else 'computed'}"]
        for side_index, contact_name, foot_index, toe_index in sides:
            height, point, _part = foot_lowest_ground_height(
                positions, rotations, foot_index, toe_index, config
            )
            speed = self.foot_contact_speed_mps(actor, frame, foot_index, toe_index, point)
            if source_mode:
                contact = self.actor_source_contact(actor, frame, side_index)
            else:
                contact = bool(is_foot_contact(height, speed, config))
            side = "L" if contact_name.endswith("L") else "R"
            mark = "on " if contact else "off"
            lines.append(
                f"{side} {mark}  h {height:+.3f} / {config.height_threshold_m:.3f}  "
                f"v {speed:.3f} / {config.horizontal_speed_threshold_mps:.3f}"
            )
        return "\n".join(lines)

    def gl_draw_contact_metrics_overlay(
        self, width: int, height: int, actor_poses: list[tuple[Actor, np.ndarray, np.ndarray]]
    ) -> None:
        text = self.contact_metrics_text(actor_poses)
        if not text:
            return
        self.upload_contact_overlay_texture(text)
        text_w, _text_h = self.contact_overlay_texture_size
        x0 = max(12.0, float(width - text_w - 116))
        self.gl_draw_text_overlay(
            self.contact_overlay_texture_id,
            self.contact_overlay_texture_size,
            self.contact_overlay_texture_uv,
            x0,
            34.0,
            width,
            height,
        )

    def gl_draw_screen_disc(self, x: float, y: float, radius: float, color: tuple[float, float, float, float]) -> None:
        GL.glColor4f(*color)
        GL.glBegin(GL.GL_TRIANGLE_FAN)
        GL.glVertex2f(float(x), float(y))
        for i in range(25):
            angle = math.tau * i / 24.0
            GL.glVertex2f(float(x + math.cos(angle) * radius), float(y + math.sin(angle) * radius))
        GL.glEnd()

    def gl_draw_screen_arrowhead(
        self,
        tail: tuple[float, float],
        tip: tuple[float, float],
        color: tuple[float, float, float, float],
    ) -> None:
        direction = np.asarray([tip[0] - tail[0], tip[1] - tail[1]], dtype=np.float32)
        length = float(np.linalg.norm(direction))
        if length <= 1e-6:
            self.gl_draw_screen_disc(tip[0], tip[1], 5.0, color)
            return
        direction /= length
        perp = np.asarray([-direction[1], direction[0]], dtype=np.float32)
        head_length = 17.0
        head_width = 12.5
        tip_v = np.asarray(tip, dtype=np.float32)
        base = tip_v - direction * head_length
        left = base + perp * (head_width * 0.5)
        right = base - perp * (head_width * 0.5)
        shadow = (max(0.0, color[0] * 0.45), max(0.0, color[1] * 0.45), max(0.0, color[2] * 0.45), color[3])
        highlight = (min(1.0, color[0] + 0.24), min(1.0, color[1] + 0.24), min(1.0, color[2] + 0.24), color[3])
        GL.glColor4f(*shadow)
        GL.glBegin(GL.GL_TRIANGLES)
        GL.glVertex2f(float(tip_v[0] + 1.5), float(tip_v[1] + 1.5))
        GL.glVertex2f(float(left[0] + 1.5), float(left[1] + 1.5))
        GL.glVertex2f(float(right[0] + 1.5), float(right[1] + 1.5))
        GL.glEnd()
        GL.glColor4f(*color)
        GL.glBegin(GL.GL_TRIANGLES)
        GL.glVertex2f(float(tip_v[0]), float(tip_v[1]))
        GL.glVertex2f(float(left[0]), float(left[1]))
        GL.glVertex2f(float(right[0]), float(right[1]))
        GL.glEnd()
        GL.glColor4f(*highlight)
        GL.glLineWidth(1.2)
        GL.glBegin(GL.GL_LINES)
        GL.glVertex2f(float(tip_v[0] - direction[0] * 3.0), float(tip_v[1] - direction[1] * 3.0))
        GL.glVertex2f(float((tip_v[0] + left[0]) * 0.5), float((tip_v[1] + left[1]) * 0.5))
        GL.glEnd()

    def gl_draw_gizmo_overlay(self, width: int, height: int) -> None:
        self.gizmo_hits.clear()
        actor = self.selected_model_actor()
        if actor is None or not self.gizmo_visible or self.playing:
            return
        origin = actor.offset.copy()
        length = max(0.25, min(0.85, self.extent * 0.18))
        axes = {
            "x": (np.asarray([length, 0.0, 0.0], dtype=np.float32), (1.0, 0.25, 0.28, 0.96)),
            "y": (np.asarray([0.0, length, 0.0], dtype=np.float32), (0.34, 0.86, 0.42, 0.96)),
            "z": (np.asarray([0.0, 0.0, length], dtype=np.float32), (0.39, 0.64, 1.0, 0.96)),
        }
        o = self.screen_project(origin)
        if not (math.isfinite(o[0]) and math.isfinite(o[1])):
            return

        points: dict[str, tuple[float, float]] = {}
        for name, (delta, _color) in axes.items():
            tip = self.screen_project(origin + delta)
            if math.isfinite(tip[0]) and math.isfinite(tip[1]):
                points[name] = (tip[0], tip[1])

        ox, oy = o[0], o[1]
        self.gizmo_hits["plane"] = (ox - 13.0, oy - 13.0, ox + 13.0, oy + 13.0)
        for name, (tx, ty) in points.items():
            self.gizmo_hits[name] = (
                min(ox, tx) - 9.0,
                min(oy, ty) - 9.0,
                max(ox, tx) + 9.0,
                max(oy, ty) + 9.0,
            )

        GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT | GL.GL_LINE_BIT | GL.GL_CURRENT_BIT)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_LINE_SMOOTH)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPushMatrix()
        GL.glLoadIdentity()
        GL.glOrtho(0.0, float(width), float(height), 0.0, -1.0, 1.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        GL.glLoadIdentity()

        GL.glLineWidth(3.5)
        for name, (_delta, color) in axes.items():
            tip = points.get(name)
            if tip is None:
                continue
            GL.glColor4f(*color)
            GL.glBegin(GL.GL_LINES)
            GL.glVertex2f(float(ox), float(oy))
            GL.glVertex2f(float(tip[0]), float(tip[1]))
            GL.glEnd()
            self.gl_draw_screen_arrowhead((ox, oy), tip, color)
        self.gl_draw_screen_disc(ox, oy, 8.0, (1.0, 1.0, 1.0, 0.9))
        self.gl_draw_screen_disc(ox, oy, 3.3, (0.08, 0.09, 0.11, 0.95))

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopMatrix()
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPopMatrix()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glPopAttrib()

    def gl_call_transformed(
        self,
        list_name: str,
        origin: np.ndarray,
        axis_x: np.ndarray,
        axis_y: np.ndarray,
        axis_z: np.ndarray,
    ) -> None:
        list_id = self.gl_lists.get(list_name)
        if list_id is None:
            self.init_gl_resources()
            list_id = self.gl_lists[list_name]
        matrix = np.asarray(
            [
                [axis_x[0], axis_y[0], axis_z[0], origin[0]],
                [axis_x[1], axis_y[1], axis_z[1], origin[1]],
                [axis_x[2], axis_y[2], axis_z[2], origin[2]],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        GL.glPushMatrix()
        GL.glMultMatrixf(matrix.T)
        GL.glCallList(list_id)
        GL.glPopMatrix()

    def render_gl(self) -> None:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        GL.glViewport(0, 0, width, height)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluPerspective(45.0, width / height, 0.001, 500.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        cam, target, up = self.camera_render_view()
        GLU.gluLookAt(
            float(cam[0]),
            float(cam[1]),
            float(cam[2]),
            float(target[0]),
            float(target[1]),
            float(target[2]),
            float(up[0]),
            float(up[1]),
            float(up[2]),
        )
        self.gl_modelview = np.asarray(GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX), dtype=np.float64)
        self.gl_projection = np.asarray(GL.glGetDoublev(GL.GL_PROJECTION_MATRIX), dtype=np.float64)
        self.gl_viewport = np.asarray(GL.glGetIntegerv(GL.GL_VIEWPORT), dtype=np.int32)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, (-float(LIGHT_DIR[0]), -float(LIGHT_DIR[1]), -float(LIGHT_DIR[2]), 0.0))

        self.gl_draw_floor()
        if self.penetration_mode:
            self.gl_draw_penetration_line()
        actor_poses: list[tuple[Actor, np.ndarray, np.ndarray]] = []
        for actor in self.actors:
            if not actor.visible:
                continue
            pose = self.actor_pose_for_display(actor)
            if pose is None:
                continue
            positions, rotations = pose
            actor_poses.append((actor, positions, rotations))
        for actor, positions, rotations in actor_poses:
            self.gl_draw_actor_shadows_cached(actor, positions, rotations)
        for actor, positions, rotations in actor_poses:
            self.gl_draw_actor(actor, positions, rotations)
        if self.show_foot_height_var.get() or self.show_foot_contact_var.get():
            for actor, positions, rotations in actor_poses:
                self.gl_draw_foot_contact_debug(actor, positions, rotations)
        if self.show_trajectory_var.get():
            self.gl_draw_controller_trajectory()
        self.gl_draw_gizmo_overlay(width, height)
        self.gl_draw_contact_metrics_overlay(width, height, actor_poses)
        self.gl_draw_camera_overlay(width, height)
        GL.glFlush()

    def hex_to_rgb(self, color: str) -> tuple[float, float, float]:
        return (
            int(color[1:3], 16) / 255.0,
            int(color[3:5], 16) / 255.0,
            int(color[5:7], 16) / 255.0,
        )

    def gl_color(self, color: str, alpha: float = 1.0) -> None:
        r, g, b = self.hex_to_rgb(color)
        GL.glColor4f(r, g, b, alpha)

    def gl_draw_floor(self) -> None:
        GL.glDisable(GL.GL_LIGHTING)
        self.gl_color(FLOOR, 1.0)
        size = max(18.0, self.extent * 5.0)
        GL.glBegin(GL.GL_QUADS)
        GL.glVertex3f(-size, 0.0, -size)
        GL.glVertex3f(size, 0.0, -size)
        GL.glVertex3f(size, 0.0, size)
        GL.glVertex3f(-size, 0.0, size)
        GL.glEnd()
        GL.glEnable(GL.GL_LIGHTING)

    def gl_draw_penetration_line(self) -> None:
        target = self.camera_target()
        right = self.camera_right_vector()
        size = max(12.0, self.extent * 5.0)
        a = np.asarray([target[0], 0.004, target[2]], dtype=np.float32) - right * size
        b = np.asarray([target[0], 0.004, target[2]], dtype=np.float32) + right * size
        GL.glDisable(GL.GL_LIGHTING)
        GL.glLineWidth(2.0)
        GL.glColor4f(1.0, 1.0, 1.0, 0.92)
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f(float(a[0]), float(a[1]), float(a[2]))
        GL.glVertex3f(float(b[0]), float(b[1]), float(b[2]))
        GL.glEnd()
        GL.glEnable(GL.GL_LIGHTING)

    def gl_draw_actor(self, actor: Actor, positions: np.ndarray, rotations: np.ndarray) -> None:
        if actor.clip is None:
            return
        if self.volumes_var.get():
            self.gl_draw_volumes(actor, positions, rotations)
            return
        self.gl_draw_skeleton(actor, positions)

    def gl_draw_skeleton(self, actor: Actor, positions: np.ndarray) -> None:
        GL.glDisable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_LINE_SMOOTH)
        GL.glLineWidth(3.0 if actor.kind == "model" else 3.6)
        self.gl_color(actor.color, 1.0)
        name_to_index = {name: i for i, name in enumerate(actor.bone_names)}
        chains = [
            ("pelvis", "spine_01", "spine_02", "spine_03", "spine_04", "spine_05", "neck_01", "neck_02", "head"),
            ("spine_05", "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l"),
            ("spine_05", "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r"),
            ("pelvis", "thigh_l", "calf_l", "foot_l", "ball_l"),
            ("pelvis", "thigh_r", "calf_r", "foot_r", "ball_r"),
        ]
        for chain in chains:
            GL.glBegin(GL.GL_LINE_STRIP)
            for name in chain:
                index = name_to_index.get(name)
                if index is None:
                    continue
                p = positions[index]
                GL.glVertex3f(float(p[0]), float(p[1]), float(p[2]))
            GL.glEnd()
        GL.glEnable(GL.GL_LIGHTING)

    def gl_draw_volumes(
        self,
        actor: Actor,
        positions: np.ndarray,
        rotations: np.ndarray,
        color: str | None = None,
        alpha: float = 1.0,
        rounded_caps: bool = True,
    ) -> None:
        name_to_index = {name: i for i, name in enumerate(actor.bone_names)}
        draw_color = actor.color if color is None else color
        self.gl_color(draw_color, alpha)
        self.gl_draw_torso_boxes(positions, name_to_index, draw_color, alpha)
        self.gl_draw_hip_spheres(positions, name_to_index, draw_color, alpha)
        self.gl_draw_joint_spheres(positions, name_to_index, draw_color, alpha)
        rounded = self.rounded_capsule_specs() if rounded_caps else set()
        for a_name, b_name, radius_cm in self.capsule_specs():
            a = name_to_index.get(a_name)
            b = name_to_index.get(b_name)
            if a is not None and b is not None:
                self.gl_draw_capsule(
                    positions[a],
                    positions[b],
                    radius_cm * 0.01,
                    draw_color,
                    alpha,
                    (a_name, b_name) in rounded,
                )
        self.gl_draw_head_sphere(positions, name_to_index, draw_color, alpha)
        for ankle_name, toe_name in (("foot_l", "ball_l"), ("foot_r", "ball_r")):
            ankle = name_to_index.get(ankle_name)
            toe = name_to_index.get(toe_name)
            if ankle is not None and toe is not None:
                self.gl_draw_foot_boxes(positions, rotations, ankle, toe, draw_color, alpha)
        for hand_name, mid_name, parent_name in (
            ("hand_l", "middle_03_l", "lowerarm_l"),
            ("hand_r", "middle_03_r", "lowerarm_r"),
        ):
            hand = name_to_index.get(hand_name)
            if hand is not None:
                self.gl_draw_hand_box(
                    positions,
                    rotations,
                    hand,
                    name_to_index.get(mid_name),
                    name_to_index.get(parent_name),
                    draw_color,
                    alpha,
                )

    def unit_vec(self, vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-6:
            return fallback.astype(np.float32)
        return (vector / norm).astype(np.float32)

    def body_box_frame(
        self, positions: np.ndarray, name_to_index: dict[str, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        pelvis = name_to_index.get("pelvis")
        chest = name_to_index.get("spine_05")
        if chest is None:
            chest = name_to_index.get("spine_04")
        if pelvis is None or chest is None:
            return None
        up = self.unit_vec(sub3(positions[chest], positions[pelvis]), np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
        side_raw = None
        for left_name, right_name in (("clavicle_l", "clavicle_r"), ("upperarm_l", "upperarm_r"), ("thigh_l", "thigh_r")):
            left = name_to_index.get(left_name)
            right = name_to_index.get(right_name)
            if left is not None and right is not None:
                side_raw = sub3(positions[left], positions[right])
                break
        if side_raw is None:
            side_raw = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        side_raw = side_raw - up * dot3(side_raw, up)
        side = self.unit_vec(side_raw, np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
        forward = self.unit_vec(np.cross(side, up), np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
        side = self.unit_vec(np.cross(up, forward), side)
        return side, up, forward

    def hip_width_hint(self, positions: np.ndarray, name_to_index: dict[str, int]) -> float:
        left = name_to_index.get("thigh_l")
        right = name_to_index.get("thigh_r")
        if left is None or right is None:
            return 0.30
        width = float(np.linalg.norm(sub3(positions[left], positions[right]))) + 0.09
        return max(0.25, min(0.32, width))

    def gl_draw_torso_boxes(
        self, positions: np.ndarray, name_to_index: dict[str, int], color: str, alpha: float = 1.0
    ) -> None:
        frame = self.body_box_frame(positions, name_to_index)
        pelvis = name_to_index.get("pelvis")
        if frame is None or pelvis is None:
            return
        side, up, forward = frame
        base_width = self.hip_width_hint(positions, name_to_index)
        chest = name_to_index.get("spine_05")
        if chest is None:
            return
        base = add3(positions[pelvis], mul3(up, 0.060))
        top = add3(positions[chest], mul3(up, -0.030))
        torso_height = max(0.30, min(0.48, float(np.linalg.norm(sub3(top, base)))))
        gap = 0.0
        box_height = max(0.080, (torso_height - gap * 2.0) / 3.0)
        widths = (base_width * 0.86, base_width * 0.96, base_width * 1.06)
        depths = (base_width * 0.54, base_width * 0.57, base_width * 0.55)
        for index, (width, depth) in enumerate(zip(widths, depths)):
            center = add3(base, mul3(up, box_height * 0.5 + index * (box_height + gap)))
            dims = (width, box_height, depth)
            self.gl_draw_box(center, side, up, forward, dims, color, alpha)
        left = name_to_index.get("clavicle_l")
        right = name_to_index.get("clavicle_r")
        if left is not None and right is not None:
            left_pos = positions[left]
            right_pos = positions[right]
            shoulder_center = (left_pos + right_pos) * 0.5
            shoulder_width = max(base_width * 1.16, min(0.46, float(np.linalg.norm(sub3(left_pos, right_pos))) + 0.09))
            torso_top = add3(base, mul3(up, box_height * 3.0 + gap * 2.0))
            connector_top = add3(shoulder_center, mul3(up, -0.018))
            connector_height = max(0.052, dot3(sub3(connector_top, torso_top), up) + 0.030)
            connector_center = add3(torso_top, mul3(up, connector_height * 0.5 - 0.006))
            connector_width = (widths[-1] + shoulder_width) * 0.5
            self.gl_draw_box(
                connector_center,
                side,
                up,
                forward,
                (connector_width, connector_height, base_width * 0.34),
                color,
                alpha,
            )
            self.gl_draw_box(
                add3(shoulder_center, mul3(up, 0.006)),
                side,
                up,
                forward,
                (shoulder_width, 0.048, base_width * 0.32),
                color,
                alpha,
            )

    def gl_draw_hip_spheres(
        self, positions: np.ndarray, name_to_index: dict[str, int], color: str, alpha: float = 1.0
    ) -> None:
        frame = self.body_box_frame(positions, name_to_index)
        if frame is None:
            return
        side, up, forward = frame
        pelvis = name_to_index.get("pelvis")
        if pelvis is not None:
            pelvis_radius = 0.058
            pelvis_center = add3(positions[pelvis], mul3(up, 0.014))
            self.gl_color(color, alpha)
            self.gl_call_transformed(
                "sphere",
                pelvis_center,
                side * pelvis_radius,
                up * pelvis_radius,
                forward * pelvis_radius,
            )
        radius = 0.066
        for name in ("thigh_l", "thigh_r"):
            index = name_to_index.get(name)
            if index is None:
                continue
            self.gl_color(color, alpha)
            self.gl_call_transformed("sphere", positions[index], side * radius, up * radius, forward * radius)

    def gl_draw_joint_spheres(
        self, positions: np.ndarray, name_to_index: dict[str, int], color: str, alpha: float = 1.0
    ) -> None:
        specs = (
            ("foot_l", 0.046),
            ("foot_r", 0.046),
            ("calf_l", 0.058),
            ("calf_r", 0.058),
            ("lowerarm_l", 0.041),
            ("lowerarm_r", 0.041),
        )
        self.gl_color(color, alpha)
        for name, radius in specs:
            index = name_to_index.get(name)
            if index is None:
                continue
            self.gl_call_transformed(
                "sphere",
                positions[index],
                np.asarray([radius, 0.0, 0.0], dtype=np.float32),
                np.asarray([0.0, radius, 0.0], dtype=np.float32),
                np.asarray([0.0, 0.0, radius], dtype=np.float32),
            )

    def gl_head_sphere(self, positions: np.ndarray, name_to_index: dict[str, int]) -> tuple[np.ndarray, float] | None:
        head = name_to_index.get("head")
        neck = name_to_index.get("neck_02")
        if head is None:
            return None
        head_pos = positions[head]
        if neck is None:
            return head_pos, 0.11
        neck_pos = positions[neck]
        delta = sub3(head_pos, neck_pos)
        length = float(np.linalg.norm(delta))
        if length <= 1e-6:
            return head_pos, 0.11
        center = add3(neck_pos, mul3(delta / length, length * 0.62))
        radius = max(0.095, min(0.125, length * 0.55))
        return center, radius

    def gl_draw_head_sphere(
        self, positions: np.ndarray, name_to_index: dict[str, int], color: str, alpha: float = 1.0
    ) -> None:
        sphere = self.gl_head_sphere(positions, name_to_index)
        if sphere is None:
            return
        center, radius = sphere
        self.gl_color(color, alpha)
        self.gl_call_transformed(
            "sphere",
            center,
            np.asarray([radius, 0.0, 0.0], dtype=np.float32),
            np.asarray([0.0, radius, 0.0], dtype=np.float32),
            np.asarray([0.0, 0.0, radius], dtype=np.float32),
        )

    def gl_draw_capsule(
        self, a: np.ndarray, b: np.ndarray, radius: float, color: str, alpha: float = 1.0, rounded: bool = False
    ) -> None:
        basis = self.segment_basis(a, b)
        if basis is None:
            return
        axis, side, up = basis
        self.gl_color(color, alpha)
        length = max(1e-6, float(np.linalg.norm(sub3(b, a))))
        self.gl_call_transformed("capsule", a, axis * length, side * radius, up * radius)
        if rounded:
            self.gl_call_transformed("sphere", a, side * radius, axis * radius, up * radius)
            self.gl_call_transformed("sphere", b, side * radius, axis * radius, up * radius)

    def gl_draw_box(
        self,
        center: np.ndarray,
        axis_x: np.ndarray,
        axis_y: np.ndarray,
        axis_z: np.ndarray,
        dims: tuple[float, float, float],
        color: str,
        alpha: float = 1.0,
    ) -> None:
        self.gl_color(color, alpha)
        self.gl_call_transformed("box", center, axis_x * dims[0], axis_y * dims[1], axis_z * dims[2])

    def foot_toe_dimensions(self, _auto_foot_length: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        try:
            foot_length = float(self.foot_length_var.get())
            foot_width = float(self.foot_width_var.get())
            foot_height = float(self.foot_height_var.get())
            toe_length = float(self.toe_length_var.get())
            toe_width = float(self.toe_width_var.get())
            toe_height = float(self.toe_height_var.get())
        except tk.TclError:
            foot_length = DEFAULT_FOOT_LENGTH
            foot_width = DEFAULT_FOOT_WIDTH
            foot_height = DEFAULT_FOOT_HEIGHT
            toe_length = DEFAULT_TOE_LENGTH
            toe_width = DEFAULT_TOE_WIDTH
            toe_height = DEFAULT_TOE_HEIGHT
        return (
            (max(0.001, foot_length), max(0.001, foot_width), max(0.001, foot_height)),
            (max(0.001, toe_length), max(0.001, toe_width), max(0.001, toe_height)),
        )

    def hand_dimensions(self) -> tuple[float, float, float]:
        try:
            return (
                max(0.001, float(self.hand_length_var.get())),
                max(0.001, float(self.hand_width_var.get())),
                max(0.001, float(self.hand_height_var.get())),
            )
        except tk.TclError:
            return DEFAULT_HAND_LENGTH, DEFAULT_HAND_WIDTH, DEFAULT_HAND_HEIGHT

    def foot_box_specs(
        self, positions: np.ndarray, rotations: np.ndarray, ankle: int, toe: int
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float]]]:
        foot = positions[ankle]
        toe_pos = positions[toe]
        up = rotations[ankle, 0].copy()
        forward = rotations[ankle, 1].copy()
        side_axis = rotations[ankle, 2].copy()
        toe_vector = sub3(toe_pos, foot)
        if dot3(forward, toe_vector) < 0:
            forward = mul3(forward, -1)
        if float(up[1]) < 0:
            up = mul3(up, -1)
        ball_distance = max(0.10, min(0.28, abs(dot3(toe_vector, forward))))
        heel_length = 0.07
        auto_length = ball_distance + heel_length
        foot_dims, toe_dims = self.foot_toe_dimensions(auto_length)
        length = foot_dims[0]
        heel_back = add3(toe_pos, mul3(forward, -length))
        center = add3(add3(heel_back, mul3(forward, length * 0.5)), mul3(up, -0.006))
        toe_forward = rotations[toe, 0].copy()
        toe_up = rotations[toe, 1].copy()
        toe_side = rotations[toe, 2].copy()
        if dot3(toe_forward, toe_vector) < 0:
            toe_forward = mul3(toe_forward, -1)
        if float(toe_up[1]) < 0:
            toe_up = mul3(toe_up, -1)
        toe_center = add3(add3(toe_pos, mul3(toe_forward, toe_dims[0] * 0.5)), mul3(toe_up, -0.006))
        return [
            (center, forward, side_axis, up, foot_dims),
            (toe_center, toe_forward, toe_side, toe_up, toe_dims),
        ]

    def hand_box_spec(
        self,
        positions: np.ndarray,
        rotations: np.ndarray,
        hand: int,
        mid: int | None,
        parent: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float]]:
        hand_pos = positions[hand]
        forward = rotations[hand, 0].copy()
        side_axis = rotations[hand, 1].copy()
        up = rotations[hand, 2].copy()
        if mid is not None:
            finger_vector = sub3(positions[mid], hand_pos)
            if dot3(forward, finger_vector) < 0:
                forward = mul3(forward, -1)
        elif parent is not None:
            forearm_vector = sub3(hand_pos, positions[parent])
            length = float(np.linalg.norm(forearm_vector))
            if length > 1e-6:
                forward = forearm_vector / length
                side_axis = np.cross(up, forward)
                side_length = float(np.linalg.norm(side_axis))
                if side_length > 1e-6:
                    side_axis = side_axis / side_length
                    up = np.cross(forward, side_axis)
        if float(up[1]) < 0:
            up = mul3(up, -1)
        dims = self.hand_dimensions()
        center = add3(add3(hand_pos, mul3(forward, dims[0] * 0.5)), mul3(up, -0.0025))
        return center, forward, up, side_axis, dims

    def gl_draw_foot_boxes(
        self, positions: np.ndarray, rotations: np.ndarray, ankle: int, toe: int, color: str, alpha: float = 1.0
    ) -> None:
        for center, axis_x, axis_y, axis_z, dims in self.foot_box_specs(positions, rotations, ankle, toe):
            self.gl_draw_box(center, axis_x, axis_y, axis_z, dims, color, alpha)

    def gl_draw_hand_box(
        self,
        positions: np.ndarray,
        rotations: np.ndarray,
        hand: int,
        mid: int | None,
        parent: int | None,
        color: str,
        alpha: float = 1.0,
    ) -> None:
        center, axis_x, axis_y, axis_z, dims = self.hand_box_spec(positions, rotations, hand, mid, parent)
        self.gl_draw_box(center, axis_x, axis_y, axis_z, dims, color, alpha)

    def gl_draw_controller_trajectory(self) -> None:
        actor = self.selected_controller_actor()
        if actor is None:
            return
        if not self.controller_enabled_var.get():
            self.gl_draw_authored_root_trajectory(actor)
            return
        root = self.actor_root_world(actor)
        if root is None:
            return
        settings = self.controller_settings()
        sample = self.controller_sample if self.controller_sample.connected else ControllerSample(
            False, np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32), 0.0
        )
        state = RootMotionState(actor.controller_root_offset.copy(), actor.controller_velocity.copy(), float(actor.controller_yaw))
        movement_yaw = self.camera_movement_yaw()
        root_base = root - actor.controller_root_offset
        steps = max(1, min(48, int(math.ceil(settings.trajectory_seconds / settings.trajectory_step_seconds))))
        dt = settings.trajectory_seconds / steps
        GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT | GL.GL_LINE_BIT | GL.GL_POINT_BIT)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_LINE_SMOOTH)
        try:
            GL.glColor4f(0.42, 1.0, 0.78, 0.82)
            GL.glLineWidth(2.0)
            last = np.asarray([root[0], FOOT_CONTACT_CONFIG.ground_y + 0.018, root[2]], dtype=np.float32)
            for step in range(1, steps + 1):
                root_turn = np.asarray([sample.triggers, 0.0], dtype=np.float32)
                state = integrate_controller_root_motion(state, sample.move, root_turn, dt, settings, movement_yaw=movement_yaw)
                point = root_base + state.offset
                point = np.asarray([point[0], FOOT_CONTACT_CONFIG.ground_y + 0.018, point[2]], dtype=np.float32)
                if step % 2 == 1:
                    GL.glBegin(GL.GL_LINES)
                    GL.glVertex3f(float(last[0]), float(last[1]), float(last[2]))
                    GL.glVertex3f(float(point[0]), float(point[1]), float(point[2]))
                    GL.glEnd()
                GL.glPointSize(5.0 if step < steps else 7.0)
                GL.glBegin(GL.GL_POINTS)
                GL.glVertex3f(float(point[0]), float(point[1]), float(point[2]))
                GL.glEnd()
                last = point
        finally:
            GL.glDepthMask(GL.GL_TRUE)
            GL.glPopAttrib()

    def authored_root_trajectory_points(self, actor: Actor) -> list[np.ndarray]:
        if actor.clip is None:
            return []
        settings = self.controller_settings()
        frame = max(0, min(actor.clip.T - 1, int(round(float(self.frame)))))
        fps = max(1.0, float(actor.clip.fps))
        steps = max(1, min(48, int(math.ceil(settings.trajectory_seconds / settings.trajectory_step_seconds))))
        dt = settings.trajectory_seconds / steps
        points = []
        for step in range(0, steps + 1):
            sample_frame = max(0, min(actor.clip.T - 1, int(round(frame + step * dt * fps))))
            root = actor.clip.root_pos[sample_frame].detach().cpu().numpy().astype(np.float32) + actor.offset
            points.append(np.asarray([root[0], FOOT_CONTACT_CONFIG.ground_y + 0.018, root[2]], dtype=np.float32))
        return points

    def gl_draw_authored_root_trajectory(self, actor: Actor) -> None:
        points = self.authored_root_trajectory_points(actor)
        if len(points) < 2:
            return
        GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT | GL.GL_LINE_BIT | GL.GL_POINT_BIT)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_LINE_SMOOTH)
        try:
            GL.glColor4f(0.42, 1.0, 0.78, 0.82)
            GL.glLineWidth(2.0)
            for idx in range(1, len(points)):
                if idx % 2 == 1:
                    last = points[idx - 1]
                    point = points[idx]
                    GL.glBegin(GL.GL_LINES)
                    GL.glVertex3f(float(last[0]), float(last[1]), float(last[2]))
                    GL.glVertex3f(float(point[0]), float(point[1]), float(point[2]))
                    GL.glEnd()
                GL.glPointSize(5.0 if idx < len(points) - 1 else 7.0)
                point = points[idx]
                GL.glBegin(GL.GL_POINTS)
                GL.glVertex3f(float(point[0]), float(point[1]), float(point[2]))
                GL.glEnd()
        finally:
            GL.glDepthMask(GL.GL_TRUE)
            GL.glPopAttrib()

    def foot_contact_sides(self, name_to_index: dict[str, int]) -> list[tuple[int, str, int, int]]:
        sides = []
        for side_index, (_contact_name, foot_name, toe_name) in enumerate(
            (("contactL", "foot_l", "ball_l"), ("contactR", "foot_r", "ball_r"))
        ):
            foot_index = name_to_index.get(foot_name)
            toe_index = name_to_index.get(toe_name)
            if foot_index is not None and toe_index is not None:
                sides.append((side_index, _contact_name, foot_index, toe_index))
        return sides

    def foot_contact_config(self) -> FootContactConfig:
        try:
            return replace(
                FOOT_CONTACT_CONFIG,
                foot_length=max(0.001, float(self.foot_length_var.get())),
                foot_width=max(0.001, float(self.foot_width_var.get())),
                foot_height=max(0.001, float(self.foot_height_var.get())),
                toe_length=max(0.001, float(self.toe_length_var.get())),
                toe_width=max(0.001, float(self.toe_width_var.get())),
                toe_height=max(0.001, float(self.toe_height_var.get())),
            )
        except (AttributeError, tk.TclError, ValueError):
            return FOOT_CONTACT_CONFIG

    def foot_contact_point_for_frame(
        self, actor: Actor, frame: int, foot_index: int, toe_index: int
    ) -> np.ndarray | None:
        pose = actor.pose_for_frame(frame, generate=False)
        if pose is None:
            return None
        positions, rotations = pose
        _height, point, _part = foot_lowest_ground_height(
            positions, rotations, foot_index, toe_index, self.foot_contact_config()
        )
        return point.astype(np.float32)

    def foot_contact_speed_mps(
        self,
        actor: Actor,
        frame: int,
        foot_index: int,
        toe_index: int,
        _current_point: np.ndarray,
    ) -> float:
        if actor.clip is None or actor.frame_count <= 1:
            return 0.0
        prev_frame = 0 if frame <= 0 else frame - 1
        cur_frame = 1 if frame <= 0 else frame
        prev_pose = actor.pose_for_frame(prev_frame, generate=False)
        cur_pose = actor.pose_for_frame(cur_frame, generate=False)
        if prev_pose is None or cur_pose is None:
            return 0.0
        prev_positions, prev_rotations = prev_pose
        cur_positions, cur_rotations = cur_pose
        distance, _prev_point, _cur_point, _part = foot_sole_slide_distance(
            prev_positions,
            prev_rotations,
            cur_positions,
            cur_rotations,
            foot_index,
            toe_index,
            self.foot_contact_config(),
        )
        return float(distance * float(actor.clip.fps))

    def actor_source_contact(self, actor: Actor, frame: int, side_index: int) -> bool:
        if actor.kind == "model":
            if side_index >= 2 or not actor.generated_contacts:
                return False
            frame = max(0, min(int(frame), len(actor.generated_contacts) - 1))
            contacts = actor.generated_contacts[frame]
            return bool(float(contacts[side_index]) >= 0.5)
        contacts = actor.source_contacts
        if contacts is None or contacts.size == 0 or side_index >= contacts.shape[1]:
            return False
        frame = max(0, min(int(frame), contacts.shape[0] - 1))
        return bool(contacts[frame, side_index])

    def gl_draw_debug_point(self, point: np.ndarray, size: float = 7.0) -> None:
        GL.glPointSize(float(size))
        GL.glBegin(GL.GL_POINTS)
        GL.glVertex3f(float(point[0]), float(point[1]), float(point[2]))
        GL.glEnd()

    def gl_draw_debug_cross(self, point: np.ndarray, radius: float = 0.025) -> None:
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f(float(point[0] - radius), float(point[1]), float(point[2]))
        GL.glVertex3f(float(point[0] + radius), float(point[1]), float(point[2]))
        GL.glVertex3f(float(point[0]), float(point[1]), float(point[2] - radius))
        GL.glVertex3f(float(point[0]), float(point[1]), float(point[2] + radius))
        GL.glEnd()

    def gl_draw_wire_box(
        self,
        center: np.ndarray,
        axis_x: np.ndarray,
        axis_y: np.ndarray,
        axis_z: np.ndarray,
        dims: tuple[float, float, float],
    ) -> None:
        hx, hy, hz = float(dims[0]) * 0.5, float(dims[1]) * 0.5, float(dims[2]) * 0.5
        corners = []
        for x, y, z in (
            (-hx, -hy, -hz),
            (hx, -hy, -hz),
            (hx, hy, -hz),
            (-hx, hy, -hz),
            (-hx, -hy, hz),
            (hx, -hy, hz),
            (hx, hy, hz),
            (-hx, hy, hz),
        ):
            corners.append(add3(add3(add3(center, mul3(axis_x, x)), mul3(axis_y, y)), mul3(axis_z, z)))
        edges = (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )
        GL.glBegin(GL.GL_LINES)
        for a, b in edges:
            pa = corners[a]
            pb = corners[b]
            GL.glVertex3f(float(pa[0]), float(pa[1]), float(pa[2]))
            GL.glVertex3f(float(pb[0]), float(pb[1]), float(pb[2]))
        GL.glEnd()

    def gl_draw_foot_contact_debug(self, actor: Actor, positions: np.ndarray, rotations: np.ndarray) -> None:
        if actor.clip is None:
            return
        name_to_index = {name: i for i, name in enumerate(actor.bone_names)}
        sides = self.foot_contact_sides(name_to_index)
        if not sides:
            return
        frame = max(0, min(actor.frame_count - 1, int(round(float(self.frame)))))
        draw_height = bool(self.show_foot_height_var.get())
        draw_contact = bool(self.show_foot_contact_var.get())
        config = self.foot_contact_config()
        GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT | GL.GL_LINE_BIT | GL.GL_POINT_BIT)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_LINE_SMOOTH)
        try:
            from_source = bool(self.foot_contact_from_source_var.get())
            for side_index, _contact_name, foot_index, toe_index in sides:
                height, point, _part = foot_lowest_ground_height(
                    positions, rotations, foot_index, toe_index, config
                )
                speed = self.foot_contact_speed_mps(actor, frame, foot_index, toe_index, point)
                if from_source:
                    contact = self.actor_source_contact(actor, frame, side_index)
                else:
                    contact = bool(is_foot_contact(height, speed, config))
                color = (0.28, 1.0, 0.46, 0.98) if contact else (1.0, 0.16, 0.18, 0.98)
                if draw_contact:
                    GL.glLineWidth(3.0)
                    GL.glColor4f(*color)
                    for part, center, axis_x, axis_y, axis_z, dims in foot_toe_box_specs(
                        positions, rotations, foot_index, toe_index, config
                    ):
                        _lowest_point, _signed_height = box_lowest_point_signed_height(
                            center, axis_x, axis_y, axis_z, dims, config.ground_y
                        )
                        self.gl_draw_wire_box(center, axis_x, axis_y, axis_z, dims)
                if draw_height:
                    ground = np.asarray(
                        [point[0], config.ground_y + 0.003, point[2]], dtype=np.float32
                    )
                    line_color = (1.0, 0.22, 0.22, 0.96) if height < 0.0 else (0.62, 0.94, 1.0, 0.96)
                    GL.glLineWidth(2.5)
                    GL.glColor4f(*line_color)
                    GL.glBegin(GL.GL_LINES)
                    GL.glVertex3f(float(point[0]), float(point[1]), float(point[2]))
                    GL.glVertex3f(float(ground[0]), float(ground[1]), float(ground[2]))
                    GL.glEnd()
                    GL.glColor4f(1.0, 1.0, 1.0, 0.95)
                    self.gl_draw_debug_point(point, 7.0)
                    GL.glColor4f(line_color[0], line_color[1], line_color[2], 0.86)
                    self.gl_draw_debug_cross(ground, 0.035)
        finally:
            GL.glDepthMask(GL.GL_TRUE)
            GL.glPopAttrib()

    def actor_shadow_frame_key(self, actor: Actor) -> int:
        max_frame = max(0, actor.frame_count - 1)
        key = min(max_frame, int(self.frame))
        if actor.kind == "model":
            generated_count = min(len(actor.generated_pos), len(actor.generated_rot))
            if generated_count > 0:
                key = min(key, generated_count - 1)
        return max(0, key)

    def gl_draw_actor_shadows_cached(self, actor: Actor, positions: np.ndarray, rotations: np.ndarray) -> None:
        if actor.clip is None or not self.volumes_var.get():
            return
        if self.playing or (actor.kind == "model" and actor.controller_active):
            cached = self.shadow_lists.pop(actor.actor_id, None)
            if cached is not None:
                _cached_frame, _cached_generation, list_id = cached
                GL.glDeleteLists(list_id, 1)
            self.gl_draw_actor_shadows(actor, positions, rotations)
            return
        frame_key = self.actor_shadow_frame_key(actor)
        cached = self.shadow_lists.get(actor.actor_id)
        if cached is not None:
            cached_frame, cached_generation, list_id = cached
            if cached_frame == frame_key and cached_generation == self.shadow_generation:
                GL.glCallList(list_id)
                return
            GL.glDeleteLists(list_id, 1)
        list_id = int(GL.glGenLists(1))
        GL.glNewList(list_id, GL.GL_COMPILE)
        self.gl_draw_actor_shadows(actor, positions, rotations)
        GL.glEndList()
        self.shadow_lists[actor.actor_id] = (frame_key, self.shadow_generation, list_id)
        GL.glCallList(list_id)

    def gl_draw_actor_shadows(self, actor: Actor, positions: np.ndarray, rotations: np.ndarray) -> None:
        if actor.clip is None or not self.volumes_var.get():
            return
        GL.glPushMatrix()
        GL.glMultMatrixf(self.planar_shadow_matrix(0.012).T)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glDisable(GL.GL_BLEND)
        GL.glDepthMask(GL.GL_FALSE)
        try:
            self.gl_draw_volumes(actor, positions, rotations, blend_hex(FLOOR, "#000000", 0.42), 1.0, rounded_caps=False)
        finally:
            GL.glDepthMask(GL.GL_TRUE)
            GL.glEnable(GL.GL_BLEND)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_LIGHTING)
            GL.glPopMatrix()

    def planar_shadow_matrix(self, floor_y: float) -> np.ndarray:
        lx, ly, lz = (float(LIGHT_DIR[0]), float(LIGHT_DIR[1]), float(LIGHT_DIR[2]))
        ly = ly if abs(ly) > 1e-6 else 1e-6
        return np.asarray(
            [
                [1.0, -lx / ly, 0.0, (lx / ly) * floor_y],
                [0.0, 0.0, 0.0, floor_y],
                [0.0, -lz / ly, 1.0, (lz / ly) * floor_y],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def draw_ground_plane(self) -> None:
        return

    def draw_floor_plane(self) -> None:
        size = max(16.0, self.extent * 4.0)
        corners = [
            np.asarray([-size, 0.0, -size], dtype=np.float32),
            np.asarray([size, 0.0, -size], dtype=np.float32),
            np.asarray([size, 0.0, size], dtype=np.float32),
            np.asarray([-size, 0.0, size], dtype=np.float32),
        ]
        projected = [self.project(c) for c in corners]
        coords = [value for p in projected for value in (p[0], p[1])]
        self.scene_polygon(coords, FLOOR, None, width=1)

    def draw_actor_shadow(self, actor: Actor, positions: np.ndarray) -> None:
        if actor.clip is None or not self.volumes_var.get():
            return
        name_to_index = {name: i for i, name in enumerate(actor.bone_names)}
        for a_name, b_name, radius_cm in self.capsule_specs():
            a = name_to_index.get(a_name)
            b = name_to_index.get(b_name)
            if a is not None and b is not None:
                self.draw_shadow_capsule(positions[a], positions[b], radius_cm * 0.01)
        for ankle_name, toe_name in (("foot_l", "ball_l"), ("foot_r", "ball_r")):
            ankle = name_to_index.get(ankle_name)
            toe = name_to_index.get(toe_name)
            if ankle is not None and toe is not None:
                self.draw_shadow_capsule(positions[ankle], positions[toe], 0.075)

    def shadow_point(self, p: np.ndarray) -> np.ndarray:
        height = max(0.0, float(p[1]))
        travel = height / max(1e-6, float(LIGHT_DIR[1]))
        return np.asarray(
            [
                p[0] - LIGHT_DIR[0] * travel,
                0.018,
                p[2] - LIGHT_DIR[2] * travel,
            ],
            dtype=np.float32,
        )

    def draw_shadow_capsule(self, a: np.ndarray, b: np.ndarray, radius_world: float) -> None:
        sa_world = self.shadow_point(a)
        sb_world = self.shadow_point(b)
        axis = sb_world - sa_world
        axis[1] = 0.0
        norm = float(np.linalg.norm(axis))
        if norm < 1e-6:
            return
        axis = axis / norm
        perp = np.asarray([-axis[2], 0.0, axis[0]], dtype=np.float32)
        radius = radius_world * 1.65
        avg_height = max(0.0, float((a[1] + b[1]) * 0.5))
        softness = 1.0 / (1.0 + avg_height * 0.65)
        color = self.rgba(SHADOW, int(175 * softness + 45))
        quad = [
            sa_world + perp * radius,
            sb_world + perp * radius,
            sb_world - perp * radius,
            sa_world - perp * radius,
        ]
        coords = [value for p in [self.project(q) for q in quad] for value in (p[0], p[1])]
        self.scene_polygon(coords, color, None, width=1)

    def draw_actor_placeholder(self, actor: Actor) -> None:
        self.scene_text((20, 24 + actor.actor_id * 18), f"{actor.name}: {actor.status}", MUTED)

    def draw_actor(self, actor: Actor, positions: np.ndarray, rotations: np.ndarray) -> None:
        if actor.clip is None:
            return
        points = [self.project(positions[j]) for j in range(actor.clip.J)]
        if self.volumes_var.get():
            self.draw_volumes(actor, positions, rotations)
        else:
            self.draw_skeleton_chains(actor, points)
        if self.labels_var.get():
            for j, point in enumerate(points):
                self.scene_text((point[0] + 6, point[1] - 12), actor.bone_names[j], "#cbd3df")

    def draw_skeleton_chains(self, actor: Actor, points: list[tuple[float, float, float]]) -> None:
        name_to_index = {name: i for i, name in enumerate(actor.bone_names)}
        chains = [
            ("pelvis", "spine_01", "spine_02", "spine_03", "spine_04", "spine_05", "neck_01", "neck_02", "head"),
            ("spine_05", "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l"),
            ("spine_05", "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r"),
            ("pelvis", "thigh_l", "calf_l", "foot_l", "ball_l"),
            ("pelvis", "thigh_r", "calf_r", "foot_r", "ball_r"),
        ]
        width = 4 if actor.kind == "npz" else 3
        for chain in chains:
            coords: list[float] = []
            for name in chain:
                index = name_to_index.get(name)
                if index is None:
                    continue
                coords.extend([points[index][0], points[index][1]])
            self.scene_polyline(coords, actor.color, width=width)

    def capsule_specs(self) -> list[tuple[str, str, float]]:
        return [
            ("spine_05", "neck_01", 4.2),
            ("neck_01", "neck_02", 2.0),
            ("clavicle_l", "upperarm_l", 3.2),
            ("upperarm_l", "lowerarm_l", 4.3),
            ("lowerarm_l", "hand_l", 3.7),
            ("clavicle_r", "upperarm_r", 3.2),
            ("upperarm_r", "lowerarm_r", 4.3),
            ("lowerarm_r", "hand_r", 3.7),
            ("thigh_l", "calf_l", 6.4),
            ("calf_l", "foot_l", 5.2),
            ("thigh_r", "calf_r", 6.4),
            ("calf_r", "foot_r", 5.2),
        ]

    def rounded_capsule_specs(self) -> set[tuple[str, str]]:
        return set()

    def draw_volumes(self, actor: Actor, positions: np.ndarray, rotations: np.ndarray) -> None:
        names = actor.bone_names
        name_to_index = {name: i for i, name in enumerate(names)}
        capsule_indices = []
        for a_name, b_name, radius_cm in self.capsule_specs():
            a = name_to_index.get(a_name)
            b = name_to_index.get(b_name)
            if a is not None and b is not None:
                capsule_indices.append((a, b, radius_cm * 0.01))
        for a, b, radius in capsule_indices:
            self.draw_capsule(positions[a], positions[b], radius, actor.color, outline=False)
        self.draw_head_sphere(positions, name_to_index, actor.color)
        for ankle_name, toe_name in (("foot_l", "ball_l"), ("foot_r", "ball_r")):
            ankle = name_to_index.get(ankle_name)
            toe = name_to_index.get(toe_name)
            if ankle is not None and toe is not None:
                self.draw_foot_boxes(positions, rotations, ankle, toe, actor.color)
        for hand_name, mid_name, parent_name in (
            ("hand_l", "middle_03_l", "lowerarm_l"),
            ("hand_r", "middle_03_r", "lowerarm_r"),
        ):
            hand = name_to_index.get(hand_name)
            if hand is not None:
                self.draw_hand_box(
                    positions,
                    rotations,
                    hand,
                    name_to_index.get(mid_name),
                    name_to_index.get(parent_name),
                    actor.color,
                )

    def draw_capsule(self, a: np.ndarray, b: np.ndarray, radius_world: float, color: str, outline: bool) -> None:
        if outline:
            return
        self.draw_capsule_mesh(a, b, radius_world, color)

    def draw_head_sphere(self, positions: np.ndarray, name_to_index: dict[str, int], color: str) -> None:
        sphere = self.gl_head_sphere(positions, name_to_index)
        if sphere is None:
            return
        center, radius = sphere
        point = self.project(center)
        self.scene_ellipse((point[0], point[1]), self.world_radius_px(radius), self.lit_color(color, np.asarray([0.0, 1.0, 0.0], dtype=np.float32)))

    def segment_basis(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        axis = sub3(b, a)
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-6:
            return None
        axis = axis / norm
        ref = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(axis, ref))) > 0.88:
            ref = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        side = np.cross(axis, ref)
        side_norm = float(np.linalg.norm(side))
        if side_norm <= 1e-6:
            return None
        side = side / side_norm
        up = np.cross(side, axis)
        up_norm = float(np.linalg.norm(up))
        if up_norm <= 1e-6:
            return None
        up = up / up_norm
        return axis, side, up

    def lit_color(self, color: str, normal: np.ndarray, ambient: float = 0.38) -> str:
        light = max(0.0, float(np.dot(normal, LIGHT_DIR)))
        intensity = min(1.0, ambient + light * 0.62)
        if intensity >= 0.72:
            return blend_hex(color, "#ffffff", (intensity - 0.72) * 0.70)
        return blend_hex(color, "#000000", (0.72 - intensity) * 0.58)

    def draw_capsule_mesh(self, a: np.ndarray, b: np.ndarray, radius: float, color: str) -> None:
        basis = self.segment_basis(a, b)
        if basis is None:
            return
        _axis, side, up = basis
        segments = 10
        faces: list[tuple[float, list[float], str]] = []
        ring = []
        for i in range(segments):
            theta = math.tau * i / segments
            normal = math.cos(theta) * side + math.sin(theta) * up
            ring.append(normal.astype(np.float32))
        for i in range(segments):
            n0 = ring[i]
            n1 = ring[(i + 1) % segments]
            verts = [a + n0 * radius, b + n0 * radius, b + n1 * radius, a + n1 * radius]
            projected = [self.project(v) for v in verts]
            coords = [value for p in projected for value in (p[0], p[1])]
            z_avg = sum(p[2] for p in projected) / len(projected)
            normal = n0 + n1
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm > 1e-6:
                normal = normal / normal_norm
            faces.append((z_avg, coords, self.lit_color(color, normal)))
        for center, sign in ((a, -1.0), (b, 1.0)):
            center_projected = self.project(center)
            for i in range(segments):
                n0 = ring[i]
                n1 = ring[(i + 1) % segments]
                verts = [center, center + n1 * radius, center + n0 * radius] if sign < 0 else [center, center + n0 * radius, center + n1 * radius]
                projected = [self.project(v) for v in verts]
                coords = [value for p in projected for value in (p[0], p[1])]
                z_avg = (center_projected[2] + projected[1][2] + projected[2][2]) / 3.0
                normal = (n0 + n1) * 0.35 + (_axis * sign)
                normal_norm = float(np.linalg.norm(normal))
                if normal_norm > 1e-6:
                    normal = normal / normal_norm
                faces.append((z_avg, coords, self.lit_color(color, normal, ambient=0.45)))
        for _z, coords, fill in sorted(faces, key=lambda item: item[0]):
            self.scene_polygon(coords, fill, None, width=1)

    def draw_foot_boxes(
        self,
        positions: np.ndarray,
        rotations: np.ndarray,
        ankle: int,
        toe: int,
        color: str,
    ) -> None:
        for center, axis_x, axis_y, axis_z, dims in self.foot_box_specs(positions, rotations, ankle, toe):
            self.draw_oriented_box(center, axis_x, axis_y, axis_z, dims, color)

    def draw_hand_box(
        self,
        positions: np.ndarray,
        rotations: np.ndarray,
        hand: int,
        mid: int | None,
        parent: int | None,
        color: str,
    ) -> None:
        center, axis_x, axis_y, axis_z, dims = self.hand_box_spec(positions, rotations, hand, mid, parent)
        self.draw_oriented_box(center, axis_x, axis_y, axis_z, dims, color)

    def draw_oriented_box(
        self,
        center: np.ndarray,
        axis_x: np.ndarray,
        axis_y: np.ndarray,
        axis_z: np.ndarray,
        dims: tuple[float, float, float],
        color: str,
    ) -> None:
        hx, hy, hz = dims[0] * 0.5, dims[1] * 0.5, dims[2] * 0.5
        corners_local = [
            (-hx, -hy, -hz),
            (hx, -hy, -hz),
            (hx, hy, -hz),
            (-hx, hy, -hz),
            (-hx, -hy, hz),
            (hx, -hy, hz),
            (hx, hy, hz),
            (-hx, hy, hz),
        ]
        corners = []
        corners_world = []
        for x, y, z in corners_local:
            point = add3(add3(add3(center, mul3(axis_x, x)), mul3(axis_y, y)), mul3(axis_z, z))
            corners_world.append(point)
            corners.append(self.project(point))
        faces = [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ]
        sorted_faces = sorted(faces, key=lambda face: sum(corners[i][2] for i in face) / len(face))
        for face in sorted_faces:
            coords: list[float] = []
            for i in face:
                coords.extend([corners[i][0], corners[i][1]])
            a = corners_world[face[0]]
            b = corners_world[face[1]]
            c = corners_world[face[2]]
            normal = np.cross(sub3(b, a), sub3(c, b))
            norm = float(np.linalg.norm(normal))
            if norm > 1e-6:
                normal = normal / norm
            light = max(0.0, float(np.dot(normal, LIGHT_DIR)))
            fill = blend_hex(color, "#ffffff", 0.08 + light * 0.24)
            if light < 0.12:
                fill = blend_hex(fill, "#000000", 0.18)
            self.scene_polygon(coords, fill, None, width=1.0)

    def draw_gizmo(self) -> None:
        self.gizmo_hits.clear()
        actor = self.selected_model_actor()
        if actor is None or not self.gizmo_visible or self.playing:
            return
        origin = actor.offset.copy()
        o = self.project(origin)
        length = max(0.3, self.extent * 0.16)
        axes = {
            "x": (np.asarray([length, 0.0, 0.0], dtype=np.float32), "#f05d5e"),
            "y": (np.asarray([0.0, length, 0.0], dtype=np.float32), "#55d66b"),
            "z": (np.asarray([0.0, 0.0, length], dtype=np.float32), "#62a2ff"),
        }
        self.scene_ellipse((o[0], o[1]), 7, self.rgba("#ffffff", 210), self.rgba("#111111", 230), width=1)
        self.gizmo_hits["plane"] = (o[0] - 12, o[1] - 12, o[0] + 12, o[1] + 12)
        for name, (delta, color) in axes.items():
            tip = self.project(origin + delta)
            self.scene_line((o[0], o[1]), (tip[0], tip[1]), color, width=3.0, round_caps=True)
            self.scene_ellipse((tip[0], tip[1]), 5.5, color, self.rgba("#111111", 230), width=1)
            x0, y0 = min(o[0], tip[0]) - 8, min(o[1], tip[1]) - 8
            x1, y1 = max(o[0], tip[0]) + 8, max(o[1], tip[1]) + 8
            self.gizmo_hits[name] = (x0, y0, x1, y1)


def main() -> None:
    app = ModelViewerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
