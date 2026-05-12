from __future__ import annotations

import argparse
import json
import math
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

import numpy as np
from OpenGL import GL, GLU
from pyopengltk import OpenGLFrame


BG = "#101216"
PANEL = "#1a1d22"
PANEL_2 = "#23272f"
TEXT = "#edf1f7"
MUTED = "#9da6b5"
LINE = "#313846"
FLOOR = "#303846"
GT_BLUE = "#72a7ff"
PRED_ORANGE = "#ff9b6a"
TARGET_FPS = 60.0
TARGET_INTERVAL_MS = max(1, int(round(1000.0 / TARGET_FPS)))
FALLBACK_PLAYBACK_FPS = 30.0
LIGHT_DIR = np.asarray([-0.45, 0.85, -0.30], dtype=np.float32)
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)


def hex_to_rgb(color: str) -> tuple[float, float, float]:
    return (
        int(color[1:3], 16) / 255.0,
        int(color[3:5], 16) / 255.0,
        int(color[5:7], 16) / 255.0,
    )


def gl_color(color: str, alpha: float = 1.0) -> None:
    r, g, b = hex_to_rgb(color)
    GL.glColor4f(r, g, b, alpha)


def unit_vec(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-6:
        return fallback.astype(np.float32)
    return (v / n).astype(np.float32)


def dot3(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def add3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)


def sub3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)


def mul3(a: np.ndarray, s: float) -> np.ndarray:
    return np.asarray(a, dtype=np.float32) * float(s)


def basis_axes_from_direction(
    basis: np.ndarray,
    direction: np.ndarray,
    fallback_forward_axis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis = np.asarray(basis, dtype=np.float32)
    basis = basis / np.maximum(np.linalg.norm(basis, axis=1, keepdims=True), 1e-8)
    direction = np.asarray(direction, dtype=np.float32)
    if float(np.linalg.norm(direction)) > 1e-8:
        direction_n = direction / float(np.linalg.norm(direction))
        dots = basis @ direction_n
        forward_index = int(np.argmax(np.abs(dots)))
        forward = basis[forward_index].copy()
        if float(dots[forward_index]) < 0.0:
            forward *= -1.0
    else:
        forward_index = int(fallback_forward_axis)
        forward = basis[forward_index].copy()
    remaining = [i for i in range(3) if i != forward_index]
    up_index = max(remaining, key=lambda i: abs(float(basis[i, 1])))
    up = basis[up_index].copy()
    if float(up[1]) < 0.0:
        up *= -1.0
    side = unit_vec(np.cross(up, forward), basis[[i for i in remaining if i != up_index][0]])
    return unit_vec(forward, basis[forward_index]), side, unit_vec(up, basis[up_index])


def hand_axes_from_source(
    basis: np.ndarray,
    guide: np.ndarray,
    source_up_axis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis = np.asarray(basis, dtype=np.float32)
    basis = basis / np.maximum(np.linalg.norm(basis, axis=1, keepdims=True), 1e-8)
    forward = basis[0].copy()
    guide = np.asarray(guide, dtype=np.float32)
    if float(np.linalg.norm(guide)) > 1e-8 and dot3(forward, guide) < 0.0:
        forward *= -1.0
    up_axis = 1 if int(source_up_axis) == 3 else 2
    up = basis[up_axis].copy()
    fallback_axis = 2 if up_axis == 1 else 1
    side = unit_vec(np.cross(up, forward), basis[fallback_axis])
    return unit_vec(forward, basis[0]), side, unit_vec(up, basis[up_axis])


def replace_with_retry(tmp: Path, target: Path, attempts: int = 12, delay_seconds: float = 0.01) -> bool:
    for attempt in range(attempts):
        try:
            tmp.replace(target)
            return True
        except PermissionError:
            if attempt == attempts - 1:
                break
            time.sleep(delay_seconds)
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    return False


def write_json_atomic(path: Path, data: dict) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return replace_with_retry(tmp, path)


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


@dataclass
class LiveSnapshot:
    epoch: int
    rollout_k: int
    phase: str
    train_total: float
    fps: float
    source_up_axis: int
    body_names: list[str]
    parents: list[int]
    pred_pos: np.ndarray
    pred_rot: np.ndarray
    gt_pos: np.ndarray
    gt_rot: np.ndarray

    @property
    def agent_count(self) -> int:
        return int(self.pred_pos.shape[0])

    @property
    def frame_count(self) -> int:
        return int(self.pred_pos.shape[1])


class TrainingGLFrame(OpenGLFrame):
    def __init__(self, *args, app: "LiveTrainingViewer", **kwargs):
        self.app = app
        super().__init__(*args, **kwargs)

    def initgl(self) -> None:
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
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
        self.app.gl_ready = True

    def redraw(self) -> None:
        self.app.render_gl()


class LiveTrainingViewer(tk.Tk):
    def __init__(self, run_dir: Path, start_visualizing: bool = False, initial_loss_height: int = 120) -> None:
        super().__init__()
        self.run_dir = run_dir.resolve()
        self.live_dir = self.run_dir / "live_training"
        self.control_path = self.live_dir / "control.json"
        self.snapshot_path = self.live_dir / "snapshot.npz"
        self.status_path = self.live_dir / "status.json"
        self.loss_history_path = self.live_dir / "loss_history.csv"
        self.visualizing_var = tk.BooleanVar(value=bool(start_visualizing))
        self.show_gt_var = tk.BooleanVar(value=True)
        self.stop_requested_var = tk.BooleanVar(value=False)
        self.snapshot: LiveSnapshot | None = None
        self.snapshot_mtime = 0.0
        self.loss_history_mtime = 0.0
        self.last_loss_poll = 0.0
        self.loss_points: list[tuple[int, float, int, float]] = []
        self.frame = 0.0
        self.last_tick = time.perf_counter()
        self.quadric = None
        self.gl_ready = False
        self.slot_spacing = 2.25
        self.initial_loss_height = max(70, int(initial_loss_height))
        self.title("Stepper Training Viewer")
        self.configure(bg=BG)
        self.minsize(980, 640)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.build_ui()
        self.write_control()
        self.after(TARGET_INTERVAL_MS, self.tick)

    def build_ui(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Stepper.TFrame", background=PANEL)
        style.configure("Stepper.TButton", background=PANEL_2, foreground=TEXT, bordercolor=LINE, focusthickness=0)
        style.configure("Stepper.TCheckbutton", background=PANEL, foreground=TEXT)
        top = ttk.Frame(self, style="Stepper.TFrame")
        top.pack(side=tk.TOP, fill=tk.X)
        self.mode_button = ttk.Button(top, text="Visualise", command=self.toggle_visualizing, style="Stepper.TButton")
        self.mode_button.pack(side=tk.LEFT, padx=8, pady=7)
        self.stop_button = ttk.Button(top, text="Stop experiment", command=self.request_stop, style="Stepper.TButton")
        self.stop_button.pack(side=tk.LEFT, padx=8, pady=7)
        self.gt_check = ttk.Checkbutton(
            top,
            text="Ground truth",
            variable=self.show_gt_var,
            command=self.write_control,
            style="Stepper.TCheckbutton",
        )
        self.gt_check.pack(side=tk.LEFT, padx=8)
        self.status_var = tk.StringVar(value="headless - no training snapshots requested")
        status = ttk.Label(top, textvariable=self.status_var, background=PANEL, foreground=MUTED)
        status.pack(side=tk.LEFT, padx=12)
        self.main_pane = tk.PanedWindow(
            self,
            orient=tk.VERTICAL,
            bg=LINE,
            bd=0,
            sashwidth=7,
            sashrelief=tk.RAISED,
            showhandle=True,
            opaqueresize=True,
        )
        self.main_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = TrainingGLFrame(self.main_pane, app=self, width=980, height=520)
        self.loss_canvas = tk.Canvas(
            self.main_pane,
            height=self.initial_loss_height,
            bg="#0b0e12",
            highlightthickness=1,
            highlightbackground=LINE,
        )
        self.loss_canvas.bind("<Configure>", lambda _event: self.draw_loss_graph())
        self.main_pane.add(self.canvas, minsize=260)
        self.main_pane.add(self.loss_canvas, minsize=70)
        self.after(150, self.set_initial_loss_height)
        self.update_mode_button()

    def set_initial_loss_height(self) -> None:
        if not hasattr(self, "main_pane"):
            return
        pane_height = self.main_pane.winfo_height()
        if pane_height <= 200:
            self.after(150, self.set_initial_loss_height)
            return
        self.main_pane.sash_place(0, 0, max(260, pane_height - self.initial_loss_height))

    def init_gl_resources(self) -> None:
        self.quadric = GLU.gluNewQuadric()
        GLU.gluQuadricNormals(self.quadric, GLU.GLU_SMOOTH)

    def on_close(self) -> None:
        self.visualizing_var.set(False)
        self.write_control()
        self.destroy()

    def update_mode_button(self) -> None:
        self.mode_button.configure(text="Headless" if self.visualizing_var.get() else "Visualise")

    def toggle_visualizing(self) -> None:
        self.visualizing_var.set(not self.visualizing_var.get())
        self.update_mode_button()
        self.write_control()
        if self.visualizing_var.get():
            self.frame = 0.0
            self.load_snapshot(force=True)
            if self.snapshot is None:
                self.status_var.set("visualising - waiting for first snapshot")
            self.redraw_canvas()

    def request_stop(self) -> None:
        self.stop_requested_var.set(True)
        self.stop_button.configure(text="Stopping...", state=tk.DISABLED)
        self.status_var.set("stop requested - trainer will exit cleanly")
        self.write_control()

    def write_control(self) -> None:
        write_json_atomic(
            self.control_path,
            {
                "visualize": bool(self.visualizing_var.get()),
                "show_ground_truth": bool(self.show_gt_var.get()),
                "stop": bool(self.stop_requested_var.get()),
                "updated_at": time.time(),
            },
        )

    def load_snapshot(self, force: bool = False) -> bool:
        try:
            mtime = self.snapshot_path.stat().st_mtime
        except OSError:
            return False
        if not force and mtime <= self.snapshot_mtime:
            return False
        try:
            with np.load(self.snapshot_path, allow_pickle=False) as arrays:
                body_names = [str(x) for x in arrays["body_names"].tolist()]
                parents = [int(x) for x in arrays["parents"].tolist()]
                self.snapshot = LiveSnapshot(
                    epoch=int(arrays["epoch"][0]),
                    rollout_k=int(arrays["rollout_k"][0]),
                    phase=str(arrays["phase"][0]),
                    train_total=float(arrays["train_total"][0]),
                    fps=float(arrays["fps"][0]),
                    source_up_axis=int(arrays["source_up_axis"][0]) if "source_up_axis" in arrays.files else 2,
                    body_names=body_names,
                    parents=parents,
                    pred_pos=arrays["pred_pos"].astype(np.float32),
                    pred_rot=arrays["pred_rot"].astype(np.float32),
                    gt_pos=arrays["gt_pos"].astype(np.float32),
                    gt_rot=arrays["gt_rot"].astype(np.float32),
                )
        except (OSError, ValueError, KeyError):
            return False
        self.snapshot_mtime = mtime
        self.frame = 0.0
        return True

    def load_loss_history(self) -> bool:
        try:
            mtime = self.loss_history_path.stat().st_mtime
        except OSError:
            return False
        if mtime <= self.loss_history_mtime:
            return False
        points: list[tuple[int, float, int, float]] = []
        try:
            lines = self.loss_history_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return False
        for line in lines[1:]:
            parts = line.strip().split(",")
            if len(parts) != 4:
                continue
            try:
                points.append((int(parts[0]), float(parts[1]), int(parts[2]), float(parts[3])))
            except ValueError:
                continue
        self.loss_points = points
        self.loss_history_mtime = mtime
        self.draw_loss_graph()
        return True

    def draw_loss_graph(self) -> None:
        canvas = self.loss_canvas
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        canvas.delete("all")
        canvas.create_rectangle(0, 0, width, height, fill="#0b0e12", outline="")
        margin_l = 52
        margin_r = 12
        margin_t = 16
        margin_b = 22
        x0 = margin_l
        y0 = margin_t
        x1 = max(x0 + 1, width - margin_r)
        y1 = max(y0 + 1, height - margin_b)
        canvas.create_line(x0, y1, x1, y1, fill="#27303b")
        canvas.create_line(x0, y0, x0, y1, fill="#27303b")
        if not self.loss_points:
            canvas.create_text(12, height * 0.5, anchor="w", text="total loss graph waiting", fill=MUTED)
            return
        points = self.loss_points[-600:]
        xs = [p[1] for p in points]
        ys = [p[3] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        if max_x <= min_x:
            max_x = min_x + 1.0
        if max_y <= min_y:
            pad = max(1e-6, abs(max_y) * 0.05)
            min_y -= pad
            max_y += pad
        else:
            pad = (max_y - min_y) * 0.08
            min_y -= pad
            max_y += pad
        coords: list[float] = []
        for _epoch, elapsed, _k, loss in points:
            x = x0 + (elapsed - min_x) / (max_x - min_x) * (x1 - x0)
            y = y1 - (loss - min_y) / (max_y - min_y) * (y1 - y0)
            coords.extend((x, y))
        if len(coords) >= 4:
            canvas.create_line(*coords, fill=PRED_ORANGE, width=2, smooth=True)
        last_epoch, last_elapsed, last_k, last_loss = self.loss_points[-1]
        canvas.create_text(
            10,
            8,
            anchor="nw",
            text=f"total loss {last_loss:.5f}   epoch {last_epoch}   K{last_k}   {last_elapsed:.1f}s",
            fill=TEXT,
        )
        canvas.create_text(x0, y1 + 6, anchor="nw", text=f"{min_x:.1f}s", fill=MUTED)
        canvas.create_text(x1, y1 + 6, anchor="ne", text=f"{max_x:.1f}s", fill=MUTED)
        canvas.create_text(8, y0, anchor="nw", text=f"{max_y:.3g}", fill=MUTED)
        canvas.create_text(8, y1 - 12, anchor="nw", text=f"{min_y:.3g}", fill=MUTED)

    def tick(self) -> None:
        now = time.perf_counter()
        dt = max(0.0, min(0.1, now - self.last_tick))
        self.last_tick = now
        if now - self.last_loss_poll >= 0.25:
            self.last_loss_poll = now
            self.load_loss_history()
        if self.visualizing_var.get():
            self.load_snapshot()
            snapshot = self.snapshot
            if snapshot is not None and snapshot.frame_count > 0:
                playback_fps = snapshot.fps if snapshot.fps > 1e-3 else FALLBACK_PLAYBACK_FPS
                self.frame = min(self.frame + dt * playback_fps, max(0.0, float(snapshot.frame_count - 1)))
                self.status_var.set(
                    f"visualising - epoch {snapshot.epoch} K{snapshot.rollout_k} "
                    f"{snapshot.phase} loss {snapshot.train_total:.5f}"
                )
            else:
                self.status_var.set("visualising - waiting for first snapshot")
            self.redraw_canvas()
        else:
            status = read_json(self.status_path)
            suffix = ""
            if status.get("epoch") is not None:
                suffix = f" - training epoch {status.get('epoch')} K{status.get('rollout_k')}"
            elif self.loss_points:
                epoch, _elapsed, rollout_k, loss = self.loss_points[-1]
                suffix = f" - epoch {epoch} K{rollout_k} loss {loss:.5f}"
            self.status_var.set(f"headless - no training snapshots requested{suffix}")
        self.after(TARGET_INTERVAL_MS, self.tick)

    def redraw_canvas(self) -> None:
        if not self.gl_ready or self.canvas.winfo_width() <= 1 or self.canvas.winfo_height() <= 1:
            return
        self.canvas.tkMakeCurrent()
        self.canvas.redraw()
        self.canvas.tkSwapBuffers()

    def camera_view(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        target = np.asarray([0.0, 0.78, 0.0], dtype=np.float32)
        distance = 6.4
        yaw = -0.62
        pitch = -0.22
        cam = target + np.asarray(
            [
                math.sin(yaw) * math.cos(pitch) * distance,
                math.sin(-pitch) * distance + 1.0,
                math.cos(yaw) * math.cos(pitch) * distance,
            ],
            dtype=np.float32,
        )
        return cam, target, np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

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
        cam, target, up = self.camera_view()
        GLU.gluLookAt(
            float(cam[0]), float(cam[1]), float(cam[2]),
            float(target[0]), float(target[1]), float(target[2]),
            float(up[0]), float(up[1]), float(up[2]),
        )
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, (-float(LIGHT_DIR[0]), -float(LIGHT_DIR[1]), -float(LIGHT_DIR[2]), 0.0))
        self.draw_floor()
        snapshot = self.snapshot
        if not self.visualizing_var.get() or snapshot is None:
            self.draw_empty_state(width, height, snapshot is not None)
            GL.glFlush()
            return
        frame = self.display_frame_index(snapshot)
        for agent_i in range(snapshot.agent_count):
            offset = self.agent_slot_offset(agent_i, snapshot.agent_count)
            anchor = snapshot.pred_pos[agent_i, 0, self.pelvis_index(snapshot)].copy()
            anchor[1] = 0.0
            if self.show_gt_var.get():
                gt_pos = snapshot.gt_pos[agent_i, frame] - anchor + offset
                gt_rot = snapshot.gt_rot[agent_i, frame]
                self.draw_actor(snapshot, gt_pos, gt_rot, GT_BLUE, 0.38)
            pred_pos = snapshot.pred_pos[agent_i, frame] - anchor + offset
            pred_rot = snapshot.pred_rot[agent_i, frame]
            self.draw_actor(snapshot, pred_pos, pred_rot, PRED_ORANGE, 0.92)
        GL.glFlush()

    def display_frame_index(self, snapshot: LiveSnapshot) -> int:
        frame_count = max(1, snapshot.frame_count)
        return int(self.frame) % frame_count

    def draw_empty_state(self, width: int, height: int, has_snapshot: bool) -> None:
        GL.glDisable(GL.GL_LIGHTING)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPushMatrix()
        GL.glLoadIdentity()
        GLU.gluOrtho2D(0, width, height, 0)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        GL.glLoadIdentity()
        try:
            GL.glColor4f(0.85, 0.90, 0.98, 0.70)
            GL.glLineWidth(2.0)
            cx = width * 0.5
            cy = height * 0.42
            size = 42.0
            GL.glBegin(GL.GL_LINE_LOOP)
            GL.glVertex2f(cx - size, cy - size)
            GL.glVertex2f(cx + size, cy - size)
            GL.glVertex2f(cx + size, cy + size)
            GL.glVertex2f(cx - size, cy + size)
            GL.glEnd()
            GL.glBegin(GL.GL_LINES)
            GL.glVertex2f(cx - size * 0.62, cy)
            GL.glVertex2f(cx + size * 0.62, cy)
            GL.glVertex2f(cx, cy - size * 0.62)
            GL.glVertex2f(cx, cy + size * 0.62)
            GL.glEnd()
            if has_snapshot:
                GL.glColor4f(1.0, 0.61, 0.42, 0.75)
            else:
                GL.glColor4f(0.45, 0.65, 1.0, 0.55)
            GL.glPointSize(8.0)
            GL.glBegin(GL.GL_POINTS)
            for dx, dy in ((-18, -18), (18, -18), (-18, 18), (18, 18)):
                GL.glVertex2f(cx + dx, cy + dy)
            GL.glEnd()
        finally:
            GL.glPopMatrix()
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glPopMatrix()
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glEnable(GL.GL_LIGHTING)

    def pelvis_index(self, snapshot: LiveSnapshot) -> int:
        try:
            return snapshot.body_names.index("pelvis")
        except ValueError:
            return 0

    def agent_slot_offset(self, agent_i: int, agent_count: int) -> np.ndarray:
        if agent_count <= 1:
            return np.zeros(3, dtype=np.float32)
        slots = (
            (-0.5, 0.5),
            (0.5, 0.5),
            (-0.5, -0.5),
            (0.5, -0.5),
        )
        sx, sz = slots[agent_i % len(slots)]
        return np.asarray([sx * self.slot_spacing, 0.0, sz * self.slot_spacing], dtype=np.float32)

    def draw_floor(self) -> None:
        GL.glDisable(GL.GL_LIGHTING)
        gl_color(FLOOR, 0.28)
        size = 6.0
        GL.glBegin(GL.GL_QUADS)
        GL.glVertex3f(-size, 0.0, -size)
        GL.glVertex3f(size, 0.0, -size)
        GL.glVertex3f(size, 0.0, size)
        GL.glVertex3f(-size, 0.0, size)
        GL.glEnd()
        GL.glLineWidth(1.0)
        gl_color("#222832", 0.8)
        GL.glBegin(GL.GL_LINES)
        step = 0.25
        for i in range(int(-size / step), int(size / step) + 1):
            x = i * step
            GL.glVertex3f(x, 0.002, -size)
            GL.glVertex3f(x, 0.002, size)
            GL.glVertex3f(-size, 0.002, x)
            GL.glVertex3f(size, 0.002, x)
        GL.glEnd()
        GL.glEnable(GL.GL_LIGHTING)

    def draw_actor(self, snapshot: LiveSnapshot, positions: np.ndarray, rotations: np.ndarray, color: str, alpha: float) -> None:
        self.draw_volumes(snapshot, positions, rotations, color, alpha)
        self.draw_skeleton(snapshot, positions, color, min(1.0, alpha + 0.08))

    def draw_skeleton(self, snapshot: LiveSnapshot, positions: np.ndarray, color: str, alpha: float) -> None:
        GL.glDisable(GL.GL_LIGHTING)
        GL.glLineWidth(2.0)
        gl_color(color, alpha)
        chains = [
            ("pelvis", "spine_01", "spine_02", "spine_03", "spine_04", "spine_05", "neck_01", "neck_02", "head"),
            ("spine_05", "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l"),
            ("spine_05", "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r"),
            ("pelvis", "thigh_l", "calf_l", "foot_l", "ball_l"),
            ("pelvis", "thigh_r", "calf_r", "foot_r", "ball_r"),
        ]
        name_to_index = {name: i for i, name in enumerate(snapshot.body_names)}
        for chain in chains:
            GL.glBegin(GL.GL_LINE_STRIP)
            for name in chain:
                idx = name_to_index.get(name)
                if idx is None:
                    continue
                p = positions[idx]
                GL.glVertex3f(float(p[0]), float(p[1]), float(p[2]))
            GL.glEnd()
        GL.glEnable(GL.GL_LIGHTING)

    def draw_volumes(self, snapshot: LiveSnapshot, positions: np.ndarray, rotations: np.ndarray, color: str, alpha: float) -> None:
        name_to_index = {name: i for i, name in enumerate(snapshot.body_names)}
        for child, parent in enumerate(snapshot.parents):
            if parent < 0 or parent >= len(snapshot.parents):
                continue
            child_name = snapshot.body_names[child]
            parent_name = snapshot.body_names[parent]
            if child_name.startswith("ball_") or child_name.startswith("finger_"):
                continue
            radius = self.segment_radius(parent_name, child_name)
            self.draw_capsule(positions[parent], positions[child], radius, color, alpha * 0.55)
        for name, radius in (("pelvis", 0.058), ("head", 0.105), ("foot_l", 0.04), ("foot_r", 0.04)):
            idx = name_to_index.get(name)
            if idx is not None:
                self.draw_sphere(positions[idx], radius, color, alpha * 0.72)
        for ankle_name, toe_name in (("foot_l", "ball_l"), ("foot_r", "ball_r")):
            ankle = name_to_index.get(ankle_name)
            toe = name_to_index.get(toe_name)
            if ankle is not None and toe is not None:
                self.draw_foot_boxes(positions, rotations, ankle, toe, color, alpha * 0.62)
        for hand_name, parent_name in (("hand_l", "lowerarm_l"), ("hand_r", "lowerarm_r")):
            hand = name_to_index.get(hand_name)
            parent = name_to_index.get(parent_name)
            if hand is not None:
                self.draw_hand_box(positions, rotations, hand, parent, snapshot.source_up_axis, color, alpha * 0.62)

    def segment_radius(self, a: str, b: str) -> float:
        pair = f"{a}/{b}"
        if "thigh" in pair:
            return 0.058
        if "calf" in pair:
            return 0.050
        if "upperarm" in pair:
            return 0.041
        if "lowerarm" in pair:
            return 0.034
        if "spine" in pair or "neck" in pair:
            return 0.044
        if "clavicle" in pair:
            return 0.027
        return 0.026

    def draw_sphere(self, center: np.ndarray, radius: float, color: str, alpha: float) -> None:
        if self.quadric is None:
            return
        GL.glPushMatrix()
        GL.glTranslatef(float(center[0]), float(center[1]), float(center[2]))
        gl_color(color, alpha)
        GLU.gluSphere(self.quadric, float(radius), 16, 10)
        GL.glPopMatrix()

    def draw_capsule(self, a: np.ndarray, b: np.ndarray, radius: float, color: str, alpha: float) -> None:
        if self.quadric is None:
            return
        axis = sub3(b, a)
        length = float(np.linalg.norm(axis))
        if length <= 1e-6:
            return
        z = axis / length
        x = unit_vec(np.cross(np.asarray([0.0, 1.0, 0.0], dtype=np.float32), z), np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
        y = unit_vec(np.cross(z, x), np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 0] = x
        matrix[:3, 1] = y
        matrix[:3, 2] = z
        matrix[:3, 3] = a
        GL.glPushMatrix()
        GL.glMultMatrixf(matrix.T)
        gl_color(color, alpha)
        GLU.gluCylinder(self.quadric, float(radius), float(radius), length, 12, 1)
        GL.glPopMatrix()

    def draw_box(
        self,
        center: np.ndarray,
        axis_x: np.ndarray,
        axis_y: np.ndarray,
        axis_z: np.ndarray,
        dims: tuple[float, float, float],
        color: str,
        alpha: float,
    ) -> None:
        x = unit_vec(axis_x, np.asarray([1.0, 0.0, 0.0], dtype=np.float32)) * (dims[0] * 0.5)
        y = unit_vec(axis_y, np.asarray([0.0, 1.0, 0.0], dtype=np.float32)) * (dims[1] * 0.5)
        z = unit_vec(axis_z, np.asarray([0.0, 0.0, 1.0], dtype=np.float32)) * (dims[2] * 0.5)
        corners = [
            center - x - y - z, center + x - y - z, center + x + y - z, center - x + y - z,
            center - x - y + z, center + x - y + z, center + x + y + z, center - x + y + z,
        ]
        faces = ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0))
        gl_color(color, alpha)
        GL.glBegin(GL.GL_QUADS)
        for face in faces:
            for idx in face:
                p = corners[idx]
                GL.glVertex3f(float(p[0]), float(p[1]), float(p[2]))
        GL.glEnd()

    def draw_foot_boxes(self, positions: np.ndarray, rotations: np.ndarray, ankle: int, toe: int, color: str, alpha: float) -> None:
        foot = positions[ankle]
        toe_pos = positions[toe]
        toe_vector = sub3(toe_pos, foot)
        forward, side_axis, up = basis_axes_from_direction(rotations[ankle], toe_vector, 1)
        foot_dims = (0.17, 0.11, 0.064)
        toe_dims = (0.065, 0.11, 0.064)
        heel_back = add3(toe_pos, mul3(forward, -foot_dims[0]))
        center = add3(add3(heel_back, mul3(forward, foot_dims[0] * 0.5)), mul3(up, -0.006))
        self.draw_box(center, forward, side_axis, up, foot_dims, color, alpha)
        toe_forward, toe_side, toe_up = basis_axes_from_direction(rotations[toe], toe_vector, 0)
        toe_center = add3(add3(toe_pos, mul3(toe_forward, toe_dims[0] * 0.5)), mul3(toe_up, -0.006))
        self.draw_box(toe_center, toe_forward, toe_side, toe_up, toe_dims, color, alpha)

    def draw_hand_box(
        self,
        positions: np.ndarray,
        rotations: np.ndarray,
        hand: int,
        parent: int | None,
        source_up_axis: int,
        color: str,
        alpha: float,
    ) -> None:
        hand_pos = positions[hand]
        guide = rotations[hand, 0].copy()
        if parent is not None:
            forearm_vector = sub3(hand_pos, positions[parent])
            length = float(np.linalg.norm(forearm_vector))
            if length > 1e-6:
                guide = forearm_vector / length
        forward, side_axis, up = hand_axes_from_source(rotations[hand], guide, source_up_axis)
        dims = (0.09, 0.068, 0.032)
        center = add3(add3(hand_pos, mul3(forward, dims[0] * 0.5)), mul3(up, -0.0025))
        self.draw_box(center, forward, up, side_axis, dims, color, alpha)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight OpenGL viewer for live training snapshots.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--start-visualizing", action="store_true")
    parser.add_argument("--loss-height", type=int, default=120)
    args = parser.parse_args()
    app = LiveTrainingViewer(Path(args.run_dir), args.start_visualizing, args.loss_height)
    app.mainloop()


if __name__ == "__main__":
    main()
