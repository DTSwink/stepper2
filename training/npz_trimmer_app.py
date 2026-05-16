from __future__ import annotations

import argparse
import math
import sys
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import torch
from OpenGL import GL, GLU
from PIL import Image, ImageTk
from pyopengltk import OpenGLFrame

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
FBX_PIPELINE_DIR = PROJECT_ROOT / "fbx_npz_pipeline"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(FBX_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(FBX_PIPELINE_DIR))

import train_locomotion as tl


APP_ICON = PROJECT_ROOT / "training" / "assets" / "stepper_trimmer.ico"
APP_ICON_PNG = PROJECT_ROOT / "training" / "assets" / "stepper_trimmer.png"
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "ue5" / "animation_transitions_only_full"
FALLBACK_SOURCE_DIR = PROJECT_ROOT / "ue5" / "animations_omni_only" / "npz_final"

BG = "#101216"
PANEL = "#1a1d22"
PANEL_2 = "#23272f"
TEXT = "#edf1f7"
MUTED = "#9da6b5"
LINE = "#313846"
SKY = "#000000"
FLOOR = "#303846"
PATH_DIM = (0.25, 0.45, 0.40, 0.58)
PATH_BRIGHT = (0.28, 1.00, 0.72, 0.92)
CHARACTER = (0.45, 0.65, 1.00, 1.0)
CURRENT_PIN = (0.94, 0.98, 1.00, 1.0)
TRIM_GREEN = "#39d98a"
FLOOR_SIZE = 48.0
POLL_MS = 400


@dataclass
class MotionData:
    path: Path
    clip: tl.MotionClip
    positions: np.ndarray
    parents: list[int]
    names: list[str]
    root_index: int
    fps: float

    @property
    def frame_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def root_path(self) -> np.ndarray:
        return self.positions[:, self.root_index, :]

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        flat = self.positions.reshape(-1, 3)
        return flat.min(axis=0), flat.max(axis=0)


class TrimmerGLFrame(OpenGLFrame):
    def __init__(self, *args, app: "NpzTrimmerApp", **kwargs):
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
        GL.glEnable(GL.GL_POINT_SMOOTH)
        GL.glHint(GL.GL_POINT_SMOOTH_HINT, GL.GL_NICEST)
        GL.glDisable(GL.GL_LIGHTING)

    def redraw(self) -> None:
        self.app.render_gl()


class NpzTrimmerApp(tk.Tk):
    def __init__(
        self,
        source_dir: Path | None = None,
        open_first: bool = False,
        open_file: Path | None = None,
    ) -> None:
        super().__init__()
        self.withdraw()
        self.title("Stepper NPZ Trimmer")
        self.minsize(1120, 620)
        self.configure(bg=BG)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.source_dir = self.default_source_dir(source_dir)
        self.target_dir = self.target_for_source(self.source_dir)
        self.source_paths: list[Path] = []
        self.target_paths: list[Path] = []
        self.source_snapshot: tuple[tuple[str, int, int], ...] = ()
        self.target_snapshot: tuple[tuple[str, int, int], ...] = ()
        self.last_selected_list: str | None = None
        self.suppress_tree_select = False

        self.motion: MotionData | None = None
        self.opened_from_source = False
        self.playing = False
        self.last_play_tick = 0.0
        self.playback_accumulator = 0.0
        self.frame = 0
        self.trim_start = 0
        self.trim_end = 0
        self.timeline_drag: str | None = None
        self.window_drag_offset = 0
        self.anim_max_length_var = tk.StringVar(value="4000")
        self.yaw = -0.78
        self.pitch = -0.24
        self.distance = 3.2
        self.camera_target = np.asarray([0.0, 0.7, 0.0], dtype=np.float32)
        self.mouse_last: tuple[int, int] | None = None

        self.set_app_icon()
        self.configure_style()
        self.build_layout()
        self.bind("<Return>", self.on_enter_key)
        self.refresh_file_lists(force=True)
        if open_file is not None and open_file.exists():
            self.load_motion(open_file.resolve(), from_source=open_file.resolve().parent == self.source_dir)
        elif open_first and self.source_paths:
            self.load_motion(self.source_paths[0], from_source=True)
        self.after(POLL_MS, self.poll_files)
        self.after(16, self.playback_tick)
        self.after(100, self.draw)
        self.center_window()
        self.deiconify()
        self.lift()
        self.focus_force()

    def default_source_dir(self, source_dir: Path | None) -> Path:
        if source_dir is not None:
            return self.resolve_npz_folder(source_dir)
        if DEFAULT_SOURCE_DIR.exists():
            return self.resolve_npz_folder(DEFAULT_SOURCE_DIR)
        return self.resolve_npz_folder(FALLBACK_SOURCE_DIR)

    def resolve_npz_folder(self, folder: Path) -> Path:
        resolved = folder.resolve()
        if resolved.exists() and any(resolved.glob("*.npz")):
            return resolved
        nested_final = resolved / "npz_final"
        if nested_final.exists() and any(nested_final.glob("*.npz")):
            return nested_final.resolve()
        nested_raw = resolved / "npz"
        if nested_raw.exists() and any(nested_raw.glob("*.npz")):
            return nested_raw.resolve()
        return resolved

    def target_for_source(self, source: Path) -> Path:
        return source.with_name(f"{source.name}_trimmed")

    def set_app_icon(self) -> None:
        try:
            if APP_ICON.exists():
                self.iconbitmap(default=str(APP_ICON))
        except tk.TclError:
            pass
        try:
            if APP_ICON_PNG.exists():
                self.icon_photo = ImageTk.PhotoImage(Image.open(APP_ICON_PNG))
                self.iconphoto(True, self.icon_photo)
        except Exception:
            pass

    def center_window(self) -> None:
        self.update_idletasks()
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        width = min(1480, max(1120, int(screen_w * 0.78)), max(900, screen_w - 48))
        height = min(760, max(650, int(screen_h * 0.68)), max(650, screen_h - 180))
        x = 24 if screen_w <= 1700 else 40
        y = 40 if screen_h <= 1300 else max(0, (screen_h - height) // 2 - 8)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=PANEL, foreground=TEXT, fieldbackground=PANEL_2, bordercolor=LINE)
        style.configure("TFrame", background=PANEL)
        style.configure("Toolbar.TFrame", background=PANEL_2)
        style.configure("TLabel", background=PANEL, foreground=TEXT)
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED)
        style.configure("Header.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10, "bold"))
        style.configure("TButton", background=PANEL_2, foreground=TEXT, borderwidth=1, focusthickness=0)
        style.map("TButton", background=[("active", "#2b313b")])
        style.configure("Treeview", background="#14171c", foreground=TEXT, fieldbackground="#14171c", rowheight=23)
        style.configure("Treeview.Heading", background=PANEL_2, foreground=TEXT)

    def build_layout(self) -> None:
        root = ttk.Frame(self)
        root.pack(fill=tk.BOTH, expand=True)

        main = tk.PanedWindow(
            root,
            orient=tk.HORIZONTAL,
            bg=LINE,
            sashwidth=6,
            sashrelief=tk.FLAT,
            borderwidth=0,
            showhandle=False,
        )
        main.pack(fill=tk.BOTH, expand=True)

        viewer = ttk.Frame(main)
        viewer.rowconfigure(0, weight=1)
        viewer.columnconfigure(0, weight=1)

        self.canvas = TrimmerGLFrame(viewer, app=self)
        self.canvas.grid(row=0, column=0, sticky=tk.NSEW)
        self.canvas.bind("<Configure>", lambda _event: self.draw())
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)

        self.empty_hint = ttk.Label(
            viewer,
            text="Double-click a source animation to preview and trim.",
            style="Muted.TLabel",
            anchor=tk.CENTER,
        )
        self.empty_hint.place(relx=0.5, rely=0.48, anchor=tk.CENTER)

        bottom = ttk.Frame(viewer, style="Toolbar.TFrame", padding=(10, 8))
        bottom.grid(row=1, column=0, sticky=tk.EW)
        bottom.columnconfigure(1, weight=1)
        bottom.rowconfigure(0, minsize=48)

        self.trim_button = tk.Button(
            bottom,
            text="TRIM",
            bg=TRIM_GREEN,
            activebackground="#55e6a2",
            fg="#07140d",
            activeforeground="#07140d",
            relief=tk.FLAT,
            font=("Segoe UI", 12, "bold"),
            padx=22,
            command=self.trim_current_motion,
        )
        self.trim_button.grid(row=0, column=0, columnspan=3, sticky=tk.EW, pady=(0, 7))

        self.play_button = ttk.Button(bottom, text="Play", width=8, command=self.toggle_playback)
        self.play_button.grid(row=1, column=0, sticky=tk.W, padx=(0, 8))
        self.timeline = tk.Canvas(bottom, height=36, bg=PANEL_2, highlightthickness=0, cursor="hand2")
        self.timeline.grid(row=1, column=1, sticky=tk.EW)
        self.timeline.bind("<Configure>", lambda _event: self.draw_timeline())
        self.timeline.bind("<ButtonPress-1>", self.on_timeline_down)
        self.timeline.bind("<B1-Motion>", self.on_timeline_drag)
        self.timeline.bind("<ButtonRelease-1>", self.on_timeline_up)
        right_controls = ttk.Frame(bottom, style="Toolbar.TFrame")
        right_controls.grid(row=1, column=2, padx=(10, 0), sticky=tk.E)
        ttk.Label(right_controls, text="Anim max length", style="Muted.TLabel").grid(row=0, column=0, sticky=tk.E, padx=(0, 5))
        self.anim_max_length_spin = ttk.Spinbox(
            right_controls,
            from_=1,
            to=999999,
            increment=1,
            textvariable=self.anim_max_length_var,
            width=7,
            command=self.on_anim_max_length_changed,
        )
        self.anim_max_length_spin.grid(row=0, column=1, sticky=tk.E)
        self.anim_max_length_spin.bind("<Return>", self.on_anim_max_length_changed)
        self.anim_max_length_spin.bind("<FocusOut>", self.on_anim_max_length_changed)
        self.frame_label = ttk.Label(right_controls, text="Frame 0 / 0", style="Muted.TLabel", width=18, anchor=tk.E)
        self.frame_label.grid(row=1, column=0, columnspan=2, sticky=tk.E, pady=(3, 0))
        self.trim_label = ttk.Label(bottom, text="", style="Muted.TLabel", anchor=tk.W)
        self.trim_label.grid(row=2, column=0, columnspan=3, sticky=tk.EW, pady=(4, 0))

        side = ttk.Frame(main, width=560, padding=(10, 8))
        side.pack_propagate(False)
        side.columnconfigure(0, weight=1)
        side.columnconfigure(1, weight=1)
        side.rowconfigure(1, weight=1)
        main.add(viewer, minsize=520, stretch="always")
        main.add(side, minsize=360, stretch="never")
        self.main_pane = main

        source_header = ttk.Frame(side)
        source_header.grid(row=0, column=0, sticky=tk.EW, padx=(0, 5), pady=(0, 6))
        source_header.columnconfigure(0, weight=1)
        ttk.Label(source_header, text="Source Anims", style="Header.TLabel").grid(row=0, column=0, sticky=tk.W)
        ttk.Button(source_header, text="Folder...", width=9, command=self.choose_source_folder).grid(row=0, column=1, sticky=tk.E)

        target_header = ttk.Frame(side)
        target_header.grid(row=0, column=1, sticky=tk.EW, padx=(5, 0), pady=(0, 6))
        target_header.columnconfigure(0, weight=1)
        ttk.Label(target_header, text="Target Anims", style="Header.TLabel").grid(row=0, column=0, sticky=tk.W)
        ttk.Button(target_header, text="Delete", width=8, command=self.delete_selected_target).grid(
            row=0, column=1, sticky=tk.E
        )

        source_list_frame, self.source_tree = self.make_anim_list(side)
        source_list_frame.grid(row=1, column=0, sticky=tk.NSEW, padx=(0, 5))
        self.source_tree.bind("<Double-1>", self.on_source_double_click)
        self.source_tree.bind("<<TreeviewSelect>>", lambda _event: self.on_tree_selected("source"))
        self.source_tree.bind("<Return>", self.on_enter_key)

        target_list_frame, self.target_tree = self.make_anim_list(side)
        target_list_frame.grid(row=1, column=1, sticky=tk.NSEW, padx=(5, 0))
        self.target_tree.bind("<Double-1>", self.on_target_double_click)
        self.target_tree.bind("<<TreeviewSelect>>", lambda _event: self.on_tree_selected("target"))
        self.target_tree.bind("<Return>", self.on_enter_key)

        self.source_path_label = ttk.Label(side, text="", style="Muted.TLabel", wraplength=260, justify=tk.LEFT)
        self.source_path_label.grid(row=2, column=0, sticky=tk.EW, padx=(0, 5), pady=(8, 0))
        self.target_path_label = ttk.Label(side, text="", style="Muted.TLabel", wraplength=260, justify=tk.LEFT)
        self.target_path_label.grid(row=2, column=1, sticky=tk.EW, padx=(5, 0), pady=(8, 0))
        self.status_label = ttk.Label(side, text="", style="Muted.TLabel", anchor=tk.W)
        self.status_label.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=(10, 0))

    def make_anim_list(self, parent: tk.Widget) -> tuple[ttk.Frame, ttk.Treeview]:
        frame = ttk.Frame(parent)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        tree = ttk.Treeview(frame, columns=("frames",), show="tree headings", selectmode="browse")
        tree.heading("#0", text="Name")
        tree.heading("frames", text="Frames")
        tree.column("#0", width=190, minwidth=140, stretch=True)
        tree.column("frames", width=58, minwidth=52, anchor=tk.E, stretch=False)
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.grid(row=0, column=0, sticky=tk.NSEW)
        scroll.grid(row=0, column=1, sticky=tk.NS)
        return frame, tree

    def choose_source_folder(self) -> None:
        folder = filedialog.askdirectory(
            title="Select source NPZ folder",
            initialdir=str(self.source_dir if self.source_dir.exists() else PROJECT_ROOT),
        )
        if not folder:
            return
        self.set_source_folder(Path(folder))

    def set_source_folder(self, folder: Path) -> None:
        self.source_dir = self.resolve_npz_folder(folder)
        self.target_dir = self.target_for_source(self.source_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.source_snapshot = ()
        self.target_snapshot = ()
        self.last_selected_list = None
        self.refresh_file_lists(force=True)
        self.status_label.configure(text=f"Target folder ready: {self.target_dir.name}")

    def list_npz(self, folder: Path) -> list[Path]:
        if not folder.exists():
            return []
        return sorted(folder.glob("*.npz"), key=lambda item: item.name.lower())

    def folder_snapshot(self, folder: Path) -> tuple[tuple[str, int, int], ...]:
        if not folder.exists():
            return ()
        entries = []
        for path in self.list_npz(folder):
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append((path.name, int(stat.st_mtime_ns), int(stat.st_size)))
        return tuple(entries)

    def refresh_file_lists(self, force: bool = False) -> None:
        self.target_dir.mkdir(parents=True, exist_ok=True)
        source_snapshot = self.folder_snapshot(self.source_dir)
        target_snapshot = self.folder_snapshot(self.target_dir)
        source_changed = force or source_snapshot != self.source_snapshot
        target_changed = force or target_snapshot != self.target_snapshot
        if source_changed:
            self.source_snapshot = source_snapshot
            self.source_paths = self.list_npz(self.source_dir)
            self.populate_tree(self.source_tree, self.source_paths)
        if target_changed or source_changed:
            self.target_snapshot = target_snapshot
            self.target_paths = self.list_npz(self.target_dir)
            self.populate_tree(self.target_tree, self.target_paths, min_rows=len(self.source_paths))
        self.source_path_label.configure(text=str(self.source_dir))
        self.target_path_label.configure(text=str(self.target_dir))

    def populate_tree(self, tree: ttk.Treeview, paths: list[Path], min_rows: int | None = None) -> None:
        selected_name = None
        selection = tree.selection()
        if selection:
            selected_name = tree.item(selection[0], "text")
        was_suppressed = self.suppress_tree_select
        self.suppress_tree_select = True
        try:
            tree.delete(*tree.get_children())
            restore_id = None
            for idx, path in enumerate(paths):
                frames = self.quick_frame_count(path)
                item_id = str(idx)
                tree.insert("", tk.END, iid=item_id, text=path.name, values=(frames,))
                if path.name == selected_name:
                    restore_id = item_id
            filler_count = max(0, (min_rows or 0) - len(paths))
            for filler_idx in range(filler_count):
                tree.insert("", tk.END, iid=f"blank-{filler_idx}", text="", values=("",))
            if restore_id is not None:
                tree.selection_set(restore_id)
        finally:
            self.suppress_tree_select = was_suppressed

    def quick_frame_count(self, path: Path) -> str:
        try:
            with np.load(path, allow_pickle=False) as arrays:
                if "frame_count" in arrays.files:
                    return str(int(np.asarray(arrays["frame_count"]).item()))
                if "global_joint_pos" in arrays.files:
                    return str(int(arrays["global_joint_pos"].shape[0]))
        except Exception:
            return "?"
        return "?"

    def poll_files(self) -> None:
        self.refresh_file_lists()
        self.after(POLL_MS, self.poll_files)

    def on_tree_selected(self, list_name: str) -> None:
        if self.suppress_tree_select:
            return
        if list_name == "source":
            path = self.selected_path(self.source_tree, self.source_paths)
        else:
            path = self.selected_path(self.target_tree, self.target_paths)
        if path is not None:
            self.last_selected_list = list_name

    def on_source_double_click(self, _event: tk.Event) -> None:
        path = self.path_from_tree_event(self.source_tree, self.source_paths, _event)
        if path is not None:
            self.last_selected_list = "source"
            self.load_motion(path, from_source=True)

    def on_target_double_click(self, _event: tk.Event) -> None:
        path = self.path_from_tree_event(self.target_tree, self.target_paths, _event)
        if path is not None:
            self.last_selected_list = "target"
            self.load_motion(path, from_source=False)

    def on_enter_key(self, event: tk.Event) -> str | None:
        widget = event.widget if event is not None else self.focus_get()
        widget_class = ""
        try:
            widget_class = str(widget.winfo_class())
        except tk.TclError:
            pass
        if widget_class in {"Entry", "TEntry", "Spinbox", "TSpinbox"}:
            return None
        if widget is self.source_tree:
            self.last_selected_list = "source"
        elif widget is self.target_tree:
            self.last_selected_list = "target"
        candidate = self.selected_launch_candidate()
        if candidate is None:
            return None
        path, from_source = candidate
        if self.load_motion(path, from_source=from_source):
            self.start_playback()
        return "break"

    def selected_launch_candidate(self) -> tuple[Path, bool] | None:
        list_order: list[str] = []
        if self.last_selected_list in {"source", "target"}:
            list_order.append(self.last_selected_list)
        focus = self.focus_get()
        if focus is self.source_tree and "source" not in list_order:
            list_order.append("source")
        if focus is self.target_tree and "target" not in list_order:
            list_order.append("target")
        for fallback in ("source", "target"):
            if fallback not in list_order:
                list_order.append(fallback)
        for list_name in list_order:
            if list_name == "source":
                path = self.selected_path(self.source_tree, self.source_paths)
                if path is not None:
                    return path, True
            else:
                path = self.selected_path(self.target_tree, self.target_paths)
                if path is not None:
                    return path, False
        return None

    def delete_selected_target(self) -> None:
        path = self.selected_path(self.target_tree, self.target_paths)
        if path is None:
            self.status_label.configure(text="Select a target animation to delete.")
            return
        if path.parent.resolve() != self.target_dir.resolve():
            self.status_label.configure(text="Delete skipped: selected file is outside the target folder.")
            return
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            messagebox.showerror("Delete failed", str(exc))
            return
        if self.motion is not None and self.motion.path.resolve() == path.resolve():
            self.stop_playback()
            self.motion = None
            self.opened_from_source = False
            self.frame = 0
            self.trim_start = 0
            self.trim_end = 0
            self.empty_hint.place(relx=0.5, rely=0.48, anchor=tk.CENTER)
            self.draw_timeline()
            self.draw()
        self.refresh_file_lists(force=True)
        self.status_label.configure(text=f"Deleted target: {path.name}")

    def path_from_tree_event(self, tree: ttk.Treeview, paths: list[Path], event: tk.Event) -> Path | None:
        item = tree.identify_row(event.y)
        if not item:
            return None
        tree.selection_set(item)
        tree.focus(item)
        try:
            index = int(item)
        except ValueError:
            return None
        if index < 0 or index >= len(paths):
            return None
        return paths[index]

    def selected_path(self, tree: ttk.Treeview, paths: list[Path]) -> Path | None:
        selection = tree.selection()
        if not selection:
            return None
        try:
            index = int(selection[0])
        except ValueError:
            return None
        if index < 0 or index >= len(paths):
            return None
        return paths[index]

    def load_motion(self, path: Path, from_source: bool) -> bool:
        camera_state = self.capture_camera_state()
        self.stop_playback()
        try:
            cfg = tl.TrainConfig()
            cfg.use_torch_compile = False
            clip = tl.MotionClip(path, cfg)
            positions = clip.global_pos.detach().cpu().numpy().astype(np.float32)
            root_index = clip.body_names.index("root") if "root" in clip.body_names else 0
            motion = MotionData(
                path=path,
                clip=clip,
                positions=positions,
                parents=[int(parent) for parent in clip.parents_body_list],
                names=list(clip.body_names),
                root_index=root_index,
                fps=float(clip.fps),
            )
        except Exception as exc:
            messagebox.showerror("Animation load failed", str(exc))
            return False
        self.motion = motion
        self.restore_camera_state(camera_state)
        self.opened_from_source = bool(from_source)
        self.frame = 0
        self.trim_start = 0
        self.trim_end = max(0, motion.frame_count - 1)
        self.apply_anim_max_length_to_range(redraw=False)
        self.empty_hint.place_forget()
        self.update_trim_controls()
        self.draw_timeline()
        self.draw()
        origin = "source" if from_source else "target"
        self.status_label.configure(text=f"Loaded {origin}: {path.name}")
        return True

    def capture_camera_state(self) -> tuple[float, float, float, np.ndarray]:
        return float(self.yaw), float(self.pitch), float(self.distance), self.camera_target.copy()

    def restore_camera_state(self, state: tuple[float, float, float, np.ndarray]) -> None:
        self.yaw, self.pitch, self.distance, target = state
        self.camera_target = target.copy()

    def fit_camera_to_motion(self) -> None:
        if self.motion is None:
            return
        bounds_min, bounds_max = self.motion.bounds
        root = self.motion.root_path[min(self.frame, self.motion.frame_count - 1)]
        center = (bounds_min + bounds_max) * 0.5
        self.camera_target = np.asarray([root[0], max(0.55, center[1]), root[2]], dtype=np.float32)
        horizontal = np.linalg.norm((bounds_max - bounds_min)[[0, 2]])
        vertical = max(0.8, float(bounds_max[1] - bounds_min[1]))
        self.distance = max(2.1, min(12.0, horizontal * 0.45 + vertical * 1.65))
        self.yaw = -0.78
        self.pitch = -0.24

    def update_trim_controls(self) -> None:
        if self.motion is None or not self.opened_from_source:
            self.trim_button.grid_remove()
        else:
            self.trim_button.grid()

    def current_anim_max_length(self) -> int:
        try:
            return max(1, int(float(self.anim_max_length_var.get())))
        except (tk.TclError, ValueError):
            return 4000

    def on_anim_max_length_changed(self, _event: tk.Event | None = None) -> str | None:
        self.apply_anim_max_length_to_range(redraw=True)
        return None

    def apply_anim_max_length_to_range(self, redraw: bool = True) -> None:
        if self.motion is None or not self.opened_from_source:
            return
        frame_count = self.motion.frame_count
        if frame_count <= 0:
            return
        desired_len = min(frame_count, self.current_anim_max_length())
        center = (float(self.trim_start) + float(self.trim_end)) * 0.5
        start = int(round(center - (desired_len - 1) * 0.5))
        start = max(0, min(start, frame_count - desired_len))
        end = start + desired_len - 1
        self.trim_start = start
        self.trim_end = end
        self.frame = max(self.trim_start, min(self.frame, self.trim_end))
        if redraw:
            self.draw_timeline()
            self.draw()

    def toggle_playback(self) -> None:
        if self.motion is None:
            return
        self.playing = not self.playing
        self.last_play_tick = time.perf_counter()
        self.playback_accumulator = 0.0
        self.play_button.configure(text="Pause" if self.playing else "Play")

    def start_playback(self) -> None:
        if self.motion is None:
            return
        self.playing = True
        self.last_play_tick = time.perf_counter()
        self.playback_accumulator = 0.0
        self.play_button.configure(text="Pause")

    def stop_playback(self) -> None:
        self.playing = False
        if hasattr(self, "play_button"):
            self.play_button.configure(text="Play")

    def playback_tick(self) -> None:
        if self.playing and self.motion is not None:
            now = time.perf_counter()
            dt = max(0.0, min(0.1, now - self.last_play_tick)) if self.last_play_tick > 0.0 else 0.0
            self.last_play_tick = now
            fps = max(1.0, float(self.motion.fps))
            self.playback_accumulator += dt * fps
            step = int(self.playback_accumulator)
            if step <= 0:
                self.after(16, self.playback_tick)
                return
            self.playback_accumulator -= step
            next_frame = self.frame + step
            start_frame = self.trim_start if self.opened_from_source else 0
            end_frame = self.trim_end if self.opened_from_source else self.motion.frame_count - 1
            if next_frame > end_frame:
                next_frame = start_frame
            self.frame = max(0, min(next_frame, self.motion.frame_count - 1))
            self.draw_timeline()
            self.draw()
        self.after(16, self.playback_tick)

    def draw(self) -> None:
        try:
            if getattr(self.canvas, "context_created", False) and self.canvas.winfo_ismapped():
                self.canvas._display()
        except Exception:
            pass

    def render_gl(self) -> None:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        GL.glViewport(0, 0, width, height)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluPerspective(45.0, width / max(1, height), 0.01, 120.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        eye = self.camera_eye()
        target = self.camera_target
        GLU.gluLookAt(
            float(eye[0]),
            float(eye[1]),
            float(eye[2]),
            float(target[0]),
            float(target[1]),
            float(target[2]),
            0.0,
            1.0,
            0.0,
        )
        self.draw_floor()
        if self.motion is not None:
            self.draw_root_path()
            self.draw_skeleton()
        GL.glFlush()

    def camera_eye(self) -> np.ndarray:
        cp = math.cos(self.pitch)
        forward = np.asarray(
            [math.sin(self.yaw) * cp, math.sin(self.pitch), math.cos(self.yaw) * cp],
            dtype=np.float32,
        )
        return self.camera_target - forward * float(self.distance)

    def draw_floor(self) -> None:
        size = self.floor_size()
        GL.glDisable(GL.GL_LIGHTING)
        GL.glColor4f(0.18, 0.22, 0.28, 1.0)
        GL.glBegin(GL.GL_QUADS)
        GL.glVertex3f(-size, 0.0, -size)
        GL.glVertex3f(size, 0.0, -size)
        GL.glVertex3f(size, 0.0, size)
        GL.glVertex3f(-size, 0.0, size)
        GL.glEnd()

    def floor_size(self) -> float:
        return FLOOR_SIZE

    def draw_root_path(self) -> None:
        if self.motion is None or self.motion.frame_count < 2:
            return
        path = self.motion.root_path
        GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_LINE_BIT | GL.GL_POINT_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_FALSE)
        try:
            self.draw_path_range(path, 0, self.motion.frame_count - 1, PATH_DIM, 2.0, dashed=True)
            if self.opened_from_source:
                self.draw_path_range(path, self.trim_start, self.trim_end, PATH_BRIGHT, 3.0, dashed=True)
            current = path[max(0, min(self.frame, self.motion.frame_count - 1))]
            GL.glPointSize(9.0)
            GL.glColor4f(*CURRENT_PIN)
            GL.glBegin(GL.GL_POINTS)
            GL.glVertex3f(float(current[0]), 0.026, float(current[2]))
            GL.glEnd()
        finally:
            GL.glDepthMask(GL.GL_TRUE)
            GL.glPopAttrib()

    def draw_path_range(
        self,
        path: np.ndarray,
        start: int,
        end: int,
        color: tuple[float, float, float, float],
        width: float,
        dashed: bool,
    ) -> None:
        start = max(0, min(start, path.shape[0] - 1))
        end = max(0, min(end, path.shape[0] - 1))
        if end <= start:
            return
        GL.glColor4f(*color)
        GL.glLineWidth(width)
        GL.glBegin(GL.GL_LINES)
        for idx in range(start + 1, end + 1):
            if dashed and idx % 2 == 0:
                continue
            a = path[idx - 1]
            b = path[idx]
            GL.glVertex3f(float(a[0]), 0.02, float(a[2]))
            GL.glVertex3f(float(b[0]), 0.02, float(b[2]))
        GL.glEnd()
        GL.glPointSize(max(4.0, width + 2.0))
        GL.glBegin(GL.GL_POINTS)
        for idx in range(start, end + 1, max(1, (end - start) // 24)):
            p = path[idx]
            GL.glVertex3f(float(p[0]), 0.024, float(p[2]))
        GL.glEnd()

    def draw_skeleton(self) -> None:
        if self.motion is None:
            return
        frame = max(0, min(self.frame, self.motion.frame_count - 1))
        pos = self.motion.positions[frame]
        GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_LINE_BIT | GL.GL_POINT_BIT)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_DEPTH_TEST)
        try:
            GL.glLineWidth(4.0)
            GL.glColor4f(*CHARACTER)
            GL.glBegin(GL.GL_LINES)
            for idx, parent in enumerate(self.motion.parents):
                if parent < 0:
                    continue
                a = pos[parent]
                b = pos[idx]
                GL.glVertex3f(float(a[0]), float(a[1]), float(a[2]))
                GL.glVertex3f(float(b[0]), float(b[1]), float(b[2]))
            GL.glEnd()
            GL.glPointSize(7.0)
            GL.glBegin(GL.GL_POINTS)
            for point in pos:
                GL.glVertex3f(float(point[0]), float(point[1]), float(point[2]))
            GL.glEnd()
        finally:
            GL.glPopAttrib()

    def on_mouse_down(self, event: tk.Event) -> None:
        self.mouse_last = (event.x, event.y)

    def on_mouse_drag(self, event: tk.Event) -> None:
        if self.mouse_last is None:
            self.mouse_last = (event.x, event.y)
            return
        last_x, last_y = self.mouse_last
        dx = event.x - last_x
        dy = event.y - last_y
        self.yaw += dx * 0.008
        self.pitch = max(-1.56, min(1.56, self.pitch + dy * 0.006))
        self.mouse_last = (event.x, event.y)
        self.draw()

    def on_mouse_up(self, _event: tk.Event) -> None:
        self.mouse_last = None

    def on_mouse_wheel(self, event: tk.Event) -> None:
        if event.delta > 0:
            self.distance *= 0.88
        else:
            self.distance *= 1.14
        self.distance = max(0.25, min(60.0, self.distance))
        self.draw()

    def timeline_bounds(self) -> tuple[int, int, int]:
        width = max(1, self.timeline.winfo_width())
        height = max(1, self.timeline.winfo_height())
        return 16, max(18, width - 16), height // 2

    def timeline_x_for_frame(self, frame: int) -> float:
        if self.motion is None or self.motion.frame_count <= 1:
            return float(self.timeline_bounds()[0])
        x0, x1, _y = self.timeline_bounds()
        t = frame / float(max(1, self.motion.frame_count - 1))
        return x0 + t * (x1 - x0)

    def frame_from_timeline_x(self, x: float) -> int:
        if self.motion is None:
            return 0
        x0, x1, _y = self.timeline_bounds()
        t = (x - x0) / max(1.0, x1 - x0)
        return max(0, min(self.motion.frame_count - 1, int(round(t * (self.motion.frame_count - 1)))))

    def timeline_hit(self, x: float, y: float) -> str:
        if self.motion is None or not self.opened_from_source:
            return "scrub"
        start_x = self.timeline_x_for_frame(self.trim_start)
        end_x = self.timeline_x_for_frame(self.trim_end)
        _x0, _x1, mid_y = self.timeline_bounds()
        if abs(x - start_x) <= 12:
            return "start"
        if abs(x - end_x) <= 12:
            return "end"
        center_x = (start_x + end_x) * 0.5
        if start_x < x < end_x and abs(y - (mid_y - 15)) <= 10:
            return "window"
        if abs(x - center_x) <= 10 and abs(y - (mid_y - 15)) <= 14:
            return "window"
        return "scrub"

    def on_timeline_down(self, event: tk.Event) -> None:
        if self.motion is None:
            return
        self.timeline_drag = self.timeline_hit(event.x, event.y)
        if self.timeline_drag in {"start", "end", "window"}:
            self.stop_playback()
        if self.timeline_drag == "window":
            click_frame = self.frame_from_timeline_x(event.x)
            self.window_drag_offset = click_frame - self.trim_start
        self.update_timeline_from_x(event.x)

    def on_timeline_drag(self, event: tk.Event) -> None:
        if self.motion is None or self.timeline_drag is None:
            return
        self.update_timeline_from_x(event.x)

    def on_timeline_up(self, event: tk.Event) -> None:
        if self.motion is not None and self.timeline_drag is not None:
            self.update_timeline_from_x(event.x)
        self.timeline_drag = None

    def update_timeline_from_x(self, x: float) -> None:
        if self.motion is None:
            return
        frame = self.frame_from_timeline_x(x)
        if self.timeline_drag == "start" and self.opened_from_source:
            new_start = max(0, min(frame, self.trim_end))
            max_len = self.current_anim_max_length()
            if self.trim_end - new_start + 1 > max_len:
                new_start = self.trim_end - max_len + 1
            self.trim_start = max(0, new_start)
            self.frame = max(self.trim_start, min(self.frame, self.trim_end))
        elif self.timeline_drag == "end" and self.opened_from_source:
            new_end = min(self.motion.frame_count - 1, max(frame, self.trim_start))
            max_len = self.current_anim_max_length()
            if new_end - self.trim_start + 1 > max_len:
                new_end = self.trim_start + max_len - 1
            self.trim_end = min(self.motion.frame_count - 1, new_end)
            self.frame = max(self.trim_start, min(self.frame, self.trim_end))
        elif self.timeline_drag == "window" and self.opened_from_source:
            window_len = self.trim_end - self.trim_start
            old_start = self.trim_start
            new_start = frame - self.window_drag_offset
            new_start = max(0, min(new_start, self.motion.frame_count - 1 - window_len))
            self.trim_start = new_start
            self.trim_end = new_start + window_len
            self.frame = max(self.trim_start, min(self.frame + (new_start - old_start), self.trim_end))
        else:
            if self.opened_from_source:
                self.frame = max(self.trim_start, min(frame, self.trim_end))
            else:
                self.frame = frame
        self.draw_timeline()
        self.draw()

    def draw_timeline(self) -> None:
        self.timeline.delete("all")
        width = max(1, self.timeline.winfo_width())
        height = max(1, self.timeline.winfo_height())
        x0, x1, y = self.timeline_bounds()
        self.timeline.create_rectangle(0, 0, width, height, fill=PANEL_2, outline="")
        self.timeline.create_line(x0, y, x1, y, fill=LINE, width=10, capstyle=tk.ROUND)
        if self.motion is None:
            self.timeline.create_text(width * 0.5, y, text="load animation", fill=MUTED)
            self.frame_label.configure(text="Frame 0 / 0")
            self.trim_label.configure(text="")
            self.trim_button.grid_remove()
            return
        max_frame = self.motion.frame_count - 1
        if self.opened_from_source:
            sx = self.timeline_x_for_frame(self.trim_start)
            ex = self.timeline_x_for_frame(self.trim_end)
            self.timeline.create_line(sx, y, ex, y, fill=TRIM_GREEN, width=10, capstyle=tk.ROUND)
            self.draw_window_handle(sx, ex, y)
            self.draw_handle(sx, y, "start")
            self.draw_handle(ex, y, "end")
            self.trim_button.grid()
            self.trim_label.configure(text=f"Trim range {self.trim_start} - {self.trim_end}  ({self.trim_end - self.trim_start + 1} frames)")
        else:
            self.trim_button.grid_remove()
            self.trim_label.configure(text="Target preview")
        px = self.timeline_x_for_frame(self.frame)
        self.timeline.create_line(px, 5, px, height - 5, fill="#edf1f7", width=2)
        self.timeline.create_oval(px - 5, y - 5, px + 5, y + 5, fill="#edf1f7", outline="#55d6a7", width=2)
        self.frame_label.configure(text=f"Frame {self.frame} / {max_frame}")

    def draw_handle(self, x: float, y: float, tag: str) -> None:
        points = [x, y - 16, x - 8, y - 4, x - 8, y + 12, x + 8, y + 12, x + 8, y - 4]
        self.timeline.create_polygon(points, fill=TRIM_GREEN, outline="#e9fff4", width=1, tags=(tag,))

    def draw_window_handle(self, start_x: float, end_x: float, y: float) -> None:
        grip_y = y - 15
        center_x = (start_x + end_x) * 0.5
        self.timeline.create_line(
            start_x + 8,
            grip_y,
            end_x - 8,
            grip_y,
            fill="#ff4b5c",
            width=3,
            capstyle=tk.ROUND,
        )
        self.timeline.create_rectangle(
            center_x - 5,
            grip_y - 6,
            center_x + 5,
            grip_y + 6,
            fill="#ff4b5c",
            outline="#ffd6db",
            width=1,
        )

    def trim_current_motion(self) -> None:
        if self.motion is None or not self.opened_from_source:
            return
        source = self.motion.path
        target = self.target_dir / source.name
        start = max(0, min(self.trim_start, self.motion.frame_count - 1))
        end = max(start, min(self.trim_end, self.motion.frame_count - 1))
        try:
            self.write_trimmed_npz(source, target, start, end)
        except Exception as exc:
            messagebox.showerror("Trim failed", str(exc))
            return
        self.status_label.configure(text=f"Trimmed {source.name}: frames {start}-{end} -> {target.name}")
        self.refresh_file_lists(force=True)

    def write_trimmed_npz(self, source: Path, target: Path, start: int, end: int) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with np.load(source, allow_pickle=True) as arrays:
            frame_count = self.source_frame_count(arrays)
            if frame_count <= 0:
                raise ValueError(f"{source.name} has no frame data.")
            start = max(0, min(start, frame_count - 1))
            end = max(start, min(end, frame_count - 1))
            payload = {}
            for key in arrays.files:
                value = arrays[key]
                if value.ndim >= 1 and value.shape[0] == frame_count:
                    payload[key] = value[start : end + 1]
                elif key == "frame_start":
                    payload[key] = np.asarray(int(np.asarray(value).item()) + start, dtype=value.dtype)
                elif key == "frame_end":
                    base_start = int(np.asarray(arrays["frame_start"]).item()) if "frame_start" in arrays.files else 0
                    payload[key] = np.asarray(base_start + end, dtype=value.dtype)
                elif key == "frame_count":
                    payload[key] = np.asarray(end - start + 1, dtype=value.dtype)
                else:
                    payload[key] = value
            if "frame_count" not in payload:
                payload["frame_count"] = np.asarray(end - start + 1, dtype=np.int32)
        temp = target.with_name(f".{target.stem}.tmp.npz")
        np.savez_compressed(temp, **payload)
        temp.replace(target)

    def source_frame_count(self, arrays: np.lib.npyio.NpzFile) -> int:
        if "global_joint_pos" in arrays.files:
            return int(arrays["global_joint_pos"].shape[0])
        for key in arrays.files:
            value = arrays[key]
            if value.ndim >= 1 and value.shape[0] > 1 and key not in {"bone_names", "bone_uids", "parents"}:
                return int(value.shape[0])
        return 0

    def on_close(self) -> None:
        self.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Stepper NPZ trimming tool.")
    parser.add_argument("--source", type=Path, default=None, help="Initial source NPZ folder.")
    parser.add_argument("--open-first", action="store_true", help="Open the first source animation on startup.")
    parser.add_argument("--open-file", type=Path, default=None, help="Open a specific NPZ on startup.")
    args = parser.parse_args()
    app = NpzTrimmerApp(source_dir=args.source, open_first=args.open_first, open_file=args.open_file)
    app.mainloop()


if __name__ == "__main__":
    main()
