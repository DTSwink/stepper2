from __future__ import annotations

import argparse
import csv
import html
import json
import math
from datetime import datetime
from pathlib import Path

import torch

import contact_physics as cp
import train_locomotion as tl
import visualize_model as vm


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_folder(folder: Path, cfg: tl.TrainConfig, cyclic: bool) -> list[tl.MotionClip]:
    return [tl.MotionClip(path, cfg, cyclic_animation=cyclic) for path in sorted(folder.glob("*.npz"))]


def signed_horizontal_angle(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_len = torch.linalg.norm(a, dim=-1)
    b_len = torch.linalg.norm(b, dim=-1)
    a_n = a / a_len.clamp_min(eps).unsqueeze(-1)
    b_n = b / b_len.clamp_min(eps).unsqueeze(-1)
    cross_y = a_n[:, 1] * b_n[:, 0] - a_n[:, 0] * b_n[:, 1]
    dot = (a_n * b_n).sum(dim=-1).clamp(-1.0, 1.0)
    angle = torch.atan2(cross_y, dot)
    return torch.where((a_len > eps) & (b_len > eps), angle, torch.zeros_like(angle))


def root_window_features(
    clip: tl.MotionClip,
    prev_idx: torch.Tensor,
    cur_idx: torch.Tensor,
    future_idx: torch.Tensor,
    cfg: tl.TrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    prev_pos, _prev_rot, prev_yaw, _prev_heading = tl.root_state(clip, prev_idx, cfg, device)
    cur_pos, _cur_rot, _cur_yaw, _cur_heading = tl.root_state(clip, cur_idx, cfg, device)
    fut_pos, _fut_rot, fut_yaw, _fut_heading = tl.root_state(clip, future_idx, cfg, device)
    root_delta_yaw = tl.wrap_angle(fut_yaw - prev_yaw)
    current_delta = (cur_pos - prev_pos)[:, [0, 2]]
    future_delta = (fut_pos - cur_pos)[:, [0, 2]]
    future_motion_bend = signed_horizontal_angle(current_delta, future_delta)
    return root_delta_yaw, future_motion_bend


def clip_vertical_yaw_points(
    clip: tl.MotionClip,
    cfg: tl.TrainConfig,
    device: torch.device,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    max_cur = clip.T - cfg.future_window - 1
    if max_cur < 1:
        raise ValueError(f"{clip.path.name} is too short for future_window={cfg.future_window}")

    cur_idx = torch.arange(1, max_cur + 1, dtype=torch.long, device=device)
    prev_idx = cur_idx - 1
    future_idx = cur_idx + cfg.future_window

    tensors = clip.tensors(device)
    prev_pos = tensors["global_pos"].index_select(0, prev_idx)
    prev_rot = tensors["global_rot"].index_select(0, prev_idx)
    cur_pos = tensors["global_pos"].index_select(0, cur_idx)
    cur_rot = tensors["global_rot"].index_select(0, cur_idx)
    yaw_speeds = cp.foot_vertical_yaw_speeds(
        prev_pos,
        prev_rot,
        cur_pos,
        cur_rot,
        tuple(clip.foot_indices),
        tuple(clip.toe_indices),
        clip.fps,
    )
    root_delta, future_motion_bend = root_window_features(clip, prev_idx, cur_idx, future_idx, cfg, device)
    planted_per_frame, side_idx, _heights = cp.planted_foot_values(
        yaw_speeds,
        cur_pos,
        cur_rot,
        tuple(clip.foot_indices),
        tuple(clip.toe_indices),
    )
    peak_i = int(torch.argmax(planted_per_frame).detach().cpu())

    rows: list[dict[str, float | int | str]] = []
    for i in range(cur_idx.numel()):
        rows.append(
            {
                "clip": clip.path.stem,
                "frame": int(cur_idx[i].detach().cpu()),
                "root_delta_yaw_deg": float(torch.rad2deg(root_delta[i]).detach().cpu()),
                "future_root_motion_bend_deg": float(torch.rad2deg(future_motion_bend[i]).detach().cpu()),
                "planted_foot_vertical_yaw_deg_per_s": float(torch.rad2deg(planted_per_frame[i]).detach().cpu()),
                "side": "L" if int(side_idx[i].detach().cpu()) == 0 else "R",
                "left_foot_vertical_yaw_deg_per_s": float(torch.rad2deg(yaw_speeds[i, 0]).detach().cpu()),
                "right_foot_vertical_yaw_deg_per_s": float(torch.rad2deg(yaw_speeds[i, 1]).detach().cpu()),
            }
        )
    return rows[peak_i], rows


@torch.no_grad()
def clip_autoreg_vertical_yaw_points(
    clip: tl.MotionClip,
    model: torch.nn.Module,
    cfg: tl.TrainConfig,
    device: torch.device,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    max_cur = clip.T - cfg.future_window - 1
    if max_cur < 2:
        raise ValueError(f"{clip.path.name} is too short for autoregressive future_window={cfg.future_window}")

    prev_idx = torch.tensor([0], dtype=torch.long, device=device)
    cur_idx = torch.tensor([1], dtype=torch.long, device=device)
    prev_pose = tl.get_pose_from_clip(clip, prev_idx, device)
    cur_pose = tl.get_pose_from_clip(clip, cur_idx, device)

    pred_pos = []
    pred_rot = []
    for idx in (prev_idx, cur_idx):
        pos, rot = tl.global_from_clip(clip, idx, cfg, device)
        pred_pos.append(pos.squeeze(0))
        pred_rot.append(rot.squeeze(0))

    for target in range(2, max_cur + 1):
        target_idx = torch.tensor([target], dtype=torch.long, device=device)
        root_pos, root_rot, _root_yaw, _heading = tl.root_state(clip, target_idx, cfg, device)
        inp = tl.build_input(clip, prev_idx, cur_idx, prev_pose, cur_pose, cfg, device)
        raw_out = tl.predict_next_raw(model, inp, cur_pose, cfg)
        next_pose, _raw_pose = tl.output_to_pose(raw_out, clip)
        global_pos, global_rot, canon_pos = tl.fk_from_pose(clip, root_pos, root_rot, next_pose, device)
        pred_pos.append(global_pos.squeeze(0))
        pred_rot.append(global_rot.squeeze(0))
        prev_pose = cur_pose
        cur_pose = {
            "pelvis_pos": next_pose["pelvis_pos"],
            "pelvis_rot6": next_pose["pelvis_rot6"],
            "nonpelvis_rot6": next_pose["nonpelvis_rot6"],
            "canon_pos": canon_pos,
            "contacts": next_pose["contacts"],
        }
        prev_idx = cur_idx
        cur_idx = target_idx

    pred_pos_t = torch.stack(pred_pos, dim=0)
    pred_rot_t = torch.stack(pred_rot, dim=0)
    eval_idx = torch.arange(2, max_cur + 1, dtype=torch.long, device=device)
    prev_eval_idx = eval_idx - 1
    future_idx = eval_idx + cfg.future_window
    yaw_speeds = cp.foot_vertical_yaw_speeds(
        pred_pos_t.index_select(0, prev_eval_idx),
        pred_rot_t.index_select(0, prev_eval_idx),
        pred_pos_t.index_select(0, eval_idx),
        pred_rot_t.index_select(0, eval_idx),
        tuple(clip.foot_indices),
        tuple(clip.toe_indices),
        clip.fps,
    )
    root_delta, future_motion_bend = root_window_features(clip, prev_eval_idx, eval_idx, future_idx, cfg, device)
    planted_per_frame, side_idx, _heights = cp.planted_foot_values(
        yaw_speeds,
        pred_pos_t.index_select(0, eval_idx),
        pred_rot_t.index_select(0, eval_idx),
        tuple(clip.foot_indices),
        tuple(clip.toe_indices),
    )
    peak_i = int(torch.argmax(planted_per_frame).detach().cpu())

    rows: list[dict[str, float | int | str]] = []
    for i in range(eval_idx.numel()):
        rows.append(
            {
                "clip": clip.path.stem,
                "frame": int(eval_idx[i].detach().cpu()),
                "root_delta_yaw_deg": float(torch.rad2deg(root_delta[i]).detach().cpu()),
                "future_root_motion_bend_deg": float(torch.rad2deg(future_motion_bend[i]).detach().cpu()),
                "planted_foot_vertical_yaw_deg_per_s": float(torch.rad2deg(planted_per_frame[i]).detach().cpu()),
                "side": "L" if int(side_idx[i].detach().cpu()) == 0 else "R",
                "left_foot_vertical_yaw_deg_per_s": float(torch.rad2deg(yaw_speeds[i, 0]).detach().cpu()),
                "right_foot_vertical_yaw_deg_per_s": float(torch.rad2deg(yaw_speeds[i, 1]).detach().cpu()),
            }
        )
    return rows[peak_i], rows


def run_detector_tests(device: torch.device) -> dict[str, float]:
    fps = 30.0
    positions = torch.zeros((1, 4, 3), dtype=torch.float32, device=device)
    positions[:, 1, 2] = 1.0
    positions[:, 2, 0] = 2.0
    positions[:, 3, 0] = 2.0
    positions[:, 3, 2] = 1.0
    foot_indices = (0, 2)
    toe_indices = (1, 3)

    def row_rot_x(angle: float) -> torch.Tensor:
        c = math.cos(angle)
        s = math.sin(angle)
        return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]], dtype=torch.float32, device=device)

    def row_rot_y(angle: float) -> torch.Tensor:
        c = math.cos(angle)
        s = math.sin(angle)
        return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float32, device=device)

    def row_rot_z(angle: float) -> torch.Tensor:
        c = math.cos(angle)
        s = math.sin(angle)
        return torch.tensor([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)

    prev_rot = torch.eye(3, dtype=torch.float32, device=device).reshape(1, 1, 3, 3).expand(1, 4, 3, 3).clone()

    def measure(rot: torch.Tensor) -> float:
        cur_rot = rot.reshape(1, 1, 3, 3).expand(1, 4, 3, 3).clone()
        speeds = cp.foot_vertical_yaw_speeds(positions, prev_rot, positions, cur_rot, foot_indices, toe_indices, fps)
        return float(torch.rad2deg(speeds.max()).detach().cpu())

    yaw_30 = measure(row_rot_y(math.radians(30.0)))
    pitch_30 = measure(row_rot_x(math.radians(30.0)))
    roll_30 = measure(row_rot_z(math.radians(30.0)))
    if abs(yaw_30 - 900.0) > 1e-2:
        raise AssertionError(f"expected 30 deg/frame yaw at 30 FPS = 900 deg/s, got {yaw_30}")
    if abs(pitch_30) > 1e-3:
        raise AssertionError(f"toe-to-heel pitch should not register as vertical yaw, got {pitch_30}")
    if abs(roll_30) > 1e-3:
        raise AssertionError(f"sole roll should not register as vertical yaw, got {roll_30}")
    return {"yaw_30_deg_per_s": yaw_30, "pitch_30_deg_per_s": pitch_30, "roll_30_deg_per_s": roll_30}


def write_html(output_dir: Path, peaks: list[dict[str, float | int | str]], cfg: tl.TrainConfig) -> Path:
    if not peaks:
        raise ValueError("no peaks to plot")
    xs = [float(row["root_delta_yaw_deg"]) for row in peaks]
    ys = [float(row["planted_foot_vertical_yaw_deg_per_s"]) for row in peaks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = 0.0, max(ys)
    pad_x = max((max_x - min_x) * 0.08, 5.0)
    pad_y = max((max_y - min_y) * 0.10, 10.0)
    min_x -= pad_x
    max_x += pad_x
    max_y += pad_y

    width = 1180
    height = 720
    left = 88
    right = 36
    top = 40
    bottom = 78
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(x: float) -> float:
        return left + (x - min_x) / max(1e-9, max_x - min_x) * plot_w

    def sy(y: float) -> float:
        return top + (max_y - y) / max(1e-9, max_y - min_y) * plot_h

    x_ticks = []
    for t in range(math.floor(min_x / 45) * 45, math.ceil(max_x / 45) * 45 + 1, 45):
        if min_x <= t <= max_x:
            x_ticks.append(t)
    y_step = 50 if max_y < 400 else 100
    y_ticks = list(range(0, int(math.ceil(max_y / y_step) * y_step) + 1, y_step))

    grid = []
    for tick in x_ticks:
        x = sx(tick)
        grid.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" class="grid"/>')
        grid.append(f'<text x="{x:.2f}" y="{height - 42}" class="tick" text-anchor="middle">{tick}</text>')
    for tick in y_ticks:
        y = sy(tick)
        grid.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" class="grid"/>')
        grid.append(f'<text x="{left - 12}" y="{y + 4:.2f}" class="tick" text-anchor="end">{tick}</text>')

    points = []
    for row in peaks:
        name = str(row["clip"])
        folder = str(row.get("folder", ""))
        color = "#59d6a5" if folder == "omni" else "#ff9b6a"
        x = sx(float(row["root_delta_yaw_deg"]))
        y = sy(float(row["planted_foot_vertical_yaw_deg_per_s"]))
        payload = html.escape(json.dumps(row), quote=True)
        points.append(f'<circle class="pt" cx="{x:.2f}" cy="{y:.2f}" r="6.5" fill="{color}" data-row="{payload}"><title>{html.escape(name)}</title></circle>')

    data_json = json.dumps(peaks)
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Foot Vertical Yaw Scatter</title>
  <style>
    :root {{ color-scheme: dark; --bg:#101216; --panel:#171b22; --text:#edf1f7; --muted:#aab4c4; --grid:#2b333e; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.4 "Segoe UI", system-ui, sans-serif; }}
    header {{ padding:16px 20px 4px; }}
    h1 {{ font-size:20px; margin:0 0 6px; }}
    .meta {{ color:var(--muted); display:flex; gap:16px; flex-wrap:wrap; }}
    main {{ padding:10px 20px 20px; }}
    svg {{ width:100%; max-width:{width}px; height:auto; background:var(--panel); border:1px solid #303845; border-radius:8px; }}
    .grid {{ stroke:var(--grid); stroke-width:1; }}
    .axis {{ stroke:#718096; stroke-width:1.5; }}
    .tick {{ fill:var(--muted); font-size:12px; }}
    .label {{ fill:var(--text); font-size:14px; font-weight:600; }}
    .pt {{ stroke:#f4f7fb; stroke-width:1.2; cursor:crosshair; }}
    .pt:hover {{ stroke-width:2.8; }}
    #tip {{ position:fixed; display:none; pointer-events:none; background:#0b0f14; color:var(--text); border:1px solid #445064; border-radius:7px; padding:8px 10px; box-shadow:0 8px 24px rgba(0,0,0,.35); white-space:pre; font-size:13px; }}
    .legend {{ margin:10px 0 0; color:var(--muted); display:flex; gap:18px; }}
    .sw {{ display:inline-block; width:12px; height:12px; border-radius:50%; margin-right:6px; vertical-align:-1px; }}
  </style>
</head>
<body>
  <header>
    <h1>Max Foot Vertical Yaw Angular Velocity vs Future Root Yaw</h1>
    <div class="meta">
      <span>{len(peaks)} animations</span>
      <span>future window: {cfg.future_window} frames ({cfg.future_window / cfg.fps:.3f}s)</span>
      <span>one point per animation: frame with highest foot vertical yaw speed</span>
    </div>
    <div class="legend"><span><i class="sw" style="background:#59d6a5"></i>omni</span><span><i class="sw" style="background:#ff9b6a"></i>transition</span></div>
  </header>
  <main>
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="scatter plot">
      {''.join(grid)}
      <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>
      <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>
      <text x="{left + plot_w * 0.5:.2f}" y="{height - 14}" class="label" text-anchor="middle">root yaw delta from previous frame to future-window end (deg)</text>
      <text x="22" y="{top + plot_h * 0.5:.2f}" class="label" text-anchor="middle" transform="rotate(-90 22 {top + plot_h * 0.5:.2f})">max foot vertical yaw speed (deg/s)</text>
      {''.join(points)}
    </svg>
  </main>
  <div id="tip"></div>
  <script>
    const rows = {data_json};
    const tip = document.getElementById('tip');
    document.querySelectorAll('.pt').forEach((el) => {{
      el.addEventListener('mousemove', (ev) => {{
        const row = JSON.parse(el.dataset.row);
        tip.style.display = 'block';
        tip.style.left = `${{ev.clientX + 14}}px`;
        tip.style.top = `${{ev.clientY + 14}}px`;
        tip.textContent =
          `${{row.clip}}\\n` +
          `folder: ${{row.folder}}\\n` +
          `frame: ${{row.frame}}  side: ${{row.side}}\\n` +
          `root delta yaw: ${{row.root_delta_yaw_deg.toFixed(2)}} deg\\n` +
          `planted-foot vertical yaw: ${{row.planted_foot_vertical_yaw_deg_per_s.toFixed(2)}} deg/s`;
      }});
      el.addEventListener('mouseleave', () => tip.style.display = 'none');
    }});
  </script>
</body>
</html>"""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def write_3d_html(
    output_dir: Path,
    series: list[tuple[str, list[dict[str, float | int | str]]]],
    cfg: tl.TrainConfig,
    checkpoint: Path | None,
) -> Path:
    all_rows = [row for _name, rows in series for row in rows]
    xs = [float(row["root_delta_yaw_deg"]) for row in all_rows]
    ys = [float(row["planted_foot_vertical_yaw_deg_per_s"]) for row in all_rows]
    zs = [float(row["future_root_motion_bend_deg"]) for row in all_rows]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = 0.0, max(ys)
    min_z, max_z = min(zs), max(zs)
    pad_x = max((max_x - min_x) * 0.08, 5.0)
    pad_y = max((max_y - min_y) * 0.10, 10.0)
    pad_z = max((max_z - min_z) * 0.08, 5.0)
    min_x -= pad_x
    max_x += pad_x
    max_y += pad_y
    min_z -= pad_z
    max_z += pad_z

    plot_data = {
        "bounds": {"x": [min_x, max_x], "y": [min_y, max_y], "z": [min_z, max_z]},
        "series": [{"name": name, "rows": rows} for name, rows in series],
    }
    plot_json = json.dumps(plot_data).replace("</", "<\\/")
    panels = "\n".join(
        f"""<section class="panel">
  <div class="panel-head">
    <h2>{html.escape(name)}</h2>
    <button type="button" class="reset-view">Reset View</button>
  </div>
  <svg class="plot" data-series="{i}" viewBox="0 0 720 620" role="img" aria-label="{html.escape(name)} 3D scatter"></svg>
</section>"""
        for i, (name, _rows) in enumerate(series)
    )
    checkpoint_text = html.escape(str(checkpoint)) if checkpoint else "none"
    page_template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>3D Foot Vertical Yaw Scatter</title>
  <style>
    :root {{ color-scheme: dark; --bg:#101216; --panel:#171b22; --text:#edf1f7; --muted:#aab4c4; --grid:#2b333e; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.4 "Segoe UI", system-ui, sans-serif; }}
    header {{ padding:16px 20px 4px; }}
    h1 {{ font-size:20px; margin:0 0 6px; }}
    .meta {{ color:var(--muted); display:flex; gap:16px; flex-wrap:wrap; }}
    main {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(560px,1fr)); gap:16px; padding:14px 20px 20px; }}
    .panel {{ background:#12161d; border:1px solid #303845; border-radius:10px; padding:10px; }}
    .panel-head {{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin:0 0 8px; }}
    h2 {{ margin:0; font-size:16px; }}
    button {{ color:var(--text); background:#232a35; border:1px solid #3d4655; border-radius:6px; padding:6px 9px; cursor:pointer; }}
    button:hover {{ border-color:#6c7a92; }}
    svg {{ width:100%; height:auto; display:block; cursor:grab; user-select:none; touch-action:none; }}
    svg.dragging {{ cursor:grabbing; }}
    .grid {{ stroke:var(--grid); stroke-width:1; }}
    .axis {{ stroke:#8d9bb0; stroke-width:1.5; }}
    .axis-label {{ fill:var(--muted); font-size:12px; }}
    .tick {{ fill:var(--muted); font-size:11px; paint-order:stroke; stroke:#171b22; stroke-width:3px; stroke-linejoin:round; }}
    .pt {{ stroke:#f4f7fb; stroke-width:1.1; cursor:crosshair; }}
    .pt:hover {{ stroke-width:2.8; }}
    #tip {{ position:fixed; display:none; pointer-events:none; background:#0b0f14; color:var(--text); border:1px solid #445064; border-radius:7px; padding:8px 10px; box-shadow:0 8px 24px rgba(0,0,0,.35); white-space:pre; font-size:13px; z-index:10; }}
    .legend {{ margin:10px 0 0; color:var(--muted); display:flex; gap:18px; }}
    .sw {{ display:inline-block; width:12px; height:12px; border-radius:50%; margin-right:6px; vertical-align:-1px; }}
  </style>
</head>
<body>
  <header>
    <h1>3D Max Foot Vertical Yaw vs Root Yaw and Future Path Bend</h1>
    <div class="meta">
      <span>__POINT_COUNT__ plotted points</span>
      <span>future window: __FUTURE_FRAMES__ frames (__FUTURE_SECONDS__s)</span>
      <span>drag either panel to rotate the 3D view</span>
      <span>checkpoint: __CHECKPOINT__</span>
    </div>
    <div class="legend"><span><i class="sw" style="background:#59d6a5"></i>omni</span><span><i class="sw" style="background:#ff9b6a"></i>transition</span></div>
  </header>
  <main>__PANELS__</main>
  <div id="tip"></div>
  <script id="plot-data" type="application/json">__PLOT_DATA__</script>
  <script>
    const data = JSON.parse(document.getElementById('plot-data').textContent);
    const tip = document.getElementById('tip');
    const bounds = data.bounds;
    const view = { yaw: -0.72, pitch: 0.42 };
    const size = { w: 720, h: 620 };

    function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
    function niceStep(span, target) {
      const raw = span / target;
      const mag = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1e-9))));
      const norm = raw / mag;
      const nice = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10;
      return nice * mag;
    }
    function ticks(lo, hi, target, forceStep=null) {
      const step = forceStep || niceStep(hi - lo, target);
      const out = [];
      for (let v = Math.ceil(lo / step) * step; v <= hi + step * 0.5; v += step) out.push(v);
      return out;
    }
    function normPoint(x, y, z) {
      return {
        x: ((x - bounds.x[0]) / Math.max(1e-9, bounds.x[1] - bounds.x[0]) - 0.5) * 2.0,
        y: ((y - bounds.y[0]) / Math.max(1e-9, bounds.y[1] - bounds.y[0])) * 1.65,
        z: ((z - bounds.z[0]) / Math.max(1e-9, bounds.z[1] - bounds.z[0]) - 0.5) * 2.0,
      };
    }
    function project(x, y, z) {
      const p = normPoint(x, y, z);
      const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw);
      const cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
      const x1 = p.x * cy - p.z * sy;
      const z1 = p.x * sy + p.z * cy;
      const y1 = p.y * cp - z1 * sp;
      const z2 = p.y * sp + z1 * cp;
      const perspective = 1.0 / (1.0 + z2 * 0.10);
      return {
        x: size.w * 0.52 + x1 * 225 * perspective,
        y: size.h * 0.72 - y1 * 260 * perspective,
        depth: z2,
      };
    }
    function svgLine(a, b, cls='axis') {
      const p1 = project(a[0], a[1], a[2]);
      const p2 = project(b[0], b[1], b[2]);
      return `<line x1="${p1.x.toFixed(2)}" y1="${p1.y.toFixed(2)}" x2="${p2.x.toFixed(2)}" y2="${p2.y.toFixed(2)}" class="${cls}"/>`;
    }
    function svgText(p, text, cls='tick', anchor='middle') {
      return `<text x="${p.x.toFixed(2)}" y="${p.y.toFixed(2)}" class="${cls}" text-anchor="${anchor}">${text}</text>`;
    }
    function fmt(v) {
      const av = Math.abs(v);
      return av >= 100 ? v.toFixed(0) : av >= 10 ? v.toFixed(1) : v.toFixed(2);
    }
    function axesHtml() {
      const x0 = bounds.x[0], x1 = bounds.x[1], y0 = bounds.y[0], y1 = bounds.y[1], z0 = bounds.z[0], z1 = bounds.z[1];
      let out = '<rect width="720" height="620" rx="8" fill="#171b22"/>';
      for (const x of ticks(x0, x1, 6, 45)) out += svgLine([x, y0, z0], [x, y0, z1], 'grid');
      for (const z of ticks(z0, z1, 6, 45)) out += svgLine([x0, y0, z], [x1, y0, z], 'grid');
      for (const y of ticks(y0, y1, 6)) {
        out += svgLine([x0, y, z0], [x1, y, z0], 'grid');
        out += svgLine([x0, y, z0], [x0, y, z1], 'grid');
      }
      out += svgLine([x0, y0, z0], [x1, y0, z0], 'axis');
      out += svgLine([x0, y0, z0], [x0, y1, z0], 'axis');
      out += svgLine([x0, y0, z0], [x0, y0, z1], 'axis');
      let p = project(x1, y0, z0); out += svgText({x:p.x+45,y:p.y+4}, 'root yaw delta deg', 'axis-label', 'start');
      p = project(x0, y1, z0); out += svgText({x:p.x+8,y:p.y-10}, 'vertical yaw deg/s', 'axis-label', 'start');
      p = project(x0, y0, z1); out += svgText({x:p.x+8,y:p.y+16}, 'future path bend deg', 'axis-label', 'start');
      for (const x of ticks(x0, x1, 6, 45)) {
        const pp = project(x, y0, z0);
        out += svgText({x:pp.x,y:pp.y+16}, fmt(x));
      }
      for (const y of ticks(y0, y1, 6)) {
        const pp = project(x0, y, z0);
        out += svgText({x:pp.x-8,y:pp.y+4}, fmt(y), 'tick', 'end');
      }
      for (const z of ticks(z0, z1, 6, 45)) {
        const pp = project(x0, y0, z);
        out += svgText({x:pp.x,y:pp.y+16}, fmt(z));
      }
      return out;
    }
    function renderPanel(svg) {
      const seriesIndex = Number(svg.dataset.series);
      const rows = data.series[seriesIndex].rows.slice().sort((a, b) => {
        const pa = project(a.root_delta_yaw_deg, a.planted_foot_vertical_yaw_deg_per_s, a.future_root_motion_bend_deg);
        const pb = project(b.root_delta_yaw_deg, b.planted_foot_vertical_yaw_deg_per_s, b.future_root_motion_bend_deg);
        return pa.depth - pb.depth;
      });
      let out = axesHtml();
      rows.forEach((row, i) => {
        const p = project(row.root_delta_yaw_deg, row.planted_foot_vertical_yaw_deg_per_s, row.future_root_motion_bend_deg);
        const color = row.folder === 'omni' ? '#59d6a5' : '#ff9b6a';
        out += `<circle class="pt" cx="${p.x.toFixed(2)}" cy="${p.y.toFixed(2)}" r="6.2" fill="${color}" data-index="${i}"><title>${row.clip}</title></circle>`;
      });
      svg.innerHTML = out;
      svg.querySelectorAll('.pt').forEach((el) => {
        el.addEventListener('mousemove', (ev) => {
          const row = rows[Number(el.dataset.index)];
        tip.style.display = 'block';
          tip.style.left = `${ev.clientX + 14}px`;
          tip.style.top = `${ev.clientY + 14}px`;
        tip.textContent =
            `${row.clip}\n` +
            `folder: ${row.folder}\n` +
            `frame: ${row.frame}  side: ${row.side}\n` +
            `root yaw delta: ${row.root_delta_yaw_deg.toFixed(2)} deg\n` +
            `future path bend: ${row.future_root_motion_bend_deg.toFixed(2)} deg\n` +
            `planted-foot vertical yaw: ${row.planted_foot_vertical_yaw_deg_per_s.toFixed(2)} deg/s`;
        });
      el.addEventListener('mouseleave', () => tip.style.display = 'none');
      });
    }
    function renderAll() {
      document.querySelectorAll('svg.plot').forEach(renderPanel);
    }
    document.querySelectorAll('svg.plot').forEach((svg) => {
      let dragging = false;
      let lastX = 0;
      let lastY = 0;
      svg.addEventListener('pointerdown', (ev) => {
        dragging = true;
        lastX = ev.clientX;
        lastY = ev.clientY;
        svg.classList.add('dragging');
        svg.setPointerCapture(ev.pointerId);
      });
      svg.addEventListener('pointermove', (ev) => {
        if (!dragging) return;
        const dx = ev.clientX - lastX;
        const dy = ev.clientY - lastY;
        lastX = ev.clientX;
        lastY = ev.clientY;
        view.yaw += dx * 0.010;
        view.pitch = clamp(view.pitch + dy * 0.008, -1.1, 1.15);
        renderAll();
      });
      svg.addEventListener('pointerup', (ev) => {
        dragging = false;
        svg.classList.remove('dragging');
        try { svg.releasePointerCapture(ev.pointerId); } catch (_e) {}
      });
      svg.addEventListener('pointerleave', () => {
        dragging = false;
        svg.classList.remove('dragging');
      });
    });
    document.querySelectorAll('.reset-view').forEach((button) => {
      button.addEventListener('click', () => {
        view.yaw = -0.72;
        view.pitch = 0.42;
        renderAll();
      });
    });
    renderAll();
  </script>
</body>
</html>"""
    page = (
        page_template.replace("__POINT_COUNT__", str(len(all_rows)))
        .replace("__FUTURE_FRAMES__", str(cfg.future_window))
        .replace("__FUTURE_SECONDS__", f"{cfg.future_window / cfg.fps:.3f}")
        .replace("__CHECKPOINT__", checkpoint_text)
        .replace("__PANELS__", panels)
        .replace("__PLOT_DATA__", plot_json)
        .replace("{{", "{")
        .replace("}}", "}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def find_latest_controller_checkpoint() -> Path:
    candidates = sorted(
        (PROJECT_ROOT / "training" / "runs").glob("*/checkpoints/checkpoint_best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        try:
            ckpt = torch.load(candidate, map_location="cpu", weights_only=False)
        except Exception:
            continue
        state = ckpt.get("model", {})
        if isinstance(state, dict) and any(str(k).startswith("net.") for k in state.keys()):
            return candidate
    raise FileNotFoundError("Could not find a controller checkpoint_best.pt")


def load_controller(checkpoint: Path, clip: tl.MotionClip, cfg: tl.TrainConfig, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    vm.apply_config_dict(cfg, ckpt.get("config", {}))
    cfg.device = str(device)
    cfg.use_torch_compile = False
    input_dim, output_dim = tl.make_batch_dims(clip, cfg)
    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    state = ckpt["model"]
    try:
        model.load_state_dict(state)
    except RuntimeError:
        stripped = {str(k).replace("_orig_mod.", "", 1): v for k, v in state.items()}
        model.load_state_dict(stripped)
    model.eval()
    return model


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--omni-folder", default="ue5/animations_omni_only_full/npz_final")
    parser.add_argument("--transition-folder", default="ue5/animations_transitions_only_full_trimmed/npz_final")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--future-window-seconds", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="", help="Controller checkpoint for autoregressive model plot. Use 'latest' for newest checkpoint_best.pt.")
    parser.add_argument("--no-model-plot", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = tl.TrainConfig()
    cfg.fps = args.fps
    cfg.future_window_seconds = args.future_window_seconds
    cfg.device = str(device)
    cfg.use_torch_compile = False

    tests = run_detector_tests(device)

    omni_folder = resolve_path(args.omni_folder)
    transition_folder = resolve_path(args.transition_folder)
    clips = [(clip, "omni") for clip in load_folder(omni_folder, cfg, cyclic=True)]
    clips += [(clip, "transition") for clip in load_folder(transition_folder, cfg, cyclic=False)]

    gt_peaks: list[dict[str, float | int | str]] = []
    gt_rows: list[dict[str, float | int | str]] = []
    for clip, folder in clips:
        peak, rows = clip_vertical_yaw_points(clip, cfg, device)
        peak["series"] = "ground_truth"
        peak["folder"] = folder
        for row in rows:
            row["series"] = "ground_truth"
            row["folder"] = folder
        gt_peaks.append(peak)
        gt_rows.extend(rows)

    checkpoint = None
    model_peaks: list[dict[str, float | int | str]] = []
    model_rows: list[dict[str, float | int | str]] = []
    if not args.no_model_plot:
        checkpoint = find_latest_controller_checkpoint() if args.checkpoint in ("", "latest") else resolve_path(args.checkpoint)
        model_cfg = tl.TrainConfig()
        model_cfg.fps = args.fps
        model_cfg.future_window_seconds = args.future_window_seconds
        model_cfg.device = str(device)
        model_cfg.use_torch_compile = False
        model = load_controller(checkpoint, clips[0][0], model_cfg, device)
        cfg = model_cfg
        for clip, folder in clips:
            peak, rows = clip_autoreg_vertical_yaw_points(clip, model, cfg, device)
            peak["series"] = "model_autoregressive"
            peak["folder"] = folder
            for row in rows:
                row["series"] = "model_autoregressive"
                row["folder"] = folder
            model_peaks.append(peak)
            model_rows.extend(rows)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = resolve_path(args.output_dir) if args.output_dir else PROJECT_ROOT / "training" / "runs" / "diagnostics" / f"foot_vertical_yaw_3d_scatter_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    series = [("Ground Truth", gt_peaks)]
    if model_peaks:
        series.append(("Model Autoregressive", model_peaks))
    html_path = write_3d_html(output_dir, series, cfg, checkpoint)
    latest = PROJECT_ROOT / "training" / "runs" / "diagnostics" / "latest_foot_vertical_yaw_scatter.html"
    latest.write_text(html_path.read_text(encoding="utf-8"), encoding="utf-8")

    write_csv(output_dir / "peaks_ground_truth.csv", gt_peaks)
    write_csv(output_dir / "per_frame_ground_truth.csv", gt_rows)
    if model_peaks:
        write_csv(output_dir / "peaks_model_autoregressive.csv", model_peaks)
        write_csv(output_dir / "per_frame_model_autoregressive.csv", model_rows)
    summary = {
        "tests": tests,
        "animation_count": len(gt_peaks),
        "future_window_frames": cfg.future_window,
        "future_window_seconds": cfg.future_window / cfg.fps,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "html": str(html_path),
        "latest_html": str(latest),
        "top_10_gt_by_vertical_yaw": sorted(gt_peaks, key=lambda x: float(x["planted_foot_vertical_yaw_deg_per_s"]), reverse=True)[:10],
        "top_10_model_by_vertical_yaw": sorted(model_peaks, key=lambda x: float(x["planted_foot_vertical_yaw_deg_per_s"]), reverse=True)[:10],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
