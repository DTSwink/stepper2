from __future__ import annotations

import argparse
import html
import json
import shutil
import time
from pathlib import Path

import numpy as np
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


def parse_fractions(text: str) -> list[float]:
    result = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        result.append(min(1.0, max(0.0, float(part))))
    return result or [0.0, 0.25, 0.5, 0.75, 1.0]


def checkpoint_path(run_dir: Path, checkpoint_name: str) -> Path:
    return run_dir / "checkpoints" / checkpoint_name


def load_rollout(npz_path: Path | None, checkpoint: Path, device: torch.device, max_frames: int | None):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    npz = npz_path or vm.infer_npz_path(ckpt, checkpoint)
    cfg = tl.TrainConfig()
    vm.apply_config_dict(cfg, ckpt.get("config", {}))
    cfg.device = str(device)
    cfg.use_torch_compile = False
    clip = tl.MotionClip(npz, cfg)
    model = vm.load_model(ckpt, clip, cfg, device)
    rollout = vm.rollout_model(model, clip, cfg, device, max_frames)
    return ckpt, npz, clip, rollout


def projection_bounds(gt_pos: np.ndarray, pred_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate([gt_pos.reshape(-1, 3), pred_pos.reshape(-1, 3)], axis=0)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    pad = np.maximum((hi - lo) * 0.10, 0.25)
    return lo - pad, hi + pad


def project(point: np.ndarray, lo: np.ndarray, hi: np.ndarray, width: int, height: int) -> tuple[float, float]:
    corners = np.array(
        [
            [lo[0], lo[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], hi[1], hi[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], lo[2]],
            [hi[0], hi[1], hi[2]],
        ],
        dtype=np.float32,
    )

    def raw(p: np.ndarray) -> np.ndarray:
        return np.array([p[0] - p[2] * 0.48, -p[1] + p[2] * 0.18], dtype=np.float32)

    projected = np.stack([raw(p) for p in corners], axis=0)
    p2 = raw(point)
    p_lo = projected.min(axis=0)
    p_hi = projected.max(axis=0)
    span = np.maximum(p_hi - p_lo, 1e-5)
    scale = min((width - 36) / span[0], (height - 42) / span[1])
    x = 18 + (p2[0] - p_lo[0]) * scale
    y = 20 + (p2[1] - p_lo[1]) * scale
    return float(x), float(y)


def svg_lines(
    pos: np.ndarray,
    parents: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    width: int,
    height: int,
    color: str,
    opacity: float,
    stroke_width: float,
) -> str:
    parts = []
    for child, parent in enumerate(parents.tolist()):
        if parent < 0:
            continue
        x1, y1 = project(pos[parent], lo, hi, width, height)
        x2, y2 = project(pos[child], lo, hi, width, height)
        parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{stroke_width:.2f}" opacity="{opacity:.3f}" stroke-linecap="round"/>'
        )
    for point in pos:
        x, y = project(point, lo, hi, width, height)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{stroke_width * 0.78:.2f}" fill="{color}" opacity="{opacity:.3f}"/>')
    return "\n".join(parts)


def svg_ground(lo: np.ndarray, hi: np.ndarray, width: int, height: int) -> str:
    z_values = np.linspace(lo[2], hi[2], 8)
    lines = []
    for z in z_values:
        x1, y1 = project(np.array([lo[0], 0.0, z], dtype=np.float32), lo, hi, width, height)
        x2, y2 = project(np.array([hi[0], 0.0, z], dtype=np.float32), lo, hi, width, height)
        lines.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#28303a" stroke-width="1"/>')
    return "\n".join(lines)


def snapshot_svg(
    gt_pos: np.ndarray,
    pred_pos: np.ndarray,
    parents: np.ndarray,
    frame: int,
    label: str,
    lo: np.ndarray,
    hi: np.ndarray,
    width: int = 420,
    height: int = 320,
) -> str:
    title = html.escape(label)
    gt = svg_lines(gt_pos[frame], parents, lo, hi, width, height, "#72a7ff", 0.78, 4.0)
    pred = svg_lines(pred_pos[frame], parents, lo, hi, width, height, "#ff9b6a", 0.88, 3.2)
    ground = svg_ground(lo, hi, width, height)
    return f"""<svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
  <rect width="{width}" height="{height}" rx="8" fill="#0f1217"/>
  {ground}
  {gt}
  {pred}
  <text x="14" y="24" fill="#edf1f7" font-size="14" font-family="Segoe UI, sans-serif">{title}</text>
</svg>"""


def write_report(
    output_dir: Path,
    checkpoint: Path,
    npz: Path,
    ckpt: dict,
    clip: tl.MotionClip,
    rollout,
    fractions: list[float],
) -> dict:
    gt_pos, _gt_rot, pred_tf_pos, _pred_tf_rot, error_tf, pred_ar_pos, _pred_ar_rot, error_ar = rollout
    frame_count = int(gt_pos.shape[0])
    sample_frames = sorted({min(frame_count - 1, max(0, int(round(frac * (frame_count - 1))))) for frac in fractions})
    lo, hi = projection_bounds(gt_pos, pred_ar_pos)
    parents = clip.parents_body.cpu().numpy().astype(int)
    gt_heights, _ = cp.foot_lowest_heights_and_points(
        torch.tensor(gt_pos, dtype=torch.float32),
        torch.tensor(_gt_rot, dtype=torch.float32),
        tuple(clip.foot_indices),
        tuple(clip.toe_indices),
        cp.DEFAULT_GEOMETRY,
    )
    pred_heights, _ = cp.foot_lowest_heights_and_points(
        torch.tensor(pred_ar_pos, dtype=torch.float32),
        torch.tensor(_pred_ar_rot, dtype=torch.float32),
        tuple(clip.foot_indices),
        tuple(clip.toe_indices),
        cp.DEFAULT_GEOMETRY,
    )
    gt_floor_penetration = (-gt_heights).clamp_min(0.0)
    pred_floor_penetration = (-pred_heights).clamp_min(0.0)
    extra_floor_penetration = (
        pred_floor_penetration.max() - gt_floor_penetration.max()
    ).clamp_min(0.0)
    svgs = [
        snapshot_svg(
            gt_pos,
            pred_ar_pos,
            parents,
            frame,
            f"{int(round(frame / max(1, frame_count - 1) * 100))}% / frame {frame}",
            lo,
            hi,
        )
        for frame in sample_frames
    ]
    metrics = {
        "checkpoint": str(checkpoint),
        "npz": str(npz),
        "epoch": ckpt.get("epoch"),
        "rollout_k": ckpt.get("rollout_k"),
        "best_val": ckpt.get("best_val"),
        "frame_count": frame_count,
        "sample_frames": sample_frames,
        "one_step_mean_joint_error_avg": float(error_tf.mean()),
        "one_step_mean_joint_error_max": float(error_tf.max()),
        "autoregressive_mean_joint_error_avg": float(error_ar.mean()),
        "autoregressive_mean_joint_error_max": float(error_ar.max()),
        "autoregressive_mean_joint_error_end": float(error_ar[-1]),
        "gt_min_foot_height_l_m": float(gt_heights[:, 0].min()),
        "gt_min_foot_height_r_m": float(gt_heights[:, 1].min()),
        "pred_min_foot_height_l_m": float(pred_heights[:, 0].min()),
        "pred_min_foot_height_r_m": float(pred_heights[:, 1].min()),
        "gt_floor_penetration_max_m": float(gt_floor_penetration.max()),
        "pred_floor_penetration_max_m": float(pred_floor_penetration.max()),
        "extra_floor_penetration_over_source_max_m": float(extra_floor_penetration),
        "updated_unix": time.time(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cards = "\n".join(f'<section class="card">{svg}</section>' for svg in svgs)
    title = f"{npz.stem} vs {checkpoint.parent.parent.name}"
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} visual report</title>
  <style>
    :root {{ color-scheme: dark; --bg:#101216; --panel:#1a1d22; --text:#edf1f7; --muted:#9da6b5; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.4 "Segoe UI", system-ui, sans-serif; }}
    header {{ padding:16px 18px 8px; }}
    h1 {{ margin:0 0 6px; font-size:18px; font-weight:650; }}
    .meta {{ color:var(--muted); display:flex; gap:12px; flex-wrap:wrap; }}
    main {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:12px; padding:12px 18px 18px; }}
    .card {{ background:var(--panel); border:1px solid rgba(255,255,255,.08); border-radius:8px; padding:8px; }}
    svg {{ width:100%; height:auto; display:block; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="meta">
      <span>epoch {metrics["epoch"]}</span>
      <span>K{metrics["rollout_k"]}</span>
      <span>AR avg {metrics["autoregressive_mean_joint_error_avg"]:.4f} m</span>
      <span>AR end {metrics["autoregressive_mean_joint_error_end"]:.4f} m</span>
      <span>one-step avg {metrics["one_step_mean_joint_error_avg"]:.4f} m</span>
      <span>foot pen max {metrics["pred_floor_penetration_max_m"]:.4f} m</span>
      <span>extra over source {metrics["extra_floor_penetration_over_source_max_m"]:.4f} m</span>
    </div>
  </header>
  <main>{cards}</main>
</body>
</html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    return metrics


def render_once(args: argparse.Namespace, checkpoint: Path, destination: Path) -> dict:
    device = torch.device(args.device)
    max_frames = None if args.max_frames <= 0 else args.max_frames
    npz = resolve_path(args.npz_path) if args.npz_path else None
    ckpt, resolved_npz, clip, rollout = load_rollout(npz, checkpoint, device, max_frames)
    return write_report(destination, checkpoint, resolved_npz, ckpt, clip, rollout, parse_fractions(args.sample_fractions))


def watch(args: argparse.Namespace) -> None:
    run_dir = resolve_path(args.run_dir)
    output_root = resolve_path(args.output_dir) if args.output_dir else run_dir / "visual_reports"
    output_root.mkdir(parents=True, exist_ok=True)
    latest_dir = output_root / "latest"
    last_mtime = -1.0
    last_size = -1
    while True:
        ckpt = checkpoint_path(run_dir, args.checkpoint_name)
        if ckpt.exists():
            stat = ckpt.stat()
            if stat.st_mtime != last_mtime or stat.st_size != last_size:
                try:
                    metrics = render_once(args, ckpt, latest_dir)
                    epoch = metrics.get("epoch", "unknown")
                    staged = output_root / f"epoch_{int(epoch):06d}_k{int(metrics.get('rollout_k') or 0):03d}"
                    if staged.exists():
                        shutil.rmtree(staged)
                    shutil.copytree(latest_dir, staged)
                    if not args.quiet:
                        print(f"visual report updated epoch={epoch} output={latest_dir / 'index.html'}", flush=True)
                    last_mtime = stat.st_mtime
                    last_size = stat.st_size
                except Exception as exc:
                    if not args.quiet:
                        print(f"visual report skipped: {exc}", flush=True)
        if args.once:
            return
        time.sleep(max(1.0, float(args.interval_seconds)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Asynchronous visual checkpoint report generator.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--npz-path", default="")
    parser.add_argument("--checkpoint-name", default="checkpoint_last.pt")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--sample-fractions", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    watch(args)


if __name__ == "__main__":
    main()
