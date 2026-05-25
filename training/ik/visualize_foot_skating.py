from __future__ import annotations

import argparse
import csv
import html
from dataclasses import fields
from pathlib import Path

import torch

try:
    from . import checkpoint_runtime as ckpt_runtime
    from . import contact_physics as cp
    from . import ik_core as tl
    from . import train_simple_ae_controller as simple_ctl
except ImportError:
    import checkpoint_runtime as ckpt_runtime
    import contact_physics as cp
    import ik_core as tl
    import train_simple_ae_controller as simple_ctl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def apply_config_dict(cfg: tl.TrainConfig, values: dict) -> None:
    valid = {field.name for field in fields(tl.TrainConfig)}
    for key, value in values.items():
        if key not in valid:
            continue
        current = getattr(cfg, key)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(cfg, key, value)


def load_model(checkpoint_path: Path, clip: tl.MotionClip, device: torch.device) -> tuple[torch.nn.Module, tl.TrainConfig, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = tl.TrainConfig()
    apply_config_dict(cfg, checkpoint.get("config", {}))
    cfg.device = str(device)
    cfg.use_torch_compile = False
    input_dim, output_dim = tl.make_batch_dims(clip, cfg)
    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, cfg, checkpoint


@torch.no_grad()
def rollout_autoreg(
    model: torch.nn.Module,
    checkpoint: dict,
    clip: tl.MotionClip,
    cfg: tl.TrainConfig,
    device: torch.device,
    frame_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ckpt_runtime.require_current_ik_controller_checkpoint(checkpoint)
    pred_pos = torch.zeros((frame_count, clip.J, 3), dtype=torch.float32, device=device)
    pred_rot = torch.zeros((frame_count, clip.J, 3, 3), dtype=torch.float32, device=device)
    gt_pos = clip.global_pos[:frame_count].to(device)
    gt_rot = clip.global_rot[:frame_count].to(device)
    pred_pos[:2] = gt_pos[:2]
    pred_rot[:2] = gt_rot[:2]

    store = simple_ctl.SimpleClipStore([clip], cfg, device)
    clip_ids = torch.zeros(1, dtype=torch.long, device=device)
    cur_idx = torch.tensor([1], dtype=torch.long, device=device)
    prev_vec, prev_pelvis, prev_payload = simple_ctl.target_state(store, clip_ids, cur_idx - 1)
    cur_vec, cur_pelvis, cur_payload = simple_ctl.target_state(store, clip_ids, cur_idx)

    for target in range(2, frame_count):
        target_idx = torch.tensor([target], dtype=torch.long, device=device)
        root_pos, root_rot, _root_yaw, _heading = store.root_state(clip_ids, target_idx)
        inp = simple_ctl.build_controller_input(
            store,
            clip_ids,
            cur_idx,
            prev_vec,
            cur_vec,
            prev_pelvis,
            cur_pelvis,
            prev_payload,
            cur_payload,
        )
        raw_out = simple_ctl.model_forward(model, inp, cur_vec, cfg)
        pred_vec = simple_ctl.clean_output_vector(raw_out, store)
        pred_pose, _raw_pose = tl.output_to_pose(pred_vec, clip)
        global_pos, global_rot, canon_pos = tl.fk_from_pose(clip, root_pos, root_rot, pred_pose, device)
        pred_pos[target] = global_pos[0]
        pred_rot[target] = global_rot[0]
        prev_vec = cur_vec
        prev_pelvis = cur_pelvis
        prev_payload = cur_payload
        cur_vec, cur_pelvis, cur_payload = simple_ctl.predicted_state_from_vector(pred_vec, store)
        cur_idx = target_idx
    return pred_pos, pred_rot


def percentile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values, q).detach().cpu())


@torch.no_grad()
def foot_series(
    positions: torch.Tensor,
    rotations: torch.Tensor,
    clip: tl.MotionClip,
) -> tuple[torch.Tensor, torch.Tensor]:
    foot_indices = tuple(int(x) for x in clip.foot_indices_tensor.tolist())
    toe_indices = tuple(int(x) for x in clip.toe_indices_tensor.tolist())
    slide = cp.foot_slide_speeds(
        positions[:-1],
        rotations[:-1],
        positions[1:],
        rotations[1:],
        foot_indices,
        toe_indices,
        clip.fps,
    )
    heights, _points = cp.foot_lowest_heights_and_points(
        positions[1:],
        rotations[1:],
        foot_indices,
        toe_indices,
    )
    return slide.detach().cpu(), heights.detach().cpu()


def planted_slide(slide: torch.Tensor, heights: torch.Tensor) -> torch.Tensor:
    planted = heights.argmin(dim=-1)
    return slide.gather(-1, planted.unsqueeze(-1)).squeeze(-1)


def summarize(
    label: str,
    pred_slide: torch.Tensor,
    pred_height: torch.Tensor,
    gt_slide: torch.Tensor,
    gt_height: torch.Tensor,
    contacts: torch.Tensor,
) -> dict[str, float | str]:
    contact = contacts > 0.5
    slide_excess_pred = planted_slide(pred_slide, pred_height)
    slide_excess_gt = planted_slide(gt_slide, gt_height)
    source_contact_pred = pred_slide[contact]
    source_contact_gt = gt_slide[contact]
    return {
        "label": label,
        "slide_excess_pred_mean": float(slide_excess_pred.mean()),
        "slide_excess_gt_mean": float(slide_excess_gt.mean()),
        "slide_excess_pred_p95": percentile(slide_excess_pred, 0.95),
        "slide_excess_gt_p95": percentile(slide_excess_gt, 0.95),
        "source_contact_pred_p95": percentile(source_contact_pred, 0.95),
        "source_contact_gt_p95": percentile(source_contact_gt, 0.95),
        "source_contact_excess_p95": percentile(torch.relu(source_contact_pred - source_contact_gt), 0.95),
        "left_pred_p95": percentile(pred_slide[:, 0], 0.95),
        "left_gt_p95": percentile(gt_slide[:, 0], 0.95),
        "right_pred_p95": percentile(pred_slide[:, 1], 0.95),
        "right_gt_p95": percentile(gt_slide[:, 1], 0.95),
    }


def transition_feature_horizon(cfg: tl.TrainConfig) -> int:
    root_lookahead_steps = max(0, int(getattr(cfg, "root_lookahead_steps", 0)))
    return max(int(cfg.future_window), root_lookahead_steps + 1)


def trainable_frame_count(clip: tl.MotionClip, cfgs: list[tl.TrainConfig], frame_count: int) -> int:
    if clip.cyclic_animation:
        return frame_count
    horizon = max((transition_feature_horizon(cfg) for cfg in cfgs), default=1)
    # Last trainable input is current=T-horizon-1, which predicts target=T-horizon.
    return max(3, min(frame_count, int(clip.T) - horizon + 1))


def svg_polyline(values: list[float], min_v: float, max_v: float, x0: int, y0: int, w: int, h: int) -> str:
    if len(values) <= 1:
        return ""
    span = max(max_v - min_v, 1e-8)
    pts = []
    for i, value in enumerate(values):
        x = x0 + w * i / (len(values) - 1)
        y = y0 + h - h * (value - min_v) / span
        pts.append(f"{x:.2f},{y:.2f}")
    return " ".join(pts)


def chart_svg(
    title: str,
    series: list[tuple[str, list[float], str]],
    contacts: list[int],
    y_label: str,
    width: int = 1060,
    height: int = 260,
) -> str:
    margin_l, margin_r, margin_t, margin_b = 68, 18, 34, 38
    x0, y0 = margin_l, margin_t
    w = width - margin_l - margin_r
    h = height - margin_t - margin_b
    vals = [v for _name, values, _color in series for v in values]
    min_v = min(0.0, min(vals) if vals else 0.0)
    max_v = max(vals) if vals else 1.0
    max_v = max(max_v, 1e-6)
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f'<text x="{x0}" y="22" class="title">{html.escape(title)}</text>',
        f'<text x="12" y="{y0 + h / 2:.1f}" class="axis" transform="rotate(-90 12 {y0 + h / 2:.1f})">{html.escape(y_label)}</text>',
        f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}" class="axis-line"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+h}" class="axis-line"/>',
    ]
    if contacts:
        n = len(contacts)
        for i, active in enumerate(contacts):
            if not active:
                continue
            x = x0 + w * i / max(1, n - 1)
            x_next = x0 + w * (i + 1) / max(1, n - 1)
            parts.append(f'<rect x="{x:.2f}" y="{y0}" width="{max(1.0, x_next-x):.2f}" height="{h}" class="contact-bg"/>')
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y0 + h - h * frac
        value = min_v + (max_v - min_v) * frac
        parts.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x0+w}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{x0-8}" y="{y+4:.2f}" text-anchor="end" class="tick">{value:.3f}</text>')
    for name, values, color in series:
        pts = svg_polyline(values, min_v, max_v, x0, y0, w, h)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        parts.append(f'<text x="{x0+w-130}" y="{y0+18+18*len(parts)%72}" fill="{color}" class="legend">{html.escape(name)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def write_report(output: Path, title: str, rows: list[dict[str, float | str]], charts: list[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    table_rows = []
    keys = list(rows[0].keys())
    for row in rows:
        table_rows.append(
            "<tr>" + "".join(f"<td>{html.escape(f'{row[k]:.5g}' if isinstance(row[k], float) else str(row[k]))}</td>" for k in keys) + "</tr>"
        )
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 18px; background: #101318; color: #d8dee9; font-family: Segoe UI, Arial, sans-serif; }}
    h1 {{ font-size: 20px; font-weight: 650; }}
    .chart {{ display: block; margin: 14px 0 24px; background: #151a21; border: 1px solid #2b333d; }}
    .title {{ fill: #e6edf3; font-size: 15px; font-weight: 650; }}
    .axis, .tick, .legend {{ font-size: 12px; fill: #aeb8c4; }}
    .axis-line {{ stroke: #53606f; stroke-width: 1.2; }}
    .grid {{ stroke: #2b333d; stroke-width: 1; }}
    .contact-bg {{ fill: #4d5968; opacity: 0.18; }}
    table {{ border-collapse: collapse; margin: 14px 0 22px; font-size: 12px; }}
    th, td {{ border: 1px solid #2b333d; padding: 5px 8px; }}
    th {{ background: #1b222b; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Gray bands are source contact frames. The key failure metric is source-contact slide: if the source foot should be planted but the generated foot is moving much faster than GT, the model is skating.</p>
  <table><thead><tr>{''.join(f'<th>{html.escape(k)}</th>' for k in keys)}</tr></thead><tbody>{''.join(table_rows)}</tbody></table>
  {''.join(charts)}
</body>
</html>
"""
    output.write_text(doc, encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize GT vs autoregressive model foot skating over time.")
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--checkpoint-path", action="append", required=True)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    npz_path = resolve_path(args.npz_path)
    checkpoint_cfgs: list[tl.TrainConfig] = []
    for checkpoint_text in args.checkpoint_path:
        checkpoint = torch.load(resolve_path(checkpoint_text), map_location="cpu", weights_only=False)
        cfg = tl.TrainConfig()
        apply_config_dict(cfg, checkpoint.get("config", {}))
        checkpoint_cfgs.append(cfg)
    first_checkpoint = torch.load(resolve_path(args.checkpoint_path[0]), map_location="cpu", weights_only=False)
    base_cfg = tl.TrainConfig()
    apply_config_dict(base_cfg, first_checkpoint.get("config", {}))
    clip = tl.MotionClip(npz_path, base_cfg)
    frame_count = clip.T if args.max_frames <= 0 else min(clip.T, args.max_frames)
    frame_count = max(3, frame_count)
    safe_frame_count = trainable_frame_count(clip, checkpoint_cfgs, frame_count)
    gt_slide, gt_height = foot_series(clip.global_pos[:frame_count].to(device), clip.global_rot[:frame_count].to(device), clip)
    source_contacts = clip.contacts[1:frame_count].cpu() > 0.5

    rows: list[dict[str, float | str]] = []
    charts: list[str] = []
    labels = args.label or []
    for i, checkpoint_text in enumerate(args.checkpoint_path):
        label = labels[i] if i < len(labels) else Path(checkpoint_text).parent.parent.name
        checkpoint_path = resolve_path(checkpoint_text)
        cfg = tl.TrainConfig()
        apply_config_dict(cfg, torch.load(checkpoint_path, map_location="cpu", weights_only=False).get("config", {}))
        clip_for_model = tl.MotionClip(npz_path, cfg)
        model, cfg, ckpt = load_model(checkpoint_path, clip_for_model, device)
        pred_pos, pred_rot = rollout_autoreg(model, ckpt, clip_for_model, cfg, device, frame_count)
        pred_slide, pred_height = foot_series(pred_pos, pred_rot, clip_for_model)
        rows.append(summarize(f"{label} [full]", pred_slide, pred_height, gt_slide, gt_height, source_contacts))
        if safe_frame_count < frame_count:
            safe_slice = slice(0, safe_frame_count - 1)
            rows.append(
                summarize(
                    f"{label} [trainable horizon]",
                    pred_slide[safe_slice],
                    pred_height[safe_slice],
                    gt_slide[safe_slice],
                    gt_height[safe_slice],
                    source_contacts[safe_slice],
                )
            )
        for side, side_name in enumerate(("Left", "Right")):
            charts.append(
                chart_svg(
                    f"{label} - {side_name} Foot Slide",
                    [
                        ("GT", gt_slide[:, side].tolist(), "#6aa1ff"),
                        ("Pred", pred_slide[:, side].tolist(), "#ff9b66"),
                    ],
                    source_contacts[:, side].to(torch.int32).tolist(),
                    "m/s",
                )
            )
            charts.append(
                chart_svg(
                    f"{label} - {side_name} Lowest Foot Height",
                    [
                        ("GT", gt_height[:, side].tolist(), "#6aa1ff"),
                        ("Pred", pred_height[:, side].tolist(), "#ff9b66"),
                    ],
                    source_contacts[:, side].to(torch.int32).tolist(),
                    "m",
                )
            )

    title = f"Foot Skating Diagnostic - {npz_path.stem}"
    write_report(resolve_path(args.output_html), title, rows, charts)
    print(f"wrote {resolve_path(args.output_html)}")
    for row in rows:
        print(
            f"{row['label']} slide_excess_p95 pred={float(row['slide_excess_pred_p95']):.4f} gt={float(row['slide_excess_gt_p95']):.4f} "
            f"source_contact_p95 pred={float(row['source_contact_pred_p95']):.4f} gt={float(row['source_contact_gt_p95']):.4f} "
            f"excess_p95={float(row['source_contact_excess_p95']):.4f}"
        )


if __name__ == "__main__":
    main()
