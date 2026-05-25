from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from .bootstrap import PROJECT_ROOT, ensure_paths
    from . import excess_envelope as env
    from . import ik_core as tl
    from . import train_simple_ae_controller as ctl
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    import excess_envelope as env
    import ik_core as tl
    import train_simple_ae_controller as ctl

ensure_paths()


RUNS_DIR = PROJECT_ROOT / "training" / "runs"
DEFAULT_CLIP_QUERY = "M_Neutral_Walk_Loop_F"


def make_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    return device


def latest_controller_checkpoint() -> Path:
    candidates: list[Path] = []
    for pattern in ("*/checkpoints/*_latest.pt", "*/checkpoints/*_last.pt"):
        for path in RUNS_DIR.glob(pattern):
            text = str(path).lower()
            if "controller" in text or "finetune" in text or "ae_output_only" in text:
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No controller checkpoints found under {RUNS_DIR}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0].resolve()


def cyclic_from_path(path: Path) -> bool:
    lower = str(path).replace("\\", "/").lower()
    return "animations_transitions_only_full_trimmed" not in lower and "transitions_only" not in lower


def checkpoint_clip_specs(ckpt: dict) -> list[tuple[Path, bool]]:
    metadata = dict(ckpt.get("metadata", {}))
    paths = metadata.get("npz_paths") or []
    if not paths:
        return ctl.resolve_clip_specs(None, str(ctl.DEFAULT_WALK_F), None)
    specs: list[tuple[Path, bool]] = []
    for raw in paths:
        path = Path(str(raw))
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        specs.append((path, cyclic_from_path(path)))
    return specs


def load_controller(path: Path, device: torch.device) -> tuple[torch.nn.Module, ctl.SimpleClipStore, tl.TrainConfig, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = tl.TrainConfig()
    ctl.apply_config_dict(cfg, ckpt["config"])
    cfg.device = str(device)
    clips = ctl.load_clips(checkpoint_clip_specs(ckpt), cfg)
    store = ctl.SimpleClipStore(clips, cfg, device)
    input_dim, output_dim = tl.make_batch_dims(store.prototype, cfg)
    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, store, cfg, ckpt


def select_clip_id(store: ctl.SimpleClipStore, query: str | None) -> int:
    needle = (query or DEFAULT_CLIP_QUERY).lower()
    for clip_id, clip in enumerate(store.clips):
        name = clip.path.name.lower()
        stem = clip.path.stem.lower()
        if needle in name or needle in stem:
            return int(clip_id)
    return 0


def frame_count_for_clip(clip: tl.MotionClip, cfg: tl.TrainConfig) -> int:
    if clip.cyclic_animation:
        return max(3, int(clip.cyclic_period) + 1)
    max_target = int(clip.T) - ctl.transition_feature_horizon(cfg)
    return max(3, max_target + 1)


@torch.no_grad()
def gt_compact_sequence(
    store: ctl.SimpleClipStore,
    clip_id: int,
    frame_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.arange(frame_count, dtype=torch.long, device=store.device)
    clip_ids = torch.full_like(idx, int(clip_id))
    vec = store.get_target_output(clip_ids, idx)
    root_pos, root_rot, _yaw, _heading = store.root_state(clip_ids, idx)
    return env.ik_foot_toe_state_from_vec(store, root_pos, root_rot, vec)


@torch.no_grad()
def model_compact_sequence(
    model: torch.nn.Module,
    store: ctl.SimpleClipStore,
    clip_id: int,
    frame_count: int,
    gt_pos: torch.Tensor,
    gt_rot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = store.device
    pred_pos = gt_pos.clone()
    pred_rot = gt_rot.clone()
    if frame_count <= 2:
        return pred_pos, pred_rot

    clip_ids = torch.full((1,), int(clip_id), dtype=torch.long, device=device)
    prev_idx = torch.zeros((1,), dtype=torch.long, device=device)
    cur_idx = torch.ones((1,), dtype=torch.long, device=device)
    prev_vec, prev_pelvis, prev_payload = ctl.target_state(store, clip_ids, prev_idx)
    cur_vec, cur_pelvis, cur_payload = ctl.target_state(store, clip_ids, cur_idx)

    for target in range(2, frame_count):
        target_idx = torch.full((1,), int(target), dtype=torch.long, device=device)
        inp = ctl.build_controller_input(
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
        raw = ctl.model_forward(model, inp, cur_vec, store.cfg)
        pred_vec = ctl.clean_output_vector(raw, store)
        root_pos, root_rot = ctl.transition_output_root_state(store, clip_ids, cur_idx)
        foot_pos, foot_rot = env.ik_foot_toe_state_from_vec(store, root_pos, root_rot, pred_vec)
        pred_pos[target] = foot_pos[0]
        pred_rot[target] = foot_rot[0]

        prev_vec, prev_pelvis, prev_payload = cur_vec, cur_pelvis, cur_payload
        cur_vec, cur_pelvis, cur_payload = ctl.advance_transition_state(store, clip_ids, cur_idx, pred_vec)
        cur_idx = target_idx
    return pred_pos, pred_rot


@torch.no_grad()
def envelope_series(
    store: ctl.SimpleClipStore,
    envelope: dict[str, object],
    clip_id: int,
    positions: torch.Tensor,
    rotations: torch.Tensor,
) -> dict[str, torch.Tensor]:
    frame_count = int(positions.shape[0])
    cur_idx = torch.arange(frame_count - 1, dtype=torch.long, device=store.device)
    clip_ids = torch.full_like(cur_idx, int(clip_id))
    linear, angular, linear_bound, angular_bound = env.envelope_values_ik_state_rows(
        store,
        envelope,
        positions[:-1],
        rotations[:-1],
        positions[1:],
        rotations[1:],
        clip_ids,
        cur_idx,
    )
    points = positions[:-1].reshape(frame_count - 1, 2, 2, 3)
    planted = points[..., 1].amin(dim=-1).argmin(dim=-1)
    return {
        "linear": linear,
        "angular": angular,
        "linear_bound": linear_bound,
        "angular_bound": angular_bound,
        "linear_excess": torch.relu(linear - linear_bound),
        "angular_excess": torch.relu(angular - angular_bound),
        "planted": planted,
    }


def tensor_list(tensor: torch.Tensor) -> list[float]:
    return [float(x) for x in tensor.detach().cpu().reshape(-1).tolist()]


def build_html(payload: dict[str, object]) -> str:
    data_json = json.dumps(payload, indent=2)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>IK Foot Envelope Viewer</title>
<style>
  :root {{
    color-scheme: dark;
    font-family: Arial, Helvetica, sans-serif;
    background: #111722;
    color: #e8eef8;
  }}
  body {{ margin: 0; padding: 24px; }}
  h1 {{ margin: 0 0 8px; font-size: 22px; font-weight: 700; }}
  h2 {{ margin: 24px 0 8px; font-size: 16px; font-weight: 700; }}
  .meta {{ color: #aeb9c8; line-height: 1.45; font-size: 13px; max-width: 1200px; }}
  .chart {{
    width: min(1280px, calc(100vw - 48px));
    height: 360px;
    background: #182131;
    border: 1px solid #2a3850;
    border-radius: 6px;
    margin-bottom: 18px;
  }}
  .legend {{ display: flex; gap: 18px; flex-wrap: wrap; margin: 10px 0 18px; color: #d9e2ee; font-size: 13px; }}
  .swatch {{ display: inline-block; width: 18px; height: 3px; vertical-align: middle; margin-right: 6px; }}
  code {{ color: #cdd8e6; }}
</style>
</head>
<body>
<h1>IK Foot Envelope Viewer</h1>
<div class="meta">
  <div><b>Checkpoint:</b> <code id="ckpt"></code></div>
  <div><b>Clip:</b> <code id="clip"></code></div>
  <div><b>Metric:</b> selected lower foot, compact IK foot+toe horizontal movement, foot yaw rate.</div>
  <div><b>Envelope:</b> GT footslide multiplied by the configured margin for this animation/situation.</div>
</div>

<div class="legend">
  <span><span class="swatch" style="background:#7ee787"></span>GT footslide</span>
  <span><span class="swatch" style="background:#f2cc60"></span>envelope</span>
  <span><span class="swatch" style="background:#58a6ff"></span>model footslide</span>
</div>

<h2>Linear Footslide (m/s)</h2>
<svg id="linear" class="chart" role="img"></svg>

<h2>Angular Foot Yaw Slide (deg/s)</h2>
<svg id="angular" class="chart" role="img"></svg>

<h2>Model Envelope Excess</h2>
<svg id="excess" class="chart" role="img"></svg>

<script>
const data = {data_json};

document.getElementById("ckpt").textContent = data.checkpoint;
document.getElementById("clip").textContent = data.clip;

function finiteMax(series) {{
  let m = 0;
  for (const s of series) for (const v of s.values) if (Number.isFinite(v)) m = Math.max(m, v);
  return m;
}}

function drawChart(id, series, yLabel) {{
  const svg = document.getElementById(id);
  const rect = svg.getBoundingClientRect();
  const w = Math.max(640, rect.width || 1200);
  const h = Math.max(300, rect.height || 360);
  svg.setAttribute("viewBox", `0 0 ${{w}} ${{h}}`);
  svg.innerHTML = "";
  const pad = {{l: 58, r: 18, t: 18, b: 42}};
  const frames = data.frames;
  const xMax = Math.max(1, frames.length - 1);
  const yMax = Math.max(1e-6, finiteMax(series) * 1.08);
  const x = i => pad.l + (w - pad.l - pad.r) * (i / xMax);
  const y = v => h - pad.b - (h - pad.t - pad.b) * (v / yMax);

  function line(x1, y1, x2, y2, color, width = 1, opacity = 1) {{
    const el = document.createElementNS("http://www.w3.org/2000/svg", "line");
    el.setAttribute("x1", x1); el.setAttribute("y1", y1);
    el.setAttribute("x2", x2); el.setAttribute("y2", y2);
    el.setAttribute("stroke", color); el.setAttribute("stroke-width", width);
    el.setAttribute("opacity", opacity);
    svg.appendChild(el);
  }}
  function text(value, x0, y0, anchor = "start") {{
    const el = document.createElementNS("http://www.w3.org/2000/svg", "text");
    el.textContent = value;
    el.setAttribute("x", x0); el.setAttribute("y", y0);
    el.setAttribute("fill", "#aeb9c8"); el.setAttribute("font-size", "11");
    el.setAttribute("text-anchor", anchor);
    svg.appendChild(el);
  }}

  for (let i = 0; i <= 5; i++) {{
    const yy = pad.t + (h - pad.t - pad.b) * (i / 5);
    line(pad.l, yy, w - pad.r, yy, "#2a3850", 1, 0.8);
    text((yMax * (1 - i / 5)).toFixed(3), pad.l - 8, yy + 4, "end");
  }}
  for (let i = 0; i <= 8; i++) {{
    const xx = pad.l + (w - pad.l - pad.r) * (i / 8);
    line(xx, pad.t, xx, h - pad.b, "#243247", 1, 0.55);
    text(String(Math.round(xMax * i / 8)), xx, h - 16, "middle");
  }}
  text(yLabel, 10, 16);
  text("frame", w - pad.r, h - 8, "end");

  for (const s of series) {{
    const points = s.values.map((v, i) => `${{x(i).toFixed(2)}},${{y(v).toFixed(2)}}`).join(" ");
    const poly = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    poly.setAttribute("points", points);
    poly.setAttribute("fill", "none");
    poly.setAttribute("stroke", s.color);
    poly.setAttribute("stroke-width", s.width || 2);
    poly.setAttribute("opacity", s.opacity || 1);
    if (s.dash) poly.setAttribute("stroke-dasharray", s.dash);
    svg.appendChild(poly);
  }}
}}

drawChart("linear", [
  {{name:"GT", values:data.linear.gt, color:"#7ee787", width:2}},
  {{name:"Envelope", values:data.linear.envelope, color:"#f2cc60", width:2}},
  {{name:"Model", values:data.linear.model, color:"#58a6ff", width:2}},
], "m/s");

drawChart("angular", [
  {{name:"GT", values:data.angular_deg.gt, color:"#7ee787", width:2}},
  {{name:"Envelope", values:data.angular_deg.envelope, color:"#f2cc60", width:2}},
  {{name:"Model", values:data.angular_deg.model, color:"#58a6ff", width:2}},
], "deg/s");

drawChart("excess", [
  {{name:"Linear excess", values:data.excess.linear, color:"#ff7b72", width:2}},
  {{name:"Angular excess", values:data.excess.angular_deg, color:"#d2a8ff", width:2}},
], "excess: m/s and deg/s");
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render per-frame IK foot envelope diagnostics.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--clip", type=str, default=DEFAULT_CLIP_QUERY)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    checkpoint = (args.checkpoint or latest_controller_checkpoint()).resolve()
    device = make_device()
    model, store, cfg, ckpt = load_controller(checkpoint, device)
    clip_id = select_clip_id(store, args.clip)
    clip = store.clips[clip_id]
    frame_count = frame_count_for_clip(clip, cfg)
    envelope = env.load_or_build_excess_envelope(store)

    gt_pos, gt_rot = gt_compact_sequence(store, clip_id, frame_count)
    model_pos, model_rot = model_compact_sequence(model, store, clip_id, frame_count, gt_pos, gt_rot)
    gt = envelope_series(store, envelope, clip_id, gt_pos, gt_rot)
    model_series = envelope_series(store, envelope, clip_id, model_pos, model_rot)

    metadata = envelope["metadata"]
    assert isinstance(metadata, dict)
    margin = float(metadata.get("margin", 1.05))
    linear_envelope = gt["linear"] * margin
    angular_envelope = gt["angular"] * margin
    linear_excess = torch.relu(model_series["linear"] - linear_envelope)
    angular_excess = torch.relu(model_series["angular"] - angular_envelope)
    payload = {
        "checkpoint": str(checkpoint),
        "clip": str(clip.path),
        "frame_count": int(frame_count),
        "frames": list(range(frame_count - 1)),
        "metadata": {
            "cache_version": int(env.CACHE_VERSION),
            "envelope": dict(metadata),
            "viewer_envelope_margin": margin,
            "viewer_envelope_definition": "gt_footslide_for_frame_times_margin",
            "checkpoint_epoch": int(ckpt.get("epoch", 0)),
            "checkpoint_rollout_k": int(ckpt.get("rollout_k", 0)),
        },
        "linear": {
            "gt": tensor_list(gt["linear"]),
            "envelope": tensor_list(linear_envelope),
            "model": tensor_list(model_series["linear"]),
        },
        "angular_deg": {
            "gt": tensor_list(torch.rad2deg(gt["angular"])),
            "envelope": tensor_list(torch.rad2deg(angular_envelope)),
            "model": tensor_list(torch.rad2deg(model_series["angular"])),
        },
        "excess": {
            "linear": tensor_list(linear_excess),
            "angular_deg": tensor_list(torch.rad2deg(angular_excess)),
        },
        "summary": {
            "viewer_model_linear_over_envelope_mean": float(linear_excess.mean().detach().cpu()),
            "viewer_model_linear_over_envelope_max": float(linear_excess.max().detach().cpu()),
            "viewer_model_angular_over_envelope_radps_mean": float(angular_excess.mean().detach().cpu()),
            "viewer_model_angular_over_envelope_radps_max": float(angular_excess.max().detach().cpu()),
            "gt_linear_excess_mean": float(gt["linear_excess"].mean().detach().cpu()),
            "gt_angular_excess_radps_mean": float(gt["angular_excess"].mean().detach().cpu()),
        },
    }

    out_path = args.output
    if out_path is None:
        out_dir = RUNS_DIR / "model_comparisons"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{checkpoint.parent.parent.name}_{clip.path.stem}_foot_envelope_viewer.html"
    elif not out_path.is_absolute():
        out_path = (PROJECT_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_html(payload), encoding="utf-8")

    print(json.dumps({"output": str(out_path), "summary": payload["summary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
