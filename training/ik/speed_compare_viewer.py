from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch

try:
    from .bootstrap import PROJECT_ROOT, ensure_paths
    from . import ik_core as tl
    from . import train_simple_ae_controller as ctl
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    import ik_core as tl
    import train_simple_ae_controller as ctl

ensure_paths()


DEFAULT_NPZ = PROJECT_ROOT / "ue5" / "animations_omni_only_full" / "npz_final" / "M_Neutral_Walk_Loop_F.npz"
DEFAULT_CKPT = (
    PROJECT_ROOT
    / "training"
    / "runs"
    / "20260524_080401_ik_walkF_fixedK32_rl_datasetmax110"
    / "checkpoints"
    / "20260524_080401_ik_walkF_fixedK32_rl_datasetmax110_last.pt"
)
DEFAULT_LIMITS = PROJECT_ROOT / "training" / "runs" / "cache" / "ik_dataset_speed_limits.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "training"
    / "runs"
    / "model_comparisons"
    / "20260524_080401_ik_walkF_datasetmax110_speed_side_by_side.html"
)


def apply_config_dict(cfg: tl.TrainConfig, values: dict) -> None:
    valid = {field.name for field in fields(tl.TrainConfig)}
    for key, value in values.items():
        if key not in valid:
            continue
        current = getattr(cfg, key)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(cfg, key, value)


def load_controller(checkpoint_path: Path, clip: tl.MotionClip, cfg: tl.TrainConfig, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    input_dim, output_dim = tl.make_batch_dims(clip, cfg)
    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def rollout_vectors(
    checkpoint_path: Path,
    npz_path: Path,
    device: torch.device,
) -> tuple[tl.MotionClip, ctl.SimpleClipStore, torch.Tensor, torch.Tensor, torch.Tensor]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = tl.TrainConfig()
    apply_config_dict(cfg, ckpt.get("config", {}))
    cfg.device = str(device)
    cfg.use_torch_compile = False
    clip = tl.MotionClip(npz_path, cfg, cyclic_animation=True)
    model = load_controller(checkpoint_path, clip, cfg, device)
    store = ctl.SimpleClipStore([clip], cfg, device)
    clip_ids = torch.zeros(clip.T, dtype=torch.long, device=device)
    frame_idx = torch.arange(clip.T, dtype=torch.long, device=device)
    gt_vec = store.get_target_output(clip_ids, frame_idx)
    one_vec = gt_vec.clone()
    ar_vec = gt_vec.clone()

    one_clip = torch.zeros(1, dtype=torch.long, device=device)
    cur_idx = torch.tensor([1], dtype=torch.long, device=device)
    prev_vec, prev_pelvis, prev_payload = ctl.target_state(store, one_clip, cur_idx - 1)
    cur_vec, cur_pelvis, cur_payload = ctl.target_state(store, one_clip, cur_idx)

    for target in range(2, clip.T):
        target_idx = torch.tensor([target], dtype=torch.long, device=device)
        one_cur_idx = torch.tensor([target - 1], dtype=torch.long, device=device)
        one_prev_vec, one_prev_pelvis, one_prev_payload = ctl.target_state(store, one_clip, one_cur_idx - 1)
        one_cur_vec, one_cur_pelvis, one_cur_payload = ctl.target_state(store, one_clip, one_cur_idx)
        one_inp = ctl.build_controller_input(
            store,
            one_clip,
            one_cur_idx,
            one_prev_vec,
            one_cur_vec,
            one_prev_pelvis,
            one_cur_pelvis,
            one_prev_payload,
            one_cur_payload,
        )
        one_raw = ctl.model_forward(model, one_inp, one_cur_vec, cfg)
        one_vec[target] = ctl.clean_output_vector(one_raw, store)[0]

        inp = ctl.build_controller_input(
            store,
            one_clip,
            cur_idx,
            prev_vec,
            cur_vec,
            prev_pelvis,
            cur_pelvis,
            prev_payload,
            cur_payload,
        )
        raw = ctl.model_forward(model, inp, cur_vec, cfg)
        pred_vec = ctl.clean_output_vector(raw, store)
        ar_vec[target] = pred_vec[0]
        prev_vec = cur_vec
        prev_pelvis = cur_pelvis
        prev_payload = cur_payload
        cur_vec, cur_pelvis, cur_payload = ctl.predicted_state_from_vector(pred_vec, store)
        cur_idx = target_idx
    return clip, store, gt_vec, one_vec, ar_vec


def payload_parts(vec: torch.Tensor, part: str) -> torch.Tensor:
    payload_start = int(vec.shape[-1]) - int(tl.IK_PAYLOAD_DIM)
    chunks: list[torch.Tensor] = []
    for spec in tl.IK_PAYLOAD_SLICES:
        sl = spec[part]
        assert isinstance(sl, slice)
        chunks.append(vec[:, payload_start + sl.start : payload_start + sl.stop])
    return torch.stack(chunks, dim=1)


def angular_speed(prev_rot6: torch.Tensor, cur_rot6: torch.Tensor, fps: float) -> torch.Tensor:
    if prev_rot6.numel() == 0:
        return prev_rot6.new_zeros(prev_rot6.shape[:-1])
    prev = tl.rotation_6d_to_matrix(prev_rot6.reshape(-1, 6))
    cur = tl.rotation_6d_to_matrix(cur_rot6.reshape(-1, 6))
    return tl.geodesic_angles(cur, prev).reshape(prev_rot6.shape[:-1]) * float(fps) * 180.0 / torch.pi


def speed_series(current: torch.Tensor, previous: torch.Tensor, fps: float) -> dict[str, torch.Tensor]:
    ee_cur = payload_parts(current, "pos")
    ee_prev = payload_parts(previous, "pos")
    ee_linear = torch.linalg.vector_norm(ee_cur - ee_prev, dim=-1).amax(dim=-1) * float(fps)

    ee_rot_cur = payload_parts(current, "rot6")
    ee_rot_prev = payload_parts(previous, "rot6")
    ee_angular = angular_speed(ee_rot_prev, ee_rot_cur, fps).amax(dim=-1)

    pelvis_linear = torch.linalg.vector_norm(current[:, :3] - previous[:, :3], dim=-1) * float(fps)
    pelvis_angular = angular_speed(previous[:, 3:9], current[:, 3:9], fps)

    payload_start = int(current.shape[-1]) - int(tl.IK_PAYLOAD_DIM)
    core_cur = current[:, 9:payload_start].reshape(current.shape[0], -1, 6)
    core_prev = previous[:, 9:payload_start].reshape(previous.shape[0], -1, 6)
    core_angular = angular_speed(core_prev, core_cur, fps).amax(dim=-1)

    zeros = current.new_zeros(1)
    return {
        "ee_linear_mps": torch.cat((zeros, ee_linear[1:])),
        "ee_angular_deg_s": torch.cat((zeros, ee_angular[1:])),
        "pelvis_linear_mps": torch.cat((zeros, pelvis_linear[1:])),
        "pelvis_angular_deg_s": torch.cat((zeros, pelvis_angular[1:])),
        "core_angular_deg_s": torch.cat((zeros, core_angular[1:])),
    }


def shifted_previous(vec: torch.Tensor) -> torch.Tensor:
    return torch.cat((vec[:1], vec[:-1]), dim=0)


def global_positions_from_vectors(
    clip: tl.MotionClip,
    store: ctl.SimpleClipStore,
    vec: torch.Tensor,
) -> torch.Tensor:
    device = vec.device
    clip_ids = torch.zeros(vec.shape[0], dtype=torch.long, device=device)
    idx = torch.arange(vec.shape[0], dtype=torch.long, device=device)
    root_pos, root_rot, _yaw, _heading = store.root_state(clip_ids, idx)
    pose, _raw = tl.output_to_pose(vec, clip)
    pos, _rot, _canon = tl.fk_from_pose(clip, root_pos, root_rot, pose, device)
    return pos


def linear_speed_from_positions(pos: torch.Tensor, fps: float) -> torch.Tensor:
    zeros = pos.new_zeros(1)
    speed = torch.linalg.vector_norm(pos[1:] - pos[:-1], dim=-1) * float(fps)
    return torch.cat((zeros, speed), dim=0)


def summarize(series: torch.Tensor, limit: float, gt_max: float) -> dict[str, float | int]:
    values = series.detach().cpu().float()
    max_value = float(values.max().item())
    return {
        "mean": float(values.mean().item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "max": max_value,
        "max_frame": int(values.argmax().item()),
        "ratio_to_gt_max": float(max_value / max(gt_max, 1e-8)),
        "ratio_to_full_limit": float(max_value / max(float(limit), 1e-8)),
        "frames_over_full_limit": int((values > float(limit)).sum().item()),
    }


def summarize_distance(series: torch.Tensor) -> dict[str, float | int]:
    values = series.detach().cpu().float()
    return {
        "mean": float(values.mean().item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "max": float(values.max().item()),
        "max_frame": int(values.argmax().item()),
        "ratio_to_gt_max": 0.0,
        "ratio_to_full_limit": 0.0,
        "frames_over_full_limit": 0,
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IK Speed Side By Side</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101216;
      --panel: #1a1d22;
      --panel2: #23272f;
      --text: #edf1f7;
      --muted: #9da6b5;
      --line: #313846;
      --grid: rgba(255,255,255,0.08);
      --gt: #72a7ff;
      --one: #e6df83;
      --ar: #ff9b6a;
      --limit: #55d6a7;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; background: var(--bg); }
    body { font: 13px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--text); }
    #app { width: 100vw; height: 100vh; display: grid; grid-template-rows: auto 1fr auto; }
    header { padding: 10px 12px; background: var(--panel); border-bottom: 1px solid var(--line); display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    select, button { height: 32px; border-radius: 6px; border: 1px solid var(--line); background: var(--panel2); color: var(--text); font: inherit; padding: 0 9px; }
    label { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); }
    input { accent-color: var(--limit); }
    #plotWrap { position: relative; min-height: 0; }
    canvas { display: block; width: 100%; height: 100%; background: #0f1217; }
    #summary { padding: 8px 12px; border-top: 1px solid var(--line); background: var(--panel); overflow-x: auto; }
    table { border-collapse: collapse; min-width: 960px; width: 100%; }
    th, td { padding: 5px 7px; border-bottom: 1px solid rgba(255,255,255,0.07); text-align: right; font-variant-numeric: tabular-nums; }
    th:first-child, td:first-child { text-align: left; }
    .pill { padding: 5px 7px; border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; color: var(--muted); }
    .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; }
  </style>
</head>
<body>
<div id="app">
  <header>
    <strong id="title"></strong>
    <select id="metric"></select>
    <label><input id="showGt" type="checkbox" checked> <span class="dot" style="background:var(--gt)"></span>GT</label>
    <label><input id="showOne" type="checkbox" checked> <span class="dot" style="background:var(--one)"></span>one-step</label>
    <label><input id="showAr" type="checkbox" checked> <span class="dot" style="background:var(--ar)"></span>autoregressive</label>
    <label><input id="showLimit" type="checkbox" checked> <span class="dot" style="background:var(--limit)"></span>full dataset 1.10x max</label>
    <span class="pill" id="readout"></span>
  </header>
  <div id="plotWrap"><canvas id="plot"></canvas></div>
  <div id="summary"><table id="table"></table></div>
</div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const data = JSON.parse(document.getElementById("payload").textContent);
const metricSelect = document.getElementById("metric");
const canvas = document.getElementById("plot");
const ctx = canvas.getContext("2d");
const checks = {
  gt: document.getElementById("showGt"),
  one: document.getElementById("showOne"),
  ar: document.getElementById("showAr"),
  limit: document.getElementById("showLimit"),
};
document.getElementById("title").textContent = data.title;
for (const key of data.metric_order) {
  const opt = document.createElement("option");
  opt.value = key;
  opt.textContent = data.metrics[key].label;
  metricSelect.appendChild(opt);
}
function resize() {
  const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(r.width * dpr));
  canvas.height = Math.max(1, Math.floor(r.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
function line(values, color, yMin, yMax, dashed=false) {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const pad = {l: 54, r: 18, t: 18, b: 32};
  const sx = (w - pad.l - pad.r) / Math.max(1, values.length - 1);
  const sy = (h - pad.t - pad.b) / Math.max(1e-8, yMax - yMin);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash(dashed ? [7, 5] : []);
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = pad.l + i * sx;
    const y = h - pad.b - (v - yMin) * sy;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.setLineDash([]);
}
function draw() {
  resize();
  const key = metricSelect.value;
  const m = data.metrics[key];
  const visible = [];
  if (checks.gt.checked) visible.push(...m.gt);
  if (checks.one.checked) visible.push(...m.one_step);
  if (checks.ar.checked) visible.push(...m.autoregressive);
  if (checks.limit.checked && m.limit > 0) visible.push(0, m.limit);
  let yMax = Math.max(...visible, 1e-6) * 1.08;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0f1217";
  ctx.fillRect(0, 0, w, h);
  const pad = {l: 54, r: 18, t: 18, b: 32};
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#9da6b5";
  ctx.font = "12px system-ui";
  for (let i = 0; i <= 5; i++) {
    const y = pad.t + (h - pad.t - pad.b) * i / 5;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    const v = yMax * (1 - i / 5);
    ctx.fillText(v.toFixed(v >= 10 ? 0 : 2), 8, y + 4);
  }
  for (let i = 0; i <= 6; i++) {
    const x = pad.l + (w - pad.l - pad.r) * i / 6;
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
    ctx.fillText(String(Math.round((m.gt.length - 1) * i / 6)), x - 8, h - 9);
  }
  if (checks.limit.checked && m.limit > 0) line(new Array(m.gt.length).fill(m.limit), "var(--limit)", 0, yMax, true);
  if (checks.gt.checked) line(m.gt, "var(--gt)", 0, yMax);
  if (checks.one.checked) line(m.one_step, "var(--one)", 0, yMax);
  if (checks.ar.checked) line(m.autoregressive, "var(--ar)", 0, yMax);
  document.getElementById("readout").textContent =
    `${m.limit > 0 ? `limit ${m.limit.toFixed(3)} ${m.unit} | ` : ""}AR max ${data.summary[key].autoregressive.max.toFixed(3)} ${m.unit} (${data.summary[key].autoregressive.ratio_to_gt_max.toFixed(2)}x WalkF GT max)`;
}
function renderTable() {
  const rows = [`<tr><th>metric</th><th>series</th><th>mean</th><th>p95</th><th>max</th><th>max frame</th><th>max / WalkF GT max</th><th>max / full limit</th><th>frames over limit</th></tr>`];
  for (const key of data.metric_order) {
    for (const name of ["gt", "one_step", "autoregressive"]) {
      const s = data.summary[key][name];
      rows.push(`<tr><td>${data.metrics[key].label}</td><td>${name}</td><td>${s.mean.toFixed(3)}</td><td>${s.p95.toFixed(3)}</td><td>${s.max.toFixed(3)}</td><td>${s.max_frame}</td><td>${s.ratio_to_gt_max.toFixed(2)}</td><td>${s.ratio_to_full_limit.toFixed(2)}</td><td>${s.frames_over_full_limit}</td></tr>`);
    }
  }
  document.getElementById("table").innerHTML = rows.join("");
}
window.addEventListener("resize", draw);
metricSelect.addEventListener("change", draw);
for (const c of Object.values(checks)) c.addEventListener("change", draw);
renderTable();
draw();
</script>
</body>
</html>
"""


def build_payload(checkpoint: Path, npz: Path, limits_path: Path, device: torch.device) -> dict[str, object]:
    clip, store, gt_vec, one_vec, ar_vec = rollout_vectors(checkpoint, npz, device)
    fps = float(clip.fps)
    gt_prev = shifted_previous(gt_vec)
    one_prev = shifted_previous(gt_vec)
    ar_prev = shifted_previous(ar_vec)
    gt = speed_series(gt_vec, gt_prev, fps)
    one = speed_series(one_vec, one_prev, fps)
    ar = speed_series(ar_vec, ar_prev, fps)
    gt_global = global_positions_from_vectors(clip, store, gt_vec)
    one_global = global_positions_from_vectors(clip, store, one_vec)
    ar_global = global_positions_from_vectors(clip, store, ar_vec)
    pelvis = int(clip.pelvis)
    gt_global_pelvis = gt_global[:, pelvis]
    one_global_pelvis = one_global[:, pelvis]
    ar_global_pelvis = ar_global[:, pelvis]
    gt["pelvis_global_linear_mps"] = linear_speed_from_positions(gt_global_pelvis, fps)
    one["pelvis_global_linear_mps"] = linear_speed_from_positions(one_global_pelvis, fps)
    ar["pelvis_global_linear_mps"] = linear_speed_from_positions(ar_global_pelvis, fps)
    gt["pelvis_global_deviation_m"] = torch.zeros_like(gt["pelvis_global_linear_mps"])
    one["pelvis_global_deviation_m"] = torch.linalg.vector_norm(one_global_pelvis - gt_global_pelvis, dim=-1)
    ar["pelvis_global_deviation_m"] = torch.linalg.vector_norm(ar_global_pelvis - gt_global_pelvis, dim=-1)
    limits_data = json.loads(limits_path.read_text(encoding="utf-8"))
    limit_map = {
        "ee_linear_mps": limits_data["limits"]["end_effector_velocity_mps"],
        "ee_angular_deg_s": limits_data["limits"]["end_effector_angular_velocity_deg_s"],
        "pelvis_linear_mps": limits_data["limits"]["pelvis_velocity_mps"],
        "pelvis_angular_deg_s": limits_data["limits"]["pelvis_angular_velocity_deg_s"],
        "core_angular_deg_s": limits_data["limits"]["core_angular_velocity_deg_s"],
        "pelvis_global_linear_mps": 0.0,
        "pelvis_global_deviation_m": 0.0,
    }
    labels = {
        "ee_linear_mps": ("EE linear", "m/s"),
        "ee_angular_deg_s": ("EE angular", "deg/s"),
        "pelvis_linear_mps": ("Pelvis linear", "m/s"),
        "pelvis_angular_deg_s": ("Pelvis angular", "deg/s"),
        "core_angular_deg_s": ("Core angular", "deg/s"),
        "pelvis_global_linear_mps": ("Pelvis global linear", "m/s"),
        "pelvis_global_deviation_m": ("Pelvis global deviation", "m"),
    }
    metrics: dict[str, object] = {}
    summary: dict[str, object] = {}
    for key, (label, unit) in labels.items():
        limit = float(limit_map[key])
        gt_max = float(gt[key].max().detach().cpu())
        metrics[key] = {
            "label": label,
            "unit": unit,
            "limit": limit,
            "gt": [float(v) for v in gt[key].detach().cpu().tolist()],
            "one_step": [float(v) for v in one[key].detach().cpu().tolist()],
            "autoregressive": [float(v) for v in ar[key].detach().cpu().tolist()],
        }
        if key.endswith("_deviation_m"):
            summary[key] = {
                "gt": summarize_distance(gt[key]),
                "one_step": summarize_distance(one[key]),
                "autoregressive": summarize_distance(ar[key]),
            }
        else:
            ratio_limit = limit if limit > 0 else float("inf")
            summary[key] = {
                "gt": summarize(gt[key], ratio_limit, gt_max),
                "one_step": summarize(one[key], ratio_limit, gt_max),
                "autoregressive": summarize(ar[key], ratio_limit, gt_max),
            }
    return {
        "title": f"{npz.stem} speed side by side / {checkpoint.parent.parent.name}",
        "fps": fps,
        "frame_count": int(clip.T),
        "metric_order": list(labels.keys()),
        "metrics": metrics,
        "summary": summary,
        "limits_source": limits_data,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GT and model IK speed quantities side by side.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--npz", default=str(DEFAULT_NPZ))
    parser.add_argument("--limits", default=str(DEFAULT_LIMITS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    payload = build_payload(Path(args.checkpoint), Path(args.npz), Path(args.limits), torch.device(args.device))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":"))), encoding="utf-8")
    print(out)
    for key in payload["metric_order"]:
        m = payload["summary"][key]
        print(
            key,
            "gt_max",
            f"{m['gt']['max']:.6g}",
            "one_max",
            f"{m['one_step']['max']:.6g}",
            "ar_max",
            f"{m['autoregressive']['max']:.6g}",
            "ar/gt",
            f"{m['autoregressive']['ratio_to_gt_max']:.3g}",
        )


if __name__ == "__main__":
    main()
