from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import torch

try:
    from .bootstrap import PROJECT_ROOT, ensure_paths
    from . import ik_core as ik
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    import ik_core as ik

ensure_paths()


DEFAULT_NPZ = PROJECT_ROOT / "ue5" / "animations_omni_only_full" / "npz_final" / "M_Neutral_Stand_Idle_Loop.npz"
DEFAULT_OUT = PROJECT_ROOT / "training" / "runs" / "model_comparisons" / "ee_theta_viewer.html"


def _tolist(t: torch.Tensor, ndigits: int = 6) -> list:
    return torch.round(t.detach().cpu().float() * (10**ndigits)).div(10**ndigits).tolist()


def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)


def _project(v: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    n = _normalize(n)
    return _normalize(v - n * (v * n).sum(dim=-1, keepdim=True))


def _signed_angle(a: torch.Tensor, b: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    a = _normalize(a)
    b = _normalize(b)
    axis = _normalize(axis)
    sin_v = (torch.cross(a, b, dim=-1) * axis).sum(dim=-1)
    cos_v = (a * b).sum(dim=-1)
    return torch.atan2(sin_v, cos_v)


def build_payload(npz: Path, frame: int) -> dict:
    cfg = ik.TrainConfig()
    cfg.pose_representation = ik.IK_POSE_REPRESENTATION
    cfg.use_torch_compile = False
    clip = ik.MotionClip(npz, cfg)
    frame = max(0, min(int(frame), int(clip.T) - 1))
    root_inv = clip.root_rot.transpose(-1, -2)
    root_rel_rot = clip.global_rot @ root_inv[:, None]

    ee_local_pole_frames: list[list[list[float]]] = []
    ee_local_pole_mean: list[list[float]] = []
    theta_stats: list[dict[str, float]] = []
    limb_data: list[dict] = []
    root_up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).reshape(1, 1, 3).expand(clip.T, 1, 3)

    for limb_i, spec in enumerate(clip.ik_limb_specs):
        start = int(spec["start"])
        mid = int(spec["mid"])
        end = int(spec["end"])
        toe = spec.get("toe")
        base = clip.root_relative_pos[:, start]
        mid_pos = clip.root_relative_pos[:, mid]
        end_pos = clip.root_relative_pos[:, end]
        axis = _normalize(end_pos - base)
        actual_pole = _project(mid_pos - base, axis)
        ee_rot = root_rel_rot[:, end]
        ee_local = torch.matmul(actual_pole.unsqueeze(1), ee_rot.transpose(-1, -2)).squeeze(1)
        ee_local = _normalize(ee_local)
        mean_local = _normalize(clip.ik_ee_pole_ref[limb_i].reshape(1, 3))[0]
        mean_ref = _project(torch.matmul(mean_local.reshape(1, 1, 3).expand(clip.T, 1, 3), ee_rot).squeeze(1), axis)
        theta = torch.rad2deg(_signed_angle(mean_ref, actual_pole, axis))
        theta_abs = theta.abs()
        sorted_abs = torch.sort(theta_abs).values
        p95_idx = min(sorted_abs.numel() - 1, max(0, int(round(0.95 * (sorted_abs.numel() - 1)))))
        toe_offset = clip.ik_toe_offsets[limb_i]
        if toe is not None and torch.linalg.norm(toe_offset) > 1e-6:
            forward_local = _normalize(toe_offset.reshape(1, 3))[0]
        else:
            distal_axis = _normalize(end_pos - mid_pos)
            distal_local = torch.matmul(distal_axis.unsqueeze(1), ee_rot.transpose(-1, -2)).squeeze(1)
            forward_local = _normalize(distal_local.mean(dim=0, keepdim=True))[0]
        up_local_frames = torch.matmul(root_up, ee_rot.transpose(-1, -2)).squeeze(1)
        up_local = _normalize(up_local_frames.mean(dim=0, keepdim=True))[0]
        up_local = up_local - forward_local * (up_local * forward_local).sum()
        if float(torch.linalg.norm(up_local)) < 1e-5:
            fallback = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
            if abs(float((fallback * forward_local).sum())) > 0.9:
                fallback = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
            up_local = fallback - forward_local * (fallback * forward_local).sum()
        up_local = _normalize(up_local.reshape(1, 3))[0]
        theta_stats.append(
            {
                "min": float(theta.min()),
                "max": float(theta.max()),
                "abs95": float(sorted_abs[p95_idx]),
                "absMax": float(theta_abs.max()),
            }
        )
        ee_local_pole_frames.append(_tolist(ee_local))
        ee_local_pole_mean.append(_tolist(mean_local))
        pole_alpha_deg = math.degrees(float(clip.ik_pole_alpha[limb_i]))
        limb_data.append(
            {
                "side": str(spec["side"]),
                "kind": str(spec["kind"]),
                "start": start,
                "mid": mid,
                "end": end,
                "toe": None if toe is None else int(toe),
                "name": f"{spec['side']} {spec['kind']} ({clip.body_names[start]} -> {clip.body_names[end]})",
                "lengths": _tolist(clip.ik_limb_lengths[limb_i]),
                "restAxis": _tolist(clip.ik_rest_axis[limb_i]),
                "restPole": _tolist(clip.ik_rest_pole[limb_i]),
                "toeOffset": _tolist(clip.ik_toe_offsets[limb_i]),
                "eeForwardLocal": _tolist(forward_local),
                "eeUpLocal": _tolist(up_local),
                "poleAlphaDeg": pole_alpha_deg,
            }
        )

    return {
        "npz": str(npz),
        "frame": frame,
        "fps": float(clip.fps),
        "bodyNames": list(clip.body_names),
        "parents": [int(x) for x in clip.parents_body_list],
        "limbs": limb_data,
        "frames": _tolist(clip.root_relative_pos),
        "rootRelRot": _tolist(root_rel_rot),
        "eeLocalPoleFrames": ee_local_pole_frames,
        "eeLocalPoleMean": ee_local_pole_mean,
        "oldPoleFloat": _tolist(clip.ik_pole_float),
        "thetaStatsDeg": theta_stats,
        "poleAlphaDeg": [float(limb["poleAlphaDeg"]) for limb in limb_data],
        "defaultAlphaDeg": float(limb_data[2]["poleAlphaDeg"]) if len(limb_data) > 2 else 10.0,
    }


HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>EE Theta IK Sandbox</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101316;
      --panel: #171d22;
      --panel2: #202832;
      --ink: #e9f0ef;
      --muted: #8fa09e;
      --line: #31414c;
      --new: #53f0c4;
      --old: #ff8a67;
      --auth: #70808b;
      --accent: #f0d36b;
    }
    * { box-sizing: border-box; }
    html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; background: var(--bg); color: var(--ink); font-family: "Aptos", "Segoe UI", sans-serif; }
    body { display: grid; grid-template-columns: 360px 1fr; }
    aside { background: linear-gradient(180deg, #151b20, #101316); border-right: 1px solid var(--line); padding: 16px; overflow: auto; }
    canvas { display: block; width: 100%; height: 100%; background: radial-gradient(circle at 50% 42%, #19232b 0, #0d1013 70%); cursor: grab; }
    canvas.dragging { cursor: grabbing; }
    h1 { font-size: 18px; margin: 0 0 8px; letter-spacing: 0.02em; }
    p { color: var(--muted); line-height: 1.35; margin: 8px 0 14px; }
    .group { border: 1px solid var(--line); background: rgba(255,255,255,0.025); border-radius: 12px; padding: 12px; margin: 10px 0; }
    label { display: block; color: var(--muted); font-size: 12px; margin: 10px 0 5px; }
    select, input[type="number"] { width: 100%; background: var(--panel2); color: var(--ink); border: 1px solid var(--line); border-radius: 8px; padding: 7px 8px; }
    input[type="range"] { width: 100%; accent-color: var(--new); }
    button { background: #22313a; color: var(--ink); border: 1px solid #3b4d58; border-radius: 9px; padding: 7px 9px; cursor: pointer; }
    button:hover { background: #2a3b45; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .row3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
    .value { float: right; color: var(--ink); font-variant-numeric: tabular-nums; }
    .legend { display: flex; flex-wrap: wrap; gap: 8px; font-size: 12px; color: var(--muted); }
    .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }
    .stat { font-family: "Cascadia Mono", Consolas, monospace; font-size: 12px; color: #b7c7c5; white-space: pre-wrap; }
    .hint { color: #9eaca9; font-size: 12px; }
  </style>
</head>
<body>
  <aside>
    <h1>EE Theta IK Sandbox</h1>
    <p>Drag an end-effector dot in the viewport. Use the theta slider to rotate the knee/elbow around the selected reference pole. The default neutral pose starts with each IK limb fully straight.</p>
    <div class="legend">
      <span><i class="swatch" style="background:var(--new)"></i>new EE-frame theta</span>
      <span><i class="swatch" style="background:var(--old)"></i>old rest theta</span>
      <span><i class="swatch" style="background:var(--auth)"></i>authored pose</span>
    </div>

    <div class="group">
      <label>Limb</label>
      <select id="limb"></select>
      <label>Frame <span id="frameValue" class="value"></span></label>
      <input id="frame" type="range" min="0" max="1" value="0" step="1">
      <div class="row3">
        <button id="resetPose">Clip target</button>
        <button id="neutralPose">Straight neutral</button>
        <button id="thetaAuthored">Authored theta</button>
      </div>
    </div>

    <div class="group">
      <label>Reference</label>
      <select id="reference">
        <option value="mean">new: stored EE-local ref</option>
        <option value="frame">new: selected-frame EE-local</option>
        <option value="old">old: rest-axis transport</option>
      </select>
      <label>Theta deg <span id="thetaValue" class="value"></span></label>
      <input id="theta" type="range" min="-60" max="60" value="0" step="0.25">
      <label>Alpha / clamp deg <span id="alphaValue" class="value"></span></label>
      <input id="alpha" type="range" min="5" max="90" value="40" step="1">
      <label><input id="showOld" type="checkbox" checked> overlay old solver with same theta</label>
      <label><input id="showColliders" type="checkbox" checked> show EE orientation colliders</label>
    </div>

    <div class="group">
      <label>End-effector position in root space</label>
      <div class="row3">
        <div><label>X <span id="xValue" class="value"></span></label><input id="targetX" type="range" min="-1" max="1" value="0" step="0.002"></div>
        <div><label>Y <span id="yValue" class="value"></span></label><input id="targetY" type="range" min="-1" max="1" value="0" step="0.002"></div>
        <div><label>Z <span id="zValue" class="value"></span></label><input id="targetZ" type="range" min="-1" max="1" value="0" step="0.002"></div>
      </div>
    </div>

    <div class="group">
      <label>End-effector rotation delta (world/root axes)</label>
      <label>World Z <span id="yawValue" class="value"></span></label>
      <input id="yaw" type="range" min="-90" max="90" value="0" step="1">
      <label>World X <span id="pitchValue" class="value"></span></label>
      <input id="pitch" type="range" min="-90" max="90" value="0" step="1">
      <label>World Y <span id="rollValue" class="value"></span></label>
      <input id="roll" type="range" min="-90" max="90" value="0" step="1">
      <div class="row3" style="margin-top:10px">
        <button id="yawMinus">Z -15</button>
        <button id="pitchMinus">X -15</button>
        <button id="rollMinus">Y -15</button>
      </div>
      <div class="row3" style="margin-top:8px">
        <button id="yawPlus">Z +15</button>
        <button id="pitchPlus">X +15</button>
        <button id="rollPlus">Y +15</button>
      </div>
      <div class="row" style="margin-top:8px">
        <button id="resetRotation">Reset rotation</button>
        <button id="zeroRotationTheta">Zero rot + theta</button>
      </div>
      <p class="hint">Hotkeys: Q/E world Z, W/S world X, A/D world Y. EE rotation feeds the green knee/elbow reference immediately.</p>
    </div>

    <div class="group">
      <label>Camera</label>
      <div class="row">
        <button id="viewFront">Front</button>
        <button id="viewSide">Side</button>
      </div>
      <div class="row" style="margin-top:8px">
        <button id="viewTop">Top</button>
        <button id="zeroTheta">Zero theta</button>
      </div>
      <p class="hint">Mouse wheel zoom. Right/middle drag pans. Left-drag on an EE handle moves it in the camera plane.</p>
    </div>

    <div class="group">
      <label>Stats</label>
      <div id="stats" class="stat"></div>
    </div>
  </aside>
  <canvas id="canvas"></canvas>

<script>
const DATA = __DATA__;
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const $ = id => document.getElementById(id);
const ui = {
  limb: $("limb"), frame: $("frame"), reference: $("reference"),
  theta: $("theta"), alpha: $("alpha"), showOld: $("showOld"), showColliders: $("showColliders"),
  targetX: $("targetX"), targetY: $("targetY"), targetZ: $("targetZ"),
  yaw: $("yaw"), pitch: $("pitch"), roll: $("roll"),
};

const state = {
  frame: DATA.frame || 0,
  limb: 2,
  thetaDeg: 0,
  alphaDeg: (DATA.poleAlphaDeg && DATA.poleAlphaDeg[2]) || DATA.defaultAlphaDeg || 10,
  refMode: "mean",
  targets: [],
  rotDeltas: [],
  yaw: -34,
  pitch: 18,
  zoom: 1.0,
  pan: [0, 0],
  dragging: null,
  lastMouse: [0, 0],
};

function defaultAlphaForLimb(limbIdx) {
  return (DATA.poleAlphaDeg && DATA.poleAlphaDeg[limbIdx]) || DATA.defaultAlphaDeg || 10;
}

function v(x=0,y=0,z=0){ return [x,y,z]; }
function add(a,b){ return [a[0]+b[0],a[1]+b[1],a[2]+b[2]]; }
function sub(a,b){ return [a[0]-b[0],a[1]-b[1],a[2]-b[2]]; }
function mul(a,s){ return [a[0]*s,a[1]*s,a[2]*s]; }
function dot(a,b){ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
function cross(a,b){ return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function len(a){ return Math.sqrt(Math.max(0, dot(a,a))); }
function norm(a){ const l=len(a); return l > 1e-8 ? mul(a,1/l) : [0,0,0]; }
function clamp(x,a,b){ return Math.max(a, Math.min(b, x)); }
function project(vv,n){ const nn=norm(n); return norm(sub(vv, mul(nn, dot(vv, nn)))); }
function rotateAroundAxis(vec, axis, angleRad) {
  const ax = norm(axis), c = Math.cos(angleRad), s = Math.sin(angleRad);
  return add(add(mul(vec,c), mul(cross(ax, vec), s)), mul(ax, dot(ax, vec) * (1-c)));
}
function rowMul(vec, m) {
  return [
    vec[0]*m[0][0] + vec[1]*m[1][0] + vec[2]*m[2][0],
    vec[0]*m[0][1] + vec[1]*m[1][1] + vec[2]*m[2][1],
    vec[0]*m[0][2] + vec[1]*m[1][2] + vec[2]*m[2][2],
  ];
}
function matMul(a,b) {
  const out = [[0,0,0],[0,0,0],[0,0,0]];
  for (let r=0;r<3;r++) for (let c=0;c<3;c++) out[r][c] = a[r][0]*b[0][c] + a[r][1]*b[1][c] + a[r][2]*b[2][c];
  return out;
}
function axisAngleRow(axis, deg) {
  const a = norm(axis), rad = deg * Math.PI / 180;
  return [
    rotateAroundAxis([1,0,0], a, rad),
    rotateAroundAxis([0,1,0], a, rad),
    rotateAroundAxis([0,0,1], a, rad),
  ];
}
function rotateRowsWorld(rows, axis, deg) {
  const rad = deg * Math.PI / 180;
  return rows.map(row => rotateAroundAxis(row, axis, rad));
}
function eeRotFor(limbIdx) {
  let rot = DATA.rootRelRot[state.frame][DATA.limbs[limbIdx].end].map(row => row.slice());
  const d = state.rotDeltas[limbIdx] || {yaw:0,pitch:0,roll:0};
  // The controls are root/world-space deltas, not local EE Euler rotations.
  if (Math.abs(d.pitch || 0) > 1e-8) rot = rotateRowsWorld(rot, [1,0,0], d.pitch);
  if (Math.abs(d.roll || 0) > 1e-8) rot = rotateRowsWorld(rot, [0,1,0], d.roll);
  if (Math.abs(d.yaw || 0) > 1e-8) rot = rotateRowsWorld(rot, [0,0,1], d.yaw);
  return rot;
}
function stablePerp(axis) {
  const a = norm(axis);
  const ref = Math.abs(a[2]) < 0.8 ? [0,0,1] : [1,0,0];
  return norm(cross(a, ref));
}
function swingOnlyTransport(restAxis, currentAxis, restPole) {
  const a = norm(restAxis), b = norm(currentAxis), c = clamp(dot(a,b), -1, 1);
  if (c > 0.999999) return restPole;
  if (c < -0.999999) return rotateAroundAxis(restPole, stablePerp(a), Math.PI);
  return rotateAroundAxis(restPole, norm(cross(a,b)), Math.acos(c));
}
function signedAngleDeg(a,b,axis) {
  const aa = norm(a), bb = norm(b), ax = norm(axis);
  return Math.atan2(dot(cross(aa,bb), ax), dot(aa,bb)) * 180 / Math.PI;
}
function referencePole(limbIdx, target, axis, eeRot, mode) {
  const limb = DATA.limbs[limbIdx];
  if (mode === "old") return project(swingOnlyTransport(limb.restAxis, axis, limb.restPole), axis);
  const local = mode === "frame" ? DATA.eeLocalPoleFrames[limbIdx][state.frame] : DATA.eeLocalPoleMean[limbIdx];
  let ref = project(rowMul(local, eeRot), axis);
  if (len(ref) < 1e-6) ref = project(swingOnlyTransport(limb.restAxis, axis, limb.restPole), axis);
  return ref;
}
function solve(limbIdx, thetaDeg, mode) {
  const limb = DATA.limbs[limbIdx], frame = DATA.frames[state.frame];
  const base = frame[limb.start];
  const targetRaw = state.targets[limbIdx];
  let delta = sub(targetRaw, base);
  let dRaw = len(delta);
  let axis = dRaw > 1e-8 ? mul(delta, 1/dRaw) : norm(limb.restAxis);
  const l1 = limb.lengths[0], l2 = limb.lengths[1];
  const minD = Math.abs(l1-l2) + 1e-5;
  const maxD = l1+l2 - 1e-5;
  const d = clamp(Math.max(dRaw, 1e-8), minD, maxD);
  const end = add(base, mul(axis, d));
  const eeRot = eeRotFor(limbIdx);
  const ref = referencePole(limbIdx, end, axis, eeRot, mode);
  const pole = rotateAroundAxis(ref, axis, thetaDeg * Math.PI / 180);
  const a = (l1*l1 - l2*l2 + d*d) / (2*d);
  const h = Math.sqrt(Math.max(0, l1*l1 - a*a));
  const mid = add(add(base, mul(axis, a)), mul(pole, h));
  let toe = null;
  if (limb.toe !== null && limb.toe !== undefined) toe = add(end, rowMul(limb.toeOffset, eeRot));
  return {base, mid, end, toe, axis, pole, ref, eeRot, clamped: Math.abs(d-dRaw) > 1e-5};
}
    function cameraBasis() {
      const yaw = state.yaw * Math.PI/180, pitch = state.pitch * Math.PI/180;
      const forward = norm([Math.sin(yaw)*Math.cos(pitch), -Math.cos(yaw)*Math.cos(pitch), Math.sin(pitch)]);
      const right = norm([Math.cos(yaw), Math.sin(yaw), 0]);
      // Keep screen-up aligned with positive Z for the z-up animation data.
      const up = norm(cross(forward, right));
      return {right, up, forward};
    }
function sceneExtent() {
  const pts = DATA.frames[state.frame];
  let maxR = 0;
  for (const p of pts) maxR = Math.max(maxR, len(p));
  return Math.max(0.8, maxR * 1.3);
}
function projectPoint(p) {
  const {right, up, forward} = cameraBasis();
  const s = Math.min(canvas.clientWidth, canvas.clientHeight) * 0.44 * state.zoom / sceneExtent();
  return {
    x: canvas.clientWidth*0.52 + state.pan[0] + dot(p,right)*s,
    y: canvas.clientHeight*0.55 + state.pan[1] - dot(p,up)*s,
    z: dot(p,forward),
    scale: s,
  };
}
function unprojectAtDepth(x,y,depth) {
  const {right, up, forward} = cameraBasis();
  const s = Math.min(canvas.clientWidth, canvas.clientHeight) * 0.44 * state.zoom / sceneExtent();
  const u = (x - canvas.clientWidth*0.52 - state.pan[0]) / s;
  const vv = -(y - canvas.clientHeight*0.55 - state.pan[1]) / s;
  return add(add(mul(right,u), mul(up,vv)), mul(forward, depth));
}
function drawLine(a,b,color,width=2,dash=[]) {
  const pa = projectPoint(a), pb = projectPoint(b);
  ctx.save(); ctx.setLineDash(dash); ctx.strokeStyle=color; ctx.lineWidth=width;
  ctx.beginPath(); ctx.moveTo(pa.x,pa.y); ctx.lineTo(pb.x,pb.y); ctx.stroke(); ctx.restore();
}
function drawDot(p,color,r=5,label="") {
  const pp = projectPoint(p);
  ctx.fillStyle=color; ctx.beginPath(); ctx.arc(pp.x,pp.y,r,0,Math.PI*2); ctx.fill();
  if (label) { ctx.fillStyle="#dbe7e5"; ctx.font="12px Segoe UI"; ctx.fillText(label, pp.x+8, pp.y-8); }
}
function drawArrow(a,b,color) {
  drawLine(a,b,color,2);
  drawDot(b,color,3);
}
function drawWireBox(center, axes, half, color, width=1.4, dash=[]) {
  const corners = [];
  for (const sx of [-1,1]) for (const sy of [-1,1]) for (const sz of [-1,1]) {
    corners.push(add(add(add(center, mul(axes[0], sx*half[0])), mul(axes[1], sy*half[1])), mul(axes[2], sz*half[2])));
  }
  const id = (sx,sy,sz) => (sx > 0 ? 4 : 0) + (sy > 0 ? 2 : 0) + (sz > 0 ? 1 : 0);
  for (const sy of [-1,1]) for (const sz of [-1,1]) drawLine(corners[id(-1,sy,sz)], corners[id(1,sy,sz)], color, width, dash);
  for (const sx of [-1,1]) for (const sz of [-1,1]) drawLine(corners[id(sx,-1,sz)], corners[id(sx,1,sz)], color, width, dash);
  for (const sx of [-1,1]) for (const sy of [-1,1]) drawLine(corners[id(sx,sy,-1)], corners[id(sx,sy,1)], color, width, dash);
}
function localFrameFromHints(limb) {
  let forwardLocal = limb.eeForwardLocal ? norm(limb.eeForwardLocal) : [1,0,0];
  if (len(forwardLocal) < 1e-5 && len(limb.toeOffset) > 1e-5) forwardLocal = norm(limb.toeOffset);
  if (len(forwardLocal) < 1e-5) forwardLocal = [1,0,0];
  let upLocal = limb.eeUpLocal ? norm(limb.eeUpLocal) : [0,0,1];
  upLocal = sub(upLocal, mul(forwardLocal, dot(upLocal, forwardLocal)));
  if (len(upLocal) < 1e-5) {
    const fallback = Math.abs(forwardLocal[2]) < 0.8 ? [0,0,1] : [0,1,0];
    upLocal = sub(fallback, mul(forwardLocal, dot(fallback, forwardLocal)));
  }
  upLocal = norm(upLocal);
  const rightLocal = norm(cross(upLocal, forwardLocal));
  upLocal = norm(cross(forwardLocal, rightLocal));
  return {forwardLocal, rightLocal, upLocal};
}
function colliderFrame(limbIdx, eeRot) {
  const limb = DATA.limbs[limbIdx];
  const local = localFrameFromHints(limb);
  const forward = norm(rowMul(local.forwardLocal, eeRot));
  const right = norm(rowMul(local.rightLocal, eeRot));
  const up = norm(rowMul(local.upLocal, eeRot));
  if (limb.kind === "leg") {
    const toeLen = Math.max(0.12, len(limb.toeOffset));
    return {
      axes: [forward, right, up],
      half: [Math.max(0.10, toeLen * 0.78), 0.045, 0.025],
      centerOffset: mul(forward, toeLen * 0.26),
      nose: true,
    };
  }
  return {
    axes: [forward, right, up],
    half: [0.105, 0.032, 0.042],
    centerOffset: mul(forward, 0.060),
    nose: true,
  };
}
function drawEeCollider(limbIdx, target, eeRot, active=false) {
  const spec = colliderFrame(limbIdx, eeRot);
  const center = add(target, spec.centerOffset);
  const edge = active ? "#f0d36b" : "#40515b";
  drawWireBox(center, spec.axes, spec.half, edge, active ? 1.8 : 1.0, active ? [] : [5,5]);
  if (spec.nose) {
    const toe = add(center, mul(spec.axes[0], spec.half[0]));
    const left = add(add(center, mul(spec.axes[0], spec.half[0]*0.55)), mul(spec.axes[1], spec.half[1]));
    const right = add(add(center, mul(spec.axes[0], spec.half[0]*0.55)), mul(spec.axes[1], -spec.half[1]));
    drawLine(left, toe, active ? "#ffe38a" : "#52636b", active ? 1.8 : 1.0);
    drawLine(right, toe, active ? "#ffe38a" : "#52636b", active ? 1.8 : 1.0);
  }
  if (!active) return;
  const axisLen = 0.12;
  drawArrow(target, add(target, mul(spec.axes[0], axisLen)), "#ff5f5f");
  drawArrow(target, add(target, mul(spec.axes[1], axisLen * 0.85)), "#6df06d");
  drawArrow(target, add(target, mul(spec.axes[2], axisLen * 0.85)), "#6da8ff");
}
function resize() {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(r.width*dpr));
  canvas.height = Math.max(1, Math.floor(r.height*dpr));
  ctx.setTransform(dpr,0,0,dpr,0,0);
}
function drawGrid() {
  const z = 0, size = 1.5, step = 0.1;
  for (let i=-size;i<=size+1e-6;i+=step) {
    const major = Math.abs(Math.round(i*10)%5)===0;
    drawLine([i,-size,z],[i,size,z], major ? "#24323c" : "#18242b", major ? 1.2 : 0.8);
    drawLine([-size,i,z],[size,i,z], major ? "#24323c" : "#18242b", major ? 1.2 : 0.8);
  }
}
function draw() {
  resize();
  ctx.clearRect(0,0,canvas.clientWidth,canvas.clientHeight);
  drawGrid();
  const frame = DATA.frames[state.frame];
  for (let i=0;i<DATA.parents.length;i++) {
    const p = DATA.parents[i];
    if (p >= 0) drawLine(frame[p], frame[i], "#61717a", 1.5);
  }
  for (let i=0;i<DATA.limbs.length;i++) {
    const limb = DATA.limbs[i];
    if (ui.showColliders.checked) drawEeCollider(i, state.targets[i], eeRotFor(i), i===state.limb);
    drawDot(state.targets[i], i===state.limb ? "#f0d36b" : "#73838c", i===state.limb ? 8 : 5, i===state.limb ? "drag EE" : "");
    if (limb.toe !== null && limb.toe !== undefined) drawDot(frame[limb.toe], "#596873", 3);
  }
  for (let i=0;i<DATA.limbs.length;i++) {
    if (i === state.limb) continue;
    const ghost = solve(i, 0, state.refMode);
    drawLine(ghost.base, ghost.mid, "#2b7d70", 2);
    drawLine(ghost.mid, ghost.end, "#2b7d70", 2);
    if (ghost.toe) drawLine(ghost.end, ghost.toe, "#2b7d70", 1.4);
  }
  const old = solve(state.limb, state.thetaDeg, "old");
  const sol = solve(state.limb, state.thetaDeg, state.refMode);
  if (ui.showOld.checked && state.refMode !== "old") {
    drawLine(old.base, old.mid, "#ff8a67", 3, [8,5]);
    drawLine(old.mid, old.end, "#ff8a67", 3, [8,5]);
    drawDot(old.mid, "#ff8a67", 5, "old");
  }
  drawLine(sol.base, sol.mid, "#53f0c4", 5);
  drawLine(sol.mid, sol.end, "#53f0c4", 5);
  if (sol.toe) drawLine(sol.end, sol.toe, "#53f0c4", 3);
  drawDot(sol.base, "#ffffff", 5, "base");
  drawDot(sol.mid, "#53f0c4", 7, "knee/elbow");
  drawDot(sol.end, "#f0d36b", 8, "EE");
  if (sol.toe) drawDot(sol.toe, "#f0d36b", 5, "toe");
  drawArrow(sol.base, add(sol.base, mul(sol.ref, 0.22)), "#8aa7ff");
  drawArrow(sol.base, add(sol.base, mul(sol.pole, 0.24)), "#53f0c4");
  const ee = sol.end, r = 0.16;
  drawArrow(ee, add(ee, mul(rowMul([1,0,0], sol.eeRot), r)), "#ff5f5f");
  drawArrow(ee, add(ee, mul(rowMul([0,1,0], sol.eeRot), r)), "#6df06d");
  drawArrow(ee, add(ee, mul(rowMul([0,0,1], sol.eeRot), r)), "#6da8ff");
  updateStats(sol, old);
}
function resetTargetsForFrame() {
  state.targets = DATA.limbs.map(l => DATA.frames[state.frame][l.end].slice());
  state.rotDeltas = DATA.limbs.map(_ => ({yaw:0,pitch:0,roll:0}));
  syncPositionSliders();
  syncRotationSliders();
}
function straightTargetForFrame(limbIdx) {
  const limb = DATA.limbs[limbIdx], frame = DATA.frames[state.frame];
  const base = frame[limb.start];
  let axis = norm(sub(frame[limb.end], base));
  if (len(axis) < 1e-5) axis = norm(limb.restAxis);
  const reach = Math.max(0.001, limb.lengths[0] + limb.lengths[1] - 1e-4);
  return add(base, mul(axis, reach));
}
function resetStraightTargetsForFrame() {
  state.targets = DATA.limbs.map((_, i) => straightTargetForFrame(i));
  state.rotDeltas = DATA.limbs.map(_ => ({yaw:0,pitch:0,roll:0}));
  syncPositionSliders();
  syncRotationSliders();
}
function setNeutralPose() {
  resetStraightTargetsForFrame();
  state.thetaDeg = 0;
  ui.theta.value = 0;
  draw();
}
function setZeroTheta() {
  state.thetaDeg = 0;
  ui.theta.value = 0;
  draw();
}
function resetSelectedRotation() {
  state.rotDeltas[state.limb] = {yaw:0,pitch:0,roll:0};
  syncRotationSliders();
  draw();
}
function rotateSelected(key, deltaDeg) {
  const d = state.rotDeltas[state.limb] || {yaw:0,pitch:0,roll:0};
  d[key] = clamp((d[key] || 0) + deltaDeg, -90, 90);
  state.rotDeltas[state.limb] = d;
  syncRotationSliders();
  draw();
}
function resetAlphaForSelectedLimb() {
  state.alphaDeg = defaultAlphaForLimb(state.limb);
  ui.alpha.value = state.alphaDeg;
  ui.theta.min = -state.alphaDeg;
  ui.theta.max = state.alphaDeg;
  state.thetaDeg = clamp(state.thetaDeg, -state.alphaDeg, state.alphaDeg);
  ui.theta.value = state.thetaDeg;
}
function syncPositionSliders() {
  const p = state.targets[state.limb];
  for (const [id,val] of [["targetX",p[0]],["targetY",p[1]],["targetZ",p[2]]]) {
    const el = ui[id], margin = 0.25;
    el.min = (val - margin - 0.8).toFixed(3);
    el.max = (val + margin + 0.8).toFixed(3);
    el.value = val;
  }
}
function syncRotationSliders() {
  const d = state.rotDeltas[state.limb] || {yaw:0,pitch:0,roll:0};
  ui.yaw.value = d.yaw; ui.pitch.value = d.pitch; ui.roll.value = d.roll;
}
function updateLabels() {
  $("frameValue").textContent = state.frame;
  $("thetaValue").textContent = state.thetaDeg.toFixed(2);
  $("alphaValue").textContent = state.alphaDeg.toFixed(0);
  const p = state.targets[state.limb];
  $("xValue").textContent = p[0].toFixed(3);
  $("yValue").textContent = p[1].toFixed(3);
  $("zValue").textContent = p[2].toFixed(3);
  const d = state.rotDeltas[state.limb] || {yaw:0,pitch:0,roll:0};
  $("yawValue").textContent = `${d.yaw.toFixed(0)} deg`;
  $("pitchValue").textContent = `${d.pitch.toFixed(0)} deg`;
  $("rollValue").textContent = `${d.roll.toFixed(0)} deg`;
}
function updateStats(sol, old) {
  const limb = DATA.limbs[state.limb], authoredMid = DATA.frames[state.frame][limb.mid];
  const d = state.rotDeltas[state.limb] || {yaw:0,pitch:0,roll:0};
  const newErr = len(sub(sol.mid, authoredMid));
  const oldErr = len(sub(old.mid, authoredMid));
  const stats = DATA.thetaStatsDeg[state.limb];
  const actual = project(sub(authoredMid, sol.base), sol.axis);
  const meanRef = referencePole(state.limb, sol.end, sol.axis, sol.eeRot, "mean");
  const authoredTheta = signedAngleDeg(meanRef, actual, sol.axis);
  $("stats").textContent =
    `clip: ${DATA.npz.split(/[\\\\/]/).pop()}\n` +
    `limb: ${limb.name}\n` +
    `new mid error vs authored: ${(newErr*100).toFixed(2)} cm\n` +
    `old mid error vs authored: ${(oldErr*100).toFixed(2)} cm\n` +
    `authored theta from mean ref: ${authoredTheta.toFixed(1)} deg\n` +
    `clip theta mean-ref range: ${stats.min.toFixed(1)} .. ${stats.max.toFixed(1)} deg\n` +
    `clip abs95 / absmax: ${stats.abs95.toFixed(1)} / ${stats.absMax.toFixed(1)} deg\n` +
    `EE world delta X/Y/Z: ${d.pitch.toFixed(0)} / ${d.roll.toFixed(0)} / ${d.yaw.toFixed(0)} deg\n` +
    `target clamped: ${sol.clamped ? "yes" : "no"}`;
  updateLabels();
}
function setThetaFromAuthored() {
  const limb = DATA.limbs[state.limb], frame = DATA.frames[state.frame];
  const base = frame[limb.start], mid = frame[limb.mid], target = state.targets[state.limb];
  const axis = norm(sub(target, base));
  const eeRot = eeRotFor(state.limb);
  const ref = referencePole(state.limb, target, axis, eeRot, state.refMode);
  const actual = project(sub(mid, base), axis);
  state.thetaDeg = clamp(signedAngleDeg(ref, actual, axis), -state.alphaDeg, state.alphaDeg);
  ui.theta.value = state.thetaDeg;
  draw();
}
function init() {
  for (let i=0;i<DATA.limbs.length;i++) {
    const opt = document.createElement("option");
    opt.value = String(i); opt.textContent = DATA.limbs[i].name;
    ui.limb.appendChild(opt);
  }
  ui.limb.value = String(state.limb);
  ui.frame.max = String(DATA.frames.length - 1);
  ui.frame.value = String(state.frame);
  ui.alpha.value = String(state.alphaDeg);
  ui.theta.min = String(-state.alphaDeg); ui.theta.max = String(state.alphaDeg);
  setNeutralPose();
  bind();
  draw();
}
function bind() {
  ui.limb.addEventListener("change", () => { state.limb = Number(ui.limb.value); syncPositionSliders(); syncRotationSliders(); resetAlphaForSelectedLimb(); setZeroTheta(); });
  ui.frame.addEventListener("input", () => { state.frame = Number(ui.frame.value); setNeutralPose(); });
  ui.reference.addEventListener("change", () => { state.refMode = ui.reference.value; setThetaFromAuthored(); });
  ui.theta.addEventListener("input", () => { state.thetaDeg = Number(ui.theta.value); draw(); });
  ui.alpha.addEventListener("input", () => { state.alphaDeg = Number(ui.alpha.value); ui.theta.min=-state.alphaDeg; ui.theta.max=state.alphaDeg; state.thetaDeg=clamp(state.thetaDeg,-state.alphaDeg,state.alphaDeg); ui.theta.value=state.thetaDeg; draw(); });
  ui.showOld.addEventListener("change", draw);
  ui.showColliders.addEventListener("change", draw);
  for (const [id,idx] of [["targetX",0],["targetY",1],["targetZ",2]]) ui[id].addEventListener("input", () => { state.targets[state.limb][idx]=Number(ui[id].value); draw(); });
  for (const [id,key] of [["yaw","yaw"],["pitch","pitch"],["roll","roll"]]) ui[id].addEventListener("input", () => { state.rotDeltas[state.limb][key]=Number(ui[id].value); draw(); });
  $("resetPose").addEventListener("click", () => { resetTargetsForFrame(); draw(); });
  $("neutralPose").addEventListener("click", setNeutralPose);
  $("thetaAuthored").addEventListener("click", setThetaFromAuthored);
  $("zeroTheta").addEventListener("click", setZeroTheta);
  $("yawMinus").addEventListener("click", () => rotateSelected("yaw", -15));
  $("yawPlus").addEventListener("click", () => rotateSelected("yaw", 15));
  $("pitchMinus").addEventListener("click", () => rotateSelected("pitch", -15));
  $("pitchPlus").addEventListener("click", () => rotateSelected("pitch", 15));
  $("rollMinus").addEventListener("click", () => rotateSelected("roll", -15));
  $("rollPlus").addEventListener("click", () => rotateSelected("roll", 15));
  $("resetRotation").addEventListener("click", resetSelectedRotation);
  $("zeroRotationTheta").addEventListener("click", () => { resetSelectedRotation(); setZeroTheta(); });
  $("viewFront").addEventListener("click", () => { state.yaw = 0; state.pitch = 5; draw(); });
  $("viewSide").addEventListener("click", () => { state.yaw = 90; state.pitch = 5; draw(); });
  $("viewTop").addEventListener("click", () => { state.yaw = 0; state.pitch = 89; draw(); });
  canvas.addEventListener("contextmenu", e => e.preventDefault());
  canvas.addEventListener("mousedown", e => {
    state.lastMouse = [e.clientX, e.clientY];
    if (e.button === 1 || e.button === 2 || e.shiftKey) { state.dragging = {kind:"pan"}; return; }
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    let best = null, bestD = 1e9;
    for (let i=0;i<state.targets.length;i++) {
      const p = projectPoint(state.targets[i]);
      const d = Math.hypot(mx-p.x, my-p.y);
      if (d < bestD) { bestD=d; best=i; }
    }
    if (best !== null && bestD < 22) {
      state.limb = best; ui.limb.value = String(best); syncPositionSliders(); syncRotationSliders();
      state.dragging = {kind:"ee", limb:best, depth:projectPoint(state.targets[best]).z};
      canvas.classList.add("dragging");
    } else {
      state.dragging = {kind:"orbit"};
    }
  });
  window.addEventListener("mouseup", () => { state.dragging=null; canvas.classList.remove("dragging"); });
  window.addEventListener("mousemove", e => {
    if (!state.dragging) return;
    const dx = e.clientX - state.lastMouse[0], dy = e.clientY - state.lastMouse[1];
    state.lastMouse = [e.clientX, e.clientY];
    if (state.dragging.kind === "pan") { state.pan[0]+=dx; state.pan[1]+=dy; draw(); return; }
    if (state.dragging.kind === "orbit") { state.yaw += dx*0.35; state.pitch = clamp(state.pitch + dy*0.25, -89, 89); draw(); return; }
    const rect = canvas.getBoundingClientRect();
    const p = unprojectAtDepth(e.clientX - rect.left, e.clientY - rect.top, state.dragging.depth);
    state.targets[state.limb] = p;
    syncPositionSliders();
    draw();
  });
  canvas.addEventListener("wheel", e => { e.preventDefault(); state.zoom = clamp(state.zoom * Math.exp(-e.deltaY * 0.001), 0.2, 6); draw(); }, {passive:false});
  window.addEventListener("keydown", e => {
    const tag = (e.target && e.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "select" || tag === "button") return;
    const k = e.key.toLowerCase();
    if (k === "q") rotateSelected("yaw", -5);
    else if (k === "e") rotateSelected("yaw", 5);
    else if (k === "w") rotateSelected("pitch", -5);
    else if (k === "s") rotateSelected("pitch", 5);
    else if (k === "a") rotateSelected("roll", -5);
    else if (k === "d") rotateSelected("roll", 5);
  });
}
init();
</script>
</body>
</html>
"""


def write_viewer(npz: Path, out: Path, frame: int, open_browser: bool) -> Path:
    data = build_payload(npz, frame)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))), encoding="utf-8")
    if open_browser:
        try:
            subprocess.Popen(["powershell", "-NoProfile", "-Command", "Start-Process", str(out)], cwd=str(PROJECT_ROOT))
        except OSError:
            pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an interactive EE-local theta IK sandbox.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    npz = args.npz if args.npz.is_absolute() else (PROJECT_ROOT / args.npz).resolve()
    out = args.out if args.out.is_absolute() else (PROJECT_ROOT / args.out).resolve()
    path = write_viewer(npz, out, args.frame, args.open)
    print(path)


if __name__ == "__main__":
    main()
