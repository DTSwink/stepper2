from __future__ import annotations

# Put the files you want to compare here. Relative paths are resolved from the
# stepper project root. Leave npz_path/checkpoint_path empty to use the newest
# training run's checkpoint_best.pt and the source NPZ recorded in that checkpoint.
npz_path = ""
checkpoint_path = ""
output_path = "training/runs/model_comparisons/model_comparison.html"

import argparse
import base64
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

try:
    from . import ik_core as tl
except ImportError:
    import ik_core as tl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model Motion Comparison - {title}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #101216;
      --panel: #1a1d22;
      --panel2: #23272f;
      --text: #edf1f7;
      --muted: #9da6b5;
      --line: #313846;
      --grid: rgba(255,255,255,0.055);
      --gt: #72a7ff;
      --pred: #ff9b6a;
      --root: #e6df83;
      --accent: #55d6a7;
      --volume: rgba(223, 203, 186, 0.30);
      --volume-line: rgba(255, 238, 222, 0.48);
      --foot: rgba(236, 218, 202, 0.40);
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; width: 100%; height: 100%; overflow: hidden; background: var(--bg); }}
    body {{ font: 13px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--text); }}
    #app {{ width: 100vw; height: 100vh; display: grid; grid-template-rows: 1fr auto; }}
    #viewport {{ position: relative; min-height: 0; }}
    canvas {{ display: block; width: 100%; height: 100%; background: #0f1217; cursor: grab; }}
    canvas:active {{ cursor: grabbing; }}
    .hud {{
      position: absolute;
      top: 12px;
      left: 12px;
      right: 12px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      pointer-events: none;
    }}
    .pill {{
      padding: 6px 8px;
      border-radius: 6px;
      background: rgba(26,29,34,0.9);
      border: 1px solid rgba(255,255,255,0.08);
      color: var(--muted);
    }}
    .pill strong {{ color: var(--text); font-weight: 650; }}
    .legend-dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; }}
    #tooltip {{
      position: absolute;
      display: none;
      padding: 5px 7px;
      border-radius: 5px;
      color: var(--text);
      background: rgba(0, 0, 0, 0.78);
      border: 1px solid rgba(255,255,255,0.14);
      pointer-events: none;
      transform: translate(10px, 10px);
      white-space: nowrap;
    }}
    #controls {{
      display: grid;
      grid-template-columns: auto auto minmax(180px, 1fr) auto auto auto auto auto auto auto;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
      background: var(--panel);
      border-top: 1px solid var(--line);
    }}
    button, input, select {{
      height: 32px;
      color: var(--text);
      background: var(--panel2);
      border: 1px solid var(--line);
      border-radius: 6px;
      font: inherit;
    }}
    button {{ min-width: 42px; padding: 0 11px; cursor: pointer; }}
    button:hover, select:hover {{ border-color: #596273; }}
    select {{ min-width: 220px; padding: 0 8px; }}
    input[type="range"] {{ width: 100%; accent-color: var(--accent); }}
    label {{ display: inline-flex; align-items: center; gap: 6px; color: var(--muted); }}
    .num {{ width: 72px; padding: 0 7px; }}
    @media (max-width: 760px) {{
      #controls {{ grid-template-columns: auto 1fr auto; }}
      #controls .wide-extra {{ display: none; }}
    }}
  </style>
</head>
<body>
  <div id="app">
    <div id="viewport">
      <canvas id="canvas"></canvas>
      <div class="hud">
        <div class="pill"><strong id="titleText">{title}</strong></div>
        <div class="pill"><strong id="frameText">0</strong> / <span id="lastFrameText">{last_frame}</span></div>
        <div class="pill"><span id="fpsText">{fps}</span> FPS</div>
        <div class="pill"><span id="boneCountText">{bone_count}</span> bones</div>
        <div class="pill"><span class="legend-dot" style="background:var(--gt)"></span>ground truth</div>
        <div class="pill"><span class="legend-dot" style="background:var(--pred)"></span>model <strong id="rolloutText">autoregressive</strong></div>
        <div class="pill">mean joint error <strong id="errText">0.000</strong> m</div>
      </div>
      <div id="tooltip"></div>
    </div>
    <div id="controls">
      <select id="runSelect" title="Choose comparison run"></select>
      <button id="play" title="Play or pause">Play</button>
      <input id="frame" type="range" min="0" max="{last_frame}" value="0" step="1">
      <label class="wide-extra">Speed <input id="speed" class="num" type="number" value="1" min="0.1" max="4" step="0.1"></label>
      <label class="wide-extra">Scale <input id="scale" class="num" type="number" value="1" min="0.05" max="10" step="0.05"></label>
      <button id="predictionMode" class="wide-extra" title="Toggle one-step prediction or full autoregressive rollout">Autoreg</button>
      <button id="mode" class="wide-extra" title="Overlay or side-by-side comparison">Overlay</button>
      <button id="volumes" class="wide-extra" title="Toggle bone orientation volumes">Hide Volumes</button>
      <button id="trails" class="wide-extra" title="Toggle recent root trails">Trails</button>
      <button id="labels" class="wide-extra" title="Toggle joint labels">Labels</button>
      <button id="reset" class="wide-extra" title="Reset camera">Reset</button>
    </div>
  </div>
  <script id="motion-data" type="application/json">{payload}</script>
  <script>
    const motions = JSON.parse(document.getElementById("motion-data").textContent);
    for (const item of motions) {{
      item.gt = new Float32Array(Uint8Array.from(atob(item.gt_b64), c => c.charCodeAt(0)).buffer);
      item.gtBasis = new Float32Array(Uint8Array.from(atob(item.gt_basis_b64), c => c.charCodeAt(0)).buffer);
      item.predOne = new Float32Array(Uint8Array.from(atob(item.pred_one_step_b64), c => c.charCodeAt(0)).buffer);
      item.predOneBasis = new Float32Array(Uint8Array.from(atob(item.pred_one_step_basis_b64), c => c.charCodeAt(0)).buffer);
      item.predAr = new Float32Array(Uint8Array.from(atob(item.pred_ar_b64), c => c.charCodeAt(0)).buffer);
      item.predArBasis = new Float32Array(Uint8Array.from(atob(item.pred_ar_basis_b64), c => c.charCodeAt(0)).buffer);
      item.errOne = new Float32Array(Uint8Array.from(atob(item.err_one_step_b64), c => c.charCodeAt(0)).buffer);
      item.errAr = new Float32Array(Uint8Array.from(atob(item.err_ar_b64), c => c.charCodeAt(0)).buffer);
      delete item.gt_b64;
      delete item.gt_basis_b64;
      delete item.pred_one_step_b64;
      delete item.pred_one_step_basis_b64;
      delete item.pred_ar_b64;
      delete item.pred_ar_basis_b64;
      delete item.err_one_step_b64;
      delete item.err_ar_b64;
    }}

    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const runSelect = document.getElementById("runSelect");
    const titleText = document.getElementById("titleText");
    const frameSlider = document.getElementById("frame");
    const frameText = document.getElementById("frameText");
    const lastFrameText = document.getElementById("lastFrameText");
    const fpsText = document.getElementById("fpsText");
    const boneCountText = document.getElementById("boneCountText");
    const errText = document.getElementById("errText");
    const rolloutText = document.getElementById("rolloutText");
    const playButton = document.getElementById("play");
    const speedInput = document.getElementById("speed");
    const scaleInput = document.getElementById("scale");
    const predictionModeButton = document.getElementById("predictionMode");
    const modeButton = document.getElementById("mode");
    const volumesButton = document.getElementById("volumes");
    const trailsButton = document.getElementById("trails");
    const labelsButton = document.getElementById("labels");
    const resetButton = document.getElementById("reset");
    const tooltip = document.getElementById("tooltip");

    let motion = motions[0];
    let T = 0;
    let J = 0;
    let parents = [];
    let names = [];
    let rootIndex = 0;
    let bounds = null;
    let center = [0, 0, 0];
    let extent = 1;
    let volumeSpecs = [];
    let footSpecs = [];
    let handSpecs = [];

    let frame = motion.initial_frame || 0;
    let playing = false;
    let showLabels = false;
    let showTrails = true;
    let showVolumes = true;
    let splitMode = false;
    let predictionMode = "autoregressive";
    let yaw = -0.75;
    let pitch = -0.18;
    let zoom = 1.0;
    let panX = 0;
    let panY = 0;
    let dragging = false;
    let panning = false;
    let lastX = 0;
    let lastY = 0;
    let lastTime = performance.now();
    let frameCarry = 0;
    let hoverName = "";

    function activateMotion(index, preserveFrame = false) {{
      motion = motions[index];
      T = motion.frame_count;
      J = motion.bone_count;
      parents = motion.parents;
      names = motion.bone_names;
      rootIndex = motion.root_index;
      bounds = motion.bounds;
      center = [
        (bounds.min[0] + bounds.max[0]) * 0.5,
        (bounds.min[1] + bounds.max[1]) * 0.5,
        (bounds.min[2] + bounds.max[2]) * 0.5
      ];
      extent = Math.max(
        bounds.max[0] - bounds.min[0],
        bounds.max[1] - bounds.min[1],
        bounds.max[2] - bounds.min[2],
        1
      );
      if (!preserveFrame) frame = motion.initial_frame || 0;
      frame = Math.max(0, Math.min(T - 1, frame));
      frameSlider.max = String(T - 1);
      frameSlider.value = String(frame);
      titleText.textContent = motion.title || "model comparison";
      lastFrameText.textContent = String(T - 1);
      fpsText.textContent = String(Math.round(motion.fps));
      boneCountText.textContent = String(J);
      const nameToIndex = new Map(names.map((name, index) => [name, index]));
      volumeSpecs = [
        ["pelvis", "spine_01", 12.5], ["spine_01", "spine_02", 13.5], ["spine_02", "spine_03", 14.5],
        ["spine_03", "spine_04", 14.5], ["spine_04", "spine_05", 13.5], ["spine_05", "neck_01", 8.5],
        ["neck_02", "head", 10],
        ["clavicle_l", "upperarm_l", 5.5], ["upperarm_l", "lowerarm_l", 5.8], ["lowerarm_l", "hand_l", 4.8],
        ["clavicle_r", "upperarm_r", 5.5], ["upperarm_r", "lowerarm_r", 5.8], ["lowerarm_r", "hand_r", 4.8],
        ["pelvis", "thigh_l", 9.5], ["thigh_l", "calf_l", 8.3], ["calf_l", "foot_l", 6.2],
        ["pelvis", "thigh_r", 9.5], ["thigh_r", "calf_r", 8.3], ["calf_r", "foot_r", 6.2]
      ].map(([a, b, r]) => ({{ a: nameToIndex.get(a), b: nameToIndex.get(b), r }})).filter(v => v.a !== undefined && v.b !== undefined);
      footSpecs = [
        {{ ankle: nameToIndex.get("foot_l"), toe: nameToIndex.get("ball_l") }},
        {{ ankle: nameToIndex.get("foot_r"), toe: nameToIndex.get("ball_r") }}
      ].filter(v => v.ankle !== undefined && v.toe !== undefined);
      handSpecs = [
        {{ bone: nameToIndex.get("hand_l"), mid: nameToIndex.get("middle_03_l") }},
        {{ bone: nameToIndex.get("hand_r"), mid: nameToIndex.get("middle_03_r") }}
      ].filter(v => v.bone !== undefined);
      frameCarry = 0;
    }}

    for (let i = 0; i < motions.length; i++) {{
      const option = document.createElement("option");
      option.value = String(i);
      option.textContent = motions[i].title || `run ${{i + 1}}`;
      runSelect.appendChild(option);
    }}
    runSelect.style.display = motions.length > 1 ? "block" : "none";
    activateMotion(0);

    function resize() {{
      const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.floor(rect.width * dpr);
      canvas.height = Math.floor(rect.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }}

    function posAt(arr, f, j, offsetX = 0) {{
      const i = (f * J + j) * 3;
      return [arr[i] + offsetX, arr[i + 1], arr[i + 2]];
    }}
    function basisAxisAt(basis, f, j, axis) {{
      const i = (f * J + j) * 9 + axis * 3;
      return [basis[i], basis[i + 1], basis[i + 2]];
    }}
    function add3(a, b) {{ return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }}
    function sub3(a, b) {{ return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }}
    function mul3(a, s) {{ return [a[0] * s, a[1] * s, a[2] * s]; }}
    function dot3(a, b) {{ return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }}
    function len3(a) {{ return Math.max(1e-6, Math.hypot(a[0], a[1], a[2])); }}

    function rotateProject(p) {{
      const scale = Number(scaleInput.value) || 1;
      const x0 = (p[0] - center[0]) * scale;
      const y0 = (p[1] - center[1]) * scale;
      const z0 = (p[2] - center[2]) * scale;
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = x0 * cy - z0 * sy;
      const z1 = x0 * sy + z0 * cy;
      const y1 = y0 * cp - z1 * sp;
      const z2 = y0 * sp + z1 * cp;
      const s = Math.min(canvas.clientWidth, canvas.clientHeight) * 0.72 * zoom / extent;
      return {{
        x: canvas.clientWidth * 0.5 + panX + x1 * s,
        y: canvas.clientHeight * 0.55 + panY - y1 * s,
        z: z2
      }};
    }}

    function drawGrid() {{
      const step = 0.5;
      const size = Math.max(3, Math.ceil(extent / step) * step);
      ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--grid").trim();
      ctx.lineWidth = 1;
      for (let i = -size; i <= size + 1e-4; i += step) {{
        const a = rotateProject([center[0] - size, 0, center[2] + i]);
        const b = rotateProject([center[0] + size, 0, center[2] + i]);
        const c = rotateProject([center[0] + i, 0, center[2] - size]);
        const d = rotateProject([center[0] + i, 0, center[2] + size]);
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(c.x, c.y); ctx.lineTo(d.x, d.y); ctx.stroke();
      }}
    }}

    function drawJoint(p, color, radius) {{
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }}

    function worldRadiusPx(radiusWorld) {{
      const scale = Number(scaleInput.value) || 1;
      return Math.max(3, radiusWorld * scale * Math.min(canvas.clientWidth, canvas.clientHeight) * 0.72 * zoom / extent);
    }}

    function drawCapsule(arr, a, b, radiusWorld, offsetX, fill, stroke, alpha) {{
      const pa = rotateProject(posAt(arr, frame, a, offsetX));
      const pb = rotateProject(posAt(arr, frame, b, offsetX));
      const radiusPx = worldRadiusPx(radiusWorld);
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = fill;
      ctx.lineWidth = radiusPx * 2;
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
      ctx.strokeStyle = stroke;
      ctx.lineWidth = Math.max(1, radiusPx * 0.08);
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
      ctx.restore();
    }}

    function drawAxisTick(center, axis, length, color) {{
      const a = rotateProject(center);
      const b = rotateProject(add3(center, mul3(axis, length)));
      ctx.save();
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.85;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      ctx.restore();
    }}

    function drawOrientedBox(center, axisX, axisY, axisZ, dims, stroke, fill, alpha) {{
      const hx = dims[0] * 0.5, hy = dims[1] * 0.5, hz = dims[2] * 0.5;
      const corners = [
        [-hx,-hy,-hz], [ hx,-hy,-hz], [ hx, hy,-hz], [-hx, hy,-hz],
        [-hx,-hy, hz], [ hx,-hy, hz], [ hx, hy, hz], [-hx, hy, hz]
      ].map(c => rotateProject(add3(add3(add3(center, mul3(axisX, c[0])), mul3(axisY, c[1])), mul3(axisZ, c[2]))));
      const faces = [
        [0,1,2,3], [4,5,6,7], [0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7]
      ].map(face => ({{
        face,
        z: face.reduce((acc, i) => acc + corners[i].z, 0) / face.length
      }})).sort((a, b) => a.z - b.z);
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.fillStyle = fill;
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 1.1;
      for (const item of faces) {{
        ctx.beginPath();
        for (let k = 0; k < item.face.length; k++) {{
          const p = corners[item.face[k]];
          if (k === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
        }}
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }}
      ctx.restore();
    }}

    function drawFootBlock(arr, basis, ankle, toe, offsetX, color, alpha) {{
      const foot = posAt(arr, frame, ankle, offsetX);
      const toePos = posAt(arr, frame, toe, offsetX);
      let up = basisAxisAt(basis, frame, ankle, 0);
      let forward = basisAxisAt(basis, frame, ankle, 1);
      const sideAxis = basisAxisAt(basis, frame, ankle, 2);
      const toeVector = sub3(toePos, foot);
      if (dot3(forward, toeVector) < 0) forward = mul3(forward, -1);
      if (up[1] < 0) up = mul3(up, -1);
      const ballDistance = Math.max(0.10, Math.min(0.28, Math.abs(dot3(toeVector, forward))));
      const heelLength = 0.07;
      const length = ballDistance + heelLength;
      const width = 0.11;
      const height = 0.064;
      const ballBack = add3(toePos, mul3(forward, -ballDistance));
      const heelBack = add3(ballBack, mul3(forward, -heelLength));
      const c = add3(add3(heelBack, mul3(forward, length * 0.5)), mul3(up, -0.006));
      drawOrientedBox(c, forward, sideAxis, up, [length, width, height], color, getComputedStyle(document.documentElement).getPropertyValue("--foot").trim(), alpha);
      drawAxisTick(c, forward, length * 0.5, color);
    }}

    function drawToeBlock(arr, basis, ankle, toe, offsetX, color, alpha) {{
      const foot = posAt(arr, frame, ankle, offsetX);
      const toePos = posAt(arr, frame, toe, offsetX);
      let forward = basisAxisAt(basis, frame, toe, 0);
      let up = basisAxisAt(basis, frame, toe, 1);
      const sideAxis = basisAxisAt(basis, frame, toe, 2);
      const toeVector = sub3(toePos, foot);
      if (dot3(forward, toeVector) < 0) forward = mul3(forward, -1);
      if (up[1] < 0) up = mul3(up, -1);
      const toeLength = 0.065;
      const c = add3(add3(toePos, mul3(forward, toeLength * 0.5)), mul3(up, -0.006));
      drawOrientedBox(c, forward, sideAxis, up, [toeLength, 0.11, 0.064], color, "rgba(236, 218, 202, 0.34)", alpha);
      drawAxisTick(c, forward, toeLength * 0.5, color);
    }}

    function drawHandBox(arr, basis, spec, offsetX, color, alpha) {{
      const bone = spec.bone;
      const hand = posAt(arr, frame, bone, offsetX);
      let forward = basisAxisAt(basis, frame, bone, 0);
      const sideAxis = basisAxisAt(basis, frame, bone, 1);
      let up = basisAxisAt(basis, frame, bone, 2);
      if (spec.mid !== undefined) {{
        const fingerVector = sub3(posAt(arr, frame, spec.mid, offsetX), hand);
        if (dot3(forward, fingerVector) < 0) forward = mul3(forward, -1);
      }}
      if (up[1] < 0) up = mul3(up, -1);
      const c = add3(add3(hand, mul3(forward, 0.052)), mul3(up, -0.0025));
      drawOrientedBox(c, forward, up, sideAxis, [0.125, 0.115, 0.038], color, "rgba(236, 218, 202, 0.38)", alpha);
      drawAxisTick(c, forward, 0.08, color);
    }}

    function drawVolumesFor(arr, basis, color, offsetX, alpha) {{
      if (!showVolumes) return;
      const fill = getComputedStyle(document.documentElement).getPropertyValue("--volume").trim();
      for (const v of volumeSpecs) drawCapsule(arr, v.a, v.b, v.r * 0.01, offsetX, fill, color, alpha);
      for (const h of handSpecs) drawHandBox(arr, basis, h, offsetX, color, alpha);
      for (const f of footSpecs) {{
        drawFootBlock(arr, basis, f.ankle, f.toe, offsetX, color, alpha);
        drawToeBlock(arr, basis, f.ankle, f.toe, offsetX, color, alpha);
      }}
    }}

    function drawSkeleton(arr, color, offsetX, alpha, width) {{
      const points = Array.from({{ length: J }}, (_, j) => rotateProject(posAt(arr, frame, j, offsetX)));
      const edges = [];
      for (let j = 0; j < J; j++) {{
        const p = parents[j];
        if (p >= 0) edges.push([points[j].z + points[p].z, j, p]);
      }}
      edges.sort((a, b) => a[0] - b[0]);
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      for (const [, j, p] of edges) {{
        ctx.beginPath();
        ctx.moveTo(points[p].x, points[p].y);
        ctx.lineTo(points[j].x, points[j].y);
        ctx.stroke();
      }}
      for (let j = 0; j < J; j++) {{
        drawJoint(points[j], color, j === rootIndex ? 4.5 : 2.6);
      }}
      ctx.globalAlpha = 1;
      return points;
    }}

    function drawTrailsFor(arr, color, offsetX) {{
      if (!showTrails) return;
      const start = Math.max(0, frame - 45);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.42;
      ctx.beginPath();
      for (let f = start; f <= frame; f++) {{
        const p = rotateProject(posAt(arr, f, rootIndex, offsetX));
        if (f === start) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }}
      ctx.stroke();
      ctx.globalAlpha = 1;
    }}

    function draw() {{
      resize();
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      drawGrid();
      const gtColor = getComputedStyle(document.documentElement).getPropertyValue("--gt").trim();
      const predColor = getComputedStyle(document.documentElement).getPropertyValue("--pred").trim();
      const pred = predictionMode === "autoregressive" ? motion.predAr : motion.predOne;
      const predBasis = predictionMode === "autoregressive" ? motion.predArBasis : motion.predOneBasis;
      const err = predictionMode === "autoregressive" ? motion.errAr : motion.errOne;
      const sep = splitMode ? extent * 0.42 : 0;
      drawTrailsFor(motion.gt, gtColor, -sep);
      drawTrailsFor(pred, predColor, sep);
      drawVolumesFor(motion.gt, motion.gtBasis, gtColor, -sep, splitMode ? 0.36 : 0.22);
      drawVolumesFor(pred, predBasis, predColor, sep, 0.42);
      const gtPoints = drawSkeleton(motion.gt, gtColor, -sep, splitMode ? 0.9 : 0.54, splitMode ? 4.0 : 5.0);
      const predPoints = drawSkeleton(pred, predColor, sep, 0.92, 3.2);

      if (!splitMode) {{
        ctx.strokeStyle = "rgba(255,255,255,0.18)";
        ctx.lineWidth = 1;
        for (let j = 0; j < J; j += 3) {{
          ctx.beginPath();
          ctx.moveTo(gtPoints[j].x, gtPoints[j].y);
          ctx.lineTo(predPoints[j].x, predPoints[j].y);
          ctx.stroke();
        }}
      }}

      if (showLabels) {{
        ctx.font = "11px system-ui, sans-serif";
        ctx.fillStyle = "rgba(237,241,247,0.72)";
        for (let j = 0; j < J; j++) {{
          const p = predPoints[j];
          ctx.fillText(names[j], p.x + 5, p.y - 5);
        }}
      }}

      if (splitMode) {{
        ctx.font = "15px system-ui, sans-serif";
        ctx.fillStyle = gtColor;
        ctx.fillText("ground truth", 16, canvas.clientHeight - 18);
        ctx.fillStyle = predColor;
        ctx.fillText("model", canvas.clientWidth - 70, canvas.clientHeight - 18);
      }}
      frameSlider.value = frame;
      frameText.textContent = String(frame);
      errText.textContent = err[frame].toFixed(4);
      rolloutText.textContent = predictionMode === "autoregressive" ? "autoregressive" : "one-step";
      playButton.textContent = playing ? "Pause" : "Play";
      predictionModeButton.textContent = predictionMode === "autoregressive" ? "Autoreg" : "One-step";
      modeButton.textContent = splitMode ? "Split" : "Overlay";
      volumesButton.textContent = showVolumes ? "Hide Volumes" : "Volumes";
      trailsButton.textContent = showTrails ? "Hide Trails" : "Trails";
      labelsButton.textContent = showLabels ? "Hide Labels" : "Labels";
    }}

    function tick(now) {{
      const dt = Math.min(0.1, (now - lastTime) / 1000);
      lastTime = now;
      if (playing) {{
        frameCarry += dt * motion.fps * (Number(speedInput.value) || 1);
        while (frameCarry >= 1) {{
          frame = (frame + 1) % T;
          frameCarry -= 1;
        }}
      }}
      draw();
      requestAnimationFrame(tick);
    }}

    canvas.addEventListener("mousemove", (event) => {{
      if (dragging) {{
        const dx = event.clientX - lastX;
        const dy = event.clientY - lastY;
        if (panning) {{
          panX += dx;
          panY += dy;
        }} else {{
          yaw += dx * 0.006;
          pitch = Math.max(-1.35, Math.min(1.35, pitch + dy * 0.006));
        }}
        lastX = event.clientX;
        lastY = event.clientY;
      }}
      tooltip.style.display = hoverName ? "block" : "none";
      tooltip.style.left = event.clientX + "px";
      tooltip.style.top = event.clientY + "px";
      tooltip.textContent = hoverName;
    }});
    canvas.addEventListener("mousedown", (event) => {{
      dragging = true;
      panning = event.button === 1 || event.shiftKey;
      lastX = event.clientX;
      lastY = event.clientY;
    }});
    window.addEventListener("mouseup", () => {{ dragging = false; }});
    canvas.addEventListener("wheel", (event) => {{
      event.preventDefault();
      zoom = Math.max(0.15, Math.min(8, zoom * Math.exp(-event.deltaY * 0.001)));
    }}, {{ passive: false }});
    frameSlider.addEventListener("input", () => {{
      frame = Number(frameSlider.value);
      frameCarry = 0;
    }});
    runSelect.addEventListener("change", () => {{
      activateMotion(Number(runSelect.value), true);
    }});
    playButton.addEventListener("click", () => {{ playing = !playing; }});
    predictionModeButton.addEventListener("click", () => {{
      predictionMode = predictionMode === "autoregressive" ? "one_step" : "autoregressive";
    }});
    modeButton.addEventListener("click", () => {{ splitMode = !splitMode; }});
    volumesButton.addEventListener("click", () => {{ showVolumes = !showVolumes; }});
    trailsButton.addEventListener("click", () => {{ showTrails = !showTrails; }});
    labelsButton.addEventListener("click", () => {{ showLabels = !showLabels; }});
    resetButton.addEventListener("click", () => {{
      yaw = -0.75; pitch = -0.18; zoom = 1.0; panX = 0; panY = 0;
    }});
    window.addEventListener("keydown", (event) => {{
      if (event.code === "Space") {{ event.preventDefault(); playing = !playing; }}
      if (event.key === "ArrowRight") frame = Math.min(T - 1, frame + 1);
      if (event.key === "ArrowLeft") frame = Math.max(0, frame - 1);
    }});
    window.addEventListener("resize", resize);
    requestAnimationFrame(tick);
  </script>
</body>
</html>
"""


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_optional_path(path_text: str | None) -> Path | None:
    if path_text is None or not str(path_text).strip():
        return None
    return resolve_path(str(path_text))


def find_latest_checkpoint() -> Path:
    run_root = PROJECT_ROOT / "training" / "runs"
    run_dirs = [path for path in run_root.iterdir() if path.is_dir()] if run_root.exists() else []
    candidates: list[tuple[float, Path, Path]] = []
    for run_dir in run_dirs:
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue
        checkpoints = [p for p in ckpt_dir.glob("*.pt") if p.is_file()]
        if not checkpoints:
            continue
        newest_mtime = max(p.stat().st_mtime for p in checkpoints)
        preferred = sorted(
            [p for p in checkpoints if p.stem.endswith("_best") or "_best_" in p.stem],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not preferred:
            preferred = sorted(checkpoints, key=lambda p: p.stat().st_mtime, reverse=True)
        candidates.append((newest_mtime, run_dir, preferred[0]))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {run_root}")
    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    return candidates[0][2].resolve()


def infer_npz_path(ckpt: dict, checkpoint: Path) -> Path:
    metadata = ckpt.get("metadata", {})
    source_paths = metadata.get("source_npz_paths", [])
    if not source_paths:
        source_paths = metadata.get("npz_paths", [])
    if source_paths:
        return resolve_path(str(source_paths[0]))
    npz_folder = metadata.get("npz_folder")
    if npz_folder:
        folder = resolve_path(str(npz_folder))
        npz_files = sorted(folder.glob("*.npz"))
        if npz_files:
            return npz_files[0].resolve()
    raise ValueError(
        "No --npz-path was provided and the checkpoint does not record a usable source NPZ "
        f"({checkpoint})"
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


def load_model(checkpoint: dict, clip: tl.MotionClip, cfg: tl.TrainConfig, device: torch.device) -> tl.MLPController:
    input_dim, output_dim = tl.make_batch_dims(clip, cfg)
    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def rollout_model(
    model: torch.nn.Module,
    clip: tl.MotionClip,
    cfg: tl.TrainConfig,
    device: torch.device,
    max_frames: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_count = clip.T if max_frames is None else min(clip.T, max(3, max_frames))
    frame_idx = torch.arange(frame_count, dtype=torch.long, device=device)
    gt_pos_t, gt_rot_t = tl.global_from_clip(clip, frame_idx, cfg, device)
    gt_pos = gt_pos_t.detach().cpu().numpy().astype(np.float32)
    gt_rot = gt_rot_t.detach().cpu().numpy().astype(np.float32)
    pred_ar_pos = np.zeros_like(gt_pos)
    pred_one_step_pos = np.zeros_like(gt_pos)
    pred_ar_rot = np.zeros_like(gt_rot)
    pred_one_step_rot = np.zeros_like(gt_rot)

    pred_ar_pos[:2] = gt_pos[:2]
    pred_one_step_pos[:2] = gt_pos[:2]
    pred_ar_rot[:2] = gt_rot[:2]
    pred_one_step_rot[:2] = gt_rot[:2]

    prev_idx = torch.tensor([0], dtype=torch.long)
    cur_idx = torch.tensor([1], dtype=torch.long)
    prev_pose = tl.get_pose_from_clip(clip, prev_idx, device)
    cur_pose = tl.get_pose_from_clip(clip, cur_idx, device)
    prev_pose, cur_pose = tl.maybe_apply_initial_offsets(clip, prev_idx, cur_idx, prev_pose, cur_pose, cfg, device)
    if cfg.freefall_body_height_offset_m != 0.0:
        tensors = clip.tensors(device)
        prev_root_pos = tensors["root_pos"].index_select(0, prev_idx.to(device))
        prev_root_rot = tensors["root_rot"].index_select(0, prev_idx.to(device))
        cur_root_pos = tensors["root_pos"].index_select(0, cur_idx.to(device))
        cur_root_rot = tensors["root_rot"].index_select(0, cur_idx.to(device))
        init_prev_pos, init_prev_rot, _ = tl.fk_from_pose(clip, prev_root_pos, prev_root_rot, prev_pose, device)
        init_cur_pos, init_cur_rot, _ = tl.fk_from_pose(clip, cur_root_pos, cur_root_rot, cur_pose, device)
        pred_ar_pos[0] = init_prev_pos[0].detach().cpu().numpy()
        pred_ar_rot[0] = init_prev_rot[0].detach().cpu().numpy()
        if frame_count > 1:
            pred_ar_pos[1] = init_cur_pos[0].detach().cpu().numpy()
            pred_ar_rot[1] = init_cur_rot[0].detach().cpu().numpy()

    for target in range(2, frame_count):
        target_idx = torch.tensor([target], dtype=torch.long)
        root_pos, root_rot, _yaw, _heading = tl.root_state(clip, target_idx, cfg, device)

        one_prev_idx = torch.tensor([target - 2], dtype=torch.long)
        one_cur_idx = torch.tensor([target - 1], dtype=torch.long)
        one_prev_pose = tl.get_pose_from_clip(clip, one_prev_idx, device)
        one_cur_pose = tl.get_pose_from_clip(clip, one_cur_idx, device)
        one_inp = tl.build_input(clip, one_prev_idx, one_cur_idx, one_prev_pose, one_cur_pose, cfg, device)
        one_raw_out = tl.predict_next_raw(model, one_inp, one_cur_pose, cfg)
        one_pose, _ = tl.output_to_pose(one_raw_out, clip)
        one_global_pos, one_global_rot, _ = tl.fk_from_pose(clip, root_pos, root_rot, one_pose, device)
        pred_one_step_pos[target] = one_global_pos[0].detach().cpu().numpy()
        pred_one_step_rot[target] = one_global_rot[0].detach().cpu().numpy()

        inp = tl.build_input(clip, prev_idx, cur_idx, prev_pose, cur_pose, cfg, device)
        raw_out = tl.predict_next_raw(model, inp, cur_pose, cfg)
        pred_pose, _ = tl.output_to_pose(raw_out, clip)
        global_pos, global_rot, canon_pos = tl.fk_from_pose(clip, root_pos, root_rot, pred_pose, device)

        pred_ar_pos[target] = global_pos[0].detach().cpu().numpy()
        pred_ar_rot[target] = global_rot[0].detach().cpu().numpy()

        prev_pose = cur_pose
        cur_pose = tl.next_pose_from_prediction(pred_pose, canon_pos)
        prev_idx = cur_idx
        cur_idx = target_idx

    error_ar = np.linalg.norm(pred_ar_pos - gt_pos, axis=-1).mean(axis=-1).astype(np.float32)
    error_one_step = np.linalg.norm(pred_one_step_pos - gt_pos, axis=-1).mean(axis=-1).astype(np.float32)
    return gt_pos, gt_rot, pred_one_step_pos, pred_one_step_rot, error_one_step, pred_ar_pos, pred_ar_rot, error_ar


def make_payload(
    clip: tl.MotionClip,
    gt_pos: np.ndarray,
    gt_rot: np.ndarray,
    pred_one_step_pos: np.ndarray,
    pred_one_step_rot: np.ndarray,
    error_one_step: np.ndarray,
    pred_ar_pos: np.ndarray,
    pred_ar_rot: np.ndarray,
    error_ar: np.ndarray,
) -> dict:
    both = np.concatenate((gt_pos, pred_one_step_pos, pred_ar_pos), axis=1)
    bounds_min = both.reshape(-1, 3).min(axis=0)
    bounds_max = both.reshape(-1, 3).max(axis=0)
    return {
        "title": "",
        "frame_count": int(gt_pos.shape[0]),
        "bone_count": int(gt_pos.shape[1]),
        "fps": float(clip.fps),
        "bone_names": clip.body_names,
        "parents": clip.parents_body.cpu().numpy().astype(int).tolist(),
        "root_index": int(clip.pelvis),
        "initial_frame": int(min(max(2, gt_pos.shape[0] // 2), gt_pos.shape[0] - 1)),
        "bounds": {"min": bounds_min.tolist(), "max": bounds_max.tolist()},
        "gt_b64": base64.b64encode(np.ascontiguousarray(gt_pos, dtype=np.float32).tobytes()).decode("ascii"),
        "gt_basis_b64": base64.b64encode(np.ascontiguousarray(gt_rot, dtype=np.float32).tobytes()).decode("ascii"),
        "pred_one_step_b64": base64.b64encode(np.ascontiguousarray(pred_one_step_pos, dtype=np.float32).tobytes()).decode("ascii"),
        "pred_one_step_basis_b64": base64.b64encode(np.ascontiguousarray(pred_one_step_rot, dtype=np.float32).tobytes()).decode("ascii"),
        "pred_ar_b64": base64.b64encode(np.ascontiguousarray(pred_ar_pos, dtype=np.float32).tobytes()).decode("ascii"),
        "pred_ar_basis_b64": base64.b64encode(np.ascontiguousarray(pred_ar_rot, dtype=np.float32).tobytes()).decode("ascii"),
        "err_one_step_b64": base64.b64encode(np.ascontiguousarray(error_one_step, dtype=np.float32).tobytes()).decode("ascii"),
        "err_ar_b64": base64.b64encode(np.ascontiguousarray(error_ar, dtype=np.float32).tobytes()).decode("ascii"),
    }


def write_html(payload: dict | list[dict], output: Path, title: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payloads = payload if isinstance(payload, list) else [payload]
    html = HTML_TEMPLATE.format(
        title=title,
        last_frame=payloads[0]["frame_count"] - 1,
        fps=f"{payloads[0]['fps']:.0f}",
        bone_count=payloads[0]["bone_count"],
        payload=json.dumps(payloads, separators=(",", ":")),
    )
    output.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize autoregressive model rollout against NPZ ground truth.")
    parser.add_argument("--npz-path", default=npz_path)
    parser.add_argument("--checkpoint-path", default=checkpoint_path)
    parser.add_argument("--output-path", default=output_path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    checkpoint = resolve_optional_path(args.checkpoint_path) or find_latest_checkpoint()
    output = resolve_path(args.output_path)
    device = torch.device(args.device)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    npz = resolve_optional_path(args.npz_path) or infer_npz_path(ckpt, checkpoint)
    cfg = tl.TrainConfig()
    apply_config_dict(cfg, ckpt.get("config", {}))
    cfg.device = str(device)
    cfg.use_torch_compile = False

    clip = tl.MotionClip(npz, cfg)
    model = load_model(ckpt, clip, cfg, device)
    gt_pos, gt_rot, pred_one_step_pos, pred_one_step_rot, error_one_step, pred_ar_pos, pred_ar_rot, error_ar = rollout_model(
        model, clip, cfg, device, args.max_frames
    )
    payload = make_payload(
        clip,
        gt_pos,
        gt_rot,
        pred_one_step_pos,
        pred_one_step_rot,
        error_one_step,
        pred_ar_pos,
        pred_ar_rot,
        error_ar,
    )
    title = f"{npz.stem} vs {checkpoint.parent.parent.name}"
    payload["title"] = title
    write_html(payload, output, title)

    print(f"wrote {output}")
    print(f"checkpoint {checkpoint}")
    print(f"npz {npz}")
    print(f"frames {payload['frame_count']} bones {payload['bone_count']} fps {payload['fps']:.0f}")
    print(f"one_step_mean_joint_error_start {float(error_one_step[0]):.6f}")
    print(f"one_step_mean_joint_error_end {float(error_one_step[-1]):.6f}")
    print(f"one_step_mean_joint_error_avg {float(error_one_step.mean()):.6f}")
    print(f"one_step_mean_joint_error_max {float(error_one_step.max()):.6f}")
    print(f"autoregressive_mean_joint_error_start {float(error_ar[0]):.6f}")
    print(f"autoregressive_mean_joint_error_end {float(error_ar[-1]):.6f}")
    print(f"autoregressive_mean_joint_error_avg {float(error_ar.mean()):.6f}")
    print(f"autoregressive_mean_joint_error_max {float(error_ar.max()):.6f}")


if __name__ == "__main__":
    main()
