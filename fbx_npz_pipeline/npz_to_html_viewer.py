from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NPZ Motion Viewer - {title}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #111317;
      --panel: #1a1d22;
      --panel2: #23272f;
      --text: #eceff4;
      --muted: #9aa3b2;
      --line: #313743;
      --accent: #54d6a3;
      --left: #73a8ff;
      --right: #ff9f6e;
      --center: #e6df83;
      --volume: rgba(223, 203, 186, 0.34);
      --volume-line: rgba(255, 238, 222, 0.52);
      --foot: rgba(236, 218, 202, 0.44);
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ width: 100%; height: 100%; margin: 0; overflow: hidden; }}
    body {{
      font: 13px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
    }}
    #app {{ width: 100vw; height: 100vh; display: grid; grid-template-rows: 1fr auto; }}
    #viewport {{ position: relative; min-height: 0; }}
    canvas {{ width: 100%; height: 100%; display: block; background: #101216; cursor: grab; }}
    canvas:active {{ cursor: grabbing; }}
    .hud {{
      position: absolute;
      top: 12px;
      left: 12px;
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      pointer-events: none;
    }}
    .pill {{
      background: rgba(26, 29, 34, 0.9);
      border: 1px solid rgba(255,255,255,0.08);
      padding: 6px 8px;
      border-radius: 6px;
      color: var(--muted);
    }}
    .pill strong {{ color: var(--text); font-weight: 600; }}
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
      grid-template-columns: auto minmax(160px, 1fr) auto auto auto auto auto auto auto;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
      background: var(--panel);
      border-top: 1px solid var(--line);
    }}
    button, select, input {{
      font: inherit;
      color: var(--text);
      background: var(--panel2);
      border: 1px solid var(--line);
      border-radius: 6px;
      height: 32px;
    }}
    button {{ min-width: 42px; padding: 0 11px; cursor: pointer; }}
    button:hover {{ border-color: #596273; }}
    input[type="range"] {{ width: 100%; accent-color: var(--accent); }}
    label {{ color: var(--muted); display: inline-flex; align-items: center; gap: 6px; }}
    .num {{ width: 70px; padding: 0 7px; }}
    .small {{ width: 84px; padding: 0 7px; }}
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
        <div class="pill"><strong>{title}</strong></div>
        <div class="pill"><strong id="frameText">0</strong> / {last_frame}</div>
        <div class="pill">{fps} FPS</div>
        <div class="pill">{bone_count} bones</div>
      </div>
      <div id="tooltip"></div>
    </div>
    <div id="controls">
      <button id="play" title="Play or pause">Play</button>
      <input id="frame" type="range" min="0" max="{last_frame}" value="0" step="1">
      <label class="wide-extra">Speed <input id="speed" class="small" type="number" value="1" min="0.1" max="4" step="0.1"></label>
      <label class="wide-extra">Scale <input id="scale" class="small" type="number" value="1" min="0.05" max="10" step="0.05"></label>
      <button id="reset" class="wide-extra" title="Reset camera">Reset</button>
      <button id="volumes" class="wide-extra" title="Toggle limb volumes">Hide Volumes</button>
      <button id="helpers" class="wide-extra" title="Toggle IK, twist, weapon, and attach nodes">Helpers</button>
      <button id="details" class="wide-extra" title="Toggle fingers and detail bones">Details</button>
      <button id="labels" class="wide-extra" title="Toggle joint labels">Labels</button>
    </div>
  </div>
  <script id="motion-data" type="application/json">{payload}</script>
  <script>
    const motion = JSON.parse(document.getElementById("motion-data").textContent);
    motion.positions = new Float32Array(Uint8Array.from(atob(motion.positions_b64), c => c.charCodeAt(0)).buffer);
    motion.basis = new Float32Array(Uint8Array.from(atob(motion.basis_b64), c => c.charCodeAt(0)).buffer);
    delete motion.positions_b64;
    delete motion.basis_b64;

    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const frameSlider = document.getElementById("frame");
    const frameText = document.getElementById("frameText");
    const playButton = document.getElementById("play");
    const speedInput = document.getElementById("speed");
    const scaleInput = document.getElementById("scale");
    const resetButton = document.getElementById("reset");
    const labelsButton = document.getElementById("labels");
    const volumesButton = document.getElementById("volumes");
    const helpersButton = document.getElementById("helpers");
    const detailsButton = document.getElementById("details");
    const tooltip = document.getElementById("tooltip");

    let frame = 0;
    let playing = false;
    let showLabels = false;
    let showVolumes = true;
    let showHelpers = false;
    let showDetails = false;
    let yaw = -0.78;
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

    const parents = motion.parents;
    const names = motion.bone_names;
    const nameToIndex = new Map(names.map((name, index) => [name, index]));
    const T = motion.frame_count;
    const J = motion.bone_count;
    const bounds = motion.bounds;
    const center = [
      (bounds.min[0] + bounds.max[0]) * 0.5,
      (bounds.min[1] + bounds.max[1]) * 0.5,
      (bounds.min[2] + bounds.max[2]) * 0.5
    ];
    const extent = Math.max(
      bounds.max[0] - bounds.min[0],
      bounds.max[1] - bounds.min[1],
      bounds.max[2] - bounds.min[2],
      1
    );

    function resize() {{
      const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.floor(rect.width * dpr);
      canvas.height = Math.floor(rect.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }}

    function posAt(f, j) {{
      const i = (f * J + j) * 3;
      return [motion.positions[i], motion.positions[i + 1], motion.positions[i + 2]];
    }}

    function basisAxisAt(f, j, axis) {{
      const i = (f * J + j) * 9 + axis * 3;
      return [motion.basis[i], motion.basis[i + 1], motion.basis[i + 2]];
    }}

    function add3(a, b) {{ return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }}
    function sub3(a, b) {{ return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }}
    function mul3(a, s) {{ return [a[0] * s, a[1] * s, a[2] * s]; }}
    function dot3(a, b) {{ return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }}
    function len3(a) {{ return Math.max(1e-6, Math.hypot(a[0], a[1], a[2])); }}

    function isHelperName(name) {{
      return name.includes("_twist_") ||
        name.startsWith("ik_") ||
        name.startsWith("weapon_") ||
        name === "attach";
    }}

    function isDetailName(name) {{
      return name.includes("thumb") ||
        name.includes("index") ||
        name.includes("middle") ||
        name.includes("ring") ||
        name.includes("pinky") ||
        name.includes("metacarpal");
    }}

    function shouldDrawJoint(j) {{
      if (!showHelpers && isHelperName(names[j])) return false;
      if (!showDetails && isDetailName(names[j])) return false;
      return true;
    }}

    const anatomicalEdges = [
      ["root", "pelvis"],
      ["pelvis", "spine_01"], ["spine_01", "spine_02"], ["spine_02", "spine_03"], ["spine_03", "spine_04"], ["spine_04", "spine_05"],
      ["spine_05", "neck_01"], ["neck_01", "neck_02"], ["neck_02", "head"],
      ["spine_05", "clavicle_l"], ["clavicle_l", "upperarm_l"], ["upperarm_l", "lowerarm_l"], ["lowerarm_l", "hand_l"],
      ["spine_05", "clavicle_r"], ["clavicle_r", "upperarm_r"], ["upperarm_r", "lowerarm_r"], ["lowerarm_r", "hand_r"],
      ["pelvis", "thigh_l"], ["thigh_l", "calf_l"], ["calf_l", "foot_l"],
      ["pelvis", "thigh_r"], ["thigh_r", "calf_r"], ["calf_r", "foot_r"],
      ["hand_l", "index_01_l"], ["index_01_l", "index_02_l"], ["index_02_l", "index_03_l"],
      ["hand_l", "middle_01_l"], ["middle_01_l", "middle_02_l"], ["middle_02_l", "middle_03_l"],
      ["hand_l", "ring_01_l"], ["ring_01_l", "ring_02_l"], ["ring_02_l", "ring_03_l"],
      ["hand_l", "pinky_01_l"], ["pinky_01_l", "pinky_02_l"], ["pinky_02_l", "pinky_03_l"],
      ["hand_l", "thumb_01_l"], ["thumb_01_l", "thumb_02_l"], ["thumb_02_l", "thumb_03_l"],
      ["hand_r", "index_01_r"], ["index_01_r", "index_02_r"], ["index_02_r", "index_03_r"],
      ["hand_r", "middle_01_r"], ["middle_01_r", "middle_02_r"], ["middle_02_r", "middle_03_r"],
      ["hand_r", "ring_01_r"], ["ring_01_r", "ring_02_r"], ["ring_02_r", "ring_03_r"],
      ["hand_r", "pinky_01_r"], ["pinky_01_r", "pinky_02_r"], ["pinky_02_r", "pinky_03_r"],
      ["hand_r", "thumb_01_r"], ["thumb_01_r", "thumb_02_r"], ["thumb_02_r", "thumb_03_r"]
    ].map(([a, b]) => [nameToIndex.get(a), nameToIndex.get(b)]).filter(([a, b]) => a !== undefined && b !== undefined);

    const volumeSpecs = [
      ["pelvis", "spine_01", 12.5], ["spine_01", "spine_02", 13.5], ["spine_02", "spine_03", 14.5],
      ["spine_03", "spine_04", 14.5], ["spine_04", "spine_05", 13.5], ["spine_05", "neck_01", 8.5],
      ["neck_02", "head", 10],
      ["clavicle_l", "upperarm_l", 5.5], ["upperarm_l", "lowerarm_l", 5.8], ["lowerarm_l", "hand_l", 4.8],
      ["clavicle_r", "upperarm_r", 5.5], ["upperarm_r", "lowerarm_r", 5.8], ["lowerarm_r", "hand_r", 4.8],
      ["pelvis", "thigh_l", 9.5], ["thigh_l", "calf_l", 8.3], ["calf_l", "foot_l", 6.2],
      ["pelvis", "thigh_r", 9.5], ["thigh_r", "calf_r", 8.3], ["calf_r", "foot_r", 6.2]
    ].map(([a, b, r]) => ({{ a: nameToIndex.get(a), b: nameToIndex.get(b), r }})).filter(v => v.a !== undefined && v.b !== undefined);

    const footSpecs = [
      {{ ankle: nameToIndex.get("foot_l"), toe: nameToIndex.get("ball_l"), side: "l" }},
      {{ ankle: nameToIndex.get("foot_r"), toe: nameToIndex.get("ball_r"), side: "r" }}
    ].filter(v => v.ankle !== undefined && v.toe !== undefined);

    const handSpecs = [
      {{ bone: nameToIndex.get("hand_l"), mid: nameToIndex.get("middle_03_l"), side: "l" }},
      {{ bone: nameToIndex.get("hand_r"), mid: nameToIndex.get("middle_03_r"), side: "r" }}
    ].filter(v => v.bone !== undefined);

    function colorFor(name) {{
      const n = name.toLowerCase();
      if (n.endsWith("_l") || n.includes("left")) return getComputedStyle(document.documentElement).getPropertyValue("--left").trim();
      if (n.endsWith("_r") || n.includes("right")) return getComputedStyle(document.documentElement).getPropertyValue("--right").trim();
      return getComputedStyle(document.documentElement).getPropertyValue("--center").trim();
    }}

    function rotateProject(p) {{
      const s = Number(scaleInput.value) || 1;
      let x = (p[0] - center[0]) * s;
      let y = (p[1] - center[1]) * s;
      let z = (p[2] - center[2]) * s;

      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = cy * x + sy * z;
      const z1 = -sy * x + cy * z;
      const y1 = cp * y - sp * z1;
      const z2 = sp * y + cp * z1;

      const rect = canvas.getBoundingClientRect();
      const base = Math.min(rect.width, rect.height) * 0.68 * zoom / extent;
      const perspective = 1 / (1 + z2 / (extent * 3.0));
      return {{
        x: rect.width * 0.5 + panX + x1 * base * perspective,
        y: rect.height * 0.56 + panY - y1 * base * perspective,
        z: z2,
        p: perspective,
        scalePx: base * perspective
      }};
    }}

    function projectWorld(p) {{
      return rotateProject(p);
    }}

    function drawCapsule(a, b, radiusWorld, fill, stroke) {{
      const pa = pointsCache[a];
      const pb = pointsCache[b];
      const radiusPx = Math.max(4, radiusWorld * 0.5 * (pa.scalePx + pb.scalePx));
      ctx.save();
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

    function drawFootBlock(ankle, toe, side) {{
      const foot = posAt(frame, ankle);
      const toePos = posAt(frame, toe);
      let up = basisAxisAt(frame, ankle, 0);
      let forward = basisAxisAt(frame, ankle, 1);
      const sideAxis = basisAxisAt(frame, ankle, 2);
      const toeVector = sub3(toePos, foot);
      if (dot3(forward, toeVector) < 0) forward = mul3(forward, -1);
      if (up[1] < 0) up = mul3(up, -1);

      const ballDistance = Math.max(10, Math.min(28, Math.abs(dot3(toeVector, forward))));
      const heelLength = 7.0;
      const length = ballDistance + heelLength;
      const width = 11.0;
      const height = 6.4;
      const ballBack = add3(toePos, mul3(forward, -ballDistance));
      const heelBack = add3(ballBack, mul3(forward, -heelLength));
      const center = add3(add3(heelBack, mul3(forward, length * 0.5)), mul3(up, -0.6));
      drawOrientedBox(center, forward, sideAxis, up, [length, width, height], colorFor(names[ankle]), getComputedStyle(document.documentElement).getPropertyValue("--foot").trim());
      drawAxisTick(center, forward, length * 0.5, colorFor(names[ankle]));
    }}

    function drawToeBlock(ankle, toe) {{
      const foot = posAt(frame, ankle);
      const toePos = posAt(frame, toe);
      let forward = basisAxisAt(frame, toe, 0);
      let up = basisAxisAt(frame, toe, 1);
      const sideAxis = basisAxisAt(frame, toe, 2);
      const toeVector = sub3(toePos, foot);
      if (dot3(forward, toeVector) < 0) forward = mul3(forward, -1);
      if (up[1] < 0) up = mul3(up, -1);
      const toeLength = 6.5;
      const center = add3(add3(toePos, mul3(forward, toeLength * 0.5)), mul3(up, -0.6));
      drawOrientedBox(center, forward, sideAxis, up, [toeLength, 11.0, 6.4], colorFor(names[toe]), "rgba(236, 218, 202, 0.34)");
      drawAxisTick(center, forward, toeLength * 0.5, colorFor(names[toe]));
    }}

    function drawHandBox(spec) {{
      const bone = spec.bone;
      const hand = posAt(frame, bone);
      let forward = basisAxisAt(frame, bone, 0);
      const sideAxis = basisAxisAt(frame, bone, 1);
      let up = basisAxisAt(frame, bone, 2);
      if (spec.mid !== undefined) {{
        const fingerVector = sub3(posAt(frame, spec.mid), hand);
        if (dot3(forward, fingerVector) < 0) forward = mul3(forward, -1);
      }}
      if (up[1] < 0) up = mul3(up, -1);
      const center = add3(add3(hand, mul3(forward, 5.2)), mul3(up, -0.25));
      drawOrientedBox(center, forward, up, sideAxis, [12.5, 11.5, 3.8], colorFor(names[bone]), "rgba(236, 218, 202, 0.38)");
      drawAxisTick(center, forward, 8, colorFor(names[bone]));
    }}

    function drawAxisTick(center, axis, length, color) {{
      const a = projectWorld(center);
      const b = projectWorld(add3(center, mul3(axis, length)));
      ctx.save();
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.9;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      ctx.restore();
    }}

    function drawOrientedBox(center, axisX, axisY, axisZ, dims, stroke, fill) {{
      const hx = dims[0] * 0.5, hy = dims[1] * 0.5, hz = dims[2] * 0.5;
      const corners = [
        [-hx,-hy,-hz], [ hx,-hy,-hz], [ hx, hy,-hz], [-hx, hy,-hz],
        [-hx,-hy, hz], [ hx,-hy, hz], [ hx, hy, hz], [-hx, hy, hz]
      ].map(c => projectWorld(add3(add3(add3(center, mul3(axisX, c[0])), mul3(axisY, c[1])), mul3(axisZ, c[2]))));
      const faces = [
        [0,1,2,3], [4,5,6,7], [0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7]
      ].map(face => ({{
        face,
        z: face.reduce((acc, i) => acc + corners[i].z, 0) / face.length
      }})).sort((a, b) => a.z - b.z);
      ctx.save();
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

    function drawGrid() {{
      const rect = canvas.getBoundingClientRect();
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.lineWidth = 1;
      const step = 40;
      for (let x = (rect.width * 0.5 + panX) % step; x < rect.width; x += step) {{
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, rect.height); ctx.stroke();
      }}
      for (let y = (rect.height * 0.56 + panY) % step; y < rect.height; y += step) {{
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(rect.width, y); ctx.stroke();
      }}
    }}

    let pointsCache = [];

    function draw() {{
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      drawGrid();

      const points = Array.from({{ length: J }}, (_, j) => rotateProject(posAt(frame, j)));
      pointsCache = points;
      const edges = [];
      for (const [parent, child] of anatomicalEdges) {{
        if (shouldDrawJoint(parent) && shouldDrawJoint(child)) {{
          edges.push([points[child].z + points[parent].z, child, parent]);
        }}
      }}
      if (showHelpers) {{
        for (let j = 0; j < J; j++) {{
          if (parents[j] >= 0 && isHelperName(names[j])) {{
            edges.push([points[j].z + points[parents[j]].z, j, parents[j]]);
          }}
        }}
      }}
      edges.sort((a, b) => a[0] - b[0]);

      if (showVolumes) {{
        const fill = getComputedStyle(document.documentElement).getPropertyValue("--volume").trim();
        const stroke = getComputedStyle(document.documentElement).getPropertyValue("--volume-line").trim();
        const sortedVolumes = volumeSpecs.slice().sort((a, b) => points[a.a].z + points[a.b].z - points[b.a].z - points[b.b].z);
        for (const v of sortedVolumes) drawCapsule(v.a, v.b, v.r, fill, stroke);
        for (const h of handSpecs) drawHandBox(h);
        for (const f of footSpecs) {{
          drawFootBlock(f.ankle, f.toe, f.side);
          drawToeBlock(f.ankle, f.toe);
        }}
      }}

      for (const [, child, parent] of edges) {{
        const a = points[parent], b = points[child];
        ctx.strokeStyle = showHelpers && isHelperName(names[child]) ? "rgba(180,185,196,0.38)" : colorFor(names[child]);
        ctx.globalAlpha = showHelpers && isHelperName(names[child]) ? 0.55 : 0.86;
        ctx.lineWidth = Math.max(1.2, 3.3 * b.p);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }}
      ctx.globalAlpha = 1;

      for (let j = 0; j < J; j++) {{
        if (!shouldDrawJoint(j)) continue;
        const q = points[j];
        ctx.fillStyle = showHelpers && isHelperName(names[j]) ? "rgba(180,185,196,0.68)" : colorFor(names[j]);
        ctx.beginPath();
        ctx.arc(q.x, q.y, Math.max(2.2, 4.2 * q.p), 0, Math.PI * 2);
        ctx.fill();
        if (showLabels && q.p > 0.6) {{
          ctx.fillStyle = "rgba(236,239,244,0.72)";
          ctx.font = "11px system-ui, sans-serif";
          ctx.fillText(names[j], q.x + 6, q.y - 5);
        }}
      }}

      frameSlider.value = frame;
      frameText.textContent = String(frame);
    }}

    function animate(now) {{
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
      requestAnimationFrame(animate);
    }}

    function updateTooltip(e) {{
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      let best = null;
      for (let j = 0; j < J; j++) {{
        const q = rotateProject(posAt(frame, j));
        const d = Math.hypot(q.x - mx, q.y - my);
        if (d < 9 && (!best || d < best.d)) best = {{ d, name: names[j] }};
      }}
      if (best) {{
        tooltip.style.display = "block";
        tooltip.style.left = `${{e.clientX - rect.left}}px`;
        tooltip.style.top = `${{e.clientY - rect.top}}px`;
        tooltip.textContent = best.name;
        hoverName = best.name;
      }} else {{
        tooltip.style.display = "none";
        hoverName = "";
      }}
    }}

    playButton.addEventListener("click", () => {{
      playing = !playing;
      playButton.textContent = playing ? "Pause" : "Play";
    }});
    frameSlider.addEventListener("input", () => {{
      frame = Number(frameSlider.value);
      playing = false;
      playButton.textContent = "Play";
    }});
    resetButton.addEventListener("click", () => {{
      yaw = -0.78; pitch = -0.18; zoom = 1; panX = 0; panY = 0;
    }});
    labelsButton.addEventListener("click", () => {{
      showLabels = !showLabels;
      labelsButton.textContent = showLabels ? "Hide Labels" : "Labels";
    }});
    volumesButton.addEventListener("click", () => {{
      showVolumes = !showVolumes;
      volumesButton.textContent = showVolumes ? "Hide Volumes" : "Volumes";
    }});
    helpersButton.addEventListener("click", () => {{
      showHelpers = !showHelpers;
      helpersButton.textContent = showHelpers ? "Hide Helpers" : "Helpers";
    }});
    detailsButton.addEventListener("click", () => {{
      showDetails = !showDetails;
      detailsButton.textContent = showDetails ? "Hide Details" : "Details";
    }});

    canvas.addEventListener("pointerdown", e => {{
      dragging = true;
      panning = e.shiftKey || e.button === 1;
      lastX = e.clientX;
      lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    }});
    canvas.addEventListener("pointerup", e => {{
      dragging = false;
      canvas.releasePointerCapture(e.pointerId);
    }});
    canvas.addEventListener("pointermove", e => {{
      if (dragging) {{
        const dx = e.clientX - lastX;
        const dy = e.clientY - lastY;
        lastX = e.clientX;
        lastY = e.clientY;
        if (panning) {{
          panX += dx;
          panY += dy;
        }} else {{
          yaw += dx * 0.008;
          pitch = Math.max(-1.35, Math.min(1.35, pitch + dy * 0.008));
        }}
      }}
      updateTooltip(e);
    }});
    canvas.addEventListener("wheel", e => {{
      e.preventDefault();
      zoom *= Math.exp(-e.deltaY * 0.001);
      zoom = Math.max(0.12, Math.min(8, zoom));
    }}, {{ passive: false }});

    window.addEventListener("resize", resize);
    window.addEventListener("keydown", e => {{
      if (e.code === "Space") {{
        e.preventDefault();
        playButton.click();
      }} else if (e.code === "ArrowRight") {{
        frame = Math.min(T - 1, frame + 1);
      }} else if (e.code === "ArrowLeft") {{
        frame = Math.max(0, frame - 1);
      }}
    }});

    resize();
    requestAnimationFrame(animate);
  </script>
</body>
</html>
"""


def canonicalize_for_view(data: np.lib.npyio.NpzFile, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    basis = np.asarray(data["global_matrix"][:, :, :3, :3], dtype=np.float32)
    up_axis = int(data["axis_up_axis"]) if "axis_up_axis" in data.files else 2
    axis_info = {"up_axis": up_axis, "canonicalized": False}

    if up_axis == 3:
        # FBX Z-up/Unreal-style source -> viewer/training Y-up convention:
        # canonical x = source y, canonical y = source z, canonical z = source x.
        perm = [1, 2, 0]
        positions = positions[..., perm].copy()
        basis = basis[..., :, perm].copy()
        axis_info["canonicalized"] = True
        axis_info["mapping"] = "z_up_to_y_up_yzx"

    return positions, basis, axis_info


def make_payload(npz_path: Path) -> dict:
    data = np.load(npz_path)
    positions = np.asarray(data["global_joint_pos"], dtype=np.float32)
    positions, basis, axis_info = canonicalize_for_view(data, positions)
    parents = np.asarray(data["parents"], dtype=np.int32)
    bone_names = [str(x) for x in data["bone_names"]]
    fps = float(data["fps"])

    finite = np.isfinite(positions)
    if not finite.all():
        raise ValueError("global_joint_pos contains non-finite values")

    return {
        "source_npz": str(npz_path),
        "frame_count": int(positions.shape[0]),
        "bone_count": int(positions.shape[1]),
        "fps": fps,
        "axis": axis_info,
        "parents": parents.tolist(),
        "bone_names": bone_names,
        "bounds": {
            "min": positions.reshape(-1, 3).min(axis=0).tolist(),
            "max": positions.reshape(-1, 3).max(axis=0).tolist(),
        },
        "positions_b64": base64.b64encode(positions.tobytes()).decode("ascii"),
        "basis_b64": base64.b64encode(basis.tobytes()).decode("ascii"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a self-contained HTML viewer from a motion NPZ.")
    parser.add_argument("npz", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    npz_path = args.npz.resolve()
    payload = make_payload(npz_path)
    output = args.output
    if output is None:
        output = Path("data") / "visualizations" / f"{npz_path.stem}_viewer.html"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    title = npz_path.stem
    html = HTML_TEMPLATE.format(
        title=title,
        fps=f"{payload['fps']:.3g}",
        bone_count=payload["bone_count"],
        last_frame=payload["frame_count"] - 1,
        payload=json.dumps(payload, separators=(",", ":")),
    )
    output.write_text(html, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
