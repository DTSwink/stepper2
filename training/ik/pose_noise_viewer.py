from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import torch

try:
    from .bootstrap import PROJECT_ROOT, ensure_paths
    from . import ik_core as tl
    from . import train_full_ae_envelope as full
    from . import train_simple_ae_controller as ctl
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    import ik_core as tl
    import train_full_ae_envelope as full
    import train_simple_ae_controller as ctl

ensure_paths()


RUNS_DIR = PROJECT_ROOT / "training" / "runs"
DEFAULT_OUTPUT = RUNS_DIR / "model_comparisons" / "pose_noise_viewer.html"


def b64_tensor(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().float().numpy()
    return base64.b64encode(arr.tobytes()).decode("ascii")


def rot_noise_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    return tl.axis_angle_to_row_matrix(torch.nn.functional.normalize(axis, dim=-1, eps=1e-8), angle)


def rot6_slices(store: ctl.SimpleClipStore) -> list[slice]:
    slices = [slice(3, 9)]
    cursor = 9
    for _ in range(store.Jcore):
        slices.append(slice(cursor, cursor + 6))
        cursor += 6
    payload_start = cursor
    for spec in tl.IK_PAYLOAD_SLICES:
        rot_slice = spec["rot6"]
        assert isinstance(rot_slice, slice)
        slices.append(slice(payload_start + rot_slice.start, payload_start + rot_slice.stop))
    return slices


def position_slices(store: ctl.SimpleClipStore) -> list[slice]:
    slices = [slice(0, 3)]
    payload_start = 9 + store.Jcore * 6
    for spec in tl.IK_PAYLOAD_SLICES:
        pos_slice = spec["pos"]
        assert isinstance(pos_slice, slice)
        slices.append(slice(payload_start + pos_slice.start, payload_start + pos_slice.stop))
    return slices


def scalar_noise_slices(store: ctl.SimpleClipStore) -> list[slice]:
    slices: list[slice] = []
    payload_start = 9 + store.Jcore * 6
    for spec in tl.IK_PAYLOAD_SLICES:
        pole_slice = spec["pole"]
        toe_slice = spec["toe_float"]
        assert isinstance(pole_slice, slice)
        slices.append(slice(payload_start + pole_slice.start, payload_start + pole_slice.stop))
        if toe_slice is not None:
            assert isinstance(toe_slice, slice)
            slices.append(slice(payload_start + toe_slice.start, payload_start + toe_slice.stop))
    return slices


def fk_positions_and_rotations(
    store: ctl.SimpleClipStore,
    vec: torch.Tensor,
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pose, _raw = tl.output_to_pose(vec, store.prototype)
    pos, rot, _canon = tl.fk_from_pose(store.prototype, root_pos, root_rot, pose, store.device)
    return pos, rot


def build_payload(
    sample_count: int,
    noise_count: int,
    level_count: int,
    pos_sigma_m: float,
    rot_sigma_deg: float,
    scalar_sigma: float,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    cfg = ctl.make_cfg(device, {})
    clips = ctl.load_clips(full.full_specs(), cfg)
    store = ctl.SimpleClipStore(clips, cfg, device)
    pool = ctl.build_training_start_pool(store, 1)
    sample_count = min(max(1, int(sample_count)), int(pool.row_count))
    noise_count = max(1, int(noise_count))
    level_count = max(2, int(level_count))

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    rows = torch.randperm(int(pool.row_count), device=device, generator=gen)[:sample_count]
    clip_ids = pool.clip_ids.index_select(0, rows)
    frame_idx = pool.starts.index_select(0, rows)
    root_pos, root_rot, _yaw, _heading = store.root_state(clip_ids, frame_idx)
    base_vec = store.get_target_output(clip_ids, frame_idx)
    base_pos, base_rot = fk_positions_and_rotations(store, base_vec, root_pos, root_rot)

    vec_dim = int(base_vec.shape[-1])
    pos_slices = position_slices(store)
    rot_slices = rot6_slices(store)
    scalar_slices = scalar_noise_slices(store)
    rot_group_count = len(rot_slices)

    pos_noise = torch.randn((sample_count, noise_count, len(pos_slices), 3), device=device, generator=gen)
    scalar_noise = torch.randn((sample_count, noise_count, len(scalar_slices)), device=device, generator=gen)
    rot_axes = torch.randn((sample_count, noise_count, rot_group_count, 3), device=device, generator=gen)
    rot_angle_unit = torch.randn((sample_count, noise_count, rot_group_count), device=device, generator=gen)

    levels = torch.linspace(0.0, 1.0, level_count, device=device)
    all_vecs = torch.empty((sample_count, noise_count, level_count, vec_dim), device=device)
    rot_sigma_rad = float(rot_sigma_deg) * 3.141592653589793 / 180.0
    for li, amount in enumerate(levels):
        vec = base_vec[:, None, :].expand(sample_count, noise_count, vec_dim).clone()
        for pi, sl in enumerate(pos_slices):
            vec[:, :, sl] = vec[:, :, sl] + amount * float(pos_sigma_m) * pos_noise[:, :, pi]
        for si, sl in enumerate(scalar_slices):
            vec[:, :, sl] = vec[:, :, sl] + amount * float(scalar_sigma) * scalar_noise[:, :, si, None]
        for ri, sl in enumerate(rot_slices):
            d6 = base_vec[:, sl].reshape(sample_count, 1, 6).expand(sample_count, noise_count, 6)
            base_r = tl.rotation_6d_to_matrix(d6.reshape(-1, 6)).reshape(sample_count, noise_count, 3, 3)
            delta = rot_noise_matrix(
                rot_axes[:, :, ri].reshape(-1, 3),
                (amount * rot_sigma_rad * rot_angle_unit[:, :, ri]).reshape(-1),
            ).reshape(sample_count, noise_count, 3, 3)
            vec[:, :, sl] = tl.rotmat_to_6d(delta @ base_r)
        all_vecs[:, :, li] = vec

    flat_vec = all_vecs.reshape(sample_count * noise_count * level_count, vec_dim)
    flat_clip = clip_ids[:, None, None].expand(sample_count, noise_count, level_count).reshape(-1)
    flat_frame = frame_idx[:, None, None].expand(sample_count, noise_count, level_count).reshape(-1)
    flat_root_pos, flat_root_rot, _fyaw, _fhead = store.root_state(flat_clip, flat_frame)
    noisy_pos, noisy_rot = fk_positions_and_rotations(store, flat_vec, flat_root_pos, flat_root_rot)
    noisy_pos = noisy_pos.reshape(sample_count, noise_count, level_count, store.J, 3)
    noisy_rot = noisy_rot.reshape(sample_count, noise_count, level_count, store.J, 3, 3)

    sample_labels = [
        f"{Path(store.clips[int(ci)].path).stem} / frame {int(fi)}"
        for ci, fi in zip(clip_ids.detach().cpu().tolist(), frame_idx.detach().cpu().tolist())
    ]
    important_names = [
        "pelvis",
        "spine_03",
        "clavicle_l",
        "clavicle_r",
        "hand_l",
        "hand_r",
        "foot_l",
        "foot_r",
        "ball_l",
        "ball_r",
    ]
    axis_bones = [i for i, name in enumerate(store.prototype.body_names) if name in important_names]
    both = torch.cat((base_pos.reshape(-1, 3), noisy_pos.reshape(-1, 3)), dim=0)
    return {
        "title": "Pose Noise Viewer",
        "sample_count": int(sample_count),
        "noise_count": int(noise_count),
        "level_count": int(level_count),
        "bone_count": int(store.J),
        "parents": store.prototype.parents_body.cpu().numpy().astype(int).tolist(),
        "bone_names": store.prototype.body_names,
        "root_index": int(store.prototype.pelvis),
        "axis_bones": axis_bones,
        "sample_labels": sample_labels,
        "levels": [float(x) for x in levels.detach().cpu().tolist()],
        "noise_recipe": {
            "pos_sigma_m_at_1": float(pos_sigma_m),
            "rot_sigma_deg_at_1": float(rot_sigma_deg),
            "pole_toe_sigma_at_1": float(scalar_sigma),
            "meaning": "viewer amount scales Gaussian noise stddev for pose vector positions, rot6 rotations, pole and toe floats",
        },
        "bounds": {
            "min": [float(x) for x in both.min(dim=0).values.detach().cpu().tolist()],
            "max": [float(x) for x in both.max(dim=0).values.detach().cpu().tolist()],
        },
        "base_pos_b64": b64_tensor(base_pos),
        "base_rot_b64": b64_tensor(base_rot),
        "noisy_pos_b64": b64_tensor(noisy_pos),
        "noisy_rot_b64": b64_tensor(noisy_rot),
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pose Noise Viewer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101216;
      --panel: #1a1d22;
      --panel2: #23272f;
      --text: #edf1f7;
      --muted: #9da6b5;
      --line: #313846;
      --grid: rgba(255,255,255,0.055);
      --gt: #72a7ff;
      --noise: #ff9b6a;
      --accent: #55d6a7;
      --x: #ff5f6d;
      --y: #52d273;
      --z: #58a6ff;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; background: var(--bg); }
    body { font: 13px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--text); }
    #app { width: 100vw; height: 100vh; display: grid; grid-template-rows: 1fr auto; }
    #viewport { position: relative; min-height: 0; }
    canvas { display: block; width: 100%; height: 100%; background: #0f1217; cursor: grab; }
    canvas:active { cursor: grabbing; }
    .hud { position: absolute; top: 12px; left: 12px; right: 12px; display: flex; gap: 8px; flex-wrap: wrap; pointer-events: none; }
    .pill { padding: 6px 8px; border-radius: 6px; background: rgba(26,29,34,0.9); border: 1px solid rgba(255,255,255,0.08); color: var(--muted); }
    .pill strong { color: var(--text); font-weight: 650; }
    .legend-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; }
    #controls { display: grid; grid-template-columns: auto auto minmax(220px, 1fr) auto auto auto auto; gap: 10px; align-items: center; padding: 10px 12px; background: var(--panel); border-top: 1px solid var(--line); }
    button, input { height: 32px; color: var(--text); background: var(--panel2); border: 1px solid var(--line); border-radius: 6px; font: inherit; }
    button { min-width: 42px; padding: 0 11px; cursor: pointer; }
    button:hover { border-color: #596273; }
    input[type="range"] { width: 100%; accent-color: var(--accent); }
    label { display: inline-flex; align-items: center; gap: 7px; color: var(--muted); }
    .num { width: 76px; padding: 0 7px; }
    @media (max-width: 820px) { #controls { grid-template-columns: auto auto 1fr; } .wide-extra { display: none; } }
  </style>
</head>
<body>
  <div id="app">
    <div id="viewport">
      <canvas id="canvas"></canvas>
      <div class="hud">
        <div class="pill"><strong>Pose Noise Viewer</strong></div>
        <div class="pill">pose <strong id="poseText">0</strong> / <span id="poseMax"></span></div>
        <div class="pill">noise <strong id="noiseText">0</strong> / <span id="noiseMax"></span></div>
        <div class="pill">amount <strong id="amountText">0.00</strong></div>
        <div class="pill">mean delta <strong id="errText">0.000</strong> m</div>
        <div class="pill"><span class="legend-dot" style="background:var(--gt)"></span>dataset pose</div>
        <div class="pill"><span class="legend-dot" style="background:var(--noise)"></span>noisy start pose</div>
        <div class="pill"><span class="legend-dot" style="background:var(--x)"></span>X <span class="legend-dot" style="background:var(--y)"></span>Y <span class="legend-dot" style="background:var(--z)"></span>Z</div>
        <div class="pill"><strong id="sampleLabel"></strong></div>
      </div>
    </div>
    <div id="controls">
      <button id="poseButton" title="Draw another pose from the dataset">Draw Pose</button>
      <button id="noiseButton" title="Sample another noise direction">Sample Noise</button>
      <label>Amount <input id="amount" type="range" min="0" max="1" value="0.25" step="0.01"></label>
      <input id="amountNum" class="num" type="number" min="0" max="1" value="0.25" step="0.01">
      <button id="mode" class="wide-extra">Overlay</button>
      <button id="labels" class="wide-extra">Labels</button>
      <button id="reset" class="wide-extra">Reset</button>
    </div>
  </div>
  <script id="payload" type="application/json">__PAYLOAD__</script>
  <script>
    const data = JSON.parse(document.getElementById("payload").textContent);
    const f32 = b64 => new Float32Array(Uint8Array.from(atob(b64), c => c.charCodeAt(0)).buffer);
    const basePos = f32(data.base_pos_b64);
    const baseRot = f32(data.base_rot_b64);
    const noisyPos = f32(data.noisy_pos_b64);
    const noisyRot = f32(data.noisy_rot_b64);
    const N = data.sample_count, S = data.noise_count, L = data.level_count, J = data.bone_count;
    const parents = data.parents, names = data.bone_names, axisBones = data.axis_bones;
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const amount = document.getElementById("amount");
    const amountNum = document.getElementById("amountNum");
    const poseText = document.getElementById("poseText");
    const noiseText = document.getElementById("noiseText");
    const amountText = document.getElementById("amountText");
    const errText = document.getElementById("errText");
    const sampleLabel = document.getElementById("sampleLabel");
    document.getElementById("poseMax").textContent = String(N - 1);
    document.getElementById("noiseMax").textContent = String(S - 1);

    let sample = 0, noise = 0, splitMode = false, showLabels = false;
    let yaw = -0.72, pitch = -0.18, zoom = 1.0, panX = 0, panY = 0;
    let dragging = false, panning = false, lastX = 0, lastY = 0;
    let dpr = 1, extent = 1;
    const boundsMin = data.bounds.min, boundsMax = data.bounds.max;
    extent = Math.max(0.5, boundsMax[0] - boundsMin[0], boundsMax[1] - boundsMin[1], boundsMax[2] - boundsMin[2]);

    function baseIndex(n, j, k) { return (n * J + j) * 3 + k; }
    function noisyIndex(n, s, l, j, k) { return ((((n * S + s) * L + l) * J + j) * 3 + k); }
    function baseRotIndex(n, j, r, c) { return ((n * J + j) * 9 + r * 3 + c); }
    function noisyRotIndex(n, s, l, j, r, c) { return (((((n * S + s) * L + l) * J + j) * 9) + r * 3 + c); }
    function currentLevels() {
      const a = Math.max(0, Math.min(1, Number(amount.value) || 0));
      const x = a * (L - 1);
      const lo = Math.floor(x);
      const hi = Math.min(L - 1, lo + 1);
      return { a, lo, hi, t: x - lo };
    }
    function lerp(a, b, t) { return a + (b - a) * t; }
    function baseP(j) { return [basePos[baseIndex(sample,j,0)], basePos[baseIndex(sample,j,1)], basePos[baseIndex(sample,j,2)]]; }
    function noisyP(j) {
      const q = currentLevels();
      return [
        lerp(noisyPos[noisyIndex(sample,noise,q.lo,j,0)], noisyPos[noisyIndex(sample,noise,q.hi,j,0)], q.t),
        lerp(noisyPos[noisyIndex(sample,noise,q.lo,j,1)], noisyPos[noisyIndex(sample,noise,q.hi,j,1)], q.t),
        lerp(noisyPos[noisyIndex(sample,noise,q.lo,j,2)], noisyPos[noisyIndex(sample,noise,q.hi,j,2)], q.t),
      ];
    }
    function baseAxis(j, r) { return [baseRot[baseRotIndex(sample,j,r,0)], baseRot[baseRotIndex(sample,j,r,1)], baseRot[baseRotIndex(sample,j,r,2)]]; }
    function noisyAxis(j, r) {
      const q = currentLevels();
      return [
        lerp(noisyRot[noisyRotIndex(sample,noise,q.lo,j,r,0)], noisyRot[noisyRotIndex(sample,noise,q.hi,j,r,0)], q.t),
        lerp(noisyRot[noisyRotIndex(sample,noise,q.lo,j,r,1)], noisyRot[noisyRotIndex(sample,noise,q.hi,j,r,1)], q.t),
        lerp(noisyRot[noisyRotIndex(sample,noise,q.lo,j,r,2)], noisyRot[noisyRotIndex(sample,noise,q.hi,j,r,2)], q.t),
      ];
    }
    function sub(a,b){ return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
    function add(a,b){ return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
    function mul(a,s){ return [a[0]*s, a[1]*s, a[2]*s]; }
    function norm(a){ return Math.hypot(a[0],a[1],a[2]); }
    function rotateProject(p, offsetX = 0) {
      let x = p[0] + offsetX, y = p[1], z = p[2];
      const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = cy*x - sy*z;
      const z1 = sy*x + cy*z;
      const y1 = cp*y - sp*z1;
      const z2 = sp*y + cp*z1;
      const s = Math.min(canvas.clientWidth, canvas.clientHeight) * 0.72 * zoom / extent;
      return { x: canvas.clientWidth * 0.5 + panX + x1*s, y: canvas.clientHeight * 0.58 + panY - y1*s, z: z2 };
    }
    function resize() {
      const ndpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      if (canvas.width !== Math.floor(rect.width * ndpr) || canvas.height !== Math.floor(rect.height * ndpr)) {
        dpr = ndpr; canvas.width = Math.floor(rect.width*dpr); canvas.height = Math.floor(rect.height*dpr);
        ctx.setTransform(dpr,0,0,dpr,0,0);
      }
    }
    function drawGrid() {
      const y = 0, size = extent * 1.8, step = 0.25;
      ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--grid");
      ctx.lineWidth = 1;
      for (let x = -size; x <= size; x += step) {
        const a = rotateProject([x,y,-size]), b = rotateProject([x,y,size]);
        ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
      }
      for (let z = -size; z <= size; z += step) {
        const a = rotateProject([-size,y,z]), b = rotateProject([size,y,z]);
        ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
      }
    }
    function drawJoint(p, color, r) { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(p.x,p.y,r,0,Math.PI*2); ctx.fill(); }
    function drawSkeleton(posFn, color, offsetX, alpha, width) {
      const points = Array.from({length:J}, (_,j)=>rotateProject(posFn(j), offsetX));
      const edges = [];
      for (let j=0;j<J;j++) { const p = parents[j]; if (p >= 0) edges.push([points[j].z + points[p].z, j, p]); }
      edges.sort((a,b)=>a[0]-b[0]);
      ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = "round"; ctx.lineJoin = "round";
      for (const [,j,p] of edges) { ctx.beginPath(); ctx.moveTo(points[p].x,points[p].y); ctx.lineTo(points[j].x,points[j].y); ctx.stroke(); }
      for (let j=0;j<J;j++) drawJoint(points[j], color, j === data.root_index ? 4.5 : 2.8);
      ctx.globalAlpha = 1;
      return points;
    }
    function drawAxes(posFn, axisFn, offsetX, alpha) {
      const colors = ["#ff5f6d", "#52d273", "#58a6ff"];
      ctx.globalAlpha = alpha; ctx.lineWidth = 2;
      const len = extent * 0.045;
      for (const j of axisBones) {
        const p0 = posFn(j);
        const s0 = rotateProject(p0, offsetX);
        for (let r=0; r<3; r++) {
          const p1 = add(p0, mul(axisFn(j,r), len));
          const s1 = rotateProject(p1, offsetX);
          ctx.strokeStyle = colors[r];
          ctx.beginPath(); ctx.moveTo(s0.x,s0.y); ctx.lineTo(s1.x,s1.y); ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;
    }
    function drawLabels(points) {
      if (!showLabels) return;
      ctx.font = "11px system-ui, sans-serif"; ctx.fillStyle = "rgba(237,241,247,0.72)";
      for (let j=0;j<J;j++) ctx.fillText(names[j], points[j].x + 5, points[j].y - 5);
    }
    function updateHud() {
      const a = Number(amount.value) || 0;
      poseText.textContent = String(sample);
      noiseText.textContent = String(noise);
      amountText.textContent = a.toFixed(2);
      sampleLabel.textContent = data.sample_labels[sample] || "";
      const recipe = data.noise_recipe;
      amountNum.value = a.toFixed(2);
      let sum = 0;
      for (let j=0;j<J;j++) sum += norm(sub(noisyP(j), baseP(j)));
      errText.textContent = (sum / J).toFixed(4);
      amount.title = `at 1.0: pos sigma ${recipe.pos_sigma_m_at_1}m, rotation sigma ${recipe.rot_sigma_deg_at_1}deg, pole/toe sigma ${recipe.pole_toe_sigma_at_1}`;
    }
    function draw() {
      resize(); ctx.clearRect(0,0,canvas.clientWidth,canvas.clientHeight); drawGrid();
      const gtColor = getComputedStyle(document.documentElement).getPropertyValue("--gt").trim();
      const nzColor = getComputedStyle(document.documentElement).getPropertyValue("--noise").trim();
      const sep = splitMode ? extent * 0.42 : 0;
      const gtPts = drawSkeleton(baseP, gtColor, -sep, splitMode ? 0.9 : 0.45, splitMode ? 4.0 : 5.0);
      const nzPts = drawSkeleton(noisyP, nzColor, sep, 0.95, 3.2);
      drawAxes(baseP, baseAxis, -sep, splitMode ? 0.62 : 0.28);
      drawAxes(noisyP, noisyAxis, sep, 0.86);
      if (!splitMode) {
        ctx.strokeStyle = "rgba(255,255,255,0.18)"; ctx.lineWidth = 1;
        for (let j=0;j<J;j+=3) { ctx.beginPath(); ctx.moveTo(gtPts[j].x,gtPts[j].y); ctx.lineTo(nzPts[j].x,nzPts[j].y); ctx.stroke(); }
      }
      drawLabels(nzPts);
      updateHud();
      requestAnimationFrame(draw);
    }
    function randomInt(n) { return Math.floor(Math.random() * n); }
    document.getElementById("poseButton").addEventListener("click", () => { sample = randomInt(N); });
    document.getElementById("noiseButton").addEventListener("click", () => { noise = randomInt(S); });
    document.getElementById("mode").addEventListener("click", (e) => { splitMode = !splitMode; e.target.textContent = splitMode ? "Split" : "Overlay"; });
    document.getElementById("labels").addEventListener("click", (e) => { showLabels = !showLabels; e.target.textContent = showLabels ? "Hide Labels" : "Labels"; });
    document.getElementById("reset").addEventListener("click", () => { yaw=-0.72; pitch=-0.18; zoom=1; panX=0; panY=0; });
    amount.addEventListener("input", () => { amountNum.value = Number(amount.value).toFixed(2); });
    amountNum.addEventListener("input", () => { amount.value = Math.max(0, Math.min(1, Number(amountNum.value) || 0)); });
    canvas.addEventListener("mousedown", e => { dragging = true; panning = e.button === 1 || e.shiftKey; lastX=e.clientX; lastY=e.clientY; });
    window.addEventListener("mouseup", () => dragging = false);
    canvas.addEventListener("mousemove", e => {
      if (!dragging) return;
      const dx=e.clientX-lastX, dy=e.clientY-lastY; lastX=e.clientX; lastY=e.clientY;
      if (panning) { panX += dx; panY += dy; }
      else { yaw += dx*0.006; pitch = Math.max(-1.35, Math.min(1.35, pitch + dy*0.006)); }
    });
    canvas.addEventListener("wheel", e => { e.preventDefault(); zoom = Math.max(0.15, Math.min(8, zoom * Math.exp(-e.deltaY*0.001))); }, {passive:false});
    window.addEventListener("keydown", e => {
      if (e.key === "p") sample = randomInt(N);
      if (e.key === "n") noise = randomInt(S);
    });
    requestAnimationFrame(draw);
  </script>
</body>
</html>
"""


def write_html(payload: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    text = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an interactive IK pose-noise visualizer.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--noises", type=int, default=12)
    parser.add_argument("--levels", type=int, default=9)
    parser.add_argument("--pos-sigma-m", type=float, default=0.12)
    parser.add_argument("--rot-sigma-deg", type=float, default=25.0)
    parser.add_argument("--scalar-sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = build_payload(
        sample_count=args.samples,
        noise_count=args.noises,
        level_count=args.levels,
        pos_sigma_m=args.pos_sigma_m,
        rot_sigma_deg=args.rot_sigma_deg,
        scalar_sigma=args.scalar_sigma,
        seed=args.seed,
        device=device,
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    write_html(payload, output)
    print(json.dumps({"output": str(output), "samples": payload["sample_count"], "noises": payload["noise_count"]}, indent=2))


if __name__ == "__main__":
    main()
