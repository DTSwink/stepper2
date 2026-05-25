from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .bootstrap import PROJECT_ROOT, ensure_paths
    from . import ik_core as tl
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    import ik_core as tl

ensure_paths()


RUNS_DIR = PROJECT_ROOT / "training" / "runs"
DEFAULT_NPZ = PROJECT_ROOT / "ue5" / "animations_omni_only_full" / "npz_final" / "M_Neutral_Walk_Loop_F.npz"
DEFAULT_OUTPUT = RUNS_DIR / "model_comparisons" / "ik_speed_limit_tuner.html"


def _rows(tensor) -> list:
    return tensor.detach().cpu().float().tolist()


def _flat_rows(tensor) -> list:
    return tensor.detach().cpu().float().reshape(tensor.shape[0], -1).tolist()


def build_payload(npz_path: Path, frame: int) -> dict[str, object]:
    cfg = tl.TrainConfig()
    cfg.pose_representation = tl.IK_POSE_REPRESENTATION
    cfg.live_viewer = False
    cfg.visual_reporter = False
    clip = tl.MotionClip(npz_path, cfg, cyclic_animation=True)
    frame = max(0, min(int(frame), clip.T - 1))
    root_rel_pos = clip.root_relative_pos[frame]

    limb_payload: list[dict[str, object]] = []
    for limb_i, spec in enumerate(clip.ik_limb_specs):
        start = int(spec["start"])
        mid = int(spec["mid"])
        end = int(spec["end"])
        toe = spec.get("toe")
        toe_i = int(toe) if toe is not None else -1
        base = root_rel_pos[start]
        mid_pos = root_rel_pos[mid]
        end_pos = root_rel_pos[end]
        raw_axis = end_pos - base
        axis = raw_axis / raw_axis.norm().clamp_min(1e-6)
        pole = mid_pos - base
        pole = pole - axis * (pole * axis).sum()
        pole = pole / pole.norm().clamp_min(1e-6)
        toe_offset = root_rel_pos[toe_i] - end_pos if toe_i >= 0 else end_pos.new_zeros(3)
        limb_payload.append(
            {
                "kind": str(spec["kind"]),
                "side": str(spec["side"]),
                "start": start,
                "mid": mid,
                "end": end,
                "toe": toe_i,
                "lengths": [float(v) for v in clip.ik_limb_lengths[limb_i].detach().cpu().tolist()],
                "initial_pole": [float(v) for v in pole.detach().cpu().tolist()],
                "toe_offset": [float(v) for v in toe_offset.detach().cpu().tolist()],
            }
        )

    return {
        "title": "IK Speed Limit Tuner",
        "clip": str(npz_path),
        "clip_name": npz_path.stem,
        "fps": float(clip.fps),
        "frame": int(frame),
        "bone_names": clip.body_names,
        "parents": [int(v) for v in clip.parents_body_list],
        "pelvis": int(clip.pelvis),
        "core_indices": [int(v) for v in clip.core_non_pelvis],
        "limbs": limb_payload,
        "local_offsets": _rows(clip.local_offsets),
        "initial_local_rot": _flat_rows(clip.local_rot[frame]),
        "initial_pelvis_pos": [float(v) for v in clip.pelvis_local_pos[frame].detach().cpu().tolist()],
        "initial_root_relative_pos": _rows(root_rel_pos),
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IK Speed Limit Tuner</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101216;
      --panel: #1a1d22;
      --panel2: #23272f;
      --text: #edf1f7;
      --muted: #9da6b5;
      --line: #313846;
      --grid: rgba(255,255,255,0.065);
      --initial: #72a7ff;
      --pose: #ff9b6a;
      --accent: #55d6a7;
      --limit: #e6df83;
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
    .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; }
    #controls {
      display: grid;
      grid-template-columns: auto auto auto auto minmax(150px, 1fr) minmax(150px, 1fr) minmax(150px, 1fr) minmax(150px, 1fr) auto;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
      background: var(--panel);
      border-top: 1px solid var(--line);
    }
    button, input { height: 32px; color: var(--text); background: var(--panel2); border: 1px solid var(--line); border-radius: 6px; font: inherit; }
    button { padding: 0 11px; cursor: pointer; white-space: nowrap; }
    button:hover { border-color: #596273; }
    input[type="range"] { width: 100%; accent-color: var(--accent); }
    label { display: grid; grid-template-columns: auto 1fr auto; align-items: center; gap: 7px; color: var(--muted); min-width: 0; }
    .value { color: var(--text); font-variant-numeric: tabular-nums; min-width: 48px; text-align: right; }
    .small { width: 66px; padding: 0 7px; }
    @media (max-width: 920px) {
      #controls { grid-template-columns: auto auto auto; }
      label { grid-column: span 3; }
    }
  </style>
</head>
<body>
  <div id="app">
    <div id="viewport">
      <canvas id="canvas"></canvas>
      <div class="hud">
        <div class="pill"><strong id="title"></strong></div>
        <div class="pill">source <strong id="clip"></strong></div>
        <div class="pill">tick <strong id="tick">0</strong> @ <span id="fps"></span> FPS</div>
        <div class="pill"><span class="dot" style="background:var(--initial)"></span>reset pose</div>
        <div class="pill"><span class="dot" style="background:var(--pose)"></span>rate-limited jiggle</div>
        <div class="pill">reach clamps <strong id="clamps">0</strong></div>
      </div>
    </div>
    <div id="controls">
      <button id="play">Pause</button>
      <button id="resetPose">Reset Pose</button>
      <button id="newDirs">New Random</button>
      <label>Speed <input id="speed" class="small" type="number" min="0.1" max="4" step="0.1" value="1"></label>
      <label>End eff m/s <input id="ee" type="range" min="0" max="6" step="0.01" value="4"><span class="value" id="eeV"></span></label>
      <label>Pelvis m/s <input id="pelvisV" type="range" min="0" max="4" step="0.01" value="2"><span class="value" id="pelvisVV"></span></label>
      <label>Pelvis deg/s <input id="pelvisW" type="range" min="0" max="720" step="1" value="360"><span class="value" id="pelvisWV"></span></label>
      <label>Core deg/s <input id="coreW" type="range" min="0" max="720" step="1" value="360"><span class="value" id="coreWV"></span></label>
      <button id="resetCam">Reset View</button>
    </div>
  </div>
  <script id="payload" type="application/json">__PAYLOAD__</script>
  <script>
    const data = JSON.parse(document.getElementById("payload").textContent);
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const titleEl = document.getElementById("title");
    const clipEl = document.getElementById("clip");
    const fpsEl = document.getElementById("fps");
    const tickEl = document.getElementById("tick");
    const clampsEl = document.getElementById("clamps");
    const playBtn = document.getElementById("play");
    const resetPoseBtn = document.getElementById("resetPose");
    const newDirsBtn = document.getElementById("newDirs");
    const resetCamBtn = document.getElementById("resetCam");
    const speedInput = document.getElementById("speed");
    const eeInput = document.getElementById("ee");
    const pelvisVInput = document.getElementById("pelvisV");
    const pelvisWInput = document.getElementById("pelvisW");
    const coreWInput = document.getElementById("coreW");
    const eeV = document.getElementById("eeV");
    const pelvisVV = document.getElementById("pelvisVV");
    const pelvisWV = document.getElementById("pelvisWV");
    const coreWV = document.getElementById("coreWV");

    const J = data.bone_names.length;
    const ID = [1,0,0, 0,1,0, 0,0,1];
    const coreSet = new Set(data.core_indices);
    const limbEndToIndex = new Map(data.limbs.map((l, i) => [l.end, i]));
    const initialPos = data.initial_root_relative_pos.map(p => p.slice());
    const frameDt = 1 / Math.max(1, data.fps || 60);
    let yaw = -0.72;
    let pitch = -0.18;
    let zoom = 1.2;
    let panX = 0;
    let panY = 0;
    let dragging = false;
    let panning = false;
    let lastMouse = [0, 0];
    let playing = true;
    let tick = 0;
    let clampCount = 0;
    let acc = 0;
    let lastTs = 0;
    let dirClock = 0;
    const directionInterval = 0.18;

    titleEl.textContent = data.title;
    clipEl.textContent = `${data.clip_name} / frame ${data.frame}`;
    fpsEl.textContent = (data.fps || 60).toFixed(1);

    function clone3(v) { return [v[0], v[1], v[2]]; }
    function add(a,b) { return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
    function sub(a,b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
    function mul(v,s) { return [v[0]*s, v[1]*s, v[2]*s]; }
    function dot(a,b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
    function cross(a,b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
    function len(v) { return Math.hypot(v[0], v[1], v[2]); }
    function norm(v) {
      const l = len(v);
      if (l < 1e-8) return [1, 0, 0];
      return [v[0] / l, v[1] / l, v[2] / l];
    }
    function randomDir() {
      let v = [Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1];
      return norm(v);
    }
    function vecMat(v, m) {
      return [
        v[0]*m[0] + v[1]*m[3] + v[2]*m[6],
        v[0]*m[1] + v[1]*m[4] + v[2]*m[7],
        v[0]*m[2] + v[1]*m[5] + v[2]*m[8],
      ];
    }
    function matMul(a, b) {
      return [
        a[0]*b[0]+a[1]*b[3]+a[2]*b[6], a[0]*b[1]+a[1]*b[4]+a[2]*b[7], a[0]*b[2]+a[1]*b[5]+a[2]*b[8],
        a[3]*b[0]+a[4]*b[3]+a[5]*b[6], a[3]*b[1]+a[4]*b[4]+a[5]*b[7], a[3]*b[2]+a[4]*b[5]+a[5]*b[8],
        a[6]*b[0]+a[7]*b[3]+a[8]*b[6], a[6]*b[1]+a[7]*b[4]+a[8]*b[7], a[6]*b[2]+a[7]*b[5]+a[8]*b[8],
      ];
    }
    function axisAngleRow(axis, angle) {
      const a = norm(axis);
      const x = a[0], y = a[1], z = a[2];
      const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
      return [
        t*x*x+c,     t*x*y+s*z, t*x*z-s*y,
        t*x*y-s*z,   t*y*y+c,   t*y*z+s*x,
        t*x*z+s*y,   t*y*z-s*x, t*z*z+c,
      ];
    }
    function projectToPlane(v, axis) {
      const n = norm(axis);
      return norm(sub(v, mul(n, dot(v, n))));
    }
    function stablePole(axis) {
      const ref = Math.abs(axis[1]) < 0.8 ? [0, 1, 0] : [1, 0, 0];
      return norm(cross(axis, ref));
    }

    function makeState() {
      return {
        pelvisPos: clone3(data.initial_pelvis_pos),
        localRot: data.initial_local_rot.map(r => r.slice()),
        eeTargets: data.limbs.map(l => clone3(initialPos[l.end])),
        dirs: null,
      };
    }
    let state = makeState();

    function newDirections() {
      state.dirs = {
        pelvisV: randomDir(),
        pelvisW: randomDir(),
        coreW: data.core_indices.map(() => randomDir()),
        ee: data.limbs.map(() => randomDir()),
      };
      dirClock = 0;
    }
    newDirections();

    function resetPose() {
      state = makeState();
      newDirections();
      tick = 0;
      clampCount = 0;
      acc = 0;
      dirClock = 0;
    }

    function computePose(mutating) {
      const pos = Array.from({length: J}, () => [0,0,0]);
      const rot = Array.from({length: J}, () => ID.slice());
      const offsets = data.local_offsets.map(o => o.slice());
      offsets[data.pelvis] = clone3(state.pelvisPos);
      for (let j = 0; j < J; j++) {
        const parent = data.parents[j];
        let localRot = ID;
        if (j === data.pelvis || coreSet.has(j)) localRot = state.localRot[j];
        if (parent < 0) {
          rot[j] = localRot.slice();
          pos[j] = offsets[j].slice();
        } else {
          rot[j] = matMul(localRot, rot[parent]);
          pos[j] = add(vecMat(offsets[j], rot[parent]), pos[parent]);
        }
      }
      for (let li = 0; li < data.limbs.length; li++) {
        const limb = data.limbs[li];
        const base = pos[limb.start];
        let end = state.eeTargets[li].slice();
        const l1 = limb.lengths[0];
        const l2 = limb.lengths[1];
        const maxReach = Math.max(1e-5, l1 + l2 - 1e-4);
        const minReach = Math.max(1e-5, Math.abs(l1 - l2) + 1e-4);
        let raw = sub(end, base);
        let dRaw = len(raw);
        if (dRaw > maxReach) {
          end = add(base, mul(norm(raw), maxReach));
          if (mutating) {
            state.eeTargets[li] = end.slice();
            clampCount += 1;
          }
          raw = sub(end, base);
          dRaw = len(raw);
        }
        const axis = norm(raw);
        const d = Math.max(minReach, Math.min(maxReach, dRaw));
        let pole = projectToPlane(limb.initial_pole, axis);
        if (len(pole) < 1e-5) pole = stablePole(axis);
        const a = (l1*l1 - l2*l2 + d*d) / (2 * d);
        const h = Math.sqrt(Math.max(l1*l1 - a*a, 0));
        const mid = add(add(base, mul(axis, a)), mul(pole, h));
        pos[limb.mid] = mid;
        pos[limb.end] = end;
        if (limb.toe >= 0) pos[limb.toe] = add(end, limb.toe_offset);
      }
      return pos;
    }

    function step(dt) {
      if (!state.dirs) newDirections();
      const eeSpeed = Number(eeInput.value);
      const pelvisSpeed = Number(pelvisVInput.value);
      const pelvisAng = Number(pelvisWInput.value) * Math.PI / 180;
      const coreAng = Number(coreWInput.value) * Math.PI / 180;
      state.pelvisPos = add(state.pelvisPos, mul(state.dirs.pelvisV, pelvisSpeed * dt));
      if (pelvisAng > 0) state.localRot[data.pelvis] = matMul(axisAngleRow(state.dirs.pelvisW, pelvisAng * dt), state.localRot[data.pelvis]);
      for (let i = 0; i < data.core_indices.length; i++) {
        const bone = data.core_indices[i];
        if (coreAng > 0) state.localRot[bone] = matMul(axisAngleRow(state.dirs.coreW[i], coreAng * dt), state.localRot[bone]);
      }
      for (let i = 0; i < data.limbs.length; i++) {
        state.eeTargets[i] = add(state.eeTargets[i], mul(state.dirs.ee[i], eeSpeed * dt));
      }
      computePose(true);
      tick += 1;
      dirClock += dt;
      if (dirClock >= directionInterval) newDirections();
    }

    function resize() {
      const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
      const w = canvas.clientWidth || window.innerWidth;
      const h = canvas.clientHeight || window.innerHeight;
      canvas.width = Math.max(1, Math.floor(w * dpr));
      canvas.height = Math.max(1, Math.floor(h * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    window.addEventListener("resize", resize);

    function cameraPoint(p) {
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = p[0] * cy - p[2] * sy;
      const z1 = p[0] * sy + p[2] * cy;
      const y1 = p[1] * cp - z1 * sp;
      const z2 = p[1] * sp + z1 * cp;
      const scale = 145 * zoom / Math.max(0.35, 1 + z2 * 0.09);
      return [canvas.clientWidth * 0.5 + panX + x1 * scale, canvas.clientHeight * 0.61 + panY - y1 * scale, z2];
    }
    function drawLine(a, b, color, width, alpha = 1) {
      const pa = cameraPoint(a), pb = cameraPoint(b);
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(pa[0], pa[1]);
      ctx.lineTo(pb[0], pb[1]);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    function drawJoint(p, color, r, alpha = 1) {
      const pp = cameraPoint(p);
      ctx.globalAlpha = alpha;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(pp[0], pp[1], r, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    }
    function drawGrid() {
      const span = 2.5;
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.lineWidth = 1;
      for (let i = -10; i <= 10; i++) {
        drawLine([-span, 0, i * span / 10], [span, 0, i * span / 10], "rgba(255,255,255,0.055)", 1, 1);
        drawLine([i * span / 10, 0, -span], [i * span / 10, 0, span], "rgba(255,255,255,0.055)", 1, 1);
      }
    }
    function boneColor(j) {
      if (j === data.pelvis) return "var(--limit)";
      if (coreSet.has(j)) return "#d7dee9";
      if (limbEndToIndex.has(j)) return "var(--accent)";
      return "var(--pose)";
    }
    function drawSkeleton(pos, baseColor, width, alpha) {
      for (let j = 0; j < J; j++) {
        const parent = data.parents[j];
        if (parent >= 0) drawLine(pos[parent], pos[j], baseColor || boneColor(j), width, alpha);
      }
      for (let j = 0; j < J; j++) {
        if (j === data.pelvis || limbEndToIndex.has(j)) drawJoint(pos[j], baseColor || boneColor(j), 4.2, alpha);
      }
    }
    function drawEffectorTargets() {
      for (let i = 0; i < data.limbs.length; i++) {
        const target = state.eeTargets[i];
        const color = data.limbs[i].kind === "leg" ? "#72a7ff" : "#55d6a7";
        drawJoint(target, color, 5.5, 0.9);
      }
    }
    function draw() {
      resize();
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      drawGrid();
      const current = computePose(false);
      drawSkeleton(initialPos, "var(--initial)", 2, 0.28);
      drawSkeleton(current, null, 3, 0.95);
      drawEffectorTargets();
      eeV.textContent = Number(eeInput.value).toFixed(2);
      pelvisVV.textContent = Number(pelvisVInput.value).toFixed(2);
      pelvisWV.textContent = Math.round(Number(pelvisWInput.value));
      coreWV.textContent = Math.round(Number(coreWInput.value));
      tickEl.textContent = String(tick);
      clampsEl.textContent = String(clampCount);
    }
    function animate(ts) {
      if (!lastTs) lastTs = ts;
      const speed = Math.max(0.05, Number(speedInput.value) || 1);
      const realDt = Math.min(0.5, Math.max(0, (ts - lastTs) / 1000));
      lastTs = ts;
      if (playing) {
        acc += realDt * speed;
        while (acc >= frameDt) {
          step(frameDt);
          acc -= frameDt;
        }
      }
      draw();
    }

    playBtn.addEventListener("click", () => {
      playing = !playing;
      playBtn.textContent = playing ? "Pause" : "Play";
    });
    resetPoseBtn.addEventListener("click", resetPose);
    newDirsBtn.addEventListener("click", newDirections);
    resetCamBtn.addEventListener("click", () => { yaw = -0.72; pitch = -0.18; zoom = 1.2; panX = 0; panY = 0; });
    for (const input of [eeInput, pelvisVInput, pelvisWInput, coreWInput]) input.addEventListener("input", draw);

    canvas.addEventListener("mousedown", (e) => {
      dragging = true;
      panning = e.shiftKey || e.button === 1 || e.button === 2;
      lastMouse = [e.clientX, e.clientY];
    });
    window.addEventListener("mouseup", () => { dragging = false; });
    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      const dx = e.clientX - lastMouse[0], dy = e.clientY - lastMouse[1];
      lastMouse = [e.clientX, e.clientY];
      if (panning) {
        panX += dx;
        panY += dy;
      } else {
        yaw += dx * 0.008;
        pitch = Math.max(-1.25, Math.min(1.05, pitch + dy * 0.006));
      }
    });
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      zoom = Math.max(0.25, Math.min(5, zoom * Math.exp(-e.deltaY * 0.001)));
    }, { passive: false });
    canvas.addEventListener("contextmenu", e => e.preventDefault());

    resize();
    draw();
    window.setInterval(() => animate(performance.now()), 1000 / 60);
  </script>
</body>
</html>
"""


def write_viewer(npz_path: Path, output: Path, frame: int) -> Path:
    payload = build_payload(npz_path, frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    html = HTML.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    output.write_text(html, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an IK rate-limit visual tuner.")
    parser.add_argument("--npz", default=str(DEFAULT_NPZ))
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output = write_viewer(Path(args.npz).resolve(), Path(args.output).resolve(), args.frame)
    print(output)


if __name__ == "__main__":
    main()
