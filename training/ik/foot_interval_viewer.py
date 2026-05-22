from __future__ import annotations

import json
from pathlib import Path

import torch

try:
    from .bootstrap import PROJECT_ROOT, ensure_paths
    from . import foot_harshness_audit as audit
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    import foot_harshness_audit as audit

ensure_paths()

OUTPUT = PROJECT_ROOT / "training" / "runs" / "model_comparisons" / "foot_pin_interval_audit.html"


def series_for_foot(pos: torch.Tensor, rot: torch.Tensor, foot_i: int) -> dict[str, list[float]]:
    slide, rot_deg, height = audit.foot_motion_series(pos, rot, foot_i)
    foot_pos = pos[:, foot_i]
    return {
        "slide": [float(x) for x in slide.detach().cpu()],
        "rot": [float(x) for x in rot_deg.detach().cpu()],
        "height": [float(x) for x in height.detach().cpu()],
        "x": [float(x) for x in foot_pos[:, 0].detach().cpu()],
        "z": [float(x) for x in foot_pos[:, 2].detach().cpu()],
    }


def build_payload(checkpoint: Path) -> dict:
    device = audit.make_device()
    model, store, cfg = audit.load_controller(checkpoint, device)
    gt_pos, gt_rot = audit.gt_sequence(store)
    pred_pos, pred_rot = audit.rollout_sequence(model, store, cfg)
    metric = audit.pinned_metric(gt_pos, gt_rot, pred_pos, pred_rot, store.prototype)
    indices = audit.foot_indices(store.prototype)
    feet = {}
    for foot_name, foot_i in indices.items():
        gt_slide, gt_rot_deg, gt_height = audit.foot_motion_series(gt_pos, gt_rot, foot_i)
        pred_slide, pred_rot_deg, pred_height = audit.foot_motion_series(pred_pos, pred_rot, foot_i)
        gt_interval = tuple(metric[foot_name]["gt_interval"])
        pred_interval = tuple(metric[foot_name]["pred_interval"])
        feet[foot_name] = {
            "metric": metric[foot_name],
            "gt": series_for_foot(gt_pos, gt_rot, foot_i),
            "pred": series_for_foot(pred_pos, pred_rot, foot_i),
            "gtIntervalScore": audit.interval_score(gt_slide, gt_rot_deg, gt_height, *gt_interval),
            "predIntervalScore": audit.interval_score(pred_slide, pred_rot_deg, pred_height, *pred_interval),
        }
    return {
        "checkpoint": str(checkpoint),
        "frameCount": int(gt_pos.shape[0]),
        "metric": metric,
        "feet": feet,
    }


def write_html(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload)
    output.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Foot Pin Interval Audit</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #101318;
      --panel: #171c23;
      --grid: #2b3542;
      --text: #e8eef7;
      --muted: #97a6b8;
      --gt: #68d391;
      --pred: #63b3ed;
      --gtBand: rgba(104, 211, 145, 0.18);
      --predBand: rgba(99, 179, 237, 0.18);
      --mid: rgba(255, 214, 102, 0.28);
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, Segoe UI, sans-serif;
    }}
    header {{
      padding: 18px 22px 10px;
      border-bottom: 1px solid #26303b;
    }}
    h1 {{
      font-size: 18px;
      margin: 0 0 8px;
      letter-spacing: 0;
    }}
    .sub {{ color: var(--muted); font-size: 12px; }}
    main {{ padding: 16px 22px 32px; display: grid; gap: 18px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(130px, 1fr));
      gap: 10px;
    }}
    .stat, .foot {{
      background: var(--panel);
      border: 1px solid #26303b;
      border-radius: 6px;
      padding: 12px;
    }}
    .label {{ color: var(--muted); font-size: 12px; }}
    .value {{ font-size: 19px; margin-top: 3px; }}
    .foot {{ display: grid; gap: 12px; }}
    .foot h2 {{ margin: 0; font-size: 16px; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    svg {{ width: 100%; height: 230px; display: block; background: #0d1117; border: 1px solid #26303b; border-radius: 6px; }}
    .legend {{ display: flex; gap: 16px; color: var(--muted); font-size: 12px; align-items: center; flex-wrap: wrap; }}
    .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: -1px; }}
    .gt {{ background: var(--gt); }} .pred {{ background: var(--pred); }}
    .note {{ color: var(--muted); font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #26303b; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <header>
    <h1>Foot Pin Interval Audit</h1>
    <div class="sub" id="checkpoint"></div>
  </header>
  <main>
    <section class="summary" id="summary"></section>
    <section class="note">
      Wide translucent bands are detected planted intervals. Those full intervals are now the scored windows.
      GT and prediction choose their intervals separately, so phase drift is not scored as foot sliding.
    </section>
    <section id="feet"></section>
  </main>
  <script>
    const DATA = {data};
    const W = 760, H = 230, P = {{l: 42, r: 16, t: 18, b: 32}};
    const colors = getComputedStyle(document.documentElement);
    document.getElementById("checkpoint").textContent = DATA.checkpoint;

    function fmt(x, n=3) {{ return Number(x).toFixed(n); }}
    function summaryCard(label, value) {{
      return `<div class="stat"><div class="label">${{label}}</div><div class="value">${{value}}</div></div>`;
    }}
    document.getElementById("summary").innerHTML = [
      summaryCard("Mean Slide Ratio", fmt(DATA.metric.mean_slide_ratio, 2) + "x"),
      summaryCard("Mean Rot Ratio", fmt(DATA.metric.mean_rot_ratio, 2) + "x"),
      summaryCard("Combined Score", fmt(DATA.metric.score, 2)),
      summaryCard("Frames", DATA.frameCount)
    ].join("");

    function scale(values, minOverride=null, maxOverride=null) {{
      const min = minOverride ?? Math.min(...values);
      const max = maxOverride ?? Math.max(...values);
      const span = Math.max(1e-8, max - min);
      return v => P.t + (H - P.t - P.b) * (1 - (v - min) / span);
    }}
    function xScale(n) {{
      return i => P.l + (W - P.l - P.r) * i / Math.max(1, n - 1);
    }}
    function poly(values, sy) {{
      const sx = xScale(values.length);
      return values.map((v, i) => `${{sx(i).toFixed(1)}},${{sy(v).toFixed(1)}}`).join(" ");
    }}
    function rect(interval, color, n, y=0, h=H) {{
      const sx = xScale(n);
      const x = sx(interval[0]);
      const w = Math.max(1, sx(interval[1]) - sx(interval[0]));
      return `<rect x="${{x}}" y="${{y}}" width="${{w}}" height="${{h}}" fill="${{color}}" />`;
    }}
    function chart(title, gt, pred, gtInterval, predInterval, gtMid, predMid, unit) {{
      const all = gt.concat(pred);
      const sy = scale(all, 0, Math.max(...all) * 1.1 + 1e-6);
      const n = gt.length;
      const grid = [0, .25, .5, .75, 1].map(f => {{
        const y = P.t + (H-P.t-P.b)*f;
        return `<line x1="${{P.l}}" y1="${{y}}" x2="${{W-P.r}}" y2="${{y}}" stroke="var(--grid)" />`;
      }}).join("");
      return `<div>
        <svg viewBox="0 0 ${{W}} ${{H}}">
          ${{rect(gtInterval, "var(--gtBand)", n)}}
          ${{rect(predInterval, "var(--predBand)", n)}}
          ${{grid}}
          <text x="12" y="16" fill="var(--text)" font-size="12">${{title}} (${{unit}})</text>
          <polyline points="${{poly(gt, sy)}}" fill="none" stroke="var(--gt)" stroke-width="2"/>
          <polyline points="${{poly(pred, sy)}}" fill="none" stroke="var(--pred)" stroke-width="2"/>
          <text x="${{P.l}}" y="${{H-8}}" fill="var(--muted)" font-size="11">frame</text>
        </svg>
      </div>`;
    }}
    function pathChart(title, gt, pred, metric) {{
      const xs = gt.x.concat(pred.x), zs = gt.z.concat(pred.z);
      const sx = scale(xs, Math.min(...xs), Math.max(...xs));
      const sz = scale(zs, Math.min(...zs), Math.max(...zs));
      const pts = (s) => s.x.map((x, i) => `${{P.l + (W-P.l-P.r)*(x-Math.min(...xs))/(Math.max(...xs)-Math.min(...xs)+1e-8)}},${{P.t + (H-P.t-P.b)*(1-(s.z[i]-Math.min(...zs))/(Math.max(...zs)-Math.min(...zs)+1e-8))}}`).join(" ");
      const mark = (s, mid, color) => {{
        const out = [];
        for (let i = mid[0]; i <= mid[1] && i < s.x.length; i++) {{
          const x = P.l + (W-P.l-P.r)*(s.x[i]-Math.min(...xs))/(Math.max(...xs)-Math.min(...xs)+1e-8);
          const y = P.t + (H-P.t-P.b)*(1-(s.z[i]-Math.min(...zs))/(Math.max(...zs)-Math.min(...zs)+1e-8));
          out.push(`<circle cx="${{x}}" cy="${{y}}" r="3" fill="${{color}}" />`);
        }}
        return out.join("");
      }};
      return `<svg viewBox="0 0 ${{W}} ${{H}}">
        <text x="12" y="16" fill="var(--text)" font-size="12">${{title}} foot path x/z</text>
        <polyline points="${{pts(gt)}}" fill="none" stroke="var(--gt)" stroke-width="2"/>
        <polyline points="${{pts(pred)}}" fill="none" stroke="var(--pred)" stroke-width="2"/>
        ${{mark(gt, metric.gt_scored_interval, "var(--gt)")}}
        ${{mark(pred, metric.pred_scored_interval, "var(--pred)")}}
      </svg>`;
    }}

    const feetEl = document.getElementById("feet");
    feetEl.innerHTML = Object.entries(DATA.feet).map(([name, foot]) => {{
      const m = foot.metric;
      return `<article class="foot">
        <h2>${{name}}</h2>
        <div class="legend">
          <span><span class="dot gt"></span>GT</span>
          <span><span class="dot pred"></span>Prediction</span>
          <span>GT interval ${{m.gt_interval[0]}}-${{m.gt_interval[1]}}</span>
          <span>Pred interval ${{m.pred_interval[0]}}-${{m.pred_interval[1]}}</span>
        </div>
        <table>
          <tr><th>Metric</th><th>GT middle</th><th>Pred middle</th><th>Ratio</th></tr>
          <tr><td>slide m/frame</td><td>${{fmt(m.gt_slide_m_per_frame, 5)}}</td><td>${{fmt(m.pred_slide_m_per_frame, 5)}}</td><td>${{fmt(m.slide_ratio, 2)}}x</td></tr>
          <tr><td>rotation deg/frame</td><td>${{fmt(m.gt_rot_deg_per_frame, 3)}}</td><td>${{fmt(m.pred_rot_deg_per_frame, 3)}}</td><td>${{fmt(m.rot_ratio, 2)}}x</td></tr>
        </table>
        <div class="grid2">
          ${{chart("Slide", foot.gt.slide, foot.pred.slide, m.gt_interval, m.pred_interval, m.gt_scored_interval, m.pred_scored_interval, "m/frame")}}
          ${{chart("Rotation", foot.gt.rot, foot.pred.rot, m.gt_interval, m.pred_interval, m.gt_scored_interval, m.pred_scored_interval, "deg/frame")}}
        </div>
        ${{pathChart(name, foot.gt, foot.pred, m)}}
      </article>`;
    }}).join("");
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    payload = build_payload(audit.BASE_CONTROLLER)
    write_html(payload, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
