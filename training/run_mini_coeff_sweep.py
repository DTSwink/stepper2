from __future__ import annotations

import csv
import re
import subprocess
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".tools" / "python310" / "python.exe"
RUNS = ROOT / "training" / "runs"
LOGS = RUNS / "launch_logs"

BASELINE = RUNS / "20260516_044547_hybrid_canonbasis_from_bestk32_constantk32_w010_resumeopt" / "checkpoints" / "checkpoint_best_k32.pt"
PRIOR_1F = RUNS / "20260517_161820_denoise_rootlook1_lat32_n0p05_e360" / "checkpoints" / "checkpoint_best.pt"
PRIOR_COMPAT = RUNS / "20260517_215153_denoise_rootlook16_compatfixed_mini_e240" / "checkpoints" / "checkpoint_best.pt"
PERIODIC = RUNS / "mini_datasets" / "walkF_plus_stand45" / "periodic"
NONPERIODIC = RUNS / "mini_datasets" / "walkF_plus_stand45" / "nonperiodic"
R45 = NONPERIODIC / "M_Neutral_Stand_Turn_045_R.npz"
L45 = NONPERIODIC / "M_Neutral_Stand_Turn_045_L.npz"
WALKF = PERIODIC / "M_Neutral_Walk_Loop_F.npz"


COMBOS = [
    ("ew015_cw005", 0.15, 0.05),
    ("ew030_cw005", 0.30, 0.05),
    ("ew030_cw010", 0.30, 0.10),
    ("ew030_cw020", 0.30, 0.20),
    ("ew060_cw010", 0.60, 0.10),
    ("ew100_cw010", 1.00, 0.10),
]


def run(cmd: list[str], log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with code {proc.returncode}: {' '.join(cmd)}\n{proc.stdout[-4000:]}")
    return proc.stdout


def parse_metric(text: str, name: str) -> float:
    match = re.search(rf"{re.escape(name)}\s+([-+0-9.eE]+)", text)
    if not match:
        raise ValueError(f"missing metric {name!r} in output:\n{text}")
    return float(match.group(1))


def checkpoint_epoch(path: Path) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return int(ckpt.get("epoch", -1)), float(ckpt.get("best_val", float("nan")))


def parse_elapsed_for_epoch(log_text: str, epoch: int) -> float:
    pattern = rf"epoch={epoch:04d}.*?elapsed_s=([-+0-9.eE]+)"
    match = re.search(pattern, log_text)
    if not match:
        return float("nan")
    return float(match.group(1))


def evaluate(run_name: str, checkpoint: Path, clip: Path, side: str) -> dict[str, float]:
    out_html = RUNS / "model_comparisons" / f"{run_name}_{side}.html"
    cmd = [
        str(PYTHON),
        "training/visualize_model.py",
        "--npz-path",
        str(clip),
        "--checkpoint-path",
        str(checkpoint),
        "--output-path",
        str(out_html),
        "--device",
        "cpu",
        "--max-frames",
        "50",
    ]
    text = run(cmd, LOGS / f"{run_name}_{side}_eval.log")
    return {
        f"{side}_ar_avg": parse_metric(text, "autoregressive_mean_joint_error_avg"),
        f"{side}_ar_end": parse_metric(text, "autoregressive_mean_joint_error_end"),
        f"{side}_ar_max": parse_metric(text, "autoregressive_mean_joint_error_max"),
        f"{side}_tf_avg": parse_metric(text, "one_step_mean_joint_error_avg"),
    }


def main() -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    summary_dir = RUNS / "coeff_sweeps" / f"{stamp}_mini_doubleae"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for label, extra_weight, compat_weight in COMBOS:
        run_name = f"{stamp}_sweep_{label}"
        print(f"\n=== {run_name} extra={extra_weight} compat={compat_weight} ===", flush=True)
        cmd = [
            str(PYTHON),
            "training/train_locomotion_ae_prior.py",
            "--periodic-folder-path",
            str(PERIODIC),
            "--nonperiodic-folder-path",
            str(NONPERIODIC),
            "--prior-checkpoint",
            str(PRIOR_1F),
            "--prior-weight",
            "1.0",
            "--extra-prior-checkpoint",
            str(PRIOR_COMPAT),
            "--extra-prior-weight",
            str(extra_weight),
            "--compatibility-score-weight",
            str(compat_weight),
            "--resume-checkpoint",
            str(BASELINE),
            "--run-name",
            run_name,
            "--device",
            "cuda",
            "--hidden-dim",
            "512",
            "--num-hidden-layers",
            "2",
            "--learning-rate",
            "5e-5",
            "--max-epochs",
            "170",
            "--rollout-schedule",
            "8,16,32",
            "--initial-rollout-k",
            "8",
            "--curriculum-min-epochs",
            "15",
            "--curriculum-stall-patience-epochs",
            "20",
            "--curriculum-max-epochs-per-stage",
            "45",
            "--training-loop",
            "agents",
            "--agent-sampling",
            "random",
            "--agent-batches-per-epoch",
            "1",
            "--gradient-accumulation-batches",
            "1",
            "--periodic-sampling-weight",
            "1",
            "--nonperiodic-sampling-weight",
            "1",
            "--agent-min-cohort-steps",
            "8",
            "--no-contact-physics-losses",
            "--slide-excess-loss-weight",
            "0",
            "--yaw-excess-loss-weight",
            "0",
            "--motion-floor-loss-weight",
            "0",
            "--diagnostic-metrics-every-epochs",
            "0",
            "--save-live-every-epochs",
            "0",
            "--no-live-viewer",
            "--no-visual-reporter",
            "--no-compile",
        ]
        log_text = run(cmd, LOGS / f"{run_name}.out.log")
        ckpt = RUNS / run_name / "checkpoints" / "checkpoint_best_k32.pt"
        if not ckpt.exists():
            ckpt = RUNS / run_name / "checkpoints" / "checkpoint_best.pt"
        epoch, best_val = checkpoint_epoch(ckpt)
        elapsed = parse_elapsed_for_epoch(log_text, epoch)
        row: dict[str, object] = {
            "run_name": run_name,
            "extra_prior_weight": extra_weight,
            "compatibility_score_weight": compat_weight,
            "checkpoint": str(ckpt),
            "best_epoch": epoch,
            "best_val": best_val,
            "best_elapsed_s": elapsed,
        }
        row.update(evaluate(run_name, ckpt, R45, "R45"))
        row.update(evaluate(run_name, ckpt, L45, "L45"))
        row.update(evaluate(run_name, ckpt, WALKF, "WalkF"))
        row["mean_ar_avg"] = (float(row["R45_ar_avg"]) + float(row["L45_ar_avg"])) / 2.0
        row["mean_turn_walk_ar_avg"] = (
            float(row["R45_ar_avg"]) + float(row["L45_ar_avg"]) + float(row["WalkF_ar_avg"])
        ) / 3.0
        row["max_ar_max"] = max(float(row["R45_ar_max"]), float(row["L45_ar_max"]))
        row["max_turn_walk_ar_max"] = max(
            float(row["R45_ar_max"]), float(row["L45_ar_max"]), float(row["WalkF_ar_max"])
        )
        rows.append(row)
        print(
            f"{run_name}: mean_ar_avg={row['mean_ar_avg']:.6f} "
            f"mean_turn_walk={row['mean_turn_walk_ar_avg']:.6f} "
            f"R45={row['R45_ar_avg']:.6f} L45={row['L45_ar_avg']:.6f} WalkF={row['WalkF_ar_avg']:.6f} "
            f"epoch={epoch} elapsed={elapsed:.1f}s",
            flush=True,
        )
        with (summary_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    best = min(rows, key=lambda r: float(r["mean_turn_walk_ar_avg"]))
    print("\n=== BEST ===", flush=True)
    print(best, flush=True)
    print(f"summary: {summary_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
