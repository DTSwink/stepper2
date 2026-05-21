from __future__ import annotations

import csv
import math
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import inspect_foot_sliding as footmon
import train_locomotion as tl
import visualize_model as vm


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".tools" / "python310" / "python.exe"
RUNS = ROOT / "training" / "runs"
LOGS = RUNS / "launch_logs"

BASELINE = (
    RUNS
    / "20260516_044547_hybrid_canonbasis_from_bestk32_constantk32_w010_resumeopt"
    / "checkpoints"
    / "checkpoint_best_k32.pt"
)
PRIOR_1F = RUNS / "20260517_161820_denoise_rootlook1_lat32_n0p05_e360" / "checkpoints" / "checkpoint_best.pt"
PRIOR_ROOTLOOK16 = (
    RUNS / "20260517_184310_denoise_rootlook16_fullae_lat32_dampedcompat035_n0p05_e300" / "checkpoints" / "checkpoint_best.pt"
)

MINI_ROOT = ROOT / "training" / "mini_datasets" / "20260518_idle_back_turn45_circle"
PERIODIC = MINI_ROOT / "periodic_npz_final"
NONPERIODIC = MINI_ROOT / "nonperiodic_npz_final"


@dataclass(frozen=True)
class Candidate:
    label: str
    k_schedule: str
    k_weights: str
    kmax: int
    lr: str
    loss_kind: str
    slide_weight: str = "0"
    slide_threshold: str = "0"
    yaw_weight: str = "0"
    excess_envelope: bool = False
    epochs: int = 100


def run(cmd: list[str], log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            chunks.append(line)
            log.write(line)
            log.flush()
        proc.wait()
    output = "".join(chunks)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with code {proc.returncode}: {' '.join(cmd)}\n{output[-5000:]}")
    return output


def checkpoint_summary(path: Path) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return int(ckpt.get("epoch", -1)), float(ckpt.get("best_val", float("nan")))


def parse_elapsed(log_text: str, epoch: int) -> float:
    match = re.search(rf"epoch={epoch:04d}.*?elapsed_s=([-+0-9.eE]+)", log_text)
    return float(match.group(1)) if match else float("nan")


def clip_paths() -> list[Path]:
    return sorted(PERIODIC.glob("*.npz")) + sorted(NONPERIODIC.glob("*.npz"))


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path: Path, device_text: str = "cuda") -> dict[str, float | str]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = tl.TrainConfig()
    vm.apply_config_dict(cfg, ckpt.get("config", {}))
    device = torch.device(device_text if torch.cuda.is_available() or device_text == "cpu" else "cpu")
    cfg.device = str(device)
    cfg.use_torch_compile = False

    clips = [tl.MotionClip(path, cfg) for path in clip_paths()]
    model = vm.load_model(ckpt, clips[0], cfg, device)
    model.eval()

    rows: list[dict[str, float | str]] = []
    for clip in clips:
        frame_count = max(3, int(clip.T))
        gt_pos, gt_rot, _tf_pos, _tf_rot, _err_tf, pred_pos, pred_rot, err_ar = vm.rollout_model(
            model,
            clip,
            cfg,
            device,
            frame_count,
        )
        contacts = clip.contacts[:frame_count].to(device)
        gt_pos_t = torch.as_tensor(gt_pos, device=device, dtype=torch.float32)
        gt_rot_t = torch.as_tensor(gt_rot, device=device, dtype=torch.float32)
        pred_pos_t = torch.as_tensor(pred_pos, device=device, dtype=torch.float32)
        pred_rot_t = torch.as_tensor(pred_rot, device=device, dtype=torch.float32)
        gt_slide = footmon.slide_metrics(gt_pos_t, gt_rot_t, contacts, clip, clip.fps, near_height_threshold=0.025)
        pred_slide = footmon.slide_metrics(pred_pos_t, pred_rot_t, contacts, clip, clip.fps, near_height_threshold=0.025)
        per_bone_error = torch.linalg.norm(pred_pos_t - gt_pos_t, dim=-1)
        ar_per_frame = per_bone_error.mean(dim=1)
        rows.append(
            {
                "clip": Path(clip.path).stem,
                "ar_avg": float(per_bone_error.mean().detach().cpu()),
                "ar_end": float(ar_per_frame[-1].detach().cpu()),
                "ar_max": float(ar_per_frame.max().detach().cpu()),
                "pred_slide_p95": float(pred_slide["contact_or_near_p95_mps"]),
                "gt_slide_p95": float(gt_slide["contact_or_near_p95_mps"]),
                "slide_delta_p95": float(pred_slide["contact_or_near_p95_mps"] - gt_slide["contact_or_near_p95_mps"]),
                "pred_min_height": float(pred_slide["min_height_m"]),
            }
        )

    ar_avgs = np.asarray([float(row["ar_avg"]) for row in rows], dtype=np.float64)
    ar_maxes = np.asarray([float(row["ar_max"]) for row in rows], dtype=np.float64)
    slide_delta = np.asarray([float(row["slide_delta_p95"]) for row in rows], dtype=np.float64)
    min_heights = np.asarray([float(row["pred_min_height"]) for row in rows], dtype=np.float64)
    worst_idx = int(np.argmax(ar_avgs))
    return {
        "checkpoint": str(checkpoint_path),
        "mean_ar_avg": float(ar_avgs.mean()),
        "worst_ar_avg": float(ar_avgs.max()),
        "worst_ar_clip": str(rows[worst_idx]["clip"]),
        "mean_ar_max": float(ar_maxes.mean()),
        "worst_ar_max": float(ar_maxes.max()),
        "mean_slide_delta_p95": float(slide_delta.mean()),
        "worst_slide_delta_p95": float(slide_delta.max()),
        "min_pred_height": float(min_heights.min()),
        **{f"{row['clip']}_ar_avg": float(row["ar_avg"]) for row in rows},
    }


def candidate_cmd(candidate: Candidate, run_name: str) -> list[str]:
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
        str(PRIOR_ROOTLOOK16),
        "--extra-prior-weight",
        "0.30",
        "--compatibility-score-weight",
        "0.05",
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
        candidate.lr,
        "--batch-size",
        "256",
        "--max-epochs",
        str(candidate.epochs),
        "--rollout-schedule",
        candidate.k_schedule,
        "--initial-rollout-k",
        str(candidate.kmax),
        "--mixed-rollout-cohorts",
        "--mixed-rollout-cohort-schedule",
        candidate.k_schedule,
        "--mixed-rollout-cohort-weights",
        candidate.k_weights,
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
        "2",
        "--agent-min-cohort-steps",
        "2",
        "--no-contact-physics-losses",
        "--slide-excess-loss-weight",
        candidate.slide_weight,
        "--turn-slide-bound-divisor",
        "20",
        "--yaw-excess-loss-weight",
        candidate.yaw_weight,
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
    cmd.append("--excess-envelope" if candidate.excess_envelope else "--no-excess-envelope")
    if candidate.excess_envelope:
        cmd.extend(["--excess-envelope-knn", "16", "--excess-envelope-margin", "1.05", "--yaw-excess-scale-radps", "0"])
    return cmd


def main() -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    summary_dir = RUNS / "mini_sweeps" / f"{stamp}_idle_back_turn45_circle"
    summary_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        Candidate("nofoot_k64_lr5e5", "2,4,8,16,32,64", "5,10,15,20,25,35", 64, "5e-5", "none", epochs=320),
        Candidate("nofoot_k32_lr5e5", "2,4,8,16,32", "5,10,20,30,35", 32, "5e-5", "none", epochs=260),
        Candidate("oldslide_w010_k32_lr5e5", "2,4,8,16,32", "5,10,20,30,35", 32, "5e-5", "old", "0.10", "0.2135299310088158", "0", False, 220),
        Candidate("oldslide_w028_k32_lr5e5", "2,4,8,16,32", "5,10,20,30,35", 32, "5e-5", "old", "0.28", "0.2135299310088158", "0", False, 220),
        Candidate("env_w050_y010_k32_lr1e5", "2,4,8,16,32", "5,10,20,30,35", 32, "1e-5", "envelope", "0.50", "0", "0.10", True, 220),
    ]

    rows: list[dict[str, object]] = []
    summary_path = summary_dir / "summary.csv"
    for candidate in candidates:
        run_name = f"{stamp}_mini_{candidate.label}"
        print(f"\n=== {run_name} ===", flush=True)
        print(candidate, flush=True)
        log_text = run(candidate_cmd(candidate, run_name), LOGS / f"{run_name}.out.log")
        run_dir = RUNS / run_name
        checkpoints = [run_dir / "checkpoints" / "checkpoint_best.pt"]
        last = run_dir / "checkpoints" / "checkpoint_last.pt"
        if last.exists():
            checkpoints.append(last)

        best_row: dict[str, object] | None = None
        for checkpoint in checkpoints:
            epoch, best_val = checkpoint_summary(checkpoint)
            metrics = evaluate_checkpoint(checkpoint)
            row: dict[str, object] = {
                "candidate": candidate.label,
                "run_name": run_name,
                "loss_kind": candidate.loss_kind,
                "k_schedule": candidate.k_schedule,
                "k_weights": candidate.k_weights,
                "kmax": candidate.kmax,
                "lr": candidate.lr,
                "slide_weight": candidate.slide_weight,
                "slide_threshold": candidate.slide_threshold,
                "yaw_weight": candidate.yaw_weight,
                "excess_envelope": candidate.excess_envelope,
                "checkpoint_name": checkpoint.name,
                "epoch": epoch,
                "best_val": best_val,
                "elapsed_s_at_epoch": parse_elapsed(log_text, epoch),
                **metrics,
            }
            if best_row is None or float(row["worst_ar_avg"]) < float(best_row["worst_ar_avg"]):
                best_row = row
        assert best_row is not None
        rows.append(best_row)
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"best {candidate.label}: worst_ar={float(best_row['worst_ar_avg']):.5f} "
            f"mean_ar={float(best_row['mean_ar_avg']):.5f} worst_clip={best_row['worst_ar_clip']} "
            f"slide_delta={float(best_row['worst_slide_delta_p95']):+.4f} "
            f"ckpt={best_row['checkpoint_name']} epoch={best_row['epoch']}",
            flush=True,
        )

    ordered = sorted(rows, key=lambda row: (float(row["worst_ar_avg"]), float(row["mean_ar_avg"])))
    print("\n=== RANKED BY WORST CLIP AR AVG ===", flush=True)
    for i, row in enumerate(ordered, 1):
        print(
            f"{i:02d} {row['candidate']} worst={float(row['worst_ar_avg']):.5f} "
            f"mean={float(row['mean_ar_avg']):.5f} clip={row['worst_ar_clip']} "
            f"slide_delta={float(row['worst_slide_delta_p95']):+.4f} checkpoint={row['checkpoint']}",
            flush=True,
        )
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
