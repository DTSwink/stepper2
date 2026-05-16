from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def kaggle_output_root() -> Path:
    root = Path("/kaggle/working")
    if root.exists():
        return root
    return PROJECT_ROOT / "training" / "kaggle_output_mirror"


def mirror_checkpoints(run_name: str) -> None:
    src = PROJECT_ROOT / "training" / "runs" / run_name / "checkpoints"
    dst = kaggle_output_root() / "stepper_checkpoints" / run_name / "checkpoints"
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(src.glob("*.pt")):
        target = dst / path.name
        if not target.exists() or target.stat().st_size != path.stat().st_size or target.stat().st_mtime < path.stat().st_mtime:
            shutil.copy2(path, target)
            print(f"mirrored checkpoint at {target}", flush=True)
        copied.append(path.name)
    manifest = dst.parent / "CHECKPOINTS_ARE_HERE.txt"
    manifest.write_text(
        "Stepper Kaggle checkpoint mirror\n"
        f"run_name={run_name}\n"
        f"source={src}\n"
        f"mirrored_to={dst}\n"
        "Download from the Kaggle Output tab after stopping or completing the run.\n"
        "Files:\n"
        + "\n".join(copied)
        + "\n",
        encoding="utf-8",
    )


def start_checkpoint_mirror(run_name: str, interval_seconds: float = 300.0) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()

    def loop() -> None:
        while not stop_event.wait(interval_seconds):
            try:
                mirror_checkpoints(run_name)
            except Exception as exc:
                print(f"checkpoint mirror skipped: {exc}", flush=True)

    thread = threading.Thread(target=loop, name="checkpoint-mirror", daemon=True)
    thread.start()
    return stop_event, thread


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    os.chdir(PROJECT_ROOT)

    try:
        import torch

        if torch.cuda.is_available():
            print(f"cuda: {torch.cuda.get_device_name(0)}", flush=True)
        else:
            print("cuda: unavailable, training will run on CPU", flush=True)
    except Exception as exc:
        print(f"torch probe failed: {exc}", flush=True)

    run_name = os.environ.get("STEPPER_RUN_NAME", "kaggle_k111_from_best_k64")
    max_train_seconds = os.environ.get("STEPPER_MAX_TRAIN_SECONDS", "0")
    batch_size = os.environ.get("STEPPER_BATCH_SIZE", "256")
    learning_rate = os.environ.get("STEPPER_LEARNING_RATE", "1e-6")
    footslide_weight = os.environ.get("STEPPER_FOOTSLIDE_WEIGHT", "0.10")
    footslide_threshold = os.environ.get("STEPPER_FOOTSLIDE_THRESHOLD_MPS", "0.2135299310088158")
    max_epochs = os.environ.get("STEPPER_MAX_EPOCHS", "100000")
    initial_k = os.environ.get("STEPPER_INITIAL_ROLLOUT_K", "111")
    prior_checkpoint = os.environ.get(
        "STEPPER_PRIOR_CHECKPOINT",
        str(
            PROJECT_ROOT
            / "training"
            / "runs"
            / "ae_poseaware_hybrid_datasetrefresh_20260515_175707"
            / "checkpoints"
            / "checkpoint_best.pt"
        ),
    )
    resume_checkpoint = os.environ.get(
        "STEPPER_RESUME_CHECKPOINT",
        str(
            PROJECT_ROOT
            / "training"
            / "runs"
            / "hybrid_datasetrefresh_packed_scheduleK_from_epoch2040_w010_20260515_180333"
            / "checkpoints"
            / "checkpoint_best_k64.pt"
        ),
    )

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "training" / "train_locomotion_ae_prior.py"),
        "--periodic-folder-path",
        str(PROJECT_ROOT / "ue5" / "animations_omni_only_full"),
        "--nonperiodic-folder-path",
        str(PROJECT_ROOT / "ue5" / "animations_transitions_only_full_trimmed"),
        "--prior-checkpoint",
        prior_checkpoint,
        "--resume-checkpoint",
        resume_checkpoint,
        "--run-name",
        run_name,
        "--device",
        os.environ.get("STEPPER_DEVICE", "cuda"),
        "--hidden-dim",
        os.environ.get("STEPPER_HIDDEN_DIM", "512"),
        "--num-hidden-layers",
        os.environ.get("STEPPER_NUM_HIDDEN_LAYERS", "2"),
        "--batch-size",
        batch_size,
        "--training-loop",
        "agents",
        "--agent-sampling",
        "random",
        "--agent-batch-clips",
        "0",
        "--packed-agent-rollout",
        "--agent-batches-per-epoch",
        os.environ.get("STEPPER_AGENT_BATCHES_PER_EPOCH", "1"),
        "--gradient-accumulation-batches",
        os.environ.get("STEPPER_GRAD_ACCUM_BATCHES", "1"),
        "--periodic-sampling-weight",
        os.environ.get("STEPPER_PERIODIC_SAMPLING_WEIGHT", "2"),
        "--nonperiodic-sampling-weight",
        os.environ.get("STEPPER_NONPERIODIC_SAMPLING_WEIGHT", "1"),
        "--agent-min-cohort-steps",
        os.environ.get("STEPPER_AGENT_MIN_COHORT_STEPS", "8"),
        "--rollout-schedule",
        os.environ.get("STEPPER_ROLLOUT_SCHEDULE", "1,2,4,8,16,32,64,111"),
        "--initial-rollout-k",
        initial_k,
        "--curriculum-min-epochs",
        os.environ.get("STEPPER_CURRICULUM_MIN_EPOCHS", "80"),
        "--curriculum-stall-patience-epochs",
        os.environ.get("STEPPER_CURRICULUM_STALL_PATIENCE_EPOCHS", "160"),
        "--curriculum-max-epochs-per-stage",
        os.environ.get("STEPPER_CURRICULUM_MAX_EPOCHS_PER_STAGE", "400"),
        "--curriculum-min-eligible-clip-visits",
        os.environ.get("STEPPER_CURRICULUM_MIN_ELIGIBLE_CLIP_VISITS", "0.5"),
        "--learning-rate",
        learning_rate,
        "--max-epochs",
        max_epochs,
        "--diagnostic-metrics-every-epochs",
        os.environ.get("STEPPER_DIAGNOSTIC_METRICS_EVERY_EPOCHS", "10"),
        "--save-live-every-epochs",
        os.environ.get("STEPPER_SAVE_EVERY_EPOCHS", "20"),
        "--visual-report-save-every-epochs",
        os.environ.get("STEPPER_VISUAL_REPORT_SAVE_EVERY_EPOCHS", "50"),
        "--visual-report-interval-seconds",
        os.environ.get("STEPPER_VISUAL_REPORT_INTERVAL_SECONDS", "60"),
        "--visual-report-device",
        os.environ.get("STEPPER_VISUAL_REPORT_DEVICE", "cpu"),
        "--visual-report-max-frames",
        os.environ.get("STEPPER_VISUAL_REPORT_MAX_FRAMES", "180"),
        "--simple-footslide-loss-weight",
        footslide_weight,
        "--simple-footslide-threshold-mps",
        footslide_threshold,
        "--simple-footslide-gt-margin",
        os.environ.get("STEPPER_FOOTSLIDE_GT_MARGIN", "1.05"),
        "--no-contact-physics-losses",
        "--no-live-viewer",
        "--no-visual-reporter",
    ]

    if env_flag("STEPPER_FINAL_STAGE_RANDOM_ROLLOUT", True):
        cmd.append("--final-stage-random-rollout")
    if env_flag("STEPPER_TORCH_COMPILE", False):
        cmd.append("--compile")
    else:
        cmd.append("--no-compile")
    if env_flag("STEPPER_RESUME_OPTIMIZER", False):
        cmd.append("--resume-optimizer")
    if max_train_seconds and float(max_train_seconds) > 0.0:
        cmd.extend(["--max-train-seconds", max_train_seconds])

    mirror_stop, mirror_thread = start_checkpoint_mirror(run_name)
    try:
        print("running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    finally:
        mirror_stop.set()
        try:
            mirror_thread.join(timeout=5.0)
        except RuntimeError:
            pass
        try:
            mirror_checkpoints(run_name)
            print(
                "checkpoint mirror:",
                kaggle_output_root() / "stepper_checkpoints" / run_name,
                flush=True,
            )
        except Exception as exc:
            print(f"final checkpoint mirror failed: {exc}", flush=True)


if __name__ == "__main__":
    main()
