from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "training" / "runs"
DEFAULT_PERIODIC = PROJECT_ROOT / "ue5" / "animations_omni_only_full" / "npz_final"
DEFAULT_NONPERIODIC = PROJECT_ROOT / "ue5" / "animations_transitions_only_full_trimmed" / "npz_final"
STAMP_RE_PREFIXES = ("_ik_",)


def output_root() -> Path:
    root = Path("/kaggle/working")
    return root if root.exists() else PROJECT_ROOT / "training" / "ik" / "kaggle_output_mirror"


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_text(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def add_dataset_args(cmd: list[str]) -> None:
    npz = env_text("STEPPER_NPZ")
    periodic = env_text("STEPPER_PERIODIC_FOLDER", str(DEFAULT_PERIODIC))
    nonperiodic = env_text("STEPPER_NONPERIODIC_FOLDER", str(DEFAULT_NONPERIODIC))
    if npz:
        cmd.extend(["--npz", npz])
    if periodic:
        cmd.extend(["--periodic-folder", periodic])
    if nonperiodic:
        cmd.extend(["--nonperiodic-folder", nonperiodic])


def run_dirs_for_label(label: str) -> list[Path]:
    safe = label.strip()
    for marker in STAMP_RE_PREFIXES:
        if marker in safe and safe[:8].isdigit():
            safe = safe.split(marker, 1)[1]
            break
    safe = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in safe.strip("_")) or "run"
    return sorted(RUNS_DIR.glob(f"*_ik_{safe}"), key=lambda path: path.stat().st_mtime, reverse=True)


def mirror_outputs(labels: list[str]) -> None:
    mirror_root = output_root() / "stepper_ik_outputs"
    mirror_root.mkdir(parents=True, exist_ok=True)
    run_dirs: list[Path] = []
    for label in labels:
        run_dirs.extend(run_dirs_for_label(label))
    if not run_dirs:
        run_dirs = [path for path in RUNS_DIR.glob("*_ik_*") if path.is_dir()]
    mirror_patterns = [
        "checkpoints/*.pt",
        "config*.json",
        "run_status.json",
        "REPORT.md",
        "ae_research_results.csv",
        "ae_research_results.json",
    ]
    for run_dir in run_dirs:
        rel_root = mirror_root / run_dir.name
        sources: list[Path] = []
        for pattern in mirror_patterns:
            sources.extend(run_dir.glob(pattern))
        for src in sources:
            if not src.exists() or not src.is_file():
                continue
            dst = rel_root / src.relative_to(run_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or dst.stat().st_size != src.stat().st_size or dst.stat().st_mtime < src.stat().st_mtime:
                shutil.copy2(src, dst)
                print(f"mirrored {src} -> {dst}", flush=True)
    manifest = mirror_root / "IK_OUTPUTS_ARE_HERE.txt"
    manifest.write_text(
        "Stepper IK Kaggle output mirror\n"
        f"source={RUNS_DIR}\n"
        f"mirrored_to={mirror_root}\n"
        "Download checkpoints/configs from the Kaggle Output tab.\n",
        encoding="utf-8",
    )


def start_mirror(labels: list[str], interval_seconds: float = 300.0) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()

    def loop() -> None:
        while not stop_event.wait(interval_seconds):
            try:
                mirror_outputs(labels)
            except Exception as exc:
                print(f"output mirror skipped: {exc}", flush=True)

    thread = threading.Thread(target=loop, name="ik-output-mirror", daemon=True)
    thread.start()
    return stop_event, thread


def supervised_cmd() -> tuple[list[str], list[str]]:
    label = env_text("STEPPER_RUN_LABEL", "kaggle_full_supervised")
    cmd = [sys.executable, "training/ik/train.py", "--run-label", label]
    add_dataset_args(cmd)
    init_checkpoint = env_text("STEPPER_INIT_CHECKPOINT")
    if init_checkpoint:
        cmd.extend(["--init-checkpoint", init_checkpoint])
    cmd.extend(["--train-steps", env_text("STEPPER_TRAIN_STEPS", "100000")])
    if env_flag("STEPPER_LOAD_OPTIMIZER", True):
        cmd.append("--load-optimizer")
    if env_flag("STEPPER_RESUME_STEP_FROM_CHECKPOINT", True):
        cmd.append("--resume-step-from-checkpoint")
    if env_flag("STEPPER_DISABLE_CUDA_GRAPH", False):
        raise RuntimeError("STEPPER_DISABLE_CUDA_GRAPH is forbidden; supervised IK training is CUDA-graph-only.")
    return cmd, [label]


def simple_ae_cmd() -> tuple[list[str], list[str]]:
    label = env_text("STEPPER_AE_LABEL", env_text("STEPPER_RUN_LABEL", "kaggle_simple_ae"))
    cmd = [sys.executable, "training/ik/train_simple_autoencoder.py", "--run-label", label]
    add_dataset_args(cmd)
    return cmd, [label]


def ae_envelope_cmd() -> tuple[list[str], list[str]]:
    ae_label = env_text("STEPPER_AE_LABEL", "kaggle_full_vanilla_ae")
    baseline_label = env_text("STEPPER_BASELINE_LABEL", "kaggle_full_ae_controller_baseline")
    refined_label = env_text("STEPPER_REFINED_LABEL", "kaggle_full_ae_controller_refined")
    final_label = env_text("STEPPER_FINAL_LABEL", "kaggle_full_ae_controller_random_init")
    cmd = [
        sys.executable,
        "training/ik/train_full_ae_envelope.py",
        "--phase",
        env_text("STEPPER_PHASE", "all"),
        "--ae-label",
        ae_label,
        "--baseline-label",
        baseline_label,
        "--refined-label",
        refined_label,
        "--final-label",
        final_label,
    ]
    add_dataset_args(cmd)
    for env_name, arg_name in (
        ("STEPPER_AE_CHECKPOINT", "--ae-checkpoint"),
        ("STEPPER_BASELINE_CHECKPOINT", "--baseline-checkpoint"),
        ("STEPPER_REFINED_CHECKPOINT", "--refined-checkpoint"),
        ("STEPPER_WEIGHTS_JSON", "--weights-json"),
    ):
        value = env_text(env_name)
        if value:
            cmd.extend([arg_name, value])
    pose_noise = env_text("STEPPER_POSE_NOISE")
    if pose_noise:
        cmd.extend(["--pose-noise", pose_noise])
    if env_flag("STEPPER_DISABLE_CUDA_GRAPH", False):
        raise RuntimeError("STEPPER_DISABLE_CUDA_GRAPH is forbidden; CUDA graph is the default training contract.")
    return cmd, [ae_label, baseline_label, refined_label, final_label]


def ae_research_cmd() -> tuple[list[str], list[str]]:
    label = env_text("STEPPER_RUN_LABEL", "kaggle_ae_research")
    cmd = [
        sys.executable,
        "training/ik/kaggle_ae_research.py",
        "--run-label",
        label,
        "--train-steps",
        env_text("STEPPER_TRAIN_STEPS", "4000"),
        "--controller-steps",
        env_text("STEPPER_AE_RESEARCH_CONTROLLER_STEPS", "3000"),
    ]
    for env_name, arg_name in (
        ("STEPPER_AE_RESEARCH_PERIODIC_NAMES", "--periodic-names"),
        ("STEPPER_AE_RESEARCH_NONPERIODIC_NAMES", "--nonperiodic-names"),
        ("STEPPER_AE_RESEARCH_MAX_VARIANTS", "--max-variants"),
        ("STEPPER_AE_RESEARCH_VARIANT_NAMES", "--variant-names"),
        ("STEPPER_AE_RESEARCH_EVAL_ROWS", "--eval-rows"),
        ("STEPPER_AE_RESEARCH_CONTROLLER_PERIODIC_NAMES", "--controller-periodic-names"),
        ("STEPPER_AE_RESEARCH_CONTROLLER_NONPERIODIC_NAMES", "--controller-nonperiodic-names"),
        ("STEPPER_AE_RESEARCH_CONTROLLER_POSE_NOISE", "--controller-pose-noise"),
        ("STEPPER_AE_RESEARCH_WINDOW_FRAMES", "--window-frames"),
    ):
        value = env_text(env_name)
        if value:
            cmd.extend([arg_name, value])
    if env_flag("STEPPER_AE_RESEARCH_SKIP_ONE_FRAME_BASELINE", False):
        cmd.append("--skip-one-frame-baseline")
    return cmd, [label]


def command_for_mode() -> tuple[list[str], list[str]]:
    mode = env_text("STEPPER_IK_MODE", "supervised").lower()
    if mode == "supervised":
        return supervised_cmd()
    if mode in {"ae", "simple_ae", "simple-autoencoder"}:
        return simple_ae_cmd()
    if mode in {"ae_envelope", "envelope", "full_ae_envelope"}:
        return ae_envelope_cmd()
    if mode in {"ae_research", "research", "auto_ae_research"}:
        return ae_research_cmd()
    raise ValueError(f"Unknown STEPPER_IK_MODE={mode!r}")


def main() -> None:
    os.chdir(PROJECT_ROOT)
    try:
        import torch

        if torch.cuda.is_available():
            print(f"cuda: {torch.cuda.get_device_name(0)}", flush=True)
        else:
            print("cuda: unavailable", flush=True)
    except Exception as exc:
        print(f"torch probe failed: {exc}", flush=True)

    cmd, labels = command_for_mode()
    mirror_stop, mirror_thread = start_mirror(labels, float(env_text("STEPPER_MIRROR_INTERVAL_SECONDS", "300")))
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
            mirror_outputs(labels)
        except Exception as exc:
            print(f"final output mirror failed: {exc}", flush=True)


if __name__ == "__main__":
    main()
