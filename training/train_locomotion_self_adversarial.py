from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(text: str | Path) -> Path:
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def run_command(cmd: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(cmd), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n\n" + "=" * 100 + "\n")
        log.write(" ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"command failed with code {code}: {' '.join(cmd)}")


def checkpoint_meta(path: Path) -> dict[str, object]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": str(path),
        "epoch": ckpt.get("epoch"),
        "rollout_k": ckpt.get("rollout_k"),
        "best_val": ckpt.get("best_val", ckpt.get("best")),
        "priors": ckpt.get("metadata", {}).get("ae_prior_checkpoints"),
        "prior_weights": ckpt.get("metadata", {}).get("ae_prior_weights"),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def split_semicolon(text: str) -> list[str]:
    return [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]


def key_clip_paths(args: argparse.Namespace) -> list[Path]:
    names = split_semicolon(args.key_clip_names)
    folders = [resolve_path(args.periodic_folder_path), resolve_path(args.nonperiodic_folder_path)]
    found: list[Path] = []
    for name in names:
        matches: list[Path] = []
        for folder in folders:
            if folder.exists():
                matches.extend(folder.glob(f"*{name}*.npz"))
        if not matches:
            print(f"warning: key clip pattern {name!r} found no clips", flush=True)
            continue
        found.append(matches[0])
    return found


def evaluate_key_clips(
    args: argparse.Namespace,
    run_root: Path,
    cycle: int,
    controller_checkpoint: Path,
    prior_checkpoint: Path,
) -> None:
    clips = key_clip_paths(args)
    if not clips:
        return
    diag_dir = run_root / "diagnostics" / f"cycle_{cycle:02d}"
    for clip in clips:
        safe_name = clip.stem.replace(" ", "_")
        foot_html = diag_dir / f"{safe_name}_footskating.html"
        cmd = [
            sys.executable,
            "training/visualize_rollout_foot_skating.py",
            "--npz-path",
            str(clip),
            "--checkpoint-path",
            str(controller_checkpoint),
            "--label",
            f"cycle_{cycle:02d}",
            "--output-html",
            str(foot_html),
            "--device",
            args.eval_device,
        ]
        run_command(cmd, PROJECT_ROOT, run_root / "logs" / f"eval_footskate_cycle_{cycle:02d}.log")
    lab_csv = diag_dir / "ae_lab_cases.csv"
    cmd = [
        sys.executable,
        "training/evaluate_ae_lab_cases.py",
        "--prior-checkpoint",
        str(prior_checkpoint),
        "--prior-label",
        f"cycle_{cycle:02d}_prior",
        "--bad-checkpoint",
        str(controller_checkpoint),
        "--bad-label",
        f"cycle_{cycle:02d}_controller",
        "--output-csv",
        str(lab_csv),
        "--device",
        args.eval_device,
    ]
    for clip in clips:
        cmd.extend(["--npz-path", str(clip)])
    run_command(cmd, PROJECT_ROOT, run_root / "logs" / f"eval_ae_cycle_{cycle:02d}.log")


def train_model_aware_prior(
    args: argparse.Namespace,
    run_root: Path,
    cycle: int,
    controller_checkpoint: Path,
    prior_checkpoint: Path,
    fake_buffer_path: Path,
) -> Path:
    run_name = f"{run_root.name}/ae_cycle_{cycle:02d}"
    cmd = [
        sys.executable,
        "training/train_model_aware_transition_ae.py",
        "--folder-path",
        args.folder_path,
        "--periodic-folder-path",
        args.periodic_folder_path,
        "--nonperiodic-folder-path",
        args.nonperiodic_folder_path,
        "--model-checkpoint",
        str(controller_checkpoint),
        "--init-prior-checkpoint",
        str(prior_checkpoint),
        "--run-name",
        run_name,
        "--device",
        args.device,
        "--learning-rate",
        str(args.ae_learning_rate),
        "--batch-size",
        str(args.ae_batch_size),
        "--max-epochs",
        str(args.ae_epochs_per_cycle),
        "--input-noise-std",
        str(args.ae_input_noise_std),
        "--fake-margin",
        str(args.fake_margin),
        "--fake-weight",
        str(args.fake_weight),
        "--real-weight",
        str(args.real_weight),
        "--compatibility-real-weight",
        str(args.compatibility_real_weight),
        "--compatibility-fake-weight",
        str(args.compatibility_fake_weight),
        "--fake-starts-per-clip",
        str(args.fake_starts_per_clip),
        "--fake-rollout-steps",
        str(args.fake_rollout_steps),
        "--fake-buffer-path",
        str(fake_buffer_path),
        "--fake-buffer-max-rows",
        str(args.fake_buffer_max_rows),
        "--hard-negative-keep-fraction",
        str(args.hard_negative_keep_fraction),
        "--eval-every-epochs",
        str(args.ae_eval_every_epochs),
        "--stall-patience-epochs",
        str(args.ae_stall_patience_epochs),
    ]
    run_command(cmd, PROJECT_ROOT, run_root / "logs" / f"ae_cycle_{cycle:02d}.log")
    ckpt = PROJECT_ROOT / "training" / "runs" / run_name / "checkpoints" / "checkpoint_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    return ckpt


def train_controller(
    args: argparse.Namespace,
    run_root: Path,
    cycle: int,
    controller_checkpoint: Path,
    prior_checkpoint: Path,
) -> Path:
    run_name = f"{run_root.name}/controller_cycle_{cycle:02d}"
    cmd = [
        sys.executable,
        "training/train_locomotion_ae_prior.py",
        "--periodic-folder-path",
        args.periodic_folder_path,
        "--nonperiodic-folder-path",
        args.nonperiodic_folder_path,
        "--prior-checkpoint",
        str(prior_checkpoint),
        "--prior-weight",
        str(args.prior_weight),
        "--compatibility-score-weight",
        str(args.controller_compatibility_score_weight),
        "--resume-checkpoint",
        str(controller_checkpoint),
        "--run-name",
        run_name,
        "--device",
        args.device,
        "--hidden-dim",
        str(args.controller_hidden_dim),
        "--num-hidden-layers",
        str(args.controller_num_hidden_layers),
        "--learning-rate",
        str(args.controller_learning_rate),
        "--batch-size",
        str(args.controller_batch_size),
        "--training-loop",
        "agents",
        "--agent-sampling",
        "random",
        "--agent-batches-per-epoch",
        str(args.agent_batches_per_epoch),
        "--packed-agent-rollout",
        "--agent-batch-clips",
        "0",
        "--periodic-sampling-weight",
        str(args.periodic_sampling_weight),
        "--nonperiodic-sampling-weight",
        str(args.nonperiodic_sampling_weight),
        "--max-epochs",
        str(args.controller_epochs_per_cycle),
        "--rollout-schedule",
        args.rollout_schedule,
        "--curriculum-max-epochs-per-stage",
        str(args.curriculum_max_epochs_per_stage),
        "--curriculum-stall-patience-epochs",
        str(args.curriculum_stall_patience_epochs),
        "--curriculum-min-epochs",
        str(args.curriculum_min_epochs),
        "--curriculum-min-delta",
        str(args.curriculum_min_delta),
        "--mixed-rollout-cohorts",
        "--mixed-rollout-cohort-schedule",
        args.mixed_rollout_cohort_schedule,
        "--mixed-rollout-cohort-weights",
        args.mixed_rollout_cohort_weights,
        "--no-contact-physics-losses",
        "--no-live-viewer",
        "--no-visual-reporter",
        "--diagnostic-metrics-every-epochs",
        "0",
        "--timed-checkpoint-interval-minutes",
        str(args.timed_checkpoint_interval_minutes),
    ]
    for extra_prior in args.extra_prior_checkpoint:
        cmd.extend(["--extra-prior-checkpoint", str(resolve_path(extra_prior))])
    for extra_weight in args.extra_prior_weight:
        cmd.extend(["--extra-prior-weight", str(extra_weight)])
    if args.simple_footslide_loss_weight > 0.0:
        cmd.extend(
            [
                "--simple-footslide-loss-weight",
                str(args.simple_footslide_loss_weight),
                "--simple-footslide-threshold-mps",
                str(args.simple_footslide_threshold_mps),
                "--simple-footslide-gt-margin",
                str(args.simple_footslide_gt_margin),
                "--turn-idle-footslide-tolerance-divisor",
                str(args.turn_idle_footslide_tolerance_divisor),
            ]
        )
    if args.compile:
        cmd.append("--compile")
    run_command(cmd, PROJECT_ROOT, run_root / "logs" / f"controller_cycle_{cycle:02d}.log")
    ckpt = PROJECT_ROOT / "training" / "runs" / run_name / "checkpoints" / f"checkpoint_best_k{args.expected_best_k:02d}.pt"
    if not ckpt.exists():
        ckpt = PROJECT_ROOT / "training" / "runs" / run_name / "checkpoints" / "checkpoint_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end hard-negative AE/controller training loop.")
    parser.add_argument("--folder-path", default="data/npz_final")
    parser.add_argument("--periodic-folder-path", required=True)
    parser.add_argument("--nonperiodic-folder-path", required=True)
    parser.add_argument("--start-controller-checkpoint", required=True)
    parser.add_argument("--start-prior-checkpoint", required=True)
    parser.add_argument("--run-name", default="self_adversarial_locomotion")
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument(
        "--key-clip-names",
        default=(
            "Idle;Walk_Loop_F;Circle_Strafe_L;Circle_Strafe_R;"
            "Stand_Turn_045_L;Stand_Turn_045_R;Box_RL_B_Lfoot;Box_RL_B_Rfoot"
        ),
    )
    parser.add_argument("--ae-epochs-per-cycle", type=int, default=80)
    parser.add_argument("--ae-learning-rate", type=float, default=3e-4)
    parser.add_argument("--ae-batch-size", type=int, default=1024)
    parser.add_argument("--ae-input-noise-std", type=float, default=0.02)
    parser.add_argument("--ae-eval-every-epochs", type=int, default=10)
    parser.add_argument("--ae-stall-patience-epochs", type=int, default=0)
    parser.add_argument("--fake-margin", type=float, default=0.03)
    parser.add_argument("--fake-weight", type=float, default=1.0)
    parser.add_argument("--real-weight", type=float, default=1.0)
    parser.add_argument("--compatibility-real-weight", type=float, default=0.5)
    parser.add_argument("--compatibility-fake-weight", type=float, default=0.0)
    parser.add_argument("--fake-starts-per-clip", type=int, default=8)
    parser.add_argument("--fake-rollout-steps", type=int, default=32)
    parser.add_argument("--fake-buffer-max-rows", type=int, default=200000)
    parser.add_argument("--hard-negative-keep-fraction", type=float, default=0.65)
    parser.add_argument("--controller-epochs-per-cycle", type=int, default=420)
    parser.add_argument("--controller-learning-rate", type=float, default=5e-5)
    parser.add_argument("--controller-batch-size", type=int, default=64)
    parser.add_argument("--controller-hidden-dim", type=int, default=512)
    parser.add_argument("--controller-num-hidden-layers", type=int, default=2)
    parser.add_argument("--controller-compatibility-score-weight", type=float, default=0.0)
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--extra-prior-checkpoint", action="append", default=[])
    parser.add_argument("--extra-prior-weight", action="append", type=float, default=[])
    parser.add_argument("--simple-footslide-loss-weight", type=float, default=0.0)
    parser.add_argument("--simple-footslide-threshold-mps", type=float, default=0.0)
    parser.add_argument("--simple-footslide-gt-margin", type=float, default=1.05)
    parser.add_argument("--turn-idle-footslide-tolerance-divisor", type=float, default=1.0)
    parser.add_argument("--agent-batches-per-epoch", type=int, default=1)
    parser.add_argument("--periodic-sampling-weight", type=float, default=1.0)
    parser.add_argument("--nonperiodic-sampling-weight", type=float, default=1.0)
    parser.add_argument("--rollout-schedule", default="2,4,8,16,32")
    parser.add_argument("--mixed-rollout-cohort-schedule", default="2,4,8,16,32")
    parser.add_argument("--mixed-rollout-cohort-weights", default="5,15,20,30,40")
    parser.add_argument("--expected-best-k", type=int, default=32)
    parser.add_argument("--curriculum-max-epochs-per-stage", type=int, default=120)
    parser.add_argument("--curriculum-stall-patience-epochs", type=int, default=60)
    parser.add_argument("--curriculum-min-epochs", type=int, default=30)
    parser.add_argument("--curriculum-min-delta", type=float, default=1e-6)
    parser.add_argument("--timed-checkpoint-interval-minutes", type=float, default=30.0)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluate-every-cycle", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = PROJECT_ROOT / "training" / "runs" / f"{stamp}_{args.run_name}"
    run_root.mkdir(parents=True, exist_ok=True)
    fake_buffer = run_root / "hard_negative_buffer.pt"
    write_json(run_root / "config.json", vars(args))

    controller = resolve_path(args.start_controller_checkpoint)
    prior = resolve_path(args.start_prior_checkpoint)
    rows: list[dict[str, object]] = []
    print(f"self_adversarial run_root={run_root}", flush=True)
    print(f"start controller={controller}", flush=True)
    print(f"start prior={prior}", flush=True)

    for cycle in range(1, args.cycles + 1):
        cycle_start = time.perf_counter()
        print(f"\n=== cycle {cycle}/{args.cycles}: update AE ===", flush=True)
        prior = train_model_aware_prior(args, run_root, cycle, controller, prior, fake_buffer)
        print(f"\n=== cycle {cycle}/{args.cycles}: train controller ===", flush=True)
        controller = train_controller(args, run_root, cycle, controller, prior)
        if args.evaluate_every_cycle:
            print(f"\n=== cycle {cycle}/{args.cycles}: diagnostics ===", flush=True)
            evaluate_key_clips(args, run_root, cycle, controller, prior)
        row = {
            "cycle": cycle,
            "elapsed_s": time.perf_counter() - cycle_start,
            "controller_checkpoint": str(controller),
            "prior_checkpoint": str(prior),
            "controller_meta": json.dumps(checkpoint_meta(controller)),
            "prior_meta": json.dumps(checkpoint_meta(prior)),
        }
        rows.append(row)
        append_csv(run_root / "cycle_summary.csv", row)
        write_json(run_root / "latest.json", row)
        print(f"cycle {cycle} done controller={controller} prior={prior}", flush=True)

    print(f"finished self-adversarial loop run_root={run_root}", flush=True)


if __name__ == "__main__":
    main()
