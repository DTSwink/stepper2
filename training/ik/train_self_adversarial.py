from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

try:
    from . import ik_core as tl
    from . import train_model_aware_transition_ae as model_aware
except ImportError:
    import ik_core as tl
    import train_model_aware_transition_ae as model_aware


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Canonical recipe branch: GT-difference hard negatives, reconstruction-only AE,
# geometric fake replay, and pure AE-prior controller training.
AE_EPOCHS = 80
AE_LEARNING_RATE = 3e-4
AE_BATCH_SIZE = 1024
AE_INPUT_NOISE_STD = 0.02
AE_FAKE_MARGIN = 0.04
AE_FAKE_WEIGHT = 1.0
AE_REAL_WEIGHT = 1.25
AE_FAKE_STARTS_PER_CLIP = 10
AE_FAKE_ROLLOUT_STEPS = 32
AE_HARD_NEGATIVE_KEEP_FRACTION = 0.7
AE_HARD_NEGATIVE_MODE = "low_energy_high_gtdiff"
AE_FAKE_REPLAY_DECAY = 0.25
AE_FAKE_BUFFER_MAX_ROWS = 0
AE_EVAL_EVERY_EPOCHS = 10
AE_STALL_PATIENCE_EPOCHS = 0
AE_CONDITIONAL_ROOT_WINDOW = False

CONTROLLER_EPOCHS = 620
CONTROLLER_LEARNING_RATE = 5e-5
CONTROLLER_BATCH_SIZE = 64
CONTROLLER_HIDDEN_DIM = 512
CONTROLLER_HIDDEN_LAYERS = 2
CONTROLLER_PRIOR_WEIGHT = 1.0
CONTROLLER_AGENT_BATCHES_PER_EPOCH = 1
PERIODIC_SAMPLING_WEIGHT = 1.0
NONPERIODIC_SAMPLING_WEIGHT = 1.0
ROLLOUT_SCHEDULE = "2,4,8,16,32"
MIXED_ROLLOUT_COHORT_SCHEDULE = "2,4,8,16,32"
MIXED_ROLLOUT_COHORT_WEIGHTS = "5,15,20,30,40"
EXPECTED_BEST_K = 32
CURRICULUM_MAX_EPOCHS_PER_STAGE = 120
CURRICULUM_STALL_PATIENCE_EPOCHS = 60
CURRICULUM_MIN_EPOCHS = 30
CURRICULUM_MIN_DELTA = 1e-6
CONTROLLER_STOP_ON_FINAL_STALL = True
CONTROLLER_SAVE_LIVE_EVERY_EPOCHS = 0
TIMED_CHECKPOINT_INTERVAL_MINUTES = 30.0
GT_RULE_STARTS_PER_CLIP = 10
GT_RULE_ROLLOUT_STEPS = 32
GT_RULE_TOLERANCE_P95 = 0.003

DEFAULT_KEY_CLIPS = (
    "Idle;Walk_Loop_F;Circle_Strafe_L;Circle_Strafe_R;"
    "Stand_Turn_045_L;Stand_Turn_045_R;Box_RL_B_Lfoot;Box_RL_B_Rfoot"
)


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


def write_recipe_config(run_root: Path, args: argparse.Namespace) -> None:
    write_json(
        run_root / "config.json",
        {
            "exposed_args": vars(args),
            "hardcoded_recipe": {
                "ae": {
                    "epochs": AE_EPOCHS,
                    "learning_rate": AE_LEARNING_RATE,
                    "batch_size": AE_BATCH_SIZE,
                    "input_noise_std": AE_INPUT_NOISE_STD,
                    "fake_margin": AE_FAKE_MARGIN,
                    "fake_weight": AE_FAKE_WEIGHT,
                    "real_weight": AE_REAL_WEIGHT,
                    "compatibility_real_weight": 0.0,
                    "compatibility_fake_weight": 0.0,
                    "fake_starts_per_clip": AE_FAKE_STARTS_PER_CLIP,
                    "fake_rollout_steps": AE_FAKE_ROLLOUT_STEPS,
                    "fake_buffer": "geometric_generation_replay",
                    "fake_replay_decay": AE_FAKE_REPLAY_DECAY,
                    "fake_buffer_max_rows": AE_FAKE_BUFFER_MAX_ROWS,
                    "fake_replay_sampling": "generation_weighted_normalized",
                    "hard_negative_keep_fraction": AE_HARD_NEGATIVE_KEEP_FRACTION,
                    "hard_negative_mode": AE_HARD_NEGATIVE_MODE,
                    "conditional_root_window": AE_CONDITIONAL_ROOT_WINDOW,
                },
                "controller": {
                    "epochs": CONTROLLER_EPOCHS,
                    "learning_rate": CONTROLLER_LEARNING_RATE,
                    "batch_size": CONTROLLER_BATCH_SIZE,
                    "hidden_dim": CONTROLLER_HIDDEN_DIM,
                    "num_hidden_layers": CONTROLLER_HIDDEN_LAYERS,
                    "prior_weight": CONTROLLER_PRIOR_WEIGHT,
                    "compatibility_score_weight": 0.0,
                    "contact_physics_losses": "disabled",
                    "slide_excess_losses": "disabled",
                    "compile": "disabled",
                    "clip_sampling": "mixed per-row random clips",
                    "rollout_schedule": ROLLOUT_SCHEDULE,
                    "mixed_rollout_cohort_schedule": MIXED_ROLLOUT_COHORT_SCHEDULE,
                    "mixed_rollout_cohort_weights": MIXED_ROLLOUT_COHORT_WEIGHTS,
                    "stop_on_final_stall": CONTROLLER_STOP_ON_FINAL_STALL,
                    "save_live_every_epochs": CONTROLLER_SAVE_LIVE_EVERY_EPOCHS,
                },
                "gt_rule": {
                    "starts_per_clip": GT_RULE_STARTS_PER_CLIP,
                    "rollout_steps": GT_RULE_ROLLOUT_STEPS,
                    "tolerance_p95": GT_RULE_TOLERANCE_P95,
                },
                "pose_representation": args.pose_representation,
            },
        },
    )


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


def evaluate_gt_diff_rule(
    args: argparse.Namespace,
    run_root: Path,
    cycle: int,
    controller_checkpoint: Path,
    previous_p95: float | None,
) -> tuple[dict[str, object], bool]:
    device = torch.device(args.eval_device)
    controller_ckpt = torch.load(controller_checkpoint, map_location="cpu", weights_only=False)
    locomotion_cfg = tl.TrainConfig()
    model_aware.apply_config_dict(locomotion_cfg, controller_ckpt.get("config", {}))
    locomotion_cfg.device = str(device)
    locomotion_cfg.use_torch_compile = False
    tl.apply_cuda_performance_settings(locomotion_cfg, device)
    clip_specs = tl.clip_specs_from_folders(
        args.periodic_folder_path,
        args.periodic_folder_path or None,
        args.nonperiodic_folder_path or None,
    )
    clips = tl.load_clips_from_specs(clip_specs, locomotion_cfg)
    controller = model_aware.load_controller(controller_checkpoint, clips, locomotion_cfg, device)
    generated = model_aware.collect_generated_feature_batch(
        controller,
        clips,
        locomotion_cfg,
        device,
        GT_RULE_STARTS_PER_CLIP,
        GT_RULE_ROLLOUT_STEPS,
    )
    values = generated.gt_difference_sum_m.float()
    p95 = float(torch.quantile(values, 0.95).item())
    row: dict[str, object] = {
        "cycle": cycle,
        "mean_gt_diff": float(values.mean().item()),
        "p95_gt_diff": p95,
        "max_gt_diff": float(values.max().item()),
        "rollout_count": int(values.numel()),
        "starts_per_clip": GT_RULE_STARTS_PER_CLIP,
        "rollout_steps": GT_RULE_ROLLOUT_STEPS,
        "previous_p95_gt_diff": "" if previous_p95 is None else previous_p95,
        "tolerance_p95": GT_RULE_TOLERANCE_P95,
    }
    if previous_p95 is None:
        decision = "baseline"
        should_stop = False
        delta = 0.0
    else:
        delta = p95 - previous_p95
        should_stop = delta > GT_RULE_TOLERANCE_P95
        decision = "break" if should_stop else "accept"
    row["delta_p95_gt_diff"] = delta
    row["decision"] = decision
    append_csv(run_root / "gt_diff_summary.csv", row)
    write_json(run_root / "gt_rule_decision.json", row)
    print(
        f"GT rule cycle={cycle} mean={row['mean_gt_diff']:.6f} "
        f"p95={p95:.6f} max={row['max_gt_diff']:.6f} decision={decision}",
        flush=True,
    )
    return row, should_stop


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
        args.periodic_folder_path,
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
        str(AE_LEARNING_RATE),
        "--batch-size",
        str(AE_BATCH_SIZE),
        "--max-epochs",
        str(AE_EPOCHS),
        "--input-noise-std",
        str(AE_INPUT_NOISE_STD),
        "--fake-margin",
        str(AE_FAKE_MARGIN),
        "--fake-weight",
        str(AE_FAKE_WEIGHT),
        "--real-weight",
        str(AE_REAL_WEIGHT),
        "--compatibility-real-weight",
        "0",
        "--compatibility-fake-weight",
        "0",
        "--fake-starts-per-clip",
        str(AE_FAKE_STARTS_PER_CLIP),
        "--fake-rollout-steps",
        str(AE_FAKE_ROLLOUT_STEPS),
        "--fake-buffer-path",
        str(fake_buffer_path),
        "--fake-buffer-max-rows",
        str(AE_FAKE_BUFFER_MAX_ROWS),
        "--fake-buffer-source-cycle",
        str(cycle),
        "--fake-replay-decay",
        str(AE_FAKE_REPLAY_DECAY),
        "--hard-negative-keep-fraction",
        str(AE_HARD_NEGATIVE_KEEP_FRACTION),
        "--hard-negative-mode",
        AE_HARD_NEGATIVE_MODE,
        "--pose-representation",
        args.pose_representation,
        "--eval-every-epochs",
        str(AE_EVAL_EVERY_EPOCHS),
        "--stall-patience-epochs",
        str(AE_STALL_PATIENCE_EPOCHS),
    ]
    if AE_CONDITIONAL_ROOT_WINDOW:
        cmd.append("--conditional-root-window")
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
        str(CONTROLLER_PRIOR_WEIGHT),
        "--compatibility-score-weight",
        "0",
        "--resume-checkpoint",
        str(controller_checkpoint),
        "--run-name",
        run_name,
        "--device",
        args.device,
        "--hidden-dim",
        str(CONTROLLER_HIDDEN_DIM),
        "--pose-representation",
        args.pose_representation,
        "--num-hidden-layers",
        str(CONTROLLER_HIDDEN_LAYERS),
        "--learning-rate",
        str(CONTROLLER_LEARNING_RATE),
        "--batch-size",
        str(CONTROLLER_BATCH_SIZE),
        "--training-loop",
        "agents",
        "--agent-sampling",
        "random",
        "--agent-batches-per-epoch",
        str(CONTROLLER_AGENT_BATCHES_PER_EPOCH),
        "--periodic-sampling-weight",
        str(PERIODIC_SAMPLING_WEIGHT),
        "--nonperiodic-sampling-weight",
        str(NONPERIODIC_SAMPLING_WEIGHT),
        "--max-epochs",
        str(CONTROLLER_EPOCHS),
        "--rollout-schedule",
        ROLLOUT_SCHEDULE,
        "--curriculum-max-epochs-per-stage",
        str(CURRICULUM_MAX_EPOCHS_PER_STAGE),
        "--curriculum-stall-patience-epochs",
        str(CURRICULUM_STALL_PATIENCE_EPOCHS),
        "--curriculum-min-epochs",
        str(CURRICULUM_MIN_EPOCHS),
        "--curriculum-min-delta",
        str(CURRICULUM_MIN_DELTA),
        "--mixed-rollout-cohorts",
        "--mixed-rollout-cohort-schedule",
        MIXED_ROLLOUT_COHORT_SCHEDULE,
        "--mixed-rollout-cohort-weights",
        MIXED_ROLLOUT_COHORT_WEIGHTS,
        "--no-contact-physics-losses",
        "--no-live-viewer",
        "--no-visual-reporter",
        "--diagnostic-metrics-every-epochs",
        "0",
        "--timed-checkpoint-interval-minutes",
        str(TIMED_CHECKPOINT_INTERVAL_MINUTES),
        "--save-live-every-epochs",
        str(CONTROLLER_SAVE_LIVE_EVERY_EPOCHS),
    ]
    if CONTROLLER_STOP_ON_FINAL_STALL:
        cmd.append("--stop-on-final-stall")
    run_command(cmd, PROJECT_ROOT, run_root / "logs" / f"controller_cycle_{cycle:02d}.log")
    ckpt = PROJECT_ROOT / "training" / "runs" / run_name / "checkpoints" / f"checkpoint_best_k{EXPECTED_BEST_K:02d}.pt"
    if not ckpt.exists():
        ckpt = PROJECT_ROOT / "training" / "runs" / run_name / "checkpoints" / "checkpoint_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Canonical Stepper self-adversarial reconstruction-only recipe. "
            "Only dataset/start/run controls are exposed; fragile training flags are hard-coded."
        )
    )
    parser.add_argument("--periodic-folder-path", required=True)
    parser.add_argument("--nonperiodic-folder-path", required=True)
    parser.add_argument("--start-controller-checkpoint", required=True)
    parser.add_argument("--start-prior-checkpoint", required=True)
    parser.add_argument("--run-name", default="gtdiff_recononly_recipe")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--pose-representation", choices=("rot6", "ik_markers"), default="rot6")
    parser.add_argument("--key-clip-names", default=DEFAULT_KEY_CLIPS)
    parser.add_argument("--no-evaluate", action="store_true")
    args = parser.parse_args()

    run_root = PROJECT_ROOT / "training" / "runs" / tl.date_prefixed_run_name(args.run_name)
    run_root.mkdir(parents=True, exist_ok=True)
    fake_buffer = run_root / "geometric_fake_replay_buffer.pt"
    write_recipe_config(run_root, args)

    controller = resolve_path(args.start_controller_checkpoint)
    prior = resolve_path(args.start_prior_checkpoint)
    rows: list[dict[str, object]] = []
    print(f"self_adversarial run_root={run_root}", flush=True)
    print(
        f"recipe=GT-diff reconstruction-only geometric fake replay decay={AE_FAKE_REPLAY_DECAY:g} "
        f"conditional_root_window={AE_CONDITIONAL_ROOT_WINDOW} pose_representation={args.pose_representation}",
        flush=True,
    )
    print(f"start controller={controller}", flush=True)
    print(f"start prior={prior}", flush=True)
    previous_gt_p95: float | None = None

    for cycle in range(1, args.cycles + 1):
        cycle_start = time.perf_counter()
        print(f"\n=== cycle {cycle}/{args.cycles}: update AE ===", flush=True)
        prior = train_model_aware_prior(args, run_root, cycle, controller, prior, fake_buffer)
        print(f"\n=== cycle {cycle}/{args.cycles}: train controller ===", flush=True)
        controller = train_controller(args, run_root, cycle, controller, prior)
        if not args.no_evaluate:
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
        gt_row, should_stop = evaluate_gt_diff_rule(args, run_root, cycle, controller, previous_gt_p95)
        previous_gt_p95 = float(gt_row["p95_gt_diff"])
        print(f"cycle {cycle} done controller={controller} prior={prior}", flush=True)
        if should_stop:
            print(f"stopping after cycle {cycle}: GT p95 broke monotonic rule", flush=True)
            break

    print(f"finished self-adversarial loop run_root={run_root}", flush=True)


if __name__ == "__main__":
    main()
