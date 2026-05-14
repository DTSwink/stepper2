from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

import train_locomotion as tl
import train_locomotion_ae_prior as ae_train
import transition_autoencoder as tae


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def apply_config_dict(cfg: tl.TrainConfig, values: dict) -> None:
    valid = {field.name for field in fields(tl.TrainConfig)}
    for key, value in values.items():
        if key not in valid:
            continue
        current = getattr(cfg, key)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(cfg, key, value)


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {' '.join(command)}")


def latest_ae_report(run_dir: Path) -> dict[str, float]:
    report_path = run_dir / "model_aware_ae_report.csv"
    with report_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No AE report rows in {report_path}")
    return {key: float(value) for key, value in rows[-1].items()}


def load_controller(checkpoint_path: Path, cfg: tl.TrainConfig, clip: tl.MotionClip, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    apply_config_dict(cfg, checkpoint.get("config", {}))
    cfg.device = str(device)
    cfg.use_torch_compile = False
    input_dim, output_dim = tl.make_batch_dims(clip, cfg)
    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_model_ae_score(
    folder_path: Path,
    model_checkpoint: Path,
    prior_checkpoints: list[Path],
    device: torch.device,
    cyclic_animation: bool,
    rollout_k: int,
    batch_size: int,
    batches: int,
    seed: int,
) -> dict[str, float]:
    cfg = tl.TrainConfig()
    cfg.cyclic_animation = cyclic_animation
    initial_clips = tl.load_clips(folder_path, cfg)
    model = load_controller(model_checkpoint, cfg, initial_clips[0], device)
    cfg.cyclic_animation = cyclic_animation
    cfg.batch_size = batch_size
    cfg.ae_score_loss = "mse"
    cfg.ae_huber_delta = 1.0
    clips = tl.load_clips(folder_path, cfg)
    priors = ae_train.load_prior_bundle(prior_checkpoints, device)

    rng = random.Random(seed)
    losses = []
    scores = []
    motions = []
    for batch_i in range(batches):
        clip_ids = []
        starts = []
        for row in range(batch_size):
            ci = (batch_i * batch_size + row) % len(clips)
            clip = clips[ci]
            max_start = max(1, clip.cyclic_period - 1 if cyclic_animation else clip.T - rollout_k - 1)
            clip_ids.append(ci)
            starts.append(rng.randint(1, max_start))
        loss, scalars = ae_train.run_batch_ae(
            model,
            priors,
            clips,
            (torch.tensor(clip_ids, dtype=torch.long), torch.tensor(starts, dtype=torch.long)),
            cfg,
            rollout_k,
            device,
            compute_diagnostics=False,
            compatibility_score_weight=0.0,
        )
        losses.append(float(loss.detach().cpu()))
        scores.append(float(scalars["ae_score"]))
        motions.append(float(scalars["canon_step_rms"]))
    return {
        "loss": float(np.mean(losses)),
        "ae_score": float(np.mean(scores)),
        "motion_rms": float(np.mean(motions)),
    }


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_foot_slide_monitor(
    folder_path: str,
    checkpoint_path: Path,
    label: str,
    output_csv: Path,
    device: str,
    cyclic_animation: bool,
    log_path: Path,
) -> dict[str, float]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "training" / "inspect_foot_sliding.py"),
        "--folder-path",
        folder_path,
        "--checkpoint-path",
        str(checkpoint_path),
        "--label",
        label,
        "--output-csv",
        str(output_csv),
        "--device",
        device,
    ]
    command.append("--cyclic-animation" if cyclic_animation else "--no-cyclic-animation")
    run_command(command, log_path)
    with output_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pred = np.asarray([float(row["pred_contact_p95_mps"]) for row in rows], dtype=np.float64)
    gt = np.asarray([float(row["gt_contact_p95_mps"]) for row in rows], dtype=np.float64)
    return {
        "foot_slide_pred_contact_p95_mean": float(pred.mean()),
        "foot_slide_gt_contact_p95_mean": float(gt.mean()),
        "foot_slide_delta_contact_p95_mean": float(pred.mean() - gt.mean()),
        "foot_slide_pred_contact_p95_max": float(pred.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Alternate model-aware AE training and AE-prior model finetuning.")
    parser.add_argument("--folder-path", default="ue5/animations_omni_only/npz_final")
    parser.add_argument("--initial-model-checkpoint", required=True)
    parser.add_argument("--initial-prior-checkpoint", required=True)
    parser.add_argument(
        "--anchor-prior-checkpoint",
        action="append",
        default=[],
        help="Frozen AE prior checkpoint to average with each dynamically updated AE during model training.",
    )
    parser.add_argument("--run-prefix", default="modelaware_loop")
    parser.add_argument("--output-dir", default="training/runs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cyclic-animation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--rollout-k", type=int, default=111)
    parser.add_argument("--model-learning-rate", type=float, default=5e-6)
    parser.add_argument("--model-max-epochs", type=int, default=120)
    parser.add_argument("--model-stall-patience-epochs", type=int, default=18)
    parser.add_argument("--model-min-epochs", type=int, default=20)
    parser.add_argument("--model-min-delta", type=float, default=1e-5)
    parser.add_argument("--model-min-improvement", type=float, default=0.001)
    parser.add_argument("--model-batch-size", type=int, default=256)
    parser.add_argument("--agent-batches-per-epoch", type=int, default=1)
    parser.add_argument("--ae-max-epochs", type=int, default=120)
    parser.add_argument("--ae-learning-rate", type=float, default=1e-4)
    parser.add_argument("--ae-batch-size", type=int, default=512)
    parser.add_argument("--ae-fake-margin", type=float, default=0.02)
    parser.add_argument("--ae-fake-weight", type=float, default=1.0)
    parser.add_argument("--ae-fake-starts-per-clip", type=int, default=16)
    parser.add_argument("--ae-fake-rollout-steps", type=int, default=0)
    parser.add_argument("--ae-stall-patience-epochs", type=int, default=35)
    parser.add_argument("--ae-min-fake-margin-success", type=float, default=0.60)
    parser.add_argument("--ae-min-energy-gap", type=float, default=0.005)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    output_dir = tl.resolve_path(args.output_dir)
    folder = tl.resolve_path(args.folder_path)
    device = torch.device(args.device)
    python = sys.executable
    loop_dir = output_dir / args.run_prefix
    loop_dir.mkdir(parents=True, exist_ok=True)

    current_model = tl.resolve_path(args.initial_model_checkpoint)
    current_prior = tl.resolve_path(args.initial_prior_checkpoint)
    anchor_priors = [tl.resolve_path(path) for path in args.anchor_prior_checkpoint]
    summary_rows: list[dict[str, object]] = []

    baseline_slide = run_foot_slide_monitor(
        args.folder_path,
        current_model,
        f"{args.run_prefix}_baseline",
        loop_dir / "foot_sliding_baseline.csv",
        args.device,
        args.cyclic_animation,
        loop_dir / "foot_sliding_baseline.log",
    )

    for cycle in range(1, args.cycles + 1):
        ae_run = f"{args.run_prefix}_ae_cycle{cycle:02d}"
        ae_run_dir = output_dir / ae_run
        ae_cmd = [
            python,
            str(PROJECT_ROOT / "training" / "train_model_aware_transition_ae.py"),
            "--folder-path",
            args.folder_path,
            "--model-checkpoint",
            str(current_model),
            "--init-prior-checkpoint",
            str(current_prior),
            "--run-name",
            ae_run,
            "--output-dir",
            args.output_dir,
            "--learning-rate",
            str(args.ae_learning_rate),
            "--max-epochs",
            str(args.ae_max_epochs),
            "--batch-size",
            str(args.ae_batch_size),
            "--fake-margin",
            str(args.ae_fake_margin),
            "--fake-weight",
            str(args.ae_fake_weight),
            "--fake-starts-per-clip",
            str(args.ae_fake_starts_per_clip),
            "--fake-rollout-steps",
            str(args.ae_fake_rollout_steps),
            "--stall-patience-epochs",
            str(args.ae_stall_patience_epochs),
            "--seed",
            str(args.seed + cycle),
            "--device",
            args.device,
        ]
        if args.cyclic_animation:
            ae_cmd.append("--cyclic-animation")
        else:
            ae_cmd.append("--no-cyclic-animation")
        run_command(ae_cmd, loop_dir / f"{ae_run}.log")

        ae_report = latest_ae_report(ae_run_dir)
        ae_can_tell = (
            ae_report["fake_margin_success"] >= args.ae_min_fake_margin_success
            and ae_report["gap"] >= args.ae_min_energy_gap
        )
        current_prior = ae_run_dir / "checkpoints" / "checkpoint_best.pt"

        eval_seed = args.seed + 1000 + cycle
        initial_eval = evaluate_model_ae_score(
            folder,
            current_model,
            [current_prior, *anchor_priors],
            device,
            args.cyclic_animation,
            args.rollout_k,
            args.eval_batch_size,
            args.eval_batches,
            eval_seed,
        )

        row: dict[str, object] = {
            "cycle": cycle,
            "ae_run": ae_run,
            "ae_fake_margin_success": ae_report["fake_margin_success"],
            "ae_energy_gap": ae_report["gap"],
            "ae_real_energy_mean": ae_report["real_energy_mean"],
            "ae_fake_energy_mean": ae_report["fake_energy_mean"],
            "anchor_prior_count": len(anchor_priors),
            "model_initial_ae_score": initial_eval["loss"],
            "model_initial_motion_rms": initial_eval["motion_rms"],
            "baseline_foot_slide_pred_contact_p95_mean": baseline_slide["foot_slide_pred_contact_p95_mean"],
            "baseline_foot_slide_gt_contact_p95_mean": baseline_slide["foot_slide_gt_contact_p95_mean"],
            "stop_reason": "",
        }
        if not ae_can_tell:
            row["stop_reason"] = "ae_cannot_classify"
            summary_rows.append(row)
            write_summary(loop_dir / "summary.csv", summary_rows)
            print("Stopping: model-aware AE can no longer separate generated transitions from real transitions.", flush=True)
            break

        model_run = f"{args.run_prefix}_model_cycle{cycle:02d}"
        model_run_dir = output_dir / model_run
        model_cmd = [
            python,
            str(PROJECT_ROOT / "training" / "train_locomotion_ae_prior.py"),
            "--folder-path",
            args.folder_path,
            "--prior-checkpoint",
            str(current_prior),
            "--resume-checkpoint",
            str(current_model),
            "--run-name",
            model_run,
            "--device",
            args.device,
            "--hidden-dim",
            "512",
            "--num-hidden-layers",
            "2",
            "--learning-rate",
            str(args.model_learning_rate),
            "--batch-size",
            str(args.model_batch_size),
            "--training-loop",
            "agents",
            "--agent-sampling",
            "random",
            "--agent-batch-clips",
            "1",
            "--agent-batches-per-epoch",
            str(args.agent_batches_per_epoch),
            "--rollout-schedule",
            str(args.rollout_k),
            "--max-epochs",
            str(args.model_max_epochs),
            "--curriculum-max-epochs-per-stage",
            str(args.model_max_epochs),
            "--curriculum-stall-patience-epochs",
            str(args.model_stall_patience_epochs),
            "--curriculum-min-epochs",
            str(args.model_min_epochs),
            "--curriculum-min-delta",
            str(args.model_min_delta),
            "--diagnostic-metrics-every-epochs",
            "10",
            "--save-live-every-epochs",
            "20",
            "--no-visual-reporter",
            "--no-contact-physics-losses",
            "--no-compile",
            "--stop-on-final-stall",
        ]
        for anchor_prior in anchor_priors:
            model_cmd.extend(["--extra-prior-checkpoint", str(anchor_prior)])
        if args.cyclic_animation:
            model_cmd.append("--cyclic-animation")
        else:
            model_cmd.append("--no-cyclic-animation")
        run_command(model_cmd, loop_dir / f"{model_run}.log")

        best_model = model_run_dir / "checkpoints" / "checkpoint_best.pt"
        last_model = model_run_dir / "checkpoints" / "checkpoint_last.pt"
        final_model = best_model if best_model.exists() else last_model
        final_eval = evaluate_model_ae_score(
            folder,
            final_model,
            [current_prior, *anchor_priors],
            device,
            args.cyclic_animation,
            args.rollout_k,
            args.eval_batch_size,
            args.eval_batches,
            eval_seed,
        )
        improvement = initial_eval["loss"] - final_eval["loss"]
        slide_metrics = run_foot_slide_monitor(
            args.folder_path,
            final_model,
            f"{args.run_prefix}_cycle{cycle:02d}",
            loop_dir / f"foot_sliding_cycle{cycle:02d}.csv",
            args.device,
            args.cyclic_animation,
            loop_dir / f"foot_sliding_cycle{cycle:02d}.log",
        )
        row.update(
            {
                "model_run": model_run,
                "model_checkpoint": str(final_model),
                "model_final_ae_score": final_eval["loss"],
                "model_final_motion_rms": final_eval["motion_rms"],
                "model_ae_improvement": improvement,
                **slide_metrics,
            }
        )
        if improvement < args.model_min_improvement:
            row["stop_reason"] = "model_cannot_reduce_ae_loss"
            summary_rows.append(row)
            write_summary(loop_dir / "summary.csv", summary_rows)
            print("Stopping: model could not reduce the current AE loss enough from its initialized state.", flush=True)
            break

        summary_rows.append(row)
        write_summary(loop_dir / "summary.csv", summary_rows)
        current_model = final_model

    (loop_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Wrote loop summary to {loop_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
