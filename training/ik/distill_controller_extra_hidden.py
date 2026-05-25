from __future__ import annotations

import argparse
import json
import math
import os
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

try:
    from .bootstrap import PROJECT_ROOT, ensure_paths
    from .naming import checkpoint_path, ik_run_id
    from . import ik_core as tl
    from . import train_simple_ae_controller as ctl
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    from naming import checkpoint_path, ik_run_id
    import ik_core as tl
    import train_simple_ae_controller as ctl

ensure_paths()


RUNS_DIR = PROJECT_ROOT / "training" / "runs"
DEFAULT_PERIODIC_FOLDER = PROJECT_ROOT / "ue5" / "animations_omni_only_full" / "npz_final"
DEFAULT_NONPERIODIC_FOLDER = PROJECT_ROOT / "ue5" / "animations_transitions_only_full_trimmed" / "npz_final"

BATCH_SIZE = 4096
ROLLOUT_K = 64
TRAIN_STEPS = 50000
LEARNING_RATE = 5e-5
LOG_EVERY_STEPS = 100
CHECKPOINT_EVERY_MINUTES = 30.0
EXTRA_HIDDEN_LAYERS = 1


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_teacher_checkpoint(path: Path, device: torch.device) -> tuple[dict, tl.TrainConfig]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in ckpt or "config" not in ckpt:
        raise ValueError(f"Not an IK controller checkpoint: {path}")
    cfg = tl.TrainConfig()
    ctl.apply_config_dict(cfg, ckpt.get("config", {}))
    cfg.pose_representation = tl.IK_POSE_REPRESENTATION
    cfg.device = str(device)
    cfg.batch_size = int(BATCH_SIZE)
    cfg.use_torch_compile = False
    cfg.live_viewer = False
    cfg.visual_reporter = False
    return ckpt, cfg


def load_store(
    cfg: tl.TrainConfig,
    device: torch.device,
    npz_text: str | None,
    periodic_text: str | None,
    nonperiodic_text: str | None,
) -> tuple[list[tl.MotionClip], ctl.SimpleClipStore, list[tuple[Path, bool]]]:
    specs = ctl.resolve_clip_specs(npz_text, periodic_text, nonperiodic_text)
    clips = ctl.load_clips(specs, cfg)
    store = ctl.SimpleClipStore(clips, cfg, device)
    return clips, store, specs


def make_teacher_and_student(
    teacher_ckpt: dict,
    teacher_cfg: tl.TrainConfig,
    input_dim: int,
    output_dim: int,
    extra_hidden_layers: int,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, tl.TrainConfig, list[str]]:
    teacher = tl.MLPController(input_dim, output_dim, teacher_cfg).to(device)
    teacher.load_state_dict(teacher_ckpt["model"])
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    student_cfg = deepcopy(teacher_cfg)
    student_cfg.num_hidden_layers = int(teacher_cfg.num_hidden_layers) + int(extra_hidden_layers)
    student_cfg.hidden_dim = int(teacher_cfg.hidden_dim)
    student = tl.MLPController(input_dim, output_dim, student_cfg).to(device)
    copied = copy_teacher_prefix_into_deeper_student(teacher, student, int(teacher_cfg.num_hidden_layers), int(student_cfg.num_hidden_layers))
    return teacher, student, student_cfg, copied


def copy_teacher_prefix_into_deeper_student(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    teacher_hidden_layers: int,
    student_hidden_layers: int,
) -> list[str]:
    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    copied: list[str] = []

    hidden_entries = int(teacher_hidden_layers) * 3
    for name, value in teacher_state.items():
        parts = name.split(".")
        if len(parts) >= 3 and parts[0] == "net" and parts[1].isdigit() and int(parts[1]) < hidden_entries:
            if name in student_state and student_state[name].shape == value.shape:
                student_state[name] = value.detach().clone()
                copied.append(name)

    teacher_output_prefix = f"net.{int(teacher_hidden_layers) * 3}."
    student_output_prefix = f"net.{int(student_hidden_layers) * 3}."
    for name, value in teacher_state.items():
        if not name.startswith(teacher_output_prefix):
            continue
        target_name = student_output_prefix + name[len(teacher_output_prefix) :]
        if target_name in student_state and student_state[target_name].shape == value.shape:
            student_state[target_name] = value.detach().clone()
            copied.append(f"{name} -> {target_name}")

    student.load_state_dict(student_state)
    return copied


def distill_rollout_loss(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    store: ctl.SimpleClipStore,
    cfg: tl.TrainConfig,
    start_pools: dict[int, ctl.StartPool],
    rollout_k: int,
    batch_size: int,
) -> torch.Tensor:
    max_k = max(1, int(rollout_k))
    effective_k = ctl.sample_effective_rollout_k(batch_size, max_k, store.device)
    clip_ids, starts = ctl.sample_rollout_rows(start_pools, effective_k)

    prev_vec, prev_pelvis, prev_payload = ctl.target_state(store, clip_ids, starts - 1)
    cur_vec, cur_pelvis, cur_payload = ctl.target_state(store, clip_ids, starts)
    cur_idx = starts

    total = torch.zeros((), dtype=torch.float32, device=store.device)
    total_weight = torch.zeros((), dtype=torch.float32, device=store.device)

    for step in range(max_k):
        active = step < effective_k
        if not bool(active.any()):
            break

        inp = ctl.build_controller_input(
            store, clip_ids, cur_idx, prev_vec, cur_vec, prev_pelvis, cur_pelvis, prev_payload, cur_payload
        )
        with torch.no_grad():
            teacher_raw = ctl.model_forward(teacher, inp, cur_vec, cfg)
            teacher_vec = ctl.clean_output_vector(teacher_raw, store)
            teacher_state_vec, teacher_pelvis, teacher_payload = ctl.advance_transition_state(
                store, clip_ids, cur_idx, teacher_vec
            )

        student_raw = ctl.model_forward(student, inp, cur_vec, cfg)
        student_vec = ctl.clean_output_vector(student_raw, store)

        per_row = (student_vec - teacher_vec).square().mean(dim=-1)
        active_f = active.to(dtype=per_row.dtype)
        total = total + (per_row * active_f).sum()
        total_weight = total_weight + active_f.sum()

        continuing = (step + 1) < effective_k
        if not bool(continuing.any()):
            break

        reset = ctl.training_reset_rows(store, clip_ids, cur_idx, continuing)
        advance = continuing & (~reset)
        reset_starts = ctl.sample_same_clip_training_starts(store, clip_ids)
        reset_prev_vec, reset_prev_pelvis, reset_prev_payload = ctl.target_state(store, clip_ids, reset_starts - 1)
        reset_cur_vec, reset_cur_pelvis, reset_cur_payload = ctl.target_state(store, clip_ids, reset_starts)

        reset_mask = reset[:, None]
        advance_mask = advance[:, None]
        prev_vec = torch.where(reset_mask, reset_prev_vec, torch.where(advance_mask, cur_vec, prev_vec))
        prev_pelvis = torch.where(reset_mask, reset_prev_pelvis, torch.where(advance_mask, cur_pelvis, prev_pelvis))
        prev_payload = torch.where(reset_mask, reset_prev_payload, torch.where(advance_mask, cur_payload, prev_payload))
        cur_vec = torch.where(reset_mask, reset_cur_vec, torch.where(advance_mask, teacher_state_vec, cur_vec))
        cur_pelvis = torch.where(reset_mask, reset_cur_pelvis, torch.where(advance_mask, teacher_pelvis, cur_pelvis))
        cur_payload = torch.where(reset_mask, reset_cur_payload, torch.where(advance_mask, teacher_payload, cur_payload))
        cur_idx = torch.where(reset, reset_starts, torch.where(continuing, cur_idx + 1, cur_idx))

    return total / total_weight.clamp_min(1.0)


@torch.no_grad()
def distill_probe_loss(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    store: ctl.SimpleClipStore,
    cfg: tl.TrainConfig,
    start_pools: dict[int, ctl.StartPool],
    rollout_k: int,
    batch_size: int,
) -> torch.Tensor:
    teacher_was_training = teacher.training
    student_was_training = student.training
    teacher.eval()
    student.eval()
    loss = distill_rollout_loss(teacher, student, store, cfg, start_pools, rollout_k, batch_size)
    teacher.train(teacher_was_training)
    student.train(student_was_training)
    return loss.detach()


def save_distill_checkpoint(
    run_dir: Path,
    run_id: str,
    tag: str,
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best: float,
    student_cfg: tl.TrainConfig,
    metadata: dict,
) -> Path:
    path = checkpoint_path(run_dir, run_id, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tl.checkpoint_payload(student, optimizer, step, best, ROLLOUT_K, student_cfg, metadata), path)
    return path


def write_readable_config(
    run_dir: Path,
    args: argparse.Namespace,
    teacher_cfg: tl.TrainConfig,
    student_cfg: tl.TrainConfig,
    metadata: dict,
) -> None:
    important = {
        "mode": "controller_distillation",
        "teacher_checkpoint": str(metadata["teacher_checkpoint"]),
        "run_label": str(args.run_label),
        "distillation_target": "teacher cleaned next IK output on teacher rollout states",
        "teacher_hidden_dim": int(teacher_cfg.hidden_dim),
        "student_hidden_dim": int(student_cfg.hidden_dim),
        "teacher_hidden_layers": int(teacher_cfg.num_hidden_layers),
        "student_hidden_layers": int(student_cfg.num_hidden_layers),
        "extra_hidden_layers": int(args.extra_hidden_layers),
        "rollout_k": int(args.rollout_k),
        "batch_size": int(args.batch_size),
        "steps": int(args.steps),
        "learning_rate": float(args.learning_rate),
        "checkpoint_every_minutes": float(args.checkpoint_every_minutes),
        "npz": str(args.npz or ""),
        "periodic_folder": str(args.periodic_folder),
        "nonperiodic_folder": str(args.nonperiodic_folder),
    }
    payload = {
        "important": important,
        "args": vars(args),
        "metadata": metadata,
        "teacher_config": asdict(teacher_cfg),
        "student_config": asdict(student_cfg),
    }
    (run_dir / "config_readable.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def assert_tensorboard_event_file(tb_dir: Path) -> None:
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline:
        if any(path.stat().st_size > 0 for path in tb_dir.glob("events.out.tfevents*")):
            return
        time.sleep(0.05)
    raise RuntimeError(f"TensorBoard event file was not created in {tb_dir}")


def default_teacher_checkpoint_from_env() -> str:
    return os.environ.get("STEPPER_TEACHER_CHECKPOINT", "")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill an IK controller checkpoint into a same-width controller with one extra hidden layer."
    )
    parser.add_argument("--teacher-checkpoint", default=default_teacher_checkpoint_from_env())
    parser.add_argument("--npz", default=None)
    parser.add_argument("--periodic-folder", default=str(DEFAULT_PERIODIC_FOLDER))
    parser.add_argument("--nonperiodic-folder", default=str(DEFAULT_NONPERIODIC_FOLDER))
    parser.add_argument("--run-label", default="distill_extra_hidden")
    parser.add_argument("--extra-hidden-layers", type=int, default=EXTRA_HIDDEN_LAYERS)
    parser.add_argument("--rollout-k", type=int, default=ROLLOUT_K)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY_STEPS)
    parser.add_argument("--checkpoint-every-minutes", type=float, default=CHECKPOINT_EVERY_MINUTES)
    args = parser.parse_args()

    if not args.teacher_checkpoint:
        raise ValueError("Pass --teacher-checkpoint or set STEPPER_TEACHER_CHECKPOINT.")
    if int(args.extra_hidden_layers) < 1:
        raise ValueError("--extra-hidden-layers must be >= 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    teacher_path = resolve_path(args.teacher_checkpoint)
    teacher_ckpt, teacher_cfg = load_teacher_checkpoint(teacher_path, device)
    teacher_cfg.batch_size = int(args.batch_size)
    clips, store, specs = load_store(teacher_cfg, device, args.npz, args.periodic_folder, args.nonperiodic_folder)
    input_dim, output_dim = tl.make_batch_dims(clips[0], teacher_cfg)
    teacher, student, student_cfg, copied_names = make_teacher_and_student(
        teacher_ckpt, teacher_cfg, input_dim, output_dim, int(args.extra_hidden_layers), device
    )

    optimizer = ctl.make_adamw(student.parameters(), float(args.learning_rate), device)
    rollout_values = ctl.rollout_values_for(int(args.rollout_k)) if ctl.mixed_rollout_enabled(int(args.rollout_k)) else (int(args.rollout_k),)
    start_pools = ctl.build_training_start_pools(store, rollout_values)
    max_pool = start_pools[int(args.rollout_k)]
    batch_size = min(int(args.batch_size), int(max_pool.row_count))
    rollout_stats = ctl.rollout_stat_summary(batch_size, int(args.rollout_k))

    run_id = ik_run_id(args.run_label)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = run_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir), flush_secs=1)

    metadata = {
        "teacher_checkpoint": str(teacher_path),
        "teacher_epoch": int(teacher_ckpt.get("epoch", 0)),
        "teacher_rollout_k": int(teacher_ckpt.get("rollout_k", 0)),
        "teacher_best_val": float(teacher_ckpt.get("best_val", math.inf)),
        "npz_paths": [str(path) for path, _cyclic in specs],
        "npz_folders": [{"path": str(path.parent), "cyclic": bool(cyclic)} for path, cyclic in specs],
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
        "clip_count": int(len(clips)),
        "batch_size": int(batch_size),
        "rollout_values": [int(k) for k in rollout_values],
        "rollout_stats": rollout_stats,
        "pool_rows": {str(int(k)): int(pool.row_count) for k, pool in start_pools.items()},
        "copied_teacher_parameters": copied_names,
        "policy": {
            "loss": "distill_teacher_clean_output_mse",
            "state_distribution": "teacher_rollout_with_same_clip_resets",
            "student_extra_hidden_layers": int(args.extra_hidden_layers),
            "student_same_width_as_teacher": True,
            "pose_representation": tl.IK_POSE_REPRESENTATION,
            "output_reference_root": tl.OUTPUT_REFERENCE_ROOT,
            "output_prediction_mode": tl.normalized_output_prediction_mode(),
            "state_reference_root": tl.STATE_REFERENCE_ROOT,
        },
    }
    write_readable_config(run_dir, args, teacher_cfg, student_cfg, metadata)
    writer.add_text("config/json", f"```json\n{json.dumps({'important': metadata['policy'], 'metadata': metadata}, indent=2)}\n```", 0)
    writer.add_scalar("run/started", 1.0, 0)
    writer.add_scalar("curriculum/rollout_k", int(args.rollout_k), 0)
    writer.add_scalar("curriculum/effective_rollout_k_mean", float(rollout_stats["effective_k_mean"]), 0)
    writer.add_scalar("curriculum/effective_rollout_k_max", float(rollout_stats["effective_k_max"]), 0)
    writer.flush()
    assert_tensorboard_event_file(tb_dir)
    ctl.refresh_tensorboard_async()

    best = math.inf
    last_periodic = time.perf_counter()
    init_path = save_distill_checkpoint(run_dir, run_id, "init", student, optimizer, 0, best, student_cfg, metadata)
    print(f"saved init checkpoint {init_path}", flush=True)
    print(
        f"{run_id}: teacher_layers={teacher_cfg.num_hidden_layers} student_layers={student_cfg.num_hidden_layers} "
        f"hidden_dim={student_cfg.hidden_dim} clips={len(clips)} batch={batch_size} K={args.rollout_k}",
        flush=True,
    )

    start_time = time.perf_counter()
    for step in range(1, int(args.steps) + 1):
        student.train()
        optimizer.zero_grad(set_to_none=True)
        loss = distill_rollout_loss(teacher, student, store, teacher_cfg, start_pools, int(args.rollout_k), batch_size)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        best = min(best, loss_value)
        if step == 1 or step % int(args.log_every) == 0:
            elapsed = time.perf_counter() - start_time
            writer.add_scalar("loss/distill", loss_value, step)
            writer.add_scalar("loss/best", best, step)
            writer.add_scalar("time/elapsed_s", elapsed, step)
            writer.add_scalar("time/steps_per_s", float(step) / max(elapsed, 1e-6), step)
            writer.flush()
            latest = save_distill_checkpoint(run_dir, run_id, "latest", student, optimizer, step, best, student_cfg, metadata)
            print(f"{run_id}: step={step} loss={loss_value:.8g} best={best:.8g} latest={latest}", flush=True)

        now = time.perf_counter()
        if now - last_periodic >= float(args.checkpoint_every_minutes) * 60.0:
            path = save_distill_checkpoint(run_dir, run_id, f"periodic_step{step}", student, optimizer, step, best, student_cfg, metadata)
            print(f"saved periodic checkpoint {path}", flush=True)
            last_periodic = now

    final_loss = float(
        distill_probe_loss(teacher, student, store, teacher_cfg, start_pools, int(args.rollout_k), batch_size).cpu()
    )
    writer.add_scalar("loss/final_probe", final_loss, int(args.steps))
    writer.add_scalar("run/completed", 1.0, int(args.steps))
    writer.flush()
    last_path = save_distill_checkpoint(run_dir, run_id, "last", student, optimizer, int(args.steps), best, student_cfg, metadata)
    print(f"saved last checkpoint {last_path}", flush=True)


if __name__ == "__main__":
    main()
