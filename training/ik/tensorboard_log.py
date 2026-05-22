from __future__ import annotations

import time
from typing import Mapping


CONTROLLER_LOG_EVERY_STEPS = 20


def should_log_controller_step(step: int) -> bool:
    return int(step) == 1 or int(step) % CONTROLLER_LOG_EVERY_STEPS == 0


def log_controller_start(
    writer,
    *,
    rollout_k: int,
    effective_rollout_k_mean: float | None = None,
    effective_rollout_k_max: float | None = None,
    linear_weight: float | None = None,
    angular_weight: float | None = None,
) -> None:
    writer.add_scalar("run/started", 1.0, 0)
    writer.add_scalar("curriculum/rollout_k", int(rollout_k), 0)
    if effective_rollout_k_mean is not None:
        writer.add_scalar("curriculum/effective_rollout_k_mean", float(effective_rollout_k_mean), 0)
    if effective_rollout_k_max is not None:
        writer.add_scalar("curriculum/effective_rollout_k_max", float(effective_rollout_k_max), 0)
    if linear_weight is not None:
        writer.add_scalar("weights/linear_excess", float(linear_weight), 0)
    if angular_weight is not None:
        writer.add_scalar("weights/angular_excess", float(angular_weight), 0)


def log_controller_loss(
    writer,
    *,
    step: int,
    total_loss: float,
    parts: Mapping[str, float] | None = None,
    linear_weight: float = 0.0,
    angular_weight: float = 0.0,
    has_envelope_loss: bool = False,
    loss_scale: float = 1.0,
) -> None:
    if not parts:
        return

    scale = float(loss_scale)
    writer.add_scalar("loss/ae_score", float(parts.get("ae", total_loss)) * scale, int(step))
    if not has_envelope_loss:
        return

    linear = float(parts.get("linear", 0.0))
    angular = float(parts.get("angular", 0.0))
    weighted_linear = linear * float(linear_weight) * scale
    weighted_angular = angular * float(angular_weight) * scale
    writer.add_scalar("loss/weighted_slide_excess", weighted_linear, int(step))
    writer.add_scalar("loss/weighted_yaw_excess", weighted_angular, int(step))


def log_controller_curriculum(
    writer,
    *,
    step: int,
    rollout_k: int,
    effective_rollout_k_mean: float | None = None,
    effective_rollout_k_max: float | None = None,
    stalls: int | None = None,
    start_time: float | None = None,
) -> None:
    writer.add_scalar("curriculum/rollout_k", int(rollout_k), int(step))
    if effective_rollout_k_mean is not None:
        writer.add_scalar("curriculum/effective_rollout_k_mean", float(effective_rollout_k_mean), int(step))
    if effective_rollout_k_max is not None:
        writer.add_scalar("curriculum/effective_rollout_k_max", float(effective_rollout_k_max), int(step))
    if stalls is not None:
        writer.add_scalar("curriculum/stalls", int(stalls), int(step))
    if start_time is not None:
        writer.add_scalar("time/elapsed_s", time.perf_counter() - float(start_time), int(step))


def log_controller_eval_summary(writer, *, step: int, before: Mapping[str, object], after: Mapping[str, object]) -> None:
    writer.add_scalar("eval/before_foot_pin_score", float(before["foot_pin_score"]), int(step))
    writer.add_scalar("eval/after_foot_pin_score", float(after["foot_pin_score"]), int(step))
    writer.add_scalar("eval/before_mean_slide_ratio", float(before["mean_slide_ratio"]), int(step))
    writer.add_scalar("eval/after_mean_slide_ratio", float(after["mean_slide_ratio"]), int(step))
    writer.add_scalar("eval/before_mean_rot_ratio", float(before["mean_rot_ratio"]), int(step))
    writer.add_scalar("eval/after_mean_rot_ratio", float(after["mean_rot_ratio"]), int(step))
    writer.add_scalar("run/completed", 1.0, int(step))
