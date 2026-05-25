from __future__ import annotations

from collections.abc import Iterator
from typing import Any


CURRENT_IK_CONTROLLER_LOSSES = frozenset(
    {
        "simple_ae_output_reconstruction",
        "simple_ae_output_reconstruction_with_optional_envelope",
        "rl_only",
    }
)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def checkpoint_metadata(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(checkpoint.get("metadata"))


def checkpoint_policy(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(checkpoint_metadata(checkpoint).get("policy"))


def checkpoint_is_autoencoder(checkpoint: dict[str, Any]) -> bool:
    schema = checkpoint.get("schema")
    return isinstance(schema, dict) and "total_dim" in schema


def is_current_ik_controller_checkpoint(checkpoint: Any) -> bool:
    """Return true for the one supported IK controller runtime.

    This intentionally does not depend only on policy.loss. Objective names keep
    changing during experiments, while the runtime contract is stable: these
    checkpoints carry the flat current/previous pose vector, velocity, root
    feature input and must be rolled out with train_simple_ae_controller helpers.
    """

    if not isinstance(checkpoint, dict) or checkpoint_is_autoencoder(checkpoint):
        return False
    if "model" not in checkpoint:
        return False
    metadata = checkpoint_metadata(checkpoint)
    policy = checkpoint_policy(checkpoint)
    if metadata.get("simple_ae_checkpoint"):
        return True
    if policy.get("loss") in CURRENT_IK_CONTROLLER_LOSSES:
        return True
    current_policy_keys = {
        "ae_loss_weight",
        "ae_score_output_only",
        "ae_scores_raw_output",
        "rl_loss_enabled",
        "rl_loss",
        "rl_grad_clip_norm",
        "zero_loss_stop_threshold",
    }
    return any(key in policy for key in current_policy_keys)


def require_current_ik_controller_checkpoint(checkpoint: Any, path: object = "") -> None:
    if is_current_ik_controller_checkpoint(checkpoint):
        return
    suffix = f": {path}" if path else ""
    if checkpoint_uses_ik(checkpoint):
        raise ValueError(
            "Unsupported legacy IK checkpoint. Current IK viewers and tools use exactly one "
            f"flat-vector IK controller runtime{suffix}."
        )
    raise ValueError(f"Checkpoint is not a current IK controller checkpoint{suffix}.")


def _walk_dicts(value: Any) -> Iterator[dict[str, Any]]:
    if not isinstance(value, dict):
        return
    yield value
    for child in value.values():
        if isinstance(child, dict):
            yield from _walk_dicts(child)
        elif isinstance(child, list):
            for item in child:
                if isinstance(item, dict):
                    yield from _walk_dicts(item)


def checkpoint_uses_ik(checkpoint: Any) -> bool:
    if not isinstance(checkpoint, dict):
        return False
    for values in _walk_dicts(checkpoint):
        pose_representation = values.get("pose_representation")
        if isinstance(pose_representation, str) and "ik" in pose_representation.lower():
            return True
    return False


def runtime_name(checkpoint: Any) -> str:
    return "ik_controller" if is_current_ik_controller_checkpoint(checkpoint) else "unsupported"
