from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from .bootstrap import PROJECT_ROOT, ensure_paths
    from . import ik_core as tl
    from . import train_full_ae_envelope as full
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    import ik_core as tl
    import train_full_ae_envelope as full

ensure_paths()


DEFAULT_OUTPUT = PROJECT_ROOT / "training" / "runs" / "cache" / "ik_dataset_speed_limits.json"


def _transition_indices(clip: tl.MotionClip) -> tuple[torch.Tensor, torch.Tensor]:
    if clip.cyclic_animation:
        count = max(1, int(clip.cyclic_period))
        cur = torch.arange(count, dtype=torch.long)
        nxt = torch.remainder(cur + 1, count)
        return cur, nxt
    count = max(0, int(clip.T) - 1)
    cur = torch.arange(count, dtype=torch.long)
    return cur, cur + 1


def _argmax_record(values: torch.Tensor, clip: tl.MotionClip, label: str) -> dict[str, object]:
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return {"value": 0.0, "clip": str(clip.path), "frame": 0, "slot": 0, "label": label}
    flat_i = int(flat.argmax().item())
    value = float(flat[flat_i].item())
    if values.ndim == 1:
        frame_i = flat_i
        slot_i = 0
    else:
        slot_count = int(values.shape[1])
        frame_i = flat_i // slot_count
        slot_i = flat_i % slot_count
    return {
        "value": value,
        "clip": str(clip.path),
        "frame": int(frame_i),
        "slot": int(slot_i),
        "label": label,
    }


def _update_best(best: dict[str, dict[str, object]], key: str, record: dict[str, object]) -> None:
    if key not in best or float(record["value"]) > float(best[key]["value"]):
        best[key] = record


def _payload_part(payload: torch.Tensor, part: str) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for spec in tl.IK_PAYLOAD_SLICES:
        sl = spec[part]
        assert isinstance(sl, slice)
        chunks.append(payload[:, sl])
    return torch.stack(chunks, dim=1)


def _slot_names() -> list[str]:
    return [str(spec["end"]) for spec in tl.IK_LIMB_SPECS]


def _update_slot_best(
    slot_max: dict[str, torch.Tensor],
    slot_best: dict[str, list[dict[str, object] | None]],
    key: str,
    values: torch.Tensor,
    clip: tl.MotionClip,
    names: list[str],
) -> None:
    max_values, max_indices = values.max(dim=0)
    for slot_i, value_tensor in enumerate(max_values):
        value = float(value_tensor.item())
        if value > float(slot_max[key][slot_i].item()):
            slot_max[key][slot_i] = value_tensor.detach().cpu()
            slot_best[key][slot_i] = {
                "value": value,
                "clip": str(clip.path),
                "frame": int(max_indices[slot_i].item()),
                "slot": int(slot_i),
                "slot_name": names[slot_i],
            }


def compute_dataset_limits(margin: float = 1.10) -> dict[str, object]:
    cfg = tl.TrainConfig()
    cfg.pose_representation = tl.IK_POSE_REPRESENTATION
    cfg.live_viewer = False
    cfg.visual_reporter = False

    best: dict[str, dict[str, object]] = {}
    names = _slot_names()
    slot_max = {
        "end_effector_root_location_m": torch.zeros(len(names), dtype=torch.float32),
        "end_effector_root_rotation_deg": torch.zeros(len(names), dtype=torch.float32),
    }
    slot_best: dict[str, list[dict[str, object] | None]] = {
        key: [None for _name in names] for key in slot_max
    }
    specs = full.full_specs()
    for path, cyclic in specs:
        clip = tl.MotionClip(path, cfg, cyclic_animation=cyclic)
        cur, nxt = _transition_indices(clip)
        if cur.numel() == 0:
            continue
        fps = float(clip.fps)

        pelvis_horizontal = torch.linalg.vector_norm(
            torch.stack((clip.pelvis_local_pos[:, 0], clip.pelvis_local_pos[:, 2]), dim=-1),
            dim=-1,
        )
        _update_best(best, "pelvis_root_horizontal_m", _argmax_record(pelvis_horizontal, clip, "pelvis root-local horizontal distance"))

        pelvis_static_rot = tl.rotation_6d_to_matrix(clip.pelvis_rot6)
        identity = torch.eye(3, dtype=pelvis_static_rot.dtype, device=pelvis_static_rot.device).expand_as(pelvis_static_rot)
        pelvis_root_angle = tl.geodesic_angles(pelvis_static_rot, identity) * 180.0 / torch.pi
        _update_best(best, "pelvis_root_rotation_deg", _argmax_record(pelvis_root_angle, clip, "pelvis root-local rotation angle"))

        ee_static_pos = _payload_part(clip.ik_payload, "pos")
        ee_static_loc = torch.linalg.vector_norm(ee_static_pos, dim=-1)
        _update_best(best, "end_effector_root_location_m", _argmax_record(ee_static_loc, clip, "IK hand/foot root-local location distance"))
        _update_slot_best(slot_max, slot_best, "end_effector_root_location_m", ee_static_loc, clip, names)

        ee_static_rot = tl.rotation_6d_to_matrix(_payload_part(clip.ik_payload, "rot6").reshape(-1, 6))
        identity = torch.eye(3, dtype=ee_static_rot.dtype, device=ee_static_rot.device).expand_as(ee_static_rot)
        ee_static_angle = tl.geodesic_angles(ee_static_rot, identity).reshape(clip.ik_payload.shape[0], len(names)) * 180.0 / torch.pi
        _update_best(best, "end_effector_root_rotation_deg", _argmax_record(ee_static_angle, clip, "IK hand/foot root-local rotation angle"))
        _update_slot_best(slot_max, slot_best, "end_effector_root_rotation_deg", ee_static_angle, clip, names)

        ee_pos_cur = _payload_part(clip.ik_payload.index_select(0, cur), "pos")
        ee_pos_nxt = _payload_part(clip.ik_payload.index_select(0, nxt), "pos")
        ee_lin = torch.linalg.vector_norm(ee_pos_nxt - ee_pos_cur, dim=-1) * fps
        _update_best(best, "end_effector_velocity_mps", _argmax_record(ee_lin, clip, "IK hand/foot root-local position speed"))

        ee_rot_cur = tl.rotation_6d_to_matrix(_payload_part(clip.ik_payload.index_select(0, cur), "rot6").reshape(-1, 6))
        ee_rot_nxt = tl.rotation_6d_to_matrix(_payload_part(clip.ik_payload.index_select(0, nxt), "rot6").reshape(-1, 6))
        ee_ang = tl.geodesic_angles(ee_rot_nxt, ee_rot_cur).reshape(cur.numel(), -1) * fps * 180.0 / torch.pi
        _update_best(best, "end_effector_angular_velocity_deg_s", _argmax_record(ee_ang, clip, "IK hand/foot root-local rotation speed"))

        pelvis_lin = torch.linalg.vector_norm(
            clip.pelvis_local_pos.index_select(0, nxt) - clip.pelvis_local_pos.index_select(0, cur), dim=-1
        ) * fps
        _update_best(best, "pelvis_velocity_mps", _argmax_record(pelvis_lin, clip, "pelvis local position speed"))

        pelvis_rot_cur = tl.rotation_6d_to_matrix(clip.pelvis_rot6.index_select(0, cur))
        pelvis_rot_nxt = tl.rotation_6d_to_matrix(clip.pelvis_rot6.index_select(0, nxt))
        pelvis_ang = tl.geodesic_angles(pelvis_rot_nxt, pelvis_rot_cur) * fps * 180.0 / torch.pi
        _update_best(best, "pelvis_angular_velocity_deg_s", _argmax_record(pelvis_ang, clip, "pelvis local rotation speed"))

        core_cur = clip.core_non_pelvis_rot6.index_select(0, cur)
        core_nxt = clip.core_non_pelvis_rot6.index_select(0, nxt)
        core_rot_cur = tl.rotation_6d_to_matrix(core_cur.reshape(-1, 6))
        core_rot_nxt = tl.rotation_6d_to_matrix(core_nxt.reshape(-1, 6))
        core_ang = tl.geodesic_angles(core_rot_nxt, core_rot_cur).reshape(cur.numel(), -1) * fps * 180.0 / torch.pi
        _update_best(best, "core_angular_velocity_deg_s", _argmax_record(core_ang, clip, "core local rotation speed"))

    raw_max = {key: float(record["value"]) for key, record in best.items()}
    limits = {key: float(value) * float(margin) for key, value in raw_max.items()}
    per_slot_raw_max = {
        key: {names[i]: float(values[i].item()) for i in range(len(names))}
        for key, values in slot_max.items()
    }
    per_slot_limits = {
        key: {name: float(value) * float(margin) for name, value in values.items()}
        for key, values in per_slot_raw_max.items()
    }
    return {
        "margin": float(margin),
        "dataset": {
            "periodic_folder": str(full.PERIODIC_FOLDER),
            "nonperiodic_folder": str(full.NONPERIODIC_FOLDER),
            "clip_count": len(specs),
        },
        "raw_max": raw_max,
        "limits": limits,
        "per_slot_raw_max": per_slot_raw_max,
        "per_slot_limits": per_slot_limits,
        "argmax": best,
        "per_slot_argmax": slot_best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute IK RL speed limits from the full dataset.")
    parser.add_argument("--margin", type=float, default=1.10)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    data = compute_dataset_limits(args.margin)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
