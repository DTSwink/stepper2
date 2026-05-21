from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from make_model_reconstructable_npz import (
    load_npz_payload,
    make_matrix,
    rotation_6d,
    write_npz_atomic,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IK_DIR = PROJECT_ROOT / "training" / "ik"
if str(IK_DIR) not in sys.path:
    sys.path.insert(0, str(IK_DIR))

import ik_core as tl  # noqa: E402


STALE_OR_OBSOLETE_KEYS = {
    "local_quat_xyzw",
    "fbx_lcl_rotation_euler_xyz",
    "default_lcl_rotation_euler_xyz",
    "contact_names",
    "contacts",
    "contact_height_m",
    "contact_speed_mps",
    "contact_closest_point_m",
    "contact_closest_part",
    "contact_lowest_point_m",
    "contact_lowest_part",
    "contact_slide_distance_m",
    "contact_slide_point_prev_m",
    "contact_slide_point_cur_m",
    "contact_slide_part",
    "contact_height_threshold_m",
    "contact_speed_threshold_mps",
}


def decanonicalize_positions(pos: np.ndarray, up_axis: int) -> np.ndarray:
    if int(up_axis) == 3:
        return np.stack((pos[..., 0], -pos[..., 2], pos[..., 1]), axis=-1)
    return pos


def decanonicalize_rotations(rot: np.ndarray, up_axis: int) -> np.ndarray:
    if int(up_axis) != 3:
        return rot
    p = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    return p @ rot @ p.T


def exact_geodesic_angles(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    equal = (pred - target).abs().amax(dim=(-1, -2)) <= 1e-7
    pred = tl.rotation_6d_to_matrix(tl.rotmat_to_6d(pred))
    target = tl.rotation_6d_to_matrix(tl.rotmat_to_6d(target))
    delta = pred @ target.transpose(-1, -2)
    trace = delta.diagonal(dim1=-1, dim2=-2).sum(dim=-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.where(equal, torch.zeros_like(cos), torch.acos(cos))


def ik_metrics(npz_path: Path) -> dict[str, float]:
    cfg = tl.TrainConfig()
    cfg.pose_representation = tl.IK_POSE_REPRESENTATION
    cfg.cyclic_animation = False
    clip = tl.MotionClip(npz_path, cfg, cyclic_animation=False)
    idx = torch.arange(clip.T, dtype=torch.long)
    pose = clip.pose_at(idx)
    pos, rot, _canon = tl.fk_from_pose(clip, clip.root_pos, clip.root_rot, pose, torch.device("cpu"))
    pos_delta = torch.linalg.norm(pos - clip.global_pos, dim=-1)
    rot_delta = exact_geodesic_angles(rot.reshape(-1, 3, 3), clip.global_rot.reshape(-1, 3, 3))
    return {
        "frame0_max_position_delta_m": float(pos_delta[:1].max().detach().cpu()),
        "all_max_position_delta_m": float(pos_delta.max().detach().cpu()),
        "frame0_max_rotation_delta_deg": float((rot_delta[: clip.J].max() * 180.0 / np.pi).detach().cpu()),
        "all_max_rotation_delta_deg": float((rot_delta.max() * 180.0 / np.pi).detach().cpu()),
        "frame0_max_rotation_matrix_abs_delta": float((rot[:1] - clip.global_rot[:1]).abs().max().detach().cpu()),
        "all_max_rotation_matrix_abs_delta": float((rot - clip.global_rot).abs().max().detach().cpu()),
    }


def bake_npz(input_npz: Path, output_npz: Path, report_json: Path | None = None) -> dict:
    payload = load_npz_payload(input_npz)
    names = [str(x) for x in payload["bone_names"]]
    up_axis = int(payload["axis_up_axis"]) if "axis_up_axis" in payload else 2

    cfg = tl.TrainConfig()
    cfg.pose_representation = tl.IK_POSE_REPRESENTATION
    cfg.cyclic_animation = False
    clip = tl.MotionClip(input_npz, cfg, cyclic_animation=False)
    idx = torch.arange(clip.T, dtype=torch.long)
    pose = clip.pose_at(idx)
    decoded_pos, decoded_rot, _canon = tl.fk_from_pose(
        clip,
        clip.root_pos,
        clip.root_rot,
        pose,
        torch.device("cpu"),
    )

    source_global_pos = (
        tl.canonicalize_positions(
            torch.tensor(np.asarray(payload["global_joint_pos"], dtype=np.float32), dtype=torch.float32)
            * float(cfg.position_unit_scale),
            up_axis,
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    source_global_rot = (
        tl.canonicalize_rotations(
            torch.tensor(np.asarray(payload["global_matrix"], dtype=np.float32)[..., :3, :3], dtype=torch.float32),
            up_axis,
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    source_local_pos = (
        tl.canonicalize_positions(
            torch.tensor(np.asarray(payload["fbx_lcl_translation"], dtype=np.float32), dtype=torch.float32)
            * float(cfg.position_unit_scale),
            up_axis,
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    source_local_rot = (
        tl.canonicalize_rotations(
            torch.tensor(np.asarray(payload["local_matrix"], dtype=np.float32)[..., :3, :3], dtype=torch.float32),
            up_axis,
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    model_global_pos = source_global_pos.copy()
    model_global_rot = source_global_rot.copy()
    model_local_pos = source_local_pos.copy()
    model_local_rot6 = tl.rotmat_to_6d(torch.tensor(source_local_rot, dtype=torch.float32)).detach().cpu().numpy().astype(np.float32)

    root_index = names.index("root")
    model_global_pos[:, root_index] = clip.root_pos.detach().cpu().numpy().astype(np.float32)
    model_global_rot[:, root_index] = clip.root_rot.detach().cpu().numpy().astype(np.float32)
    body_local_pos = np.broadcast_to(
        clip.local_offsets.detach().cpu().numpy()[None],
        (clip.T, clip.J, 3),
    ).copy()
    body_local_pos[:, clip.pelvis] = pose["pelvis_pos"].detach().cpu().numpy().astype(np.float32)
    body_local_rot6 = tl.rotmat_to_6d(clip.local_rot).detach().cpu().numpy().astype(np.float32)
    body_local_rot6[:, clip.pelvis] = pose["pelvis_rot6"].detach().cpu().numpy().astype(np.float32)
    if clip.core_non_pelvis:
        body_local_rot6[:, clip.core_non_pelvis] = pose["core_nonpelvis_rot6"].detach().cpu().numpy().astype(np.float32)
    for body_i, full_i in enumerate(clip.keep_full):
        model_global_pos[:, full_i] = decoded_pos[:, body_i].detach().cpu().numpy().astype(np.float32)
        model_global_rot[:, full_i] = decoded_rot[:, body_i].detach().cpu().numpy().astype(np.float32)
        model_local_pos[:, full_i] = body_local_pos[:, body_i]
        model_local_rot6[:, full_i] = body_local_rot6[:, body_i]
    model_local_rot = (
        tl.rotation_6d_to_matrix(torch.tensor(model_local_rot6, dtype=torch.float32).reshape(-1, 6))
        .reshape(clip.T, len(names), 3, 3)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    global_pos = decanonicalize_positions(model_global_pos / float(cfg.position_unit_scale), up_axis).astype(np.float32)
    global_rot = decanonicalize_rotations(model_global_rot, up_axis).astype(np.float32)
    local_pos = decanonicalize_positions(model_local_pos / float(cfg.position_unit_scale), up_axis).astype(np.float32)
    local_rot = decanonicalize_rotations(model_local_rot, up_axis).astype(np.float32)

    payload["local_matrix"] = make_matrix(local_rot, local_pos)
    payload["global_matrix"] = make_matrix(global_rot, global_pos)
    payload["local_rotation_6d"] = rotation_6d(local_rot)
    payload["local_translation"] = local_pos.astype(np.float32)
    payload["global_joint_pos"] = global_pos.astype(np.float32)
    payload["fbx_lcl_translation"] = local_pos.astype(np.float32)
    payload["default_lcl_translation"] = local_pos[0].astype(np.float32)
    payload["default_local_matrix"] = payload["local_matrix"][0].astype(np.float32)
    payload["default_global_matrix"] = payload["global_matrix"][0].astype(np.float32)
    payload["model_global_joint_pos_m"] = model_global_pos
    payload["model_global_matrix"] = make_matrix(model_global_rot, model_global_pos)
    payload["model_local_matrix"] = make_matrix(model_local_rot, model_local_pos)
    payload["model_local_rotation_6d"] = model_local_rot6
    payload["model_lcl_translation_m"] = model_local_pos.astype(np.float32)
    payload["model_default_lcl_translation_m"] = model_local_pos[0].astype(np.float32)
    payload["model_ik_payload"] = pose["ik_payload"].detach().cpu().numpy().astype(np.float32)
    payload["model_ik_rest_axis"] = clip.ik_rest_axis.detach().cpu().numpy().astype(np.float32)
    payload["model_ik_rest_pole"] = clip.ik_rest_pole.detach().cpu().numpy().astype(np.float32)
    payload["model_ik_limb_lengths"] = clip.ik_limb_lengths.detach().cpu().numpy().astype(np.float32)
    payload["model_ik_local_pole_axis"] = clip.ik_local_pole_axis.detach().cpu().numpy().astype(np.float32)
    payload["model_ik_toe_offsets"] = clip.ik_toe_offsets.detach().cpu().numpy().astype(np.float32)
    payload["model_ik_toe_axis"] = clip.ik_toe_axis.detach().cpu().numpy().astype(np.float32)
    payload["ik_reconstructable"] = np.array(True, dtype=np.bool_)
    payload["ik_reconstruction_version"] = np.array(1, dtype=np.int32)
    payload["ik_reconstruction_contract"] = np.array(
        "Payload42 fixed point: model-space arrays and model_ik_payload decode to final NPZ transforms"
    )
    for key in STALE_OR_OBSOLETE_KEYS:
        payload.pop(key, None)

    write_npz_atomic(output_npz, payload)
    metrics = ik_metrics(output_npz)
    summary = {
        "input_npz": str(input_npz),
        "output_npz": str(output_npz),
        "frames": int(global_pos.shape[0]),
        "bones": int(global_pos.shape[1]),
        **metrics,
    }
    if report_json is not None:
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def bake_folder(input_folder: Path, output_folder: Path, report_folder: Path | None = None) -> list[dict]:
    paths = sorted(input_folder.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {input_folder}")
    summaries = []
    for path in paths:
        report = None if report_folder is None else report_folder / f"{path.stem}_ik_reconstructable.json"
        summaries.append(bake_npz(path, output_folder / path.name, report))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Bake NPZ files to a fixed point of the Payload42 IK decoder.")
    parser.add_argument("input", type=Path, help="Input .npz file or folder.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .npz file or folder. Defaults to in-place.")
    parser.add_argument("--report", type=Path, default=None, help="Optional report .json file or folder.")
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve() if args.output is not None else input_path
    report_path = args.report.resolve() if args.report is not None else None
    if input_path.is_dir():
        summaries = bake_folder(input_path, output_path, report_path)
    else:
        output_npz = output_path
        if output_npz.suffix.lower() != ".npz":
            output_npz = output_npz / input_path.name
        report_json = report_path
        if report_json is not None and report_json.suffix.lower() != ".json":
            report_json = report_json / f"{input_path.stem}_ik_reconstructable.json"
        summaries = [bake_npz(input_path, output_npz, report_json)]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
