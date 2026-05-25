from __future__ import annotations

import math
from pathlib import Path

import torch

try:
    from .bootstrap import PROJECT_ROOT, ensure_paths
    from . import ik_core as tl
    from . import checkpoint_runtime as ik_runtime
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    import ik_core as tl
    import checkpoint_runtime as ik_runtime

ensure_paths()


DEFAULT_NPZ = PROJECT_ROOT / "ue5" / "animations_omni_only_full" / "npz_final" / "M_Neutral_Walk_Loop_F.npz"


def _pose_clone(pose: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in pose.items()}


def _clip(path: Path = DEFAULT_NPZ) -> tl.MotionClip:
    cfg = tl.TrainConfig()
    cfg.pose_representation = tl.IK_POSE_REPRESENTATION
    cfg.use_torch_compile = False
    return tl.MotionClip(path, cfg, cyclic_animation=True)


def _decode(clip: tl.MotionClip, pose: dict[str, torch.Tensor], idx: torch.Tensor) -> torch.Tensor:
    pos, _rot, _canon = tl.fk_from_pose(clip, clip.root_pos.index_select(0, idx), clip.root_rot.index_select(0, idx), pose, torch.device("cpu"))
    return pos


def _rotate_ee_rows_around_chain_axis(
    clip: tl.MotionClip,
    pose: dict[str, torch.Tensor],
    limb_i: int,
    angle_rad: float,
) -> None:
    payload = pose["ik_payload"]
    spec = tl.IK_PAYLOAD_SLICES[limb_i]
    pos_slice = spec["pos"]
    rot_slice = spec["rot6"]
    assert isinstance(pos_slice, slice)
    assert isinstance(rot_slice, slice)
    limb = clip.ik_limb_specs[limb_i]
    base = clip.root_relative_pos[1:2, int(limb["start"])]
    end = payload[:, pos_slice]
    axis = tl.normalize(end - base)
    rot = tl.rotation_6d_to_matrix(payload[:, rot_slice])
    axis_rows = axis[:, None, :].expand_as(rot)
    angle_rows = torch.full(rot.shape[:-1], angle_rad, dtype=rot.dtype)
    payload[:, rot_slice] = tl.rotmat_to_6d(tl.rotate_around_axis(rot, axis_rows, angle_rows))


def test_schema_and_clamps() -> None:
    clip = _clip()
    expected = torch.tensor(
        [tl.IK_ARM_POLE_ALPHA, tl.IK_ARM_POLE_ALPHA, tl.IK_LEG_POLE_ALPHA, tl.IK_LEG_POLE_ALPHA],
        dtype=torch.float32,
    )
    assert tl.IK_SCHEMA_VERSION == 2
    assert tl.IK_POLE_REFERENCE == "ee_frame"
    assert torch.allclose(clip.ik_pole_alpha, expected, atol=1e-7)
    assert clip.ik_ee_pole_ref.shape == (len(clip.ik_limb_specs), 3)
    assert float(clip.ik_pole_float.abs().max()) <= 1.0 + 1e-6


def test_payload_cleaning_clamps_poles_and_toes() -> None:
    clip = _clip()
    payload = clip.ik_payload[:2].clone()
    for spec in tl.IK_PAYLOAD_SLICES:
        pole_slice = spec["pole"]
        toe_slice = spec["toe_float"]
        assert isinstance(pole_slice, slice)
        payload[:, pole_slice] = 99.0
        if toe_slice is not None:
            assert isinstance(toe_slice, slice)
            payload[:, toe_slice] = -99.0
    cleaned = tl.clean_ik_payload(payload)
    for spec in tl.IK_PAYLOAD_SLICES:
        pole_slice = spec["pole"]
        toe_slice = spec["toe_float"]
        assert isinstance(pole_slice, slice)
        assert float(cleaned[:, pole_slice].max()) <= 1.0
        if toe_slice is not None:
            assert isinstance(toe_slice, slice)
            assert float(cleaned[:, toe_slice].min()) >= -1.0


def test_decode_clamps_same_as_unit_pole_float() -> None:
    clip = _clip()
    idx = torch.tensor([1], dtype=torch.long)
    base_pose = clip.pose_at(idx)
    for limb_i, spec in enumerate(tl.IK_PAYLOAD_SLICES):
        pole_slice = spec["pole"]
        assert isinstance(pole_slice, slice)
        pose_one = _pose_clone(base_pose)
        pose_huge = _pose_clone(base_pose)
        pose_one["ik_payload"][:, pole_slice] = 1.0
        pose_huge["ik_payload"][:, pole_slice] = 8.0
        pos_one = _decode(clip, pose_one, idx)
        pos_huge = _decode(clip, pose_huge, idx)
        mid = int(clip.ik_limb_specs[limb_i]["mid"])
        assert float((pos_one[:, mid] - pos_huge[:, mid]).abs().max()) < 1e-6


def test_ee_rotation_changes_knee_or_elbow_reference() -> None:
    clip = _clip()
    idx = torch.tensor([1], dtype=torch.long)
    moving_limbs = 0
    for limb_i, spec in enumerate(tl.IK_PAYLOAD_SLICES):
        pole_slice = spec["pole"]
        assert isinstance(pole_slice, slice)
        pose_a = _pose_clone(clip.pose_at(idx))
        pose_b = _pose_clone(clip.pose_at(idx))
        pose_a["ik_payload"][:, pole_slice] = 0.0
        pose_b["ik_payload"][:, pole_slice] = 0.0
        _rotate_ee_rows_around_chain_axis(clip, pose_b, limb_i, math.radians(20.0))
        pos_a = _decode(clip, pose_a, idx)
        pos_b = _decode(clip, pose_b, idx)
        mid = int(clip.ik_limb_specs[limb_i]["mid"])
        delta = float(torch.linalg.norm(pos_a[:, mid] - pos_b[:, mid], dim=-1).max())
        if delta > 1e-4:
            moving_limbs += 1
    assert moving_limbs >= 2


def test_residual_prediction_is_hard_coded_global() -> None:
    previous_mode = tl.OUTPUT_PREDICTION_MODE
    try:
        tl.OUTPUT_PREDICTION_MODE = "absolute"
        cfg = tl.TrainConfig()
        assert cfg.predict_residual is False
        cfg.predict_residual = True
        assert cfg.predict_residual is False

        tl.OUTPUT_PREDICTION_MODE = "residual"
        cfg = tl.TrainConfig()
        assert cfg.predict_residual is True
        cfg.predict_residual = False
        assert cfg.predict_residual is True
    finally:
        tl.OUTPUT_PREDICTION_MODE = previous_mode

    checkpoint = {
        "model": {},
        "config": {"predict_residual": True, "output_prediction_mode": "residual"},
        "metadata": {
            "policy": {
                "loss": "supervised_rollout",
                "output_reference_root": "current",
                "output_prediction_mode": "residual",
                "ik_schema_version": 2,
                "ik_pole_reference": "ee_frame",
            }
        },
    }
    assert ik_runtime.is_current_ik_controller_checkpoint(checkpoint)


def main() -> None:
    test_schema_and_clamps()
    test_payload_cleaning_clamps_poles_and_toes()
    test_decode_clamps_same_as_unit_pole_float()
    test_ee_rotation_changes_knee_or_elbow_reference()
    test_residual_prediction_is_hard_coded_global()
    print("ee theta contract tests passed")


if __name__ == "__main__":
    main()
