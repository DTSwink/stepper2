from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

try:
    from .bootstrap import PROJECT_ROOT, ensure_paths
    from . import ik_core as tl
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    import ik_core as tl

ensure_paths()


DEFAULT_NPZ = PROJECT_ROOT / "ue5" / "animations_omni_only_full" / "npz_final" / "M_Neutral_Walk_Loop_F.npz"


def resolve_path(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def max_pos(pos: torch.Tensor, gt: torch.Tensor, bone: int) -> float:
    return float((pos[:, bone] - gt[:, bone]).norm(dim=-1).max().detach().cpu())


def max_rot(rot: torch.Tensor, gt: torch.Tensor, bone: int) -> float:
    err = exact_geodesic_angles(rot[:, bone], gt[:, bone]).max()
    return float((err * 180.0 / math.pi).detach().cpu())


def frame0_rot(rot: torch.Tensor, gt: torch.Tensor, bone: int) -> float:
    err = exact_geodesic_angles(rot[:1, bone], gt[:1, bone])[0]
    return float((err * 180.0 / math.pi).detach().cpu())


def exact_geodesic_angles(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    equal = (pred - target).abs().amax(dim=(-1, -2)) <= 1e-7
    pred_clean = tl.rotation_6d_to_matrix(tl.rotmat_to_6d(pred))
    target_clean = tl.rotation_6d_to_matrix(tl.rotmat_to_6d(target))
    delta = pred_clean @ target_clean.transpose(-1, -2)
    trace = delta.diagonal(dim1=-1, dim2=-2).sum(dim=-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.where(equal, torch.zeros_like(cos), torch.acos(cos))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check IK payload encode/decode against NPZ ground truth.")
    parser.add_argument("--npz", default=str(DEFAULT_NPZ))
    args = parser.parse_args()

    cfg = tl.TrainConfig()
    cfg.pose_representation = tl.IK_POSE_REPRESENTATION
    cfg.cyclic_animation = False
    clip = tl.MotionClip(resolve_path(args.npz), cfg, cyclic_animation=False)
    idx = torch.arange(clip.T, dtype=torch.long)
    pose = clip.pose_at(idx)
    pos, rot, _canon = tl.fk_from_pose(clip, clip.root_pos, clip.root_rot, pose, torch.device("cpu"))

    endpoint_names = ("hand_l", "hand_r", "foot_l", "foot_r")
    toe_names = ("ball_l", "ball_r")
    mid_names = ("lowerarm_l", "lowerarm_r", "calf_l", "calf_r")

    print(f"clip={clip.path}")
    print(f"frames={clip.T} bodies={clip.J} core={clip.Jcore} payload_dim={clip.ik_payload.shape[-1]}")
    print(f"pole_reference={tl.IK_POLE_REFERENCE} pole_alpha_deg={[(float(x) * 180.0 / math.pi) for x in clip.ik_pole_alpha]}")
    print(f"pole_axes={clip.ik_local_pole_axis.tolist()}")
    print(f"toe_axes={clip.ik_toe_axis.tolist()}")

    failures: list[str] = []
    for name in endpoint_names + toe_names + mid_names:
        if name not in clip.body_names:
            continue
        bone = clip.body_names.index(name)
        p = max_pos(pos, clip.global_pos, bone)
        r = max_rot(rot, clip.global_rot, bone)
        r0 = frame0_rot(rot, clip.global_rot, bone)
        print(f"{name:12s} pos_max_m={p:.8f} rot_f0_deg={r0:.4f} rot_max_deg={r:.4f}")
        if name in endpoint_names and (p > 1e-5 or r > 0.25):
            failures.append(f"{name} endpoint round trip exceeded strict tolerance")
        if name in toe_names and (p > 2e-5 or r > 6.0):
            failures.append(f"{name} toe round trip exceeded hinge tolerance")
        if name in mid_names and p > 0.02:
            failures.append(f"{name} mid-joint position exceeded tolerance")

    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
