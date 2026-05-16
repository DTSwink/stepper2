from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_locomotion as tl
import train_locomotion_ae_prior as ae_prior


def resolve_npz_folder(path: Path) -> Path:
    path = tl.resolve_path(path)
    if path.name == "npz_final":
        return path
    candidate = path / "npz_final"
    if candidate.exists():
        return candidate
    return path


def load_folder(path: Path, cfg: tl.TrainConfig, cyclic: bool) -> list[tl.MotionClip]:
    folder = resolve_npz_folder(path)
    files = sorted(folder.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {folder}")
    return [tl.MotionClip(file, cfg, cyclic_animation=cyclic) for file in files]


def feature_slices(clip: tl.MotionClip, cfg: tl.TrainConfig) -> tuple[slice, slice]:
    pose_dim = 3 + 6 + clip.Jn * 6 + clip.J * 3
    if cfg.use_contact_state:
        pose_dim += 2
    velocity_dim = 3 + clip.J * 3
    root_start = pose_dim * 2 + velocity_dim
    root_slice = slice(root_start, root_start + 3)
    future_slice = slice(root_start + 3, root_start + 3 + cfg.future_window * 4)
    return root_slice, future_slice


def root_feature_for_clip(
    clip: tl.MotionClip,
    cur_frame: int,
    cfg: tl.TrainConfig,
    device: torch.device,
) -> torch.Tensor:
    prev = torch.tensor([cur_frame - 1], dtype=torch.long, device=device)
    cur = torch.tensor([cur_frame], dtype=torch.long, device=device)
    return tl.root_delta_feature(clip, prev, cur, cfg, device)[0].detach().cpu()


def check_tail_forward_features(
    clips: list[tl.MotionClip],
    cfg: tl.TrainConfig,
    device: torch.device,
    required_names: list[str],
) -> list[dict[str, object]]:
    by_name = {clip.path.stem: clip for clip in clips}
    reports: list[dict[str, object]] = []
    missing = [name for name in required_names if name not in by_name]
    if missing:
        raise AssertionError(f"Missing required diagnostic clips: {missing}")
    for name in required_names:
        clip = by_name[name]
        tail = int(clip.T) - 1
        root_feat = root_feature_for_clip(clip, tail, cfg, device)
        prev_pos, _prev_rot, prev_yaw, _prev_heading = tl.root_state(
            clip, torch.tensor([tail - 1], dtype=torch.long, device=device), cfg, device
        )
        cur_pos, _cur_rot, cur_yaw, _cur_heading = tl.root_state(
            clip, torch.tensor([tail], dtype=torch.long, device=device), cfg, device
        )
        report = {
            "clip": name,
            "tail_frame": tail,
            "root_feature_dx_dz_dyaw": [float(x) for x in root_feat.tolist()],
            "world_delta_xz_m": [float(cur_pos[0, 0] - prev_pos[0, 0]), float(cur_pos[0, 2] - prev_pos[0, 2])],
            "prev_yaw_deg": float(torch.rad2deg(prev_yaw[0]).detach().cpu()),
            "cur_yaw_deg": float(torch.rad2deg(cur_yaw[0]).detach().cpu()),
        }
        reports.append(report)
        if root_feat[1] <= 0.05:
            raise AssertionError(
                f"{name} tail should condition as forward local +Z, got root feature {report['root_feature_dx_dz_dyaw']}"
            )
    return reports


def check_packed_matches_unpacked(
    clips: list[tl.MotionClip],
    cfg: tl.TrainConfig,
    device: torch.device,
    max_clips: int = 8,
) -> list[dict[str, object]]:
    selected = clips[:max_clips]
    packed = ae_prior.PackedClips(selected, cfg, device)
    reports: list[dict[str, object]] = []
    for clip_id, clip in enumerate(selected):
        safe_last = max(1, min(int(clip.T) - int(cfg.future_window) - 1, int(clip.cyclic_period) - 1))
        candidates = sorted({1, max(1, safe_last // 2), safe_last})
        for cur_frame in candidates:
            prev = torch.tensor([cur_frame - 1], dtype=torch.long, device=device)
            cur = torch.tensor([cur_frame], dtype=torch.long, device=device)
            clip_ids = torch.tensor([clip_id], dtype=torch.long, device=device)
            prev_pose = tl.get_pose_from_clip(clip, prev, device)
            cur_pose = tl.get_pose_from_clip(clip, cur, device)
            unpacked = tl.build_input(clip, prev, cur, prev_pose, cur_pose, cfg, device)
            packed_in = ae_prior.packed_build_input(packed, clip_ids, prev, cur, prev_pose, cur_pose, cfg)
            max_abs = float((unpacked - packed_in).abs().max().detach().cpu())
            root_slice, future_slice = feature_slices(clip, cfg)
            root_abs = float((unpacked[:, root_slice] - packed_in[:, root_slice]).abs().max().detach().cpu())
            future_abs = float((unpacked[:, future_slice] - packed_in[:, future_slice]).abs().max().detach().cpu())
            reports.append(
                {
                    "clip": clip.path.stem,
                    "frame": int(cur_frame),
                    "max_abs": max_abs,
                    "root_abs": root_abs,
                    "future_abs": future_abs,
                }
            )
            if max_abs > 1e-5:
                raise AssertionError(
                    f"Packed input does not match unpacked for {clip.path.name} frame {cur_frame}: max_abs={max_abs}"
                )
    return reports


def check_canonical_heading_invariance(
    periodic: list[tl.MotionClip],
    nonperiodic: list[tl.MotionClip],
    cfg: tl.TrainConfig,
    device: torch.device,
) -> list[dict[str, object]]:
    loops = {clip.path.stem: clip for clip in periodic}
    transitions = {clip.path.stem: clip for clip in nonperiodic}
    loop = loops.get("M_Neutral_Walk_Loop_F")
    if loop is None:
        return []
    reports: list[dict[str, object]] = []
    pose_frame = min(33, loop.T - 1)
    pose = tl.get_pose_from_clip(loop, torch.tensor([pose_frame], dtype=torch.long, device=device), device)
    loop_root = tl.root_state(loop, torch.tensor([pose_frame], dtype=torch.long, device=device), cfg, device)
    _loop_pos, _loop_rot, loop_canon = tl.fk_from_pose(loop, loop_root[0], loop_root[1], pose, device)
    for name in ("M_Neutral_Walk_Spin_LL_to_F_Lfoot", "M_Neutral_Walk_Spin_LL_to_F_Rfoot"):
        transition = transitions.get(name)
        if transition is None:
            continue
        frame = int(transition.T) - 1
        transition_root = tl.root_state(transition, torch.tensor([frame], dtype=torch.long, device=device), cfg, device)
        _pos, _rot, canon = tl.fk_from_pose(transition, transition_root[0], transition_root[1], pose, device)
        max_abs = float((canon - loop_canon).abs().max().detach().cpu())
        report = {
            "clip": name,
            "pose_source": loop.path.stem,
            "pose_frame": int(pose_frame),
            "root_frame": int(frame),
            "max_abs": max_abs,
        }
        reports.append(report)
        if max_abs > 1e-5:
            raise AssertionError(f"Canonical pose basis is not heading-invariant for {name}: max_abs={max_abs}")
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight checks for locomotion motion-feature conventions.")
    parser.add_argument(
        "--periodic-folder",
        type=Path,
        default=Path("ue5/animations_omni_only_full"),
        help="Periodic FBX folder or npz_final folder.",
    )
    parser.add_argument(
        "--nonperiodic-folder",
        type=Path,
        default=Path("ue5/animations_transitions_only_full_trimmed"),
        help="Non-periodic FBX folder or npz_final folder.",
    )
    parser.add_argument("--future-window-seconds", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("training/runs/diagnostics/motion_feature_preflight.json"))
    args = parser.parse_args()

    cfg = tl.TrainConfig()
    cfg.future_window_seconds = float(args.future_window_seconds)
    device = torch.device(args.device)

    periodic = load_folder(args.periodic_folder, cfg, cyclic=True)
    nonperiodic = load_folder(args.nonperiodic_folder, cfg, cyclic=False)
    transition_required = [
        "M_Neutral_Walk_Spin_LL_to_F_Rfoot",
        "M_Neutral_Walk_Spin_LL_to_F_Lfoot",
        "M_Neutral_Walk_Turn_R_180_Lfoot",
        "M_Neutral_Walk_Turn_R_180_Rfoot",
        "M_Neutral_Walk_Turn_R_090_Lfoot",
        "M_Neutral_Walk_Turn_R_090_Rfoot",
    ]
    transition_required = [name for name in transition_required if any(c.path.stem == name for c in nonperiodic)]
    if len(transition_required) < 2:
        raise AssertionError("Could not find enough spin/turn diagnostic clips in the non-periodic folder.")

    report = {
        "periodic_folder": str(resolve_npz_folder(args.periodic_folder)),
        "nonperiodic_folder": str(resolve_npz_folder(args.nonperiodic_folder)),
        "future_window": int(cfg.future_window),
        "tail_forward_checks": check_tail_forward_features(nonperiodic, cfg, device, transition_required),
        "canonical_heading_invariance": check_canonical_heading_invariance(periodic, nonperiodic, cfg, device),
        "packed_vs_unpacked_periodic": check_packed_matches_unpacked(periodic, cfg, device),
        "packed_vs_unpacked_nonperiodic": check_packed_matches_unpacked(nonperiodic, cfg, device),
    }
    out_path = tl.resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["tail_forward_checks"], indent=2))
    print(f"OK: motion feature preflight passed -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
