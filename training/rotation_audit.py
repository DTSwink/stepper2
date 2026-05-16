from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))

import contact_physics as cp
import train_locomotion as tl
import train_locomotion_ae_prior as ae_prior
import transition_autoencoder as tae


def max_abs(x: torch.Tensor) -> float:
    return float(x.detach().abs().max().cpu()) if x.numel() else 0.0


def metric(rows: list[dict[str, object]], name: str, value: float, threshold: float) -> None:
    rows.append(
        {
            "name": name,
            "value": float(value),
            "threshold": float(threshold),
            "status": "PASS" if abs(float(value)) <= float(threshold) else "FAIL",
        }
    )


def random_rotations(n: int, device: torch.device) -> torch.Tensor:
    q = torch.randn(n, 4, device=device)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    row0 = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), dim=-1)
    row1 = torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), dim=-1)
    row2 = torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def assert_rotation_matrices(rows: list[dict[str, object]], name: str, rot: torch.Tensor, threshold: float) -> None:
    eye = torch.eye(3, dtype=rot.dtype, device=rot.device)
    ortho = max_abs(rot @ rot.transpose(-1, -2) - eye)
    det = max_abs(torch.linalg.det(rot) - 1.0)
    metric(rows, f"{name}: orthonormal", ortho, threshold)
    metric(rows, f"{name}: det+1", det, threshold)


def load_clips(args: argparse.Namespace, cfg: tl.TrainConfig) -> list[tl.MotionClip]:
    specs = tl.clip_specs_from_folders(None, args.periodic_folder, args.nonperiodic_folder)
    clips = tl.load_clips_from_specs(specs, cfg)
    if args.max_clips > 0 and len(clips) > args.max_clips:
        # Keep named stress clips when present, then fill with a deterministic sample.
        important = {
            "M_Neutral_Walk_Loop_F",
            "M_Neutral_Walk_Turn_R_090_Lfoot",
            "M_Neutral_Walk_Turn_R_180_Lfoot",
            "M_Neutral_Walk_Spin_LL_to_F_Lfoot",
            "M_Neutral_Walk_Arc_F_Wide_L",
        }
        picked = [clip for clip in clips if clip.path.stem in important]
        rest = [clip for clip in clips if clip.path.stem not in important]
        random.Random(1234).shuffle(rest)
        clips = (picked + rest)[: args.max_clips]
    return clips


def gather_frames(clip: tl.MotionClip, count: int, device: torch.device) -> torch.Tensor:
    lo = 1
    hi = max(lo + 1, clip.cyclic_period if clip.cyclic_animation else clip.T - 2)
    if hi <= lo:
        return torch.ones((1,), dtype=torch.long, device=device)
    values = torch.linspace(lo, hi, steps=min(count, hi - lo + 1), device=device).round().long()
    return values.unique()


def gather_future_safe_frames(clip: tl.MotionClip, cfg: tl.TrainConfig, count: int, device: torch.device) -> torch.Tensor:
    lo = 1
    hi = max(lo, clip.cyclic_period - 1 if clip.cyclic_animation else ae_prior.clip_future_safe_current_max(clip, cfg))
    if hi <= lo:
        return torch.ones((1,), dtype=torch.long, device=device)
    values = torch.linspace(lo, hi, steps=min(count, hi - lo + 1), device=device).round().long()
    return values.unique()


def unpacked_build_input_rows(
    clips: list[tl.MotionClip],
    clip_ids: torch.Tensor,
    prev_idx: torch.Tensor,
    cur_idx: torch.Tensor,
    cfg: tl.TrainConfig,
    device: torch.device,
) -> torch.Tensor:
    out = torch.empty((clip_ids.numel(), tl.make_batch_dims(clips[0], cfg)[0]), dtype=torch.float32, device=device)
    for ci in sorted(set(int(x) for x in clip_ids.cpu().tolist())):
        mask = (clip_ids == ci).nonzero(as_tuple=False).flatten()
        clip = clips[ci]
        prev = prev_idx.index_select(0, mask)
        cur = cur_idx.index_select(0, mask)
        prev_pose = tl.get_pose_from_clip(clip, prev, device)
        cur_pose = tl.get_pose_from_clip(clip, cur, device)
        out[mask] = tl.build_input(clip, prev, cur, prev_pose, cur_pose, cfg, device)
    return out


def manual_root_delta_feature(pos_a, rot_a, pos_b, rot_b, cfg: tl.TrainConfig) -> torch.Tensor:
    yaw_a = tl.heading_yaw_from_root(rot_a)
    yaw_b = tl.heading_yaw_from_root(rot_b)
    heading_a = tl.yaw_to_row_matrix(yaw_a)
    local = torch.matmul((pos_b - pos_a).unsqueeze(1), heading_a).squeeze(1)
    return torch.stack(
        (
            local[:, 0] / cfg.max_speed_scale_final,
            local[:, 2] / cfg.max_speed_scale_final,
            tl.wrap_angle(yaw_b - yaw_a) / cfg.max_turn_rate_scale_final,
        ),
        dim=-1,
    )


def manual_future_features(positions: list[torch.Tensor], rotations: list[torch.Tensor], cfg: tl.TrainConfig) -> torch.Tensor:
    cur_pos = positions[0]
    cur_rot = rotations[0]
    cur_yaw = tl.heading_yaw_from_root(cur_rot)
    cur_heading = tl.yaw_to_row_matrix(cur_yaw)
    parts = []
    for k in range(1, cfg.future_window + 1):
        fut_pos = positions[k]
        fut_yaw = tl.heading_yaw_from_root(rotations[k])
        local = torch.matmul((fut_pos - cur_pos).unsqueeze(1), cur_heading).squeeze(1)
        scale = k * cfg.max_speed_scale_final
        dyaw = tl.wrap_angle(fut_yaw - cur_yaw)
        parts.append(
            torch.stack(
                (
                    torch.clamp(local[:, 0] / scale, -2.0, 2.0),
                    torch.clamp(local[:, 2] / scale, -2.0, 2.0),
                    torch.cos(dyaw),
                    torch.sin(dyaw),
                ),
                dim=-1,
            )
        )
    return torch.cat(parts, dim=-1)


def run_audit(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    device = torch.device("cpu")
    cfg = tl.TrainConfig()
    cfg.device = "cpu"
    cfg.use_contact_state = False
    cfg.zero_contact_state = False
    cfg.future_window_seconds = args.future_window_seconds
    random.seed(1234)
    torch.manual_seed(1234)

    rows: list[dict[str, object]] = []
    notes: list[dict[str, object]] = []
    clips = load_clips(args, cfg)
    packed = ae_prior.PackedClips(clips, cfg, device)

    # 1. Raw rotation and 6D representation sanity.
    rand_rot = random_rotations(512, device)
    rt = tl.rotation_6d_to_matrix(tl.rotmat_to_6d(rand_rot))
    metric(rows, "6d random rotation roundtrip", max_abs(rt - rand_rot), 3e-5)
    assert_rotation_matrices(rows, "6d cleaned random rotation", rt, 3e-5)
    degenerate = torch.zeros((4, 6), dtype=torch.float32, device=device)
    degenerate[0] = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], device=device)
    degenerate[1] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device)
    degenerate[2] = torch.tensor([0.0, 1.0, 0.0, 0.0, 1e-10, 0.0], device=device)
    degenerate[3] = torch.tensor([0.0, 0.0, 1e-10, 0.0, 0.0, 0.0], device=device)
    assert_rotation_matrices(rows, "6d degenerate fallback rotation", tl.rotation_6d_to_matrix(degenerate), 3e-5)

    data_rot_chunks = []
    for clip in clips:
        idx = gather_frames(clip, args.frames_per_clip, device)
        data_rot_chunks.append(clip.tensors(device)["global_rot"].index_select(0, idx))
    data_rot = torch.cat(data_rot_chunks, dim=0)
    assert_rotation_matrices(rows, "dataset global rotations", data_rot, 3e-4)
    data_rt = tl.rotation_6d_to_matrix(tl.rotmat_to_6d(data_rot.reshape(-1, 3, 3))).reshape_as(data_rot)
    metric(rows, "6d dataset global rotation roundtrip", max_abs(data_rt - data_rot), 3e-4)

    # 2. Source axis conversion: p_c = p_s P must agree with R_c = P^-1 R_s P.
    src_v = torch.randn(256, 3, device=device)
    src_r = random_rotations(256, device)
    converted_vr = tl.canonicalize_positions(torch.matmul(src_v.unsqueeze(1), src_r).squeeze(1), 3)
    converted_sep = torch.matmul(
        tl.canonicalize_positions(src_v, 3).unsqueeze(1),
        tl.canonicalize_rotations(src_r, 3),
    ).squeeze(1)
    metric(rows, "axis-up conversion vector-rotation consistency", max_abs(converted_vr - converted_sep), 3e-5)

    # 3. FK must reconstruct stored globals for true dataset poses.
    fk_pos_errs = []
    fk_rot_errs = []
    for clip in clips:
        idx = gather_frames(clip, args.frames_per_clip, device)
        pose = tl.get_pose_from_clip(clip, idx, device)
        root_pos, root_rot, _yaw, _heading = tl.root_state(clip, idx, cfg, device)
        pos, rot, canon = tl.fk_from_pose(clip, root_pos, root_rot, pose, device)
        gt_pos, gt_rot = tl.global_from_clip(clip, idx, cfg, device)
        fk_pos_errs.append((pos - gt_pos).abs().max())
        fk_rot_errs.append((rot - gt_rot).abs().max())
        # This is a representation sanity check, not a pure rotation check:
        # some FBX clips contain small animated local translations on non-root
        # bones, while the controller FK intentionally uses fixed rest offsets.
        metric(rows, f"{clip.path.stem}: fixed-offset FK vs stored canonical", max_abs(canon - pose["canon_pos"]), 4e-2)
    metric(rows, "fixed-offset FK dataset position reconstruction", float(torch.stack(fk_pos_errs).max()), 4e-2)
    metric(rows, "FK dataset rotation reconstruction", float(torch.stack(fk_rot_errs).max()), 3e-4)

    # 4. Global yaw equivariance: same local pose under a rotated root must rotate globals but preserve canon.
    equiv_pos_errs = []
    equiv_rot_errs = []
    equiv_canon_errs = []
    yaw_angles = torch.tensor([0.37, math.pi / 2.0, math.pi, -2.13], device=device)
    for clip in clips[: min(len(clips), args.equivariance_clips)]:
        idx = gather_frames(clip, min(args.frames_per_clip, 4), device)
        pose = tl.get_pose_from_clip(clip, idx, device)
        root_pos, root_rot, _yaw, _heading = tl.root_state(clip, idx, cfg, device)
        pos, rot, canon = tl.fk_from_pose(clip, root_pos, root_rot, pose, device)
        for yaw in yaw_angles:
            q = tl.yaw_to_row_matrix(yaw.reshape(()))
            q_batch = q.expand(root_rot.shape[0], 3, 3)
            pos_q, rot_q, canon_q = tl.fk_from_pose(
                clip,
                torch.matmul(root_pos.unsqueeze(1), q_batch).squeeze(1),
                root_rot @ q_batch,
                pose,
                device,
            )
            equiv_pos_errs.append((pos_q - torch.matmul(pos, q)).abs().max())
            equiv_rot_errs.append((rot_q - rot @ q).abs().max())
            equiv_canon_errs.append((canon_q - canon).abs().max())
    metric(rows, "FK global-yaw equivariance positions", float(torch.stack(equiv_pos_errs).max()), 3e-4)
    metric(rows, "FK global-yaw equivariance rotations", float(torch.stack(equiv_rot_errs).max()), 3e-4)
    metric(rows, "FK canonical global-yaw invariance", float(torch.stack(equiv_canon_errs).max()), 3e-4)

    # 5. Root/future feature invariance under synthetic global yaw rotations.
    root_feat_errs = []
    future_feat_errs = []
    for clip in clips[: min(len(clips), args.equivariance_clips)]:
        idx = gather_frames(clip, min(args.frames_per_clip, 8), device)
        idx = idx[idx + cfg.future_window < (clip.cyclic_period if clip.cyclic_animation else clip.T)]
        if idx.numel() == 0:
            continue
        prev = idx - 1
        original_root = tl.root_delta_feature(clip, prev, idx, cfg, device)
        original_future = tl.future_root_features(clip, idx, cfg, device)
        root_positions = []
        root_rotations = []
        for k in range(0, cfg.future_window + 1):
            p, r, _y, _h = tl.root_state(clip, idx + k, cfg, device)
            root_positions.append(p)
            root_rotations.append(r)
        prev_p, prev_r, _py, _ph = tl.root_state(clip, prev, cfg, device)
        for yaw in yaw_angles:
            q = tl.yaw_to_row_matrix(yaw.reshape(())).expand(idx.numel(), 3, 3)
            cur_p = torch.matmul(root_positions[0].unsqueeze(1), q).squeeze(1)
            cur_r = root_rotations[0] @ q
            prev_p_q = torch.matmul(prev_p.unsqueeze(1), q).squeeze(1)
            prev_r_q = prev_r @ q
            rotated_root = manual_root_delta_feature(prev_p_q, prev_r_q, cur_p, cur_r, cfg)
            fut_positions = [torch.matmul(p.unsqueeze(1), q).squeeze(1) for p in root_positions]
            fut_rotations = [r @ q for r in root_rotations]
            rotated_future = manual_future_features(fut_positions, fut_rotations, cfg)
            root_feat_errs.append((rotated_root - original_root).abs().max())
            future_feat_errs.append((rotated_future - original_future).abs().max())
    metric(rows, "root delta feature global-yaw invariance", float(torch.stack(root_feat_errs).max()), 3e-4)
    metric(rows, "future root feature global-yaw invariance", float(torch.stack(future_feat_errs).max()), 3e-4)

    # 6. Packed path must match per-clip path, including cyclic multi-cycle root states.
    clip_id_values = []
    idx_values = []
    for ci, clip in enumerate(clips):
        idx = gather_future_safe_frames(clip, cfg, max(2, args.frames_per_clip // 2), device)
        if clip.cyclic_animation:
            idx = torch.cat((idx, idx + clip.cyclic_period, idx + 2 * clip.cyclic_period))
        clip_id_values.append(torch.full_like(idx, ci))
        idx_values.append(idx)
    clip_ids = torch.cat(clip_id_values).long()
    cur_idx = torch.cat(idx_values).long()
    prev_idx = cur_idx - 1
    packed_prev = packed.get_pose(clip_ids, prev_idx)
    packed_cur = packed.get_pose(clip_ids, cur_idx)
    packed_inp = ae_prior.packed_build_input(packed, clip_ids, prev_idx, cur_idx, packed_prev, packed_cur, cfg)
    unpacked_inp = unpacked_build_input_rows(clips, clip_ids, prev_idx, cur_idx, cfg, device)
    metric(rows, "packed vs unpacked build_input", max_abs(packed_inp - unpacked_inp), 3e-4)

    packed_root = packed.root_state(clip_ids, cur_idx)
    root_pos_parts = []
    root_rot_parts = []
    root_yaw_parts = []
    root_heading_parts = []
    for ci in sorted(set(int(x) for x in clip_ids.cpu().tolist())):
        mask = (clip_ids == ci).nonzero(as_tuple=False).flatten()
        p, r, y, h = tl.root_state(clips[ci], cur_idx.index_select(0, mask), cfg, device)
        root_pos_parts.append((mask, p))
        root_rot_parts.append((mask, r))
        root_yaw_parts.append((mask, y))
        root_heading_parts.append((mask, h))
    ref_pos = torch.empty_like(packed_root[0])
    ref_rot = torch.empty_like(packed_root[1])
    ref_yaw = torch.empty_like(packed_root[2])
    ref_heading = torch.empty_like(packed_root[3])
    for target, parts in ((ref_pos, root_pos_parts), (ref_rot, root_rot_parts), (ref_yaw, root_yaw_parts), (ref_heading, root_heading_parts)):
        for mask, value in parts:
            target[mask] = value
    metric(rows, "packed vs unpacked root_state pos", max_abs(packed_root[0] - ref_pos), 3e-4)
    metric(rows, "packed vs unpacked root_state rot", max_abs(packed_root[1] - ref_rot), 3e-4)
    metric(rows, "packed vs unpacked root_state yaw", max_abs(tl.wrap_angle(packed_root[2] - ref_yaw)), 3e-4)
    metric(rows, "packed vs unpacked root_state heading", max_abs(packed_root[3] - ref_heading), 3e-4)

    packed_pos, packed_rot, packed_canon = packed.fk_from_pose(clip_ids, packed_root[0], packed_root[1], packed_cur)
    unpacked_pos_rows = torch.empty_like(packed_pos)
    unpacked_rot_rows = torch.empty_like(packed_rot)
    unpacked_canon_rows = torch.empty_like(packed_canon)
    for ci in sorted(set(int(x) for x in clip_ids.cpu().tolist())):
        mask = (clip_ids == ci).nonzero(as_tuple=False).flatten()
        p, r, c = tl.fk_from_pose(
            clips[ci],
            packed_root[0].index_select(0, mask),
            packed_root[1].index_select(0, mask),
            {k: v.index_select(0, mask) for k, v in packed_cur.items()},
            device,
        )
        unpacked_pos_rows[mask] = p
        unpacked_rot_rows[mask] = r
        unpacked_canon_rows[mask] = c
    metric(rows, "packed vs unpacked FK pos", max_abs(packed_pos - unpacked_pos_rows), 3e-4)
    metric(rows, "packed vs unpacked FK rot", max_abs(packed_rot - unpacked_rot_rows), 3e-4)
    metric(rows, "packed vs unpacked FK canon", max_abs(packed_canon - unpacked_canon_rows), 3e-4)

    next_pose = packed.get_pose(clip_ids, cur_idx + 1)
    packed_feature = ae_prior.packed_transition_feature_from_next_pose(
        packed, clip_ids, prev_idx, cur_idx, packed_prev, packed_cur, next_pose, cfg
    )
    unpacked_feature = torch.empty_like(packed_feature)
    for ci in sorted(set(int(x) for x in clip_ids.cpu().tolist())):
        mask = (clip_ids == ci).nonzero(as_tuple=False).flatten()
        clip = clips[ci]
        local_prev = prev_idx.index_select(0, mask)
        local_cur = cur_idx.index_select(0, mask)
        prev_pose = tl.get_pose_from_clip(clip, local_prev, device)
        cur_pose = tl.get_pose_from_clip(clip, local_cur, device)
        local_next = tl.get_pose_from_clip(clip, local_cur + 1, device)
        unpacked_feature[mask] = tae.transition_feature_from_next_pose(
            clip, local_prev, local_cur, prev_pose, cur_pose, local_next, cfg, device
        )
    metric(rows, "packed vs unpacked transition feature", max_abs(packed_feature - unpacked_feature), 3e-4)

    # 7. Contact slide geometry should be invariant to a global yaw rotation too.
    slide_errs = []
    for clip in clips[: min(len(clips), args.equivariance_clips)]:
        idx = gather_frames(clip, min(args.frames_per_clip, 8), device)
        idx = idx[idx + 1 < (clip.cyclic_period if clip.cyclic_animation else clip.T)]
        if idx.numel() == 0:
            continue
        cur_pos, cur_rot = tl.global_from_clip(clip, idx, cfg, device)
        nxt_pos, nxt_rot = tl.global_from_clip(clip, idx + 1, cfg, device)
        speeds = cp.foot_slide_speeds(
            cur_pos, cur_rot, nxt_pos, nxt_rot, clip.foot_indices, clip.toe_indices, clip.fps
        )
        for yaw in yaw_angles:
            q = tl.yaw_to_row_matrix(yaw.reshape(())).expand(cur_pos.shape[0], 3, 3)
            speeds_q = cp.foot_slide_speeds(
                torch.matmul(cur_pos, q),
                cur_rot @ q[:, None, :, :],
                torch.matmul(nxt_pos, q),
                nxt_rot @ q[:, None, :, :],
                clip.foot_indices,
                clip.toe_indices,
                clip.fps,
            )
            slide_errs.append((speeds_q - speeds).abs().max())
    metric(rows, "foot slide speed global-yaw invariance", float(torch.stack(slide_errs).max()), 3e-4)

    failures = [row for row in rows if row["status"] != "PASS"]
    notes.append({"loaded_clips": len(clips), "failures": len(failures)})
    return rows, notes


def main() -> int:
    parser = argparse.ArgumentParser(description="Rotation/basis invariant audit for Stepper training code.")
    parser.add_argument("--periodic-folder", default=str(PROJECT_ROOT / "ue5" / "animations_omni_only_full"))
    parser.add_argument("--nonperiodic-folder", default=str(PROJECT_ROOT / "ue5" / "animations_transitions_only_full_trimmed"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "training" / "runs" / "diagnostics" / "rotation_audit.json"))
    parser.add_argument("--max-clips", type=int, default=0, help="0 audits all clips.")
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--equivariance-clips", type=int, default=24)
    parser.add_argument("--future-window-seconds", type=float, default=0.25)
    args = parser.parse_args()

    rows, notes = run_audit(args)
    payload = {"metrics": rows, "summary": notes}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 1 if any(row["status"] != "PASS" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
