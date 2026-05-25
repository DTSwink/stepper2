from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

try:
    from .bootstrap import PROJECT_ROOT, ensure_paths
    from . import excess_envelope as env
    from . import ik_core as tl
    from . import train_simple_ae_controller as ctl
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    import excess_envelope as env
    import ik_core as tl
    import train_simple_ae_controller as ctl

ensure_paths()


DEFAULT_NPZ = PROJECT_ROOT / "ue5" / "animations_omni_only_full" / "npz_final" / "M_Neutral_Walk_Loop_F.npz"
TURN_45L_NPZ = (
    PROJECT_ROOT
    / "ue5"
    / "animations_transitions_only_full_trimmed_turn_in_place"
    / "npz_final"
    / "M_Neutral_Stand_Turn_045_L.npz"
)


def _store(path: Path = DEFAULT_NPZ) -> ctl.SimpleClipStore:
    device = torch.device("cpu")
    cfg = tl.TrainConfig()
    cfg.pose_representation = tl.IK_POSE_REPRESENTATION
    cfg.device = str(device)
    cfg.use_torch_compile = False
    clip = tl.MotionClip(path, cfg, cyclic_animation=True)
    return ctl.SimpleClipStore([clip], cfg, device)


def test_transition_output_reference_switch_contract() -> None:
    previous_mode = tl.OUTPUT_REFERENCE_ROOT
    try:
        store = _store()
        clip = store.prototype
        clip_ids = torch.zeros(8, dtype=torch.long)
        cur_idx = torch.arange(1, 9, dtype=torch.long)

        for mode in ("current", "future"):
            tl.OUTPUT_REFERENCE_ROOT = mode
            out = ctl.transition_target_output(store, clip_ids, cur_idx)
            out_pose, _raw = tl.output_to_pose(out, clip)
            root_pos, root_rot = ctl.transition_output_root_state(store, clip_ids, cur_idx)
            pred_pos, pred_rot, _canon = tl.fk_from_pose(clip, root_pos, root_rot, out_pose, torch.device("cpu"))
            gt_pos, gt_rot = tl.global_from_clip(clip, cur_idx + 1, store.cfg, torch.device("cpu"))

            state_vec, _pelvis, _payload = ctl.advance_transition_state(store, clip_ids, cur_idx, out)
            gt_state_vec = store.get_target_output(clip_ids, cur_idx + 1)

            assert float((pred_pos - gt_pos).abs().max()) < 1e-5
            assert float((pred_rot - gt_rot).abs().max()) < 1e-5
            assert float((state_vec - gt_state_vec).abs().max()) < 1e-5
    finally:
        tl.OUTPUT_REFERENCE_ROOT = previous_mode


def test_envelope_loss_uses_active_transition_output_root() -> None:
    previous_mode = tl.OUTPUT_REFERENCE_ROOT
    try:
        store = _store()
        envelope = env.load_or_build_excess_envelope(store, env.ExcessEnvelopeConfig(knn=8))
        clip_ids = torch.zeros(12, dtype=torch.long)
        cur_idx = torch.arange(1, 13, dtype=torch.long)

        for mode in ("current", "future"):
            tl.OUTPUT_REFERENCE_ROOT = mode
            cur_vec = store.get_target_output(clip_ids, cur_idx)
            pred_vec = ctl.transition_target_output(store, clip_ids, cur_idx)
            cur_root_pos, cur_root_rot, _cur_yaw, _cur_heading = store.root_state(clip_ids, cur_idx)
            pred_root_pos, pred_root_rot = ctl.transition_output_root_state(store, clip_ids, cur_idx)
            cur_foot_pos, cur_foot_rot = env.ik_foot_toe_state_from_vec(store, cur_root_pos, cur_root_rot, cur_vec)
            pred_foot_pos, pred_foot_rot = env.ik_foot_toe_state_from_vec(store, pred_root_pos, pred_root_rot, pred_vec)
            linear_rows, angular_rows = env.envelope_excess_ik_state_rows(
                store,
                envelope,
                cur_foot_pos,
                cur_foot_rot,
                pred_foot_pos,
                pred_foot_rot,
                clip_ids,
                cur_idx,
            )

            assert float(linear_rows.abs().max()) < 1e-6
            assert float(angular_rows.abs().max()) < 1e-6
    finally:
        tl.OUTPUT_REFERENCE_ROOT = previous_mode


def test_noncyclic_root_features_are_clamped_near_tail() -> None:
    device = torch.device("cpu")
    cfg = tl.TrainConfig()
    cfg.pose_representation = tl.IK_POSE_REPRESENTATION
    cfg.device = str(device)
    clip = tl.MotionClip(TURN_45L_NPZ, cfg, cyclic_animation=False)
    store = ctl.SimpleClipStore([clip], cfg, device)
    old_unfilled_start = int(clip.T) - ctl.transition_feature_horizon(cfg)
    idx = torch.arange(old_unfilled_start, int(clip.T), dtype=torch.long)
    clip_ids = torch.zeros_like(idx)
    features = store.get_input_root_features(clip_ids, idx)

    assert torch.isfinite(features).all()
    assert float(features.abs().sum(dim=1).max()) > 0.0


class _ZeroController(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.shape[0], self.output_dim), dtype=x.dtype, device=x.device)


def test_residual_zero_output_current_statue_future_hover_contract() -> None:
    previous_root = tl.OUTPUT_REFERENCE_ROOT
    previous_prediction = tl.OUTPUT_PREDICTION_MODE
    try:
        for root_mode in ("current", "future"):
            tl.OUTPUT_REFERENCE_ROOT = root_mode
            tl.OUTPUT_PREDICTION_MODE = "residual"
            store = _store()
            clip = store.prototype
            input_dim, output_dim = tl.make_batch_dims(clip, store.cfg)
            model = _ZeroController(output_dim)
            clip_ids = torch.zeros(1, dtype=torch.long)
            cur_idx = torch.tensor([10], dtype=torch.long)
            prev_vec, prev_pelvis, prev_payload = ctl.target_state(store, clip_ids, cur_idx - 1)
            cur_vec, cur_pelvis, cur_payload = ctl.target_state(store, clip_ids, cur_idx)
            inp = ctl.build_controller_input(
                store,
                clip_ids,
                cur_idx,
                prev_vec,
                cur_vec,
                prev_pelvis,
                cur_pelvis,
                prev_payload,
                cur_payload,
            )
            assert int(inp.shape[-1]) == int(input_dim)
            pred_vec = ctl.clean_output_vector(ctl.model_forward(model, inp, cur_vec, store.cfg), store)
            assert float((pred_vec - cur_vec).abs().max()) < 1e-5

            pred_pose, _raw = tl.output_to_pose(pred_vec, clip)
            pred_root_pos, pred_root_rot = ctl.transition_output_root_state(store, clip_ids, cur_idx)
            pred_pos, _pred_rot, _canon = tl.fk_from_pose(clip, pred_root_pos, pred_root_rot, pred_pose, store.device)
            cur_pos, _cur_rot = tl.global_from_clip(clip, cur_idx, store.cfg, store.device)

            mean_shift = float((pred_pos - cur_pos).norm(dim=-1).mean())
            if root_mode == "current":
                assert mean_shift < 1e-5
            else:
                assert mean_shift > 1e-3
    finally:
        tl.OUTPUT_REFERENCE_ROOT = previous_root
        tl.OUTPUT_PREDICTION_MODE = previous_prediction
