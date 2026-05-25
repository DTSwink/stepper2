"""Diagnose foot-contact behaviour of a trained IK controller.

For each provided NPZ clip, run a long autoregressive rollout of the controller
(starting from frame 0 of the clip) and report:

  - per-foot world height stats and "below contact threshold" duty cycle;
  - hover ratio (fraction of frames where BOTH feet are above the threshold);
  - per-foot horizontal world-speed stats while in soft contact (foot slide);
  - gait period estimate from L-foot contact crossings.

This is a pure-evaluation script: no gradients, no training. It uses dataset
NPZs only as a source of root-motion *commands* (clip root_pos / root_rot,
which determine the future_window / root_features the controller consumes).
The body pose comes entirely from the controller.

Usage:
    python -m training.ik.rl_kin_diagnostic \
        --checkpoint <path-to-controller-checkpoint.pt> \
        --npz <clip1.npz> [<clip2.npz> ...] \
        [--frames 300] [--contact-threshold-m 0.10]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

try:
    from . import ik_core as tl
    from .train_simple_ae_controller import (
        SimpleClipStore,
        build_controller_input,
        clean_output_vector,
        advance_transition_state,
        target_state,
        model_forward,
    )
    from .rl_loss import foot_world_positions
except ImportError:
    _HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(_HERE))
    import ik_core as tl  # type: ignore
    from train_simple_ae_controller import (  # type: ignore
        SimpleClipStore,
        build_controller_input,
        clean_output_vector,
        advance_transition_state,
        target_state,
        model_forward,
    )
    from rl_loss import foot_world_positions  # type: ignore


def load_controller(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, tl.TrainConfig, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_payload = ckpt.get("config")
    if isinstance(config_payload, dict):
        cfg = tl.TrainConfig()
        for key, value in config_payload.items():
            if hasattr(cfg, key):
                if isinstance(getattr(cfg, key), tuple) and isinstance(value, list):
                    value = tuple(value)
                setattr(cfg, key, value)
    else:
        cfg = tl.TrainConfig()
    cfg.device = str(device)
    cfg.pose_representation = tl.IK_POSE_REPRESENTATION
    cfg.predict_residual = tl.output_prediction_uses_residual()
    cfg.zero_init_output = tl.output_prediction_uses_residual()
    metadata = ckpt.get("metadata") or {}
    input_dim = int(metadata.get("input_dim", 0)) or None
    output_dim = int(metadata.get("output_dim", 0)) or None
    if input_dim is None or output_dim is None:
        raise RuntimeError("checkpoint metadata missing input/output dims")
    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def rollout_clip(
    model: torch.nn.Module,
    store: SimpleClipStore,
    clip_id: int,
    n_frames: int,
    cfg: tl.TrainConfig,
) -> dict[str, torch.Tensor]:
    device = store.device
    clip_ids = torch.tensor([clip_id], dtype=torch.long, device=device)
    cur_idx = torch.tensor([0], dtype=torch.long, device=device)
    prev_vec, prev_pelvis, prev_payload = target_state(store, clip_ids, cur_idx)
    cur_vec, cur_pelvis, cur_payload = target_state(store, clip_ids, cur_idx)

    foot_world_history: list[torch.Tensor] = []
    pelvis_world_history: list[torch.Tensor] = []
    root_pos_history: list[torch.Tensor] = []
    root_rot_history: list[torch.Tensor] = []

    root_pos_t, root_rot_t, _yaw, _h = store.root_state(clip_ids, cur_idx)
    foot_world_t = foot_world_positions(cur_vec, root_pos_t, root_rot_t)
    foot_world_history.append(foot_world_t)
    pelvis_world_history.append(root_pos_t + torch.einsum("blk,bkj->blj", cur_vec[:, None, :3], root_rot_t).squeeze(1))
    root_pos_history.append(root_pos_t)
    root_rot_history.append(root_rot_t)

    for _step in range(int(n_frames) - 1):
        inp = build_controller_input(
            store, clip_ids, cur_idx, prev_vec, cur_vec, prev_pelvis, cur_pelvis, prev_payload, cur_payload
        )
        raw = model_forward(model, inp, cur_vec, cfg)
        pred_vec = clean_output_vector(raw, store)

        next_vec, next_pelvis, next_payload = advance_transition_state(store, clip_ids, cur_idx, pred_vec)

        prev_vec, prev_pelvis, prev_payload = cur_vec, cur_pelvis, cur_payload
        cur_vec, cur_pelvis, cur_payload = next_vec, next_pelvis, next_payload
        cur_idx = cur_idx + 1

        rp, rr, _yaw, _h = store.root_state(clip_ids, cur_idx)
        foot_world_t = foot_world_positions(cur_vec, rp, rr)
        foot_world_history.append(foot_world_t)
        pelvis_world_history.append(rp + torch.einsum("blk,bkj->blj", cur_vec[:, None, :3], rr).squeeze(1))
        root_pos_history.append(rp)
        root_rot_history.append(rr)

    return {
        "foot_world": torch.cat(foot_world_history, dim=0),
        "pelvis_world": torch.cat(pelvis_world_history, dim=0),
        "root_pos": torch.cat(root_pos_history, dim=0),
        "root_rot": torch.stack(root_rot_history, dim=0).squeeze(1),
    }


def gait_period_estimate(contact_signal: torch.Tensor, fps: float) -> float:
    """Estimate gait period (seconds) by counting rising edges of contact."""
    sig = contact_signal.cpu().numpy()
    rises = 0
    for i in range(1, len(sig)):
        if sig[i] > 0.5 and sig[i - 1] <= 0.5:
            rises += 1
    if rises < 1:
        return float("nan")
    return (len(sig) - 1) / fps / max(1, rises)


def report_clip(name: str, foot_world: torch.Tensor, pelvis_world: torch.Tensor, root_pos: torch.Tensor, threshold_m: float, fps: float) -> dict:
    height = foot_world[..., 1]  # (T, 2)
    horiz_v = (foot_world[1:, :, (0, 2)] - foot_world[:-1, :, (0, 2)]).norm(dim=-1) * float(fps)

    contact = (height < threshold_m).float()
    duty_l = float(contact[:, 0].mean())
    duty_r = float(contact[:, 1].mean())
    both_off = ((1.0 - contact[:, 0]) * (1.0 - contact[:, 1]))
    hover_ratio = float(both_off.mean())

    in_contact_speed_l = horiz_v[:, 0][(contact[:-1, 0] > 0.5)]
    in_contact_speed_r = horiz_v[:, 1][(contact[:-1, 1] > 0.5)]
    slide_l = float(in_contact_speed_l.mean()) if in_contact_speed_l.numel() > 0 else float("nan")
    slide_r = float(in_contact_speed_r.mean()) if in_contact_speed_r.numel() > 0 else float("nan")
    slide_l_max = float(in_contact_speed_l.max()) if in_contact_speed_l.numel() > 0 else float("nan")
    slide_r_max = float(in_contact_speed_r.max()) if in_contact_speed_r.numel() > 0 else float("nan")

    period_l = gait_period_estimate(contact[:, 0], fps)
    period_r = gait_period_estimate(contact[:, 1], fps)

    pelvis_horiz = pelvis_world[:, (0, 2)] - root_pos[:, (0, 2)]
    pelvis_horiz_dev = float(pelvis_horiz.norm(dim=-1).mean())
    pelvis_height_mean = float(pelvis_world[:, 1].mean())

    h_min = float(height.min())
    h_max = float(height.max())
    h_mean = float(height.mean())

    print(f"\n=== {name} ===")
    print(f"frames: {height.shape[0]}  fps: {fps:.1f}  contact_threshold_m: {threshold_m:.3f}")
    print(f"foot_y     l: min={float(height[:,0].min()):+.4f}  max={float(height[:,0].max()):+.4f}  mean={float(height[:,0].mean()):+.4f}")
    print(f"foot_y     r: min={float(height[:,1].min()):+.4f}  max={float(height[:,1].max()):+.4f}  mean={float(height[:,1].mean()):+.4f}")
    print(f"contact_duty l={duty_l:.3f}  r={duty_r:.3f}    hover_ratio_both_off={hover_ratio:.3f}")
    print(f"slide_in_contact l: mean={slide_l:.3f}  max={slide_l_max:.3f}  m/s")
    print(f"slide_in_contact r: mean={slide_r:.3f}  max={slide_r_max:.3f}  m/s")
    print(f"gait_period_estimate_s: l={period_l:.3f}  r={period_r:.3f}")
    print(f"pelvis horizontal dev from root (m, world): {pelvis_horiz_dev:.4f}")
    print(f"pelvis world y mean: {pelvis_height_mean:.3f}")

    return {
        "duty_l": duty_l,
        "duty_r": duty_r,
        "hover_ratio": hover_ratio,
        "slide_l_mean": slide_l,
        "slide_r_mean": slide_r,
        "period_l_s": period_l,
        "period_r_s": period_r,
        "pelvis_horiz_dev_m": pelvis_horiz_dev,
        "pelvis_world_y_mean": pelvis_height_mean,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose foot/gait behaviour of a trained IK controller.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--npz", nargs="+", required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--contact-threshold-m", type=float, default=0.10)
    parser.add_argument("--cyclic", nargs="+", default=None,
                        help="Per-clip cyclic flags ('1' or '0'). Defaults to 1 (cyclic) for each clip.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint).resolve()
    model, cfg, _ckpt = load_controller(checkpoint_path, device)
    print(f"loaded controller from {checkpoint_path}")
    print(f"  input_dim={model.net[0].in_features} output_dim={model.net[-1].out_features}")
    print(f"  fps={cfg.fps}")

    cyclic_flags = [True] * len(args.npz) if args.cyclic is None else [bool(int(v)) for v in args.cyclic]
    if len(cyclic_flags) != len(args.npz):
        raise ValueError("--cyclic must match number of --npz entries")

    clips = [tl.MotionClip(Path(p).resolve(), cfg, cyclic_animation=c) for p, c in zip(args.npz, cyclic_flags)]
    first_names = clips[0].body_names
    first_parents = clips[0].parents_body_list
    for clip in clips[1:]:
        if clip.body_names != first_names or clip.parents_body_list != first_parents:
            raise ValueError(f"skeleton mismatch: {clip.path}")

    store = SimpleClipStore(clips, cfg, device)

    for clip_id, clip in enumerate(clips):
        n_frames = min(int(args.frames), int(clip.T) - 1) if not clip.cyclic_animation else int(args.frames)
        out = rollout_clip(model, store, clip_id, n_frames, cfg)
        report_clip(
            clip.path.name,
            out["foot_world"],
            out["pelvis_world"],
            out["root_pos"],
            float(args.contact_threshold_m),
            float(cfg.fps),
        )


if __name__ == "__main__":
    main()
