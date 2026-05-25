from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

try:
    from .bootstrap import PROJECT_ROOT, ensure_paths
    from .naming import checkpoint_path, ik_run_id
    from . import ik_core as tl
    from . import train_simple_autoencoder as ae_tr
    from . import train_simple_ae_controller as ctl
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    from naming import checkpoint_path, ik_run_id
    import ik_core as tl
    import train_simple_autoencoder as ae_tr
    import train_simple_ae_controller as ctl

ensure_paths()

WALK_F = PROJECT_ROOT / "ue5" / "animations_omni_only_full" / "npz_final" / "M_Neutral_Walk_Loop_F.npz"
RUNS_DIR = PROJECT_ROOT / "training" / "runs"
BASE_AE = (
    RUNS_DIR
    / "20260522_013146_ik_simple_ae_walkF_probe"
    / "checkpoints"
    / "20260522_013146_ik_simple_ae_walkF_probe_best.pt"
)
BASE_CONTROLLER = (
    RUNS_DIR
    / "20260522_041619_ik_walkF_ae_output_only_pure"
    / "checkpoints"
    / "20260522_041619_ik_walkF_ae_output_only_pure_last.pt"
)
RESULTS_PATH = PROJECT_ROOT / "training" / "ik" / "foot_harshness_results.json"
METRIC_VERSION = "height_window_full_interval_plus_onestep_v3"


def make_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    return device


def load_store(
    cfg: tl.TrainConfig,
    device: torch.device,
    clip_path: Path = WALK_F,
    cyclic: bool = True,
) -> ctl.SimpleClipStore:
    clip = tl.MotionClip(clip_path, cfg, cyclic_animation=cyclic)
    return ctl.SimpleClipStore([clip], cfg, device)


def foot_indices(clip: tl.MotionClip) -> dict[str, int]:
    return {name: int(clip.body_names.index(name)) for name in ("foot_l", "foot_r")}


@torch.no_grad()
def gt_sequence(store: ctl.SimpleClipStore) -> tuple[torch.Tensor, torch.Tensor]:
    clip = store.prototype
    idx = torch.arange(int(clip.T), dtype=torch.long, device=store.device)
    clip_ids = torch.zeros_like(idx)
    pose = store.get_pose(clip_ids, idx)
    root_pos, root_rot, _yaw, _heading = store.root_state(clip_ids, idx)
    return fk_full_by_clip(store, clip_ids, root_pos, root_rot, pose)


@torch.no_grad()
def fk_rotations_by_clip(
    store: ctl.SimpleClipStore,
    clip_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    pose: dict[str, torch.Tensor],
) -> torch.Tensor:
    out = torch.empty((clip_ids.shape[0], store.J, 3, 3), dtype=root_pos.dtype, device=store.device)
    for clip_id in clip_ids.unique().tolist():
        rows = (clip_ids == int(clip_id)).nonzero(as_tuple=False).flatten()
        _pos, rot, _canon = tl.fk_from_pose(
            store.clips[int(clip_id)],
            root_pos.index_select(0, rows),
            root_rot.index_select(0, rows),
            ctl.pose_rows(pose, rows),
            store.device,
        )
        out[rows] = rot
    return out


@torch.no_grad()
def fk_full_by_clip(
    store: ctl.SimpleClipStore,
    clip_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    pose: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    out_pos = torch.empty((clip_ids.shape[0], store.J, 3), dtype=root_pos.dtype, device=store.device)
    out_rot = torch.empty((clip_ids.shape[0], store.J, 3, 3), dtype=root_pos.dtype, device=store.device)
    for clip_id in clip_ids.unique().tolist():
        rows = (clip_ids == int(clip_id)).nonzero(as_tuple=False).flatten()
        pos, rot, _canon = tl.fk_from_pose(
            store.clips[int(clip_id)],
            root_pos.index_select(0, rows),
            root_rot.index_select(0, rows),
            ctl.pose_rows(pose, rows),
            store.device,
        )
        out_pos[rows] = pos
        out_rot[rows] = rot
    return out_pos, out_rot


@torch.no_grad()
def rollout_sequence(model: torch.nn.Module, store: ctl.SimpleClipStore, cfg: tl.TrainConfig) -> tuple[torch.Tensor, torch.Tensor]:
    clip = store.prototype
    device = store.device
    frame_count = int(clip.T)
    pred_pos = torch.empty((frame_count, store.J, 3), dtype=torch.float32, device=device)
    pred_rot = torch.empty((frame_count, store.J, 3, 3), dtype=torch.float32, device=device)

    prev_idx = torch.tensor([0], dtype=torch.long, device=device)
    cur_idx = torch.tensor([1], dtype=torch.long, device=device)
    clip_ids = torch.zeros((1,), dtype=torch.long, device=device)
    prev_vec, prev_pelvis, prev_payload = ctl.target_state(store, clip_ids, prev_idx)
    cur_vec, cur_pelvis, cur_payload = ctl.target_state(store, clip_ids, cur_idx)

    init_pose = store.get_pose(torch.zeros((2,), dtype=torch.long, device=device), torch.tensor([0, 1], dtype=torch.long, device=device))
    init_root_pos, init_root_rot, _yaw, _heading = store.root_state(
        torch.zeros((2,), dtype=torch.long, device=device), torch.tensor([0, 1], dtype=torch.long, device=device)
    )
    init_pos, init_rot = fk_full_by_clip(
        store, torch.zeros((2,), dtype=torch.long, device=device), init_root_pos, init_root_rot, init_pose
    )
    pred_pos[:2] = init_pos
    pred_rot[:2] = init_rot

    for target in range(2, frame_count):
        target_idx = torch.tensor([target], dtype=torch.long, device=device)
        inp = ctl.build_controller_input(
            store, clip_ids, cur_idx, prev_vec, cur_vec, prev_pelvis, cur_pelvis, prev_payload, cur_payload
        )
        raw = ctl.model_forward(model, inp, cur_vec, cfg)
        pred_vec = ctl.clean_output_vector(raw, store)
        pred_pose, _raw_pose = tl.output_to_pose(pred_vec, clip)
        root_pos, root_rot, _yaw, _heading = store.root_state(clip_ids, cur_idx)
        global_pos, global_rot = fk_full_by_clip(store, clip_ids, root_pos, root_rot, pred_pose)
        pred_pos[target] = global_pos[0]
        pred_rot[target] = global_rot[0]

        prev_vec, prev_pelvis, prev_payload = cur_vec, cur_pelvis, cur_payload
        cur_vec, cur_pelvis, cur_payload = ctl.advance_transition_state(store, clip_ids, cur_idx, pred_vec)
        cur_idx = target_idx
    return pred_pos, pred_rot


@torch.no_grad()
def one_step_sequence(model: torch.nn.Module, store: ctl.SimpleClipStore, cfg: tl.TrainConfig) -> tuple[torch.Tensor, torch.Tensor]:
    clip = store.prototype
    device = store.device
    frame_count = int(clip.T)
    pred_pos = torch.empty((frame_count, store.J, 3), dtype=torch.float32, device=device)
    pred_rot = torch.empty((frame_count, store.J, 3, 3), dtype=torch.float32, device=device)

    init_pose = store.get_pose(torch.zeros((2,), dtype=torch.long, device=device), torch.tensor([0, 1], dtype=torch.long, device=device))
    init_root_pos, init_root_rot, _yaw, _heading = store.root_state(
        torch.zeros((2,), dtype=torch.long, device=device), torch.tensor([0, 1], dtype=torch.long, device=device)
    )
    init_pos, init_rot = fk_full_by_clip(
        store, torch.zeros((2,), dtype=torch.long, device=device), init_root_pos, init_root_rot, init_pose
    )
    pred_pos[:2] = init_pos
    pred_rot[:2] = init_rot

    if frame_count <= 2:
        return pred_pos, pred_rot

    target_idx = torch.arange(2, frame_count, dtype=torch.long, device=device)
    clip_ids = torch.zeros_like(target_idx)
    prev_idx = target_idx - 2
    cur_idx = target_idx - 1
    prev_vec, prev_pelvis, prev_payload = ctl.target_state(store, clip_ids, prev_idx)
    cur_vec, cur_pelvis, cur_payload = ctl.target_state(store, clip_ids, cur_idx)
    inp = ctl.build_controller_input(
        store, clip_ids, cur_idx, prev_vec, cur_vec, prev_pelvis, cur_pelvis, prev_payload, cur_payload
    )
    raw = ctl.model_forward(model, inp, cur_vec, cfg)
    pred_vec = ctl.clean_output_vector(raw, store)
    pred_pose, _raw_pose = tl.output_to_pose(pred_vec, clip)
    root_pos, root_rot, _yaw, _heading = store.root_state(clip_ids, cur_idx)
    pred_pos[2:], pred_rot[2:] = fk_full_by_clip(store, clip_ids, root_pos, root_rot, pred_pose)
    return pred_pos, pred_rot


def rotation_angle_deg(rot: torch.Tensor) -> torch.Tensor:
    rel = rot[:-1].transpose(-1, -2) @ rot[1:]
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def rotation_error_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    rel = pred @ target.transpose(-1, -2)
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def contiguous_runs(mask: torch.Tensor) -> list[tuple[int, int]]:
    vals = mask.detach().cpu().tolist()
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(vals):
        if value and start is None:
            start = i
        if (not value or i + 1 == len(vals)) and start is not None:
            end = i + 1 if value and i + 1 == len(vals) else i
            if end > start:
                runs.append((start, end))
            start = None
    return runs


def interval_score(slide: torch.Tensor, rot_deg: torch.Tensor, height: torch.Tensor, start: int, end: int) -> float:
    h0 = float(height.min().detach().cpu())
    h_penalty = torch.relu(height[start:end] - (h0 + 0.035))
    score = slide[start:end] + 0.0025 * rot_deg[start:end] + 2.0 * h_penalty
    return float(score.mean().detach().cpu())


def height_interval_score(height: torch.Tensor, start: int, end: int) -> float:
    h0 = float(height.min().detach().cpu())
    low = (height[start:end] - h0).abs().mean()
    dh = (height[start + 1 : end] - height[start : end - 1]).abs().mean() if end - start > 1 else low * 0.0
    return float((low + 2.0 * dh).detach().cpu())


def choose_gt_pin_interval(slide: torch.Tensor, rot_deg: torch.Tensor, height: torch.Tensor) -> tuple[int, int]:
    h_thresh = torch.quantile(height, 0.35) + 0.025
    s_thresh = torch.quantile(slide, 0.45)
    r_thresh = torch.quantile(rot_deg, 0.70)
    mask = (height <= h_thresh) & (slide <= s_thresh) & (rot_deg <= r_thresh)
    runs = [run for run in contiguous_runs(mask) if run[1] - run[0] >= 6]
    if runs:
        return min(runs, key=lambda run: (interval_score(slide, rot_deg, height, run[0], run[1]), -(run[1] - run[0])))
    length = max(8, int(slide.numel()) // 6)
    return choose_best_window(slide, rot_deg, height, length)


def choose_low_height_window(height: torch.Tensor, length: int) -> tuple[int, int]:
    length = min(max(4, int(length)), int(height.numel()))
    best_start = 0
    best_score = math.inf
    for start in range(0, int(height.numel()) - length + 1):
        score = height_interval_score(height, start, start + length)
        if score < best_score:
            best_score = score
            best_start = start
    return best_start, best_start + length


def choose_best_window(slide: torch.Tensor, rot_deg: torch.Tensor, height: torch.Tensor, length: int) -> tuple[int, int]:
    length = min(max(4, int(length)), int(slide.numel()))
    best_start = 0
    best_score = math.inf
    for start in range(0, int(slide.numel()) - length + 1):
        score = interval_score(slide, rot_deg, height, start, start + length)
        if score < best_score:
            best_score = score
            best_start = start
    return best_start, best_start + length


def foot_motion_series(pos: torch.Tensor, rot: torch.Tensor, foot_i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    foot_pos = pos[:, foot_i]
    slide = (foot_pos[1:, [0, 2]] - foot_pos[:-1, [0, 2]]).norm(dim=-1)
    rot_deg = rotation_angle_deg(rot[:, foot_i])
    height = 0.5 * (foot_pos[:-1, 1] + foot_pos[1:, 1])
    return slide, rot_deg, height


def pinned_metric(
    gt_pos: torch.Tensor,
    gt_rot: torch.Tensor,
    pred_pos: torch.Tensor,
    pred_rot: torch.Tensor,
    clip: tl.MotionClip,
) -> dict[str, object]:
    idx = foot_indices(clip)
    out: dict[str, object] = {}
    slide_ratios: list[float] = []
    rot_ratios: list[float] = []
    for foot_name, foot_i in idx.items():
        gt_slide, gt_rot_deg, gt_height = foot_motion_series(gt_pos, gt_rot, foot_i)
        pred_slide, pred_rot_deg, pred_height = foot_motion_series(pred_pos, pred_rot, foot_i)
        gt_interval = choose_gt_pin_interval(gt_slide, gt_rot_deg, gt_height)
        target_len = gt_interval[1] - gt_interval[0]
        pred_interval = choose_low_height_window(pred_height, target_len)
        gt_eval = gt_interval
        pred_eval = pred_interval
        gt_slide_mean = float(gt_slide[gt_eval[0] : gt_eval[1]].mean().detach().cpu())
        gt_rot_mean = float(gt_rot_deg[gt_eval[0] : gt_eval[1]].mean().detach().cpu())
        pred_slide_mean = float(pred_slide[pred_eval[0] : pred_eval[1]].mean().detach().cpu())
        pred_rot_mean = float(pred_rot_deg[pred_eval[0] : pred_eval[1]].mean().detach().cpu())
        slide_ratio = pred_slide_mean / max(gt_slide_mean, 1e-8)
        rot_ratio = pred_rot_mean / max(gt_rot_mean, 1e-8)
        slide_ratios.append(slide_ratio)
        rot_ratios.append(rot_ratio)
        out[foot_name] = {
            "gt_interval": gt_interval,
            "pred_interval": pred_interval,
            "gt_scored_interval": gt_eval,
            "pred_scored_interval": pred_eval,
            "gt_slide_m_per_frame": gt_slide_mean,
            "pred_slide_m_per_frame": pred_slide_mean,
            "slide_ratio": slide_ratio,
            "gt_rot_deg_per_frame": gt_rot_mean,
            "pred_rot_deg_per_frame": pred_rot_mean,
            "rot_ratio": rot_ratio,
        }
    out["mean_slide_ratio"] = sum(slide_ratios) / len(slide_ratios)
    out["mean_rot_ratio"] = sum(rot_ratios) / len(rot_ratios)
    out["score"] = 0.5 * (float(out["mean_slide_ratio"]) + float(out["mean_rot_ratio"]))
    return out


def one_step_metric(
    gt_pos: torch.Tensor,
    gt_rot: torch.Tensor,
    pred_pos: torch.Tensor,
    pred_rot: torch.Tensor,
    clip: tl.MotionClip,
) -> dict[str, object]:
    pos_err = (pred_pos - gt_pos).norm(dim=-1)
    rot_err = rotation_error_deg(pred_rot, gt_rot)
    body_indices = torch.arange(clip.J, dtype=torch.long, device=gt_pos.device)
    eval_slice = slice(2, None)
    foot_names = [name for name in ("foot_l", "foot_r", "ball_l", "ball_r") if name in clip.body_names]
    foot_idx = torch.tensor([clip.body_names.index(name) for name in foot_names], dtype=torch.long, device=gt_pos.device)
    ee_names = [name for name in ("hand_l", "hand_r", "foot_l", "foot_r", "ball_l", "ball_r") if name in clip.body_names]
    ee_idx = torch.tensor([clip.body_names.index(name) for name in ee_names], dtype=torch.long, device=gt_pos.device)
    pos_eval = pos_err[eval_slice]
    rot_eval = rot_err[eval_slice]
    return {
        "frames_evaluated": int(pos_eval.shape[0]),
        "pos_mean_m": float(pos_eval.mean().detach().cpu()),
        "pos_p95_m": float(torch.quantile(pos_eval.reshape(-1), 0.95).detach().cpu()),
        "pos_max_m": float(pos_eval.max().detach().cpu()),
        "pos_end_m": float(pos_err[-1].mean().detach().cpu()),
        "rot_mean_deg": float(rot_eval.mean().detach().cpu()),
        "rot_p95_deg": float(torch.quantile(rot_eval.reshape(-1), 0.95).detach().cpu()),
        "rot_max_deg": float(rot_eval.max().detach().cpu()),
        "rot_end_deg": float(rot_err[-1].mean().detach().cpu()),
        "ee_pos_mean_m": float(pos_eval.index_select(1, ee_idx).mean().detach().cpu()),
        "foot_pos_mean_m": float(pos_eval.index_select(1, foot_idx).mean().detach().cpu()),
        "foot_rot_mean_deg": float(rot_eval.index_select(1, foot_idx).mean().detach().cpu()),
        "body_count": int(body_indices.numel()),
    }


def load_controller(path: Path, device: torch.device) -> tuple[torch.nn.Module, ctl.SimpleClipStore, tl.TrainConfig]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = tl.TrainConfig()
    ctl.apply_config_dict(cfg, ckpt["config"])
    cfg.device = str(device)
    store = load_store(cfg, device)
    input_dim, output_dim = tl.make_batch_dims(store.prototype, cfg)
    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, store, cfg


@torch.no_grad()
def evaluate_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    model, store, cfg = load_controller(path, device)
    gt_pos, gt_rot = gt_sequence(store)
    ar_pos, ar_rot = rollout_sequence(model, store, cfg)
    step_pos, step_rot = one_step_sequence(model, store, cfg)
    return {
        "foot_pin": pinned_metric(gt_pos, gt_rot, ar_pos, ar_rot, store.prototype),
        "one_step": one_step_metric(gt_pos, gt_rot, step_pos, step_rot, store.prototype),
    }


def metric_is_current(entry: dict[str, object]) -> bool:
    return entry.get("metric_version") == METRIC_VERSION


def refresh_metric(entry: dict[str, object], checkpoint_key: str, device: torch.device) -> None:
    entry["metric"] = evaluate_checkpoint(Path(str(entry[checkpoint_key])), device)
    entry["metric_version"] = METRIC_VERSION


def train_autoencoder_variant(label: str, device: torch.device, latent_dim: int, hidden_dim: int, steps: int) -> Path:
    existing = latest_checkpoint_for_label(label, "best")
    if existing is not None:
        print(f"reuse AE {existing}", flush=True)
        return existing
    torch.manual_seed(1234)
    cfg = ae_tr.SimpleAEConfig(latent_dim=latent_dim, hidden_dim=hidden_dim, train_steps=steps)
    locomotion_cfg = ae_tr.make_locomotion_cfg(device)
    clips = ae_tr.load_clips([(WALK_F, True)], locomotion_cfg)
    raw_features, _clip_ids, _cur_indices, schema = ae_tr.collect_controller_features(clips, locomotion_cfg, device)
    x_cpu, mean_cpu, std_cpu = ae_tr.normalize_features(raw_features, cfg.std_floor)
    x = x_cpu.to(device)
    mean = mean_cpu.to(device)
    std = std_cpu.to(device)
    model = ae_tr.SimpleAutoencoder(x.shape[1], cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    batch_size = min(int(cfg.batch_size), int(x.shape[0]))
    for step in range(1, steps + 1):
        rows = ae_tr.batch_indices(torch.arange(x.shape[0], device=device), batch_size)
        batch = x.index_select(0, rows)
        loss = F.mse_loss(model(batch), batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    run_id = ik_run_id(label)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ae_tr.checkpoint_payload(
        model,
        optimizer,
        cfg,
        locomotion_cfg,
        schema,
        mean,
        std,
        int(steps),
        float(loss.detach().cpu()),
        {
            "npz_paths": [str(WALK_F)],
            "row_count": int(x.shape[0]),
            "audit": "foot_harshness",
            "variant": label,
        },
    )
    path = checkpoint_path(run_dir, run_id, "best")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    return path


def train_output_only_autoencoder_variant(label: str, device: torch.device, latent_dim: int, hidden_dim: int, steps: int) -> Path:
    existing = latest_checkpoint_for_label(label, "best")
    if existing is not None:
        print(f"reuse AE {existing}", flush=True)
        return existing
    torch.manual_seed(4321)
    cfg = ae_tr.SimpleAEConfig(latent_dim=latent_dim, hidden_dim=hidden_dim, train_steps=steps)
    locomotion_cfg = ae_tr.make_locomotion_cfg(device)
    clips = ae_tr.load_clips([(WALK_F, True)], locomotion_cfg)
    raw_features, _clip_ids, _cur_indices, schema = ae_tr.collect_controller_features(clips, locomotion_cfg, device)
    x_cpu, mean_cpu, std_cpu = ae_tr.normalize_features(raw_features, cfg.std_floor)
    x = x_cpu.to(device)
    mean = mean_cpu.to(device)
    std = std_cpu.to(device)
    out_start = int(schema["target_output_start"])
    out_end = int(schema["target_output_end"])
    model = ae_tr.SimpleAutoencoder(x.shape[1], cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    batch_size = min(int(cfg.batch_size), int(x.shape[0]))
    all_rows = torch.arange(x.shape[0], device=device)
    loss = torch.zeros((), device=device)
    for step in range(1, steps + 1):
        rows = ae_tr.batch_indices(all_rows, batch_size)
        batch = x.index_select(0, rows)
        recon = model(batch)
        loss = F.mse_loss(recon[:, out_start:out_end], batch[:, out_start:out_end])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    run_id = ik_run_id(label)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ae_tr.checkpoint_payload(
        model,
        optimizer,
        cfg,
        locomotion_cfg,
        schema,
        mean,
        std,
        int(steps),
        float(loss.detach().cpu()),
        {
            "npz_paths": [str(WALK_F)],
            "row_count": int(x.shape[0]),
            "audit": "foot_harshness",
            "variant": label,
            "ae_training_loss": "target_output_only",
            "negative_examples": False,
            "temporal_window": 1,
        },
    )
    path = checkpoint_path(run_dir, run_id, "best")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    return path


def output_payload_start(schema: dict[str, object]) -> int:
    return int(schema["target_output_start"]) + int(schema["output_dim"]) - int(tl.IK_PAYLOAD_DIM)


def output_foot_slices(schema: dict[str, object], foot_name: str) -> dict[str, slice]:
    base = output_payload_start(schema)
    spec = next(s for s in tl.IK_PAYLOAD_SLICES if s["end"] == foot_name)
    out: dict[str, slice] = {}
    for key in ("pos", "rot6", "pole", "toe_float"):
        value = spec[key]
        if isinstance(value, slice):
            out[key] = slice(base + int(value.start), base + int(value.stop))
    return out


def pinned_foot_masks(
    clips: list[tl.MotionClip],
    clip_ids: torch.Tensor,
    cur_indices: torch.Tensor,
) -> torch.Tensor:
    masks = torch.zeros((int(cur_indices.numel()), 2), dtype=torch.bool)
    target_frames = cur_indices + 1
    for clip_id, clip in enumerate(clips):
        clip_rows = clip_ids == int(clip_id)
        for foot_slot, foot_name in enumerate(("foot_l", "foot_r")):
            foot_i = foot_indices(clip)[foot_name]
            slide, rot_deg, height = foot_motion_series(clip.global_pos, clip.global_rot, foot_i)
            start, end = choose_gt_pin_interval(slide, rot_deg, height)
            foot_rows = clip_rows & (target_frames >= int(start)) & (target_frames <= int(end))
            masks[:, foot_slot] |= foot_rows
    return masks


def corrupt_pinned_output(
    rows_raw: torch.Tensor,
    pinned_masks: torch.Tensor,
    schema: dict[str, object],
) -> torch.Tensor:
    out = rows_raw.clone()
    for foot_slot, foot_name in enumerate(("foot_l", "foot_r")):
        active = pinned_masks[:, foot_slot]
        count = int(active.sum().item())
        if count == 0:
            continue
        slices = output_foot_slices(schema, foot_name)

        pos = out[active, slices["pos"]]
        horizontal = torch.randn((count, 2), dtype=out.dtype, device=out.device) * 0.055
        vertical = torch.randn((count, 1), dtype=out.dtype, device=out.device) * 0.012
        pos_noise = torch.cat((horizontal[:, 0:1], vertical, horizontal[:, 1:2]), dim=-1)
        out[active, slices["pos"]] = pos + pos_noise

        rot6 = out[active, slices["rot6"]]
        out[active, slices["rot6"]] = ctl.fast_clean_6d(rot6 + torch.randn_like(rot6) * 0.35)

        pole = out[active, slices["pole"]]
        out[active, slices["pole"]] = pole + torch.randn_like(pole) * 0.35
        toe_slice = slices.get("toe_float")
        if toe_slice is not None:
            toe = out[active, toe_slice]
            out[active, toe_slice] = toe + torch.randn_like(toe) * 0.45
    return out


def output_weights(schema: dict[str, object], device: torch.device, profile: str) -> torch.Tensor:
    weights = torch.ones((int(schema["output_dim"]),), dtype=torch.float32, device=device)
    if not profile:
        return weights
    payload_offset = int(schema["output_dim"]) - int(tl.IK_PAYLOAD_DIM)
    pos_w = 1.0
    rot_w = 1.0
    pole_w = 1.0
    toe_w = 1.0
    if profile == "foot_x3":
        pos_w, rot_w, pole_w, toe_w = 4.0, 3.0, 2.0, 3.0
    elif profile == "foot_x6":
        pos_w, rot_w, pole_w, toe_w = 8.0, 5.0, 3.0, 5.0
    elif profile == "foot_pos_x6":
        pos_w, rot_w, pole_w, toe_w = 8.0, 1.5, 1.0, 1.5
    else:
        raise ValueError(f"Unknown output weight profile: {profile}")
    for spec in tl.IK_PAYLOAD_SLICES:
        if spec["kind"] != "leg":
            continue
        for key, value, weight in (
            ("pos", spec["pos"], pos_w),
            ("rot6", spec["rot6"], rot_w),
            ("pole", spec["pole"], pole_w),
            ("toe_float", spec["toe_float"], toe_w),
        ):
            if isinstance(value, slice):
                weights[payload_offset + int(value.start) : payload_offset + int(value.stop)] = float(weight)
    return weights


def foot_delta_feature(
    controller_input: torch.Tensor,
    output: torch.Tensor,
    schema: dict[str, object],
    cfg: tl.TrainConfig,
) -> torch.Tensor:
    input_dim = int(schema["input_dim"])
    output_dim = int(schema["output_dim"])
    pose_dim = int(schema["pose_dim"])
    payload_start = output_dim - int(tl.IK_PAYLOAD_DIM)
    if int(cfg.future_window) < 1:
        raise ValueError("foot_delta_feature needs the controller root feature layout")

    deltas: list[torch.Tensor] = []
    for spec in tl.IK_PAYLOAD_SLICES:
        if spec["kind"] != "leg":
            continue
        pos_slice = spec["pos"]
        assert isinstance(pos_slice, slice)
        cur = controller_input[:, payload_start + int(pos_slice.start) : payload_start + int(pos_slice.stop)]
        nxt = output[:, payload_start + int(pos_slice.start) : payload_start + int(pos_slice.stop)]
        deltas.append(nxt - cur)
    return torch.cat(deltas, dim=-1)


def append_foot_delta_feature(
    controller_input: torch.Tensor,
    output: torch.Tensor,
    schema: dict[str, object],
    cfg: tl.TrainConfig,
) -> torch.Tensor:
    return torch.cat((controller_input, output, foot_delta_feature(controller_input, output, schema, cfg)), dim=-1)


def make_foot_delta_score_fn(schema: dict[str, object], cfg: tl.TrainConfig, derived_weight: float):
    input_dim = int(schema["input_dim"])
    output_dim = int(schema["output_dim"])
    derived_start = int(schema["foot_delta_start"])
    derived_end = int(schema["foot_delta_end"])
    derived_weight = float(derived_weight)

    def score_rows(
        ae: ae_tr.SimpleAutoencoder,
        mean: torch.Tensor,
        std: torch.Tensor,
        controller_input: torch.Tensor,
        predicted_output: torch.Tensor,
    ) -> torch.Tensor:
        feature = append_foot_delta_feature(controller_input, predicted_output, schema, cfg)
        x = (feature - mean) / std
        recon = ae(x)
        out_err = (recon[:, input_dim : input_dim + output_dim] - x[:, input_dim : input_dim + output_dim]).square().mean(dim=-1)
        delta_err = (recon[:, derived_start:derived_end] - x[:, derived_start:derived_end]).square().mean(dim=-1)
        return out_err + derived_weight * delta_err

    return score_rows


def train_foot_delta_autoencoder_variant(
    label: str,
    device: torch.device,
    latent_dim: int,
    hidden_dim: int,
    steps: int,
    derived_weight: float = 2.0,
    clip_path: Path = WALK_F,
    cyclic: bool = True,
) -> Path:
    existing = latest_checkpoint_for_label(label, "best")
    if existing is not None:
        print(f"reuse AE {existing}", flush=True)
        return existing
    torch.manual_seed(9753)
    cfg = ae_tr.SimpleAEConfig(latent_dim=latent_dim, hidden_dim=hidden_dim, train_steps=steps)
    locomotion_cfg = ae_tr.make_locomotion_cfg(device)
    clips = ae_tr.load_clips([(clip_path, cyclic)], locomotion_cfg)
    raw_features, _clip_ids, _cur_indices, schema = ae_tr.collect_controller_features(clips, locomotion_cfg, device)
    base_input_dim = int(schema["input_dim"])
    base_output_dim = int(schema["output_dim"])
    controller_input = raw_features[:, :base_input_dim].to(device)
    output = raw_features[:, base_input_dim : base_input_dim + base_output_dim].to(device)
    derived = foot_delta_feature(controller_input, output, schema, locomotion_cfg).detach().cpu()
    aug_features = torch.cat((raw_features, derived), dim=-1)
    aug_schema = dict(schema)
    feature_root = "current_root" if tl.output_reference_uses_current_root() else "future_root"
    aug_schema["feature"] = f"controller_input_plus_{feature_root}_transition_output_plus_foot_delta"
    aug_schema["base_total_dim"] = int(schema["total_dim"])
    aug_schema["total_dim"] = int(aug_features.shape[-1])
    aug_schema["foot_delta_start"] = int(schema["total_dim"])
    aug_schema["foot_delta_end"] = int(aug_features.shape[-1])
    aug_schema["foot_delta_dim"] = int(derived.shape[-1])
    aug_schema["foot_delta_weight"] = float(derived_weight)

    x_cpu, mean_cpu, std_cpu = ae_tr.normalize_features(aug_features, cfg.std_floor)
    x = x_cpu.to(device)
    mean = mean_cpu.to(device)
    std = std_cpu.to(device)
    out_start = int(aug_schema["target_output_start"])
    out_end = int(aug_schema["target_output_end"])
    d_start = int(aug_schema["foot_delta_start"])
    d_end = int(aug_schema["foot_delta_end"])
    model = ae_tr.SimpleAutoencoder(x.shape[1], cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    batch_size = min(int(cfg.batch_size), int(x.shape[0]))
    all_rows = torch.arange(x.shape[0], device=device)
    loss = torch.zeros((), device=device)
    for step in range(1, steps + 1):
        rows = ae_tr.batch_indices(all_rows, batch_size)
        batch = x.index_select(0, rows)
        recon = model(batch)
        out_loss = F.mse_loss(recon[:, out_start:out_end], batch[:, out_start:out_end])
        delta_loss = F.mse_loss(recon[:, d_start:d_end], batch[:, d_start:d_end])
        loss = out_loss + float(derived_weight) * delta_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    run_id = ik_run_id(label)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ae_tr.checkpoint_payload(
        model,
        optimizer,
        cfg,
        locomotion_cfg,
        aug_schema,
        mean,
        std,
        int(steps),
        float(loss.detach().cpu()),
        {
            "npz_paths": [str(clip_path)],
            "cyclic": bool(cyclic),
            "row_count": int(x.shape[0]),
            "audit": "foot_harshness",
            "variant": label,
            "ae_training_loss": "target_output_plus_foot_delta",
            "negative_examples": False,
            "temporal_window": 1,
            "contact_information": False,
            "derived_from_existing_controller_io": True,
        },
    )
    path = checkpoint_path(run_dir, run_id, "best")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    return path


def feature_weights(schema: dict[str, object], device: torch.device, profile: str) -> torch.Tensor:
    weights = torch.ones((int(schema["total_dim"]),), dtype=torch.float32, device=device)
    if not profile:
        return weights
    out = output_weights(schema, device, profile)
    out_start = int(schema["target_output_start"])
    weights[out_start : out_start + int(schema["output_dim"])] = out

    pose_dim = int(schema["pose_dim"])
    payload_offset = pose_dim - int(tl.IK_PAYLOAD_DIM)
    in_payload_weights = out[-int(tl.IK_PAYLOAD_DIM) :]
    weights[payload_offset:pose_dim] = in_payload_weights
    weights[pose_dim + payload_offset : 2 * pose_dim] = in_payload_weights
    return weights


def make_weighted_score_fn(
    schema: dict[str, object],
    device: torch.device,
    profile: str = "",
    topk_fraction: float = 0.0,
):
    out_weights = output_weights(schema, device, profile).reshape(1, -1) if profile else None
    topk_fraction = float(topk_fraction)

    def score_rows(
        ae: ae_tr.SimpleAutoencoder,
        mean: torch.Tensor,
        std: torch.Tensor,
        controller_input: torch.Tensor,
        predicted_output: torch.Tensor,
    ) -> torch.Tensor:
        input_dim = int(controller_input.shape[-1])
        feature = torch.cat((controller_input, predicted_output), dim=-1)
        x = (feature - mean) / std
        recon = ae(x)
        err = (recon[:, input_dim:] - x[:, input_dim:]).square()
        if out_weights is not None:
            err = err * out_weights.to(device=err.device, dtype=err.dtype)
        if topk_fraction > 0.0:
            k = max(1, min(err.shape[-1], int(round(err.shape[-1] * topk_fraction))))
            return torch.topk(err, k=k, dim=-1).values.mean(dim=-1)
        if out_weights is not None:
            return err.sum(dim=-1) / out_weights.sum().to(device=err.device, dtype=err.dtype).clamp_min(1.0)
        return err.mean(dim=-1)

    return score_rows


def train_weighted_autoencoder_variant(
    label: str,
    device: torch.device,
    latent_dim: int,
    hidden_dim: int,
    steps: int,
    profile: str,
) -> Path:
    existing = latest_checkpoint_for_label(label, "best")
    if existing is not None:
        print(f"reuse AE {existing}", flush=True)
        return existing
    torch.manual_seed(1357)
    cfg = ae_tr.SimpleAEConfig(latent_dim=latent_dim, hidden_dim=hidden_dim, train_steps=steps)
    locomotion_cfg = ae_tr.make_locomotion_cfg(device)
    clips = ae_tr.load_clips([(WALK_F, True)], locomotion_cfg)
    raw_features, _clip_ids, _cur_indices, schema = ae_tr.collect_controller_features(clips, locomotion_cfg, device)
    x_cpu, mean_cpu, std_cpu = ae_tr.normalize_features(raw_features, cfg.std_floor)
    x = x_cpu.to(device)
    mean = mean_cpu.to(device)
    std = std_cpu.to(device)
    weights = feature_weights(schema, device, profile).reshape(1, -1)
    model = ae_tr.SimpleAutoencoder(x.shape[1], cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    batch_size = min(int(cfg.batch_size), int(x.shape[0]))
    all_rows = torch.arange(x.shape[0], device=device)
    loss = torch.zeros((), device=device)
    for step in range(1, steps + 1):
        rows = ae_tr.batch_indices(all_rows, batch_size)
        batch = x.index_select(0, rows)
        loss = ((model(batch) - batch).square() * weights).sum() / weights.sum().clamp_min(1.0) / batch.shape[0]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    run_id = ik_run_id(label)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ae_tr.checkpoint_payload(
        model,
        optimizer,
        cfg,
        locomotion_cfg,
        schema,
        mean,
        std,
        int(steps),
        float(loss.detach().cpu()),
        {
            "npz_paths": [str(WALK_F)],
            "row_count": int(x.shape[0]),
            "audit": "foot_harshness",
            "variant": label,
            "weight_profile": profile,
            "negative_examples": False,
            "temporal_window": 1,
        },
    )
    path = checkpoint_path(run_dir, run_id, "best")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    return path


def train_foot_x3_autoencoder_variant(label: str, device: torch.device, latent_dim: int, hidden_dim: int, steps: int) -> Path:
    return train_weighted_autoencoder_variant(label, device, latent_dim, hidden_dim, steps, "foot_x3")


def train_foot_x6_autoencoder_variant(label: str, device: torch.device, latent_dim: int, hidden_dim: int, steps: int) -> Path:
    return train_weighted_autoencoder_variant(label, device, latent_dim, hidden_dim, steps, "foot_x6")


def train_corrupt_autoencoder_variant(label: str, device: torch.device, latent_dim: int, hidden_dim: int, steps: int) -> Path:
    existing = latest_checkpoint_for_label(label, "best")
    if existing is not None:
        print(f"reuse AE {existing}", flush=True)
        return existing
    torch.manual_seed(2468)
    cfg = ae_tr.SimpleAEConfig(latent_dim=latent_dim, hidden_dim=hidden_dim, train_steps=steps)
    locomotion_cfg = ae_tr.make_locomotion_cfg(device)
    clips = ae_tr.load_clips([(WALK_F, True)], locomotion_cfg)
    raw_features, clip_ids, cur_indices, schema = ae_tr.collect_controller_features(clips, locomotion_cfg, device)
    x_cpu, mean_cpu, std_cpu = ae_tr.normalize_features(raw_features, cfg.std_floor)
    raw = raw_features.to(device)
    x = x_cpu.to(device)
    mean = mean_cpu.to(device)
    std = std_cpu.to(device)
    pinned_masks = pinned_foot_masks(clips, clip_ids, cur_indices).to(device)
    model = ae_tr.SimpleAutoencoder(x.shape[1], cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    batch_size = min(int(cfg.batch_size), int(x.shape[0]))
    all_rows = torch.arange(x.shape[0], device=device)
    loss = torch.zeros((), device=device)
    for step in range(1, steps + 1):
        rows = ae_tr.batch_indices(all_rows, batch_size)
        clean = x.index_select(0, rows)
        raw_batch = raw.index_select(0, rows)
        mask_batch = pinned_masks.index_select(0, rows)
        corrupt_raw = corrupt_pinned_output(raw_batch, mask_batch, schema)
        corrupt = (corrupt_raw - mean) / std
        inp = torch.cat((clean, corrupt), dim=0)
        target = torch.cat((clean, clean), dim=0)
        loss = F.mse_loss(model(inp), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    run_id = ik_run_id(label)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ae_tr.checkpoint_payload(
        model,
        optimizer,
        cfg,
        locomotion_cfg,
        schema,
        mean,
        std,
        int(steps),
        float(loss.detach().cpu()),
        {
            "npz_paths": [str(WALK_F)],
            "row_count": int(x.shape[0]),
            "audit": "foot_harshness",
            "variant": label,
            "denoising": "pinned_foot_output_corruption",
            "pinned_rows": int(pinned_masks.any(dim=1).sum().item()),
        },
    )
    path = checkpoint_path(run_dir, run_id, "best")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    return path


def train_controller(
    label: str,
    ae_path: Path,
    device: torch.device,
    score_profile: str = "",
    score_topk_fraction: float = 0.0,
    stop_after_k: int | None = None,
    clip_path: Path = WALK_F,
    cyclic: bool = True,
) -> Path:
    existing = latest_checkpoint_for_label(label, "last")
    if existing is not None:
        print(f"reuse controller {existing}", flush=True)
        return existing
    torch.manual_seed(5678)
    ae, mean, std, ae_ckpt = ctl.load_simple_ae(ae_path, device)
    cfg = ctl.make_cfg(device, ae_ckpt)
    store = load_store(cfg, device, clip_path=clip_path, cyclic=cyclic)
    input_dim, output_dim = tl.make_batch_dims(store.prototype, cfg)
    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    optimizer = ctl.make_adamw(model.parameters(), ctl.LEARNING_RATE, device, capturable=bool(device.type == "cuda"))
    run_id = ik_run_id(label)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"), flush_secs=1)
    metadata = {
        "npz_paths": [str(clip_path)],
        "cyclic": bool(cyclic),
        "simple_ae_checkpoint": str(ae_path),
        "audit": "foot_harshness",
        "policy": {
            "loss": "simple_ae_output_reconstruction",
            "ae_score_output_only": True,
            "score_weight_profile": score_profile,
            "score_topk_fraction": float(score_topk_fraction),
            "stop_after_k": int(stop_after_k) if stop_after_k is not None else None,
            "pose_representation": tl.IK_POSE_REPRESENTATION,
            "output_reference_root": tl.OUTPUT_REFERENCE_ROOT,
            "output_prediction_mode": tl.normalized_output_prediction_mode(),
            "state_reference_root": tl.STATE_REFERENCE_ROOT,
        },
        "rollout_schedule": [int(k) for k in ctl.ROLLOUT_SCHEDULE],
        "rollout_stage_steps": [int(n) for n in ctl.ROLLOUT_STAGE_STEPS],
        "stage_learning_rates": {str(k): float(ctl.stage_learning_rate(k)) for k in ctl.ROLLOUT_SCHEDULE},
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
    }
    ctl.save_controller_checkpoint(run_dir, run_id, "init", model, optimizer, 0, float("inf"), 0, cfg, metadata)
    writer.add_text("config/json", f"```json\n{json.dumps({'config': asdict(cfg), 'metadata': metadata}, indent=2)}\n```", 0)
    writer.flush()
    ctl.refresh_tensorboard_async()
    step = 0
    last_loss = float("inf")
    t0 = time.perf_counter()
    original_score_rows = ctl.ae_score_rows
    ae_schema = dict(ae_ckpt["schema"])
    if str(ae_schema.get("feature", "")) in {
        "controller_input_plus_current_root_transition_output_plus_foot_delta",
        "controller_input_plus_future_root_transition_output_plus_foot_delta",
    }:
        ctl.ae_score_rows = make_foot_delta_score_fn(
            ae_schema,
            cfg,
            derived_weight=float(ae_schema.get("foot_delta_weight", 2.0)),
        )
    elif score_profile or float(score_topk_fraction) > 0.0:
        ctl.ae_score_rows = make_weighted_score_fn(
            ae_schema,
            device,
            profile=score_profile,
            topk_fraction=float(score_topk_fraction),
        )
    try:
        final = None
        final_k = int(ctl.ROLLOUT_K)
        for stage_idx, stage_k in enumerate(ctl.ROLLOUT_SCHEDULE):
            stage_steps = int(ctl.ROLLOUT_STAGE_STEPS[stage_idx])
            ctl.set_optimizer_lr(optimizer, ctl.stage_learning_rate(int(stage_k)))
            rollout_values = ctl.rollout_values_for(int(stage_k)) if ctl.mixed_rollout_enabled(int(stage_k)) else (int(stage_k),)
            start_pools = ctl.build_start_pools(store, rollout_values)
            max_pool = start_pools[int(stage_k)]
            batch_size = min(ctl.BATCH_SIZE, max_pool.row_count)
            stepper = ctl.make_pure_ae_stepper(model, optimizer, ae, mean, std, store, int(stage_k), batch_size, start_pools)
            print(f"{label}: stage K={stage_k} steps={stage_steps} batch={batch_size} stepper={stepper.kind}", flush=True)
            for stage_step in range(1, stage_steps + 1):
                step += 1
                loss = stepper.step()
                last_loss = float(loss.detach().cpu())
                if stage_step == 1 or stage_step % ctl.LOG_EVERY == 0 or stage_step == stage_steps:
                    writer.add_scalar("loss/train_ae", last_loss, step)
                    writer.add_scalar("curriculum/rollout_k", int(stage_k), step)
                    writer.add_scalar("time/elapsed_s", time.perf_counter() - t0, step)
                    writer.flush()
                    print(f"{label}: step={step} K={stage_k} loss={last_loss:.6g}", flush=True)
            ctl.save_controller_checkpoint(run_dir, run_id, f"stage_K{int(stage_k)}", model, optimizer, step, last_loss, int(stage_k), cfg, metadata)
            del stepper
            final = ctl.save_controller_checkpoint(run_dir, run_id, "last", model, optimizer, step, last_loss, int(stage_k), cfg, metadata)
            final_k = int(stage_k)
            if stop_after_k is not None and int(stage_k) >= int(stop_after_k):
                break
        assert final is not None
        metadata["stopped_after_k"] = final_k
        return final
    finally:
        ctl.ae_score_rows = original_score_rows
        writer.close()


def latest_checkpoint_for_label(label: str, tag: str) -> Path | None:
    matches = sorted(RUNS_DIR.glob(f"*_ik_{label}/checkpoints/*_{tag}.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def load_existing_results() -> dict[str, object]:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {}


def write_results(results: dict[str, object]) -> None:
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")


def run_ae_controller_variant(
    results: dict[str, object],
    key: str,
    label: str,
    device: torch.device,
    latent_dim: int,
    hidden_dim: int,
    steps: int = 12000,
    ae_trainer=train_autoencoder_variant,
    score_profile: str = "",
    score_topk_fraction: float = 0.0,
) -> None:
    if key in results:
        if not metric_is_current(results[key]):
            print(f"refresh metric {key}", flush=True)
            refresh_metric(results[key], "controller_checkpoint", device)
            write_results(results)
        print(f"reuse result {key}", flush=True)
        print(json.dumps(results[key], indent=2), flush=True)
        return
    ae_path = ae_trainer(f"{label}_ae", device, latent_dim=latent_dim, hidden_dim=hidden_dim, steps=steps)
    ctl_path = train_controller(
        f"{label}_controller",
        ae_path,
        device,
        score_profile=score_profile,
        score_topk_fraction=float(score_topk_fraction),
    )
    results[key] = {
        "ae_checkpoint": str(ae_path),
        "controller_checkpoint": str(ctl_path),
        "metric": evaluate_checkpoint(ctl_path, device),
        "metric_version": METRIC_VERSION,
    }
    print(json.dumps(results[key], indent=2), flush=True)
    write_results(results)


def main() -> None:
    device = make_device()
    results = load_existing_results()
    if "baseline_current" not in results:
        print("evaluating baseline", flush=True)
        results["baseline_current"] = {
            "checkpoint": str(BASE_CONTROLLER),
            "metric": evaluate_checkpoint(BASE_CONTROLLER, device),
            "metric_version": METRIC_VERSION,
        }
        write_results(results)
    elif not metric_is_current(results["baseline_current"]):
        print("refresh metric baseline_current", flush=True)
        refresh_metric(results["baseline_current"], "checkpoint", device)
        write_results(results)
    print(json.dumps(results["baseline_current"], indent=2), flush=True)

    print("option 1a: tighter AE", flush=True)
    run_ae_controller_variant(
        results,
        "option1_tight_ae",
        "foot_audit_opt1_tight",
        device,
        latent_dim=8,
        hidden_dim=256,
    )

    print("option 1b: wider AE", flush=True)
    run_ae_controller_variant(
        results,
        "option1_wide_ae",
        "foot_audit_opt1_wide",
        device,
        latent_dim=64,
        hidden_dim=768,
    )

    print("option 2: corrupted planted-foot AE", flush=True)
    run_ae_controller_variant(
        results,
        "option2_corrupt_ae",
        "foot_audit_opt2_corrupt",
        device,
        latent_dim=32,
        hidden_dim=512,
        ae_trainer=train_corrupt_autoencoder_variant,
    )

    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
