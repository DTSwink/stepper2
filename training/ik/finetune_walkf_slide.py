from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

try:
    from .bootstrap import PROJECT_ROOT, ensure_paths
    from .naming import ik_run_id
    from . import excess_envelope as env
    from . import foot_harshness_audit as audit
    from . import ik_core as tl
    from . import train_full_ae_envelope as full_train
    from . import train_simple_ae_controller as ctl
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    from naming import ik_run_id
    import excess_envelope as env
    import foot_harshness_audit as audit
    import ik_core as tl
    import train_full_ae_envelope as full_train
    import train_simple_ae_controller as ctl


ensure_paths()


RUNS_DIR = PROJECT_ROOT / "training" / "runs"
WALK_F = PROJECT_ROOT / "ue5" / "animations_omni_only_full" / "npz_final" / "M_Neutral_Walk_Loop_F.npz"
TEMP_BASELINE = (
    RUNS_DIR
    / "20260522_163750_ik_full_vanilla_ae_controller_refined_stall"
    / "checkpoints"
    / "20260522_163750_ik_full_vanilla_ae_controller_refined_stall_init_from_checkpoint.pt"
)
FULL_AE = (
    RUNS_DIR
    / "20260522_073652_ik_full_vanilla_ae_all"
    / "checkpoints"
    / "20260522_073652_ik_full_vanilla_ae_all_best.pt"
)
WEIGHTS_JSON = RUNS_DIR / "cache" / "ik_excess_envelopes" / "latest_full_refinement_weights.json"
RUN_LABEL = "walkF_from_refined_init_slide_finetune"
TRAIN_STEPS = 1500
ROLLOUT_K = 32


def make_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    return device


def load_controller(path: Path, device: torch.device) -> tuple[torch.nn.Module, torch.optim.Optimizer, tl.TrainConfig, ctl.SimpleClipStore, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = tl.TrainConfig()
    ctl.apply_config_dict(cfg, ckpt["config"])
    cfg.device = str(device)
    cfg.use_torch_compile = False
    clip = tl.MotionClip(WALK_F, cfg, cyclic_animation=True)
    store = ctl.SimpleClipStore([clip], cfg, device)
    input_dim, output_dim = tl.make_batch_dims(store.prototype, cfg)
    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    model.load_state_dict(ckpt["model"])
    optimizer = ctl.make_adamw(model.parameters(), ctl.stage_learning_rate(ROLLOUT_K), device)
    if "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            ctl.set_optimizer_lr(optimizer, ctl.stage_learning_rate(ROLLOUT_K))
        except Exception as exc:
            print(f"optimizer state ignored for walk fine-tune: {exc}", flush=True)
    return model, optimizer, cfg, store, ckpt


def summarize_metric(metric: dict[str, object]) -> dict[str, object]:
    foot_pin = metric["foot_pin"]  # type: ignore[index]
    one_step = metric["one_step"]  # type: ignore[index]
    assert isinstance(foot_pin, dict)
    assert isinstance(one_step, dict)
    out: dict[str, object] = {
        "foot_pin_score": foot_pin["score"],
        "mean_slide_ratio": foot_pin["mean_slide_ratio"],
        "mean_rot_ratio": foot_pin["mean_rot_ratio"],
        "one_step_pos_mean_m": one_step["pos_mean_m"],
        "one_step_rot_mean_deg": one_step["rot_mean_deg"],
        "one_step_foot_pos_mean_m": one_step["foot_pos_mean_m"],
        "one_step_foot_rot_mean_deg": one_step["foot_rot_mean_deg"],
    }
    for foot in ("foot_l", "foot_r"):
        row = foot_pin[foot]
        assert isinstance(row, dict)
        out[f"{foot}_pred_slide_m_per_frame"] = row["pred_slide_m_per_frame"]
        out[f"{foot}_gt_slide_m_per_frame"] = row["gt_slide_m_per_frame"]
        out[f"{foot}_slide_ratio"] = row["slide_ratio"]
        out[f"{foot}_pred_rot_deg_per_frame"] = row["pred_rot_deg_per_frame"]
        out[f"{foot}_gt_rot_deg_per_frame"] = row["gt_rot_deg_per_frame"]
        out[f"{foot}_rot_ratio"] = row["rot_ratio"]
        out[f"{foot}_gt_interval"] = row["gt_interval"]
        out[f"{foot}_pred_interval"] = row["pred_interval"]
    return out


def evaluate(path: Path, device: torch.device) -> dict[str, object]:
    return summarize_metric(audit.evaluate_checkpoint(path, device))


def save_checkpoint(
    run_dir: Path,
    run_id: str,
    tag: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    cfg: tl.TrainConfig,
    metadata: dict,
) -> Path:
    path = run_dir / "checkpoints" / f"{run_id}_{tag}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tl.checkpoint_payload(model, optimizer, step, loss, ROLLOUT_K, cfg, metadata), path)
    return path


def main() -> None:
    if not TEMP_BASELINE.exists():
        raise FileNotFoundError(f"Temp baseline checkpoint not found: {TEMP_BASELINE}")
    if not FULL_AE.exists():
        raise FileNotFoundError(f"Full AE checkpoint not found: {FULL_AE}")
    if not WEIGHTS_JSON.exists():
        raise FileNotFoundError(f"Refinement weights not found: {WEIGHTS_JSON}")

    device = make_device()
    before = evaluate(TEMP_BASELINE, device)
    print("before", json.dumps(before, indent=2), flush=True)

    ae, mean, std, _ae_ckpt = ctl.load_simple_ae(FULL_AE, device)
    model, optimizer, cfg, store, source_ckpt = load_controller(TEMP_BASELINE, device)
    weights = json.loads(WEIGHTS_JSON.read_text(encoding="utf-8"))
    linear_weight = float(weights["linear_weight"])
    angular_weight = float(weights["angular_weight"])
    envelope = env.load_or_build_excess_envelope(store)
    sanity = env.groundtruth_sanity(store, envelope)
    if max(sanity.values()) > 1e-6:
        raise RuntimeError(f"Walk_F GT exceeds its own envelope: {sanity}")

    rollout_values = ctl.rollout_values_for(ROLLOUT_K)
    start_pools = ctl.build_start_pools(store, rollout_values)
    batch_size = min(int(ctl.BATCH_SIZE), int(start_pools[ROLLOUT_K].row_count))
    run_id = ik_run_id(RUN_LABEL)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "npz_paths": [str(WALK_F)],
        "source_checkpoint": str(TEMP_BASELINE),
        "simple_ae_checkpoint": str(FULL_AE),
        "weights_json": str(WEIGHTS_JSON),
        "linear_weight": linear_weight,
        "angular_weight": angular_weight,
        "train_steps": int(TRAIN_STEPS),
        "rollout_k": int(ROLLOUT_K),
        "batch_size": int(batch_size),
        "policy": "walkF_only_AE_plus_foot_envelope_finetune",
        "source_epoch": int(source_ckpt.get("epoch", 0)),
    }
    (run_dir / "config.json").write_text(json.dumps({"config": asdict(cfg), "metadata": metadata}, indent=2), encoding="utf-8")
    save_checkpoint(run_dir, run_id, "init", model, optimizer, 0, float("inf"), cfg, metadata)

    t0 = time.perf_counter()
    last_loss = float("inf")
    last_parts: dict[str, float] = {}
    model.train()
    for step in range(1, TRAIN_STEPS + 1):
        loss, parts = full_train.rollout_loss(
            model,
            ae,
            mean,
            std,
            store,
            ROLLOUT_K,
            batch_size,
            start_pools,
            envelope,  # type: ignore[arg-type]
            linear_weight,
            angular_weight,
            False,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        last_parts = {name: float(value.detach().cpu()) for name, value in parts.items()}
        if step == 1 or step % 100 == 0 or step == TRAIN_STEPS:
            print(
                f"step={step} loss={last_loss:.6g} ae={last_parts.get('ae', 0.0):.6g} "
                f"linear={last_parts.get('linear', 0.0):.6g} angular={last_parts.get('angular', 0.0):.6g} "
                f"elapsed_s={time.perf_counter() - t0:.1f}",
                flush=True,
            )
        if step % 500 == 0:
            save_checkpoint(run_dir, run_id, "latest", model, optimizer, step, last_loss, cfg, metadata)

    final_path = save_checkpoint(run_dir, run_id, "last", model, optimizer, TRAIN_STEPS, last_loss, cfg, metadata)
    after = evaluate(final_path, device)
    print("after", json.dumps(after, indent=2), flush=True)
    result = {
        "temp_baseline_checkpoint": str(TEMP_BASELINE),
        "finetuned_checkpoint": str(final_path),
        "before": before,
        "after": after,
        "last_train_loss": last_loss,
        "last_train_parts": last_parts,
    }
    out_path = run_dir / "walkf_slide_finetune_metrics.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"metrics_json={out_path}", flush=True)
    print(f"checkpoint={final_path}", flush=True)


if __name__ == "__main__":
    main()
