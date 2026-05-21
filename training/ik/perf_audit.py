from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

try:
    from .bootstrap import PROJECT_ROOT, ensure_paths
    from . import ik_core as tl
    from . import train_ae_prior as ae_prior
    from .train import (
        BATCH_SIZE,
        DEFAULT_WALK_F,
        ROLLOUT_K,
        build_start_pools,
        mixed_rollout_enabled,
        load_clips,
        make_adamw,
        make_cfg,
        make_supervised_stepper,
        resolve_clip_specs,
        rollout_values_for,
        start_pool_summary,
    )
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    import ik_core as tl
    import train_ae_prior as ae_prior
    from train import (
        BATCH_SIZE,
        DEFAULT_WALK_F,
        ROLLOUT_K,
        build_start_pools,
        mixed_rollout_enabled,
        load_clips,
        make_adamw,
        make_cfg,
        make_supervised_stepper,
        resolve_clip_specs,
        rollout_values_for,
        start_pool_summary,
    )

ensure_paths()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench(label: str, fn, device: torch.device, iters: int, warmup: int = 2) -> dict[str, float | str]:
    for _ in range(max(0, warmup)):
        fn()
    sync(device)
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - start)
    return {
        "label": label,
        "iters": iters,
        "warmup": warmup,
        "mean_ms": 1000.0 * sum(times) / max(1, len(times)),
        "min_ms": 1000.0 * min(times),
        "max_ms": 1000.0 * max(times),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Performance audit for contained IK training paths.")
    parser.add_argument("--npz", default=str(DEFAULT_WALK_F))
    parser.add_argument("--periodic-folder", default=None)
    parser.add_argument("--nonperiodic-folder", default=None)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    cfg = make_cfg(device)
    cfg.batch_size = int(BATCH_SIZE)
    clip_specs = resolve_clip_specs(args.npz, args.periodic_folder, args.nonperiodic_folder)
    clips = load_clips(clip_specs, cfg)
    input_dim, output_dim = tl.make_batch_dims(clips[0], cfg)
    model = tl.MLPController(input_dim, output_dim, cfg).to(device)
    optimizer = make_adamw(model.parameters(), cfg.learning_rate, device, capturable=(device.type == "cuda"))
    store = ae_prior.ClipStore(clips, cfg, device)
    rollout_k = int(ROLLOUT_K)
    rollout_values = rollout_values_for(rollout_k) if mixed_rollout_enabled(rollout_k) else (int(rollout_k),)
    start_pools = build_start_pools(
        store,
        rollout_values,
        require_all_clips=not mixed_rollout_enabled(rollout_k),
    )
    max_pool_clip_ids, max_pool_starts = start_pools[int(rollout_k)]
    row_count = int(max_pool_starts.sum().detach().cpu())
    batch_size = min(int(BATCH_SIZE), row_count)
    stepper = make_supervised_stepper(model, optimizer, store, cfg, rollout_k, batch_size, start_pools)

    def supervised_rollout_step() -> None:
        stepper.step()

    rows = [
        bench("supervised_rollout_optimizer_step", supervised_rollout_step, device, args.iters),
    ]
    report = {
        "clip_count": len(clips),
        "npz_paths": [str(path) for path, _cyclic in clip_specs[:8]],
        "npz_path_note": "truncated" if len(clip_specs) > 8 else "complete",
        "device": str(device),
        "rows": int(row_count),
        "eligible_clip_count": int(max_pool_clip_ids.numel()),
        "batch_size": batch_size,
        "rollout_k": int(rollout_k),
        "mixed_rollout": bool(mixed_rollout_enabled(rollout_k)),
        "training_step": stepper.kind,
        "rollout_values": [int(k) for k in rollout_values],
        "start_pools": start_pool_summary(start_pools),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "benchmarks": rows,
        "policy": {
            "gpu_resident_rollout": "always",
            "random_clip_per_row": "always",
            "ik_run_name": "YYYYMMDD_HHMMSS_ik_<label>",
        },
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
