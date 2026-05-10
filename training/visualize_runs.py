from __future__ import annotations

# Build one comparison HTML containing multiple checkpoint rollouts.
npz_path = "data/npz_final/testcasc.npz"
output_path = "training/runs/model_comparisons/model_comparison.html"
checkpoint_name = "checkpoint_best.pt"

import argparse
from pathlib import Path

import torch

import train_locomotion as tl
import visualize_model as vm


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def runs_root() -> Path:
    return resolve_path("training/runs")


def find_latest_runs(limit: int, checkpoint_file: str) -> list[Path]:
    candidates: list[tuple[float, Path]] = []
    for run_dir in runs_root().iterdir():
        ckpt = run_dir / "checkpoints" / checkpoint_file
        if ckpt.exists():
            candidates.append((ckpt.stat().st_mtime, run_dir))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [run for _, run in candidates[:limit]]


def parse_run_names(text: str | None, recent: int, checkpoint_file: str) -> list[Path]:
    if recent > 0:
        return find_latest_runs(recent, checkpoint_file)
    if text is None or text.strip() == "latest":
        return find_latest_runs(1, checkpoint_file)
    result = []
    for name in text.split(","):
        name = name.strip()
        if not name:
            continue
        result.append(runs_root() / name)
    return result


def checkpoint_for_run(run_dir: Path, checkpoint_file: str) -> Path:
    checkpoint = run_dir / "checkpoints" / checkpoint_file
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def payload_for_checkpoint(npz: Path, checkpoint: Path, device: torch.device, max_frames: int | None) -> tuple[dict, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = tl.TrainConfig()
    vm.apply_config_dict(cfg, ckpt.get("config", {}))
    cfg.device = str(device)
    cfg.use_torch_compile = False

    clip = tl.MotionClip(npz, cfg)
    model = vm.load_model(ckpt, clip, cfg, device)
    gt_pos, gt_rot, pred_tf_pos, pred_tf_rot, error_tf, pred_ar_pos, pred_ar_rot, error_ar = vm.rollout_model(
        model, clip, cfg, device, max_frames
    )
    payload = vm.make_payload(clip, gt_pos, gt_rot, pred_tf_pos, pred_tf_rot, error_tf, pred_ar_pos, pred_ar_rot, error_ar)
    payload["title"] = f"{npz.stem} vs {checkpoint.parent.parent.name}"
    metrics = {
        "one_step_avg": float(error_tf.mean()),
        "one_step_max": float(error_tf.max()),
        "autoregressive_avg": float(error_ar.mean()),
        "autoregressive_max": float(error_ar.max()),
        "epoch": ckpt.get("epoch"),
        "rollout_k": ckpt.get("rollout_k"),
        "best_val": ckpt.get("best_val"),
    }
    return payload, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize multiple model checkpoints in one selectable HTML viewer.")
    parser.add_argument("--npz-path", default=npz_path)
    parser.add_argument("--run-names", default="latest", help='Comma-separated run folder names, or "latest".')
    parser.add_argument("--include-recent", type=int, default=0, help="Use N most recent runs with the checkpoint.")
    parser.add_argument("--checkpoint-name", default=checkpoint_name)
    parser.add_argument("--output-path", default=output_path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    npz = resolve_path(args.npz_path)
    output = resolve_path(args.output_path)
    device = torch.device(args.device)
    run_dirs = parse_run_names(args.run_names, args.include_recent, args.checkpoint_name)
    if not run_dirs:
        raise ValueError("No runs selected.")

    payloads = []
    for run_dir in run_dirs:
        checkpoint = checkpoint_for_run(run_dir, args.checkpoint_name)
        payload, metrics = payload_for_checkpoint(npz, checkpoint, device, args.max_frames)
        payloads.append(payload)
        print(
            f"{run_dir.name}: epoch={metrics['epoch']} K={metrics['rollout_k']} "
            f"best={metrics['best_val']:.6f} ar_avg={metrics['autoregressive_avg']:.6f} "
            f"ar_max={metrics['autoregressive_max']:.6f}"
        )

    title = f"{npz.stem} model comparisons"
    vm.write_html(payloads, output, title)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
