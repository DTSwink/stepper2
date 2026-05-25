from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

try:
    from .bootstrap import PROJECT_ROOT, ensure_paths
    from . import visualize
except ImportError:
    from bootstrap import PROJECT_ROOT, ensure_paths
    import visualize

ensure_paths()


DEFAULT_OUTPUT = PROJECT_ROOT / "training" / "runs" / "model_comparisons" / "model_comparison.html"


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def checkpoint_signature(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))


def wait_until_stable(path: Path, seconds: float = 0.5) -> tuple[str, int, int]:
    previous = checkpoint_signature(path)
    deadline = time.perf_counter() + max(0.0, float(seconds))
    while time.perf_counter() < deadline:
        time.sleep(0.1)
        current = checkpoint_signature(path)
        if current != previous:
            previous = current
            deadline = time.perf_counter() + max(0.0, float(seconds))
    return previous


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuously regenerate an HTML model viewer from the newest checkpoint.")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--npz-path", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--checkpoint-name-contains",
        default="",
        help="Optional case-insensitive substring filter for the checkpoint path/run name.",
    )
    args = parser.parse_args()

    output = resolve_path(args.output_path)
    npz = resolve_path(args.npz_path) if str(args.npz_path).strip() else None
    device = torch.device(args.device)
    last_signature: tuple[str, int, int] | None = None

    print(f"watch_latest_viewer output={output}", flush=True)
    while True:
        try:
            checkpoint = visualize.find_latest_checkpoint(args.checkpoint_name_contains)
            signature = wait_until_stable(checkpoint)
            if signature != last_signature:
                info = visualize.render_checkpoint_to_html(
                    checkpoint=checkpoint,
                    output=output,
                    npz=npz,
                    device=device,
                    max_frames=args.max_frames,
                )
                print(
                    "rendered "
                    f"checkpoint={info['checkpoint']} "
                    f"root={info.get('output_reference_root', '?')} "
                    f"prediction={info.get('output_prediction_mode', '?')} "
                    f"one_step_max_m={info['one_step_max']:.6f} "
                    f"autoregressive_max_m={info['autoregressive_max']:.6f}",
                    flush=True,
                )
                last_signature = signature
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"viewer refresh skipped: {exc}", flush=True)
        time.sleep(max(0.5, float(args.poll_seconds)))


if __name__ == "__main__":
    main()
