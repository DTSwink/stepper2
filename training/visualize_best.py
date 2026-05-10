from __future__ import annotations

# Fast default: visualize the most recently updated checkpoint_best.pt.
# Set run_name to a specific run folder if you do not want automatic latest-run detection.
run_name = "latest"
npz_path = "data/npz_final/testcasc.npz"
checkpoint_name = "checkpoint_best.pt"
output_path = "training/runs/model_comparisons/model_comparison.html"
device = "cuda"
open_browser = True

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def find_latest_run(runs_root: Path, checkpoint_file: str) -> Path:
    candidates = []
    for run_dir in runs_root.iterdir():
        ckpt = run_dir / "checkpoints" / checkpoint_file
        if ckpt.exists():
            candidates.append((ckpt.stat().st_mtime, run_dir))
    if not candidates:
        raise FileNotFoundError(f"No runs with checkpoints/{checkpoint_file} under {runs_root}")
    return max(candidates, key=lambda item: item[0])[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a run's best supervised checkpoint quickly.")
    parser.add_argument("--run-name", default=run_name, help='Run folder name, or "latest".')
    parser.add_argument("--npz-path", default=npz_path)
    parser.add_argument("--checkpoint-path", default=None, help="Explicit checkpoint path. Overrides --run-name.")
    parser.add_argument("--checkpoint-name", default=checkpoint_name)
    parser.add_argument("--output-path", default=output_path)
    parser.add_argument("--device", default=device)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    runs_root = resolve_path("training/runs")
    if args.checkpoint_path is not None:
        checkpoint_path = resolve_path(args.checkpoint_path)
        selected_run = checkpoint_path.parents[1].name if checkpoint_path.parent.name == "checkpoints" else "custom"
    else:
        if args.run_name == "latest":
            run_dir = find_latest_run(runs_root, args.checkpoint_name)
        else:
            run_dir = runs_root / args.run_name
        checkpoint_path = run_dir / "checkpoints" / args.checkpoint_name
        selected_run = run_dir.name

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    out_path = resolve_path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "training" / "visualize_model.py"),
        "--npz-path",
        str(resolve_path(args.npz_path)),
        "--checkpoint-path",
        str(checkpoint_path),
        "--output-path",
        str(out_path),
        "--device",
        args.device,
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    print(f"visualized_run={selected_run}")
    print(f"checkpoint={checkpoint_path}")
    print(f"output={out_path}")
    if open_browser and not args.no_open:
        webbrowser.open(out_path.as_uri())


if __name__ == "__main__":
    main()
