from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def kaggle_executable() -> str:
    current = PROJECT_ROOT / ".tools" / "kaggle_py312" / "Scripts" / "kaggle.exe"
    if current.exists():
        return str(current)
    bundled = PROJECT_ROOT / ".tools" / "python310" / "Scripts" / "kaggle.exe"
    return str(bundled if bundled.exists() else "kaggle")


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", help="Kaggle kernel id, for example madmad74100/stepper-k111-fork")
    parser.add_argument("--output-dir", default="training/runs/kaggle_sync")
    parser.add_argument("--interval-seconds", type=float, default=120.0)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    output_dir = (PROJECT_ROOT / args.output_dir / args.kernel.replace("/", "__")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    kaggle = kaggle_executable()

    while True:
        status = run([kaggle, "kernels", "status", args.kernel])
        print(status.stdout.strip(), flush=True)

        out = run([kaggle, "kernels", "output", args.kernel, "-p", str(output_dir), "-o", "-q"])
        print(out.stdout.strip(), flush=True)
        event_files = list(output_dir.rglob("events.out.tfevents*"))
        if event_files:
            print(f"TensorBoard logdir: {output_dir}", flush=True)
            print(f"Run locally: tensorboard --logdir \"{output_dir}\"", flush=True)
        else:
            print("No TensorBoard event files downloaded yet. Kaggle may only expose outputs after the kernel finishes.", flush=True)

        if not args.loop:
            break
        time.sleep(max(10.0, args.interval_seconds))


if __name__ == "__main__":
    main()
