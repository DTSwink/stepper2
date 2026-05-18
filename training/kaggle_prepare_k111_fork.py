from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_DIR = PROJECT_ROOT / "training" / "kaggle_k111_fork"
DEFAULT_DATASET_SLUG = "stepper-k111-fork"
DEFAULT_KERNEL_SLUG = "stepper-k111-fork"


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "stepper-k111-fork"


def title_from_slug(slug: str) -> str:
    return slug.replace("-", " ")


def read_kaggle_username() -> str | None:
    token_path = Path.home() / ".kaggle" / "kaggle.json"
    if not token_path.exists():
        return None
    try:
        return json.loads(token_path.read_text(encoding="utf-8")).get("username")
    except Exception:
        return None


def safe_clean_dir(path: Path) -> None:
    path = path.resolve()
    root = PROJECT_ROOT.resolve()
    if root not in path.parents:
        raise ValueError(f"refusing to clean outside project: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_matching_files(src: Path, dst: Path, suffixes: tuple[str, ...]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.is_file() and path.name in {"requirements.txt", "README.md"}:
            shutil.copy2(path, dst / path.name)
        elif path.is_file() and path.suffix.lower() in suffixes:
            shutil.copy2(path, dst / path.name)


def copy_relative_file(src: Path, dst_root: Path) -> None:
    src = src.resolve()
    rel = src.relative_to(PROJECT_ROOT)
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def kaggle_payload_path(src: Path) -> str:
    rel = src.resolve().relative_to(PROJECT_ROOT)
    return str(PurePosixPath("/kaggle/working/stepper", *rel.parts))


def checkpoint_referenced_base_prior(path: Path) -> Path | None:
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    base = ckpt.get("base_prior_checkpoint")
    if not base:
        return None
    base_path = Path(str(base))
    if not base_path.exists():
        text = str(base).replace("\\", "/")
        for marker in ("/stepper/", "stepper/"):
            if marker in text:
                candidate = PROJECT_ROOT / text.split(marker, 1)[1]
                if candidate.exists():
                    return candidate.resolve()
    return base_path.resolve()


def write_notebook(
    path: Path,
    dataset_slug: str,
    run_name: str,
    rollout_schedule: str,
    initial_rollout_k: int,
    final_stage_random_rollout: bool,
    enable_tensorboard_tunnel: bool,
    env_vars: dict[str, str] | None = None,
) -> None:
    env_vars = env_vars or {}
    tb_tunnel_source = [
        "import os, re, stat, subprocess, sys, time, urllib.request\n",
        "from pathlib import Path\n",
        "logdir = Path('/kaggle/working/stepper/training/runs')\n",
        "logdir.mkdir(parents=True, exist_ok=True)\n",
        "tb_cmd = [sys.executable, '-m', 'tensorboard.main', '--logdir', str(logdir), '--host', '0.0.0.0', '--port', '6006']\n",
        "tb_proc = subprocess.Popen(tb_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)\n",
        "cloudflared = Path('/kaggle/working/cloudflared')\n",
        "if not cloudflared.exists():\n",
        "    urllib.request.urlretrieve('https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64', cloudflared)\n",
        "    cloudflared.chmod(cloudflared.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)\n",
        "tunnel_cmd = [str(cloudflared), 'tunnel', '--url', 'http://127.0.0.1:6006', '--no-autoupdate']\n",
        "tunnel_proc = subprocess.Popen(tunnel_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n",
        "deadline = time.time() + 90\n",
        "tb_url = None\n",
        "while time.time() < deadline:\n",
        "    line = tunnel_proc.stdout.readline() if tunnel_proc.stdout else ''\n",
        "    if line:\n",
        "        print('[cloudflared]', line.rstrip(), flush=True)\n",
        "        match = re.search(r'https://[^\\s]+\\.trycloudflare\\.com', line)\n",
        "        if match:\n",
        "            tb_url = match.group(0)\n",
        "            break\n",
        "    elif tunnel_proc.poll() is not None:\n",
        "        break\n",
        "if tb_url:\n",
        "    print('TENSORBOARD_URL=' + tb_url, flush=True)\n",
        "else:\n",
        "    print('TENSORBOARD_URL unavailable; use Kaggle logs/output after run finishes.', flush=True)\n",
    ]
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Stepper Kaggle fork\n",
                "\n",
                "This notebook unpacks the Stepper payload, starts TensorBoard, then resumes training from the selected checkpoint.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import os, shutil\n",
                f"DATASET_SLUG = {dataset_slug!r}\n",
                "input_root = Path('/kaggle/input')\n",
                "input_dir = input_root / DATASET_SLUG\n",
                "dst = Path('/kaggle/working/stepper')\n",
                "if dst.exists():\n",
                "    shutil.rmtree(dst)\n",
                "src = input_dir / 'stepper'\n",
                "zip_src = input_dir / 'stepper.zip'\n",
                "if not src.exists() and not zip_src.exists():\n",
                "    stepper_dirs = [p for p in input_root.rglob('stepper') if p.is_dir()]\n",
                "    stepper_zips = [p for p in input_root.rglob('stepper.zip') if p.is_file()]\n",
                "    if stepper_dirs:\n",
                "        src = stepper_dirs[0]\n",
                "    elif stepper_zips:\n",
                "        zip_src = stepper_zips[0]\n",
                "if src.exists():\n",
                "    shutil.copytree(src, dst)\n",
                "elif zip_src.exists():\n",
                "    shutil.unpack_archive(str(zip_src), '/kaggle/working')\n",
                "else:\n",
                "    visible = [str(p) for p in list(input_root.rglob('*'))[:80]]\n",
                "    raise FileNotFoundError(f'Could not find stepper/ or stepper.zip under {input_root}; visible={visible}')\n",
                "os.chdir(dst)\n",
                "print('workspace:', dst)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%load_ext tensorboard\n",
                "%tensorboard --logdir /kaggle/working/stepper/training/runs\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": tb_tunnel_source
            if enable_tensorboard_tunnel
            else ["print('TensorBoard tunnel disabled')\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, subprocess, sys\n",
                f"os.environ.setdefault('STEPPER_RUN_NAME', {run_name!r})\n",
                f"os.environ.setdefault('STEPPER_ROLLOUT_SCHEDULE', {rollout_schedule!r})\n",
                f"os.environ.setdefault('STEPPER_INITIAL_ROLLOUT_K', {str(initial_rollout_k)!r})\n",
                f"os.environ.setdefault('STEPPER_FINAL_STAGE_RANDOM_ROLLOUT', {'1' if final_stage_random_rollout else '0'!r})\n",
                *[
                    f"os.environ.setdefault({key!r}, {value!r})\n"
                    for key, value in sorted(env_vars.items())
                ],
                "subprocess.run([sys.executable, 'training/kaggle_run_k111.py'], check=True)\n",
            ],
        },
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=2), encoding="utf-8")


def run_cli(cmd: list[str]) -> None:
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def kaggle_executable() -> str:
    current_cli = PROJECT_ROOT / ".tools" / "kaggle_py312" / "Scripts" / "kaggle.exe"
    if current_cli.exists():
        return str(current_cli)
    bundled_legacy_cli = PROJECT_ROOT / ".tools" / "python310" / "Scripts" / "kaggle.exe"
    if bundled_legacy_cli.exists():
        return str(bundled_legacy_cli)
    return "kaggle"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--kaggle-username", default=None)
    parser.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG)
    parser.add_argument("--kernel-slug", default=DEFAULT_KERNEL_SLUG)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--upload", action="store_true", help="Create or version the Kaggle dataset after preparing it.")
    parser.add_argument("--push-kernel", action="store_true", help="Push the Kaggle notebook after preparing it.")
    parser.add_argument("--version-dataset", action="store_true", help="Use kaggle datasets version instead of create.")
    parser.add_argument("--accelerator", default="NvidiaTeslaT4", help="Kaggle accelerator machine shape.")
    parser.add_argument("--rollout-schedule", default="1,2,4,8,16,32,64,111")
    parser.add_argument("--initial-rollout-k", type=int, default=111)
    parser.add_argument("--no-final-stage-random-rollout", action="store_true")
    parser.add_argument("--no-tensorboard-tunnel", action="store_true")
    parser.add_argument(
        "--prior-checkpoint-path",
        default=str(
            PROJECT_ROOT
            / "training"
            / "runs"
            / "ae_poseaware_hybrid_datasetrefresh_20260515_175707"
            / "checkpoints"
            / "checkpoint_best.pt"
        ),
    )
    parser.add_argument(
        "--extra-prior-checkpoint-path",
        action="append",
        default=[],
        help="Additional AE prior checkpoint to copy into the Kaggle payload and pass to training.",
    )
    parser.add_argument(
        "--extra-prior-weight",
        action="append",
        default=[],
        help="Weight for each --extra-prior-checkpoint-path, in the same order.",
    )
    parser.add_argument(
        "--resume-checkpoint-path",
        default=str(
            PROJECT_ROOT
            / "training"
            / "runs"
            / "hybrid_datasetrefresh_packed_scheduleK_from_epoch2040_w010_20260515_180333"
            / "checkpoints"
            / "checkpoint_best_k64.pt"
        ),
    )
    parser.add_argument("--footslide-weight", default="0.10")
    parser.add_argument("--footslide-threshold-mps", default=None)
    parser.add_argument("--learning-rate", default="1e-6")
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--kaggle-periodic-folder-path", default=None)
    parser.add_argument("--kaggle-nonperiodic-folder-path", default=None)
    args = parser.parse_args()

    username = args.kaggle_username or read_kaggle_username()
    if not username:
        username = "KAGGLE_USERNAME_HERE"
    dataset_slug = slugify(args.dataset_slug)
    kernel_slug = slugify(args.kernel_slug)
    run_name = args.run_name or f"kaggle_k111_from_best_k64_{time.strftime('%Y%m%d_%H%M%S')}"

    work_dir = Path(args.work_dir).resolve()
    dataset_dir = work_dir / "dataset"
    kernel_dir = work_dir / "kernel"
    safe_clean_dir(work_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir.mkdir(parents=True, exist_ok=True)

    payload_root = dataset_dir / "stepper"
    payload_root.mkdir(parents=True, exist_ok=True)

    for name in ["README.md", "requirements.txt"]:
        src = PROJECT_ROOT / name
        if src.exists():
            shutil.copy2(src, payload_root / name)
    copy_matching_files(PROJECT_ROOT / "training", payload_root / "training", (".py", ".md", ".txt"))
    copy_matching_files(PROJECT_ROOT / "fbx_npz_pipeline", payload_root / "fbx_npz_pipeline", (".py", ".md", ".txt"))

    for rel_dir in [
        Path("ue5") / "animations_omni_only_full" / "npz_final",
        Path("ue5") / "animations_transitions_only_full_trimmed" / "npz_final",
        Path("ue5") / "animations_transitions_only_full_trimmed_turn_in_place" / "npz_final",
    ]:
        src = PROJECT_ROOT / rel_dir
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copytree(src, payload_root / rel_dir)

    prior_checkpoint = Path(args.prior_checkpoint_path).resolve()
    resume_checkpoint = Path(args.resume_checkpoint_path).resolve()
    extra_prior_checkpoints = [Path(path).resolve() for path in args.extra_prior_checkpoint_path]
    if args.extra_prior_weight and len(args.extra_prior_weight) > len(extra_prior_checkpoints):
        raise ValueError("--extra-prior-weight was provided more times than --extra-prior-checkpoint-path")
    required_files = [prior_checkpoint, resume_checkpoint, *extra_prior_checkpoints]
    base_prior = checkpoint_referenced_base_prior(prior_checkpoint)
    if base_prior is not None:
        required_files.append(base_prior)
    for extra_prior in extra_prior_checkpoints:
        base_prior = checkpoint_referenced_base_prior(extra_prior)
        if base_prior is not None:
            required_files.append(base_prior)
    for src in required_files:
        if not src.exists():
            raise FileNotFoundError(src)
        copy_relative_file(src, payload_root)

    dataset_id = f"{username}/{dataset_slug}"
    kernel_id = f"{username}/{kernel_slug}"
    dataset_meta = {
        "title": "Stepper K111 Fork Payload",
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps(dataset_meta, indent=2), encoding="utf-8")
    (dataset_dir / "datasets-metadata.json").write_text(json.dumps(dataset_meta, indent=2), encoding="utf-8")

    notebook_path = kernel_dir / "stepper_k111_fork.ipynb"
    env_vars = {
        "STEPPER_FOOTSLIDE_WEIGHT": str(args.footslide_weight),
        "STEPPER_LEARNING_RATE": str(args.learning_rate),
        "STEPPER_PRIOR_CHECKPOINT": kaggle_payload_path(prior_checkpoint),
        "STEPPER_RESUME_CHECKPOINT": kaggle_payload_path(resume_checkpoint),
        "STEPPER_RESUME_OPTIMIZER": "1" if args.resume_optimizer else "0",
    }
    if args.footslide_threshold_mps is not None:
        env_vars["STEPPER_FOOTSLIDE_THRESHOLD_MPS"] = str(args.footslide_threshold_mps)
    if extra_prior_checkpoints:
        env_vars["STEPPER_EXTRA_PRIOR_CHECKPOINTS"] = ";".join(
            kaggle_payload_path(path) for path in extra_prior_checkpoints
        )
        env_vars["STEPPER_EXTRA_PRIOR_WEIGHTS"] = ";".join(str(weight) for weight in args.extra_prior_weight)
    if args.kaggle_periodic_folder_path is not None:
        env_vars["STEPPER_PERIODIC_FOLDER_PATH"] = str(args.kaggle_periodic_folder_path)
    if args.kaggle_nonperiodic_folder_path is not None:
        env_vars["STEPPER_NONPERIODIC_FOLDER_PATH"] = str(args.kaggle_nonperiodic_folder_path)

    write_notebook(
        notebook_path,
        dataset_slug,
        run_name,
        args.rollout_schedule,
        args.initial_rollout_k,
        not args.no_final_stage_random_rollout,
        not args.no_tensorboard_tunnel,
        env_vars,
    )
    kernel_meta = {
        "id": kernel_id,
        "title": title_from_slug(kernel_slug),
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": args.accelerator,
        "dataset_sources": [dataset_id],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(kernel_meta, indent=2), encoding="utf-8")

    print(f"prepared Kaggle dataset payload: {dataset_dir}", flush=True)
    print(f"prepared Kaggle notebook:       {kernel_dir}", flush=True)
    print(f"dataset id: {dataset_id}", flush=True)
    print(f"kernel id:  {kernel_id}", flush=True)

    kaggle = kaggle_executable()
    if args.upload:
        if args.version_dataset:
            run_cli([kaggle, "datasets", "version", "-p", str(dataset_dir), "-m", "refresh K111 fork payload", "-r", "zip"])
        else:
            run_cli([kaggle, "datasets", "create", "-p", str(dataset_dir), "-r", "zip"])
    if args.push_kernel:
        run_cli([kaggle, "kernels", "push", "-p", str(kernel_dir), "--accelerator", args.accelerator])


if __name__ == "__main__":
    main()
