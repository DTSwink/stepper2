from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IK_DIR = PROJECT_ROOT / "training" / "ik"
DEFAULT_WORK_DIR = IK_DIR / "kaggle_payload"
DEFAULT_PERIODIC = PROJECT_ROOT / "ue5" / "animations_omni_only_full" / "npz_final"
DEFAULT_NONPERIODIC = PROJECT_ROOT / "ue5" / "animations_transitions_only_full_trimmed" / "npz_final"
DEFAULT_DATASET_SLUG = "stepper-ik-payload"
DEFAULT_KERNEL_SLUG = "stepper-ik-trainer"


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "stepper-ik"


def read_kaggle_username() -> str | None:
    token = Path.home() / ".kaggle" / "kaggle.json"
    if not token.exists():
        return None
    try:
        return json.loads(token.read_text(encoding="utf-8")).get("username")
    except Exception:
        return None


def safe_clean_dir(path: Path) -> None:
    path = path.resolve()
    root = PROJECT_ROOT.resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"refusing to clean outside project: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ignore_payload(_dir: str, names: list[str]) -> set[str]:
    ignored = {"__pycache__", ".pytest_cache"}
    ignored.update(name for name in names if name.startswith("kaggle_payload"))
    ignored.update(name for name in names if name.startswith("kaggle_results"))
    ignored.update(name for name in names if name.startswith("kaggle_output_mirror"))
    ignored.update(name for name in names if name.endswith((".pyc", ".pyo")))
    return ignored


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copytree(src, dst, ignore=ignore_payload)


def copy_file_to_payload(src: Path, payload_root: Path) -> str:
    src = src.resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    try:
        rel = src.relative_to(PROJECT_ROOT)
    except ValueError:
        digest = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:10]
        rel = Path("kaggle_external") / digest / src.name
    dst = payload_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(PurePosixPath("/kaggle/working/stepper", *rel.parts))


def copy_dataset(src: Path, payload_root: Path) -> str:
    src = src.resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    rel = src.relative_to(PROJECT_ROOT)
    copy_tree(src, payload_root / rel)
    return str(PurePosixPath("/kaggle/working/stepper", *rel.parts))


def enabled_text(text: str) -> bool:
    return text.strip().lower() not in {"", "none", "null", "__empty__"}


def kaggle_executable() -> str:
    current = PROJECT_ROOT / ".tools" / "kaggle_py312" / "Scripts" / "kaggle.exe"
    if current.exists():
        return str(current)
    bundled = PROJECT_ROOT / ".tools" / "python310" / "Scripts" / "kaggle.exe"
    return str(bundled if bundled.exists() else "kaggle")


def write_notebook(
    path: Path,
    dataset_slug: str,
    env_vars: dict[str, str],
    enable_tensorboard_tunnel: bool,
) -> None:
    tunnel_source = [
        "import os, re, stat, subprocess, sys, time, urllib.request\n",
        "from pathlib import Path\n",
        "logdir = Path('/kaggle/working/stepper/training/runs')\n",
        "logdir.mkdir(parents=True, exist_ok=True)\n",
        "tb_proc = subprocess.Popen([sys.executable, '-m', 'tensorboard.main', '--logdir', str(logdir), '--host', '0.0.0.0', '--port', '6006'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)\n",
        "cloudflared = Path('/kaggle/working/cloudflared')\n",
        "if not cloudflared.exists():\n",
        "    urllib.request.urlretrieve('https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64', cloudflared)\n",
        "    cloudflared.chmod(cloudflared.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)\n",
        "proc = subprocess.Popen([str(cloudflared), 'tunnel', '--url', 'http://127.0.0.1:6006', '--no-autoupdate'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n",
        "deadline = time.time() + 90\n",
        "url = None\n",
        "while time.time() < deadline:\n",
        "    line = proc.stdout.readline() if proc.stdout else ''\n",
        "    if line:\n",
        "        print('[cloudflared]', line.rstrip(), flush=True)\n",
        "        match = re.search(r'https://[^\\s]+\\.trycloudflare\\.com', line)\n",
        "        if match:\n",
        "            url = match.group(0)\n",
        "            break\n",
        "    elif proc.poll() is not None:\n",
        "        break\n",
        "print('TENSORBOARD_URL=' + url if url else 'TENSORBOARD_URL unavailable', flush=True)\n",
    ]
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Stepper IK Kaggle trainer\n"],
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
                "dst = Path('/kaggle/working/stepper')\n",
                "if dst.exists():\n",
                "    shutil.rmtree(dst)\n",
                "src = input_root / DATASET_SLUG / 'stepper'\n",
                "zip_src = input_root / DATASET_SLUG / 'stepper.zip'\n",
                "if src.exists():\n",
                "    shutil.copytree(src, dst)\n",
                "elif zip_src.exists():\n",
                "    shutil.unpack_archive(str(zip_src), '/kaggle/working')\n",
                "else:\n",
                "    candidates = [p for p in input_root.rglob('stepper') if p.is_dir()]\n",
                "    if candidates:\n",
                "        shutil.copytree(candidates[0], dst)\n",
                "    else:\n",
                "        raise FileNotFoundError('Could not find stepper payload under /kaggle/input')\n",
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
                "from pathlib import Path\n",
                "import subprocess, sys\n",
                "if Path('requirements.txt').exists():\n",
                "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=False)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["%load_ext tensorboard\n", "%tensorboard --logdir /kaggle/working/stepper/training/runs\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": tunnel_source if enable_tensorboard_tunnel else ["print('TensorBoard tunnel disabled')\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, subprocess, sys\n",
                *[f"os.environ[{key!r}] = {value!r}\n" for key, value in sorted(env_vars.items())],
                "subprocess.run([sys.executable, 'training/ik/kaggle_run.py'], check=True)\n",
            ],
        },
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def write_script(
    path: Path,
    dataset_slug: str,
    env_vars: dict[str, str],
    enable_tensorboard_tunnel: bool,
) -> None:
    tunnel_source = """
import re
import stat
import time
import urllib.request

logdir = Path('/kaggle/working/stepper/training/runs')
logdir.mkdir(parents=True, exist_ok=True)
tb_proc = subprocess.Popen([sys.executable, '-m', 'tensorboard.main', '--logdir', str(logdir), '--host', '0.0.0.0', '--port', '6006'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
cloudflared = Path('/kaggle/working/cloudflared')
if not cloudflared.exists():
    print('downloading cloudflared', flush=True)
    urllib.request.urlretrieve('https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64', cloudflared)
    cloudflared.chmod(cloudflared.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
proc = subprocess.Popen([str(cloudflared), 'tunnel', '--url', 'http://127.0.0.1:6006', '--no-autoupdate'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
deadline = time.time() + 90
url = None
while time.time() < deadline:
    line = proc.stdout.readline() if proc.stdout else ''
    if line:
        print('[cloudflared]', line.rstrip(), flush=True)
        match = re.search(r'https://[^\\s]+\\.trycloudflare\\.com', line)
        if match:
            url = match.group(0)
            break
    elif proc.poll() is not None:
        break
print('TENSORBOARD_URL=' + url if url else 'TENSORBOARD_URL unavailable', flush=True)
"""
    env_lines = "\n".join(f"os.environ[{key!r}] = {value!r}" for key, value in sorted(env_vars.items()))
    tensorboard_block = tunnel_source if enable_tensorboard_tunnel else "print('TensorBoard tunnel disabled', flush=True)"
    script = f"""from pathlib import Path
import os
import shutil
import subprocess
import sys

print('stepper IK Kaggle script start', flush=True)
DATASET_SLUG = {dataset_slug!r}
input_root = Path('/kaggle/input')
dst = Path('/kaggle/working/stepper')
if dst.exists():
    print('removing old workspace', dst, flush=True)
    shutil.rmtree(dst)
src = input_root / DATASET_SLUG / 'stepper'
zip_src = input_root / DATASET_SLUG / 'stepper.zip'
if src.exists():
    print('copying workspace from', src, flush=True)
    shutil.copytree(src, dst)
elif zip_src.exists():
    print('unpacking workspace from', zip_src, flush=True)
    shutil.unpack_archive(str(zip_src), '/kaggle/working')
else:
    candidates = [p for p in input_root.rglob('stepper') if p.is_dir()]
    if candidates:
        print('copying workspace from fallback', candidates[0], flush=True)
        shutil.copytree(candidates[0], dst)
    else:
        raise FileNotFoundError('Could not find stepper payload under /kaggle/input')
os.chdir(dst)
print('workspace:', dst, flush=True)

if Path('requirements.txt').exists():
    print('installing requirements', flush=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=False)

{tensorboard_block}

{env_lines}
print('running training/ik/kaggle_run.py', flush=True)
subprocess.run([sys.executable, '-u', 'training/ik/kaggle_run.py'], check=True)
print('stepper IK Kaggle script done', flush=True)
"""
    path.write_text(script, encoding="utf-8")


def run_cli(cmd: list[str]) -> None:
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an IK-only Kaggle dataset/kernel payload.")
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--kaggle-username", default=None)
    parser.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG)
    parser.add_argument("--kernel-slug", default=DEFAULT_KERNEL_SLUG)
    parser.add_argument("--mode", choices=("supervised", "simple_ae", "ae_envelope", "ae_research"), default="supervised")
    parser.add_argument("--run-label", default="kaggle_full_supervised")
    parser.add_argument("--ae-label", default="")
    parser.add_argument("--baseline-label", default="")
    parser.add_argument("--refined-label", default="")
    parser.add_argument("--final-label", default="")
    parser.add_argument("--periodic-folder", default=str(DEFAULT_PERIODIC))
    parser.add_argument("--nonperiodic-folder", default=str(DEFAULT_NONPERIODIC))
    parser.add_argument("--npz", default="")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--ae-checkpoint", default="")
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--refined-checkpoint", default="")
    parser.add_argument("--weights-json", default="")
    parser.add_argument("--phase", default="all")
    parser.add_argument("--pose-noise", default="")
    parser.add_argument("--train-steps", default="100000")
    parser.add_argument("--ae-research-controller-steps", default="3000")
    parser.add_argument("--ae-research-max-variants", default="")
    parser.add_argument("--ae-research-variant-names", default="")
    parser.add_argument("--ae-research-eval-rows", default="")
    parser.add_argument("--ae-research-periodic-names", default="")
    parser.add_argument("--ae-research-nonperiodic-names", default="")
    parser.add_argument("--ae-research-controller-periodic-names", default="")
    parser.add_argument("--ae-research-controller-nonperiodic-names", default="")
    parser.add_argument("--ae-research-controller-pose-noise", default="")
    parser.add_argument("--ae-research-window-frames", default="")
    parser.add_argument("--ae-research-skip-one-frame-baseline", action="store_true")
    parser.add_argument("--load-optimizer", action="store_true")
    parser.add_argument("--allow-cuda-graph", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--version-dataset", action="store_true")
    parser.add_argument("--push-kernel", action="store_true")
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--no-tensorboard-tunnel", action="store_true")
    parser.add_argument("--script-kernel", action="store_true")
    args = parser.parse_args()

    username = args.kaggle_username or read_kaggle_username() or "KAGGLE_USERNAME_HERE"
    dataset_slug = slugify(args.dataset_slug)
    kernel_slug = slugify(args.kernel_slug)
    dataset_id = f"{username}/{dataset_slug}"
    kernel_id = f"{username}/{kernel_slug}"

    work_dir = Path(args.work_dir).resolve()
    dataset_dir = work_dir / "dataset"
    kernel_dir = work_dir / "kernel"
    safe_clean_dir(work_dir)
    payload_root = dataset_dir / "stepper"
    payload_root.mkdir(parents=True, exist_ok=True)
    kernel_dir.mkdir(parents=True, exist_ok=True)

    for name in ("README.md", "requirements.txt"):
        src = PROJECT_ROOT / name
        if src.exists():
            shutil.copy2(src, payload_root / name)
    training_requirements = PROJECT_ROOT / "training" / "requirements.txt"
    if training_requirements.exists():
        dst_requirements = payload_root / "training" / "requirements.txt"
        dst_requirements.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(training_requirements, dst_requirements)
    copy_tree(IK_DIR, payload_root / "training" / "ik")
    copy_tree(PROJECT_ROOT / "fbx_npz_pipeline", payload_root / "fbx_npz_pipeline")

    env_vars = {
        "STEPPER_IK_MODE": args.mode,
        "STEPPER_RUN_LABEL": args.run_label,
        "STEPPER_TRAIN_STEPS": str(args.train_steps),
        "STEPPER_LOAD_OPTIMIZER": "1" if args.load_optimizer else "0",
        "STEPPER_RESUME_STEP_FROM_CHECKPOINT": "1",
        "STEPPER_DISABLE_CUDA_GRAPH": "0" if args.allow_cuda_graph else "1",
    }
    if args.ae_label:
        env_vars["STEPPER_AE_LABEL"] = args.ae_label
    if args.baseline_label:
        env_vars["STEPPER_BASELINE_LABEL"] = args.baseline_label
    if args.refined_label:
        env_vars["STEPPER_REFINED_LABEL"] = args.refined_label
    elif args.mode == "ae_envelope" and str(args.phase).lower() == "refine":
        env_vars["STEPPER_REFINED_LABEL"] = args.run_label
    if args.final_label:
        env_vars["STEPPER_FINAL_LABEL"] = args.final_label
    if args.ae_research_controller_steps:
        env_vars["STEPPER_AE_RESEARCH_CONTROLLER_STEPS"] = str(args.ae_research_controller_steps)
    if args.ae_research_max_variants:
        env_vars["STEPPER_AE_RESEARCH_MAX_VARIANTS"] = str(args.ae_research_max_variants)
    if args.ae_research_variant_names:
        env_vars["STEPPER_AE_RESEARCH_VARIANT_NAMES"] = str(args.ae_research_variant_names)
    if args.ae_research_eval_rows:
        env_vars["STEPPER_AE_RESEARCH_EVAL_ROWS"] = str(args.ae_research_eval_rows)
    if args.ae_research_periodic_names:
        env_vars["STEPPER_AE_RESEARCH_PERIODIC_NAMES"] = str(args.ae_research_periodic_names)
    if args.ae_research_nonperiodic_names:
        env_vars["STEPPER_AE_RESEARCH_NONPERIODIC_NAMES"] = str(args.ae_research_nonperiodic_names)
    if args.ae_research_controller_periodic_names:
        env_vars["STEPPER_AE_RESEARCH_CONTROLLER_PERIODIC_NAMES"] = str(args.ae_research_controller_periodic_names)
    if args.ae_research_controller_nonperiodic_names:
        env_vars["STEPPER_AE_RESEARCH_CONTROLLER_NONPERIODIC_NAMES"] = str(args.ae_research_controller_nonperiodic_names)
    if args.ae_research_controller_pose_noise:
        env_vars["STEPPER_AE_RESEARCH_CONTROLLER_POSE_NOISE"] = str(args.ae_research_controller_pose_noise)
    if args.ae_research_window_frames:
        env_vars["STEPPER_AE_RESEARCH_WINDOW_FRAMES"] = str(args.ae_research_window_frames)
    if args.ae_research_skip_one_frame_baseline:
        env_vars["STEPPER_AE_RESEARCH_SKIP_ONE_FRAME_BASELINE"] = "1"
    if args.npz:
        env_vars["STEPPER_NPZ"] = copy_file_to_payload(Path(args.npz), payload_root)
    if enabled_text(args.periodic_folder):
        env_vars["STEPPER_PERIODIC_FOLDER"] = copy_dataset(Path(args.periodic_folder), payload_root)
    if enabled_text(args.nonperiodic_folder):
        env_vars["STEPPER_NONPERIODIC_FOLDER"] = copy_dataset(Path(args.nonperiodic_folder), payload_root)

    checkpoint_map = {
        "STEPPER_INIT_CHECKPOINT": args.init_checkpoint,
        "STEPPER_AE_CHECKPOINT": args.ae_checkpoint,
        "STEPPER_BASELINE_CHECKPOINT": args.baseline_checkpoint,
        "STEPPER_REFINED_CHECKPOINT": args.refined_checkpoint,
        "STEPPER_WEIGHTS_JSON": args.weights_json,
    }
    for env_name, src_text in checkpoint_map.items():
        if src_text:
            env_vars[env_name] = copy_file_to_payload(Path(src_text), payload_root)
    if args.phase:
        env_vars["STEPPER_PHASE"] = args.phase
    if args.pose_noise:
        env_vars["STEPPER_POSE_NOISE"] = str(args.pose_noise)

    dataset_meta = {"title": "Stepper IK Payload", "id": dataset_id, "licenses": [{"name": "CC0-1.0"}]}
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps(dataset_meta, indent=2), encoding="utf-8")
    (dataset_dir / "datasets-metadata.json").write_text(json.dumps(dataset_meta, indent=2), encoding="utf-8")

    if args.script_kernel:
        code_path = kernel_dir / "stepper_ik_trainer.py"
        write_script(code_path, dataset_slug, env_vars, not args.no_tensorboard_tunnel)
        kernel_type = "script"
    else:
        code_path = kernel_dir / "stepper_ik_trainer.ipynb"
        write_notebook(code_path, dataset_slug, env_vars, not args.no_tensorboard_tunnel)
        kernel_type = "notebook"
    kernel_meta = {
        "id": kernel_id,
        "title": kernel_slug.replace("-", " "),
        "code_file": code_path.name,
        "language": "python",
        "kernel_type": kernel_type,
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

    print(f"prepared IK Kaggle dataset payload: {dataset_dir}", flush=True)
    print(f"prepared IK Kaggle kernel:          {kernel_dir}", flush=True)
    print(f"dataset id: {dataset_id}", flush=True)
    print(f"kernel id:  {kernel_id}", flush=True)

    kaggle = kaggle_executable()
    if args.upload:
        if args.version_dataset:
            run_cli([kaggle, "datasets", "version", "-p", str(dataset_dir), "-m", "refresh IK payload", "-r", "zip"])
        else:
            run_cli([kaggle, "datasets", "create", "-p", str(dataset_dir), "-r", "zip"])
    if args.push_kernel:
        run_cli([kaggle, "kernels", "push", "-p", str(kernel_dir), "--accelerator", args.accelerator])


if __name__ == "__main__":
    main()
