from __future__ import annotations

import re
import time
from pathlib import Path


STAMP_RE = re.compile(r"^\d{8}_\d{6}_(?:ik_)?")


def clean_label(label: str) -> str:
    text = STAMP_RE.sub("", str(label).strip())
    text = text.replace("\\", "_").replace("/", "_").strip("_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text or "run"


def ik_run_id(label: str, now: float | None = None) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now or time.time()))
    return f"{stamp}_ik_{clean_label(label)}"


def checkpoint_path(run_dir: Path, run_id: str, tag: str) -> Path:
    return run_dir / "checkpoints" / f"{run_id}_{tag}.pt"
