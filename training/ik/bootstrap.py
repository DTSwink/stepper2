from __future__ import annotations

import sys
from pathlib import Path


IK_DIR = Path(__file__).resolve().parent
TRAINING_DIR = IK_DIR.parent
PROJECT_ROOT = TRAINING_DIR.parent


def ensure_paths() -> None:
    for path in (IK_DIR, PROJECT_ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
