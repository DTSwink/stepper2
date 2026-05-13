from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class VisualReportBridge:
    def __init__(
        self,
        run_dir: Path,
        *,
        npz_path: Path | None = None,
        checkpoint_name: str = "checkpoint_last.pt",
        interval_seconds: float = 60.0,
        device: str = "cpu",
        max_frames: int = 180,
    ) -> None:
        self.run_dir = run_dir
        self.npz_path = npz_path
        self.checkpoint_name = checkpoint_name
        self.interval_seconds = interval_seconds
        self.device = device
        self.max_frames = max_frames
        self.process: subprocess.Popen | None = None
        self._stdout = None
        self._stderr = None

    def start(self) -> None:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "training" / "visual_reporter.py"),
            "--run-dir",
            str(self.run_dir),
            "--checkpoint-name",
            self.checkpoint_name,
            "--interval-seconds",
            str(self.interval_seconds),
            "--device",
            self.device,
            "--max-frames",
            str(self.max_frames),
        ]
        if self.npz_path is not None:
            cmd.extend(["--npz-path", str(self.npz_path)])
        output_dir = self.run_dir / "visual_reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        self._stdout = open(output_dir / "reporter_stdout.log", "a", encoding="utf-8")
        self._stderr = open(output_dir / "reporter_stderr.log", "a", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=self._stdout,
                stderr=self._stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
        except Exception:
            self._stdout.close()
            self._stderr.close()
            self._stdout = None
            self._stderr = None
            raise

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self._stdout is not None:
            self._stdout.close()
            self._stdout = None
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None
