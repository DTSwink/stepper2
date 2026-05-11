from __future__ import annotations

import argparse
import html
import http.server
import socketserver
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = PROJECT_ROOT / "data" / "npz_final" / "testcasc.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "training" / "runs" / "model_comparisons" / "model_comparison.html"


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def newest_checkpoint(checkpoint_name: str) -> Path:
    runs_dir = PROJECT_ROOT / "training" / "runs"
    candidates: list[Path] = []
    if runs_dir.exists():
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name == "model_comparisons":
                continue
            checkpoint = run_dir / "checkpoints" / checkpoint_name
            if checkpoint.exists():
                candidates.append(checkpoint)
    if not candidates and checkpoint_name != "checkpoint_best.pt":
        return newest_checkpoint("checkpoint_best.pt")
    if not candidates:
        raise FileNotFoundError(f"No checkpoints named {checkpoint_name!r} found under {runs_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint_path:
        return resolve_path(args.checkpoint_path)
    if args.run_dir:
        return resolve_path(args.run_dir) / "checkpoints" / args.checkpoint_name
    return newest_checkpoint(args.checkpoint_name)


def regenerate(args: argparse.Namespace) -> Path:
    checkpoint = resolve_checkpoint(args)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    output = resolve_path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "training" / "visualize_model.py"),
        "--npz-path",
        str(resolve_path(args.npz_path)),
        "--checkpoint-path",
        str(checkpoint),
        "--output-path",
        str(output),
        "--device",
        args.device,
    ]
    if args.max_frames is not None:
        cmd += ["--max-frames", str(args.max_frames)]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return output


class ReusableThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_handler(args: argparse.Namespace):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path not in ("/", "/model_comparison.html"):
                self.send_error(404, "Only /model_comparison.html is served by this viewer.")
                return
            try:
                output = regenerate(args)
                data = output.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.end_headers()
                self.wfile.write(data)
            except Exception as exc:
                message = (
                    "<!doctype html><meta charset='utf-8'>"
                    "<title>Viewer refresh failed</title>"
                    "<body style='font-family: sans-serif; background:#111; color:#eee'>"
                    "<h1>Viewer refresh failed</h1>"
                    f"<pre>{html.escape(str(exc))}</pre>"
                    "</body>"
                ).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(message)))
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.end_headers()
                self.wfile.write(message)

        def log_message(self, fmt: str, *values: object) -> None:
            print(f"{self.address_string()} - {fmt % values}", flush=True)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the model comparison viewer and regenerate it only when the browser refreshes."
    )
    parser.add_argument("--npz-path", default=str(DEFAULT_NPZ))
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint-name", default="checkpoint_last.pt")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8017)
    parser.add_argument("--once", action="store_true", help="Regenerate once and exit.")
    args = parser.parse_args()

    if args.once:
        output = regenerate(args)
        print(f"Generated {output}")
        return

    url = f"http://{args.host}:{args.port}/model_comparison.html"
    checkpoint_hint = args.checkpoint_path or (
        str(resolve_path(args.run_dir) / "checkpoints" / args.checkpoint_name) if args.run_dir else f"newest {args.checkpoint_name}"
    )
    print(f"Serving {url}", flush=True)
    print(f"Regenerates on refresh from: {checkpoint_hint}", flush=True)
    with ReusableThreadingHTTPServer((args.host, args.port), make_handler(args)) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
