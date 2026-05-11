"""Launch the full Jetson live-camera SkyScouter pipeline.

This is the non-smoke runtime entrypoint. It runs until the camera stops or the
operator presses Ctrl+C. The pipeline is log-only/advisory: no MAVLink, no
ESP32 command relay, and no actuator authority.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent


CONFIGS = {
    "pytorch": "configs/jetson_live_camera_pytorch_full.yaml",
    "tensorrt": "configs/deploy_jetson_yolo11s_stage2_multiclass_capped_engine_1080p.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Jetson live SkyScouter pipeline.")
    parser.add_argument("--backend", choices=sorted(CONFIGS), default="tensorrt")
    parser.add_argument("--config", default=None, help="Optional config override.")
    parser.add_argument("--output", default=None, help="Optional output directory.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--operator-view", action="store_true",
                        help="Enable live annotated operator view.")
    parser.add_argument("--no-operator-view", action="store_true",
                        help="Disable live annotated operator view.")
    parser.add_argument("--operator-view-mode", choices=["mjpeg", "window", "both"], default=None)
    parser.add_argument("--operator-view-host", default=None)
    parser.add_argument("--operator-view-port", type=int, default=None)
    parser.add_argument("--operator-view-max-width", type=int, default=None)
    parser.add_argument("--operator-view-jpeg-quality", type=int, default=None)
    return parser.parse_args()


def default_output_dir(backend: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("data") / "outputs" / f"jetson_live_{backend}_{stamp}"


def main() -> int:
    args = parse_args()
    config = Path(args.config or CONFIGS[args.backend])
    output = Path(args.output) if args.output else default_output_dir(args.backend)
    if not config.exists():
        raise FileNotFoundError(f"Config not found: {config}")

    cmd = [
        args.python,
        "scripts/run_pipeline.py",
        "--config",
        str(config),
        "--output",
        str(output),
    ]
    if args.operator_view:
        cmd.append("--operator-view")
    if args.no_operator_view:
        cmd.append("--no-operator-view")
    if args.operator_view_mode is not None:
        cmd.extend(["--operator-view-mode", args.operator_view_mode])
    if args.operator_view_host is not None:
        cmd.extend(["--operator-view-host", args.operator_view_host])
    if args.operator_view_port is not None:
        cmd.extend(["--operator-view-port", str(args.operator_view_port)])
    if args.operator_view_max_width is not None:
        cmd.extend(["--operator-view-max-width", str(args.operator_view_max_width)])
    if args.operator_view_jpeg_quality is not None:
        cmd.extend(["--operator-view-jpeg-quality", str(args.operator_view_jpeg_quality)])
    print("Launching full live pipeline:")
    print(" ".join(cmd))
    print("Press Ctrl+C to stop and finalize the manifest cleanly.")
    result = subprocess.run(cmd, cwd=str(_ROOT), check=False)
    return int(result.returncode)


if __name__ == "__main__":
    sys.exit(main())
