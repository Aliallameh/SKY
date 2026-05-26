"""Launch the full Jetson live-camera SkyScouter pipeline.

This is the non-smoke runtime entrypoint. It runs until the camera stops or the
operator presses Ctrl+C. The pipeline is log-only/advisory: no MAVLink, no
ESP32 command relay, and no actuator authority.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent


def _venv_python() -> str:
    """Return the venv python so subprocesses inherit site-packages.

    sys.executable resolves symlinks back to the system interpreter, which
    bypasses the active venv's site-packages. We need to pass the symlink
    path so Python finds pyvenv.cfg next to it and activates the venv.
    """
    # 1. Respect an explicit VIRTUAL_ENV env var (venv activated in shell)
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        candidate = Path(venv) / "bin" / "python3"
        if candidate.exists():
            return str(candidate)
    # 2. Jetson convention: .venv_jetson at repo root
    candidate = _ROOT / ".venv_jetson" / "bin" / "python3"
    if candidate.exists():
        return str(candidate)
    return sys.executable


CONFIGS = {
    "pytorch": "configs/jetson_live_camera_pytorch_full.yaml",
    "tensorrt": "configs/deploy_jetson_yolov26_lrdd_v2_siyi_a8_mini_1080p.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Jetson live SkyScouter pipeline.")
    parser.add_argument("--backend", choices=sorted(CONFIGS), default="tensorrt")
    parser.add_argument("--config", default=None, help="Optional config override.")
    parser.add_argument("--output", default=None, help="Optional output directory.")
    parser.add_argument("--python", default=_venv_python())
    parser.add_argument("--operator-view", action="store_true",
                        help="Enable live annotated operator view.")
    parser.add_argument("--no-operator-view", action="store_true",
                        help="Disable live annotated operator view.")
    parser.add_argument("--operator-view-mode", choices=["mjpeg", "window", "both"], default=None)
    parser.add_argument("--operator-view-host", default=None)
    parser.add_argument("--operator-view-port", type=int, default=None)
    parser.add_argument("--operator-view-max-width", type=int, default=None)
    parser.add_argument("--operator-view-jpeg-quality", type=int, default=None)
    parser.add_argument("--operator-view-fullscreen", action="store_true")
    parser.add_argument("--operator-view-windowed", action="store_true")
    parser.add_argument("--operator-view-display-width", type=int, default=None)
    parser.add_argument("--operator-view-display-height", type=int, default=None)
    parser.add_argument("--operator-view-window-x", type=int, default=None)
    parser.add_argument("--operator-view-window-y", type=int, default=None)
    parser.add_argument("--operator-view-window-backend", choices=["opencv", "gstreamer"], default=None)
    parser.add_argument("--operator-view-display-fps", type=int, default=None)
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
    if args.operator_view_fullscreen:
        cmd.append("--operator-view-fullscreen")
    if args.operator_view_windowed:
        cmd.append("--operator-view-windowed")
    if args.operator_view_display_width is not None:
        cmd.extend(["--operator-view-display-width", str(args.operator_view_display_width)])
    if args.operator_view_display_height is not None:
        cmd.extend(["--operator-view-display-height", str(args.operator_view_display_height)])
    if args.operator_view_window_x is not None:
        cmd.extend(["--operator-view-window-x", str(args.operator_view_window_x)])
    if args.operator_view_window_y is not None:
        cmd.extend(["--operator-view-window-y", str(args.operator_view_window_y)])
    if args.operator_view_window_backend is not None:
        cmd.extend(["--operator-view-window-backend", args.operator_view_window_backend])
    if args.operator_view_display_fps is not None:
        cmd.extend(["--operator-view-display-fps", str(args.operator_view_display_fps)])
    # Add the nvidia/cu12/lib dir from the venv so the dynamic linker can find
    # libcudss.so.0 (required by Jetson AI Lab torch, not shipped by JetPack).
    # Only this one directory is added; all other CUDA libs are left to the
    # system JetPack paths to avoid cuBLAS/cuDNN version conflicts.
    env = os.environ.copy()
    cudss_lib = (
        Path(args.python).parent.parent
        / "lib" / "python3.10" / "site-packages" / "nvidia" / "cu12" / "lib"
    )
    if cudss_lib.is_dir():
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = str(cudss_lib) + (":" + existing if existing else "")

    print("Launching full live pipeline:")
    print(" ".join(cmd))
    print("Press Ctrl+C to stop and finalize the manifest cleanly.")
    result = subprocess.run(cmd, cwd=str(_ROOT), env=env, check=False)
    return int(result.returncode)


if __name__ == "__main__":
    sys.exit(main())
