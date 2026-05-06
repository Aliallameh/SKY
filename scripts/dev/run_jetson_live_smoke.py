"""Run the repeatable Jetson live-camera smoke sequence.

This orchestrates:
1. preflight diagnostics
2. LiveCameraSource 300-frame smoke
3. full SkyScouter live-camera pipeline smoke

It intentionally stays log-only. It does not start MAVLink, ESP32, or any
command transport.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Jetson live camera smoke sequence.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--backend", default="v4l2")
    parser.add_argument("--pipeline-config", default="configs/jetson_live_camera_pytorch.yaml")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--probe-camera-preflight", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("data") / "outputs" / f"jetson_live_smoke_{stamp}"


def run_step(name: str, cmd: list[str], cwd: Path, log_dir: Path) -> Dict[str, Any]:
    print(f"\n== {name} ==")
    print(" ".join(cmd))
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    log_path = log_dir / f"{name}.log"
    log_path.write_text(
        f"$ {' '.join(cmd)}\n\n# STDOUT\n{result.stdout}\n\n# STDERR\n{result.stderr}\n",
        encoding="utf-8",
    )
    ok = result.returncode == 0
    print(f"{name}: {'PASS' if ok else 'FAIL'} returncode={result.returncode}")
    return {
        "name": name,
        "ok": ok,
        "returncode": result.returncode,
        "started_utc": started,
        "log": str(log_path),
        "cmd": cmd,
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    logs_dir = out_dir / "logs"
    pipeline_out = out_dir / "pipeline"
    logs_dir.mkdir(parents=True, exist_ok=True)

    steps: list[Dict[str, Any]] = []

    if not args.skip_preflight:
        preflight_cmd = [
            args.python,
            "scripts/dev/jetson_preflight_check.py",
            "--camera-index",
            str(args.device_index),
            "--camera-device",
            f"/dev/video{args.device_index}",
            "--config",
            args.pipeline_config,
            "--output-dir",
            str(out_dir / "preflight"),
        ]
        if args.probe_camera_preflight:
            preflight_cmd.append("--probe-camera")
        steps.append(run_step("preflight", preflight_cmd, _ROOT, logs_dir))

    camera_cmd = [
        args.python,
        "scripts/dev/test_live_camera_source.py",
        "--device-index",
        str(args.device_index),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--fps",
        str(args.fps),
        "--fourcc",
        args.fourcc,
        "--backend",
        args.backend,
        "--max-frames",
        str(args.frames),
        "--warmup-frames",
        str(args.warmup_frames),
    ]
    steps.append(run_step("live_camera_source", camera_cmd, _ROOT, logs_dir))

    if not args.skip_pipeline:
        pipeline_cmd = [
            args.python,
            "scripts/run_pipeline.py",
            "--config",
            args.pipeline_config,
            "--output",
            str(pipeline_out),
        ]
        steps.append(run_step("pipeline", pipeline_cmd, _ROOT, logs_dir))

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output_dir": str(out_dir),
        "steps": steps,
        "overall_ok": all(step["ok"] for step in steps),
        "expected_pipeline_artifacts": [
            str(pipeline_out / "annotated.mp4"),
            str(pipeline_out / "target_states.jsonl"),
            str(pipeline_out / "guidance_hints.jsonl"),
            str(pipeline_out / "diagnostics.csv"),
            str(pipeline_out / "manifest.json"),
        ],
    }
    report_path = out_dir / "jetson_live_smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote smoke report: {report_path}")
    return 0 if report["overall_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
