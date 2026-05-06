"""Benchmark a YOLO .pt or TensorRT .engine detector on video or USB camera.

This is intentionally detector-focused. It measures model inference latency
without the tracker, lock state, video writer, or overlay renderer.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SkyScouter YOLO detector backend.")
    parser.add_argument("--model", required=True, help="Path to .pt or .engine model.")
    parser.add_argument(
        "--source",
        required=True,
        help="Video path, integer camera index such as 0, or /dev/videoN.",
    )
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="0")
    parser.add_argument("--frames", type=int, default=300, help="Measured frames after warmup.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument("--camera-fps", type=float, default=None)
    parser.add_argument("--fourcc", default=None, help="Camera FOURCC such as MJPG.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Defaults to data/outputs/benchmarks/<timestamp>.json.",
    )
    return parser.parse_args()


def open_capture(args: argparse.Namespace) -> Tuple[cv2.VideoCapture, str, bool]:
    source = args.source
    is_camera = source.isdigit() or source.startswith("/dev/video")
    if source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2 if is_camera else cv2.CAP_ANY)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    if is_camera:
        if args.camera_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        if args.camera_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        if args.camera_fps:
            cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
        if args.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc[:4]))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    source_id = f"camera:{source}" if is_camera else str(Path(source).resolve())
    return cap, source_id, is_camera


def frames_from_capture(cap: cv2.VideoCapture) -> Iterator[np.ndarray]:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = (len(sorted_values) - 1) * pct
    lower = int(idx)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = idx - lower
    return sorted_values[lower] * (1.0 - frac) + sorted_values[upper] * frac


def cpu_snapshot() -> Dict[str, Any]:
    try:
        import psutil  # type: ignore

        proc = psutil.Process()
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "process_rss_mb": round(proc.memory_info().rss / (1024 * 1024), 2),
            "system_memory_percent": psutil.virtual_memory().percent,
        }
    except Exception as exc:
        return {"error": str(exc)}


def cuda_snapshot() -> Dict[str, Any]:
    try:
        import torch  # type: ignore

        snap: Dict[str, Any] = {
            "torch": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
        }
        if torch.cuda.is_available():
            snap["device_name"] = torch.cuda.get_device_name(0)
            snap["memory_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
            snap["memory_reserved_mb"] = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2)
        return snap
    except Exception as exc:
        return {"error": str(exc)}


def default_output_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("data") / "outputs" / "benchmarks" / f"detector_benchmark_{stamp}.json"


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    from ultralytics import YOLO  # type: ignore

    model = YOLO(str(model_path))
    cap, source_id, is_camera = open_capture(args)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    latencies_ms: list[float] = []
    detections_per_frame: list[int] = []
    processed = 0
    warm = 0
    failures = 0
    start = time.perf_counter()

    try:
        for frame in frames_from_capture(cap):
            is_warmup = warm < args.warmup
            t0 = time.perf_counter()
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            det_count = 0
            if results and results[0].boxes is not None:
                det_count = int(len(results[0].boxes))

            if is_warmup:
                warm += 1
                continue

            latencies_ms.append(elapsed_ms)
            detections_per_frame.append(det_count)
            processed += 1
            if processed >= args.frames:
                break
    finally:
        cap.release()

    elapsed_s = time.perf_counter() - start
    if not latencies_ms:
        raise RuntimeError("No measured frames were processed.")

    measured_s = sum(latencies_ms) / 1000.0
    inference_fps = processed / measured_s if measured_s > 0 else 0.0
    end_to_end_fps = (warm + processed) / elapsed_s if elapsed_s > 0 else 0.0
    dropped_estimate = 0
    if is_camera and fps > 0:
        expected = int(round(elapsed_s * fps))
        dropped_estimate = max(0, expected - warm - processed - failures)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": str(model_path),
        "model_type": model_path.suffix.lower().lstrip("."),
        "source": source_id,
        "source_is_camera": is_camera,
        "source_resolution": {"width": width, "height": height, "fps_reported": fps},
        "settings": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "device": args.device,
            "warmup_frames": args.warmup,
            "measured_frames": args.frames,
        },
        "results": {
            "frames_measured": processed,
            "latency_ms_avg": round(statistics.fmean(latencies_ms), 3),
            "latency_ms_median": round(statistics.median(latencies_ms), 3),
            "latency_ms_p50": round(percentile(latencies_ms, 0.50), 3),
            "latency_ms_p95": round(percentile(latencies_ms, 0.95), 3),
            "latency_ms_min": round(min(latencies_ms), 3),
            "latency_ms_max": round(max(latencies_ms), 3),
            "inference_only_fps": round(inference_fps, 2),
            "end_to_end_fps": round(end_to_end_fps, 2),
            "detections_per_frame_avg": round(statistics.fmean(detections_per_frame), 3),
            "dropped_frames_estimate": dropped_estimate,
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "cpu": cpu_snapshot(),
            "cuda": cuda_snapshot(),
        },
    }

    out = Path(args.output) if args.output else default_output_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary["results"], indent=2))
    print(f"Wrote benchmark summary: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
