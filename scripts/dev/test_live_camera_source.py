"""Smoke-test the SkyScouter LiveCameraSource on Jetson.

This tests the repo frame source directly, not just raw OpenCV. It should
report the same negotiated camera mode as the verified direct OpenCV smoke:
MJPG, 1280x720, 30 fps, approximately 28-30 captured FPS.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from skyscouter.io.live_camera_source import LiveCameraSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test Jetson live camera source.")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--backend", default="v4l2")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument("--source-id", default="jetson_4k_zoom_camera")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = LiveCameraSource(
        device_index=args.device_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=args.fourcc,
        backend=args.backend,
        source_id=args.source_id,
        max_frames=args.max_frames,
        warmup_frames=args.warmup_frames,
        strict=True,
    )
    try:
        print("opened: True")
        print(f"settings: {source.get_negotiated_settings()}")
        start = time.perf_counter()
        count = 0
        last_shape = None
        for frame in source:
            count += 1
            last_shape = frame.image_bgr.shape
            if args.print_every > 0 and count % args.print_every == 0:
                print(
                    f"frame {count} shape {frame.image_bgr.shape} "
                    f"capture_time_s={frame.capture_time_s:.3f} "
                    f"timestamp_utc={frame.timestamp_utc}"
                )
        elapsed = time.perf_counter() - start
        observed_fps = count / elapsed if elapsed > 0 else 0.0
        print(f"total_frames: {count}")
        print(f"seconds: {elapsed:.2f}")
        print(f"approx_fps: {observed_fps:.2f}")
        print(f"last_shape: {last_shape}")
        return 0 if count == args.max_frames else 2
    finally:
        source.close()


if __name__ == "__main__":
    sys.exit(main())
