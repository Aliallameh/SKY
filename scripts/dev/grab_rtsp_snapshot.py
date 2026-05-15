"""
Grab a single frame (or a small burst) from the SIYI A8 Mini RTSP stream
and save it to Jetson disk. Faster and more convenient than the SDK's
"take photo" command (which writes to the camera's microSD card and is
not retrievable over UDP).

Usage:
    # One frame, default output path
    python scripts/dev/grab_rtsp_snapshot.py

    # Custom output path
    python scripts/dev/grab_rtsp_snapshot.py --out /tmp/test.jpg

    # Burst of 5 frames spaced ~250 ms apart, written into a directory
    python scripts/dev/grab_rtsp_snapshot.py --count 5 --interval 0.25 \
        --out-dir data/snapshots/

    # JPEG with chosen quality (default 95) or PNG (lossless)
    python scripts/dev/grab_rtsp_snapshot.py --format png

Notes:
  * The first read off RTSP may discard up to ~30 frames while the decoder
    syncs to the next IDR (keyframe). The --warmup-frames flag controls this.
  * Snapshot is the same 1920x1080 H.265-decoded frame the perception
    pipeline sees, not the full 4K-capable sensor output. For a full-
    resolution still, use the SIYI SDK photo command (siyi_gimbal_bench.py
    --photo --send) which writes to the camera's SD card.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Snapshot the SIYI A8 Mini RTSP stream to a file")
    p.add_argument("--rtsp-url", default="rtsp://192.168.144.25:8554/main.264",
                   help="RTSP URL of the SIYI A8 Mini main stream")
    p.add_argument("--out", default=None,
                   help="Output file path. Default: data/snapshots/siyi_a8_<UTC>.jpg")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (used in --count > 1 mode, or to override default location).")
    p.add_argument("--count", type=int, default=1, help="Number of snapshots to grab.")
    p.add_argument("--interval", type=float, default=0.0,
                   help="Seconds between snapshots when --count > 1. Default 0 (back-to-back).")
    p.add_argument("--warmup-frames", type=int, default=15,
                   help="Frames to discard before first capture (lets the H.265 decoder sync). Default 15.")
    p.add_argument("--format", choices=["jpg", "jpeg", "png"], default="jpg",
                   help="Output file format.")
    p.add_argument("--jpeg-quality", type=int, default=95,
                   help="JPEG quality 1-100 (default 95). Ignored for PNG.")
    p.add_argument("--open-timeout-s", type=float, default=10.0,
                   help="How long to wait for the RTSP stream to open. Default 10 s.")
    return p.parse_args()


def open_rtsp(url: str, timeout_s: float) -> cv2.VideoCapture:
    # Force TCP transport (matches the rest of the pipeline).
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    t0 = time.monotonic()
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    while not cap.isOpened():
        if time.monotonic() - t0 > timeout_s:
            raise RuntimeError(f"RTSP open timed out after {timeout_s:.1f} s: {url}")
        time.sleep(0.1)
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    return cap


def main() -> int:
    args = parse_args()
    ext = "jpg" if args.format in ("jpg", "jpeg") else "png"

    def stamp_path(i: int) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")[:-3]
        suffix = f"_{i:02d}" if args.count > 1 else ""
        return Path(f"siyi_a8_{ts}{suffix}.{ext}")

    if args.out and args.count > 1:
        print("WARNING: --out is ignored when --count > 1; outputs will go to --out-dir.")
    if args.count > 1:
        out_dir = Path(args.out_dir) if args.out_dir else Path("data/snapshots")
        out_dir.mkdir(parents=True, exist_ok=True)
        targets = [out_dir / stamp_path(i) for i in range(args.count)]
    elif args.out:
        targets = [Path(args.out)]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(args.out_dir) if args.out_dir else Path("data/snapshots")
        out_dir.mkdir(parents=True, exist_ok=True)
        targets = [out_dir / stamp_path(0)]

    print(f"Opening RTSP: {args.rtsp_url}")
    cap = open_rtsp(args.rtsp_url, args.open_timeout_s)

    # Request full HD from decoder/camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("Backend:", cap.getBackendName())
    print("Width :", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("FPS   :", cap.get(cv2.CAP_PROP_FPS))
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(f"Stream opened: reported {w}x{h}  warmup={args.warmup_frames} frames")

    # Warm up — discard until we get the first decoded frame, then a few more
    # to clear any residual decoder lag.
    warmed = 0
    deadline = time.monotonic() + 6.0
    while warmed < args.warmup_frames:
        ok, frame = cap.read()
        if ok and frame is not None:
            warmed += 1
        if time.monotonic() > deadline:
            print("WARNING: warmup did not complete in 6 s; proceeding with whatever the decoder has.")
            break

    written: list[Path] = []
    for i, path in enumerate(targets):
        if i > 0 and args.interval > 0:
            time.sleep(args.interval)
        # Drain stale buffered frames so we get the most recent one
        for _ in range(2):
            cap.grab()
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"ERROR: frame read failed at i={i}; aborting.")
            break
        print("Decoded frame size:", frame.shape[1], "x", frame.shape[0])
        
        if ext == "jpg":
            params = [int(cv2.IMWRITE_JPEG_QUALITY), int(max(1, min(100, args.jpeg_quality)))]
        else:
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]

        ok = cv2.imwrite(str(path), frame, params)
        if not ok:
            print(f"ERROR: failed to write {path}")
            break
        size = path.stat().st_size
        # print(f"  wrote {path}  ({frame.shape[1]}x{frame.shape[0]}, {size//1024} KiB)")
        written.append(path)

    cap.release()

    if not written:
        return 1
    print()
    print(f"{len(written)} file{'s' if len(written) != 1 else ''} written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
