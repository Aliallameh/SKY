"""
Interactive capture tool for camera-intrinsic calibration.

Opens the SIYI A8 Mini RTSP stream, runs live chessboard detection, and lets
you capture frames where the board is well-resolved. Default board is 7x8
squares (6x7 inner corners), 25 mm squares — pass --cols / --rows to override.

Capture target: 20-30 frames covering distance, pose, and frame-region
diversity. The on-screen overlay tells you whether the board is currently
detected so you don't waste captures.

Controls (focus must be on the OpenCV window, not the terminal):
    SPACE    -> save current frame to the session directory
    Q / ESC  -> quit
    R        -> remove the last capture (in case it was a duplicate / blur)

Output: data/calibration/raw/<timestamp>/frame_XXXX.png
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture chessboard frames for calibration")
    p.add_argument("--rtsp-url", default="rtsp://192.168.144.25:8554/main.264",
                   help="RTSP URL of the SIYI A8 Mini main stream")
    p.add_argument("--cols", type=int, default=6,
                   help="Number of INNER corners across (squares_across - 1). Default 6 for 7x8 board.")
    p.add_argument("--rows", type=int, default=7,
                   help="Number of INNER corners tall (squares_tall - 1). Default 7 for 7x8 board.")
    p.add_argument("--square-mm", type=float, default=25.0,
                   help="Square side length in millimeters. Default 25.")
    p.add_argument("--out-dir", default=None,
                   help="Where to save captures. Default data/calibration/raw/<timestamp>/")
    p.add_argument("--max-captures", type=int, default=40, help="Stop after this many captures.")
    return p.parse_args()


def open_rtsp(url: str) -> cv2.VideoCapture:
    # Force TCP transport; matches the rest of the pipeline.
    import os
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open RTSP source: {url}")
    return cap


def main() -> int:
    args = parse_args()
    pattern_size = (int(args.cols), int(args.rows))  # OpenCV expects (cols, rows) in inner-corner count

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"data/calibration/raw/{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the session metadata next to the frames so the solver can read it.
    (out_dir / "session.txt").write_text(
        f"pattern_cols_inner={pattern_size[0]}\n"
        f"pattern_rows_inner={pattern_size[1]}\n"
        f"square_mm={args.square_mm}\n"
        f"rtsp_url={args.rtsp_url}\n"
        f"created_utc={stamp}\n"
    )

    print(f"Session    : {out_dir}")
    print(f"Pattern    : {pattern_size[0]} x {pattern_size[1]} inner corners (board {pattern_size[0]+1} x {pattern_size[1]+1} squares)")
    print(f"Square     : {args.square_mm} mm")
    print(f"RTSP       : {args.rtsp_url}")
    print()
    print("Controls (focus the OpenCV window):")
    print("  SPACE  capture current frame")
    print("  R      remove last capture")
    print("  Q/ESC  quit")
    print()

    cap = open_rtsp(args.rtsp_url)
    win = "calibration capture - press SPACE to grab"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    captures: list[Path] = []
    last_frame: np.ndarray | None = None

    chessboard_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.02)
            continue
        last_frame = frame
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=chessboard_flags)
        if found:
            cv2.drawChessboardCorners(display, pattern_size, corners, True)
            status = "BOARD DETECTED"
            color = (0, 255, 0)
        else:
            status = "no board"
            color = (0, 0, 255)

        h, w = display.shape[:2]
        cv2.rectangle(display, (0, 0), (w, 56), (0, 0, 0), -1)
        cv2.putText(display, status, (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        cv2.putText(display, f"captured: {len(captures)} / target ~25", (12, 47),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.imshow(win, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord(" "):
            if not found:
                print("  (skipped — no chessboard found in current frame)")
                continue
            idx = len(captures)
            path = out_dir / f"frame_{idx:04d}.png"
            cv2.imwrite(str(path), frame)
            captures.append(path)
            print(f"  + captured {path.name}  ({len(captures)} total)")
            if len(captures) >= args.max_captures:
                print(f"  reached --max-captures={args.max_captures}, stopping")
                break
        if key == ord("r"):
            if captures:
                rm = captures.pop()
                try:
                    rm.unlink()
                except FileNotFoundError:
                    pass
                print(f"  - removed {rm.name}  ({len(captures)} total)")

    cap.release()
    cv2.destroyAllWindows()

    print()
    print(f"Done. {len(captures)} frames saved to: {out_dir}")
    if len(captures) < 15:
        print("WARNING: fewer than 15 captures. Calibration will be unstable; capture more.")
    else:
        print(f"Next step:")
        print(f"  /home/office/SKY/.venv_jetson/bin/python /home/office/SKY/scripts/dev/solve_calibration.py \\")
        print(f"      --session {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
