"""
Render an enhanced overlay video from an existing Skyscouter run.

This avoids re-running detector inference when target_states.jsonl and
guidance_hints.jsonl already contain the tracked bbox and guidance geometry.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from skyscouter.output.annotator import STATE_COLORS  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render enhanced bbox geometry overlay from a run directory")
    p.add_argument("--video", required=True, help="Original input video")
    p.add_argument("--run-dir", required=True, help="Run directory containing target_states.jsonl")
    p.add_argument("--output", default=None, help="Output MP4 path")
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def color_for_state(state: str) -> Tuple[int, int, int]:
    return STATE_COLORS.get(state, (200, 200, 200))


def put_text(
    img: Any,
    text: str,
    xy: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: float = 0.5,
    thickness: int = 1,
) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_text_block(
    img: Any,
    lines: Iterable[str],
    anchor_xy: Tuple[int, int],
    color: Tuple[int, int, int],
) -> None:
    lines = list(lines)
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    pad = 7
    line_h = 18
    widths = [cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines]
    block_w = max(widths) + pad * 2
    block_h = line_h * len(lines) + pad * 2
    ih, iw = img.shape[:2]
    x = max(0, int(anchor_xy[0]))
    y = max(0, int(anchor_xy[1]))
    if x + block_w >= iw:
        x = max(0, iw - block_w - 2)
    if y + block_h >= ih:
        y = max(0, ih - block_h - 2)

    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + block_w, y + block_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
    cv2.rectangle(img, (x, y), (x + block_w, y + block_h), color, 1)
    for i, line in enumerate(lines):
        put_text(img, line, (x + pad, y + pad + 11 + i * line_h), color, scale, thickness)


def draw_hud(img: Any, state: Dict[str, Any], hint: Optional[Dict[str, Any]], frame_index: int) -> None:
    h, w = img.shape[:2]
    strip_h = 86
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, strip_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, img, 0.32, 0, img)

    lock_state = str(state.get("lock_state") or "NO_CUE")
    color = color_for_state(lock_state)
    guidance_valid = bool(state.get("guidance_valid"))
    gv_text = "GUIDANCE VALID" if guidance_valid else "GUIDANCE INVALID"
    gv_color = (80, 220, 80) if guidance_valid else (180, 180, 180)
    conf = float(state.get("confidence") or 0.0)
    lock_q = float(state.get("lock_quality") or 0.0)
    latency = float(state.get("latency_ms") or 0.0)
    put_text(img, f"STATE  {lock_state}", (10, 27), color, 0.72, 2)
    put_text(img, gv_text, (260, 27), gv_color, 0.72, 2)
    put_text(
        img,
        f"frame {frame_index}   confidence {conf:.2f}   lock quality {lock_q:.2f}   latency {latency:.0f} ms",
        (10, 57),
        (235, 235, 235),
        0.53,
        1,
    )
    if hint:
        bearing = hint.get("filtered_bearing_error_deg")
        elevation = hint.get("filtered_elevation_error_deg")
        yaw = float(hint.get("yaw_rate_cmd_deg_s") or 0.0)
        if bearing is not None and elevation is not None:
            put_text(
                img,
                f"bearing {float(bearing):+.2f} deg   elevation {float(elevation):+.2f} deg   yaw proposal {yaw:+.1f} deg/s",
                (10, 80),
                (255, 255, 255),
                0.5,
                1,
            )


def draw_guidance(img: Any, hint: Optional[Dict[str, Any]]) -> Tuple[float, float]:
    h, w = img.shape[:2]
    frame_center = [w * 0.5, h * 0.5]
    if hint:
        frame_center = hint.get("frame_center_px") or frame_center
    fc = (int(round(frame_center[0])), int(round(frame_center[1])))
    valid = bool(hint and hint.get("valid"))
    center_color = (255, 255, 255) if valid else (140, 140, 140)
    cv2.line(img, (fc[0] - 12, fc[1]), (fc[0] + 12, fc[1]), center_color, 1, cv2.LINE_AA)
    cv2.line(img, (fc[0], fc[1] - 12), (fc[0], fc[1] + 12), center_color, 1, cv2.LINE_AA)

    target_center = hint.get("target_center_px") if hint else None
    if target_center:
        tc = (int(round(target_center[0])), int(round(target_center[1])))
        cv2.circle(img, tc, 4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.line(img, fc, tc, (0, 255, 255), 1, cv2.LINE_AA)
    return float(frame_center[0]), float(frame_center[1])


def draw_bbox(img: Any, state: Dict[str, Any], frame_center: Tuple[float, float]) -> None:
    bbox = state.get("bbox_xywh")
    if not bbox:
        return
    x, y, bw, bh = [float(v) for v in bbox]
    x2 = x + bw
    y2 = y + bh
    cx = x + bw * 0.5
    cy = y + bh * 0.5
    dx = cx - frame_center[0]
    dy = cy - frame_center[1]

    lock_state = str(state.get("lock_state") or "NO_CUE")
    color = color_for_state(lock_state)
    thickness = 3 if state.get("track_id") is not None else 1
    cv2.rectangle(img, (round(x), round(y)), (round(x2), round(y2)), color, thickness)

    label = f"ID {state.get('track_id')}  conf {float(state.get('confidence') or 0.0):.2f}"
    put_text(img, label, (round(x), max(0, round(y) - 6)), color, 0.52, 2)
    lines = [
        f"box {bw:.0f} x {bh:.0f} px",
        f"top-left x {x:.0f}  y {y:.0f}",
        f"center   x {cx:.0f}  y {cy:.0f}",
        f"from center dx {dx:+.0f}  dy {dy:+.0f} px",
    ]
    draw_text_block(img, lines, (round(x2) + 8, round(y)), color)


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else run_dir / "annotated_geometry.mp4"
    states = read_jsonl(run_dir / "target_states.jsonl")
    hints_path = run_dir / "guidance_hints.jsonl"
    hints = read_jsonl(hints_path) if hints_path.exists() else []

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = min(total, len(states))

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, fps),
        (width, height),
    )
    if not writer.isOpened():
        raise IOError(f"Could not open writer: {output}")

    try:
        for idx in range(count):
            ok, frame = cap.read()
            if not ok:
                break
            state = states[idx]
            hint = hints[idx] if idx < len(hints) else None
            frame_center = draw_guidance(frame, hint)
            draw_bbox(frame, state, frame_center)
            draw_hud(frame, state, hint, idx)
            writer.write(frame)
    finally:
        writer.release()
        cap.release()

    print(f"Rendered {count} frames to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
