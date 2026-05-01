"""
Video annotator.

Draws bounding boxes, track IDs, and the lock state onto frames and writes an
output video. This is the primary visual verification artifact for bench
replay: Ali and Erfan watch the annotated output and immediately see whether
detection, tracking, and lock are behaving sensibly on real footage.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..schemas import LockState
from ..tracking.base_tracker import Track


# Color palette (BGR) per lock state. Red is reserved for failure/loss states;
# committed locks use green so the overlay reads naturally in review videos.
STATE_COLORS = {
    LockState.NO_CUE.value:       (140, 140, 140),  # gray
    LockState.CUED.value:         (255, 200, 0),    # cyan-blue
    LockState.SEARCHING.value:    (0, 200, 255),    # amber
    LockState.ACQUIRED.value:     (0, 230, 255),    # yellow
    LockState.TRACKING.value:     (255, 210, 0),    # cyan
    LockState.LOCKED.value:       (80, 220, 80),    # green
    LockState.STRIKE_READY.value: (255, 120, 0),    # blue
    LockState.LOST.value:         (0, 0, 255),      # red
    LockState.ABORTED.value:      (0, 0, 160),      # dark red
}


class VideoAnnotator:
    """Writes an MP4 with bounding boxes and state overlay."""

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float = 30.0,
        guidance_overlay_cfg: Optional[Dict[str, Any]] = None,
    ):
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # mp4v works on most ffmpeg/OpenCV builds. avc1 may be unavailable.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self._path), fourcc, max(1.0, fps), (width, height)
        )
        if not self._writer.isOpened():
            raise IOError(f"Could not open video writer for {self._path}")

        self._width = width
        self._height = height
        self._guidance_overlay_cfg = guidance_overlay_cfg or {"enabled": False}

    def annotate(
        self,
        image_bgr: np.ndarray,
        tracks: List[Track],
        primary_track_id: Optional[int],
        lock_state: str,
        guidance_valid: bool,
        confidence: float,
        lock_quality: float,
        latency_ms: float,
        frame_index: int,
        guidance_hint: Optional[Any] = None,
    ) -> np.ndarray:
        """Return an annotated copy of the frame."""
        img = image_bgr.copy()

        # Draw every track box (de-emphasized) and the primary one (emphasized)
        for tr in tracks:
            x1 = int(tr.detection.x)
            y1 = int(tr.detection.y)
            x2 = int(tr.detection.x + tr.detection.w)
            y2 = int(tr.detection.y + tr.detection.h)
            is_primary = (tr.track_id == primary_track_id)
            color = STATE_COLORS.get(lock_state, (200, 200, 200)) if is_primary else (180, 180, 180)
            thickness = 3 if is_primary else 1
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            label = f"ID {tr.track_id}  {tr.detection.class_label}  conf {tr.detection.confidence:.2f}"
            cv2.putText(
                img, label, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA
            )
            if bool(self._guidance_overlay_cfg.get("draw_bbox_geometry", False)):
                self._draw_bbox_geometry(
                    img,
                    x=float(tr.detection.x),
                    y=float(tr.detection.y),
                    w=float(tr.detection.w),
                    h=float(tr.detection.h),
                    color=color,
                    anchor_xy=(x2 + 8, y1),
                    frame_center=getattr(guidance_hint, "frame_center_px", None),
                )
            # STRIKE_READY gets a second outer border
            if is_primary and lock_state == LockState.STRIKE_READY.value:
                cv2.rectangle(img, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), color, 1)

        # HUD strip: state, confidence, lock_quality, latency, frame
        self._draw_hud(
            img,
            lock_state=lock_state,
            guidance_valid=guidance_valid,
            confidence=confidence,
            lock_quality=lock_quality,
            latency_ms=latency_ms,
            frame_index=frame_index,
        )
        if guidance_hint is not None and bool(self._guidance_overlay_cfg.get("enabled", False)):
            self._draw_guidance_overlay(img, guidance_hint)

        return img

    def write(self, image_bgr: np.ndarray) -> None:
        # Ensure correct size
        if image_bgr.shape[1] != self._width or image_bgr.shape[0] != self._height:
            image_bgr = cv2.resize(image_bgr, (self._width, self._height))
        self._writer.write(image_bgr)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None  # type: ignore

    def _draw_hud(
        self,
        img: np.ndarray,
        *,
        lock_state: str,
        guidance_valid: bool,
        confidence: float,
        lock_quality: float,
        latency_ms: float,
        frame_index: int,
    ) -> None:
        h, w = img.shape[:2]
        # Background strip
        strip_h = 86
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, strip_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.68, img, 0.32, 0, img)

        color = STATE_COLORS.get(lock_state, (200, 200, 200))
        gv_text = "GUIDANCE VALID" if guidance_valid else "GUIDANCE INVALID"
        gv_color = (80, 220, 80) if guidance_valid else (180, 180, 180)

        line1 = f"STATE  {lock_state}"
        line2 = (
            f"frame {frame_index}   confidence {confidence:.2f}   "
            f"lock quality {lock_quality:.2f}   latency {latency_ms:.0f} ms"
        )
        cv2.putText(img, line1, (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
        cv2.putText(img, gv_text, (260, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, gv_color, 2, cv2.LINE_AA)
        cv2.putText(img, line2, (10, 57),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.53, (235, 235, 235), 1, cv2.LINE_AA)

    def _draw_guidance_overlay(self, img: np.ndarray, hint: Any) -> None:
        cfg = self._guidance_overlay_cfg
        frame_center = getattr(hint, "frame_center_px", None)
        target_center = getattr(hint, "target_center_px", None)
        if frame_center is None:
            h, w = img.shape[:2]
            frame_center = [w * 0.5, h * 0.5]
        fc = (int(round(frame_center[0])), int(round(frame_center[1])))
        color = (255, 255, 255) if getattr(hint, "valid", False) else (140, 140, 140)

        if bool(cfg.get("draw_optical_center", True)):
            cv2.line(img, (fc[0] - 12, fc[1]), (fc[0] + 12, fc[1]), color, 1, cv2.LINE_AA)
            cv2.line(img, (fc[0], fc[1] - 12), (fc[0], fc[1] + 12), color, 1, cv2.LINE_AA)

        if target_center is not None:
            tc = (int(round(target_center[0])), int(round(target_center[1])))
            if bool(cfg.get("draw_target_center", True)):
                cv2.circle(img, tc, 4, (0, 255, 255), -1, cv2.LINE_AA)
            if bool(cfg.get("draw_error_vector", True)):
                cv2.line(img, fc, tc, (0, 255, 255), 1, cv2.LINE_AA)

        if bool(cfg.get("draw_text", True)):
            bearing = getattr(hint, "filtered_bearing_error_deg", None)
            elevation = getattr(hint, "filtered_elevation_error_deg", None)
            yaw_rate = getattr(hint, "yaw_rate_cmd_deg_s", 0.0)
            if bearing is None:
                text = "bearing n/a   yaw proposal 0.0 deg/s"
            else:
                text = (
                    f"bearing {bearing:+.2f} deg   elevation {elevation:+.2f} deg   "
                    f"yaw proposal {yaw_rate:+.1f} deg/s"
                )
            cv2.putText(
                img,
                text,
                (10, min(img.shape[0] - 10, 80)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _draw_bbox_geometry(
        self,
        img: np.ndarray,
        *,
        x: float,
        y: float,
        w: float,
        h: float,
        color: Tuple[int, int, int],
        anchor_xy: Tuple[int, int],
        frame_center: Optional[List[float]] = None,
    ) -> None:
        cx = x + w * 0.5
        cy = y + h * 0.5
        if frame_center is None:
            ih, iw = img.shape[:2]
            frame_center = [iw * 0.5, ih * 0.5]
        dx = cx - float(frame_center[0])
        dy = cy - float(frame_center[1])

        lines = [
            f"box {w:.0f} x {h:.0f} px",
            f"top-left x {x:.0f}  y {y:.0f}",
            f"center   x {cx:.0f}  y {cy:.0f}",
            f"from center dx {dx:+.0f}  dy {dy:+.0f} px",
        ]
        self._draw_text_block(img, lines, anchor_xy, color)

    def _draw_text_block(
        self,
        img: np.ndarray,
        lines: List[str],
        anchor_xy: Tuple[int, int],
        color: Tuple[int, int, int],
    ) -> None:
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
        x = int(anchor_xy[0])
        y = int(anchor_xy[1])
        if x + block_w >= iw:
            x = max(0, iw - block_w - 2)
        if y + block_h >= ih:
            y = max(0, ih - block_h - 2)
        x = max(0, x)
        y = max(0, y)

        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + block_w, y + block_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
        cv2.rectangle(img, (x, y), (x + block_w, y + block_h), color, 1)
        for i, line in enumerate(lines):
            cv2.putText(
                img,
                line,
                (x + pad, y + pad + 11 + i * line_h),
                font,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
