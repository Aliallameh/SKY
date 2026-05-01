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


# Color palette (BGR) per lock state. Visible on sky and ground backgrounds.
STATE_COLORS = {
    LockState.NO_CUE.value:       (140, 140, 140),  # gray
    LockState.CUED.value:         (255, 200, 0),    # cyan-blue
    LockState.SEARCHING.value:    (0, 200, 255),    # orange
    LockState.ACQUIRED.value:     (0, 255, 255),    # yellow
    LockState.TRACKING.value:     (0, 255, 0),      # green
    LockState.LOCKED.value:       (0, 0, 255),      # red
    LockState.STRIKE_READY.value: (0, 0, 255),      # red (with thicker border)
    LockState.LOST.value:         (128, 0, 255),    # magenta
    LockState.ABORTED.value:      (60, 60, 60),     # dark gray
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
            label = f"id={tr.track_id} {tr.detection.class_label} {tr.detection.confidence:.2f}"
            cv2.putText(
                img, label, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
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
        strip_h = 56
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, strip_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

        color = STATE_COLORS.get(lock_state, (200, 200, 200))
        gv_text = "GUIDANCE_VALID" if guidance_valid else "guidance_invalid"
        gv_color = (0, 0, 255) if guidance_valid else (180, 180, 180)

        line1 = f"STATE: {lock_state}    {gv_text}"
        line2 = (
            f"frame={frame_index}  conf={confidence:.2f}  "
            f"lock_q={lock_quality:.2f}  latency={latency_ms:.1f} ms"
        )
        cv2.putText(img, line1, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.putText(img, line2, (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, gv_color, 1, cv2.LINE_AA)

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
                text = "bearing: n/a  yaw proposal: 0.0 deg/s"
            else:
                text = (
                    f"bearing={bearing:+.2f} deg  elev={elevation:+.2f} deg  "
                    f"yaw_prop={yaw_rate:+.1f} deg/s"
                )
            cv2.putText(
                img,
                text,
                (10, min(img.shape[0] - 10, 74)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
