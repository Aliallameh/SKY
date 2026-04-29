"""
Simple IoU-based tracker.

A small, auditable, dependency-free tracker that implements the BaseTracker
interface. Sufficient for Sprint 0 bench replay. Can be replaced with full
ByteTrack later via the BaseTracker interface — the rest of the pipeline
does not change.

Algorithm:
  1. For each new detection, find the live track with the highest IoU above
     `match_threshold`. Greedy matching, sorted by descending IoU.
  2. Unmatched detections start tentative tracks.
  3. Tentative tracks are confirmed once they have `min_track_length` hits.
  4. Tracks unmatched for `track_buffer_frames` are killed.
  5. Before matching, optionally warp existing track centers with the
     EgoMotionCompensator (no-op until IMU is wired in).

Why not import a third-party tracker for Sprint 0:
  - Auditable: we can read every line.
  - Zero extra dependencies.
  - Same interface as a "real" ByteTrack drop-in for Sprint 1.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from ..perception.base_detector import Detection
from .base_tracker import BaseTracker, Track
from .ego_motion import EgoMotionCompensator, IdentityEgoMotion


def _iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.as_xyxy()
    bx1, by1, bx2, by2 = b.as_xyxy()
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center_distance(a: Detection, b: Detection) -> float:
    return ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5


class SimpleIoUTracker(BaseTracker):

    HISTORY_LEN = 64  # max points retained per track for kinematic features

    def __init__(
        self,
        match_threshold: float = 0.3,
        center_match_threshold_px: float = 80.0,
        track_buffer_frames: int = 30,
        min_track_length: int = 3,
        ego_motion: Optional[EgoMotionCompensator] = None,
    ):
        if not 0.0 <= match_threshold <= 1.0:
            raise ValueError("match_threshold must be in [0, 1]")
        if track_buffer_frames < 0:
            raise ValueError("track_buffer_frames must be >= 0")
        if min_track_length < 1:
            raise ValueError("min_track_length must be >= 1")

        self._match_threshold = match_threshold
        self._center_match_threshold_px = float(center_match_threshold_px)
        self._track_buffer_frames = track_buffer_frames
        self._min_track_length = min_track_length
        self._ego = ego_motion or IdentityEgoMotion()

        self._tracks: Dict[int, Track] = {}
        self._next_id: int = 1
        self._last_capture_time_s: Optional[float] = None

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._last_capture_time_s = None

    # ---- internal helpers ----

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _ego_compensate(self, dt_seconds: float) -> None:
        """Warp every live track's last detection center per ego-motion."""
        if dt_seconds <= 0:
            return
        for tr in self._tracks.values():
            cx, cy = self._ego.warp_point(tr.detection.cx, tr.detection.cy, dt_seconds)
            # We update the predicted center in the detection so IoU matching
            # uses the warped position. Width/height are unchanged.
            d = tr.detection
            new_x = cx - d.w / 2.0
            new_y = cy - d.h / 2.0
            tr.detection = Detection(
                x=new_x, y=new_y, w=d.w, h=d.h,
                confidence=d.confidence,
                class_id=d.class_id,
                class_label=d.class_label,
            )

    # ---- main update ----

    def update(
        self,
        detections: List[Detection],
        frame_index: int,
        capture_time_s: float,
        image_bgr=None,
    ) -> List[Track]:
        # Compute dt and run ego-motion compensation on existing tracks
        dt = 0.0
        if self._last_capture_time_s is not None:
            dt = max(0.0, capture_time_s - self._last_capture_time_s)
        self._ego_compensate(dt)
        self._last_capture_time_s = capture_time_s

        # Build candidate (track_id, detection_index, iou) pairs
        track_ids = list(self._tracks.keys())
        candidates: List[Tuple[float, int, int]] = []  # (match_score, track_id, det_idx)
        for tid in track_ids:
            tr = self._tracks[tid]
            for di, det in enumerate(detections):
                iou_val = _iou(tr.detection, det)
                if iou_val >= self._match_threshold:
                    candidates.append((1.0 + iou_val, tid, di))
                    continue

                dist = _center_distance(tr.detection, det)
                if dist <= self._center_match_threshold_px:
                    prev_area = max(1.0, tr.detection.area)
                    det_area = max(1.0, det.area)
                    area_ratio = min(prev_area, det_area) / max(prev_area, det_area)
                    if area_ratio >= 0.15:
                        distance_score = 1.0 - (dist / max(1.0, self._center_match_threshold_px))
                        candidates.append((distance_score * area_ratio, tid, di))

        candidates.sort(reverse=True)  # highest IoU first

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for iou_val, tid, di in candidates:
            if tid in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(tid)
            matched_dets.add(di)
            tr = self._tracks[tid]
            tr.detection = detections[di]
            tr.age_frames += 1
            tr.hits += 1
            tr.time_since_update = 0
            tr.confidence = detections[di].confidence
            tr.center_history.append((tr.cx, tr.cy, capture_time_s))
            while len(tr.center_history) > self.HISTORY_LEN:
                tr.center_history.popleft()

        # Age unmatched tracks
        for tid in track_ids:
            if tid in matched_tracks:
                continue
            tr = self._tracks[tid]
            tr.age_frames += 1
            tr.time_since_update += 1

        # Kill stale tracks
        dead = [
            tid for tid, tr in self._tracks.items()
            if tr.time_since_update > self._track_buffer_frames
        ]
        for tid in dead:
            del self._tracks[tid]

        # Spawn new tracks from unmatched detections
        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            tid = self._new_id()
            tr = Track(
                track_id=tid,
                detection=det,
                age_frames=1,
                hits=1,
                time_since_update=0,
                confidence=det.confidence,
                min_confirmed_hits=self._min_track_length,
            )
            tr.center_history.append((det.cx, det.cy, capture_time_s))
            self._tracks[tid] = tr

        # Return live tracks (confirmed and tentative both — the caller decides)
        return list(self._tracks.values())

    @property
    def min_track_length(self) -> int:
        return self._min_track_length
