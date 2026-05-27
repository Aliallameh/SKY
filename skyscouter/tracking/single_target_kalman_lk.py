"""Single-target Kalman + LK optical-flow tracker for airborne lock.

This tracker is adapted from the user's monocular-object-localization project,
but rewritten to satisfy Skyscouter's BaseTracker interface and airborne target
semantics. It treats detector output as proposals. The active target identity is
owned by a constant-velocity bbox Kalman filter, lightweight appearance memory,
and short-gap Lucas-Kanade propagation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..perception.base_detector import Detection
from .base_tracker import BaseTracker, Track
from .inter_frame_lk import (
    InterFrameLKThread,
    find_features_in_region,
    lk_step_with_backcheck,
    _MIN_TRACKED_POINTS,
)


def _bbox_to_z(d: Detection) -> np.ndarray:
    return np.array([d.cx, d.cy, max(1.0, d.w), max(1.0, d.h)], dtype=np.float64)


def _z_to_detection(
    z: np.ndarray,
    *,
    confidence: float,
    class_label: str,
    source: str,
    frame_wh: Optional[Tuple[int, int]] = None,
) -> Detection:
    cx, cy, w, h = [float(v) for v in z[:4]]
    w = max(1.0, w)
    h = max(1.0, h)
    x = cx - 0.5 * w
    y = cy - 0.5 * h
    if frame_wh is not None:
        fw, fh = frame_wh
        x = float(np.clip(x, 0, max(0, fw - 1)))
        y = float(np.clip(y, 0, max(0, fh - 1)))
        w = float(np.clip(w, 1, max(1, fw - x)))
        h = float(np.clip(h, 1, max(1, fh - y)))
    return Detection(
        x=x,
        y=y,
        w=w,
        h=h,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        class_id=0,
        class_label=class_label,
        source=source,
    )


def _iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.as_xyxy()
    bx1, by1, bx2, by2 = b.as_xyxy()
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    denom = a.area + b.area - inter
    return 0.0 if denom <= 0 else float(inter / denom)


def _center_distance(a: Detection, b: Detection) -> float:
    return float(((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5)


def _is_boundary_clipped(d: Detection) -> bool:
    return d.x <= 1.0 or d.y <= 1.0


def _is_airborne_label(label: str) -> bool:
    return str(label).lower().strip() in {
        "airplane",
        "aircraft",
        "drone",
        "uas",
        "airborne_candidate",
        "drone_candidate",
        "uas_candidate",
    }


class AppearanceModel:
    """Tiny HSV-HS histogram identity memory."""

    def __init__(self, bins_h: int = 24, bins_s: int = 16):
        self._bins_h = bins_h
        self._bins_s = bins_s
        self._hist: Optional[np.ndarray] = None
        self._samples = 0

    @property
    def ready(self) -> bool:
        return self._hist is not None and self._samples >= 3

    def update(self, frame_bgr: np.ndarray, d: Detection, conf: float) -> None:
        hist = self._extract(frame_bgr, d)
        if hist is None:
            return
        alpha = float(np.clip(0.08 + 0.20 * conf, 0.08, 0.28))
        if self._hist is None:
            self._hist = hist
        else:
            self._hist = (1.0 - alpha) * self._hist + alpha * hist
            total = float(self._hist.sum())
            if total > 0:
                self._hist = self._hist / total
        self._samples += 1

    def similarity(self, frame_bgr: Optional[np.ndarray], d: Detection) -> float:
        if frame_bgr is None or self._hist is None:
            return 0.5
        hist = self._extract(frame_bgr, d)
        if hist is None:
            return 0.0
        dist = cv2.compareHist(
            self._hist.astype(np.float32),
            hist.astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA,
        )
        return float(np.clip(1.0 - dist, 0.0, 1.0))

    def _extract(self, frame_bgr: np.ndarray, d: Detection) -> Optional[np.ndarray]:
        h_img, w_img = frame_bgr.shape[:2]
        x1 = int(np.clip(round(d.x), 0, max(0, w_img - 2)))
        y1 = int(np.clip(round(d.y), 0, max(0, h_img - 2)))
        x2 = int(np.clip(round(d.x + d.w), x1 + 1, max(1, w_img - 1)))
        y2 = int(np.clip(round(d.y + d.h), y1 + 1, max(1, h_img - 1)))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([0, 5, 10], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        hist = cv2.calcHist([hsv], [0, 1], mask, [self._bins_h, self._bins_s], [0, 180, 0, 256])
        total = float(hist.sum())
        if total <= 1e-6:
            return None
        return (hist / total).astype(np.float32)


class LKFlowPropagator:
    """Sparse optical-flow bbox propagation used as YOLO-rate fallback.

    This runs once per YOLO frame (5-7 fps) and bridges the gap between
    YOLO detections when the inter-frame LK thread is not wired up (e.g.
    offline video playback).  For live RTSP streams, InterFrameLKThread
    handles this at full camera FPS (30 fps) and this propagator is only
    consulted as a secondary check.

    Key improvements over original:
    - Uses shared find_features_in_region (expanded 35% search area)
    - Forward-backward consistency check via lk_step_with_backcheck
    - Lower min_points default (4 instead of 8) for tiny targets
    - maxLevel=4 instead of 3
    """

    def __init__(self, min_points: int = 4):
        self._min_points = max(int(min_points), _MIN_TRACKED_POINTS)
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_det: Optional[Detection] = None
        self._prev_points: Optional[np.ndarray] = None
        self._last_points: Optional[np.ndarray] = None

    def reset(self, gray: np.ndarray, d: Detection) -> None:
        self._prev_gray = gray.copy()
        self._prev_det = d
        self._prev_points = find_features_in_region(gray, d)
        self._last_points = None

    def accept_prediction(self, gray: np.ndarray, d: Detection) -> None:
        if self._last_points is None or len(self._last_points) < self._min_points:
            return
        self._prev_gray = gray.copy()
        self._prev_det = d
        self._prev_points = self._last_points.astype(np.float32).reshape(-1, 1, 2)

    def predict(self, gray: np.ndarray) -> Tuple[Optional[Detection], float, int]:
        self._last_points = None
        if self._prev_gray is None or self._prev_det is None or self._prev_points is None:
            return None, 0.0, 0
        if len(self._prev_points) < self._min_points:
            return None, 0.0, int(len(self._prev_points))

        old_good, new_good, delta = lk_step_with_backcheck(
            self._prev_gray, gray, self._prev_points
        )

        if delta is None or new_good is None or len(new_good) < self._min_points:
            n = len(new_good) if new_good is not None else 0
            return None, 0.0, int(n)

        self._last_points = new_good.astype(np.float32).reshape(-1, 1, 2)

        moved = Detection(
            x=float(self._prev_det.x + delta[0]),
            y=float(self._prev_det.y + delta[1]),
            w=self._prev_det.w,
            h=self._prev_det.h,
            confidence=self._prev_det.confidence,
            class_id=self._prev_det.class_id,
            class_label=self._prev_det.class_label,
            source="lk_optical_flow",
        )
        residual = np.linalg.norm((new_good - old_good) - delta[None, :], axis=1)
        residual_med = float(np.median(residual)) if len(residual) else 999.0
        point_score = min(1.0, len(new_good) / 20.0)   # scaled to 20 pts instead of 35
        residual_score = max(0.0, 1.0 - residual_med / 8.0)
        quality = float(np.clip(0.65 * point_score + 0.35 * residual_score, 0.0, 1.0))
        return moved, quality, int(len(new_good))


class SingleTargetKalmanLKTracker(BaseTracker):
    """One active airborne target with Kalman, appearance, and LK bridge."""

    def __init__(
        self,
        min_track_length: int = 3,
        track_buffer_frames: int = 30,
        lk_min_points: int = 4,
        reacquisition_radius_px: float = 160.0,
        max_primary_switches_per_second: float = 1.0,
        max_prediction_only_frames: int = 6,
        min_prediction_confidence: float = 0.05,
    ):
        self._min_track_length = int(min_track_length)
        self._max_age = int(track_buffer_frames)
        self._lk_min_points = int(lk_min_points)
        self._reacq_radius = float(reacquisition_radius_px)
        self._max_switches_per_second = float(max_primary_switches_per_second)
        self._max_prediction_only_frames = int(max_prediction_only_frames)
        self._min_prediction_confidence = float(min_prediction_confidence)
        self._appearance = AppearanceModel()
        self._flow = LKFlowPropagator(min_points=lk_min_points)

        # Inter-frame LK thread — runs LK at full camera FPS (30 fps) so the
        # tracked position stays within 5-10 px between YOLO frames instead of
        # drifting 30-50 px at 5-7 fps.  Activated via feed_inter_frame().
        self._inter_lk = InterFrameLKThread()
        self._inter_lk.start()

        self._next_id = 1
        self._track_id: Optional[int] = None
        self._x: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None
        self._track: Optional[Track] = None
        self._age_since_update = 0
        self._hits = 0
        self._last_time: Optional[float] = None
        self._last_gray: Optional[np.ndarray] = None

    def feed_inter_frame(self, frame_bgr: np.ndarray) -> None:
        """Feed every raw camera frame to the 30fps inter-frame LK thread.

        Call this from the camera reader thread on EVERY decoded frame,
        not just the frames that reach the YOLO detector.  When wired up
        via RtspStreamSource.register_frame_observer(), the LK thread
        tracks the target at camera FPS between YOLO updates.
        """
        self._inter_lk.feed_frame(frame_bgr)

    def reset(self) -> None:
        self._inter_lk.stop()
        self.__init__(
            min_track_length=self._min_track_length,
            track_buffer_frames=self._max_age,
            lk_min_points=self._lk_min_points,
            reacquisition_radius_px=self._reacq_radius,
            max_primary_switches_per_second=self._max_switches_per_second,
            max_prediction_only_frames=self._max_prediction_only_frames,
            min_prediction_confidence=self._min_prediction_confidence,
        )

    def update(
        self,
        detections: List[Detection],
        frame_index: int,
        capture_time_s: float,
        image_bgr: Optional[np.ndarray] = None,
    ) -> List[Track]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr is not None else None
        frame_wh = None if image_bgr is None else (image_bgr.shape[1], image_bgr.shape[0])
        dt = 1.0
        if self._last_time is not None:
            dt = max(1e-3, capture_time_s - self._last_time)
        self._last_time = capture_time_s

        if self._x is None:
            best = self._choose_initial_detection(detections)
            if best is None:
                self._last_gray = gray
                return []
            return [self._init_track(best, capture_time_s, gray, image_bgr)]

        predicted = self._predict(dt, frame_wh)

        # ------------------------------------------------------------------
        # Inter-frame LK (30 fps thread) — primary bridge between YOLO frames
        # ------------------------------------------------------------------
        inter_det = None
        inter_quality = 0.0
        inter_points = 0
        inter_result = self._inter_lk.get_latest()
        if inter_result is not None:
            icx, icy, iw, ih, iq, ipts = inter_result
            if iq >= 0.20:   # only use if the thread tracked with reasonable quality
                inter_det = Detection(
                    x=float(icx - iw / 2.0),
                    y=float(icy - ih / 2.0),
                    w=float(iw),
                    h=float(ih),
                    confidence=float(np.clip(0.15 + 0.55 * iq, 0.05, 0.72)),
                    class_id=self._track.detection.class_id if self._track else 0,
                    class_label=self._track.detection.class_label if self._track else "drone",
                    source="lk_inter_frame",
                )
                inter_quality = iq
                inter_points = ipts

        # ------------------------------------------------------------------
        # Per-YOLO-frame LK fallback (for offline video / when thread unavailable)
        # ------------------------------------------------------------------
        flow_det = None
        flow_quality = 0.0
        flow_points = 0
        if gray is not None and inter_det is None:
            # Only run the per-frame LK when the inter-frame thread has no result
            flow_det, flow_quality, flow_points = self._flow.predict(gray)
            if flow_det is not None:
                flow_det.confidence = float(np.clip(0.18 + 0.48 * flow_quality, 0.05, 0.70))

        # Prefer inter-frame LK over per-YOLO LK when both available
        best_lk_det    = inter_det    if inter_det    is not None else flow_det
        best_lk_qual   = inter_quality if inter_det   is not None else flow_quality
        best_lk_points = inter_points  if inter_det   is not None else flow_points

        # ------------------------------------------------------------------
        # YOLO association
        # ------------------------------------------------------------------
        best_det, score = self._associate(predicted, detections, image_bgr)
        rescue_det = self._rescue_candidate(predicted, best_det, detections)
        if rescue_det is not None and (
            best_det is None
            or _is_boundary_clipped(predicted)
            or rescue_det.confidence >= best_det.confidence + 0.08
        ):
            best_det = rescue_det
            score = max(score, 0.65 + 0.25 * rescue_det.confidence)
        reentry_det = self._stale_reentry_candidate(predicted, best_det, detections)
        if reentry_det is not None:
            best_det = reentry_det
            score = max(score, 0.24 + 0.35 * reentry_det.confidence)
        matched = best_det is not None and score >= 0.18

        # Fall back to LK (inter-frame preferred) when YOLO has no match
        lk_source_tags = {"lk_optical_flow", "lk_inter_frame"}
        if not matched and best_lk_det is not None and best_lk_qual >= 0.20:
            best_det = best_lk_det
            score = best_lk_qual * 0.55
            matched = True
            flow_points = best_lk_points
            flow_quality = best_lk_qual

        if matched and best_det is not None:
            is_lk_source = best_det.source in lk_source_tags
            self._age_since_update = 0 if not is_lk_source else self._age_since_update + 1
            self._hits += 1
            self._measurement_update(_bbox_to_z(best_det), best_det.confidence)
            det = _z_to_detection(
                self._x[:4],
                confidence=best_det.confidence,
                class_label=best_det.class_label,
                source=best_det.source,
                frame_wh=frame_wh,
            )
            status = "flow" if is_lk_source else "detected"
            if gray is not None:
                if is_lk_source:
                    self._flow.accept_prediction(gray, det)
                else:
                    # Confirmed YOLO hit — resync BOTH the per-frame LK and the
                    # inter-frame LK thread to the freshly updated Kalman position
                    self._flow.reset(gray, det)
                    self._inter_lk.reset_from_yolo(gray, det)
            if image_bgr is not None and not is_lk_source:
                self._appearance.update(image_bgr, det, best_det.confidence)
        else:
            self._age_since_update += 1
            det = predicted
            det.confidence = float(max(0.01, det.confidence * (0.75 ** self._age_since_update)))
            status = "predicted"

        self._last_gray = gray
        if (
            status in {"flow", "predicted"}
            and self._age_since_update > self._max_prediction_only_frames
        ) or (
            status == "predicted" and det.confidence < self._min_prediction_confidence
        ):
            # A Kalman-only box after several missed detector frames is not an
            # observation, and LK flow is still only a bridge between detector
            # hits. Emitting either for too long makes the overlay look like a
            # false detection and can visually chase empty sky after a target
            # exits or reverses direction. Clear state so the next real detector
            # hit starts fresh instead of dragging stale motion.
            self._clear_track()
            return []

        if self._age_since_update > self._max_age:
            # Track stale beyond the buffer. Don't keep dead-reckoning the
            # Kalman state across the gap: with a constant-velocity model,
            # the prediction drifts arbitrarily far from the drone's actual
            # position once it reappears, and _associate() then rejects the
            # fresh detector hit because it's outside the (predicted-)reacq
            # radius. Clear internal state so the next update() takes the
            # _choose_initial_detection() branch and re-acquires from the
            # strongest available detection. This is the detector→track
            # re-init pattern used by the Anti-UAV reference implementation.
            self._clear_track()
            return []

        # Report the best LK stats available for diagnostics
        diag_flow_points = inter_points if inter_points > 0 else flow_points
        diag_flow_quality = inter_quality if inter_points > 0 else flow_quality

        tr = self._make_track(
            det,
            capture_time_s,
            status=status,
            matched_detection=matched,
            flow_points=diag_flow_points,
            flow_quality=diag_flow_quality,
            association_score=score,
        )
        self._track = tr
        return [tr]

    def _clear_track(self) -> None:
        self._x = None
        self._P = None
        self._track = None
        self._track_id = None
        self._hits = 0
        self._age_since_update = 0
        self._appearance = AppearanceModel()
        self._flow = LKFlowPropagator(min_points=self._lk_min_points)
        self._inter_lk.clear()  # stop the 30fps thread from tracking stale position

    @property
    def min_track_length(self) -> int:
        return self._min_track_length

    def _choose_initial_detection(self, detections: List[Detection]) -> Optional[Detection]:
        if not detections:
            return None
        return max(detections, key=lambda d: (d.confidence, d.area))

    def _init_track(
        self,
        d: Detection,
        capture_time_s: float,
        gray: Optional[np.ndarray],
        image_bgr: Optional[np.ndarray],
    ) -> Track:
        self._track_id = self._next_id
        self._next_id += 1
        self._x = np.zeros(8, dtype=np.float64)
        self._x[:4] = _bbox_to_z(d)
        self._P = np.diag([30.0, 30.0, 20.0, 20.0, 120.0, 120.0, 40.0, 40.0]).astype(np.float64)
        self._age_since_update = 0
        self._hits = 1
        if gray is not None:
            self._flow.reset(gray, d)
        if image_bgr is not None:
            self._appearance.update(image_bgr, d, d.confidence)
        tr = self._make_track(d, capture_time_s, status="detected", matched_detection=True)
        self._track = tr
        return tr

    def _predict(self, dt_seconds: float, frame_wh: Optional[Tuple[int, int]]) -> Detection:
        if self._x is None or self._P is None or self._track is None:
            raise RuntimeError("Tracker has not been initialized")
        dt_frames = max(1e-3, dt_seconds * 30.0)
        F = np.eye(8, dtype=np.float64)
        F[0, 4] = dt_frames
        F[1, 5] = dt_frames
        F[2, 6] = dt_frames
        F[3, 7] = dt_frames
        Q = np.diag([8.0, 8.0, 3.0, 3.0, 2.0, 2.0, 0.9, 0.9]).astype(np.float64)
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q
        self._x[2] = max(4.0, self._x[2])
        self._x[3] = max(4.0, self._x[3])
        return _z_to_detection(
            self._x[:4],
            confidence=self._track.confidence,
            class_label=self._track.detection.class_label,
            source="kalman_prediction",
            frame_wh=frame_wh,
        )

    def _measurement_update(self, z: np.ndarray, conf: float) -> None:
        if self._x is None or self._P is None:
            raise RuntimeError("Tracker has not been initialized")
        H = np.zeros((4, 8), dtype=np.float64)
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0
        sigma = 22.0 - 16.0 * float(np.clip(conf, 0.0, 1.0))
        R = np.diag([sigma * sigma, sigma * sigma, 0.8 * sigma * sigma, 0.8 * sigma * sigma])
        innovation = z - H @ self._x
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)
        self._x = self._x + K @ innovation
        I = np.eye(8, dtype=np.float64)
        self._P = (I - K @ H) @ self._P @ (I - K @ H).T + K @ R @ K.T
        self._x[2] = max(4.0, self._x[2])
        self._x[3] = max(4.0, self._x[3])

    def _associate(
        self,
        predicted: Detection,
        detections: List[Detection],
        image_bgr: Optional[np.ndarray],
    ) -> Tuple[Optional[Detection], float]:
        if not detections:
            return None, 0.0
        pred_diag = max(1.0, (predicted.w ** 2 + predicted.h ** 2) ** 0.5)
        best_det: Optional[Detection] = None
        best_score = -1.0
        for det in detections:
            dist = _center_distance(predicted, det)
            if self._age_since_update <= self._max_age and dist > self._reacq_radius + pred_diag:
                continue
            iou = _iou(predicted, det)
            center_score = max(0.0, 1.0 - dist / max(1.0, self._reacq_radius + pred_diag))
            area_ratio = min(predicted.area, det.area) / max(1.0, max(predicted.area, det.area))
            area_score = max(0.0, 1.0 - abs(np.log(max(area_ratio, 1e-3))))
            appearance_score = self._appearance.similarity(image_bgr, det)
            score = (
                0.26 * iou
                + 0.30 * center_score
                + 0.16 * area_score
                + 0.18 * appearance_score
                + 0.10 * det.confidence
            )
            if self._appearance.ready and appearance_score < 0.12 and iou < 0.10:
                score *= 0.35
            if score > best_score:
                best_score = score
                best_det = det
        return best_det, float(max(0.0, best_score))

    def _rescue_candidate(
        self,
        predicted: Detection,
        current_match: Optional[Detection],
        detections: List[Detection],
    ) -> Optional[Detection]:
        if not detections:
            return None
        strongest = max(detections, key=lambda d: (d.confidence, d.area))
        if strongest.confidence < 0.74:
            return None
        if _center_distance(predicted, strongest) <= self._reacq_radius:
            return strongest
        active_boundary = _is_boundary_clipped(predicted) or (
            current_match is not None and _is_boundary_clipped(current_match)
        )
        if not active_boundary:
            return None
        # A clipped edge track is often tree/building clutter. If the detector
        # sees a much stronger high-sky candidate, let the single target tracker
        # rescue to it instead of remaining guidance-valid on the frame edge.
        reference_y = current_match.cy if current_match is not None else predicted.cy
        if strongest.cy >= reference_y:
            return None
        return strongest

    def _stale_reentry_candidate(
        self,
        predicted: Detection,
        current_match: Optional[Detection],
        detections: List[Detection],
    ) -> Optional[Detection]:
        if not detections:
            return None

        candidates = [
            d for d in detections
            if d.confidence >= 0.20 and _is_airborne_label(d.class_label)
        ]
        if not candidates:
            return None
        if self._age_since_update <= 0 and current_match is None and len(candidates) != 1:
            return None

        pred_diag = max(1.0, (predicted.w ** 2 + predicted.h ** 2) ** 0.5)
        max_reentry_dist = max(self._reacq_radius * 1.25, self._reacq_radius + 2.0 * pred_diag)
        if current_match is not None and current_match in candidates:
            candidates = [current_match]

        scored = []
        for det in candidates:
            dist = _center_distance(predicted, det)
            if dist > max_reentry_dist:
                continue
            # When the current state is already prediction-only, a fresh
            # detector measurement is more trustworthy than continuing to
            # dead-reckon the Kalman box, even if the blended association score
            # is weak because the old box shape/appearance no longer overlaps.
            scored.append((det.confidence, -dist, det.area, det))
        if not scored:
            return None
        scored.sort(reverse=True, key=lambda item: item[:3])
        return scored[0][3]

    def _make_track(
        self,
        d: Detection,
        capture_time_s: float,
        *,
        status: str,
        matched_detection: bool,
        flow_points: int = 0,
        flow_quality: float = 0.0,
        association_score: float = 0.0,
    ) -> Track:
        if self._track_id is None:
            raise RuntimeError("Track ID missing")
        if self._track is None:
            history = None
            age_frames = 1
        else:
            history = self._track.center_history
            age_frames = self._track.age_frames + 1
        tr = Track(
            track_id=self._track_id,
            detection=d,
            age_frames=age_frames,
            hits=self._hits,
            time_since_update=self._age_since_update,
            confidence=d.confidence,
            min_confirmed_hits=self._min_track_length,
            status=status,
            matched_detection=matched_detection,
            source=d.source,
            flow_points=flow_points,
            flow_quality=flow_quality,
            association_score=association_score,
        )
        if history is not None:
            tr.center_history.extend(history)
        tr.center_history.append((tr.cx, tr.cy, capture_time_s))
        while len(tr.center_history) > 64:
            tr.center_history.popleft()
        return tr
