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
    """Sparse optical-flow bbox propagation for short detector gaps."""

    def __init__(self, min_points: int = 8):
        self._min_points = int(min_points)
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_det: Optional[Detection] = None
        self._prev_points: Optional[np.ndarray] = None
        self._last_points: Optional[np.ndarray] = None

    def reset(self, gray: np.ndarray, d: Detection) -> None:
        self._prev_gray = gray.copy()
        self._prev_det = d
        self._prev_points = self._points_in_detection(gray, d)
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

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            gray,
            self._prev_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if next_pts is None or status is None:
            return None, 0.0, 0

        keep = status.reshape(-1).astype(bool)
        old = self._prev_points.reshape(-1, 2)[keep]
        new = next_pts.reshape(-1, 2)[keep]
        if len(new) < self._min_points:
            return None, 0.0, int(len(new))
        self._last_points = new.astype(np.float32).reshape(-1, 1, 2)

        delta = np.median(new - old, axis=0)
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
        residual = np.linalg.norm((new - old) - delta[None, :], axis=1)
        residual_med = float(np.median(residual)) if len(residual) else 999.0
        point_score = min(1.0, len(new) / 35.0)
        residual_score = max(0.0, 1.0 - residual_med / 12.0)
        quality = float(np.clip(0.65 * point_score + 0.35 * residual_score, 0.0, 1.0))
        return moved, quality, int(len(new))

    def _points_in_detection(self, gray: np.ndarray, d: Detection) -> np.ndarray:
        mask = np.zeros_like(gray)
        h_img, w_img = gray.shape[:2]
        x1 = int(np.clip(round(d.x), 0, max(0, w_img - 2)))
        y1 = int(np.clip(round(d.y), 0, max(0, h_img - 2)))
        x2 = int(np.clip(round(d.x + d.w), x1 + 1, max(1, w_img - 1)))
        y2 = int(np.clip(round(d.y + d.h), y1 + 1, max(1, h_img - 1)))
        pad_x = max(1, int(0.05 * max(1, x2 - x1)))
        pad_y = max(1, int(0.05 * max(1, y2 - y1)))
        mask[y1 + pad_y:y2 - pad_y, x1 + pad_x:x2 - pad_x] = 255
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=80,
            qualityLevel=0.01,
            minDistance=4,
            blockSize=5,
            mask=mask,
        )
        if pts is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return pts.astype(np.float32)


class SingleTargetKalmanLKTracker(BaseTracker):
    """One active airborne target with Kalman, appearance, and LK bridge."""

    def __init__(
        self,
        min_track_length: int = 3,
        track_buffer_frames: int = 30,
        lk_min_points: int = 8,
        reacquisition_radius_px: float = 160.0,
        max_primary_switches_per_second: float = 1.0,
    ):
        self._min_track_length = int(min_track_length)
        self._max_age = int(track_buffer_frames)
        self._lk_min_points = int(lk_min_points)
        self._reacq_radius = float(reacquisition_radius_px)
        self._max_switches_per_second = float(max_primary_switches_per_second)
        self._appearance = AppearanceModel()
        self._flow = LKFlowPropagator(min_points=lk_min_points)
        self._next_id = 1
        self._track_id: Optional[int] = None
        self._x: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None
        self._track: Optional[Track] = None
        self._age_since_update = 0
        self._hits = 0
        self._last_time: Optional[float] = None
        self._last_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.__init__(
            min_track_length=self._min_track_length,
            track_buffer_frames=self._max_age,
            lk_min_points=self._lk_min_points,
            reacquisition_radius_px=self._reacq_radius,
            max_primary_switches_per_second=self._max_switches_per_second,
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
        flow_det = None
        flow_quality = 0.0
        flow_points = 0
        if gray is not None:
            flow_det, flow_quality, flow_points = self._flow.predict(gray)
            if flow_det is not None:
                flow_det.confidence = float(np.clip(0.18 + 0.48 * flow_quality, 0.05, 0.70))

        best_det, score = self._associate(predicted, detections, image_bgr)
        rescue_det = self._rescue_candidate(predicted, best_det, detections)
        if rescue_det is not None and (
            best_det is None
            or _is_boundary_clipped(predicted)
            or rescue_det.confidence >= best_det.confidence + 0.08
        ):
            best_det = rescue_det
            score = max(score, 0.65 + 0.25 * rescue_det.confidence)
        matched = best_det is not None and score >= 0.18
        if not matched and flow_det is not None and flow_quality >= 0.25:
            best_det = flow_det
            score = flow_quality * 0.55
            matched = True

        if matched and best_det is not None:
            self._age_since_update = 0 if best_det.source != "lk_optical_flow" else self._age_since_update + 1
            self._hits += 1
            self._measurement_update(_bbox_to_z(best_det), best_det.confidence)
            det = _z_to_detection(
                self._x[:4],
                confidence=best_det.confidence,
                class_label=best_det.class_label,
                source=best_det.source,
                frame_wh=frame_wh,
            )
            status = "flow" if best_det.source == "lk_optical_flow" else "detected"
            if gray is not None:
                if best_det.source == "lk_optical_flow":
                    self._flow.accept_prediction(gray, det)
                else:
                    self._flow.reset(gray, det)
            if image_bgr is not None and best_det.source != "lk_optical_flow":
                self._appearance.update(image_bgr, det, best_det.confidence)
        else:
            self._age_since_update += 1
            det = predicted
            det.confidence = float(max(0.01, det.confidence * (0.75 ** self._age_since_update)))
            status = "predicted"

        self._last_gray = gray
        if self._age_since_update > self._max_age:
            return []

        tr = self._make_track(
            det,
            capture_time_s,
            status=status,
            matched_detection=matched,
            flow_points=flow_points,
            flow_quality=flow_quality,
            association_score=score,
        )
        self._track = tr
        return [tr]

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
