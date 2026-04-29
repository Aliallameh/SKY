"""
Classical airborne target candidate detector.

This backend is for the PRD Phase 1 reacquisition problem: small, dark aerial
targets against sky/cloud backgrounds. It is not a semantic drone classifier.
It produces `airborne_candidate` detections that can be promoted to lock only
through temporal stability gates in the lock state machine.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .base_detector import BaseDetector, Detection


class AirborneCvDetector(BaseDetector):
    """Detect compact high-contrast dark objects in the sky ROI."""

    def __init__(
        self,
        input_size: int = 640,
        sky_roi_y_max_fraction: float = 0.62,
        local_background_kernel: int = 31,
        dark_contrast_threshold: float = 24.0,
        min_area_px: int = 20,
        max_area_px: int = 1200,
        min_width_px: int = 4,
        max_width_px: int = 90,
        min_height_px: int = 4,
        max_height_px: int = 60,
        min_aspect_ratio: float = 0.35,
        max_aspect_ratio: float = 8.0,
        max_detections: int = 40,
        model_version: str = "airborne_cv_v1",
    ):
        if not 0.0 < sky_roi_y_max_fraction <= 1.0:
            raise ValueError("sky_roi_y_max_fraction must be in (0, 1]")
        if local_background_kernel < 3:
            raise ValueError("local_background_kernel must be >= 3")
        if local_background_kernel % 2 == 0:
            local_background_kernel += 1

        self._input_size = int(input_size)
        self._sky_roi_y_max_fraction = float(sky_roi_y_max_fraction)
        self._kernel = int(local_background_kernel)
        self._threshold = float(dark_contrast_threshold)
        self._min_area = int(min_area_px)
        self._max_area = int(max_area_px)
        self._min_w = int(min_width_px)
        self._max_w = int(max_width_px)
        self._min_h = int(min_height_px)
        self._max_h = int(max_height_px)
        self._min_ar = float(min_aspect_ratio)
        self._max_ar = float(max_aspect_ratio)
        self._max_detections = int(max_detections)
        self._model_version = model_version

    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        h, w = image_bgr.shape[:2]
        roi_h = max(1, min(h, int(round(h * self._sky_roi_y_max_fraction))))
        gray = cv2.cvtColor(image_bgr[:roi_h], cv2.COLOR_BGR2GRAY)

        background = cv2.GaussianBlur(gray, (self._kernel, self._kernel), 0)
        dark_response = cv2.subtract(background, gray)
        mask = (dark_response >= self._threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))

        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        candidates = []
        frame_diag = float((w * w + h * h) ** 0.5)

        for idx in range(1, count):
            x, y, bw, bh, area = [int(v) for v in stats[idx]]
            if area < self._min_area or area > self._max_area:
                continue
            if bw < self._min_w or bw > self._max_w or bh < self._min_h or bh > self._max_h:
                continue
            aspect = bw / max(1.0, float(bh))
            if aspect < self._min_ar or aspect > self._max_ar:
                continue

            patch = dark_response[y:y + bh, x:x + bw]
            mean_contrast = float(patch.mean()) if patch.size else 0.0
            max_contrast = float(patch.max()) if patch.size else 0.0
            fill = area / max(1.0, float(bw * bh))
            size_score = min(1.0, ((bw * bw + bh * bh) ** 0.5) / max(1.0, 0.035 * frame_diag))
            contrast_score = min(1.0, max_contrast / max(self._threshold * 2.0, 1.0))
            center_y = y + bh / 2.0
            vertical_score = max(0.0, min(1.0, 1.0 - (center_y / max(1.0, roi_h)) ** 2))
            confidence = max(
                0.05,
                min(
                    0.99,
                    0.45 * contrast_score
                    + 0.20 * fill
                    + 0.20 * size_score
                    + 0.15 * vertical_score,
                ),
            )

            pad = max(2, int(round(max(bw, bh) * 0.25)))
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(w, x + bw + pad)
            y1 = min(h, y + bh + pad)
            candidates.append((
                confidence,
                Detection(
                    x=float(x0),
                    y=float(y0),
                    w=float(x1 - x0),
                    h=float(y1 - y0),
                    confidence=float(confidence),
                    class_id=0,
                    class_label="airborne_candidate",
                    source="airborne_cv",
                ),
                mean_contrast,
            ))

        candidates.sort(key=lambda item: (item[0], item[2]), reverse=True)
        return [det for _, det, _ in candidates[:self._max_detections]]

    def warmup(self) -> None:
        dummy = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
        self.detect(dummy)

    def get_model_version(self) -> str:
        return self._model_version

    def get_input_size(self) -> int:
        return self._input_size
