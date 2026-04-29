"""
Ultralytics YOLO detector backend.

This is the Phase 1 default. The model is loaded from the path configured in
detector.weights — typically a small COCO-pretrained model during bench
replay, replaced with a fine-tuned anti-UAV model for credible Phase 1
acceptance.

Honest note: COCO does not contain a "drone" class. During bench replay we
keep "airplane" (4) and "bird" (14) plus a fallback "any high-confidence
small detection" to surface drone-shaped candidates. This is a starting point,
not an acceptance configuration. Fine-tuning is required.
"""
from __future__ import annotations

from typing import List, Optional, Set

import numpy as np

from .base_detector import BaseDetector, Detection


class UltralyticsYoloDetector(BaseDetector):
    """Wraps the ultralytics YOLO API."""

    def __init__(
        self,
        weights: str,
        input_size: int = 640,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        keep_class_ids: Optional[List[int]] = None,
        fallback_keep_high_conf: bool = False,
        fallback_high_conf_threshold: float = 0.35,
        device: str = "auto",
        model_version: str = "ultralytics_yolo",
    ):
        # Lazy import so unit tests of unrelated modules don't require the
        # heavy ultralytics dependency.
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:
            raise ImportError(
                "ultralytics is required for the YOLO Ultralytics backend. "
                "Install with: pip install ultralytics"
            ) from e

        self._YOLO = YOLO
        self._weights_path = weights
        self._input_size = int(input_size)
        self._conf = float(confidence_threshold)
        self._iou = float(iou_threshold)
        self._keep_class_ids: Optional[Set[int]] = (
            set(keep_class_ids) if keep_class_ids else None
        )
        self._fallback_keep_high_conf = bool(fallback_keep_high_conf)
        self._fallback_threshold = float(fallback_high_conf_threshold)
        self._device = device if device != "auto" else None  # let ultralytics decide
        self._model_version = model_version

        # Load model — fail loudly if weights missing.
        self._model = self._YOLO(weights)
        self._class_names = self._model.names  # dict: id -> name

    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        # ultralytics accepts BGR ndarray directly (it converts internally)
        results = self._model.predict(
            source=image_bgr,
            imgsz=self._input_size,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        # Pull out tensors and convert to CPU numpy
        xyxy = result.boxes.xyxy.cpu().numpy()    # (N, 4)
        conf = result.boxes.conf.cpu().numpy()    # (N,)
        cls = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

        out: List[Detection] = []
        for i in range(len(xyxy)):
            class_id = int(cls[i])
            confidence = float(conf[i])

            keep = False
            if self._keep_class_ids is None:
                keep = True
            elif class_id in self._keep_class_ids:
                keep = True
            elif self._fallback_keep_high_conf and confidence >= self._fallback_threshold:
                # Fallback: surface high-confidence detections of any class to
                # catch drones (which COCO doesn't have a class for). This is
                # explicitly a bench-replay shim, not a production policy.
                keep = True

            if not keep:
                continue

            x1, y1, x2, y2 = xyxy[i].tolist()
            label = self._class_names.get(class_id, str(class_id))
            out.append(Detection(
                x=float(x1),
                y=float(y1),
                w=float(x2 - x1),
                h=float(y2 - y1),
                confidence=confidence,
                class_id=class_id,
                class_label=str(label),
                source="yolo_ultralytics",
            ))
        return out

    def warmup(self) -> None:
        dummy = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
        try:
            self._model.predict(
                source=dummy,
                imgsz=self._input_size,
                conf=self._conf,
                iou=self._iou,
                device=self._device,
                verbose=False,
            )
        except Exception:
            # Warmup is best-effort. Don't fail pipeline if it errors.
            pass

    def get_model_version(self) -> str:
        return self._model_version

    def get_input_size(self) -> int:
        return self._input_size
