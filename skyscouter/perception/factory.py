"""Factory: build a BaseDetector instance from a config dict."""
from __future__ import annotations

from typing import Any, Dict

from .base_detector import BaseDetector
from .stub_detector import StubDetector


def build_detector(detector_cfg: Dict[str, Any]) -> BaseDetector:
    backend = detector_cfg.get("backend", "yolo_ultralytics")

    if backend == "stub":
        return StubDetector(
            model_version=detector_cfg.get("model_version", "stub_v1"),
            input_size=int(detector_cfg.get("input_size", 640)),
        )

    if backend == "yolo_ultralytics":
        from .yolo_ultralytics import UltralyticsYoloDetector
        return UltralyticsYoloDetector(
            weights=detector_cfg["weights"],
            input_size=int(detector_cfg.get("input_size", 640)),
            confidence_threshold=float(detector_cfg.get("confidence_threshold", 0.25)),
            iou_threshold=float(detector_cfg.get("iou_threshold", 0.45)),
            keep_class_ids=detector_cfg.get("keep_class_ids"),
            fallback_keep_high_conf=bool(detector_cfg.get("fallback_keep_high_conf", False)),
            fallback_high_conf_threshold=float(detector_cfg.get("fallback_high_conf_threshold", 0.35)),
            device=detector_cfg.get("device", "auto"),
            model_version=detector_cfg.get("model_version", "ultralytics_yolo"),
        )

    if backend == "airborne_cv":
        from .airborne_cv_detector import AirborneCvDetector
        return AirborneCvDetector(
            input_size=int(detector_cfg.get("input_size", 640)),
            sky_roi_y_max_fraction=float(detector_cfg.get("sky_roi_y_max_fraction", 0.62)),
            local_background_kernel=int(detector_cfg.get("local_background_kernel", 31)),
            dark_contrast_threshold=float(detector_cfg.get("dark_contrast_threshold", 24.0)),
            min_area_px=int(detector_cfg.get("min_area_px", 20)),
            max_area_px=int(detector_cfg.get("max_area_px", 1200)),
            min_width_px=int(detector_cfg.get("min_width_px", 4)),
            max_width_px=int(detector_cfg.get("max_width_px", 90)),
            min_height_px=int(detector_cfg.get("min_height_px", 4)),
            max_height_px=int(detector_cfg.get("max_height_px", 60)),
            min_aspect_ratio=float(detector_cfg.get("min_aspect_ratio", 0.35)),
            max_aspect_ratio=float(detector_cfg.get("max_aspect_ratio", 8.0)),
            max_detections=int(detector_cfg.get("max_detections", 40)),
            model_version=detector_cfg.get("model_version", "airborne_cv_v1"),
        )

    raise ValueError(
        f"Unknown detector backend: {backend!r}. "
        f"Supported: yolo_ultralytics, airborne_cv, stub. "
        f"To add a new backend, implement BaseDetector and register it here."
    )
