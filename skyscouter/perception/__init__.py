"""Detection — base interface and concrete backends."""

from .base_detector import BaseDetector, Detection
from .yolo_ultralytics import UltralyticsYoloDetector
from .stub_detector import StubDetector
from .factory import build_detector

__all__ = [
    "BaseDetector",
    "Detection",
    "UltralyticsYoloDetector",
    "StubDetector",
    "build_detector",
]
