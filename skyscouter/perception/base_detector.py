"""
Base detector interface.

Every detector backend (Ultralytics YOLO, ONNX, RT-DETR, etc.) implements
this interface. The pipeline never imports a specific backend.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Detection:
    """A single detection in image coordinates."""
    x: float          # bbox top-left x (pixels)
    y: float          # bbox top-left y (pixels)
    w: float          # bbox width (pixels)
    h: float          # bbox height (pixels)
    confidence: float
    class_id: int
    class_label: str
    source: str = "detector"

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return self.w * self.h

    def as_xywh(self) -> List[float]:
        return [self.x, self.y, self.w, self.h]

    def as_xyxy(self) -> List[float]:
        return [self.x, self.y, self.x + self.w, self.y + self.h]


class BaseDetector(ABC):
    """Abstract detector interface."""

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR image and return a list of Detection."""
        ...

    @abstractmethod
    def warmup(self) -> None:
        """Run one inference on a dummy frame to JIT/compile and avoid first-frame
        latency spikes in the latency stack-up measurement."""
        ...

    @abstractmethod
    def get_model_version(self) -> str:
        """Stable identifier stamped into target_state.model_version."""
        ...

    @abstractmethod
    def get_input_size(self) -> int:
        ...
