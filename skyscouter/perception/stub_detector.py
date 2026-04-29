"""
Stub detector — emits no detections.

Used in unit tests and pipeline smoke tests to verify that the lock state
machine and publisher behave correctly when no detections are present (PRD §12:
NO_CUE / SEARCHING / ABORTED transitions).

This is NOT a fallback for missing model weights. If weights are missing in
production, the pipeline must fail loudly — a silent fallback to "nothing
detected" is the kind of cheating the engineering plan forbids.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .base_detector import BaseDetector, Detection


class StubDetector(BaseDetector):
    """Always returns an empty list of detections. For tests only."""

    def __init__(self, model_version: str = "stub_v1", input_size: int = 640):
        self._model_version = model_version
        self._input_size = input_size

    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        return []

    def warmup(self) -> None:
        pass

    def get_model_version(self) -> str:
        return self._model_version

    def get_input_size(self) -> int:
        return self._input_size
