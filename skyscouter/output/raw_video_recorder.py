"""Raw frame video recorder for live-camera deployment runs."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class RawVideoRecorder:
    """Writes unannotated BGR frames to an MP4 review artifact."""

    def __init__(self, output_path: str, width: int, height: int, fps: float = 30.0):
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self._path), fourcc, max(1.0, fps), (width, height)
        )
        if not self._writer.isOpened():
            raise IOError(f"Could not open raw video writer for {self._path}")
        self._width = int(width)
        self._height = int(height)

    def write(self, image_bgr: np.ndarray) -> None:
        if image_bgr.shape[1] != self._width or image_bgr.shape[0] != self._height:
            image_bgr = cv2.resize(image_bgr, (self._width, self._height))
        self._writer.write(image_bgr)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None  # type: ignore[assignment]
