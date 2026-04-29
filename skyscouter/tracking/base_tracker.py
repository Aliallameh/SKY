"""
Base tracker interface.

A tracker takes per-frame detections and produces a stream of `Track`
objects with persistent IDs. All trackers in this codebase implement this
interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from ..perception.base_detector import Detection


@dataclass
class Track:
    """A persistent track. Updated each frame the track is alive."""
    track_id: int
    detection: Detection
    age_frames: int = 0          # frames since first seen
    hits: int = 1                # number of frames matched
    time_since_update: int = 0   # frames since last detection match
    confidence: float = 0.0
    min_confirmed_hits: int = 3
    status: str = "detected"
    matched_detection: bool = True
    source: str = "detector"
    flow_points: int = 0
    flow_quality: float = 0.0
    association_score: float = 0.0
    # Recent center positions in image space, for kinematic features
    center_history: Deque[Tuple[float, float, float]] = field(default_factory=deque)  # (cx, cy, t)

    @property
    def cx(self) -> float:
        return self.detection.cx

    @property
    def cy(self) -> float:
        return self.detection.cy

    @property
    def is_confirmed(self) -> bool:
        # Tracker logic decides confirmation; this is a convenience flag.
        return self.hits >= self.min_confirmed_hits


class BaseTracker(ABC):
    """Abstract tracker interface."""

    @abstractmethod
    def update(
        self,
        detections: List[Detection],
        frame_index: int,
        capture_time_s: float,
        image_bgr: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """Process detections for a single frame and return the live tracks."""
        ...

    @abstractmethod
    def reset(self) -> None:
        ...
