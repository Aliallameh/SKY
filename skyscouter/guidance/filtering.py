"""Small deterministic filters and predictors for visual guidance."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, Optional, Tuple


@dataclass
class AngleEmaFilter:
    """Exponential moving average for bearing/elevation errors in degrees."""
    alpha: float
    _bearing_deg: Optional[float] = None
    _elevation_deg: Optional[float] = None

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")

    def update(self, bearing_deg: float, elevation_deg: float) -> Tuple[float, float]:
        if not math.isfinite(bearing_deg) or not math.isfinite(elevation_deg):
            raise ValueError("angles must be finite")
        if self._bearing_deg is None or self._elevation_deg is None:
            self._bearing_deg = float(bearing_deg)
            self._elevation_deg = float(elevation_deg)
        else:
            self._bearing_deg = self.alpha * float(bearing_deg) + (1.0 - self.alpha) * self._bearing_deg
            self._elevation_deg = self.alpha * float(elevation_deg) + (1.0 - self.alpha) * self._elevation_deg
        return self._bearing_deg, self._elevation_deg

    def reset(self) -> None:
        self._bearing_deg = None
        self._elevation_deg = None


@dataclass
class CenterPredictor:
    """Short-horizon center predictor from recent image-space track centers."""
    lead_time_s: float
    max_prediction_px: float
    history_len: int = 8
    _history: Deque[Tuple[float, float, float]] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if self.lead_time_s < 0.0:
            raise ValueError("lead_time_s must be >= 0")
        if self.max_prediction_px < 0.0:
            raise ValueError("max_prediction_px must be >= 0")
        if self.history_len < 2:
            raise ValueError("history_len must be >= 2")

    def update(
        self,
        center_px: Tuple[float, float],
        timestamp_s: float,
        external_history: Optional[Iterable[Tuple[float, float, float]]] = None,
    ) -> Optional[Tuple[float, float]]:
        if external_history is not None:
            self._history.clear()
            for point in external_history:
                self._append(point)
        self._append((center_px[0], center_px[1], timestamp_s))
        if len(self._history) < 2 or self.lead_time_s <= 0.0:
            return None

        x0, y0, t0 = self._history[0]
        x1, y1, t1 = self._history[-1]
        dt = t1 - t0
        if dt <= 0.0:
            return None
        vx = (x1 - x0) / dt
        vy = (y1 - y0) / dt
        dx = vx * self.lead_time_s
        dy = vy * self.lead_time_s
        mag = (dx * dx + dy * dy) ** 0.5
        if self.max_prediction_px > 0.0 and mag > self.max_prediction_px:
            scale = self.max_prediction_px / mag
            dx *= scale
            dy *= scale
        return (x1 + dx, y1 + dy)

    def reset(self) -> None:
        self._history.clear()

    def _append(self, point: Tuple[float, float, float]) -> None:
        x, y, t = point
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(t)):
            return
        self._history.append((float(x), float(y), float(t)))
        while len(self._history) > self.history_len:
            self._history.popleft()
