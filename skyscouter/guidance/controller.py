"""Bounded yaw-rate command proposal controller."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class YawRateController:
    """
    P-controller for yaw-rate proposals.

    Positive command means yaw right, matching the positive bearing convention.
    The output is a proposal for logging/simulation only; it is not sent to a
    vehicle by this package.
    """
    enabled: bool
    mode: str
    kp_yaw: float
    deadband_deg: float
    max_yaw_rate_deg_s: float
    max_delta_yaw_rate_deg_s: Optional[float] = None
    _last_cmd_deg_s: float = 0.0

    def __post_init__(self) -> None:
        if self.deadband_deg < 0.0:
            raise ValueError("deadband_deg must be >= 0")
        if self.max_yaw_rate_deg_s < 0.0:
            raise ValueError("max_yaw_rate_deg_s must be >= 0")
        if self.max_delta_yaw_rate_deg_s is not None and self.max_delta_yaw_rate_deg_s < 0.0:
            raise ValueError("max_delta_yaw_rate_deg_s must be >= 0")

    def compute(self, filtered_bearing_error_deg: float, *, valid: bool) -> float:
        if not self.enabled or not valid:
            self._last_cmd_deg_s = 0.0
            return 0.0
        if self.mode != "yaw_p":
            raise ValueError(f"Unsupported controller mode: {self.mode!r}")
        if abs(filtered_bearing_error_deg) <= self.deadband_deg:
            raw = 0.0
        else:
            raw = self.kp_yaw * filtered_bearing_error_deg
        saturated = max(-self.max_yaw_rate_deg_s, min(self.max_yaw_rate_deg_s, raw))
        if self.max_delta_yaw_rate_deg_s is not None:
            delta = saturated - self._last_cmd_deg_s
            max_delta = self.max_delta_yaw_rate_deg_s
            delta = max(-max_delta, min(max_delta, delta))
            saturated = self._last_cmd_deg_s + delta
        self._last_cmd_deg_s = saturated
        return float(saturated)

    def reset(self) -> None:
        self._last_cmd_deg_s = 0.0
