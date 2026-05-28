"""Bounded yaw-rate command proposal controller."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class YawRateController:
    """
    PID controller for yaw-rate proposals.

    Positive command means yaw right, matching the positive bearing convention.
    The output is a proposal for logging/simulation only; it is not sent to a
    vehicle by this package.
    """
    enabled: bool
    mode: str
    kp_yaw: float
    deadband_deg: float
    max_yaw_rate_deg_s: float
    ki_yaw: float = 0.0
    kd_yaw: float = 0.0
    max_delta_yaw_rate_deg_s: Optional[float] = None
    _last_cmd_deg_s: float = 0.0
    _integral_error_deg_s: float = 0.0
    _last_error_deg: Optional[float] = None
    _last_timestamp_s: Optional[float] = None

    def __post_init__(self) -> None:
        if self.deadband_deg < 0.0:
            raise ValueError("deadband_deg must be >= 0")
        if self.max_yaw_rate_deg_s < 0.0:
            raise ValueError("max_yaw_rate_deg_s must be >= 0")
        if self.max_delta_yaw_rate_deg_s is not None and self.max_delta_yaw_rate_deg_s < 0.0:
            raise ValueError("max_delta_yaw_rate_deg_s must be >= 0")

    def compute(
        self,
        filtered_bearing_error_deg: float,
        *,
        valid: bool,
        timestamp_s: Optional[float] = None,
    ) -> float:
        if not self.enabled or not valid:
            self.reset()
            return 0.0
        if self.mode not in {"yaw_p", "yaw_pid"}:
            raise ValueError(f"Unsupported controller mode: {self.mode!r}")

        dt_s = self._dt_s(timestamp_s)
        error_deg = float(filtered_bearing_error_deg)
        if abs(filtered_bearing_error_deg) <= self.deadband_deg:
            raw = 0.0
            self._integral_error_deg_s = 0.0
            self._last_error_deg = error_deg
        else:
            if dt_s is not None:
                self._integral_error_deg_s += error_deg * dt_s
            derivative_error_deg_s = 0.0
            if dt_s is not None and self._last_error_deg is not None:
                derivative_error_deg_s = (error_deg - self._last_error_deg) / dt_s
            raw = self.kp_yaw * error_deg
            if self.mode == "yaw_pid":
                raw += (
                    self.ki_yaw * self._integral_error_deg_s
                    + self.kd_yaw * derivative_error_deg_s
                )
            self._last_error_deg = error_deg
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
        self._integral_error_deg_s = 0.0
        self._last_error_deg = None
        self._last_timestamp_s = None

    def _dt_s(self, timestamp_s: Optional[float]) -> Optional[float]:
        if timestamp_s is None:
            return None
        timestamp = float(timestamp_s)
        if self._last_timestamp_s is None:
            self._last_timestamp_s = timestamp
            return None
        dt_s = timestamp - self._last_timestamp_s
        self._last_timestamp_s = timestamp
        if dt_s <= 0.0:
            return None
        return dt_s
