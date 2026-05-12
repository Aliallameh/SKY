"""Converts visual guidance hints into rate-limited gimbal follow commands."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, TextIO

from ..schemas import GuidanceHint
from .siyi_client import SiyiGimbalClient


@dataclass(frozen=True)
class GimbalFollowCommand:
    frame_id: int
    timestamp_utc: str
    track_id: Optional[int]
    valid: bool
    yaw_cmd: int
    pitch_cmd: int
    reason: list[str]
    dry_run: bool


class GimbalFollowController:
    """Safe, bounded image-space servo for SIYI gimbal follow."""

    def __init__(
        self,
        *,
        cfg: Dict[str, Any],
        output_dir: Path,
        run_id: Optional[str],
    ):
        self._cfg = dict(cfg)
        self._enabled = bool(self._cfg.get("enabled", False))
        self._dry_run = bool(self._cfg.get("dry_run", True))
        self._allowed_lock_states = _normalized_set(
            self._cfg.get("allowed_lock_states", ["TRACKING", "LOCKED", "STRIKE_READY"])
        )
        self._min_confidence = float(self._cfg.get("min_confidence", 0.35))
        # command_hz <= 0 disables rate limiting (useful for tests / replay).
        command_hz = float(self._cfg.get("command_hz", 10.0))
        self._command_period_s = (1.0 / command_hz) if command_hz > 0.0 else 0.0
        self._deadband_px_x = float(self._cfg.get("deadband_px_x", 40.0))
        self._deadband_px_y = float(self._cfg.get("deadband_px_y", 30.0))
        self._kp_yaw = float(self._cfg.get("kp_yaw", 35.0))
        self._kp_pitch = float(self._cfg.get("kp_pitch", 28.0))
        self._max_yaw_cmd = int(abs(float(self._cfg.get("max_yaw_cmd", 25))))
        self._max_pitch_cmd = int(abs(float(self._cfg.get("max_pitch_cmd", 20))))
        self._invert_yaw = bool(self._cfg.get("invert_yaw", False))
        self._invert_pitch = bool(self._cfg.get("invert_pitch", False))
        self._stop_after_lost_s = float(self._cfg.get("stop_after_lost_s", 0.50))
        self._last_send_s = 0.0
        self._last_valid_s: Optional[float] = None
        self._last_cmd = (0, 0)
        self._run_id = run_id

        self._client: Optional[SiyiGimbalClient] = None
        if self._enabled and not self._dry_run:
            if str(self._cfg.get("backend", "siyi_udp")) != "siyi_udp":
                raise ValueError("Only gimbal_follow.backend=siyi_udp is supported")
            self._client = SiyiGimbalClient(
                host=str(self._cfg.get("host", "192.168.144.25")),
                port=int(self._cfg.get("port", 37260)),
                timeout_s=float(self._cfg.get("timeout_s", 0.20)),
            )

        self._log_fh: Optional[TextIO] = None
        if bool(self._cfg.get("log_jsonl", True)):
            log_path = output_dir / str(self._cfg.get("log_filename", "gimbal_follow_commands.jsonl"))
            self._log_fh = log_path.open("w", encoding="utf-8")
            self.log_path = log_path
        else:
            self.log_path = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    def consume(self, hint: Optional[GuidanceHint]) -> Optional[GimbalFollowCommand]:
        """Process one guidance hint and send/log a command at the configured rate."""

        if not self._enabled:
            return None

        now = time.monotonic()
        if self._command_period_s > 0.0 and now - self._last_send_s < self._command_period_s:
            return None
        self._last_send_s = now

        cmd = self._compute_command(hint, now)
        self._send(cmd)
        self._log(cmd, hint)
        return cmd

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.stop()
            except Exception:
                pass
            self._client.close()
        if self._log_fh is not None:
            self._log_fh.flush()
            self._log_fh.close()
            self._log_fh = None

    def _compute_command(
        self,
        hint: Optional[GuidanceHint],
        now_s: float,
    ) -> GimbalFollowCommand:
        reasons = self._invalid_reasons(hint)
        if reasons:
            lost_long_enough = (
                self._last_valid_s is None
                or now_s - self._last_valid_s >= self._stop_after_lost_s
            )
            yaw, pitch = (0, 0) if lost_long_enough else self._last_cmd
            return GimbalFollowCommand(
                frame_id=0 if hint is None else int(hint.frame_id),
                timestamp_utc=_utc_now() if hint is None else hint.timestamp_utc,
                track_id=None if hint is None else hint.track_id,
                valid=False,
                yaw_cmd=yaw,
                pitch_cmd=pitch,
                reason=reasons,
                dry_run=self._dry_run,
            )

        assert hint is not None
        self._last_valid_s = now_s
        pixel_error = hint.pixel_error_px or [0.0, 0.0]
        normalized_error = hint.normalized_error or [0.0, 0.0]

        yaw_cmd = 0 if abs(float(pixel_error[0])) < self._deadband_px_x else self._kp_yaw * float(normalized_error[0])
        pitch_cmd = 0 if abs(float(pixel_error[1])) < self._deadband_px_y else self._kp_pitch * float(normalized_error[1])

        if self._invert_yaw:
            yaw_cmd = -yaw_cmd
        if self._invert_pitch:
            pitch_cmd = -pitch_cmd

        yaw_i = _clamp_int(yaw_cmd, -self._max_yaw_cmd, self._max_yaw_cmd)
        pitch_i = _clamp_int(pitch_cmd, -self._max_pitch_cmd, self._max_pitch_cmd)
        self._last_cmd = (yaw_i, pitch_i)
        return GimbalFollowCommand(
            frame_id=int(hint.frame_id),
            timestamp_utc=hint.timestamp_utc,
            track_id=hint.track_id,
            valid=True,
            yaw_cmd=yaw_i,
            pitch_cmd=pitch_i,
            reason=["ok"],
            dry_run=self._dry_run,
        )

    def _invalid_reasons(self, hint: Optional[GuidanceHint]) -> list[str]:
        if hint is None:
            return ["no_guidance_hint"]
        reasons: list[str] = []
        if not hint.valid:
            reasons.append("guidance_hint_invalid")
        if hint.pixel_error_px is None or hint.normalized_error is None:
            reasons.append("missing_image_error")
        lock_state = str(hint.source_lock_state or "").upper().strip()
        if lock_state not in self._allowed_lock_states:
            reasons.append(f"lock_state_not_allowed:{hint.source_lock_state}")
        if hint.confidence is None or float(hint.confidence) < self._min_confidence:
            reasons.append("confidence_below_minimum")
        return reasons

    def _send(self, cmd: GimbalFollowCommand) -> None:
        if self._dry_run or self._client is None:
            return
        self._client.rotate(cmd.yaw_cmd, cmd.pitch_cmd)

    def _log(self, cmd: GimbalFollowCommand, hint: Optional[GuidanceHint]) -> None:
        if self._log_fh is None:
            return
        row = {
            "schema_version": "skyscout.gimbal_follow_command.v1",
            "run_id": self._run_id,
            "frame_id": cmd.frame_id,
            "timestamp_utc": cmd.timestamp_utc,
            "track_id": cmd.track_id,
            "valid": cmd.valid,
            "yaw_cmd": cmd.yaw_cmd,
            "pitch_cmd": cmd.pitch_cmd,
            "reason": cmd.reason,
            "dry_run": cmd.dry_run,
            "source_lock_state": None if hint is None else hint.source_lock_state,
            "confidence": None if hint is None else hint.confidence,
            "pixel_error_px": None if hint is None else hint.pixel_error_px,
            "normalized_error": None if hint is None else hint.normalized_error,
            "bearing_error_deg": None if hint is None else hint.filtered_bearing_error_deg,
            "elevation_error_deg": None if hint is None else hint.filtered_elevation_error_deg,
        }
        self._log_fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalized_set(values: Iterable[Any]) -> set[str]:
    return {str(v).upper().strip() for v in values}


def _clamp_int(value: float, lo: int, hi: int) -> int:
    if not math.isfinite(float(value)):
        return 0
    return max(lo, min(hi, int(round(float(value)))))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

