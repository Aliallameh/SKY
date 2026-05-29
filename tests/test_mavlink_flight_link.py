from __future__ import annotations

import pytest

from skyscouter.flight.mavlink_link import MavlinkFlightLink
from skyscouter.schemas import GuidanceHint


def _link(tmp_path, **overrides) -> MavlinkFlightLink:
    cfg = {
        "enabled": True,
        "dry_run": True,
        "send_hz": 30.0,
        "max_heading_deg": 15.0,
        "yaw_deadband_deg": 1.0,
        "allowed_lock_states": ["TRACKING", "LOCKED", "STRIKE_READY"],
        "log_jsonl": False,
    }
    cfg.update(overrides)
    return MavlinkFlightLink(cfg=cfg, output_dir=tmp_path, run_id="test")


def _hint(**overrides) -> GuidanceHint:
    base = {
        "frame_id": 1,
        "timestamp_s": 1.0,
        "timestamp_utc": "2026-05-29T00:00:01.000Z",
        "track_id": 7,
        "valid": True,
        "source_lock_state": "TRACKING",
        "filtered_bearing_error_deg": 20.0,
        "bearing_error_deg": 20.0,
        "yaw_rate_cmd_deg_s": 3.5,
    }
    base.update(overrides)
    return GuidanceHint(**base)


def test_flight_link_uses_guidance_pid_output_as_yaw_correction(tmp_path):
    link = _link(tmp_path)
    try:
        link.consume(_hint(filtered_bearing_error_deg=20.0, yaw_rate_cmd_deg_s=3.5))
        with link._target_lock:
            assert link._target_yaw_heading_deg == pytest.approx(3.5)
    finally:
        link.close()


def test_flight_link_clamps_pid_output_to_max_heading(tmp_path):
    link = _link(tmp_path, max_heading_deg=10.0)
    try:
        link.consume(_hint(yaw_rate_cmd_deg_s=30.0))
        with link._target_lock:
            assert link._target_yaw_heading_deg == pytest.approx(10.0)
    finally:
        link.close()


def test_flight_link_deadbands_pid_output(tmp_path):
    link = _link(tmp_path, yaw_deadband_deg=1.0)
    try:
        link.consume(_hint(yaw_rate_cmd_deg_s=0.5))
        with link._target_lock:
            assert link._target_yaw_heading_deg == pytest.approx(0.0)
    finally:
        link.close()
