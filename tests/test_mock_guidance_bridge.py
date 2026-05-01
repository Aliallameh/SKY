from __future__ import annotations

import pytest

from skyscouter.bridge.mock_guidance_bridge import MockGuidanceBridge
from skyscouter.schemas import GuidanceHint


def _hint(**overrides):
    base = {
        "frame_id": 42,
        "timestamp_s": 1.4,
        "timestamp_utc": "2026-05-01T00:00:01.400Z",
        "run_id": "run_bridge",
        "track_id": 7,
        "valid": True,
        "valid_for_logging": True,
        "valid_for_actuation": False,
        "reason": ["ok"],
        "source_lock_state": "LOCKED",
        "guidance_valid_from_lock_state": True,
        "yaw_rate_cmd_deg_s": 12.0,
        "pitch_rate_cmd_deg_s": None,
    }
    base.update(overrides)
    return GuidanceHint(**base)


def _bridge(**overrides):
    base = {
        "run_id": "run_bridge",
        "calibration_id": "CAL_REVIEWED",
        "calibration_reviewed": True,
        "require_reviewed_calibration": True,
        "allowed_lock_states": ["LOCKED", "STRIKE_READY"],
        "require_guidance_hint_valid": True,
        "max_abs_yaw_rate_deg_s": 45.0,
    }
    base.update(overrides)
    return MockGuidanceBridge(**base)


def test_reviewed_calibration_and_valid_hint_emits_transport_valid_proposal():
    proposal = _bridge().consume(_hint())
    assert proposal.schema_version == "skyscout.mock_bridge_proposal.v1"
    assert proposal.valid_for_transport is True
    assert proposal.yaw_rate_cmd_deg_s == pytest.approx(12.0)
    assert proposal.pitch_rate_cmd_deg_s is None
    assert proposal.reason == ["ok"]


def test_unreviewed_calibration_suppresses_proposal():
    proposal = _bridge(calibration_reviewed=False).consume(_hint())
    assert proposal.valid_for_transport is False
    assert proposal.yaw_rate_cmd_deg_s == 0.0
    assert "calibration_not_reviewed" in proposal.reason


def test_invalid_guidance_suppresses_proposal_and_zeroes_command():
    hint = _hint(valid=False, reason=["target_state_guidance_not_valid"], yaw_rate_cmd_deg_s=20.0)
    proposal = _bridge().consume(hint)
    assert proposal.valid_for_transport is False
    assert proposal.yaw_rate_cmd_deg_s == 0.0
    assert "guidance_hint_not_valid" in proposal.reason


def test_unsafe_lock_state_suppresses_proposal():
    proposal = _bridge().consume(_hint(source_lock_state="TRACKING"))
    assert proposal.valid_for_transport is False
    assert proposal.yaw_rate_cmd_deg_s == 0.0
    assert "lock_state_not_allowed:TRACKING" in proposal.reason


def test_fault_flags_reason_suppresses_even_if_validity_requirement_is_relaxed():
    bridge = _bridge(require_guidance_hint_valid=False)
    hint = _hint(valid=False, reason=["fault_flags_active"], yaw_rate_cmd_deg_s=5.0)
    proposal = bridge.consume(hint)
    assert proposal.valid_for_transport is False
    assert proposal.yaw_rate_cmd_deg_s == 0.0
    assert "fault_flags_active" in proposal.reason


def test_yaw_command_above_bridge_limit_is_rejected_not_clamped():
    proposal = _bridge(max_abs_yaw_rate_deg_s=10.0).consume(_hint(yaw_rate_cmd_deg_s=12.0))
    assert proposal.valid_for_transport is False
    assert proposal.yaw_rate_cmd_deg_s == 0.0
    assert "command_exceeds_bridge_limit" in proposal.reason
