"""Transport-neutral mocked bridge for GuidanceHint consumption."""
from __future__ import annotations

import math
from typing import Iterable, Optional, Set

from ..schemas import BridgeProposal, GuidanceHint, LockState


class MockGuidanceBridge:
    """
    Converts GuidanceHint into auditable bridge proposal rows.

    This bridge never sends MAVLink, never opens a network transport, and never
    marks anything valid for actuation. `valid_for_transport` only means the row
    is acceptable for a future mocked transport/simulator consumer.
    """

    def __init__(
        self,
        *,
        run_id: Optional[str],
        calibration_id: str,
        calibration_reviewed: bool,
        require_reviewed_calibration: bool,
        allowed_lock_states: Iterable[str],
        require_guidance_hint_valid: bool,
        max_abs_yaw_rate_deg_s: float,
    ):
        if max_abs_yaw_rate_deg_s < 0.0:
            raise ValueError("mock_bridge.max_abs_yaw_rate_deg_s must be >= 0")
        self._run_id = run_id
        self._calibration_id = str(calibration_id)
        self._calibration_reviewed = bool(calibration_reviewed)
        self._require_reviewed_calibration = bool(require_reviewed_calibration)
        self._allowed_lock_states: Set[str] = {
            str(s).upper().strip() for s in allowed_lock_states
        }
        if not self._allowed_lock_states:
            raise ValueError("mock_bridge.allowed_lock_states must not be empty")
        self._require_guidance_hint_valid = bool(require_guidance_hint_valid)
        self._max_abs_yaw_rate_deg_s = float(max_abs_yaw_rate_deg_s)

    def consume(self, hint: Optional[GuidanceHint]) -> BridgeProposal:
        if hint is None:
            return self._suppressed_empty("missing_guidance_hint")

        reasons = []
        source_lock_state = str(hint.source_lock_state or "").upper().strip()
        yaw_rate = float(hint.yaw_rate_cmd_deg_s)

        if self._require_reviewed_calibration and not self._calibration_reviewed:
            reasons.append("calibration_not_reviewed")
        if source_lock_state not in self._allowed_lock_states:
            reasons.append(f"lock_state_not_allowed:{hint.source_lock_state}")
        if self._require_guidance_hint_valid and not hint.valid:
            reasons.append("guidance_hint_not_valid")
        if "fault_flags_active" in set(hint.reason):
            reasons.append("fault_flags_active")
        if not math.isfinite(yaw_rate):
            reasons.append("yaw_rate_not_finite")
        elif abs(yaw_rate) > self._max_abs_yaw_rate_deg_s:
            reasons.append("command_exceeds_bridge_limit")

        valid = not reasons
        proposal = BridgeProposal(
            frame_id=hint.frame_id,
            timestamp_s=hint.timestamp_s,
            timestamp_utc=hint.timestamp_utc,
            run_id=hint.run_id or self._run_id,
            track_id=hint.track_id,
            source_guidance_schema_version=hint.schema_version,
            source_lock_state=hint.source_lock_state,
            calibration_id=self._calibration_id,
            calibration_reviewed=self._calibration_reviewed,
            yaw_rate_cmd_deg_s=yaw_rate if valid else 0.0,
            pitch_rate_cmd_deg_s=hint.pitch_rate_cmd_deg_s if valid else None,
            valid_for_transport=valid,
            reason=["ok"] if valid else reasons,
            source_guidance_valid=hint.valid,
            source_guidance_reason=list(hint.reason),
        )
        proposal.enforce_safety_invariants()
        return proposal

    def _suppressed_empty(self, reason: str) -> BridgeProposal:
        proposal = BridgeProposal(
            run_id=self._run_id,
            calibration_id=self._calibration_id,
            calibration_reviewed=self._calibration_reviewed,
            valid_for_transport=False,
            reason=[reason],
        )
        proposal.enforce_safety_invariants()
        return proposal
