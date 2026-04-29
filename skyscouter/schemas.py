"""
Message schemas for Skyscouter Phase 1.

These dataclasses define the contracts between subsystems:
  - CueState:     PRD §13 — input from radar partner / cue simulator
  - TargetState:  PRD §14 — output to flight controller / log

Field names and semantics match the PRD exactly. Do not rename without
bumping schema_version and updating the contract with Parham.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any


# ---------------- Enums (PRD §12) ----------------

class LockState(str, Enum):
    """Lock lifecycle states from PRD §12."""
    NO_CUE = "NO_CUE"
    CUED = "CUED"
    SEARCHING = "SEARCHING"
    ACQUIRED = "ACQUIRED"
    TRACKING = "TRACKING"
    LOCKED = "LOCKED"
    STRIKE_READY = "STRIKE_READY"
    LOST = "LOST"
    ABORTED = "ABORTED"


class MessageType(str, Enum):
    """Onboard target message types from PRD §14."""
    TARGET_STATE = "TARGET_STATE"
    HEARTBEAT = "HEARTBEAT"
    NO_TARGET = "NO_TARGET"
    SYSTEM_FAULT = "SYSTEM_FAULT"


class CoordinateType(str, Enum):
    """CUE_STATE coordinate type from PRD §13."""
    GEODETIC = "GEODETIC"
    LOCAL_ENU = "LOCAL_ENU"
    BEARING_RANGE = "BEARING_RANGE"
    PROPRIETARY_NORMALIZED = "PROPRIETARY_NORMALIZED"


class RangeSource(str, Enum):
    """Range source enum from PRD §14."""
    CUE_RADAR = "CUE_RADAR"
    EO_BBOX_GROWTH = "EO_BBOX_GROWTH"
    EO_SIZE_PRIOR = "EO_SIZE_PRIOR"
    NONE = "NONE"


class FaultFlag(str, Enum):
    """Fault flags from PRD §14."""
    VIBRATION_HIGH = "VIBRATION_HIGH"
    BLUR_HIGH = "BLUR_HIGH"
    CUE_STALE = "CUE_STALE"
    CAMERA_DROP = "CAMERA_DROP"
    MODEL_FAULT = "MODEL_FAULT"
    IR_FAULT = "IR_FAULT"
    INS_FAULT = "INS_FAULT"
    INHIBIT = "INHIBIT"
    CALIBRATION_DRIFT = "CALIBRATION_DRIFT"


# ---------------- CUE_STATE (PRD §13) ----------------

@dataclass
class CueState:
    """Input cue from radar partner or simulator. Field names mirror PRD §13."""
    schema_version: str = "skyscout.cue_state.v1"
    cue_id: str = ""
    timestamp_utc: str = ""
    source_partner: str = "SIM"
    coordinate_type: str = CoordinateType.LOCAL_ENU.value
    target_position: Optional[Dict[str, float]] = None
    target_velocity: Optional[Dict[str, float]] = None
    position_uncertainty: Optional[Dict[str, float]] = None
    velocity_uncertainty: Optional[Dict[str, float]] = None
    class_label: Optional[str] = None
    class_confidence: Optional[float] = None
    track_update_rate_hz: Optional[float] = None
    expiry_ms: int = 1500

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------- Onboard target state (PRD §14) ----------------

@dataclass
class TargetState:
    """Onboard perception output. Consumed by flight controller. PRD §14."""
    schema_version: str = "skyscout.onboard_target.v1"
    message_type: str = MessageType.TARGET_STATE.value
    timestamp_utc: str = ""
    cue_id: Optional[str] = None
    track_id: Optional[int] = None
    lock_state: str = LockState.NO_CUE.value
    guidance_valid: bool = False
    bbox_xywh: Optional[List[float]] = None
    image_size_wh: Optional[List[int]] = None
    line_of_sight_body: Optional[Dict[str, float]] = None
    range_estimate_m: Optional[float] = None
    range_source: str = RangeSource.NONE.value
    confidence: Optional[float] = None
    lock_quality: Optional[float] = None
    latency_ms: Optional[float] = None
    cue_age_ms: Optional[float] = None
    sensor_sources: List[str] = field(default_factory=lambda: ["EO_MONO"])
    model_version: str = ""
    calibration_id: str = ""
    fault_flags: List[str] = field(default_factory=list)

    # ---- Validation ----

    def enforce_safety_invariants(self) -> None:
        """
        Enforces the PRD Guidance Safety Rule:
          - guidance_valid=true requires lock_state in {LOCKED, STRIKE_READY}
          - any non-empty fault_flags forces guidance_valid=false
        Mutates self if the invariants would be violated.
        """
        if self.fault_flags:
            self.guidance_valid = False
        if self.lock_state not in (LockState.LOCKED.value, LockState.STRIKE_READY.value):
            self.guidance_valid = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
