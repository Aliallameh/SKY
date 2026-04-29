"""
Lock state machine tests.

These exercise every transition in PRD §12 plus every STRIKE_READY gate
from §12.1. The state machine is the safety-critical part of the perception
system: a regression here is an unsafe-guidance bug.
"""
from __future__ import annotations

import pytest

from skyscouter.lock.state_machine import LockStateMachine, StrikeReadyConfig
from skyscouter.schemas import LockState, FaultFlag


# ---------- helpers ----------

def _ok_input(now_s, **overrides):
    """Default 'good track' inputs that, frame after frame, lead to LOCKED -> STRIKE_READY."""
    base = dict(
        now_s=now_s,
        is_track_alive=True,
        is_track_confirmed=True,
        class_label="drone",
        class_confidence=0.9,
        bbox_frame_fraction=0.20,           # safely above the 0.05 default
        bbox_center_norm_distance=0.10,     # well centered
        cue_active=False,
        cue_age_seconds=None,
        active_fault_flags=[],
        lock_quality=0.9,
    )
    base.update(overrides)
    return base


def _make_sm(**overrides):
    cfg = dict(
        acquired_to_tracking_frames=2,
        tracking_to_locked_frames=3,
        min_class_confidence=0.30,
        min_lock_quality=0.50,
        lost_to_aborted_seconds=0.5,
        strike_ready_cfg=StrikeReadyConfig(
            min_locked_duration_seconds=0.10,
            min_bbox_frame_fraction=0.05,
            bbox_center_window=0.5,
            max_cue_age_seconds=None,
        ),
        operator_inhibit=False,
    )
    cfg.update(overrides)
    return LockStateMachine(**cfg)


# ---------- happy path ----------

def test_initial_state_is_no_cue():
    sm = _make_sm()
    assert sm.state == LockState.NO_CUE


def test_full_lifecycle_to_strike_ready():
    sm = _make_sm()

    # Frame 1: candidate appears (no cue) -> SEARCHING
    d = sm.on_track_update(**_ok_input(0.00))
    assert d.lock_state == LockState.SEARCHING
    assert d.guidance_valid is False

    # Frame 2: ACQUIRED
    d = sm.on_track_update(**_ok_input(0.01))
    assert d.lock_state == LockState.ACQUIRED

    # Frame 3-4: stay ACQUIRED until acquired_to_tracking_frames hits
    for i in range(2):
        d = sm.on_track_update(**_ok_input(0.02 + i * 0.01))

    assert d.lock_state == LockState.TRACKING

    # Stay in TRACKING until tracking_to_locked_frames is met
    for i in range(3):
        d = sm.on_track_update(**_ok_input(0.05 + i * 0.01))

    assert d.lock_state == LockState.LOCKED
    assert d.guidance_valid is True  # LOCKED with no faults

    # Need to satisfy continuous lock duration before STRIKE_READY
    # min_locked_duration_seconds=0.10
    d = sm.on_track_update(**_ok_input(0.20))  # > 0.10 since first locked
    assert d.lock_state == LockState.STRIKE_READY
    assert d.guidance_valid is True


# ---------- STRIKE_READY gates ----------

def test_strike_ready_blocked_by_small_bbox():
    sm = _make_sm()
    # Drive to LOCKED first
    inputs = [_ok_input(t) for t in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]]
    for inp in inputs:
        d = sm.on_track_update(**inp)
    assert d.lock_state == LockState.LOCKED
    # Try STRIKE_READY but with too-small bbox
    d = sm.on_track_update(**_ok_input(0.30, bbox_frame_fraction=0.02))
    assert d.lock_state == LockState.LOCKED  # blocked


def test_strike_ready_blocked_by_off_center_bbox():
    sm = _make_sm()
    inputs = [_ok_input(t) for t in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]]
    for inp in inputs:
        d = sm.on_track_update(**inp)
    assert d.lock_state == LockState.LOCKED
    # Off-center: distance > window
    d = sm.on_track_update(**_ok_input(0.30, bbox_center_norm_distance=0.95))
    assert d.lock_state == LockState.LOCKED


def test_strike_ready_revoked_within_one_frame():
    """PRD FR-LOCK-004: any failed gate revokes STRIKE_READY immediately."""
    sm = _make_sm()
    # Reach STRIKE_READY
    for t in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
        sm.on_track_update(**_ok_input(t))
    d = sm.on_track_update(**_ok_input(0.20))
    assert d.lock_state == LockState.STRIKE_READY

    # Next frame: bbox shrinks below threshold
    d = sm.on_track_update(**_ok_input(0.21, bbox_frame_fraction=0.01))
    assert d.lock_state == LockState.LOCKED


# ---------- LOST and ABORTED ----------

def test_lost_recovery_returns_to_tracking():
    sm = _make_sm()
    # Reach LOCKED
    for t in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
        sm.on_track_update(**_ok_input(t))
    # Lose track
    d = sm.on_track_update(**_ok_input(0.07, is_track_alive=False))
    assert d.lock_state == LockState.LOST
    # Recover
    d = sm.on_track_update(**_ok_input(0.10))
    assert d.lock_state == LockState.TRACKING


def test_lost_timeout_to_aborted():
    sm = _make_sm(lost_to_aborted_seconds=0.5)
    # Reach LOCKED
    for t in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
        sm.on_track_update(**_ok_input(t))
    # Lose
    d = sm.on_track_update(**_ok_input(0.10, is_track_alive=False))
    assert d.lock_state == LockState.LOST
    # Wait beyond timeout
    d = sm.on_track_update(**_ok_input(0.70, is_track_alive=False))
    assert d.lock_state == LockState.ABORTED
    assert d.guidance_valid is False


def test_locked_revoked_when_lock_quality_fails():
    sm = _make_sm()
    for t in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
        d = sm.on_track_update(**_ok_input(t))
    assert d.lock_state == LockState.LOCKED
    assert d.guidance_valid is True

    d = sm.on_track_update(**_ok_input(0.07, lock_quality=0.0))
    assert d.lock_state == LockState.TRACKING
    assert d.guidance_valid is False


# ---------- fault flags ----------

def test_fault_flag_forces_aborted_and_invalid_guidance():
    sm = _make_sm()
    for t in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
        sm.on_track_update(**_ok_input(t))
    # Inject a fault
    d = sm.on_track_update(**_ok_input(0.10, active_fault_flags=[FaultFlag.MODEL_FAULT]))
    assert d.lock_state == LockState.ABORTED
    assert d.guidance_valid is False
    assert FaultFlag.MODEL_FAULT in d.fault_flags


def test_inhibit_forces_invalid_guidance():
    sm = _make_sm()
    sm.set_inhibit(True)
    # Even with a great track, INHIBIT must veto
    for t in [0.0, 0.01, 0.02, 0.03, 0.04]:
        d = sm.on_track_update(**_ok_input(t))
    assert d.guidance_valid is False
    assert FaultFlag.INHIBIT in d.fault_flags


# ---------- low confidence / quality ----------

def test_low_class_confidence_keeps_in_tracking():
    sm = _make_sm()
    # Reach TRACKING
    for t in [0.0, 0.01, 0.02, 0.03]:
        sm.on_track_update(**_ok_input(t))
    # Now low class confidence
    for t in [0.04, 0.05, 0.06, 0.07]:
        d = sm.on_track_update(**_ok_input(t, class_confidence=0.10))
    assert d.lock_state == LockState.TRACKING
    assert d.guidance_valid is False


def test_low_lock_quality_keeps_in_tracking():
    sm = _make_sm()
    for t in [0.0, 0.01, 0.02, 0.03]:
        sm.on_track_update(**_ok_input(t))
    for t in [0.04, 0.05, 0.06, 0.07]:
        d = sm.on_track_update(**_ok_input(t, lock_quality=0.20))
    assert d.lock_state == LockState.TRACKING


def test_bird_class_does_not_lock():
    """A track classified as bird must never reach LOCKED."""
    sm = _make_sm()
    for t in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
        d = sm.on_track_update(**_ok_input(t, class_label="bird"))
    assert d.lock_state == LockState.TRACKING
    assert d.guidance_valid is False


# ---------- guidance valid invariants ----------

def test_guidance_valid_requires_lock_or_strike_ready():
    sm = _make_sm()
    # Just SEARCHING -> guidance_valid must be False
    d = sm.on_track_update(**_ok_input(0.0))
    assert d.guidance_valid is False
    # ACQUIRED -> still False
    d = sm.on_track_update(**_ok_input(0.01))
    assert d.guidance_valid is False


# ---------- input validation ----------

def test_invalid_construction_raises():
    with pytest.raises(ValueError):
        _make_sm(acquired_to_tracking_frames=0)
    with pytest.raises(ValueError):
        _make_sm(tracking_to_locked_frames=0)
