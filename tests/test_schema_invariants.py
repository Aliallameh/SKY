"""Tests for schema invariants — guidance safety rule must not be bypassable."""
from __future__ import annotations

from skyscouter.schemas import TargetState, LockState, FaultFlag


def test_fault_flags_force_invalid_guidance():
    ts = TargetState(
        message_type="TARGET_STATE",
        lock_state=LockState.LOCKED.value,
        guidance_valid=True,  # try to claim true
        fault_flags=[FaultFlag.MODEL_FAULT.value],
    )
    ts.enforce_safety_invariants()
    assert ts.guidance_valid is False  # invariant overrode


def test_non_locked_state_forces_invalid_guidance():
    ts = TargetState(
        message_type="TARGET_STATE",
        lock_state=LockState.SEARCHING.value,
        guidance_valid=True,  # try to claim true
        fault_flags=[],
    )
    ts.enforce_safety_invariants()
    assert ts.guidance_valid is False


def test_locked_with_no_faults_remains_valid():
    ts = TargetState(
        message_type="TARGET_STATE",
        lock_state=LockState.LOCKED.value,
        guidance_valid=True,
        fault_flags=[],
    )
    ts.enforce_safety_invariants()
    assert ts.guidance_valid is True


def test_strike_ready_with_no_faults_remains_valid():
    ts = TargetState(
        message_type="TARGET_STATE",
        lock_state=LockState.STRIKE_READY.value,
        guidance_valid=True,
        fault_flags=[],
    )
    ts.enforce_safety_invariants()
    assert ts.guidance_valid is True
