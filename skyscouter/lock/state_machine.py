"""
Lock state machine — implements PRD §12 lifecycle exactly.

States:
    NO_CUE -> CUED -> SEARCHING -> ACQUIRED -> TRACKING -> LOCKED -> STRIKE_READY
    Any active state -> LOST -> ABORTED (or back to TRACKING on recovery)

The state machine is deliberately a small, deterministic, fully-tested object.
It contains NO model inference — it consumes track + cue + flags and emits a
state. This is what makes it auditable and what makes safety arguments
defensible.

It is per-track. The pipeline owns one state machine instance per track_id.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Set

from ..schemas import LockState, FaultFlag


@dataclass
class LockDecision:
    """Result of a single state-machine update."""
    lock_state: LockState
    guidance_valid: bool
    lock_quality: float
    fault_flags: List[FaultFlag]
    reason: str  # human-readable explanation, for logs


@dataclass
class StrikeReadyConfig:
    """STRIKE_READY gate thresholds from PRD §12.1."""
    min_locked_duration_seconds: float = 0.5
    min_bbox_frame_fraction: float = 0.05
    bbox_center_window: float = 0.5
    max_cue_age_seconds: Optional[float] = 5.0


class LockStateMachine:
    """One instance per track. Drives the PRD §12 lifecycle."""

    def __init__(
        self,
        acquired_to_tracking_frames: int = 3,
        tracking_to_locked_frames: int = 8,
        min_class_confidence: float = 0.30,
        min_lock_quality: float = 0.50,
        lost_to_aborted_seconds: float = 2.0,
        strike_ready_cfg: Optional[StrikeReadyConfig] = None,
        operator_inhibit: bool = False,
        acceptable_lock_labels: Optional[Iterable[str]] = None,
    ):
        if acquired_to_tracking_frames < 1:
            raise ValueError("acquired_to_tracking_frames must be >= 1")
        if tracking_to_locked_frames < 1:
            raise ValueError("tracking_to_locked_frames must be >= 1")

        self._acquired_to_tracking_frames = acquired_to_tracking_frames
        self._tracking_to_locked_frames = tracking_to_locked_frames
        self._min_class_confidence = min_class_confidence
        self._min_lock_quality = min_lock_quality
        self._lost_to_aborted_seconds = lost_to_aborted_seconds
        self._strike_cfg = strike_ready_cfg or StrikeReadyConfig()
        self._acceptable_lock_labels: Set[str] = {
            str(label).lower().strip()
            for label in (acceptable_lock_labels or ["drone", "uas"])
        }

        # Mutable state
        self._state: LockState = LockState.NO_CUE
        self._consec_in_state: int = 0
        self._first_locked_time: Optional[float] = None
        self._lost_since: Optional[float] = None
        self._operator_inhibit: bool = operator_inhibit

    # ---- public controls ----

    @property
    def state(self) -> LockState:
        return self._state

    def set_inhibit(self, inhibit: bool) -> None:
        self._operator_inhibit = bool(inhibit)

    # ---- update API ----

    def on_track_update(
        self,
        *,
        now_s: float,
        is_track_alive: bool,
        is_track_confirmed: bool,
        class_label: Optional[str],
        class_confidence: float,
        bbox_frame_fraction: float,
        bbox_center_norm_distance: float,
        cue_active: bool,
        cue_age_seconds: Optional[float],
        active_fault_flags: List[FaultFlag],
        lock_quality: float,
    ) -> LockDecision:
        """
        Drive the state machine forward by one frame.

        Args (all per-frame measurements; the state machine itself is stateless
        about how they were computed):
            now_s: capture time of this frame in seconds (from session start)
            is_track_alive: track exists this frame
            is_track_confirmed: track has hits >= min_track_length
            class_label: classifier label ('drone', 'bird', 'unknown', ...)
            class_confidence: 0..1 calibrated class confidence
            bbox_frame_fraction: max(bbox_w / frame_w, bbox_h / frame_h)
            bbox_center_norm_distance: distance of bbox center from frame
                center, normalized to half-frame; 0 = perfectly centered,
                1.0 = at frame edge.
            cue_active: whether a non-expired cue is currently held
            cue_age_seconds: age of latest used cue, or None if no cue
            active_fault_flags: any fault flags raised this frame
            lock_quality: 0..1 composite stability score (computed externally)

        Returns:
            LockDecision describing the new state, guidance flag, and reason.
        """
        flags = list(active_fault_flags)
        if self._operator_inhibit and FaultFlag.INHIBIT not in flags:
            flags.append(FaultFlag.INHIBIT)

        # Hard stop: any fault forces ABORTED state (and guidance_invalid
        # everywhere). PRD NFR-REL-001 + Guidance Safety Rule.
        if flags:
            self._transition_to(LockState.ABORTED, now_s)
            return LockDecision(
                lock_state=LockState.ABORTED,
                guidance_valid=False,
                lock_quality=lock_quality,
                fault_flags=flags,
                reason=f"fault flags active: {[f.value for f in flags]}",
            )

        # Decide state from current state
        s = self._state

        if s in (LockState.NO_CUE, LockState.ABORTED):
            # Wait for cue or fresh activity to restart
            if cue_active:
                self._transition_to(LockState.CUED, now_s)
                return self._emit(LockState.CUED, lock_quality, flags, "cue received")
            if is_track_alive and not cue_active:
                # No cue, but a candidate appeared — go straight to SEARCHING
                self._transition_to(LockState.SEARCHING, now_s)
                return self._emit(LockState.SEARCHING, lock_quality, flags,
                                   "candidate detection without cue")
            return self._emit(LockState.NO_CUE, lock_quality, flags, "idle")

        if s == LockState.CUED:
            # Move to SEARCHING when we begin scanning
            self._transition_to(LockState.SEARCHING, now_s)
            return self._emit(LockState.SEARCHING, lock_quality, flags, "search started")

        if s == LockState.SEARCHING:
            if is_track_alive:
                self._transition_to(LockState.ACQUIRED, now_s)
                return self._emit(LockState.ACQUIRED, lock_quality, flags,
                                   "candidate acquired")
            return self._emit(LockState.SEARCHING, lock_quality, flags, "searching")

        if s == LockState.ACQUIRED:
            if not is_track_alive:
                self._transition_to(LockState.LOST, now_s)
                return self._emit(LockState.LOST, lock_quality, flags,
                                   "track dropped before confirmation")
            self._consec_in_state += 1
            if self._consec_in_state >= self._acquired_to_tracking_frames and is_track_confirmed:
                self._transition_to(LockState.TRACKING, now_s)
                return self._emit(LockState.TRACKING, lock_quality, flags,
                                   "track confirmed, transitioning to TRACKING")
            return self._emit(LockState.ACQUIRED, lock_quality, flags, "acquiring")

        if s == LockState.TRACKING:
            if not is_track_alive:
                self._transition_to(LockState.LOST, now_s)
                return self._emit(LockState.LOST, lock_quality, flags,
                                   "track lost during TRACKING")

            # Drone-class confidence and lock quality gates
            normalized_label = (class_label or "").lower().strip()
            class_ok = normalized_label in self._acceptable_lock_labels and \
                       class_confidence >= self._min_class_confidence
            quality_ok = lock_quality >= self._min_lock_quality

            if class_ok and quality_ok:
                self._consec_in_state += 1
                if self._consec_in_state >= self._tracking_to_locked_frames:
                    self._transition_to(LockState.LOCKED, now_s)
                    return self._emit(LockState.LOCKED, lock_quality, flags,
                                       "lock gates satisfied")
                return self._emit(LockState.TRACKING, lock_quality, flags,
                                   f"tracking ({self._consec_in_state}/{self._tracking_to_locked_frames})")
            else:
                # Reset the consecutive counter if gates fail this frame
                self._consec_in_state = 0
                return self._emit(LockState.TRACKING, lock_quality, flags,
                                   "lock gates not met; staying in TRACKING")

        if s == LockState.LOCKED:
            if not is_track_alive:
                self._transition_to(LockState.LOST, now_s)
                return self._emit(LockState.LOST, lock_quality, flags,
                                   "track lost during LOCKED")
            if not self._lock_gates_pass(class_label, class_confidence, lock_quality):
                self._transition_to(LockState.TRACKING, now_s)
                return self._emit(LockState.TRACKING, lock_quality, flags,
                                   "LOCKED revoked; lock gates failed")

            # Check STRIKE_READY gates from PRD §12.1
            if self._strike_ready_gates_pass(
                now_s=now_s,
                bbox_frame_fraction=bbox_frame_fraction,
                bbox_center_norm_distance=bbox_center_norm_distance,
                cue_age_seconds=cue_age_seconds,
            ):
                self._transition_to(LockState.STRIKE_READY, now_s)
                return self._emit(LockState.STRIKE_READY, lock_quality, flags,
                                   "STRIKE_READY gates passed")
            return self._emit(LockState.LOCKED, lock_quality, flags,
                               "locked, STRIKE_READY gates not yet met")

        if s == LockState.STRIKE_READY:
            if not is_track_alive:
                self._transition_to(LockState.LOST, now_s)
                return self._emit(LockState.LOST, lock_quality, flags,
                                   "track lost during STRIKE_READY")
            if not self._lock_gates_pass(class_label, class_confidence, lock_quality):
                self._transition_to(LockState.TRACKING, now_s)
                return self._emit(LockState.TRACKING, lock_quality, flags,
                                   "STRIKE_READY revoked; lock gates failed")

            # PRD FR-LOCK-004: revoke STRIKE_READY immediately if any gate fails
            if not self._strike_ready_gates_pass(
                now_s=now_s,
                bbox_frame_fraction=bbox_frame_fraction,
                bbox_center_norm_distance=bbox_center_norm_distance,
                cue_age_seconds=cue_age_seconds,
            ):
                self._transition_to(LockState.LOCKED, now_s)
                return self._emit(LockState.LOCKED, lock_quality, flags,
                                   "STRIKE_READY revoked; gate failed")
            return self._emit(LockState.STRIKE_READY, lock_quality, flags,
                               "STRIKE_READY held")

        if s == LockState.LOST:
            if is_track_alive:
                # Recovery: go back to TRACKING (PRD §12)
                self._transition_to(LockState.TRACKING, now_s)
                return self._emit(LockState.TRACKING, lock_quality, flags,
                                   "track recovered")
            # Timeout to ABORTED
            if self._lost_since is not None and \
                    (now_s - self._lost_since) >= self._lost_to_aborted_seconds:
                self._transition_to(LockState.ABORTED, now_s)
                return self._emit(LockState.ABORTED, lock_quality, flags,
                                   f"LOST timeout ({self._lost_to_aborted_seconds:.1f}s) -> ABORTED")
            return self._emit(LockState.LOST, lock_quality, flags, "lost, recovering")

        # Unreachable — every state handled above.
        raise RuntimeError(f"Unhandled state: {s}")

    # ---- internals ----

    def _strike_ready_gates_pass(
        self,
        *,
        now_s: float,
        bbox_frame_fraction: float,
        bbox_center_norm_distance: float,
        cue_age_seconds: Optional[float],
    ) -> bool:
        # Continuous lock duration
        if self._first_locked_time is None:
            return False
        if (now_s - self._first_locked_time) < self._strike_cfg.min_locked_duration_seconds:
            return False
        # Bbox size in frame
        if bbox_frame_fraction < self._strike_cfg.min_bbox_frame_fraction:
            return False
        # Bbox center within central window. center_norm_distance is in [0,1]
        # where 1.0 = at frame edge. The "central X% window" from PRD means
        # center distance must be <= X/2 (since X is the total fraction).
        center_window_half = self._strike_cfg.bbox_center_window / 2.0
        if bbox_center_norm_distance > center_window_half * 2.0:
            # interpret bbox_center_window as full width in normalized
            # frame-fraction; convert: bbox_center_norm_distance is normalized
            # to half-frame, so allowed = bbox_center_window
            # (e.g., window=0.5 -> distance allowed up to 0.5 of half-frame)
            return False
        # Cue age
        if self._strike_cfg.max_cue_age_seconds is not None:
            if cue_age_seconds is not None and \
                    cue_age_seconds > self._strike_cfg.max_cue_age_seconds:
                return False
        return True

    def _lock_gates_pass(
        self,
        class_label: Optional[str],
        class_confidence: float,
        lock_quality: float,
    ) -> bool:
        normalized_label = (class_label or "").lower().strip()
        return (
            normalized_label in self._acceptable_lock_labels
            and class_confidence >= self._min_class_confidence
            and lock_quality >= self._min_lock_quality
        )

    def _transition_to(self, new_state: LockState, now_s: float) -> None:
        if new_state == self._state:
            return
        # Track first-LOCKED time for STRIKE_READY duration gate
        if new_state == LockState.LOCKED and self._first_locked_time is None:
            self._first_locked_time = now_s
        elif new_state == LockState.LOST:
            self._lost_since = now_s
        else:
            self._lost_since = None
            if new_state not in (LockState.LOCKED, LockState.STRIKE_READY):
                # leaving lock streak resets the locked duration anchor
                self._first_locked_time = None

        self._state = new_state
        self._consec_in_state = 0

    def _emit(
        self,
        state: LockState,
        lock_quality: float,
        flags: List[FaultFlag],
        reason: str,
    ) -> LockDecision:
        guidance_valid = state in (LockState.LOCKED, LockState.STRIKE_READY) and not flags
        return LockDecision(
            lock_state=state,
            guidance_valid=guidance_valid,
            lock_quality=lock_quality,
            fault_flags=flags,
            reason=reason,
        )
