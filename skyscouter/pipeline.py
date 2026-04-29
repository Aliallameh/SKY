"""
Main pipeline orchestrator.

Wires together the frame source, detector, tracker, lock state machines, and
output writers. This is the only module where these components meet; every
other module is independently testable.

Per-track state machines are kept in a dict keyed by track_id, so different
tracks have independent lifecycles. The pipeline picks one "primary" track
per frame for the published target_state — the largest confirmed track is
the default policy (configurable later).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .io.frame_source import BaseFrameSource, Frame
from .perception.base_detector import BaseDetector, Detection
from .tracking.base_tracker import BaseTracker, Track
from .lock.state_machine import LockStateMachine, StrikeReadyConfig
from .output.lock_quality import compute_lock_quality
from .schemas import (
    TargetState, MessageType, LockState, RangeSource, FaultFlag,
)
from .output.target_state_writer import TargetStateJsonlWriter
from .output.annotator import VideoAnnotator
from .output.run_logger import RunLogger
from .output.evaluation import DiagnosticsWriter, EvaluationCollector


class Pipeline:
    def __init__(
        self,
        config: Dict[str, Any],
        frame_source: BaseFrameSource,
        detector: BaseDetector,
        tracker: BaseTracker,
        run_logger: RunLogger,
        target_state_writer: Optional[TargetStateJsonlWriter] = None,
        annotator: Optional[VideoAnnotator] = None,
        diagnostics_writer: Optional[DiagnosticsWriter] = None,
        evaluation_collector: Optional[EvaluationCollector] = None,
    ):
        self._cfg = config
        self._source = frame_source
        self._detector = detector
        self._tracker = tracker
        self._logger = run_logger
        self._writer = target_state_writer
        self._annotator = annotator
        self._diagnostics = diagnostics_writer
        self._evaluation = evaluation_collector

        # Lock state machines per track_id
        self._lock_smachines: Dict[int, LockStateMachine] = {}
        self._primary_track_id: Optional[int] = None

        # Cached config blocks
        lock_cfg = config["lock"]
        sr_cfg = lock_cfg.get("strike_ready", {})
        self._strike_cfg = StrikeReadyConfig(
            min_locked_duration_seconds=float(sr_cfg.get("min_locked_duration_seconds", 0.5)),
            min_bbox_frame_fraction=float(sr_cfg.get("min_bbox_frame_fraction", 0.05)),
            bbox_center_window=float(sr_cfg.get("bbox_center_window", 0.5)),
            max_cue_age_seconds=sr_cfg.get("max_cue_age_seconds", 5.0),
        )
        self._lock_cfg = lock_cfg
        self._acceptable_lock_labels = [
            str(label).lower().strip()
            for label in lock_cfg.get("acceptable_lock_labels", ["drone", "uas"])
        ]

        camera_cfg = config["camera"]
        self._calibration_id = camera_cfg.get("calibration_id", "BENCH_REPLAY_UNCALIBRATED_v1")
        self._is_calibrated = bool(camera_cfg.get("is_calibrated", False))

        self._sensor_sources = ["EO_MONO"]  # configurable later
        self._model_version = detector.get_model_version()

    # ---- main loop ----

    def run(self) -> None:
        # Warmup (fairer first-frame latency)
        self._logger.info("Detector warmup starting")
        try:
            self._detector.warmup()
        except Exception as e:
            self._logger.warning(f"Warmup failed (non-fatal): {e}")
        self._logger.info("Pipeline starting main loop")

        for frame in self._source:
            t0 = time.perf_counter()

            # 1. Detect
            try:
                detections = self._detector.detect(frame.image_bgr)
            except Exception as e:
                self._logger.error(f"Detector error on frame {frame.frame_index}: {e}")
                detections = []
                self._publish_fault(frame, [FaultFlag.MODEL_FAULT])
                continue

            self._logger.increment_detections(len(detections))

            # 2. Track
            try:
                tracks = self._tracker.update(
                    detections,
                    frame_index=frame.frame_index,
                    capture_time_s=frame.capture_time_s,
                    image_bgr=frame.image_bgr,
                )
            except Exception as e:
                self._logger.error(f"Tracker error on frame {frame.frame_index}: {e}")
                tracks = []

            # 3. Pick primary track (largest confirmed; else largest)
            primary = self._pick_primary_track(tracks)

            # 4. Drive lock state machines per track
            primary_decision = None
            primary_lock_quality = 0.0
            if primary is not None:
                sm = self._get_or_create_state_machine(primary.track_id)
                quality = compute_lock_quality(primary)
                primary_lock_quality = quality
                bbox_frame_fraction = self._bbox_frame_fraction(
                    primary.detection, frame.width, frame.height
                )
                center_norm = self._center_norm_distance(
                    primary.detection, frame.width, frame.height
                )
                primary_decision = sm.on_track_update(
                    now_s=frame.capture_time_s,
                    is_track_alive=primary.time_since_update == 0,
                    is_track_confirmed=primary.hits >= getattr(self._tracker, "min_track_length", 3),
                    class_label=self._normalize_class_label(primary.detection.class_label),
                    class_confidence=primary.detection.confidence,
                    bbox_frame_fraction=bbox_frame_fraction,
                    bbox_center_norm_distance=center_norm,
                    cue_active=False,
                    cue_age_seconds=None,
                    active_fault_flags=[],
                    lock_quality=quality,
                )

            # Drive every other state machine forward in "track alive=False"
            # mode so they can transition LOST -> ABORTED, etc.
            for tid, sm in list(self._lock_smachines.items()):
                if primary is not None and tid == primary.track_id:
                    continue
                # find the track if still alive (might be present but not primary)
                tr = next((t for t in tracks if t.track_id == tid), None)
                is_alive = tr is not None and tr.time_since_update == 0
                sm.on_track_update(
                    now_s=frame.capture_time_s,
                    is_track_alive=is_alive,
                    is_track_confirmed=bool(
                        tr is not None and tr.hits >= getattr(self._tracker, "min_track_length", 3)
                    ),
                    class_label=self._normalize_class_label(tr.detection.class_label) if is_alive else None,
                    class_confidence=tr.detection.confidence if is_alive else 0.0,
                    bbox_frame_fraction=self._bbox_frame_fraction(
                        tr.detection, frame.width, frame.height
                    ) if tr is not None else 0.0,
                    bbox_center_norm_distance=self._center_norm_distance(
                        tr.detection, frame.width, frame.height
                    ) if tr is not None else 1.0,
                    cue_active=False,
                    cue_age_seconds=None,
                    active_fault_flags=[],
                    lock_quality=compute_lock_quality(tr) if tr is not None else 0.0,
                )

            # 5. Build target_state and publish
            latency_ms = (time.perf_counter() - t0) * 1000.0
            ts = self._build_target_state(
                frame=frame,
                primary=primary,
                primary_decision=primary_decision,
                primary_lock_quality=primary_lock_quality,
                latency_ms=latency_ms,
            )
            if self._writer is not None:
                self._writer.write(ts)
            if self._diagnostics is not None:
                self._diagnostics.write(frame.frame_index, ts, primary)
            if self._evaluation is not None:
                self._evaluation.add(frame.frame_index, ts)

            # 6. Annotate video
            if self._annotator is not None:
                annotated = self._annotator.annotate(
                    image_bgr=frame.image_bgr,
                    tracks=tracks,
                    primary_track_id=primary.track_id if primary else None,
                    lock_state=ts.lock_state,
                    guidance_valid=ts.guidance_valid,
                    confidence=ts.confidence or 0.0,
                    lock_quality=ts.lock_quality or 0.0,
                    latency_ms=latency_ms,
                    frame_index=frame.frame_index,
                )
                self._annotator.write(annotated)

            self._logger.increment_frame()

        self._logger.info("Pipeline main loop complete")

    # ---- helpers ----

    @staticmethod
    def _normalize_class_label(label: str) -> str:
        """
        Normalize a detector class label to one of the discriminator buckets.
        Until we have a real drone/bird classifier, COCO 'bird' maps to 'bird'
        and everything else (airplane, kite, generic high-conf object) maps to
        'unknown'. Pretending COCO 'airplane' = 'drone' would be cheating.
        """
        l = (label or "").lower().strip()
        if l == "bird":
            return "bird"
        if l == "drone" or l == "uas":
            return "drone"
        if l in ("airborne_candidate", "drone_candidate", "uas_candidate"):
            return l
        # COCO 'airplane' is not a drone. Mark unknown — the lock SM handles it.
        return "unknown"

    @staticmethod
    def _bbox_frame_fraction(d: Detection, fw: int, fh: int) -> float:
        if fw <= 0 or fh <= 0:
            return 0.0
        return max(d.w / fw, d.h / fh)

    @staticmethod
    def _center_norm_distance(d: Detection, fw: int, fh: int) -> float:
        if fw <= 0 or fh <= 0:
            return 1.0
        cx = d.cx
        cy = d.cy
        # Distance from frame center, normalized so 1.0 = corner of half-frame
        dx = (cx - fw / 2.0) / (fw / 2.0)
        dy = (cy - fh / 2.0) / (fh / 2.0)
        return min(1.0, (dx ** 2 + dy ** 2) ** 0.5)

    def _pick_primary_track(self, tracks: List[Track]) -> Optional[Track]:
        if not tracks:
            self._primary_track_id = None
            return None

        if self._primary_track_id is not None:
            current = next((t for t in tracks if t.track_id == self._primary_track_id), None)
            if current is not None and current.time_since_update == 0:
                return current

        # Default: largest confirmed track. If no confirmed tracks, largest.
        min_track_length = getattr(self._tracker, "min_track_length", 3)
        confirmed = [
            t for t in tracks
            if t.hits >= min_track_length and t.time_since_update == 0
        ]
        pool = confirmed if confirmed else tracks
        primary = max(
            pool,
            key=lambda t: (
                t.time_since_update == 0,
                float(t.detection.confidence),
                compute_lock_quality(t),
                t.detection.area,
            ),
        )
        self._primary_track_id = primary.track_id
        return primary

    def _get_or_create_state_machine(self, track_id: int) -> LockStateMachine:
        if track_id not in self._lock_smachines:
            self._lock_smachines[track_id] = LockStateMachine(
                acquired_to_tracking_frames=int(self._lock_cfg.get("acquired_to_tracking_frames", 3)),
                tracking_to_locked_frames=int(self._lock_cfg.get("tracking_to_locked_frames", 8)),
                min_class_confidence=float(self._lock_cfg.get("min_class_confidence", 0.30)),
                min_lock_quality=float(self._lock_cfg.get("min_lock_quality", 0.50)),
                lost_to_aborted_seconds=float(self._lock_cfg.get("lost_to_aborted_seconds", 2.0)),
                strike_ready_cfg=self._strike_cfg,
                operator_inhibit=False,
                acceptable_lock_labels=self._acceptable_lock_labels,
            )
            self._logger.increment_tracks(1)
        return self._lock_smachines[track_id]

    def _build_target_state(
        self,
        *,
        frame: Frame,
        primary: Optional[Track],
        primary_decision,
        primary_lock_quality: float,
        latency_ms: float,
    ) -> TargetState:
        if primary is None or primary_decision is None:
            ts = TargetState(
                message_type=MessageType.NO_TARGET.value,
                timestamp_utc=frame.timestamp_utc,
                lock_state=LockState.SEARCHING.value,
                guidance_valid=False,
                image_size_wh=[frame.width, frame.height],
                latency_ms=latency_ms,
                sensor_sources=list(self._sensor_sources),
                model_version=self._model_version,
                calibration_id=self._calibration_id,
                fault_flags=[],
            )
            ts.enforce_safety_invariants()
            return ts

        d = primary.detection
        ts = TargetState(
            message_type=MessageType.TARGET_STATE.value,
            timestamp_utc=frame.timestamp_utc,
            cue_id=None,
            track_id=primary.track_id,
            lock_state=primary_decision.lock_state.value,
            guidance_valid=primary_decision.guidance_valid,
            bbox_xywh=[d.x, d.y, d.w, d.h],
            image_size_wh=[frame.width, frame.height],
            line_of_sight_body=None if not self._is_calibrated else None,
            range_estimate_m=None,
            range_source=RangeSource.NONE.value,
            confidence=float(d.confidence),
            lock_quality=float(primary_lock_quality),
            latency_ms=latency_ms,
            cue_age_ms=None,
            sensor_sources=list(self._sensor_sources),
            model_version=self._model_version,
            calibration_id=self._calibration_id,
            fault_flags=[f.value for f in primary_decision.fault_flags],
        )
        ts.enforce_safety_invariants()
        return ts

    def _publish_fault(self, frame: Frame, flags: List[FaultFlag]) -> None:
        if self._writer is None:
            return
        ts = TargetState(
            message_type=MessageType.SYSTEM_FAULT.value,
            timestamp_utc=frame.timestamp_utc,
            lock_state=LockState.ABORTED.value,
            guidance_valid=False,
            image_size_wh=[frame.width, frame.height],
            sensor_sources=list(self._sensor_sources),
            model_version=self._model_version,
            calibration_id=self._calibration_id,
            fault_flags=[f.value for f in flags],
        )
        ts.enforce_safety_invariants()
        self._writer.write(ts)
