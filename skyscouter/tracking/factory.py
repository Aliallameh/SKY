"""Factory: build a tracker from a config dict."""
from __future__ import annotations

from typing import Any, Dict

from .base_tracker import BaseTracker
from .ego_motion import IdentityEgoMotion
from .single_target_kalman_lk import SingleTargetKalmanLKTracker
from .simple_tracker import SimpleIoUTracker


def build_tracker(tracker_cfg: Dict[str, Any]) -> BaseTracker:
    backend = tracker_cfg.get("backend", "bytetrack")

    # We currently implement only the simple tracker; bytetrack request maps
    # to it for Sprint 0 since the interface is identical. A real ByteTrack
    # drop-in is Sprint 1.
    if backend == "single_target_kalman_lk":
        return SingleTargetKalmanLKTracker(
            min_track_length=int(tracker_cfg.get("min_track_length", 3)),
            track_buffer_frames=int(tracker_cfg.get("track_buffer_frames", 30)),
            lk_min_points=int(tracker_cfg.get("lk_min_points", 8)),
            reacquisition_radius_px=float(tracker_cfg.get("reacquisition_radius_px", 160.0)),
            max_primary_switches_per_second=float(
                tracker_cfg.get("max_primary_switches_per_second", 1.0)
            ),
            max_prediction_only_frames=int(tracker_cfg.get("max_prediction_only_frames", 6)),
            min_prediction_confidence=float(tracker_cfg.get("min_prediction_confidence", 0.05)),
        )

    if backend in ("bytetrack", "sort", "simple"):
        ego = IdentityEgoMotion() if not tracker_cfg.get(
            "ego_motion_compensation", False
        ) else IdentityEgoMotion()
        # Note: even when ego_motion_compensation=true, we still wire the
        # IdentityEgoMotion stub for now. Real GyroDrivenEgoMotion plugs in
        # at Sprint 3.
        return SimpleIoUTracker(
            match_threshold=float(tracker_cfg.get("match_threshold", 0.3)),
            center_match_threshold_px=float(tracker_cfg.get("center_match_threshold_px", 80.0)),
            track_buffer_frames=int(tracker_cfg.get("track_buffer_frames", 30)),
            min_track_length=int(tracker_cfg.get("min_track_length", 3)),
            ego_motion=ego,
        )

    raise ValueError(f"Unknown tracker backend: {backend!r}")
