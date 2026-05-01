"""Visual bearing and yaw-alignment guidance hint computation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from ..schemas import GuidanceHint, LockState
from .camera_model import PinholeCameraModel, bbox_to_center
from .controller import YawRateController
from .filtering import AngleEmaFilter, CenterPredictor


@dataclass
class GuidanceInput:
    frame_id: int
    timestamp_s: float
    timestamp_utc: str
    frame_width: int
    frame_height: int
    bbox_xyxy: Optional[Sequence[float]]
    track_id: Optional[int]
    class_label: Optional[str]
    confidence: Optional[float]
    lock_state: str
    target_state_guidance_valid: bool
    fault_flags: Sequence[str]
    time_since_update_frames: int = 0
    center_history: Optional[Iterable[Tuple[float, float, float]]] = None


class BearingGuidanceComputer:
    """Converts tracked bbox state into a versioned log-only guidance hint."""

    def __init__(
        self,
        *,
        camera_cfg: dict,
        validity_cfg: dict,
        filtering_cfg: dict,
        controller: YawRateController,
        run_id: Optional[str] = None,
        acceptable_class_labels: Optional[Iterable[str]] = None,
        frame_width_px: Optional[int] = None,
        frame_height_px: Optional[int] = None,
    ):
        self._camera_cfg = dict(camera_cfg)
        self._validity_cfg = dict(validity_cfg)
        self._filtering_cfg = dict(filtering_cfg)
        self._controller = controller
        self._run_id = run_id
        self._frame_width_px = frame_width_px
        self._frame_height_px = frame_height_px
        self._allowed_lock_states = {
            str(s).upper().strip()
            for s in self._validity_cfg.get(
                "allowed_lock_states",
                [LockState.TRACKING.value, LockState.LOCKED.value, LockState.STRIKE_READY.value],
            )
        }
        configured_labels = self._validity_cfg.get("acceptable_class_labels")
        labels = configured_labels if configured_labels is not None else acceptable_class_labels
        self._acceptable_class_labels = {
            str(label).lower().strip()
            for label in (labels or [])
        }
        self._ema: Optional[AngleEmaFilter] = None
        if bool(self._filtering_cfg.get("enabled", True)):
            self._ema = AngleEmaFilter(alpha=float(self._filtering_cfg["ema_alpha"]))
        self._predictor: Optional[CenterPredictor] = None
        if bool(self._filtering_cfg.get("prediction_enabled", False)):
            self._predictor = CenterPredictor(
                lead_time_s=float(self._filtering_cfg["lead_time_s"]),
                max_prediction_px=float(self._filtering_cfg["max_prediction_px"]),
                history_len=int(self._filtering_cfg.get("history_len", 8)),
            )

    def compute(self, data: GuidanceInput) -> GuidanceHint:
        hint = GuidanceHint(
            frame_id=int(data.frame_id),
            timestamp_s=float(data.timestamp_s),
            timestamp_utc=data.timestamp_utc,
            run_id=self._run_id,
            track_id=data.track_id,
            source_lock_state=data.lock_state,
            guidance_valid_from_lock_state=bool(data.target_state_guidance_valid),
            lead_time_s=float(self._filtering_cfg.get("lead_time_s", 0.0)),
            controller_mode=self._controller.mode if self._controller.enabled else "disabled",
            confidence=None if data.confidence is None else float(data.confidence),
            reason=[],
            notes={"safety_scope": "log_only_command_proposal_no_actuation"},
        )

        reasons = self._validity_reasons(data)
        geometry_ok = True
        try:
            if data.bbox_xyxy is None:
                raise ValueError("missing bbox")
            bbox = _bbox_tuple(data.bbox_xyxy)
            camera = PinholeCameraModel.from_config(
                self._camera_cfg,
                frame_width_px=data.frame_width or self._frame_width_px,
                frame_height_px=data.frame_height or self._frame_height_px,
            )
            target_center = bbox_to_center(*bbox)
            frame_center = camera.frame_center()
            bearing_rad, elevation_rad = camera.pixel_error_to_angles(
                target_center[0],
                target_center[1],
                data.frame_width,
                data.frame_height,
            )
            bearing_deg = math.degrees(bearing_rad)
            elevation_deg = math.degrees(elevation_rad)
            predicted = None
            if self._predictor is not None:
                predicted = self._predictor.update(
                    target_center,
                    data.timestamp_s,
                    external_history=data.center_history,
                )
        except ValueError as exc:
            geometry_ok = False
            reasons.append(str(exc))
            bbox = None
            camera = None
            target_center = None
            frame_center = None
            bearing_rad = elevation_rad = None
            bearing_deg = elevation_deg = None
            predicted = None

        if geometry_ok and bbox is not None and target_center is not None and frame_center is not None:
            pixel_error = [target_center[0] - frame_center[0], frame_center[1] - target_center[1]]
            normalized_error = [
                pixel_error[0] / max(1.0, data.frame_width * 0.5),
                pixel_error[1] / max(1.0, data.frame_height * 0.5),
            ]
            hint.bbox_xyxy = [float(v) for v in bbox]
            hint.target_center_px = [float(target_center[0]), float(target_center[1])]
            hint.predicted_target_center_px = (
                None if predicted is None else [float(predicted[0]), float(predicted[1])]
            )
            hint.frame_center_px = [float(frame_center[0]), float(frame_center[1])]
            hint.pixel_error_px = [float(pixel_error[0]), float(pixel_error[1])]
            hint.normalized_error = [float(normalized_error[0]), float(normalized_error[1])]
            hint.bearing_error_rad = float(bearing_rad) if bearing_rad is not None else None
            hint.bearing_error_deg = float(bearing_deg) if bearing_deg is not None else None
            hint.elevation_error_rad = float(elevation_rad) if elevation_rad is not None else None
            hint.elevation_error_deg = float(elevation_deg) if elevation_deg is not None else None
            if self._ema is not None and bearing_deg is not None and elevation_deg is not None:
                filtered_bearing, filtered_elevation = self._ema.update(bearing_deg, elevation_deg)
            else:
                filtered_bearing = bearing_deg
                filtered_elevation = elevation_deg
            hint.filtered_bearing_error_deg = (
                None if filtered_bearing is None else float(filtered_bearing)
            )
            hint.filtered_elevation_error_deg = (
                None if filtered_elevation is None else float(filtered_elevation)
            )

        valid = geometry_ok and not reasons
        hint.valid = bool(valid)
        hint.valid_for_logging = bool(valid)
        hint.valid_for_actuation = False
        if valid and hint.filtered_bearing_error_deg is not None:
            hint.yaw_rate_cmd_deg_s = self._controller.compute(
                hint.filtered_bearing_error_deg,
                valid=True,
            )
        else:
            hint.yaw_rate_cmd_deg_s = self._controller.compute(0.0, valid=False)
        hint.pitch_rate_cmd_deg_s = None
        hint.reason = ["ok"] if valid else reasons
        hint.enforce_safety_invariants()
        return hint

    def _validity_reasons(self, data: GuidanceInput) -> List[str]:
        reasons: List[str] = []
        if data.track_id is None or data.bbox_xyxy is None:
            reasons.append("no_tracked_target")
        if data.frame_width <= 0 or data.frame_height <= 0:
            reasons.append("invalid_frame_shape")
        lock_state = str(data.lock_state or "").upper().strip()
        if lock_state not in self._allowed_lock_states:
            reasons.append(f"lock_state_not_allowed:{data.lock_state}")
        advisory = bool(self._validity_cfg.get("advisory_before_lock", False))
        require_ts_guidance = bool(
            self._validity_cfg.get("require_guidance_valid_target_state", True)
        )
        if require_ts_guidance and not data.target_state_guidance_valid:
            if advisory:
                reasons.append("advisory_before_lock_log_only")
            else:
                reasons.append("target_state_guidance_not_valid")
        min_conf = float(self._validity_cfg.get("min_confidence", 0.0))
        if data.confidence is None or float(data.confidence) < min_conf:
            reasons.append("confidence_below_minimum")
        max_stale = int(self._validity_cfg.get("max_stale_frames", 0))
        if int(data.time_since_update_frames) > max_stale:
            reasons.append("track_stale")
        if data.fault_flags:
            reasons.append("fault_flags_active")
        if self._acceptable_class_labels:
            label = str(data.class_label or "").lower().strip()
            if label not in self._acceptable_class_labels:
                reasons.append(f"class_label_not_allowed:{data.class_label}")

        if "advisory_before_lock_log_only" in reasons:
            reasons.remove("advisory_before_lock_log_only")
        return reasons


def _bbox_tuple(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError("bbox must contain four xyxy values")
    vals = tuple(float(v) for v in bbox)
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("bbox values must be finite")
    x1, y1, x2, y2 = vals
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive width and height")
    return vals  # type: ignore[return-value]
