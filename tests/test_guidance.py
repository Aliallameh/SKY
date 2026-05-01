from __future__ import annotations

import json
import math

import pytest

from skyscouter.guidance.bearing import BearingGuidanceComputer, GuidanceInput
from skyscouter.guidance.camera_model import PinholeCameraModel, bbox_to_center
from skyscouter.guidance.controller import YawRateController
from skyscouter.guidance.filtering import AngleEmaFilter, CenterPredictor
from skyscouter.output.guidance_writer import GuidanceHintJsonlWriter


def _camera_cfg():
    return {
        "mode": "fov",
        "horizontal_fov_deg": 90.0,
        "vertical_fov_deg": None,
        "frame_width_px": 640,
        "frame_height_px": 480,
        "fx_px": None,
        "fy_px": None,
        "cx_px": None,
        "cy_px": None,
        "assume_principal_point_center": True,
    }


def _validity_cfg(**overrides):
    cfg = {
        "advisory_before_lock": False,
        "allowed_lock_states": ["LOCKED", "STRIKE_READY"],
        "acceptable_class_labels": ["drone"],
        "min_confidence": 0.30,
        "require_guidance_valid_target_state": True,
        "max_stale_frames": 3,
    }
    cfg.update(overrides)
    return cfg


def _filtering_cfg(**overrides):
    cfg = {
        "enabled": True,
        "ema_alpha": 0.5,
        "prediction_enabled": True,
        "lead_time_s": 0.2,
        "max_prediction_px": 250,
        "history_len": 8,
    }
    cfg.update(overrides)
    return cfg


def _controller(max_delta=None):
    return YawRateController(
        enabled=True,
        mode="yaw_p",
        kp_yaw=2.0,
        deadband_deg=1.0,
        max_yaw_rate_deg_s=45.0,
        max_delta_yaw_rate_deg_s=max_delta,
    )


def _guidance_input(**overrides):
    base = {
        "frame_id": 7,
        "timestamp_s": 1.0,
        "timestamp_utc": "2026-04-30T00:00:01.000Z",
        "frame_width": 640,
        "frame_height": 480,
        "bbox_xyxy": [310, 230, 330, 250],
        "track_id": 1,
        "class_label": "drone",
        "confidence": 0.9,
        "lock_state": "LOCKED",
        "target_state_guidance_valid": True,
        "fault_flags": [],
        "time_since_update_frames": 0,
        "center_history": [(300.0, 240.0, 0.8), (320.0, 240.0, 1.0)],
    }
    base.update(overrides)
    return GuidanceInput(**base)


def test_camera_model_center_pixel_has_zero_bearing_and_elevation():
    cam = PinholeCameraModel.from_config(_camera_cfg())
    bearing, elevation = cam.pixel_error_to_angles(320, 240, 640, 480)
    assert bearing == pytest.approx(0.0)
    assert elevation == pytest.approx(0.0)


def test_camera_model_positive_bearing_when_target_is_right_of_center():
    cam = PinholeCameraModel.from_config(_camera_cfg())
    bearing, _ = cam.pixel_error_to_angles(420, 240, 640, 480)
    assert bearing > 0.0


def test_fov_derived_fx_calculation():
    cam = PinholeCameraModel.from_config(_camera_cfg())
    expected = 640 / (2 * math.tan(math.radians(90.0) / 2))
    assert cam.fx_px == pytest.approx(expected)
    assert cam.fy_px == pytest.approx(expected)


def test_bbox_center_calculation():
    assert bbox_to_center(10, 20, 30, 60) == (20, 40)


def test_invalid_bbox_produces_invalid_guidance():
    computer = BearingGuidanceComputer(
        camera_cfg=_camera_cfg(),
        validity_cfg=_validity_cfg(),
        filtering_cfg=_filtering_cfg(),
        controller=_controller(),
        acceptable_class_labels=["drone"],
    )
    hint = computer.compute(_guidance_input(bbox_xyxy=[10, 20, 5, 30]))
    assert hint.valid is False
    assert hint.yaw_rate_cmd_deg_s == 0.0
    assert any("bbox" in reason for reason in hint.reason)


def test_yaw_controller_deadband_outputs_zero():
    ctrl = _controller()
    assert ctrl.compute(0.5, valid=True) == 0.0


def test_yaw_controller_saturates_at_max_rate():
    ctrl = _controller()
    assert ctrl.compute(100.0, valid=True) == pytest.approx(45.0)


def test_invalid_guidance_returns_zero_command():
    ctrl = _controller()
    assert ctrl.compute(20.0, valid=False) == 0.0


def test_ema_filter_smooths_without_inverting_sign():
    filt = AngleEmaFilter(alpha=0.5)
    assert filt.update(10.0, 2.0) == (10.0, 2.0)
    bearing, elevation = filt.update(2.0, 0.0)
    assert 0.0 < bearing < 10.0
    assert elevation >= 0.0


def test_prediction_moves_center_in_expected_direction():
    pred = CenterPredictor(lead_time_s=0.2, max_prediction_px=250)
    out = pred.update((120.0, 100.0), 1.0, [(100.0, 100.0, 0.8), (120.0, 100.0, 1.0)])
    assert out is not None
    assert out[0] > 120.0
    assert out[1] == pytest.approx(100.0)


def test_guidance_writer_writes_valid_jsonl_rows(tmp_path):
    computer = BearingGuidanceComputer(
        camera_cfg=_camera_cfg(),
        validity_cfg=_validity_cfg(),
        filtering_cfg=_filtering_cfg(),
        controller=_controller(),
        run_id="run_test",
        acceptable_class_labels=["drone"],
    )
    hint = computer.compute(_guidance_input())
    path = tmp_path / "guidance_hints.jsonl"
    with GuidanceHintJsonlWriter(str(path)) as writer:
        writer.write(hint)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["schema_version"] == "skyscout.guidance_hint.v1"
    assert rows[0]["valid"] is True
    assert rows[0]["valid_for_actuation"] is False
    assert rows[0]["yaw_rate_cmd_deg_s"] == pytest.approx(0.0)
