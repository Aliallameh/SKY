"""
End-to-end smoke test.

Runs the pipeline against a synthetic 'video' of solid-color frames using the
stub detector. Verifies wiring: frame source -> detector -> tracker -> state
machine -> writers, without needing model weights or real video.

This catches regressions in the orchestration layer.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from skyscouter.io.frame_source import VideoFileSource
from skyscouter.perception.base_detector import BaseDetector, Detection
from skyscouter.perception.stub_detector import StubDetector
from skyscouter.tracking.simple_tracker import SimpleIoUTracker
from skyscouter.lock.state_machine import StrikeReadyConfig
from skyscouter.output.run_logger import RunLogger
from skyscouter.output.target_state_writer import TargetStateJsonlWriter
from skyscouter.output.guidance_writer import GuidanceHintJsonlWriter
from skyscouter.output.bridge_writer import BridgeProposalJsonlWriter
from skyscouter.output.annotator import VideoAnnotator
from skyscouter.pipeline import Pipeline


def _make_synthetic_video(path: str, frames: int = 12, w: int = 320, h: int = 240) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 10.0, (w, h))
    assert out.isOpened()
    for i in range(frames):
        img = np.full((h, w, 3), fill_value=(50 + i, 100, 200), dtype=np.uint8)
        out.write(img)
    out.release()


class SequenceDetector(BaseDetector):
    """Test detector that emits one configured detection list per frame."""

    def __init__(self, per_frame):
        self._per_frame = list(per_frame)
        self._idx = 0

    def detect(self, image_bgr):
        if self._idx >= len(self._per_frame):
            return []
        detections = self._per_frame[self._idx]
        self._idx += 1
        return detections

    def warmup(self):
        pass

    def get_model_version(self):
        return "sequence_test"

    def get_input_size(self):
        return 320


def _guidance_cfg(*, calibration_reviewed=False):
    return {
        "enabled": True,
        "output_jsonl": True,
        "camera": {
            "mode": "fov",
            "horizontal_fov_deg": 90.0,
            "vertical_fov_deg": None,
            "frame_width_px": None,
            "frame_height_px": None,
            "fx_px": None,
            "fy_px": None,
            "cx_px": None,
            "cy_px": None,
            "assume_principal_point_center": True,
            "calibration_reviewed": calibration_reviewed,
        },
        "validity": {
            "advisory_before_lock": False,
            "allowed_lock_states": ["LOCKED", "STRIKE_READY"],
            "acceptable_class_labels": ["drone"],
            "min_confidence": 0.30,
            "require_guidance_valid_target_state": True,
            "max_stale_frames": 0,
        },
        "filtering": {
            "enabled": True,
            "ema_alpha": 1.0,
            "prediction_enabled": True,
            "lead_time_s": 0.2,
            "max_prediction_px": 250,
            "history_len": 8,
        },
        "controller": {
            "enabled": True,
            "mode": "yaw_p",
            "kp_yaw": 2.0,
            "deadband_deg": 1.0,
            "max_yaw_rate_deg_s": 45.0,
            "max_delta_yaw_rate_deg_s": None,
            "pitch_command_enabled": False,
        },
        "overlay": {"enabled": False},
    }


def _mock_bridge_cfg():
    return {
        "enabled": True,
        "output_jsonl": True,
        "require_reviewed_calibration": True,
        "allowed_lock_states": ["LOCKED", "STRIKE_READY"],
        "require_guidance_hint_valid": True,
        "max_abs_yaw_rate_deg_s": 45.0,
    }


@pytest.fixture
def smoke_dirs(tmp_path):
    vid = tmp_path / "smoke.mp4"
    out = tmp_path / "run"
    _make_synthetic_video(str(vid))
    return str(vid), str(out)


def test_pipeline_runs_with_stub_detector(smoke_dirs):
    video_path, out_dir = smoke_dirs
    cfg = {
        "schema_version": "skyscout.config.v1",
        "source": {"type": "video_file", "frame_stride": 1, "strict": True},
        "camera": {"calibration_id": "TEST", "is_calibrated": False},
        "detector": {"backend": "stub", "model_version": "stub_test"},
        "tracker": {"backend": "simple"},
        "lock": {
            "acquired_to_tracking_frames": 2,
            "tracking_to_locked_frames": 2,
            "min_class_confidence": 0.3,
            "min_lock_quality": 0.5,
            "lost_to_aborted_seconds": 1.0,
            "strike_ready": {
                "min_locked_duration_seconds": 0.5,
                "min_bbox_frame_fraction": 0.05,
                "bbox_center_window": 0.5,
                "max_cue_age_seconds": None,
            },
        },
        "output": {
            "save_annotated_video": True,
            "save_target_states_jsonl": True,
            "save_manifest": True,
        },
    }

    src = VideoFileSource(video_path, frame_stride=1, strict=True)
    det = StubDetector()
    trk = SimpleIoUTracker()
    logger = RunLogger(out_dir)
    logger.set_config(cfg)

    w, h = src.get_resolution()
    fps = src.get_fps() or 10.0

    annot_path = Path(out_dir) / "annotated.mp4"
    jsonl_path = Path(out_dir) / "target_states.jsonl"
    annot = VideoAnnotator(str(annot_path), w, h, fps=fps)
    writer = TargetStateJsonlWriter(str(jsonl_path))

    try:
        with writer:
            pipe = Pipeline(
                config=cfg,
                frame_source=src,
                detector=det,
                tracker=trk,
                run_logger=logger,
                target_state_writer=writer,
                annotator=annot,
            )
            pipe.run()
    finally:
        annot.close()
        src.close()
        logger.finalize()

    # Verify outputs
    assert annot_path.exists()
    assert jsonl_path.exists()
    assert (Path(out_dir) / "manifest.json").exists()
    assert (Path(out_dir) / "run.log").exists()

    # Each line should be a valid TargetState JSON; with no detections,
    # message_type is NO_TARGET and lock_state is SEARCHING.
    with jsonl_path.open() as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) > 0
    for ts in lines:
        assert ts["schema_version"] == "skyscout.onboard_target.v1"
        assert ts["message_type"] in ("TARGET_STATE", "NO_TARGET", "HEARTBEAT", "SYSTEM_FAULT")
        assert ts["guidance_valid"] is False  # stub detector -> never guidance-valid


def test_pipeline_revokes_guidance_when_track_is_not_updated(tmp_path):
    video_path = tmp_path / "loss.mp4"
    out_dir = tmp_path / "run"
    _make_synthetic_video(str(video_path), frames=6)

    det = Detection(
        x=140,
        y=100,
        w=40,
        h=40,
        confidence=0.95,
        class_id=0,
        class_label="drone",
    )
    detector = SequenceDetector([[det], [det], [det], [det], [], []])
    cfg = {
        "schema_version": "skyscout.config.v1",
        "source": {"type": "video_file", "frame_stride": 1, "strict": True},
        "camera": {"calibration_id": "TEST", "is_calibrated": False},
        "detector": {"backend": "sequence", "model_version": "sequence_test"},
        "tracker": {
            "backend": "simple",
            "match_threshold": 0.3,
            "track_buffer_frames": 5,
            "min_track_length": 1,
        },
        "lock": {
            "acquired_to_tracking_frames": 1,
            "tracking_to_locked_frames": 1,
            "min_class_confidence": 0.3,
            "min_lock_quality": 0.0,
            "lost_to_aborted_seconds": 1.0,
            "strike_ready": {
                "min_locked_duration_seconds": 10.0,
                "min_bbox_frame_fraction": 0.05,
                "bbox_center_window": 0.5,
                "max_cue_age_seconds": None,
            },
        },
        "output": {"save_target_states_jsonl": True},
    }

    src = VideoFileSource(str(video_path), frame_stride=1, strict=True)
    trk = SimpleIoUTracker(match_threshold=0.3, track_buffer_frames=5, min_track_length=1)
    logger = RunLogger(str(out_dir))
    logger.set_config(cfg)
    jsonl_path = out_dir / "target_states.jsonl"
    writer = TargetStateJsonlWriter(str(jsonl_path))

    try:
        with writer:
            pipe = Pipeline(
                config=cfg,
                frame_source=src,
                detector=detector,
                tracker=trk,
                run_logger=logger,
                target_state_writer=writer,
                annotator=None,
            )
            pipe.run()
    finally:
        src.close()
        logger.finalize()

    with jsonl_path.open() as f:
        states = [json.loads(l) for l in f if l.strip()]

    assert any(s["lock_state"] == "LOCKED" and s["guidance_valid"] for s in states)
    lost_states = [s for s in states if s["lock_state"] == "LOST"]
    assert lost_states
    assert all(s["guidance_valid"] is False for s in lost_states)


def test_pipeline_writes_guidance_hints_with_sequence_detector(tmp_path):
    video_path = tmp_path / "guidance.mp4"
    out_dir = tmp_path / "run"
    _make_synthetic_video(str(video_path), frames=6, w=320, h=240)

    detections = [
        Detection(x=150, y=110, w=20, h=20, confidence=0.95, class_id=0, class_label="drone")
        for _ in range(6)
    ]
    detector = SequenceDetector([[d] for d in detections])
    cfg = {
        "schema_version": "skyscout.config.v1",
        "source": {"type": "video_file", "frame_stride": 1, "strict": True},
        "camera": {"calibration_id": "TEST", "is_calibrated": False},
        "detector": {"backend": "sequence", "model_version": "sequence_test"},
        "tracker": {
            "backend": "simple",
            "match_threshold": 0.3,
            "track_buffer_frames": 5,
            "min_track_length": 1,
        },
        "lock": {
            "acceptable_lock_labels": ["drone"],
            "acquired_to_tracking_frames": 1,
            "tracking_to_locked_frames": 1,
            "min_class_confidence": 0.3,
            "min_lock_quality": 0.0,
            "lost_to_aborted_seconds": 1.0,
            "strike_ready": {
                "min_locked_duration_seconds": 10.0,
                "min_bbox_frame_fraction": 0.01,
                "bbox_center_window": 1.0,
                "max_cue_age_seconds": None,
            },
        },
        "guidance": _guidance_cfg(),
        "output": {"save_target_states_jsonl": True},
    }

    src = VideoFileSource(str(video_path), frame_stride=1, strict=True)
    trk = SimpleIoUTracker(match_threshold=0.3, track_buffer_frames=5, min_track_length=1)
    logger = RunLogger(str(out_dir), run_id="run_guidance")
    logger.set_config(cfg)
    target_path = out_dir / "target_states.jsonl"
    guidance_path = out_dir / "guidance_hints.jsonl"
    target_writer = TargetStateJsonlWriter(str(target_path))
    guidance_writer = GuidanceHintJsonlWriter(str(guidance_path))

    try:
        with target_writer, guidance_writer:
            pipe = Pipeline(
                config=cfg,
                frame_source=src,
                detector=detector,
                tracker=trk,
                run_logger=logger,
                target_state_writer=target_writer,
                guidance_hint_writer=guidance_writer,
                annotator=None,
            )
            pipe.run()
    finally:
        src.close()
        logger.finalize()

    hints = [json.loads(line) for line in guidance_path.read_text(encoding="utf-8").splitlines()]
    assert hints
    assert any(h["valid"] for h in hints)
    valid = [h for h in hints if h["valid"]][0]
    assert valid["schema_version"] == "skyscout.guidance_hint.v1"
    assert valid["valid_for_actuation"] is False
    assert valid["bearing_error_deg"] == pytest.approx(0.0)


def test_pipeline_writes_suppressed_mock_bridge_rows_when_calibration_unreviewed(tmp_path):
    states, bridge_rows = _run_guidance_bridge_pipeline(
        tmp_path,
        calibration_reviewed=False,
    )
    assert any(s["lock_state"] == "LOCKED" and s["guidance_valid"] for s in states)
    assert bridge_rows
    assert all(row["valid_for_transport"] is False for row in bridge_rows)
    assert any("calibration_not_reviewed" in row["reason"] for row in bridge_rows)


def test_pipeline_writes_transport_valid_mock_bridge_rows_when_calibration_reviewed(tmp_path):
    _, bridge_rows = _run_guidance_bridge_pipeline(
        tmp_path,
        calibration_reviewed=True,
    )
    assert bridge_rows
    valid_rows = [row for row in bridge_rows if row["valid_for_transport"]]
    assert valid_rows
    assert valid_rows[0]["schema_version"] == "skyscout.mock_bridge_proposal.v1"
    assert valid_rows[0]["calibration_reviewed"] is True
    assert valid_rows[0]["pitch_rate_cmd_deg_s"] is None


def test_mock_bridge_enabled_requires_guidance_enabled(tmp_path):
    video_path = tmp_path / "invalid_bridge.mp4"
    _make_synthetic_video(str(video_path), frames=1)
    cfg = {
        "schema_version": "skyscout.config.v1",
        "source": {"type": "video_file", "frame_stride": 1, "strict": True},
        "camera": {"calibration_id": "TEST", "is_calibrated": False},
        "detector": {"backend": "stub", "model_version": "stub_test"},
        "tracker": {"backend": "simple"},
        "lock": {
            "acquired_to_tracking_frames": 1,
            "tracking_to_locked_frames": 1,
            "min_class_confidence": 0.3,
            "min_lock_quality": 0.0,
            "lost_to_aborted_seconds": 1.0,
            "strike_ready": {
                "min_locked_duration_seconds": 10.0,
                "min_bbox_frame_fraction": 0.01,
                "bbox_center_window": 1.0,
                "max_cue_age_seconds": None,
            },
        },
        "guidance": {"enabled": False},
        "mock_bridge": _mock_bridge_cfg(),
        "output": {"save_target_states_jsonl": True},
    }
    src = VideoFileSource(str(video_path), frame_stride=1, strict=True)
    logger = RunLogger(str(tmp_path / "run_invalid"))
    try:
        with pytest.raises(ValueError, match="requires guidance.enabled=true"):
            Pipeline(
                config=cfg,
                frame_source=src,
                detector=StubDetector(),
                tracker=SimpleIoUTracker(min_track_length=1),
                run_logger=logger,
            )
    finally:
        src.close()
        logger.finalize()


def _run_guidance_bridge_pipeline(tmp_path, *, calibration_reviewed):
    video_path = tmp_path / f"bridge_{calibration_reviewed}.mp4"
    out_dir = tmp_path / f"run_bridge_{calibration_reviewed}"
    _make_synthetic_video(str(video_path), frames=6, w=320, h=240)
    detections = [
        Detection(x=155, y=110, w=20, h=20, confidence=0.95, class_id=0, class_label="drone")
        for _ in range(6)
    ]
    detector = SequenceDetector([[d] for d in detections])
    cfg = {
        "schema_version": "skyscout.config.v1",
        "source": {"type": "video_file", "frame_stride": 1, "strict": True},
        "camera": {"calibration_id": "TEST_CAL", "is_calibrated": False},
        "detector": {"backend": "sequence", "model_version": "sequence_test"},
        "tracker": {
            "backend": "simple",
            "match_threshold": 0.3,
            "track_buffer_frames": 5,
            "min_track_length": 1,
        },
        "lock": {
            "acceptable_lock_labels": ["drone"],
            "acquired_to_tracking_frames": 1,
            "tracking_to_locked_frames": 1,
            "min_class_confidence": 0.3,
            "min_lock_quality": 0.0,
            "lost_to_aborted_seconds": 1.0,
            "strike_ready": {
                "min_locked_duration_seconds": 10.0,
                "min_bbox_frame_fraction": 0.01,
                "bbox_center_window": 1.0,
                "max_cue_age_seconds": None,
            },
        },
        "guidance": _guidance_cfg(calibration_reviewed=calibration_reviewed),
        "mock_bridge": _mock_bridge_cfg(),
        "output": {"save_target_states_jsonl": True},
    }
    src = VideoFileSource(str(video_path), frame_stride=1, strict=True)
    trk = SimpleIoUTracker(match_threshold=0.3, track_buffer_frames=5, min_track_length=1)
    logger = RunLogger(str(out_dir), run_id="run_bridge")
    logger.set_config(cfg)
    target_path = out_dir / "target_states.jsonl"
    guidance_path = out_dir / "guidance_hints.jsonl"
    bridge_path = out_dir / "mock_bridge_proposals.jsonl"
    target_writer = TargetStateJsonlWriter(str(target_path))
    guidance_writer = GuidanceHintJsonlWriter(str(guidance_path))
    bridge_writer = BridgeProposalJsonlWriter(str(bridge_path))
    try:
        with target_writer, guidance_writer, bridge_writer:
            pipe = Pipeline(
                config=cfg,
                frame_source=src,
                detector=detector,
                tracker=trk,
                run_logger=logger,
                target_state_writer=target_writer,
                guidance_hint_writer=guidance_writer,
                bridge_proposal_writer=bridge_writer,
                annotator=None,
            )
            pipe.run()
    finally:
        src.close()
        logger.finalize()

    states = [json.loads(line) for line in target_path.read_text(encoding="utf-8").splitlines()]
    rows = [json.loads(line) for line in bridge_path.read_text(encoding="utf-8").splitlines()]
    assert guidance_path.exists()
    assert bridge_path.exists()
    return states, rows
