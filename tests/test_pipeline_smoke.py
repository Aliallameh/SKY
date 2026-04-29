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
