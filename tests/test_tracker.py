"""Tests for the simple IoU tracker."""
from __future__ import annotations

from skyscouter.perception.base_detector import Detection
from skyscouter.tracking.simple_tracker import SimpleIoUTracker


def _det(x, y, w, h, conf=0.9, cls=0, label="object"):
    return Detection(x=x, y=y, w=w, h=h, confidence=conf, class_id=cls, class_label=label)


def test_assigns_persistent_id_across_frames():
    tr = SimpleIoUTracker(match_threshold=0.3, track_buffer_frames=10, min_track_length=1)
    # Same object, slightly moving
    tracks_f1 = tr.update([_det(100, 100, 50, 50)], frame_index=0, capture_time_s=0.0)
    tracks_f2 = tr.update([_det(102, 101, 50, 50)], frame_index=1, capture_time_s=0.033)
    tracks_f3 = tr.update([_det(105, 102, 50, 50)], frame_index=2, capture_time_s=0.066)
    assert len(tracks_f1) == 1
    assert len(tracks_f3) == 1
    assert tracks_f1[0].track_id == tracks_f3[0].track_id


def test_separate_objects_get_separate_ids():
    tr = SimpleIoUTracker()
    tracks = tr.update(
        [_det(0, 0, 30, 30), _det(500, 500, 30, 30)],
        frame_index=0, capture_time_s=0.0,
    )
    ids = {t.track_id for t in tracks}
    assert len(ids) == 2


def test_track_dies_after_buffer():
    tr = SimpleIoUTracker(match_threshold=0.3, track_buffer_frames=2, min_track_length=1)
    tr.update([_det(100, 100, 50, 50)], frame_index=0, capture_time_s=0.0)
    # 3 frames of no detection — past the 2-frame buffer
    tr.update([], frame_index=1, capture_time_s=0.033)
    tr.update([], frame_index=2, capture_time_s=0.066)
    tracks = tr.update([], frame_index=3, capture_time_s=0.099)
    assert len(tracks) == 0


def test_track_recovers_within_buffer():
    tr = SimpleIoUTracker(match_threshold=0.3, track_buffer_frames=5, min_track_length=1)
    tracks_a = tr.update([_det(100, 100, 50, 50)], frame_index=0, capture_time_s=0.0)
    initial_id = tracks_a[0].track_id
    tr.update([], frame_index=1, capture_time_s=0.033)
    tr.update([], frame_index=2, capture_time_s=0.066)
    # Recover at same position before buffer expires
    tracks_b = tr.update([_det(100, 100, 50, 50)], frame_index=3, capture_time_s=0.099)
    # Same track ID
    assert any(t.track_id == initial_id for t in tracks_b)


def test_is_confirmed_uses_configured_min_track_length():
    tr = SimpleIoUTracker(match_threshold=0.3, track_buffer_frames=10, min_track_length=2)
    tracks_f1 = tr.update([_det(100, 100, 50, 50)], frame_index=0, capture_time_s=0.0)
    assert tracks_f1[0].is_confirmed is False

    tracks_f2 = tr.update([_det(102, 101, 50, 50)], frame_index=1, capture_time_s=0.033)
    assert tracks_f2[0].is_confirmed is True
