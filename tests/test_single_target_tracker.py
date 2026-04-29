from __future__ import annotations

import numpy as np

from skyscouter.perception.base_detector import Detection
from skyscouter.tracking.single_target_kalman_lk import SingleTargetKalmanLKTracker


def _det(x, y, w=30, h=20, conf=0.8, label="airborne_candidate"):
    return Detection(
        x=float(x),
        y=float(y),
        w=float(w),
        h=float(h),
        confidence=float(conf),
        class_id=0,
        class_label=label,
        source="test",
    )


def _frame_with_box(x, y, w=30, h=20):
    img = np.full((240, 320, 3), 180, dtype=np.uint8)
    img[int(y):int(y + h), int(x):int(x + w)] = 30
    return img


def test_single_target_tracker_keeps_identity_with_distractor():
    trk = SingleTargetKalmanLKTracker(min_track_length=2, reacquisition_radius_px=80)
    frame = _frame_with_box(50, 60)
    t1 = trk.update([_det(50, 60)], frame_index=0, capture_time_s=0.0, image_bgr=frame)[0]
    initial_id = t1.track_id

    frame2 = _frame_with_box(55, 62)
    tracks = trk.update(
        [_det(250, 150, conf=0.99), _det(55, 62, conf=0.75)],
        frame_index=1,
        capture_time_s=1 / 30,
        image_bgr=frame2,
    )
    assert tracks[0].track_id == initial_id
    assert tracks[0].detection.cx < 120
    assert tracks[0].is_confirmed


def test_single_target_tracker_predicts_through_short_dropout():
    trk = SingleTargetKalmanLKTracker(min_track_length=1, track_buffer_frames=5, lk_min_points=1)
    t1 = trk.update([_det(50, 60)], frame_index=0, capture_time_s=0.0, image_bgr=_frame_with_box(50, 60))[0]
    tracks = trk.update([], frame_index=1, capture_time_s=1 / 30, image_bgr=_frame_with_box(52, 60))
    assert len(tracks) == 1
    assert tracks[0].track_id == t1.track_id
    assert tracks[0].time_since_update >= 1
    assert tracks[0].status in {"flow", "predicted"}


def test_single_target_tracker_reacquires_near_prediction():
    trk = SingleTargetKalmanLKTracker(min_track_length=1, track_buffer_frames=5, reacquisition_radius_px=90)
    t1 = trk.update([_det(50, 60)], frame_index=0, capture_time_s=0.0, image_bgr=_frame_with_box(50, 60))[0]
    trk.update([], frame_index=1, capture_time_s=1 / 30, image_bgr=_frame_with_box(52, 60))
    tracks = trk.update([_det(58, 63)], frame_index=2, capture_time_s=2 / 30, image_bgr=_frame_with_box(58, 63))
    assert tracks[0].track_id == t1.track_id
    assert tracks[0].time_since_update == 0
    assert tracks[0].status == "detected"


def test_single_target_tracker_rescues_from_boundary_clutter():
    trk = SingleTargetKalmanLKTracker(min_track_length=1, reacquisition_radius_px=80)
    trk.update([_det(8, 160, w=12, h=14, conf=0.64)], frame_index=0, capture_time_s=0.0, image_bgr=_frame_with_box(8, 160, 12, 14))
    trk.update([_det(2, 161, w=12, h=14, conf=0.64)], frame_index=1, capture_time_s=1 / 30, image_bgr=_frame_with_box(2, 161, 12, 14))
    tracks = trk.update(
        [_det(0, 163, w=12, h=14, conf=0.64), _det(199, 56, w=86, h=49, conf=0.81)],
        frame_index=2,
        capture_time_s=2 / 30,
        image_bgr=_frame_with_box(199, 56, 86, 49),
    )
    assert tracks[0].detection.x > 150
    assert tracks[0].detection.y < 100
