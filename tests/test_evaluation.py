from __future__ import annotations

import csv

from skyscouter.output.evaluation import EvaluationCollector
from skyscouter.schemas import TargetState


def _state(frame, bbox, track_id=1, lock_state="LOCKED", guidance=True):
    return TargetState(
        timestamp_utc=f"frame-{frame}",
        track_id=track_id,
        lock_state=lock_state,
        guidance_valid=guidance,
        bbox_xywh=bbox,
        confidence=0.8,
        lock_quality=0.9,
    )


def test_evaluation_report_with_sparse_gt(tmp_path):
    gt = tmp_path / "gt.csv"
    with gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_id",
                "x1",
                "y1",
                "x2",
                "y2",
                "review_status",
                "visibility",
                "occluded",
                "label",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "frame_id": 0,
            "x1": 10,
            "y1": 10,
            "x2": 30,
            "y2": 30,
            "review_status": "ok",
            "visibility": 1,
            "occluded": 0,
            "label": "drone",
            "notes": "",
        })
        writer.writerow({
            "frame_id": 1,
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": "",
            "review_status": "ok",
            "visibility": 0,
            "occluded": 0,
            "label": "negative",
            "notes": "",
        })

    ev = EvaluationCollector(str(gt))
    ev.add(0, _state(0, [10, 10, 20, 20], guidance=True))
    ev.add(1, _state(1, None, lock_state="SEARCHING", guidance=False))
    report = ev.build_report()

    assert report["gt"]["enabled"] is True
    assert report["gt"]["matched_positive_rate"] == 1.0
    assert report["gt"]["median_center_error_px"] == 0.0
    assert report["gt"]["false_lock_negative_frames"] == 0
