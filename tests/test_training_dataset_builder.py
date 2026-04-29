from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


def test_prepare_airborne_training_set_from_sparse_gt_video(tmp_path):
    video = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for idx in range(3):
        frame = np.full((48, 64, 3), 20 + idx * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    gt = tmp_path / "gt.csv"
    with gt.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(
            f,
            fieldnames=[
                "frame_id",
                "image",
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
        writer_csv.writeheader()
        writer_csv.writerow({
            "frame_id": 0,
            "image": "",
            "x1": 10,
            "y1": 12,
            "x2": 30,
            "y2": 28,
            "review_status": "ok",
            "visibility": 1,
            "occluded": 0,
            "label": "drone",
            "notes": "",
        })
        writer_csv.writerow({
            "frame_id": 1,
            "image": "",
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

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "test",
                "target_classes": ["drone", "bird", "airplane", "helicopter", "unknown_airborne"],
                "split": {"train": 1.0, "val": 0.0, "test": 0.0, "seed": 1},
                "sources": [
                    {
                        "name": "local",
                        "enabled": True,
                        "type": "skyscouter_sparse_gt_video",
                        "video_path": str(video),
                        "gt_csv": str(gt),
                        "include_labels": ["drone", "negative"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_airborne_training_set.py",
            "--manifest",
            str(manifest),
            "--out-dir",
            str(out_dir),
            "--link-mode",
            "copy",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    summary = json.loads((out_dir / "manifest_summary.json").read_text(encoding="utf-8"))
    assert summary["total_images"] == 2
    assert summary["splits"]["train"]["images"] == 2
    assert summary["splits"]["train"]["class_counts"]["drone"] == 1
    assert summary["splits"]["train"]["negatives"] == 1
    assert (out_dir / "data.yaml").exists()
    labels = sorted((out_dir / "labels" / "train").glob("*.txt"))
    assert len(labels) == 2
    assert any(label.read_text(encoding="utf-8").strip() == "" for label in labels)
