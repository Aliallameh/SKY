"""Classify Drone-vs-Bird dataset usability for YOLO detection training.

The currently downloaded source appears to be crop/classification folders. This
script intentionally does not fabricate boxes. If no detection annotations are
found, it writes a summary marking the dataset for later crop-classifier or
hard-negative mining work only.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.datasets.common_yolo import IMAGE_EXTS, ConversionSummary, write_conversion_summary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect/convert Drone-vs-Bird if detection boxes exist.")
    parser.add_argument("--root", default="DATASETS/3_Drone vs Bird Aerial Object Classification Dataset")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not root.exists():
        raise SystemExit(f"Drone-vs-Bird root not found: {root}")
    folder_counts = Counter()
    for child in sorted(root.iterdir()):
        if child.is_dir():
            folder_counts[child.name] = sum(1 for p in child.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    annotation_files = [
        str(p.relative_to(root))
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in {".xml", ".json", ".txt", ".csv"}
    ][:50]
    summary = ConversionSummary(
        dataset="drone_vs_bird_classification_only",
        source_root=str(root),
        output_dir=str(out_dir),
        classes=[],
        class_remap={"drone folder": "crop-level drone only", "bird folder": "crop-level bird only"},
        limit=args.limit,
        split_policy="not converted to YOLO detection",
        notes=[
            f"classification_folder_counts={dict(folder_counts)}",
            "No bounding boxes are generated here. Use this later for hard-negative mining or a crop classifier unless detection annotations are added.",
        ],
    )
    if annotation_files:
        summary.notes.append(f"annotation_like_files_found={annotation_files[:10]}")
        summary.notes.append("Detection conversion is intentionally blocked until annotation format is manually verified.")
    else:
        summary.notes.append("No detection annotation files found; direct YOLO detection training is blocked.")
    write_conversion_summary(out_dir, summary)
    print(json.dumps({"out_dir": str(out_dir), "decision": "not_yolo_detection_training", "summary": str(out_dir / "conversion_summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
