"""Convert AOD-4 YOLOv8 labels to SkyScouter Stage 2 multiclass YOLO format."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.datasets.common_yolo import (  # noqa: E402
    IMAGE_EXTS,
    ConversionSummary,
    image_size,
    parse_yolo_line,
    place_image,
    reset_yolo_dirs,
    safe_stem,
    write_conversion_summary,
    write_data_yaml,
    write_label_file,
)


CLASS_NAMES = ["drone", "bird", "airplane", "helicopter"]
SOURCE_CLASS_MAP = {
    0: 2,  # airplane -> airplane
    1: 1,  # bird -> bird
    2: 0,  # drone -> drone
    3: 3,  # helicopter -> helicopter
}
SOURCE_REMAP_DOC = {"0 airplane": "2 airplane", "1 bird": "1 bird", "2 drone": "0 drone", "3 helicopter": "3 helicopter"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AOD-4 YOLOv8 labels to Stage 2 multiclass YOLO.")
    parser.add_argument("--root", default="DATASETS/5_AOD 4 Dataset for Air Borne Object Detection")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--link-mode", choices=["copy", "hardlink", "symlink"], default="copy")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = find_aod_root(Path(args.root).resolve())
    out_dir = Path(args.out_dir).resolve()
    if root is None:
        raise SystemExit(f"AOD-4 root not found under: {args.root}")
    reset_yolo_dirs(out_dir)
    write_data_yaml(out_dir, CLASS_NAMES)
    summary = ConversionSummary(
        dataset="aod4_stage2_multiclass",
        source_root=str(root),
        output_dir=str(out_dir),
        classes=CLASS_NAMES,
        class_remap=SOURCE_REMAP_DOC | {"COCO generic supercategory": "skip/not used"},
        limit=args.limit,
        split_policy="official AOD-4 train/valid/test split",
        notes=["AOD-4 must be visually audited for drone-vs-airplane tiny-object quality before Stage 2 full training."],
    )
    written = 0
    for raw_split, out_split in (("train", "train"), ("valid", "val"), ("test", "test")):
        if args.limit is not None and written >= args.limit:
            break
        written += convert_split(root, raw_split, out_split, out_dir, summary, args, remaining=None if args.limit is None else args.limit - written)
    write_conversion_summary(out_dir, summary)
    print(json.dumps({"out_dir": str(out_dir), "summary": str(out_dir / "conversion_summary.json"), "images": summary.image_count}, indent=2))
    return 0


def find_aod_root(root: Path) -> Optional[Path]:
    for candidate in (root / "AOD 4", root):
        if (candidate / "Images").exists() and (candidate / "Annotations" / "YOLOv8 format").exists():
            return candidate
    return None


def convert_split(root: Path, raw_split: str, out_split: str, out_dir: Path, summary: ConversionSummary, args: argparse.Namespace, remaining: Optional[int]) -> int:
    images_dir = root / "Images" / raw_split
    labels_dir = root / "Annotations" / "YOLOv8 format" / raw_split / "labels"
    if not images_dir.exists():
        summary.add_skip(images_dir, "AOD images split missing")
        return 0
    if not labels_dir.exists():
        summary.add_skip(labels_dir, "AOD labels split missing")
        return 0
    written = 0
    for image_path in sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
        if remaining is not None and written >= remaining:
            break
        size = image_size(image_path)
        if size is None:
            summary.add_skip(image_path, "corrupt/unreadable image")
            continue
        width, height = size
        label_path = labels_dir / f"{image_path.stem}.txt"
        boxes = []
        if label_path.exists():
            for line_no, line in enumerate(label_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    box = parse_yolo_line(stripped, width, height, SOURCE_CLASS_MAP)
                except Exception as exc:
                    summary.add_invalid_box(f"{label_path}:{line_no}", stripped, f"invalid YOLO line: {exc}")
                    continue
                if box is None:
                    parts = stripped.split()
                    source_class = parts[0] if parts else ""
                    summary.add_invalid_box(f"{label_path}:{line_no}", stripped, f"skipped source class or degenerate box: {source_class}")
                    continue
                if not all(math.isfinite(v) for v in (box.cx, box.cy, box.w, box.h)):
                    summary.add_invalid_box(f"{label_path}:{line_no}", stripped, "NaN/inf normalized box")
                    continue
                boxes.append(box)
        stem = safe_stem(f"aod4_{out_split}_{image_path.stem}")
        dst_image = out_dir / "images" / out_split / f"{stem}{image_path.suffix.lower()}"
        dst_label = out_dir / "labels" / out_split / f"{stem}.txt"
        place_image(image_path, dst_image, args.link_mode)
        write_label_file(dst_label, boxes)
        summary.add_image(out_split, boxes, width, height)
        written += 1
    return written


if __name__ == "__main__":
    raise SystemExit(main())
