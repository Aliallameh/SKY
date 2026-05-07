"""Convert DUT-Anti-UAV Detection VOC zips to YOLO drone-only format."""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.datasets.common_yolo import (  # noqa: E402
    IMAGE_EXTS,
    ConversionSummary,
    image_size,
    place_image,
    read_voc_boxes,
    reset_yolo_dirs,
    safe_stem,
    write_conversion_summary,
    write_data_yaml,
    write_label_file,
    xyxy_to_yolo,
)


CLASS_NAMES = ["drone"]
RAW_TO_TARGET = {"uav": 0, "drone": 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DUT-Anti-UAV Detection VOC zips to YOLO.")
    parser.add_argument("--root", default="DATASETS/2_DUT-Anti-UAV")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--extract-dir", default="data/training/raw_cache/dut_antiuav_detection")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--link-mode", choices=["copy", "hardlink", "symlink"], default="copy")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    extract_dir = Path(args.extract_dir).resolve()
    if not root.exists():
        raise SystemExit(f"DUT-Anti-UAV root not found: {root}")
    reset_yolo_dirs(out_dir)
    write_data_yaml(out_dir, CLASS_NAMES)
    summary = ConversionSummary(
        dataset="dut_anti_uav_detection",
        source_root=str(root),
        output_dir=str(out_dir),
        classes=CLASS_NAMES,
        class_remap={"UAV": "0 drone", "uav": "0 drone", "drone": "0 drone"},
        limit=args.limit,
        split_policy="official Detection train/val/test zips; no frame-level reshuffle",
        notes=["Detection VOC zips are extracted to data/training/raw_cache/dut_antiuav_detection."],
    )
    extracted = extract_detection_zips(root, extract_dir, summary)
    written = 0
    for split in ("train", "val", "test"):
        if args.limit is not None and written >= args.limit:
            break
        split_root = extracted.get(split)
        if split_root is None:
            summary.add_skip(root, f"missing {split}.zip extraction")
            continue
        written += convert_split(split_root, split, out_dir, summary, args, remaining=None if args.limit is None else args.limit - written)
    write_conversion_summary(out_dir, summary)
    print(json.dumps({"out_dir": str(out_dir), "summary": str(out_dir / "conversion_summary.json"), "images": summary.image_count}, indent=2))
    return 0


def extract_detection_zips(root: Path, extract_dir: Path, summary: ConversionSummary) -> Dict[str, Path]:
    detection_dir = next((p for p in root.iterdir() if p.is_dir() and "Detection" in p.name), root)
    found: Dict[str, Path] = {}
    extract_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        existing_split = detection_dir / split
        if existing_split.exists():
            found[split] = existing_split
            summary.notes.append(f"Using already-extracted DUT Detection folder: {existing_split}")
            continue
        zip_path = detection_dir / f"{split}.zip"
        if not zip_path.exists():
            summary.add_skip(zip_path, "Detection zip missing")
            continue
        target = extract_dir / split
        marker = target / ".skyscout_extracted"
        if not marker.exists():
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(target)
            marker.write_text(str(zip_path), encoding="utf-8")
        found[split] = target
    return found


def convert_split(split_root: Path, split: str, out_dir: Path, summary: ConversionSummary, args: argparse.Namespace, remaining: Optional[int]) -> int:
    images = sorted(p for p in split_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    xml_by_stem = {p.stem: p for p in split_root.rglob("*.xml")}
    written = 0
    for image_path in images:
        if remaining is not None and written >= remaining:
            break
        size = image_size(image_path)
        if size is None:
            summary.add_skip(image_path, "corrupt/unreadable image")
            continue
        width, height = size
        xml_path = xml_by_stem.get(image_path.stem)
        boxes = []
        if xml_path is not None:
            try:
                xml_w, xml_h, objects = read_voc_boxes(xml_path)
            except Exception as exc:
                summary.add_skip(xml_path, f"invalid VOC XML: {exc}")
                continue
            width = xml_w or width
            height = xml_h or height
            for label, bbox in objects:
                target = RAW_TO_TARGET.get(label.strip().lower())
                if target is None:
                    continue
                box = xyxy_to_yolo(target, *bbox, width=width, height=height, source_label=label)
                if box is None:
                    summary.add_invalid_box(str(xml_path), bbox, "degenerate/clipped VOC box")
                    continue
                boxes.append(box)
        stem = safe_stem(f"dut_anti_uav_{split}_{image_path.relative_to(split_root)}")
        dst_image = out_dir / "images" / split / f"{stem}{image_path.suffix.lower()}"
        dst_label = out_dir / "labels" / split / f"{stem}.txt"
        place_image(image_path, dst_image, args.link_mode)
        write_label_file(dst_label, boxes)
        summary.add_image(split, boxes, width, height)
        written += 1
    return written


if __name__ == "__main__":
    raise SystemExit(main())
