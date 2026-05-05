"""Render YOLO label previews for visual gate review."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.datasets.common_yolo import IMAGE_EXTS, SPLITS, bbox_size_bin, reset_yolo_dirs, safe_stem  # noqa: E402


@dataclass
class Obj:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class ImageRecord:
    split: str
    image: Path
    label: Path
    objects: List[Obj]


PALETTE = {
    0: (0, 220, 255),
    1: (70, 220, 70),
    2: (255, 160, 40),
    3: (230, 90, 255),
    4: (180, 180, 180),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render YOLO label previews.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--random-per-split", type=int, default=50)
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--tiny", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_yaml = Path(args.data).resolve()
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(cfg.get("path", data_yaml.parent)).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()
    names = normalize_names(cfg.get("names", {}))
    out_dir = Path(args.out_dir).resolve()
    prepare_preview_dir(out_dir)
    records = collect_records(dataset_root)
    rng = random.Random(args.seed)
    selected: Dict[str, ImageRecord] = {}

    for split in SPLITS:
        split_records = [r for r in records if r.split == split]
        rng.shuffle(split_records)
        for record in split_records[: args.random_per_split]:
            selected[f"random_{split}_{record.image.stem}"] = record

    by_class: Dict[int, List[ImageRecord]] = defaultdict(list)
    tiny_records: List[ImageRecord] = []
    for record in records:
        classes = {obj.class_id for obj in record.objects}
        for class_id in classes:
            by_class[class_id].append(record)
        if any(bbox_size_bin(obj.w * obj.h) == "tiny" for obj in record.objects):
            tiny_records.append(record)
    for class_id, class_records in by_class.items():
        rng.shuffle(class_records)
        for record in class_records[: args.per_class]:
            selected[f"class_{class_id}_{record.split}_{record.image.stem}"] = record
    rng.shuffle(tiny_records)
    for record in tiny_records[: args.tiny]:
        selected[f"tiny_{record.split}_{record.image.stem}"] = record

    written = []
    for key, record in sorted(selected.items()):
        dst = out_dir / f"{safe_stem(key)}.jpg"
        if render_record(record, names, dst):
            written.append(str(dst))
    manifest = {
        "schema_version": "skyscout.yolo_preview_manifest.v1",
        "data_yaml": str(data_yaml),
        "dataset_root": str(dataset_root),
        "out_dir": str(out_dir),
        "images_rendered": len(written),
        "samples": written[:50],
    }
    (out_dir / "preview_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "images_rendered": len(written), "manifest": str(out_dir / "preview_manifest.json")}, indent=2))
    return 0


def prepare_preview_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob("*.jpg"):
        path.unlink()


def normalize_names(raw: object) -> List[str]:
    if isinstance(raw, dict):
        return [str(raw[k]) for k in sorted(raw, key=lambda x: int(x))]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    raise ValueError("data.yaml names must be list or dict")


def collect_records(dataset_root: Path) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        if not image_dir.exists() or not label_dir.exists():
            continue
        for image_path in sorted(p for p in image_dir.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            records.append(ImageRecord(split=split, image=image_path, label=label_path, objects=parse_objects(label_path)))
    return records


def parse_objects(label_path: Path) -> List[Obj]:
    objects: List[Obj] = []
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            objects.append(Obj(int(float(parts[0])), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
        except ValueError:
            continue
    return objects


def render_record(record: ImageRecord, names: Sequence[str], dst: Path) -> bool:
    image = cv2.imread(str(record.image))
    if image is None:
        return False
    height, width = image.shape[:2]
    overlay = image.copy()
    for obj in record.objects:
        x1 = int(round((obj.cx - obj.w * 0.5) * width))
        y1 = int(round((obj.cy - obj.h * 0.5) * height))
        x2 = int(round((obj.cx + obj.w * 0.5) * width))
        y2 = int(round((obj.cy + obj.h * 0.5) * height))
        color = PALETTE.get(obj.class_id, (220, 220, 220))
        label = names[obj.class_id] if 0 <= obj.class_id < len(names) else str(obj.class_id)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, max(2, int(round(width / 640))))
        tag = f"{label} {max(1, x2 - x1)}x{max(1, y2 - y1)} {bbox_size_bin(obj.w * obj.h)}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(th + 4, y1 - 6)
        cv2.rectangle(overlay, (x1, ty - th - 6), (x1 + tw + 8, ty + 4), (20, 20, 20), -1)
        cv2.putText(overlay, tag, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    header = f"{record.split} | {record.image.name} | boxes={len(record.objects)}"
    cv2.rectangle(overlay, (0, 0), (min(width, 900), 34), (20, 20, 20), -1)
    cv2.putText(overlay, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(dst), overlay))


if __name__ == "__main__":
    raise SystemExit(main())
