"""Shared utilities for gated SkyScouter YOLO dataset converters."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
SPLITS = ("train", "val", "test")
DEFAULT_SPLIT_FRACTIONS = (0.80, 0.15, 0.05)
BBOX_SIZE_BINS = {
    "tiny": "bbox_area/image_area < 0.0005",
    "small": "0.0005 <= bbox_area/image_area < 0.0025",
    "medium": "bbox_area/image_area >= 0.0025",
}


@dataclass
class YoloBox:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    source_label: str = ""


@dataclass
class ConversionSummary:
    schema_version: str = "skyscout.yolo_conversion_summary.v1"
    dataset: str = ""
    source_root: str = ""
    output_dir: str = ""
    classes: List[str] = field(default_factory=list)
    class_remap: Dict[str, object] = field(default_factory=dict)
    split_policy: str = "sequence_or_group_hash_seed_42"
    limit: Optional[int] = None
    image_count: int = 0
    label_file_count: int = 0
    object_count: int = 0
    empty_label_count: int = 0
    split_counts: Dict[str, Dict[str, object]] = field(default_factory=dict)
    object_count_per_class: Dict[str, int] = field(default_factory=dict)
    bbox_size_bins: Dict[str, int] = field(default_factory=lambda: {"tiny": 0, "small": 0, "medium": 0})
    skipped_files: List[Dict[str, str]] = field(default_factory=list)
    invalid_boxes: List[Dict[str, object]] = field(default_factory=list)
    held_out_groups: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.split_counts:
            self.split_counts = {
                split: {"images": 0, "objects": 0, "empty_labels": 0, "class_counts": {}}
                for split in SPLITS
            }

    def add_image(self, split: str, boxes: Sequence[YoloBox], width: int, height: int) -> None:
        self.image_count += 1
        self.label_file_count += 1
        self.object_count += len(boxes)
        self.split_counts[split]["images"] = int(self.split_counts[split]["images"]) + 1
        self.split_counts[split]["objects"] = int(self.split_counts[split]["objects"]) + len(boxes)
        if not boxes:
            self.empty_label_count += 1
            self.split_counts[split]["empty_labels"] = int(self.split_counts[split]["empty_labels"]) + 1
        for box in boxes:
            class_name = self.classes[box.class_id] if 0 <= box.class_id < len(self.classes) else str(box.class_id)
            self.object_count_per_class[class_name] = self.object_count_per_class.get(class_name, 0) + 1
            split_counts = self.split_counts[split]["class_counts"]
            split_counts[class_name] = split_counts.get(class_name, 0) + 1
            self.bbox_size_bins[bbox_size_bin(box.w * box.h)] += 1

    def add_skip(self, path: Path | str, reason: str) -> None:
        self.skipped_files.append({"path": str(path), "reason": reason})

    def add_invalid_box(self, source: str, box: object, reason: str) -> None:
        self.invalid_boxes.append({"source": source, "box": box, "reason": reason})


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path, base: Optional[Path] = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base or repo_root()) / path
    return path.resolve()


def safe_stem(value: str, max_len: int = 180) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_")
    if len(cleaned) > max_len:
        digest = stable_digest(cleaned)[:12]
        cleaned = f"{cleaned[: max_len - 13]}_{digest}"
    return cleaned or "sample"


def stable_digest(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="replace")).hexdigest()


def stable_fraction(value: str, seed: int = 42) -> float:
    digest = stable_digest(f"{seed}:{value}")
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def choose_split(group_key: str, fractions: Sequence[float] = DEFAULT_SPLIT_FRACTIONS, seed: int = 42) -> str:
    value = stable_fraction(group_key, seed)
    train, val, _test = fractions
    if value < train:
        return "train"
    if value < train + val:
        return "val"
    return "test"


def bbox_size_bin(area_ratio: float) -> str:
    if area_ratio < 0.0005:
        return "tiny"
    if area_ratio < 0.0025:
        return "small"
    return "medium"


def reset_yolo_dirs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("images", "labels"):
        for split in SPLITS:
            target = out_dir / kind / split
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)


def write_data_yaml(out_dir: Path, class_names: Sequence[str]) -> None:
    payload = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_conversion_summary(out_dir: Path, summary: ConversionSummary) -> None:
    (out_dir / "conversion_summary.json").write_text(
        json.dumps(asdict(summary), indent=2),
        encoding="utf-8",
    )


def write_label_file(path: Path, boxes: Sequence[YoloBox]) -> None:
    lines = [f"{b.class_id} {b.cx:.8f} {b.cy:.8f} {b.w:.8f} {b.h:.8f}" for b in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def place_image(src: Path, dst: Path, link_mode: str = "copy") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link_mode == "copy":
        shutil.copy2(src, dst)
    elif link_mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    elif link_mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown link mode: {link_mode}")


def write_image_bgr(path: Path, image_bgr: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"Failed to write image: {path}")


def image_size(path: Path) -> Optional[Tuple[int, int]]:
    img = cv2.imread(str(path))
    if img is None:
        return None
    height, width = img.shape[:2]
    return width, height


def clip_xyxy(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
    x1 = max(0.0, min(float(width), float(x1)))
    x2 = max(0.0, min(float(width), float(x2)))
    y1 = max(0.0, min(float(height), float(y1)))
    y2 = max(0.0, min(float(height), float(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def xyxy_to_yolo(
    class_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
    source_label: str = "",
) -> Optional[YoloBox]:
    clipped = clip_xyxy(x1, y1, x2, y2, width, height)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    return YoloBox(
        class_id=class_id,
        cx=((x1 + x2) * 0.5) / width,
        cy=((y1 + y2) * 0.5) / height,
        w=(x2 - x1) / width,
        h=(y2 - y1) / height,
        source_label=source_label,
    )


def xywh_to_yolo(
    class_id: int,
    x: float,
    y: float,
    w: float,
    h: float,
    width: int,
    height: int,
    source_label: str = "",
) -> Optional[YoloBox]:
    return xyxy_to_yolo(class_id, x, y, x + w, y + h, width, height, source_label=source_label)


def parse_yolo_line(line: str, width: int, height: int, class_map: Dict[int, int]) -> Optional[YoloBox]:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    source_id = int(float(parts[0]))
    if source_id not in class_map:
        return None
    cx, cy, w, h = [float(v) for v in parts[1:5]]
    if not all(math.isfinite(v) for v in (cx, cy, w, h)):
        return None
    if w <= 0 or h <= 0:
        return None
    x1 = (cx - w * 0.5) * width
    y1 = (cy - h * 0.5) * height
    x2 = (cx + w * 0.5) * width
    y2 = (cy + h * 0.5) * height
    return xyxy_to_yolo(class_map[source_id], x1, y1, x2, y2, width, height, source_label=str(source_id))


def read_voc_boxes(xml_path: Path) -> Tuple[Optional[int], Optional[int], List[Tuple[str, Tuple[float, float, float, float]]]]:
    root = ET.parse(xml_path).getroot()
    width = height = None
    size = root.find("size")
    if size is not None:
        width_node = size.find("width")
        height_node = size.find("height")
        if width_node is not None and width_node.text:
            width = int(float(width_node.text))
        if height_node is not None and height_node.text:
            height = int(float(height_node.text))
    objects: List[Tuple[str, Tuple[float, float, float, float]]] = []
    for obj in root.findall("object"):
        name_node = obj.find("name")
        box_node = obj.find("bndbox")
        if name_node is None or not name_node.text or box_node is None:
            continue
        try:
            x1 = float(box_node.findtext("xmin", "nan"))
            y1 = float(box_node.findtext("ymin", "nan"))
            x2 = float(box_node.findtext("xmax", "nan"))
            y2 = float(box_node.findtext("ymax", "nan"))
        except ValueError:
            continue
        objects.append((name_node.text.strip(), (x1, y1, x2, y2)))
    return width, height, objects


def class_name_map(names: Sequence[str]) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(names)}


def normalize_label(value: object) -> str:
    text = str(value or "").strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")
