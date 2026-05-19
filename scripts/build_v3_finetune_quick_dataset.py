"""Build the minimal V3 domain-adaptation fine-tune dataset.

This is intentionally separate from the full Stage 1/2/3 gated pipeline. It
uses the corrected local hard-case GT plus about 2,000 balanced public frames,
then fine-tunes from the promoted v2 checkpoint without resume=True.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.datasets.common_yolo import (  # noqa: E402
    IMAGE_EXTS,
    ConversionSummary,
    YoloBox,
    image_size,
    place_image,
    reset_yolo_dirs,
    safe_stem,
    stable_fraction,
    write_conversion_summary,
    write_data_yaml,
    write_label_file,
    xyxy_to_yolo,
)


CLASS_NAMES = ["drone", "bird", "airplane", "helicopter"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
AOD4_SOURCE_CLASS_MAP = {
    0: CLASS_TO_ID["airplane"],
    1: CLASS_TO_ID["bird"],
    2: CLASS_TO_ID["drone"],
    3: CLASS_TO_ID["helicopter"],
}
NEGATIVE_LABELS = {"negative", "hard_negative", "no_drone", "background"}


@dataclass
class Sample:
    source: str
    source_id: str
    image_path: Path
    boxes: List[YoloBox]
    split: str

    @property
    def classes(self) -> Set[int]:
        return {box.class_id for box in self.boxes}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build quick V3 fine-tune dataset.")
    parser.add_argument("--out-dir", default="data/training/airborne_yolo_v3_finetune_quick")
    parser.add_argument("--local-csv", default="annotations/camera_20260423_113401_turn_review_strict/drone_sparse_gt_corrected.csv")
    parser.add_argument("--v2-data", default="data/training/airborne_yolo_v2/data.yaml")
    parser.add_argument("--aod4-root", default="DATASETS/5_AOD 4 Dataset for Air Borne Object Detection")
    parser.add_argument("--public-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link-mode", choices=["copy", "hardlink", "symlink"], default="copy")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    reset_yolo_dirs(out_dir)
    write_data_yaml(out_dir, CLASS_NAMES)

    summary = ConversionSummary(
        dataset="airborne_yolo_v3_finetune_quick",
        source_root="local hard-case GT + balanced v2/public sample",
        output_dir=str(out_dir),
        classes=CLASS_NAMES,
        class_remap={
            "local label=drone": "0 drone",
            "local label=negative": "empty label",
            "AOD-4 0": "2 airplane",
            "AOD-4 1": "1 bird",
            "AOD-4 2": "0 drone",
            "AOD-4 3": "3 helicopter",
            "unknown_airborne": "skip",
        },
        limit=args.public_count,
        split_policy="local hard-case all train; public sample preserves source split when available",
        notes=[
            "Quick V3 domain-adaptation dataset. Full Stage 1/2 training is intentionally bypassed.",
            "Do not use resume=True; fine-tune starts from v2 best.pt.",
        ],
    )

    local_samples = load_local_hard_case(Path(args.local_csv).resolve(), summary)
    public_samples, public_note = load_public_samples(args, summary)
    selected_public = select_balanced(public_samples, args.public_count, args.seed)
    summary.notes.append(public_note)
    summary.notes.append(f"selected_public_images={len(selected_public)}")
    summary.notes.append(f"selected_public_class_image_counts={json.dumps(class_image_counts(selected_public), sort_keys=True)}")

    for sample in local_samples + selected_public:
        write_sample(out_dir, sample, args.link_mode, summary)

    write_conversion_summary(out_dir, summary)
    print(json.dumps({
        "out_dir": str(out_dir),
        "data_yaml": str(out_dir / "data.yaml"),
        "images": summary.image_count,
        "objects": summary.object_count,
        "class_counts": summary.object_count_per_class,
        "summary": str(out_dir / "conversion_summary.json"),
    }, indent=2))
    return 0


def load_local_hard_case(csv_path: Path, summary: ConversionSummary) -> List[Sample]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Local corrected GT CSV not found: {csv_path}")
    samples: List[Sample] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            status = str(row.get("review_status", "ok")).strip().lower()
            if status in {"skip", "draft", "ignore"}:
                continue
            label = str(row.get("label", "drone")).strip().lower()
            image_path = (csv_path.parent / str(row["image"])).resolve()
            size = image_size(image_path)
            if size is None:
                summary.add_skip(image_path, "local image missing/corrupt")
                continue
            width, height = size
            boxes: List[YoloBox] = []
            if label not in NEGATIVE_LABELS:
                if label != "drone":
                    summary.add_skip(image_path, f"unsupported local label: {label}")
                    continue
                try:
                    x1, y1, x2, y2 = [float(row[k]) for k in ("x1", "y1", "x2", "y2")]
                except (TypeError, ValueError):
                    summary.add_invalid_box(str(csv_path), row, "missing local box")
                    continue
                box = xyxy_to_yolo(CLASS_TO_ID["drone"], x1, y1, x2, y2, width, height, source_label="local_drone")
                if box is None:
                    summary.add_invalid_box(str(csv_path), row, "degenerate local box")
                    continue
                boxes.append(box)
            samples.append(
                Sample(
                    source="local_turn_hard_case",
                    source_id=f"frame_{int(row['frame_id']):05d}",
                    image_path=image_path,
                    boxes=boxes,
                    split="train",
                )
            )
    return samples


def load_public_samples(args: argparse.Namespace, summary: ConversionSummary) -> tuple[List[Sample], str]:
    v2_data = Path(args.v2_data).resolve()
    if v2_data.exists():
        return load_existing_yolo_public(v2_data, summary), f"public_source=existing_v2_data:{v2_data}"
    aod4_root = find_aod4_root(Path(args.aod4_root).resolve())
    if aod4_root is None:
        raise FileNotFoundError(f"Existing v2 data missing and AOD-4 root not found: {args.aod4_root}")
    return load_aod4_public(aod4_root, summary), f"public_source=AOD4_balanced_fallback:{aod4_root}"


def load_existing_yolo_public(data_yaml: Path, summary: ConversionSummary) -> List[Sample]:
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    source_names = normalize_names(cfg.get("names", {}))
    remap = {idx: CLASS_TO_ID[name] for idx, name in enumerate(source_names) if name in CLASS_TO_ID}
    samples: List[Sample] = []
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.exists() or not label_dir.exists():
            continue
        for image_path in sorted(p for p in image_dir.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            label_path = label_dir / f"{image_path.stem}.txt"
            boxes = read_normalized_yolo_boxes(label_path, remap)
            if boxes:
                samples.append(Sample("v2_training_data", f"{split}_{image_path.stem}", image_path, boxes, split))
    return samples


def load_aod4_public(aod4_root: Path, summary: ConversionSummary) -> List[Sample]:
    samples: List[Sample] = []
    for raw_split, out_split in (("train", "train"), ("valid", "val"), ("test", "test")):
        image_dir = aod4_root / "Images" / raw_split
        label_dir = aod4_root / "Annotations" / "YOLOv8 format" / raw_split / "labels"
        if not image_dir.exists() or not label_dir.exists():
            summary.add_skip(image_dir, "AOD-4 split missing")
            continue
        for image_path in sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            label_path = label_dir / f"{image_path.stem}.txt"
            boxes = read_normalized_yolo_boxes(label_path, AOD4_SOURCE_CLASS_MAP)
            if boxes:
                samples.append(Sample("aod4_public_fallback", f"{out_split}_{image_path.stem}", image_path, boxes, out_split))
    return samples


def read_normalized_yolo_boxes(label_path: Path, class_map: Dict[int, int]) -> List[YoloBox]:
    boxes: List[YoloBox] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parts = stripped.split()
            source_id = int(float(parts[0]))
            cx, cy, w, h = [float(v) for v in parts[1:5]]
        except (ValueError, IndexError):
            continue
        if source_id not in class_map or w <= 0.0 or h <= 0.0:
            continue
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            continue
        boxes.append(YoloBox(class_map[source_id], cx, cy, w, h, source_label=str(source_id)))
    return boxes


def select_balanced(samples: Sequence[Sample], target_count: int, seed: int) -> List[Sample]:
    target_per_class = max(1, target_count // len(CLASS_NAMES))
    by_class: Dict[int, List[Sample]] = {idx: [] for idx in range(len(CLASS_NAMES))}
    for sample in samples:
        for class_id in sample.classes:
            by_class[class_id].append(sample)
    for class_id in by_class:
        by_class[class_id] = sorted(
            by_class[class_id],
            key=lambda s: stable_fraction(f"{seed}:{class_id}:{s.source}:{s.source_id}"),
        )

    selected: Dict[str, Sample] = {}
    selected_class_counts = {idx: 0 for idx in range(len(CLASS_NAMES))}
    progress = True
    while progress and len(selected) < target_count:
        progress = False
        for class_id in range(len(CLASS_NAMES)):
            if selected_class_counts[class_id] >= target_per_class:
                continue
            while by_class[class_id]:
                sample = by_class[class_id].pop(0)
                key = f"{sample.source}:{sample.source_id}"
                if key in selected:
                    continue
                selected[key] = sample
                for present_class in sample.classes:
                    selected_class_counts[present_class] += 1
                progress = True
                break

    if len(selected) < target_count:
        leftovers = sorted(samples, key=lambda s: stable_fraction(f"{seed}:leftover:{s.source}:{s.source_id}"))
        for sample in leftovers:
            key = f"{sample.source}:{sample.source_id}"
            if key not in selected:
                selected[key] = sample
                if len(selected) >= target_count:
                    break
    return list(selected.values())


def write_sample(out_dir: Path, sample: Sample, link_mode: str, summary: ConversionSummary) -> None:
    size = image_size(sample.image_path)
    if size is None:
        summary.add_skip(sample.image_path, "cannot read selected image")
        return
    width, height = size
    stem = safe_stem(f"{sample.source}_{sample.source_id}")
    image_out = out_dir / "images" / sample.split / f"{stem}{sample.image_path.suffix.lower()}"
    label_out = out_dir / "labels" / sample.split / f"{stem}.txt"
    place_image(sample.image_path, image_out, link_mode)
    write_label_file(label_out, sample.boxes)
    summary.add_image(sample.split, sample.boxes, width, height)


def class_image_counts(samples: Sequence[Sample]) -> Dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    for sample in samples:
        for class_id in sample.classes:
            counts[CLASS_NAMES[class_id]] += 1
    return counts


def normalize_names(raw: object) -> List[str]:
    if isinstance(raw, dict):
        return [str(raw[k]) for k in sorted(raw, key=lambda x: int(x))]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    raise ValueError("data.yaml names must be list or dict")


def find_aod4_root(root: Path) -> Optional[Path]:
    for candidate in (root / "AOD 4", root):
        if (candidate / "Images").exists() and (candidate / "Annotations" / "YOLOv8 format").exists():
            return candidate
    return None


if __name__ == "__main__":
    raise SystemExit(main())
