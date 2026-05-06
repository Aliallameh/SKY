"""Merge converted YOLO sources into gated Stage 1/2/3 training datasets."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.datasets.common_yolo import (  # noqa: E402
    IMAGE_EXTS,
    ConversionSummary,
    SPLITS,
    image_size,
    place_image,
    reset_yolo_dirs,
    safe_stem,
    stable_fraction,
    write_conversion_summary,
    write_data_yaml,
    write_label_file,
    YoloBox,
)


STAGE_CLASSES = {
    "stage1": ["drone"],
    "stage2": ["drone", "bird", "airplane", "helicopter"],
    "stage3": ["drone", "bird", "airplane", "helicopter"],
}

DEFAULT_SOURCES = {
    "stage1": [
        "anti_uav_rgbt=data/training/converted/anti_uav_rgbt",
        "dut_anti_uav=data/training/converted/dut_anti_uav",
        "visiodect=data/training/converted/visiodect",
    ],
    "stage2": [
        "aod4=data/training/converted/aod4",
        "anti_uav_rgbt=data/training/converted/anti_uav_rgbt",
        "dut_anti_uav=data/training/converted/dut_anti_uav",
        "visiodect=data/training/converted/visiodect",
    ],
    "stage3": [
        "local_mavic=annotations/mavic_style_local_v3/yolo",
        "local_turn=annotations/camera_20260423_113401_turn_review_strict/yolo",
        "stage2_balanced=data/training/airborne_stage2_multiclass",
    ],
}

DEFAULT_OUT = {
    "stage1": "data/training/airborne_stage1_drone_only",
    "stage2": "data/training/airborne_stage2_multiclass",
    "stage3": "data/training/airborne_stage3_camhard_finetune",
}

DEFAULT_EXCLUDED_SOURCE_CLASSES = {
    # AOD-4 is useful as a local multiclass airborne rejection source, but its
    # drone-vs-airplane identity needs visual audit before it is allowed to
    # teach "drone". Drone identity should come from VisioDECT/Anti-UAV/DUT and
    # local Mavic-style annotations until that audit passes.
    "stage2": ["aod4:drone"],
}


@dataclass
class SourceSpec:
    name: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build staged SkyScouter YOLO datasets from converted sources.")
    parser.add_argument("--stage", choices=["stage1", "stage2", "stage3"], required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--source", action="append", default=[], help="name=path converted YOLO source. Repeatable.")
    parser.add_argument("--cap", action="append", default=[], help="name=N deterministic source image cap. Repeatable.")
    parser.add_argument(
        "--exclude-source-class",
        action="append",
        default=[],
        help=(
            "source:class_name to skip any image containing that source class. "
            "Repeatable. Example: --exclude-source-class aod4:drone"
        ),
    )
    parser.add_argument(
        "--no-default-exclusions",
        action="store_true",
        help="Disable built-in safety exclusions such as Stage 2 excluding AOD-4 drone boxes before audit.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link-mode", choices=["copy", "hardlink", "symlink"], default="copy")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    class_names = STAGE_CLASSES[args.stage]
    out_dir = Path(args.out_dir or DEFAULT_OUT[args.stage]).resolve()
    source_specs = parse_sources(args.source or DEFAULT_SOURCES[args.stage])
    caps = parse_caps(args.cap)
    excluded_source_classes = parse_excluded_source_classes(args.stage, args.exclude_source_class, args.no_default_exclusions)
    reset_yolo_dirs(out_dir)
    write_data_yaml(out_dir, class_names)
    summary = ConversionSummary(
        dataset=f"airborne_{args.stage}",
        source_root="multiple converted YOLO sources",
        output_dir=str(out_dir),
        classes=class_names,
        class_remap={"by_class_name": "source class names remapped into stage taxonomy"},
        split_policy="preserve converted source splits; source converters enforce sequence/group leakage policy",
        notes=[
            "This builder does not convert raw datasets. Run 20-sample converters, validation, and previews first.",
            "Caps are deterministic by source/image path and should be used to prevent AOD-4 domination in Stage 2.",
            "Excluded source classes skip the whole image, not just the box, so a visible excluded object is not trained as background.",
        ],
    )
    source_summaries: Dict[str, object] = {}
    for source in source_specs:
        source_summary = merge_source(
            source,
            out_dir,
            class_names,
            caps.get(source.name),
            excluded_source_classes.get(source.name, set()),
            args,
        )
        source_summaries[source.name] = source_summary
        for split, split_data in source_summary["splits"].items():
            for class_name, count in split_data.get("class_counts", {}).items():
                summary.object_count_per_class[class_name] = summary.object_count_per_class.get(class_name, 0) + int(count)
        summary.image_count += int(source_summary["images"])
        summary.label_file_count += int(source_summary["labels"])
        summary.object_count += int(source_summary["objects"])
        summary.empty_label_count += int(source_summary["empty_labels"])
        for split in SPLITS:
            dst_split = summary.split_counts[split]
            src_split = source_summary["splits"][split]
            dst_split["images"] = int(dst_split["images"]) + int(src_split["images"])
            dst_split["objects"] = int(dst_split["objects"]) + int(src_split["objects"])
            dst_split["empty_labels"] = int(dst_split["empty_labels"]) + int(src_split["empty_labels"])
            dst_counts = dst_split["class_counts"]
            for class_name, count in src_split["class_counts"].items():
                dst_counts[class_name] = dst_counts.get(class_name, 0) + int(count)
            for size_name, count in src_split["bbox_size_bins"].items():
                summary.bbox_size_bins[size_name] = summary.bbox_size_bins.get(size_name, 0) + int(count)
    summary.notes.append(f"source_summaries={json.dumps(source_summaries, sort_keys=True)}")
    write_conversion_summary(out_dir, summary)
    print(json.dumps({"stage": args.stage, "out_dir": str(out_dir), "images": summary.image_count, "objects": summary.object_count}, indent=2))
    return 0


def parse_sources(values: Sequence[str]) -> List[SourceSpec]:
    out = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--source must be name=path, got: {value}")
        name, path = value.split("=", 1)
        out.append(SourceSpec(name=name.strip(), path=Path(path.strip()).resolve()))
    return out


def parse_caps(values: Sequence[str]) -> Dict[str, int]:
    caps: Dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--cap must be name=N, got: {value}")
        name, count = value.split("=", 1)
        caps[name.strip()] = int(count)
    return caps


def parse_excluded_source_classes(stage: str, values: Sequence[str], no_default_exclusions: bool) -> Dict[str, set[str]]:
    excluded: Dict[str, set[str]] = {}
    raw_values: List[str] = []
    if not no_default_exclusions:
        raw_values.extend(DEFAULT_EXCLUDED_SOURCE_CLASSES.get(stage, []))
    raw_values.extend(values)
    for value in raw_values:
        if ":" not in value:
            raise ValueError(f"--exclude-source-class must be source:class_name, got: {value}")
        source, class_name = value.split(":", 1)
        source = source.strip()
        class_name = class_name.strip().lower()
        if not source or not class_name:
            raise ValueError(f"--exclude-source-class must be source:class_name, got: {value}")
        excluded.setdefault(source, set()).add(class_name)
    return excluded


def merge_source(
    source: SourceSpec,
    out_dir: Path,
    target_names: List[str],
    cap: Optional[int],
    excluded_source_classes: set[str],
    args: argparse.Namespace,
) -> Dict[str, object]:
    data_yaml = source.path / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Converted source data.yaml is missing for {source.name}: {data_yaml}")
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    source_root = Path(cfg.get("path", source.path))
    if not source_root.is_absolute():
        source_root = (source.path / source_root).resolve()
    source_names = normalize_names(cfg.get("names", {}))
    name_remap = build_name_remap(source_names, target_names)
    candidates = collect_candidates(source_root)
    summary = empty_source_summary(source, None)
    if cap is not None:
        candidates = select_capped_candidates(source, candidates, cap, excluded_source_classes, source_names, summary, args.seed)
    elif excluded_source_classes:
        candidates = filter_excluded_candidates(candidates, excluded_source_classes, source_names, summary)
    for split, image_path, label_path in candidates:
        boxes, skipped = read_and_remap_labels(label_path, source_names, name_remap)
        summary["skipped_labels"] += skipped
        stem = safe_stem(f"{source.name}_{split}_{image_path.stem}")
        dst_image = out_dir / "images" / split / f"{stem}{image_path.suffix.lower()}"
        dst_label = out_dir / "labels" / split / f"{stem}.txt"
        place_image(image_path, dst_image, args.link_mode)
        write_label_file(dst_label, boxes)
        size = image_size(dst_image)
        width, height = size or (1, 1)
        add_to_source_summary(summary, split, boxes, target_names, width, height)
    return summary


def normalize_names(raw: object) -> List[str]:
    if isinstance(raw, dict):
        return [str(raw[k]) for k in sorted(raw, key=lambda x: int(x))]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    raise ValueError("data.yaml names must be list or dict")


def build_name_remap(source_names: List[str], target_names: List[str]) -> Dict[int, int]:
    target_by_name = {name: idx for idx, name in enumerate(target_names)}
    remap = {}
    for source_id, source_name in enumerate(source_names):
        if source_name in target_by_name:
            remap[source_id] = target_by_name[source_name]
    return remap


def collect_candidates(source_root: Path) -> List[Tuple[str, Path, Path]]:
    candidates: List[Tuple[str, Path, Path]] = []
    for split in SPLITS:
        image_dir = source_root / "images" / split
        label_dir = source_root / "labels" / split
        if not image_dir.exists() or not label_dir.exists():
            continue
        for image_path in sorted(p for p in image_dir.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                candidates.append((split, image_path, label_path))
    return candidates


def select_capped_candidates(
    source: SourceSpec,
    candidates: List[Tuple[str, Path, Path]],
    cap: int,
    excluded_source_classes: set[str],
    source_names: List[str],
    summary: Dict[str, object],
    seed: int,
) -> List[Tuple[str, Path, Path]]:
    ordered = sorted(candidates, key=lambda item: stable_fraction(f"{seed}:{source.name}:{item[0]}:{item[1].name}"))
    selected = []
    for split, image_path, label_path in ordered:
        if excluded_source_classes and read_source_class_names(label_path, source_names) & excluded_source_classes:
            summary["skipped_images_due_to_excluded_classes"] = int(summary["skipped_images_due_to_excluded_classes"]) + 1
            continue
        selected.append((split, image_path, label_path))
        if len(selected) >= cap:
            break
    return selected


def filter_excluded_candidates(
    candidates: List[Tuple[str, Path, Path]],
    excluded_source_classes: set[str],
    source_names: List[str],
    summary: Dict[str, object],
) -> List[Tuple[str, Path, Path]]:
    filtered_candidates = []
    for split, image_path, label_path in candidates:
        present_classes = read_source_class_names(label_path, source_names)
        if present_classes & excluded_source_classes:
            summary["skipped_images_due_to_excluded_classes"] = int(summary["skipped_images_due_to_excluded_classes"]) + 1
            continue
        filtered_candidates.append((split, image_path, label_path))
    return filtered_candidates


def read_source_class_names(label_path: Path, source_names: List[str]) -> set[str]:
    names: set[str] = set()
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            source_id = int(float(parts[0]))
        except ValueError:
            continue
        if 0 <= source_id < len(source_names):
            names.add(source_names[source_id].lower())
    return names


def read_and_remap_labels(label_path: Path, source_names: List[str], remap: Dict[int, int]) -> Tuple[List[YoloBox], int]:
    boxes = []
    skipped = 0
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) < 5:
            skipped += 1
            continue
        try:
            source_id = int(float(parts[0]))
            cx, cy, w, h = [float(v) for v in parts[1:5]]
        except ValueError:
            skipped += 1
            continue
        if source_id not in remap:
            skipped += 1
            continue
        boxes.append(YoloBox(remap[source_id], cx, cy, w, h, source_label=source_names[source_id] if source_id < len(source_names) else str(source_id)))
    return boxes, skipped


def empty_source_summary(source: SourceSpec, reason: Optional[str]) -> Dict[str, object]:
    return {
        "path": str(source.path),
        "skipped_reason": reason,
        "images": 0,
        "labels": 0,
        "objects": 0,
        "empty_labels": 0,
        "skipped_labels": 0,
        "skipped_images_due_to_excluded_classes": 0,
        "splits": {split: {"images": 0, "objects": 0, "empty_labels": 0, "class_counts": {}, "bbox_size_bins": {}} for split in SPLITS},
    }


def add_to_source_summary(summary: Dict[str, object], split: str, boxes: List[YoloBox], target_names: List[str], width: int, height: int) -> None:
    from scripts.datasets.common_yolo import bbox_size_bin

    summary["images"] = int(summary["images"]) + 1
    summary["labels"] = int(summary["labels"]) + 1
    summary["objects"] = int(summary["objects"]) + len(boxes)
    split_data = summary["splits"][split]
    split_data["images"] = int(split_data["images"]) + 1
    split_data["objects"] = int(split_data["objects"]) + len(boxes)
    if not boxes:
        summary["empty_labels"] = int(summary["empty_labels"]) + 1
        split_data["empty_labels"] = int(split_data["empty_labels"]) + 1
    for box in boxes:
        name = target_names[box.class_id]
        split_data["class_counts"][name] = split_data["class_counts"].get(name, 0) + 1
        size = bbox_size_bin(box.w * box.h)
        split_data["bbox_size_bins"][size] = split_data["bbox_size_bins"].get(size, 0) + 1


if __name__ == "__main__":
    raise SystemExit(main())
