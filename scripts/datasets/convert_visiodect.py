"""Convert VisioDECT VOC annotations to YOLO drone-only format with Mavic holdout."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.datasets.common_yolo import (  # noqa: E402
    IMAGE_EXTS,
    ConversionSummary,
    choose_split,
    normalize_label,
    place_image,
    read_voc_boxes,
    reset_yolo_dirs,
    safe_stem,
    stable_fraction,
    write_conversion_summary,
    write_data_yaml,
    write_label_file,
    xyxy_to_yolo,
)


CLASS_NAMES = ["drone"]
MAVIC_TOKENS = ("mavic", "mavic_air", "mavic_enterprise", "dji_mavic", "mavic_2", "enterprise")


@dataclass
class VisioSample:
    xml_path: Path
    image_path: Path
    group_id: str
    is_mavic_like: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VisioDECT VOC labels to YOLO with Mavic-like validation holdout.")
    parser.add_argument("--root", default="DATASETS/4_VisioDECT Dataset Upload")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mavic-heldout-dir", default="data/training/validation_slices/visiodect_mavic_like")
    parser.add_argument("--limit", type=int, default=None, help="Limit non-held-out output images for prototype conversion.")
    parser.add_argument("--mavic-heldout-limit", type=int, default=None, help="Optional cap for copied held-out Mavic validation images.")
    parser.add_argument("--mavic-holdout-fraction", type=float, default=0.50)
    parser.add_argument("--min-mavic-groups-for-split", type=int, default=2)
    parser.add_argument("--min-mavic-samples-for-split", type=int, default=200)
    parser.add_argument("--link-mode", choices=["copy", "hardlink", "symlink"], default="copy")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    heldout_dir = Path(args.mavic_heldout_dir).resolve()
    if not root.exists():
        raise SystemExit(f"VisioDECT root not found: {root}")

    reset_yolo_dirs(out_dir)
    reset_yolo_dirs(heldout_dir)
    write_data_yaml(out_dir, CLASS_NAMES)
    write_data_yaml(heldout_dir, CLASS_NAMES)

    samples, discovery = discover_samples(root)
    mavic_groups = group_mavic_samples(samples)
    heldout_groups, train_mavic_groups, policy_note = decide_mavic_policy(mavic_groups, args)

    summary = ConversionSummary(
        dataset="visiodect_drone_models",
        source_root=str(root),
        output_dir=str(out_dir),
        classes=CLASS_NAMES,
        class_remap={"all VisioDECT drone model labels/folders": "0 drone"},
        limit=args.limit,
        split_policy="scenario/group hash only; no frame-level split",
        held_out_groups=sorted(heldout_groups),
        notes=[policy_note, "Mavic-like held-out samples are written separately and are excluded from this training output."],
    )
    heldout_summary = ConversionSummary(
        dataset="visiodect_mavic_like_heldout",
        source_root=str(root),
        output_dir=str(heldout_dir),
        classes=CLASS_NAMES,
        class_remap={"Mavic-like VisioDECT labels/folders": "0 drone"},
        limit=args.mavic_heldout_limit,
        split_policy="held-out validation slice; all copied samples placed in val",
        held_out_groups=sorted(heldout_groups),
        notes=[policy_note, "This slice must not be used for Stage 1/2/3 training."],
    )

    written_train = 0
    written_heldout = 0
    for sample in samples:
        if sample.group_id in heldout_groups:
            if args.mavic_heldout_limit is None or written_heldout < args.mavic_heldout_limit:
                if write_sample(sample, root, heldout_dir, "val", heldout_summary, args.link_mode):
                    written_heldout += 1
            continue
        if sample.is_mavic_like and sample.group_id not in train_mavic_groups:
            continue
        if args.limit is not None and written_train >= args.limit:
            continue
        split = choose_split(f"visiodect:{sample.group_id}", seed=args.seed)
        if write_sample(sample, root, out_dir, split, summary, args.link_mode):
            written_train += 1

    summary.notes.append(f"Discovery: {json.dumps(discovery, sort_keys=True)}")
    heldout_summary.notes.append(f"Discovery: {json.dumps(discovery, sort_keys=True)}")
    write_conversion_summary(out_dir, summary)
    write_conversion_summary(heldout_dir, heldout_summary)
    print(json.dumps({
        "out_dir": str(out_dir),
        "heldout_dir": str(heldout_dir),
        "images": summary.image_count,
        "heldout_images": heldout_summary.image_count,
        "heldout_groups": sorted(heldout_groups),
    }, indent=2))
    return 0


def discover_samples(root: Path) -> Tuple[List[VisioSample], Dict[str, object]]:
    image_index = build_image_index(root)
    samples: List[VisioSample] = []
    folder_matches: Counter = Counter()
    label_matches: Counter = Counter()
    missing_images = 0
    for xml_path in sorted(root.rglob("*.xml")):
        rel_parts = xml_path.relative_to(root).parts
        top = rel_parts[0] if rel_parts else xml_path.parent.name
        filename, labels = xml_filename_and_labels(xml_path)
        image_path = image_index.get((top, filename.lower()))
        if image_path is None:
            image_path = image_index.get(("", filename.lower()))
        if image_path is None:
            missing_images += 1
            continue
        group_id = infer_group_id(root, xml_path)
        label_is_mavic = any(is_mavic_like(label) for label in labels)
        path_is_mavic = is_mavic_like(str(xml_path.relative_to(root))) or is_mavic_like(top)
        is_mavic = label_is_mavic or path_is_mavic
        if path_is_mavic:
            folder_matches[group_id] += 1
        for label in labels:
            if is_mavic_like(label):
                label_matches[label] += 1
        samples.append(VisioSample(xml_path=xml_path, image_path=image_path, group_id=group_id, is_mavic_like=is_mavic))
    discovery = {
        "samples": len(samples),
        "missing_images": missing_images,
        "mavic_folder_group_counts": dict(folder_matches),
        "mavic_label_counts": dict(label_matches),
    }
    return samples, discovery


def build_image_index(root: Path) -> Dict[Tuple[str, str], Path]:
    index: Dict[Tuple[str, str], Path] = {}
    global_index: Dict[Tuple[str, str], Path] = {}
    for image_path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
        rel = image_path.relative_to(root)
        top = rel.parts[0] if rel.parts else ""
        index[(top, image_path.name.lower())] = image_path
        global_index[("", image_path.name.lower())] = image_path
    index.update(global_index)
    return index


def xml_filename_and_labels(xml_path: Path) -> Tuple[str, List[str]]:
    import xml.etree.ElementTree as ET

    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename") or f"{xml_path.stem}.jpg"
    labels = [node.text.strip() for node in root.findall(".//object/name") if node.text]
    return filename, labels


def infer_group_id(root: Path, xml_path: Path) -> str:
    rel = xml_path.relative_to(root)
    parts = rel.parts
    if len(parts) >= 4 and parts[1].lower() == "labels":
        return f"{parts[0]}_{parts[2].lower()}"
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1].lower()}"
    return parts[0] if parts else xml_path.parent.name


def group_mavic_samples(samples: Sequence[VisioSample]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for sample in samples:
        if sample.is_mavic_like:
            counts[sample.group_id] += 1
    return dict(counts)


def decide_mavic_policy(mavic_groups: Dict[str, int], args: argparse.Namespace) -> Tuple[set[str], set[str], str]:
    total = sum(mavic_groups.values())
    if len(mavic_groups) < args.min_mavic_groups_for_split or total < args.min_mavic_samples_for_split:
        return set(mavic_groups), set(), (
            f"Mavic-like sample count/group count below split threshold "
            f"({total} samples, {len(mavic_groups)} groups); all Mavic-like VisioDECT is held out."
        )
    heldout: set[str] = set()
    train: set[str] = set()
    for group in sorted(mavic_groups):
        if stable_fraction(f"mavic_holdout:{group}", args.seed) < args.mavic_holdout_fraction:
            heldout.add(group)
        else:
            train.add(group)
    if not heldout:
        heldout.add(sorted(mavic_groups)[0])
        train.discard(sorted(mavic_groups)[0])
    if not train and len(mavic_groups) > 1:
        moved = sorted(heldout)[-1]
        heldout.remove(moved)
        train.add(moved)
    return heldout, train, (
        f"Mavic-like samples split by scenario/group: {len(train)} train groups, {len(heldout)} held-out groups."
    )


def write_sample(sample: VisioSample, root: Path, out_dir: Path, split: str, summary: ConversionSummary, link_mode: str) -> bool:
    try:
        width, height, objects = read_voc_boxes(sample.xml_path)
    except Exception as exc:
        summary.add_skip(sample.xml_path, f"invalid VOC XML: {exc}")
        return False
    if width is None or height is None:
        summary.add_skip(sample.xml_path, "missing VOC size")
        return False
    boxes = []
    for label, bbox in objects:
        box = xyxy_to_yolo(0, *bbox, width=width, height=height, source_label=label)
        if box is None:
            summary.add_invalid_box(str(sample.xml_path), bbox, "degenerate/clipped VOC box")
            continue
        boxes.append(box)
    stem = safe_stem(f"visiodect_{sample.group_id}_{sample.image_path.stem}")
    image_out = out_dir / "images" / split / f"{stem}{sample.image_path.suffix.lower()}"
    label_out = out_dir / "labels" / split / f"{stem}.txt"
    place_image(sample.image_path, image_out, link_mode)
    write_label_file(label_out, boxes)
    summary.add_image(split, boxes, width, height)
    return True


def is_mavic_like(value: object) -> bool:
    normalized = normalize_label(value)
    return any(token in normalized for token in MAVIC_TOKENS)


if __name__ == "__main__":
    raise SystemExit(main())
