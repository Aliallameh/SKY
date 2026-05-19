"""Validate a YOLO dataset before SkyScouter training gates can pass."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.datasets.common_yolo import BBOX_SIZE_BINS, IMAGE_EXTS, SPLITS, bbox_size_bin  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO images/labels/data.yaml.")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--out", default=None, help="Validation report JSON. Defaults beside data.yaml.")
    parser.add_argument("--max-duplicate-hash-images", type=int, default=5000)
    parser.add_argument("--allow-empty", action="store_true", help="Allow datasets with zero objects.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml}")
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(cfg.get("path", data_yaml.parent)).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()
    names = normalize_names(cfg.get("names", {}))
    report = {
        "schema_version": "skyscout.yolo_validation.v1",
        "data_yaml": str(data_yaml),
        "dataset_root": str(dataset_root),
        "classes": names,
        "bbox_size_bins": BBOX_SIZE_BINS,
        "splits": {},
        "object_count_per_class": {},
        "bbox_size_counts": {"tiny": 0, "small": 0, "medium": 0},
        "errors": [],
        "warnings": [],
    }
    class_counts: Counter = Counter()
    size_counts: Counter = Counter()
    image_paths_for_hash: List[Path] = []
    for split in SPLITS:
        split_result, split_class_counts, split_size_counts, split_images = validate_split(dataset_root, split, names)
        report["splits"][split] = split_result
        class_counts.update(split_class_counts)
        size_counts.update(split_size_counts)
        image_paths_for_hash.extend(split_images)
        report["errors"].extend(split_result["errors"])
        report["warnings"].extend(split_result["warnings"])
    report["object_count_per_class"] = dict(class_counts)
    report["bbox_size_counts"] = {k: int(size_counts.get(k, 0)) for k in ("tiny", "small", "medium")}
    add_duplicate_warnings(report, image_paths_for_hash[: args.max_duplicate_hash_images])
    add_imbalance_warnings(report, names, class_counts)
    if sum(class_counts.values()) == 0 and not args.allow_empty:
        report["errors"].append("dataset contains zero objects")

    out_path = Path(args.out).resolve() if args.out else data_yaml.parent / "validation_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    status = "PASS" if not report["errors"] else "FAIL"
    print(json.dumps({
        "status": status,
        "report": str(out_path),
        "errors": len(report["errors"]),
        "warnings": len(report["warnings"]),
        "objects": sum(class_counts.values()),
        "classes": dict(class_counts),
        "bbox_size_counts": report["bbox_size_counts"],
    }, indent=2))
    return 0 if not report["errors"] else 2


def normalize_names(raw: object) -> List[str]:
    if isinstance(raw, dict):
        return [str(raw[k]) for k in sorted(raw, key=lambda x: int(x))]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    raise ValueError("data.yaml names must be list or dict")


def validate_split(dataset_root: Path, split: str, names: List[str]) -> Tuple[Dict[str, object], Counter, Counter, List[Path]]:
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    errors: List[str] = []
    warnings: List[str] = []
    class_counts: Counter = Counter()
    size_counts: Counter = Counter()
    if not images_dir.exists():
        warnings.append(f"missing images/{split}")
    if not labels_dir.exists():
        warnings.append(f"missing labels/{split}")
    images = sorted(p for p in images_dir.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS) if images_dir.exists() else []
    labels = sorted(p for p in labels_dir.glob("*.txt")) if labels_dir.exists() else []
    image_stems = {p.stem: p for p in images}
    label_stems = {p.stem: p for p in labels}
    for stem in sorted(set(image_stems) - set(label_stems)):
        errors.append(f"{split}: image has no label file: {image_stems[stem]}")
    for stem in sorted(set(label_stems) - set(image_stems)):
        errors.append(f"{split}: label has no image file: {label_stems[stem]}")
    objects = 0
    empty_labels = 0
    for stem, label_path in label_stems.items():
        if stem not in image_stems:
            continue
        text = label_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not any(line.strip() for line in text):
            empty_labels += 1
        for line_no, line in enumerate(text, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 5:
                errors.append(f"{label_path}:{line_no}: expected 5 YOLO fields")
                continue
            try:
                class_id = int(float(parts[0]))
                cx, cy, w, h = [float(v) for v in parts[1:5]]
            except ValueError:
                errors.append(f"{label_path}:{line_no}: non-numeric field")
                continue
            if class_id < 0 or class_id >= len(names):
                errors.append(f"{label_path}:{line_no}: class id {class_id} out of range 0..{len(names)-1}")
                continue
            if not all(math.isfinite(v) for v in (cx, cy, w, h)):
                errors.append(f"{label_path}:{line_no}: NaN/inf value")
                continue
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                errors.append(f"{label_path}:{line_no}: normalized values outside [0,1]")
                continue
            if w <= 0.0 or h <= 0.0:
                errors.append(f"{label_path}:{line_no}: non-positive width/height")
                continue
            class_counts[names[class_id]] += 1
            size_counts[bbox_size_bin(w * h)] += 1
            objects += 1
    return (
        {
            "images": len(images),
            "labels": len(labels),
            "objects": objects,
            "empty_labels": empty_labels,
            "class_counts": dict(class_counts),
            "bbox_size_counts": dict(size_counts),
            "errors": errors,
            "warnings": warnings,
        },
        class_counts,
        size_counts,
        images,
    )


def add_duplicate_warnings(report: Dict[str, object], image_paths: List[Path]) -> None:
    seen: Dict[str, Path] = {}
    duplicate_examples = []
    for path in image_paths:
        try:
            digest = hashlib.sha1(path.read_bytes()).hexdigest()
        except OSError:
            continue
        if digest in seen:
            duplicate_examples.append({"first": str(seen[digest]), "duplicate": str(path)})
            if len(duplicate_examples) >= 10:
                break
        else:
            seen[digest] = path
    if duplicate_examples:
        report["warnings"].append(f"duplicate image hashes detected: {duplicate_examples}")


def add_imbalance_warnings(report: Dict[str, object], names: List[str], counts: Counter) -> None:
    present = [counts.get(name, 0) for name in names]
    if not present:
        return
    nonzero = [c for c in present if c > 0]
    if len(nonzero) < len(names):
        missing = [name for name in names if counts.get(name, 0) == 0]
        report["warnings"].append(f"classes with zero objects: {missing}")
    if nonzero and max(nonzero) / max(1, min(nonzero)) >= 10:
        report["warnings"].append(f"class imbalance >=10x: {dict(counts)}")


if __name__ == "__main__":
    raise SystemExit(main())
