"""Build a unified YOLO detection dataset for airborne drone-vs-bird training.

Inputs are described by configs/training/airborne_dataset_manifest.yaml. The
builder supports:

- yolo_detection: existing YOLO images/labels with class remapping
- skyscouter_sparse_gt_video: sparse GT CSV plus source video frame extraction

The output is a standard Ultralytics-compatible dataset:

    out/
      data.yaml
      images/{train,val,test}/...
      labels/{train,val,test}/...
      manifest_summary.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class YoloObject:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class Sample:
    source_name: str
    source_id: str
    image_path: Optional[Path]
    image_bgr: Optional[object]
    width: int
    height: int
    objects: List[YoloObject]
    negative: bool = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare unified airborne YOLO training dataset")
    p.add_argument("--manifest", default="configs/training/airborne_dataset_manifest.yaml")
    p.add_argument("--out-dir", default="data/training/airborne_yolo_v1")
    p.add_argument("--link-mode", choices=("copy", "symlink", "hardlink"), default="symlink")
    p.add_argument("--include-disabled", action="store_true", help="Attempt disabled sources too")
    p.add_argument("--source", action="append", default=[], help="Source name to build, even if disabled; repeatable")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    class_names = list(manifest["target_classes"])
    class_to_id = {name: i for i, name in enumerate(class_names)}
    out_dir = Path(args.out_dir)
    _reset_output_dirs(out_dir)

    split_cfg = manifest.get("split", {})
    split_fracs = (
        float(split_cfg.get("train", 0.80)),
        float(split_cfg.get("val", 0.15)),
        float(split_cfg.get("test", 0.05)),
    )
    seed = int(split_cfg.get("seed", 42))

    summary = {
        "schema_version": "skyscout.airborne_dataset_summary.v1",
        "manifest": str(manifest_path),
        "out_dir": str(out_dir),
        "classes": class_names,
        "sources": {},
        "splits": {s: {"images": 0, "objects": 0, "negatives": 0, "class_counts": {}} for s in ("train", "val", "test")},
    }

    total_samples = 0
    selected_sources = set(args.source or [])
    for source in manifest.get("sources", []):
        source_name = str(source["name"])
        if selected_sources and source_name not in selected_sources:
            summary["sources"][source_name] = {"enabled": False, "samples": 0, "reason": "not_selected"}
            continue
        if not source.get("enabled", False) and not args.include_disabled and source_name not in selected_sources:
            summary["sources"][source["name"]] = {"enabled": False, "samples": 0, "reason": "disabled"}
            continue
        # Stream samples directly to disk — do NOT materialise into a list.
        # Video-based sources (anti_uav_rgbt, skyscouter_sparse_gt_video) hold
        # one decoded frame at a time; buffering all frames would exhaust RAM.
        source_summary = {"enabled": True, "samples": 0, "objects": 0, "negatives": 0, "class_counts": {}}
        n_written = 0
        for sample in _load_source_samples(source, class_to_id):
            split = _choose_split(f"{sample.source_name}:{sample.source_id}", split_fracs, seed)
            _write_sample(out_dir, split, sample, args.link_mode)
            total_samples += 1
            n_written += 1
            obj_count = len(sample.objects)
            source_summary["samples"] += 1
            source_summary["objects"] += obj_count
            summary["splits"][split]["images"] += 1
            summary["splits"][split]["objects"] += obj_count
            if sample.negative:
                source_summary["negatives"] += 1
                summary["splits"][split]["negatives"] += 1
            for obj in sample.objects:
                name = class_names[obj.class_id]
                source_summary["class_counts"][name] = source_summary["class_counts"].get(name, 0) + 1
                split_counts = summary["splits"][split]["class_counts"]
                split_counts[name] = split_counts.get(name, 0) + 1
            if n_written % 500 == 0:
                print(f"  [{source_name}] {n_written} samples written …", flush=True)
        summary["sources"][source["name"]] = source_summary

    _write_data_yaml(out_dir, class_names)
    summary["total_images"] = total_samples
    (out_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote YOLO dataset: {out_dir}")
    print(f"Wrote data.yaml: {out_dir / 'data.yaml'}")
    print(json.dumps(summary["splits"], indent=2))
    return 0


def _reset_output_dirs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            d = out_dir / kind / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)


def _load_source_samples(source: Dict[str, object], class_to_id: Dict[str, int]) -> Iterable[Sample]:
    source_type = str(source["type"])
    if source_type == "yolo_detection":
        yield from _load_yolo_detection(source, class_to_id)
    elif source_type == "skyscouter_sparse_gt_video":
        yield from _load_sparse_gt_video(source, class_to_id)
    elif source_type == "anti_uav_rgbt":
        yield from _load_anti_uav_rgbt(source, class_to_id)
    else:
        raise ValueError(f"Unsupported source type for {source.get('name')}: {source_type}")


def _load_yolo_detection(source: Dict[str, object], class_to_id: Dict[str, int]) -> Iterable[Sample]:
    root = Path(str(source["root"])).expanduser()
    images_dir = root / str(source.get("images_dir", "images"))
    labels_dir = root / str(source.get("labels_dir", "labels"))
    label_subdir = str(source.get("label_subdir", "")).strip("/")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found for {source['name']}: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels dir not found for {source['name']}: {labels_dir}")
    class_map = source.get("class_map", {})
    if not isinstance(class_map, dict):
        raise ValueError(f"class_map must be a mapping for {source['name']}")
    source_classes = [str(x) for x in source.get("source_classes", [])]
    numeric_map, named_map = _split_class_map(class_map, class_to_id)

    image_paths = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    for image_path in image_paths:
        rel_image = image_path.relative_to(images_dir)
        if label_subdir and len(rel_image.parts) > 1:
            label_path = labels_dir.joinpath(rel_image.parts[0], label_subdir, *rel_image.parts[1:]).with_suffix(".txt")
        else:
            label_path = labels_dir / rel_image.with_suffix(".txt")
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        height, width = img.shape[:2]
        objects = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                source_cls = parts[0]
                target_id = numeric_map.get(source_cls)
                if target_id is None and source_cls.isdigit() and source_classes:
                    idx = int(source_cls)
                    if 0 <= idx < len(source_classes):
                        target_id = named_map.get(source_classes[idx])
                if target_id is None:
                    target_id = named_map.get(source_cls)
                if target_id is None:
                    continue
                cx, cy, w, h = [float(v) for v in parts[1:5]]
                objects.append(YoloObject(target_id, cx, cy, w, h))
        yield Sample(
            source_name=str(source["name"]),
            source_id=str(image_path.relative_to(root)),
            image_path=image_path,
            image_bgr=None,
            width=width,
            height=height,
            objects=objects,
            negative=len(objects) == 0,
        )


def _load_anti_uav_rgbt(source: Dict[str, object], class_to_id: Dict[str, int]) -> Iterable[Sample]:
    """Load Anti-UAV300 RGBT sequences as YOLO detection samples.

    Dataset layout (one folder per sequence):
        <root>/<split>/<seq_name>/
            visible.mp4        — RGB video
            visible.json       — {"exist": [1,1,0,...], "gt_rect": [[x,y,w,h],...]}
            infrared.mp4       — thermal IR video  (not used here; RGB only)
            infrared.json      — same schema

    Annotation convention:
        exist[t] = 1  → drone visible at frame t, annotate as "drone"
        exist[t] = 0  → drone absent, extract as hard negative if
                         max_negatives_per_seq > 0

    Config keys (all optional):
        modality            : "rgb" (default) | "ir" | "both"
        splits              : list of sub-folder names to include
                              default: ["train", "val", "test"]
        max_positives_per_seq : int | null — cap positive frames per sequence
                              (null = no cap; use to balance with AOD-4)
        max_negatives_per_seq : int — max hard-negative frames per sequence
                              (default 0 = no negatives extracted)
        stride              : int — step between sampled frames (default 1)
        drone_label         : target class name for visible drone frames
                              (default "drone")
    """
    import random as _random

    root = Path(str(source["root"])).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Anti-UAV root not found for {source['name']}: {root}")

    modality: str = str(source.get("modality", "rgb")).lower()
    splits_to_use: List[str] = [str(s) for s in source.get("splits", ["train", "val", "test"])]
    max_pos: Optional[int] = source.get("max_positives_per_seq", None)
    if max_pos is not None:
        max_pos = int(max_pos)
    max_neg: int = int(source.get("max_negatives_per_seq", 0))
    stride: int = max(1, int(source.get("stride", 1)))
    drone_label: str = str(source.get("drone_label", "drone"))
    rng_seed: int = int(source.get("seed", 42))

    if drone_label not in class_to_id:
        raise ValueError(
            f"drone_label '{drone_label}' is not in target_classes for {source['name']}"
        )
    drone_id = class_to_id[drone_label]

    # Which (video_filename, json_filename) pairs to use per modality
    modality_pairs: List[Tuple[str, str]] = []
    if modality in ("rgb", "both"):
        modality_pairs.append(("visible.mp4", "visible.json"))
    if modality in ("ir", "both"):
        modality_pairs.append(("infrared.mp4", "infrared.json"))
    if not modality_pairs:
        raise ValueError(f"Unknown modality '{modality}' for {source['name']}")

    source_name = str(source["name"])

    for split in splits_to_use:
        split_dir = root / split
        if not split_dir.is_dir():
            print(f"[WARN] Anti-UAV split dir not found, skipping: {split_dir}", flush=True)
            continue
        seq_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
        for seq_dir in seq_dirs:
            for video_file, json_file in modality_pairs:
                video_path = seq_dir / video_file
                ann_path = seq_dir / json_file
                if not video_path.exists() or not ann_path.exists():
                    continue

                ann = json.loads(ann_path.read_text(encoding="utf-8"))
                exist_flags: List[int] = ann.get("exist", [])
                gt_rects: List[List[float]] = ann.get("gt_rect", [])
                n_frames = len(exist_flags)
                if n_frames == 0:
                    continue

                # Build candidate positive and negative frame indices
                pos_indices = [
                    i for i in range(0, n_frames, stride)
                    if i < len(exist_flags) and exist_flags[i] == 1
                ]
                neg_indices = [
                    i for i in range(0, n_frames, stride)
                    if i < len(exist_flags) and exist_flags[i] == 0
                ]

                # Apply per-sequence caps with deterministic shuffle
                rng = _random.Random(rng_seed ^ hash(str(seq_dir) + video_file))
                if max_pos is not None and len(pos_indices) > max_pos:
                    rng.shuffle(pos_indices)
                    pos_indices = pos_indices[:max_pos]
                    pos_indices.sort()
                if max_neg > 0 and len(neg_indices) > max_neg:
                    rng.shuffle(neg_indices)
                    neg_indices = neg_indices[:max_neg]
                    neg_indices.sort()
                elif max_neg == 0:
                    neg_indices = []

                all_indices = sorted(set(pos_indices) | set(neg_indices))
                if not all_indices:
                    continue

                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    print(f"[WARN] Cannot open video: {video_path}", flush=True)
                    continue

                try:
                    prev_idx = -1
                    for frame_idx in all_indices:
                        # Seek only when necessary (sequential reads are faster)
                        if frame_idx != prev_idx + 1:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ok, img = cap.read()
                        prev_idx = frame_idx
                        if not ok or img is None:
                            continue

                        height, width = img.shape[:2]
                        is_positive = (frame_idx in set(pos_indices))
                        objects: List[YoloObject] = []

                        if is_positive and frame_idx < len(gt_rects):
                            rect = gt_rects[frame_idx]
                            if len(rect) == 4:
                                x, y, w_box, h_box = float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])
                                obj = _xywh_to_yolo(drone_id, x, y, w_box, h_box, width, height)
                                if obj is not None:
                                    objects.append(obj)

                        # Skip positives where rect was degenerate
                        if is_positive and not objects:
                            continue

                        modal_tag = "rgb" if "visible" in video_file else "ir"
                        source_id = f"{split}_{seq_dir.name}_{modal_tag}_f{frame_idx:06d}"
                        yield Sample(
                            source_name=source_name,
                            source_id=source_id,
                            image_path=None,
                            image_bgr=img,
                            width=width,
                            height=height,
                            objects=objects,
                            negative=(not is_positive),
                        )
                finally:
                    cap.release()


def _xywh_to_yolo(
    class_id: int, x: float, y: float, w_box: float, h_box: float, img_w: int, img_h: int
) -> Optional[YoloObject]:
    """Convert Anti-UAV xywh (top-left origin, pixel coords) to YOLO normalised cx,cy,w,h."""
    x = max(0.0, x)
    y = max(0.0, y)
    w_box = min(w_box, img_w - x)
    h_box = min(h_box, img_h - y)
    if w_box <= 0 or h_box <= 0:
        return None
    return YoloObject(
        class_id=class_id,
        cx=(x + w_box * 0.5) / img_w,
        cy=(y + h_box * 0.5) / img_h,
        w=w_box / img_w,
        h=h_box / img_h,
    )


def _split_class_map(class_map: Dict[object, object], class_to_id: Dict[str, int]) -> Tuple[Dict[str, int], Dict[str, int]]:
    numeric: Dict[str, int] = {}
    named: Dict[str, int] = {}
    for source_cls, target_cls in class_map.items():
        target_name = str(target_cls)
        if target_name not in class_to_id:
            raise ValueError(f"Target class {target_name!r} is not in target_classes")
        target_id = class_to_id[target_name]
        key = str(source_cls)
        if key.isdigit():
            numeric[key] = target_id
        named[key] = target_id
    return numeric, named


def _load_sparse_gt_video(source: Dict[str, object], class_to_id: Dict[str, int]) -> Iterable[Sample]:
    video_path = Path(str(source["video_path"])).expanduser()
    gt_csv = Path(str(source["gt_csv"])).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found for {source['name']}: {video_path}")
    if not gt_csv.exists():
        raise FileNotFoundError(f"GT CSV not found for {source['name']}: {gt_csv}")
    include_labels = {str(x).lower() for x in source.get("include_labels", ["drone", "negative"])}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for {source['name']}: {video_path}")
    try:
        with gt_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                status = str(row.get("review_status", "ok")).lower()
                label = str(row.get("label", "drone")).lower()
                if status in {"skip", "draft", "ignore"} or label not in include_labels:
                    continue
                frame_id = int(row["frame_id"])
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, image = cap.read()
                if not ok:
                    continue
                height, width = image.shape[:2]
                objects: List[YoloObject] = []
                negative = label in {"negative", "hard_negative", "no_drone", "background"}
                if not negative:
                    if label not in class_to_id:
                        continue
                    try:
                        x1, y1, x2, y2 = [float(row[k]) for k in ("x1", "y1", "x2", "y2")]
                    except (TypeError, ValueError):
                        continue
                    obj = _xyxy_to_yolo(class_to_id[label], x1, y1, x2, y2, width, height)
                    if obj is None:
                        continue
                    objects.append(obj)
                yield Sample(
                    source_name=str(source["name"]),
                    source_id=f"frame_{frame_id:05d}",
                    image_path=None,
                    image_bgr=image,
                    width=width,
                    height=height,
                    objects=objects,
                    negative=negative,
                )
    finally:
        cap.release()


def _xyxy_to_yolo(class_id: int, x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Optional[YoloObject]:
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return YoloObject(
        class_id=class_id,
        cx=((x1 + x2) * 0.5) / width,
        cy=((y1 + y2) * 0.5) / height,
        w=(x2 - x1) / width,
        h=(y2 - y1) / height,
    )


def _choose_split(key: str, split_fracs: Sequence[float], seed: int) -> str:
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    train, val, _test = split_fracs
    if value < train:
        return "train"
    if value < train + val:
        return "val"
    return "test"


def _write_sample(out_dir: Path, split: str, sample: Sample, link_mode: str) -> None:
    safe_name = _safe_stem(f"{sample.source_name}_{sample.source_id}")
    image_out = out_dir / "images" / split / f"{safe_name}.jpg"
    label_out = out_dir / "labels" / split / f"{safe_name}.txt"
    if sample.image_path is not None:
        _place_existing_image(sample.image_path, image_out, link_mode)
    elif sample.image_bgr is not None:
        cv2.imwrite(str(image_out), sample.image_bgr)
    else:
        raise ValueError(f"Sample has no image data: {sample.source_name}:{sample.source_id}")
    label_out.write_text("\n".join(_format_obj(o) for o in sample.objects) + ("\n" if sample.objects else ""), encoding="utf-8")


def _place_existing_image(src: Path, dst: Path, link_mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link_mode == "copy":
        shutil.copy2(src, dst)
    elif link_mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def _format_obj(obj: YoloObject) -> str:
    return f"{obj.class_id} {obj.cx:.8f} {obj.cy:.8f} {obj.w:.8f} {obj.h:.8f}"


def _safe_stem(value: str) -> str:
    out = []
    for ch in value:
        out.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    stem = "".join(out).strip("_")
    if len(stem) > 180:
        digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:12]
        stem = f"{stem[:160]}_{digest}"
    return stem or "sample"


def _write_data_yaml(out_dir: Path, class_names: List[str]) -> None:
    data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
