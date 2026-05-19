"""Evaluate matched-GT semantic confusion for YOLO datasets.

This is the metric that matters for the current SkyScouter blocker: a GT drone
may be geometrically detected but semantically predicted as airplane, which
prevents semantic-safe LOCKED.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.datasets.common_yolo import IMAGE_EXTS, SPLITS, bbox_size_bin  # noqa: E402
from skyscouter.output.evaluation import bbox_iou, center_error  # noqa: E402


BBox = Tuple[float, float, float, float]
SEMANTIC_CLASSES = ("drone", "airplane", "bird", "helicopter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate matched-GT semantic confusion for YOLO predictions.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=list(SPLITS) + ["all"], default="val")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument("--iou-hit", type=float, default=0.10)
    parser.add_argument("--center-hit-px", type=float, default=35.0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is required. Use .venv_train for this script.") from exc

    data_yaml = Path(args.data).resolve()
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(cfg.get("path", data_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()
    gt_names = normalize_names(cfg.get("names", {}))
    gt_drone_ids = {idx for idx, name in enumerate(gt_names) if name.lower() in {"drone", "uas"}}
    if not gt_drone_ids:
        raise SystemExit(f"No drone GT class found in data.yaml names: {gt_names}")
    splits = SPLITS if args.split == "all" else (args.split,)
    samples = collect_samples(dataset_root, splits)
    if args.limit is not None:
        samples = samples[: args.limit]
    model = YOLO(args.model)

    per_gt: List[Dict[str, object]] = []
    fp_by_gt_class = Counter()
    drone_fp_unmatched = 0
    images_processed = 0
    for split, image_path, label_path in samples:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        gt_boxes = read_gt_boxes(label_path, width, height)
        predictions = predict_image(model, image_path, args)
        images_processed += 1
        matched_pred_indices = set()
        for gt_idx, (gt_class_id, gt_bbox) in enumerate(gt_boxes):
            match_idx, match, iou, cerr = best_match(predictions, gt_bbox)
            hit = match is not None and ((iou is not None and iou >= args.iou_hit) or (cerr is not None and cerr <= args.center_hit_px))
            if hit and match_idx is not None:
                matched_pred_indices.add(match_idx)
            pred_label = "missed"
            pred_conf = None
            if hit and match is not None:
                pred_label = match["label"]
                pred_conf = match["confidence"]
            gt_name = gt_names[gt_class_id] if gt_class_id < len(gt_names) else str(gt_class_id)
            if gt_name in {"bird", "airplane", "helicopter"} and pred_label == "drone":
                fp_by_gt_class[f"{gt_name}_to_drone"] += 1
            if gt_class_id in gt_drone_ids:
                area_ratio = bbox_area_ratio(gt_bbox, width, height)
                per_gt.append(
                    {
                        "split": split,
                        "image": str(image_path),
                        "gt_class": gt_name,
                        "gt_bbox": gt_bbox,
                        "bbox_size_bin": bbox_size_bin(area_ratio),
                        "matched": bool(hit),
                        "pred_label": pred_label,
                        "pred_confidence": pred_conf,
                        "iou": iou,
                        "center_error_px": cerr,
                    }
                )
        for idx, pred in enumerate(predictions):
            if idx not in matched_pred_indices and pred["label"] == "drone":
                drone_fp_unmatched += 1

    report = build_report(args, data_yaml, images_processed, per_gt, fp_by_gt_class, drone_fp_unmatched)
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out_path),
        "images": images_processed,
        "gt_drone_boxes": report["gt_drone_boxes"],
        "semantic_confusion": report["matched_gt_semantic_confusion"],
        "drone_recall_by_bbox_size": report["drone_recall_by_bbox_size"],
    }, indent=2))
    return 0


def normalize_names(raw: object) -> List[str]:
    if isinstance(raw, dict):
        return [str(raw[k]) for k in sorted(raw, key=lambda x: int(x))]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    raise ValueError("data.yaml names must be list or dict")


def collect_samples(dataset_root: Path, splits: Sequence[str]) -> List[Tuple[str, Path, Path]]:
    samples: List[Tuple[str, Path, Path]] = []
    for split in splits:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        if not image_dir.exists() or not label_dir.exists():
            continue
        for image_path in sorted(p for p in image_dir.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                samples.append((split, image_path, label_path))
    return samples


def read_gt_boxes(label_path: Path, width: int, height: int) -> List[Tuple[int, BBox]]:
    boxes: List[Tuple[int, BBox]] = []
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, w, h = [float(v) for v in parts[1:5]]
        except ValueError:
            continue
        x1 = (cx - w * 0.5) * width
        y1 = (cy - h * 0.5) * height
        x2 = (cx + w * 0.5) * width
        y2 = (cy + h * 0.5) * height
        boxes.append((class_id, (x1, y1, x2, y2)))
    return boxes


def predict_image(model: object, image_path: Path, args: argparse.Namespace) -> List[Dict[str, object]]:
    kwargs = {"conf": args.conf, "imgsz": args.imgsz, "verbose": False}
    if args.device is not None:
        kwargs["device"] = args.device
    result = model.predict(str(image_path), **kwargs)[0]
    names = result.names
    preds: List[Dict[str, object]] = []
    if result.boxes is None:
        return preds
    for box in result.boxes:
        xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
        class_id = int(box.cls[0].detach().cpu().item())
        conf = float(box.conf[0].detach().cpu().item())
        label = str(names.get(class_id, class_id)).lower()
        preds.append({"bbox": tuple(float(v) for v in xyxy), "class_id": class_id, "label": label, "confidence": conf})
    return preds


def best_match(predictions: List[Dict[str, object]], gt_bbox: BBox) -> Tuple[Optional[int], Optional[Dict[str, object]], Optional[float], Optional[float]]:
    if not predictions:
        return None, None, None, None
    scored = []
    for idx, pred in enumerate(predictions):
        pred_bbox = pred["bbox"]
        iou = bbox_iou(pred_bbox, gt_bbox)
        cerr = center_error(pred_bbox, gt_bbox)
        scored.append((iou, -cerr, float(pred["confidence"]), idx, pred))
    scored.sort(reverse=True, key=lambda item: item[:3])
    iou, neg_cerr, _conf, idx, pred = scored[0]
    return idx, pred, float(iou), float(-neg_cerr)


def bbox_area_ratio(bbox: BBox, width: int, height: int) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, float(width * height))


def build_report(
    args: argparse.Namespace,
    data_yaml: Path,
    images_processed: int,
    per_gt: List[Dict[str, object]],
    fp_by_gt_class: Counter,
    drone_fp_unmatched: int,
) -> Dict[str, object]:
    confusion = {
        "matched_as_drone": 0,
        "matched_as_airplane": 0,
        "matched_as_bird": 0,
        "matched_as_helicopter": 0,
        "matched_wrong_class": 0,
        "missed": 0,
    }
    label_distribution = Counter()
    center_errors = []
    for row in per_gt:
        label = str(row["pred_label"])
        label_distribution[label] += 1
        if label == "missed" or not row["matched"]:
            confusion["missed"] += 1
        elif label == "drone":
            confusion["matched_as_drone"] += 1
        elif label == "airplane":
            confusion["matched_as_airplane"] += 1
        elif label == "bird":
            confusion["matched_as_bird"] += 1
        elif label == "helicopter":
            confusion["matched_as_helicopter"] += 1
        else:
            confusion["matched_wrong_class"] += 1
        cerr = row.get("center_error_px")
        if cerr is not None and math.isfinite(float(cerr)):
            center_errors.append(float(cerr))
    by_size: Dict[str, Dict[str, float]] = {}
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in per_gt:
        grouped[str(row["bbox_size_bin"])].append(row)
    for size_name in ("tiny", "small", "medium"):
        rows = grouped.get(size_name, [])
        if not rows:
            continue
        detected = [r for r in rows if r["matched"]]
        semantic = [r for r in rows if r["matched"] and r["pred_label"] == "drone"]
        by_size[size_name] = {
            "gt_drone_boxes": len(rows),
            "geometric_recall": len(detected) / max(1, len(rows)),
            "semantic_drone_recall": len(semantic) / max(1, len(rows)),
        }
    return {
        "schema_version": "skyscout.semantic_confusion_report.v1",
        "model": str(Path(args.model).resolve()),
        "data_yaml": str(data_yaml),
        "split": args.split,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "iou_hit": args.iou_hit,
        "center_hit_px": args.center_hit_px,
        "images_processed": images_processed,
        "gt_drone_boxes": len(per_gt),
        "matched_gt_semantic_confusion": confusion,
        "drone_recall_by_bbox_size": by_size,
        "drone_label_distribution_on_gt": dict(label_distribution),
        "drone_to_airplane_confusion_rate": confusion["matched_as_airplane"] / max(1, len(per_gt)),
        "bird_to_drone_false_matches": int(fp_by_gt_class.get("bird_to_drone", 0)),
        "airplane_to_drone_false_matches": int(fp_by_gt_class.get("airplane_to_drone", 0)),
        "unmatched_drone_predictions": int(drone_fp_unmatched),
        "center_error_px": {
            "median": median(center_errors) if center_errors else None,
            "mean": sum(center_errors) / len(center_errors) if center_errors else None,
        },
        "per_gt": per_gt,
    }


if __name__ == "__main__":
    raise SystemExit(main())
