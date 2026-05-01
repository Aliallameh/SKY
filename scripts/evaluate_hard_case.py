"""Evaluate a corrected hard-case annotation packet.

This script intentionally separates detector behavior from tracker behavior:

* detector metrics answer "did YOLO see the object in this frame?"
* tracker metrics answer "did the published track follow the right object?"

It also exports the corrected CSV into a tiny YOLO-format dataset so the same
packet can later be folded into training or used as a validation slice.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from skyscouter.output.evaluation import bbox_iou, center_error  # noqa: E402
from skyscouter.perception.base_detector import Detection  # noqa: E402
from skyscouter.perception.factory import build_detector  # noqa: E402
from skyscouter.tracking.base_tracker import Track  # noqa: E402
from skyscouter.tracking.factory import build_tracker  # noqa: E402
from skyscouter.utils.config_loader import load_config  # noqa: E402


BBox = Tuple[float, float, float, float]


@dataclass
class GtRow:
    frame_id: int
    image_rel: str
    label: str
    bbox: Optional[BBox]
    visibility: str = ""
    occluded: str = ""

    @property
    def is_positive(self) -> bool:
        return self.label in {"drone", "uas", "airborne_candidate"}

    @property
    def is_negative(self) -> bool:
        return self.label in {"negative", "hard_negative", "no_drone", "background"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and evaluate a drone hard-case packet.")
    parser.add_argument("--csv", required=True, help="Corrected sparse GT CSV")
    parser.add_argument("--config", required=True, help="Pipeline config with detector/tracker settings")
    parser.add_argument("--output", required=True, help="Output directory for this hard-case report")
    parser.add_argument("--case-name", default=None, help="Human-readable hard-case name")
    parser.add_argument("--fps", type=float, default=30.0, help="Assumed FPS for tracker timestamps")
    parser.add_argument("--detector-iou-hit", type=float, default=0.10)
    parser.add_argument("--tracker-iou-hit", type=float, default=0.10)
    parser.add_argument("--center-hit-px", type=float, default=35.0)
    parser.add_argument("--no-copy-images", action="store_true", help="Write labels only; do not copy images")
    return parser.parse_args()


def read_gt_csv(path: Path) -> List[GtRow]:
    rows: List[GtRow] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            status = str(raw.get("review_status", "ok")).strip().lower()
            if status in {"skip", "draft", "ignore"}:
                continue
            label = str(raw.get("label", "drone")).strip().lower()
            bbox = None
            if label not in {"negative", "hard_negative", "no_drone", "background"}:
                try:
                    x1, y1, x2, y2 = [float(raw[k]) for k in ("x1", "y1", "x2", "y2")]
                except (TypeError, ValueError):
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue
                bbox = (x1, y1, x2, y2)
            rows.append(
                GtRow(
                    frame_id=int(raw["frame_id"]),
                    image_rel=str(raw["image"]),
                    label=label,
                    bbox=bbox,
                    visibility=str(raw.get("visibility", "")),
                    occluded=str(raw.get("occluded", "")),
                )
            )
    return rows


def detection_bbox(det: Detection) -> BBox:
    return (float(det.x), float(det.y), float(det.x + det.w), float(det.y + det.h))


def track_bbox(track: Track) -> BBox:
    return detection_bbox(track.detection)


def best_detection_match(detections: List[Detection], gt_bbox: BBox) -> Tuple[Optional[Detection], Optional[float], Optional[float]]:
    if not detections:
        return None, None, None
    scored = []
    for det in detections:
        pred = detection_bbox(det)
        iou = bbox_iou(pred, gt_bbox)
        cerr = center_error(pred, gt_bbox)
        scored.append((iou, -cerr, float(det.confidence), det))
    scored.sort(reverse=True, key=lambda item: item[:3])
    best = scored[0][3]
    return best, float(scored[0][0]), float(-scored[0][1])


def select_primary_track(tracks: List[Track], current_track_id: Optional[int]) -> Tuple[Optional[Track], Optional[int]]:
    if not tracks:
        return None, None
    if current_track_id is not None:
        current = next((t for t in tracks if t.track_id == current_track_id), None)
        if current is not None and current.time_since_update == 0:
            return current, current.track_id

    min_hits = max(getattr(t, "min_confirmed_hits", 3) for t in tracks)
    confirmed = [t for t in tracks if t.hits >= min_hits and t.time_since_update == 0]
    pool = confirmed if confirmed else tracks
    primary = max(
        pool,
        key=lambda t: (
            t.time_since_update == 0,
            float(t.detection.confidence),
            float(t.detection.area),
        ),
    )
    return primary, primary.track_id


def safe_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def is_hit(iou: Optional[float], cerr: Optional[float], iou_threshold: float, center_threshold: float) -> bool:
    return bool(
        (iou is not None and iou >= iou_threshold)
        or (cerr is not None and cerr <= center_threshold)
    )


def export_yolo_dataset(rows: List[GtRow], packet_dir: Path, output_dir: Path, copy_images: bool) -> Dict[str, Any]:
    yolo_dir = output_dir / "yolo"
    images_dir = yolo_dir / "images"
    labels_dir = yolo_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    exported = []
    for row in rows:
        src_image = packet_dir / row.image_rel
        if not src_image.exists():
            continue
        img = cv2.imread(str(src_image))
        if img is None:
            continue
        h, w = img.shape[:2]
        dst_image = images_dir / src_image.name
        if copy_images and not dst_image.exists():
            shutil.copy2(src_image, dst_image)

        label_path = labels_dir / f"{src_image.stem}.txt"
        lines: List[str] = []
        if row.is_positive and row.bbox is not None:
            x1, y1, x2, y2 = clamp_bbox(row.bbox, w, h)
            cx = ((x1 + x2) * 0.5) / w
            cy = ((y1 + y2) * 0.5) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        exported.append({"frame_id": row.frame_id, "image": str(dst_image if copy_images else src_image), "label": str(label_path)})

    data_yaml = yolo_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {yolo_dir.as_posix()}",
                "train: images",
                "val: images",
                "names:",
                "  0: drone",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "yolo_dir": str(yolo_dir),
        "data_yaml": str(data_yaml),
        "images_exported": len(exported),
        "labels_dir": str(labels_dir),
    }


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    x1 = min(max(float(x1), 0.0), float(width))
    x2 = min(max(float(x2), 0.0), float(width))
    y1 = min(max(float(y1), 0.0), float(height))
    y2 = min(max(float(y2), 0.0), float(height))
    return x1, y1, x2, y2


def summarize(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"mean": None, "median": None, "max": None}
    return {
        "mean": float(sum(vals) / len(vals)),
        "median": float(median(vals)),
        "max": float(max(vals)),
    }


def write_per_frame_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "frame_id",
        "gt_label",
        "gt_x1",
        "gt_y1",
        "gt_x2",
        "gt_y2",
        "detections",
        "det_best_class",
        "det_best_conf",
        "det_best_iou",
        "det_center_error_px",
        "det_hit",
        "track_id",
        "track_status",
        "track_source",
        "track_conf",
        "track_time_since_update",
        "track_iou",
        "track_center_error_px",
        "track_hit",
        "tracker_stale",
        "negative_false_positive",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_summary_md(path: Path, case_name: str, report: Dict[str, Any]) -> None:
    lines = [
        f"# Hard-case report: {case_name}",
        "",
        "## Summary",
        "",
        f"- Frames: {report['frames']}",
        f"- Positive drone frames: {report['positive_frames']}",
        f"- Negative frames: {report['negative_frames']}",
        f"- Detector hits: {report['detector']['hit_frames']} / {report['positive_frames']} ({report['detector']['hit_rate']:.1%})",
        f"- Tracker hits: {report['tracker']['hit_frames']} / {report['positive_frames']} ({report['tracker']['hit_rate']:.1%})",
        f"- Tracker stale frames: {len(report['tracker']['stale_frames'])}",
        f"- Negative false positives: {len(report['tracker']['negative_false_positive_frames'])}",
        "",
        "## Key Frames",
        "",
        f"- Detector miss frames: {format_frame_list(report['detector']['miss_frames'])}",
        f"- Tracker bad frames: {format_frame_list(report['tracker']['bad_frames'])}",
        f"- Stale tracker frames: {format_frame_list(report['tracker']['stale_frames'])}",
        f"- Negative false positive frames: {format_frame_list(report['tracker']['negative_false_positive_frames'])}",
        "",
        "## Interpretation",
        "",
    ]
    if report["detector"]["hit_rate"] < 0.90:
        lines.append("- Detector recall is the first bottleneck on this clip; add more turn/reversal samples before expecting the tracker to be stable.")
    else:
        lines.append("- Detector recall is acceptable on this hard case; tracker gating and stale-state behavior become the main area to tune.")
    if report["tracker"]["stale_frames"]:
        lines.append("- Tracker still emits prediction-only boxes on one or more reviewed frames; stale emission limits should stay strict.")
    if report["tracker"]["negative_false_positive_frames"]:
        lines.append("- Negative frames still have published tracks; edge/exit-frame suppression needs more work before training is blamed.")
    if not report["tracker"]["stale_frames"] and not report["tracker"]["negative_false_positive_frames"]:
        lines.append("- The stale-following failure is controlled on this packet with the current tracker settings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_frame_list(frames: List[int], limit: int = 24) -> str:
    if not frames:
        return "none"
    shown = frames[:limit]
    suffix = "" if len(frames) <= limit else f" ... (+{len(frames) - limit} more)"
    return ", ".join(str(f) for f in shown) + suffix


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv).resolve()
    packet_dir = csv_path.parent
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    case_name = args.case_name or csv_path.parent.name

    rows = read_gt_csv(csv_path)
    if not rows:
        raise RuntimeError(f"No usable rows found in {csv_path}")

    yolo_info = export_yolo_dataset(rows, packet_dir, output_dir, copy_images=not args.no_copy_images)

    cfg = load_config(args.config)
    detector = build_detector(cfg["detector"])
    tracker = build_tracker(cfg["tracker"])
    detector.warmup()

    per_frame: List[Dict[str, Any]] = []
    current_track_id: Optional[int] = None
    for row in rows:
        image_path = packet_dir / row.image_rel
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        detections = detector.detect(image)
        det_best = None
        det_iou = None
        det_cerr = None
        if row.is_positive and row.bbox is not None:
            det_best, det_iou, det_cerr = best_detection_match(detections, row.bbox)
        elif detections:
            det_best = max(detections, key=lambda d: float(d.confidence))

        tracks = tracker.update(
            detections,
            frame_index=row.frame_id,
            capture_time_s=float(row.frame_id) / max(args.fps, 1e-6),
            image_bgr=image,
        )
        primary, current_track_id = select_primary_track(tracks, current_track_id)

        track_iou = None
        track_cerr = None
        if primary is not None and row.is_positive and row.bbox is not None:
            track_iou = bbox_iou(track_bbox(primary), row.bbox)
            track_cerr = center_error(track_bbox(primary), row.bbox)

        det_hit = row.is_positive and is_hit(det_iou, det_cerr, args.detector_iou_hit, args.center_hit_px)
        track_hit = row.is_positive and is_hit(track_iou, track_cerr, args.tracker_iou_hit, args.center_hit_px)
        stale = bool(primary is not None and (primary.time_since_update > 0 or not primary.matched_detection))
        negative_fp = bool(row.is_negative and primary is not None)

        gt = row.bbox or ("", "", "", "")
        per_frame.append(
            {
                "frame_id": row.frame_id,
                "gt_label": row.label,
                "gt_x1": gt[0],
                "gt_y1": gt[1],
                "gt_x2": gt[2],
                "gt_y2": gt[3],
                "detections": len(detections),
                "det_best_class": "" if det_best is None else det_best.class_label,
                "det_best_conf": "" if det_best is None else safe_float(det_best.confidence),
                "det_best_iou": safe_float(det_iou),
                "det_center_error_px": safe_float(det_cerr),
                "det_hit": int(det_hit),
                "track_id": "" if primary is None else primary.track_id,
                "track_status": "" if primary is None else primary.status,
                "track_source": "" if primary is None else primary.source,
                "track_conf": "" if primary is None else safe_float(primary.detection.confidence),
                "track_time_since_update": "" if primary is None else primary.time_since_update,
                "track_iou": safe_float(track_iou),
                "track_center_error_px": safe_float(track_cerr),
                "track_hit": int(track_hit),
                "tracker_stale": int(stale),
                "negative_false_positive": int(negative_fp),
            }
        )

    positives = [r for r in per_frame if r["gt_label"] in {"drone", "uas", "airborne_candidate"}]
    negatives = [r for r in per_frame if r["gt_label"] in {"negative", "hard_negative", "no_drone", "background"}]
    detector_miss_frames = [int(r["frame_id"]) for r in positives if int(r["det_hit"]) == 0]
    tracker_bad_frames = [int(r["frame_id"]) for r in positives if int(r["track_hit"]) == 0]
    stale_frames = [int(r["frame_id"]) for r in per_frame if int(r["tracker_stale"]) == 1]
    negative_fp_frames = [int(r["frame_id"]) for r in negatives if int(r["negative_false_positive"]) == 1]

    report: Dict[str, Any] = {
        "schema_version": "skyscout.hard_case_report.v1",
        "case_name": case_name,
        "csv": str(csv_path),
        "config": str(Path(args.config).resolve()),
        "frames": len(per_frame),
        "positive_frames": len(positives),
        "negative_frames": len(negatives),
        "thresholds": {
            "detector_iou_hit": args.detector_iou_hit,
            "tracker_iou_hit": args.tracker_iou_hit,
            "center_hit_px": args.center_hit_px,
        },
        "yolo_export": yolo_info,
        "detector": {
            "hit_frames": sum(int(r["det_hit"]) for r in positives),
            "hit_rate": sum(int(r["det_hit"]) for r in positives) / max(1, len(positives)),
            "miss_frames": detector_miss_frames,
            "center_error_px": summarize(float_or_none(r["det_center_error_px"]) for r in positives),
            "iou": summarize(float_or_none(r["det_best_iou"]) for r in positives),
        },
        "tracker": {
            "hit_frames": sum(int(r["track_hit"]) for r in positives),
            "hit_rate": sum(int(r["track_hit"]) for r in positives) / max(1, len(positives)),
            "bad_frames": tracker_bad_frames,
            "stale_frames": stale_frames,
            "negative_false_positive_frames": negative_fp_frames,
            "center_error_px": summarize(float_or_none(r["track_center_error_px"]) for r in positives),
            "iou": summarize(float_or_none(r["track_iou"]) for r in positives),
        },
    }

    write_per_frame_csv(output_dir / "per_frame_metrics.csv", per_frame)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_summary_md(output_dir / "summary.md", case_name, report)

    print(json.dumps({
        "output": str(output_dir),
        "frames": report["frames"],
        "detector_hit_rate": report["detector"]["hit_rate"],
        "tracker_hit_rate": report["tracker"]["hit_rate"],
        "detector_miss_frames": detector_miss_frames,
        "tracker_bad_frames": tracker_bad_frames,
        "stale_frames": stale_frames,
        "negative_false_positive_frames": negative_fp_frames,
    }, indent=2))
    return 0


def float_or_none(value: Any) -> Optional[float]:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
