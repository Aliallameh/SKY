"""Diagnostics and sparse-GT evaluation for Skyscouter replay runs."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..schemas import TargetState
from ..tracking.base_tracker import Track


BBox = Tuple[float, float, float, float]


def xywh_to_xyxy(xywh: Optional[List[float]]) -> Optional[BBox]:
    if not xywh:
        return None
    x, y, w, h = [float(v) for v in xywh]
    return x, y, x + w, y + h


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    denom = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return 0.0 if denom <= 0 else float(inter / denom)


def center_error(a: BBox, b: BBox) -> float:
    acx, acy = (a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5
    bcx, bcy = (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5
    return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5)


class DiagnosticsWriter:
    """Writes frame-level debug rows for replay inspection."""

    def __init__(self, path: str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._fh,
            fieldnames=[
                "frame_index",
                "timestamp_utc",
                "message_type",
                "track_id",
                "lock_state",
                "guidance_valid",
                "x",
                "y",
                "w",
                "h",
                "confidence",
                "lock_quality",
                "tracker_status",
                "detector_source",
                "time_since_update",
                "hits",
                "flow_points",
                "flow_quality",
                "association_score",
                "latency_ms",
                "fault_flags",
            ],
        )
        self._writer.writeheader()

    def write(self, frame_index: int, state: TargetState, primary: Optional[Track]) -> None:
        bbox = state.bbox_xywh or ["", "", "", ""]
        self._writer.writerow(
            {
                "frame_index": frame_index,
                "timestamp_utc": state.timestamp_utc,
                "message_type": state.message_type,
                "track_id": "" if state.track_id is None else state.track_id,
                "lock_state": state.lock_state,
                "guidance_valid": int(bool(state.guidance_valid)),
                "x": bbox[0],
                "y": bbox[1],
                "w": bbox[2],
                "h": bbox[3],
                "confidence": "" if state.confidence is None else f"{state.confidence:.6f}",
                "lock_quality": "" if state.lock_quality is None else f"{state.lock_quality:.6f}",
                "tracker_status": "" if primary is None else primary.status,
                "detector_source": "" if primary is None else primary.source,
                "time_since_update": "" if primary is None else primary.time_since_update,
                "hits": "" if primary is None else primary.hits,
                "flow_points": "" if primary is None else primary.flow_points,
                "flow_quality": "" if primary is None else f"{primary.flow_quality:.6f}",
                "association_score": "" if primary is None else f"{primary.association_score:.6f}",
                "latency_ms": "" if state.latency_ms is None else f"{state.latency_ms:.6f}",
                "fault_flags": ",".join(state.fault_flags),
            }
        )

    def close(self) -> None:
        self._fh.close()


class EvaluationCollector:
    """Accumulates target states and evaluates them against sparse GT."""

    def __init__(self, gt_path: Optional[str] = None):
        self._states: List[Dict[str, Any]] = []
        self._gt_path = gt_path

    def add(self, frame_index: int, state: TargetState) -> None:
        self._states.append({"frame_index": frame_index, "state": state.to_dict()})

    def write_report(self, path: str) -> Dict[str, Any]:
        report = self.build_report()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def build_report(self) -> Dict[str, Any]:
        states = self._states
        locked = [r for r in states if r["state"].get("lock_state") in ("LOCKED", "STRIKE_READY")]
        guidance = [r for r in states if r["state"].get("guidance_valid")]
        track_ids = [r["state"].get("track_id") for r in states if r["state"].get("track_id") is not None]
        switches = sum(1 for a, b in zip(track_ids, track_ids[1:]) if a != b)
        base = {
            "schema_version": "skyscout.eval_report.v1",
            "frames": len(states),
            "lock_state_counts": dict(Counter(r["state"].get("lock_state") for r in states)),
            "guidance_valid_frames": len(guidance),
            "locked_frames": len(locked),
            "published_track_count": len(set(track_ids)),
            "published_track_id_switches": switches,
            "gt": {"enabled": False, "reason": "no_gt_path_configured"},
        }
        if not self._gt_path:
            return base
        gt_path = Path(self._gt_path)
        if not gt_path.exists():
            base["gt"] = {"enabled": False, "reason": f"gt_file_not_found: {gt_path}"}
            return base
        gt_rows = read_gt_csv(gt_path)
        by_frame = {int(r["frame_index"]): r["state"] for r in states}
        per_frame = []
        for gt in gt_rows:
            frame = int(gt["frame_id"])
            state = by_frame.get(frame)
            pred = xywh_to_xyxy(state.get("bbox_xywh")) if state else None
            iou = bbox_iou(pred, gt["bbox"]) if pred else None
            cerr = center_error(pred, gt["bbox"]) if pred else None
            label = str(gt.get("label", "drone")).lower()
            is_negative = label in {"negative", "hard_negative", "no_drone", "background"}
            per_frame.append(
                {
                    "frame_id": frame,
                    "matched": pred is not None,
                    "iou": iou,
                    "center_error_px": cerr,
                    "lock_state": None if state is None else state.get("lock_state"),
                    "guidance_valid": False if state is None else bool(state.get("guidance_valid")),
                    "track_id": None if state is None else state.get("track_id"),
                    "label": label,
                    "negative": is_negative,
                }
            )

        positives = [r for r in per_frame if not r["negative"]]
        negatives = [r for r in per_frame if r["negative"]]
        matched_pos = [r for r in positives if r["matched"]]
        center_errors = [float(r["center_error_px"]) for r in matched_pos if r["center_error_px"] is not None]
        ious = [float(r["iou"]) for r in matched_pos if r["iou"] is not None]
        pos_track_ids = [r["track_id"] for r in positives if r.get("track_id") is not None]
        gt_switches = sum(1 for a, b in zip(pos_track_ids, pos_track_ids[1:]) if a != b)
        false_locks = [r for r in negatives if r["guidance_valid"] or r["lock_state"] in ("LOCKED", "STRIKE_READY")]
        base["gt"] = {
            "enabled": True,
            "gt_file": str(gt_path),
            "gt_frames": len(gt_rows),
            "positive_frames": len(positives),
            "negative_frames": len(negatives),
            "matched_positive_frames": len(matched_pos),
            "matched_positive_rate": len(matched_pos) / max(1, len(positives)),
            "mean_iou": _mean(ious),
            "median_iou": None if not ious else float(median(ious)),
            "median_center_error_px": None if not center_errors else float(median(center_errors)),
            "max_center_error_px": None if not center_errors else float(max(center_errors)),
            "labeled_track_id_switches": gt_switches,
            "false_lock_negative_frames": len(false_locks),
            "passes_current_video_gate": bool(
                len(matched_pos) / max(1, len(positives)) >= 0.90
                and (not center_errors or median(center_errors) <= 25.0)
                and gt_switches <= 3
                and len(false_locks) == 0
            ),
            "per_frame": per_frame,
        }
        return base


def read_gt_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = str(row.get("review_status", "ok")).lower()
            if status in {"skip", "draft", "ignore"}:
                continue
            label = str(row.get("label", "drone")).lower()
            is_negative = label in {"negative", "hard_negative", "no_drone", "background"}
            if is_negative:
                bbox = (0.0, 0.0, 0.0, 0.0)
            else:
                try:
                    x1, y1, x2, y2 = [float(row[k]) for k in ("x1", "y1", "x2", "y2")]
                except (TypeError, ValueError):
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue
                bbox = (x1, y1, x2, y2)
            rows.append(
                {
                    "frame_id": int(row["frame_id"]),
                    "bbox": bbox,
                    "label": label,
                    "occluded": row.get("occluded", ""),
                    "visibility": row.get("visibility", ""),
                }
            )
    return rows


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    return None if not vals else float(sum(vals) / len(vals))
