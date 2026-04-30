"""Extract a highlight reel of frames where the trained detector missed a
positive sparse-GT bbox.

For each missed positive (matched=false, negative=false in eval_report.json),
this script:
  - reads the corrected GT bbox from the user's GT CSV;
  - seeks to that frame in the source video;
  - runs the trained YOLO detector at a *very* low confidence threshold to
    surface any sub-threshold signal;
  - composes a frame with the GT in green and YOLO low-conf hits in yellow,
    plus frame number / labels.

It writes both a slow-motion MP4 (each miss held ~2 sec) and per-frame JPEGs
so you can scrub through them quickly.

This is a diagnostic tool, not part of the main pipeline. It exists so we can
tell whether the missed frames are "model is blind" (no yellow boxes) vs
"model sees it but conf is below 0.25" (yellow box overlapping the green GT).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build miss-frame highlight reel")
    p.add_argument(
        "--eval-report",
        default="data/outputs/run_trained_detector_eval/eval_report.json",
    )
    p.add_argument(
        "--video",
        default=r"C:\Users\Ali\Desktop\SKY\data\videos\my_drone_chase.MP4",
    )
    p.add_argument(
        "--gt-csv",
        default=r"C:\Users\Ali\Documents\Datasets\drone_sparse_gt_corrected.csv",
    )
    p.add_argument(
        "--weights",
        default="data/training/runs/detect/yolo11s_airborne_drone_vs_bird_v1/weights/best.pt",
    )
    p.add_argument(
        "--out-dir",
        default="data/outputs/miss_highlights",
    )
    p.add_argument(
        "--low-conf",
        type=float,
        default=0.01,
        help="Surface YOLO detections down to this confidence",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=1024,
    )
    p.add_argument(
        "--hold-seconds",
        type=float,
        default=2.0,
        help="How long to display each missed frame in the highlight reel",
    )
    return p.parse_args()


def load_missed_frames(eval_report_path: Path) -> List[int]:
    data = json.loads(eval_report_path.read_text(encoding="utf-8"))
    misses = []
    for row in data.get("gt", {}).get("per_frame", []):
        if not row.get("matched") and not row.get("negative") and row.get("label") == "drone":
            misses.append(int(row["frame_id"]))
    return sorted(set(misses))


def load_gt(gt_csv: Path) -> Dict[int, Tuple[float, float, float, float, str]]:
    out: Dict[int, Tuple[float, float, float, float, str]] = {}
    with gt_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fid = int(row["frame_id"])
            except (ValueError, KeyError):
                continue
            label = row.get("label", "").strip()
            if label.lower() in ("", "negative"):
                continue
            try:
                x1 = float(row["x1"]); y1 = float(row["y1"])
                x2 = float(row["x2"]); y2 = float(row["y2"])
            except (ValueError, KeyError):
                continue
            out[fid] = (x1, y1, x2, y2, label)
    return out


def draw_box(
    img: np.ndarray,
    box: Tuple[float, float, float, float],
    color: Tuple[int, int, int],
    label: str,
    thickness: int = 3,
) -> None:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ty = max(y1 - 8, th + 4)
        cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 8, ty + 4), color, -1)
        cv2.putText(
            img, label, (x1 + 4, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA,
        )


def main() -> int:
    args = parse_args()
    eval_report = Path(args.eval_report)
    video_path = Path(args.video)
    gt_csv = Path(args.gt_csv)
    weights = Path(args.weights)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    misses = load_missed_frames(eval_report)
    print(f"Missed positive frames: {len(misses)} -> {misses}")
    if not misses:
        print("Nothing to do; gate already clean on positives.")
        return 0

    gt_map = load_gt(gt_csv)
    missing_gt = [f for f in misses if f not in gt_map]
    if missing_gt:
        print(f"WARNING: {len(missing_gt)} missed frames have no GT bbox row: {missing_gt}")

    # Lazy import — avoid forcing torch import for the no-op path
    from ultralytics import YOLO  # type: ignore
    print(f"Loading detector: {weights}")
    model = YOLO(str(weights))
    class_names = model.names

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path.name}  {w}x{h} @ {fps:.2f}fps  total={total}")

    # MP4 writer — slow-motion: hold each miss for ~hold_seconds.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    reel_path = out_dir / "miss_highlights.mp4"
    writer = cv2.VideoWriter(str(reel_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"Could not open writer: {reel_path}")

    hold_frames = max(1, int(round(fps * args.hold_seconds)))

    summary: List[dict] = []
    for fid in misses:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"  frame {fid}: read failed; skipping")
            continue

        canvas = frame.copy()

        # Draw GT (green) if available
        gt = gt_map.get(fid)
        if gt is not None:
            x1, y1, x2, y2, gt_label = gt
            draw_box(canvas, (x1, y1, x2, y2), (0, 255, 0), f"GT: {gt_label}")
            gt_ce: Optional[Tuple[float, float]] = (
                (x1 + x2) / 2.0, (y1 + y2) / 2.0,
            )
        else:
            gt_ce = None

        # Run detector at very low conf
        results = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.low_conf,
            iou=0.45,
            verbose=False,
        )
        det_count = 0
        best_iou = 0.0
        best_det_conf = 0.0
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            xyxy = results[0].boxes.xyxy.cpu().numpy()
            conf = results[0].boxes.conf.cpu().numpy()
            cls = results[0].boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                bx = tuple(xyxy[i].tolist())
                cn = float(conf[i])
                ci = int(cls[i])
                lbl = f"{class_names.get(ci, str(ci))} {cn:.2f}"
                draw_box(canvas, bx, (0, 215, 255), lbl, thickness=2)
                det_count += 1
                if gt is not None:
                    ix1 = max(bx[0], gt[0]); iy1 = max(bx[1], gt[1])
                    ix2 = min(bx[2], gt[2]); iy2 = min(bx[3], gt[3])
                    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
                    inter = iw * ih
                    a_det = (bx[2] - bx[0]) * (bx[3] - bx[1])
                    a_gt = (gt[2] - gt[0]) * (gt[3] - gt[1])
                    union = a_det + a_gt - inter
                    if union > 0:
                        iou = inter / union
                        if iou > best_iou:
                            best_iou = iou
                            best_det_conf = cn

        banner = (
            f"frame {fid}  yolo_dets@conf>={args.low_conf:.2f}={det_count}  "
            f"best_iou_vs_gt={best_iou:.2f}  best_det_conf={best_det_conf:.2f}"
        )
        cv2.rectangle(canvas, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.putText(
            canvas, banner, (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

        # Per-frame JPEG
        jpeg_path = out_dir / f"miss_{fid:06d}.jpg"
        cv2.imwrite(str(jpeg_path), canvas)

        # Hold in the reel
        for _ in range(hold_frames):
            writer.write(canvas)

        summary.append({
            "frame_id": fid,
            "yolo_low_conf_detections": det_count,
            "best_iou_vs_gt": round(best_iou, 4),
            "best_det_conf": round(best_det_conf, 4),
            "gt_present": gt is not None,
        })
        print(f"  frame {fid}: dets={det_count}  best_iou={best_iou:.2f}  best_conf={best_det_conf:.2f}")

    writer.release()
    cap.release()

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote: {reel_path}")
    print(f"Wrote per-frame JPEGs into: {out_dir}")
    print(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
