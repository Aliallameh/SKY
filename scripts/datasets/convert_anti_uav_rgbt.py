"""Convert Anti-UAV-RGBT visible-camera sequences to YOLO drone-only format."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import cv2

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.datasets.common_yolo import (  # noqa: E402
    ConversionSummary,
    choose_split,
    reset_yolo_dirs,
    safe_stem,
    stable_digest,
    write_conversion_summary,
    write_data_yaml,
    write_image_bgr,
    write_label_file,
    xywh_to_yolo,
)


CLASS_NAMES = ["drone"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Anti-UAV-RGBT RGB/visible sequences to YOLO.")
    parser.add_argument("--root", default="DATASETS/1_Anti-UAV-RGBT")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Total images to write for prototype conversion.")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-positives-per-seq", type=int, default=60)
    parser.add_argument("--max-negatives-per-seq", type=int, default=5)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Raw split folders to read.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not root.exists():
        raise SystemExit(f"Anti-UAV-RGBT root not found: {root}")
    reset_yolo_dirs(out_dir)
    write_data_yaml(out_dir, CLASS_NAMES)
    summary = ConversionSummary(
        dataset="anti_uav_rgbt_visible",
        source_root=str(root),
        output_dir=str(out_dir),
        classes=CLASS_NAMES,
        class_remap={"visible exist=1 target": "0 drone", "exist=0": "empty negative label"},
        limit=args.limit,
        notes=[
            "RGB/visible modality only; infrared is intentionally excluded for this product camera path.",
            "YOLO split is sequence-hash based; frames from one raw sequence never cross splits.",
        ],
    )

    written = 0
    for raw_split in args.splits:
        split_dir = root / raw_split
        if not split_dir.is_dir():
            summary.add_skip(split_dir, "raw split folder missing")
            continue
        for seq_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if args.limit is not None and written >= args.limit:
                break
            written += convert_sequence(seq_dir, raw_split, out_dir, summary, args, remaining=None if args.limit is None else args.limit - written)
        if args.limit is not None and written >= args.limit:
            break

    write_conversion_summary(out_dir, summary)
    print(json.dumps({"out_dir": str(out_dir), "summary": str(out_dir / "conversion_summary.json"), "images": summary.image_count}, indent=2))
    return 0


def convert_sequence(
    seq_dir: Path,
    raw_split: str,
    out_dir: Path,
    summary: ConversionSummary,
    args: argparse.Namespace,
    remaining: Optional[int],
) -> int:
    video_path = seq_dir / "visible.mp4"
    ann_path = seq_dir / "visible.json"
    if not video_path.exists() or not ann_path.exists():
        summary.add_skip(seq_dir, "missing visible.mp4 or visible.json")
        return 0
    try:
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
    except Exception as exc:
        summary.add_skip(ann_path, f"invalid json: {exc}")
        return 0
    exist = [int(v) for v in ann.get("exist", [])]
    gt_rect = ann.get("gt_rect", [])
    if not exist:
        summary.add_skip(ann_path, "empty exist array")
        return 0

    stride = max(1, int(args.stride))
    pos = [idx for idx in range(0, len(exist), stride) if exist[idx] == 1]
    neg = [idx for idx in range(0, len(exist), stride) if exist[idx] == 0]
    rng = random.Random(int(stable_digest(f"{args.seed}:{seq_dir}")[:12], 16))
    if args.max_positives_per_seq is not None and len(pos) > args.max_positives_per_seq:
        rng.shuffle(pos)
        pos = sorted(pos[: args.max_positives_per_seq])
    if args.max_negatives_per_seq > 0 and len(neg) > args.max_negatives_per_seq:
        rng.shuffle(neg)
        neg = sorted(neg[: args.max_negatives_per_seq])
    elif args.max_negatives_per_seq <= 0:
        neg = []
    selected = sorted(set(pos) | set(neg))
    if remaining is not None:
        selected = selected[: max(0, remaining)]
    if not selected:
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        summary.add_skip(video_path, "cannot open video")
        return 0
    written = 0
    split = choose_split(f"anti_uav_rgbt:{raw_split}:{seq_dir.name}", seed=args.seed)
    pos_set = set(pos)
    try:
        previous = -1
        for frame_idx in selected:
            if frame_idx != previous + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            previous = frame_idx
            if not ok or frame is None:
                summary.add_skip(f"{video_path}:{frame_idx}", "cannot decode frame")
                continue
            height, width = frame.shape[:2]
            boxes = []
            if frame_idx in pos_set:
                rect = gt_rect[frame_idx] if frame_idx < len(gt_rect) else None
                if not isinstance(rect, Sequence) or len(rect) != 4:
                    summary.add_invalid_box(f"{ann_path}:{frame_idx}", rect, "missing gt_rect for visible frame")
                    continue
                box = xywh_to_yolo(0, float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]), width, height, source_label="drone")
                if box is None:
                    summary.add_invalid_box(f"{ann_path}:{frame_idx}", rect, "degenerate/clipped box")
                    continue
                boxes.append(box)
            stem = safe_stem(f"anti_uav_rgbt_{raw_split}_{seq_dir.name}_visible_f{frame_idx:06d}")
            image_out = out_dir / "images" / split / f"{stem}.jpg"
            label_out = out_dir / "labels" / split / f"{stem}.txt"
            write_image_bgr(image_out, frame)
            write_label_file(label_out, boxes)
            summary.add_image(split, boxes, width, height)
            written += 1
    finally:
        cap.release()
    return written


if __name__ == "__main__":
    raise SystemExit(main())
