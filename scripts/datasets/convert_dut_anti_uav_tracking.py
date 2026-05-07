"""Convert DUT-Anti-UAV Tracking V0 frame/GT sequences to YOLO drone-only format."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.datasets.common_yolo import (  # noqa: E402
    ConversionSummary,
    choose_split,
    image_size,
    place_image,
    reset_yolo_dirs,
    safe_stem,
    stable_digest,
    write_conversion_summary,
    write_data_yaml,
    write_label_file,
    xywh_to_yolo,
)


CLASS_NAMES = ["drone"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DUT-Anti-UAV Tracking V0 GT to YOLO.")
    parser.add_argument("--root", default="DATASETS/2_DUT-Anti-UAV/Tracking (IEEE-TITS)")
    parser.add_argument("--frames-dir", default=None, help="Override Anti-UAV-Tracking-V0/Anti-UAV-Tracking-V0 path.")
    parser.add_argument("--gt-dir", default=None, help="Override Anti-UAV-Tracking-V0GT/Anti-UAV-Tracking-V0GT path.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Total images to write for prototype conversion.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride within each sequence.")
    parser.add_argument("--max-positives-per-seq", type=int, default=None)
    parser.add_argument("--max-negatives-per-seq", type=int, default=None)
    parser.add_argument("--link-mode", choices=["copy", "hardlink", "symlink"], default="copy")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    frames_dir = Path(args.frames_dir).resolve() if args.frames_dir else root / "Anti-UAV-Tracking-V0" / "Anti-UAV-Tracking-V0"
    gt_dir = Path(args.gt_dir).resolve() if args.gt_dir else root / "Anti-UAV-Tracking-V0GT" / "Anti-UAV-Tracking-V0GT"
    out_dir = Path(args.out_dir).resolve()

    if not frames_dir.exists():
        raise SystemExit(f"DUT tracking frames dir not found: {frames_dir}")
    if not gt_dir.exists():
        raise SystemExit(f"DUT tracking GT dir not found: {gt_dir}")

    reset_yolo_dirs(out_dir)
    write_data_yaml(out_dir, CLASS_NAMES)
    summary = ConversionSummary(
        dataset="dut_anti_uav_tracking_v0",
        source_root=str(root),
        output_dir=str(out_dir),
        classes=CLASS_NAMES,
        class_remap={
            "tracking gt x y w h with positive width/height": "0 drone",
            "-100 -100 -100 -100 / non-positive width-height": "empty negative label",
        },
        limit=args.limit,
        split_policy="video sequence hash only; no frame-level leakage",
        notes=[
            "DUT Tracking V0 is drone/UAV-only temporal supervision, useful for lock continuity and tiny-drone recall.",
            "It does not provide bird/airplane/helicopter confusers; use it as drone-positive/absent-target data, not semantic rejection data.",
        ],
    )

    written = 0
    for gt_path in sorted(gt_dir.glob("*_gt.txt")):
        if args.limit is not None and written >= args.limit:
            break
        seq_name = gt_path.stem.replace("_gt", "")
        seq_dir = frames_dir / seq_name
        if not seq_dir.exists():
            summary.add_skip(seq_dir, "matching frame sequence folder missing")
            continue
        remaining = None if args.limit is None else args.limit - written
        written += convert_sequence(seq_dir, gt_path, out_dir, summary, args, remaining)

    write_conversion_summary(out_dir, summary)
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "summary": str(out_dir / "conversion_summary.json"),
                "images": summary.image_count,
                "objects": summary.object_count,
                "empty_labels": summary.empty_label_count,
            },
            indent=2,
        )
    )
    return 0


def convert_sequence(
    seq_dir: Path,
    gt_path: Path,
    out_dir: Path,
    summary: ConversionSummary,
    args: argparse.Namespace,
    remaining: Optional[int],
) -> int:
    rows = parse_gt_rows(gt_path, summary)
    frame_paths = sorted(seq_dir.glob("*.jpg"))
    if len(rows) != len(frame_paths):
        summary.add_skip(
            seq_dir,
            f"frame/GT count mismatch: frames={len(frame_paths)} gt_rows={len(rows)}",
        )
        return 0
    if not rows:
        return 0

    selected_indices = select_indices(rows, args)
    if remaining is not None:
        selected_indices = selected_indices[: max(0, remaining)]
    if not selected_indices:
        return 0

    split = choose_split(f"dut_tracking:{seq_dir.name}", seed=args.seed)
    written = 0
    for frame_idx in selected_indices:
        image_path = frame_paths[frame_idx]
        size = image_size(image_path)
        if size is None:
            summary.add_skip(image_path, "corrupt/unreadable image")
            continue
        width, height = size
        boxes = []
        x, y, w, h = rows[frame_idx]
        if w > 0 and h > 0:
            box = xywh_to_yolo(0, x, y, w, h, width, height, source_label="drone")
            if box is None:
                summary.add_invalid_box(f"{gt_path}:{frame_idx + 1}", [x, y, w, h], "degenerate after clipping")
            else:
                boxes.append(box)
        stem = safe_stem(f"dut_tracking_{seq_dir.name}_f{frame_idx + 1:05d}")
        dst_image = out_dir / "images" / split / f"{stem}{image_path.suffix.lower()}"
        dst_label = out_dir / "labels" / split / f"{stem}.txt"
        place_image(image_path, dst_image, args.link_mode)
        write_label_file(dst_label, boxes)
        summary.add_image(split, boxes, width, height)
        written += 1
    return written


def parse_gt_rows(gt_path: Path, summary: ConversionSummary) -> List[Tuple[float, float, float, float]]:
    rows: List[Tuple[float, float, float, float]] = []
    for line_no, line in enumerate(gt_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 4:
            summary.add_invalid_box(f"{gt_path}:{line_no}", stripped, "expected 4 fields: x y w h")
            rows.append((-100.0, -100.0, -100.0, -100.0))
            continue
        try:
            x, y, w, h = (float(part) for part in parts)
        except ValueError:
            summary.add_invalid_box(f"{gt_path}:{line_no}", stripped, "non-numeric field")
            rows.append((-100.0, -100.0, -100.0, -100.0))
            continue
        rows.append((x, y, w, h))
    return rows


def select_indices(rows: Sequence[Tuple[float, float, float, float]], args: argparse.Namespace) -> List[int]:
    stride = max(1, int(args.stride))
    positive = [idx for idx in range(0, len(rows), stride) if rows[idx][2] > 0 and rows[idx][3] > 0]
    negative = [idx for idx in range(0, len(rows), stride) if rows[idx][2] <= 0 or rows[idx][3] <= 0]
    positive = cap_indices(positive, args.max_positives_per_seq, args.seed, "pos")
    negative = cap_indices(negative, args.max_negatives_per_seq, args.seed, "neg")
    return sorted(set(positive) | set(negative))


def cap_indices(indices: List[int], cap: Optional[int], seed: int, salt: str) -> List[int]:
    if cap is None or len(indices) <= cap:
        return indices
    ordered = sorted(indices, key=lambda idx: stable_digest(f"{seed}:{salt}:{idx}"))
    return sorted(ordered[:cap])


if __name__ == "__main__":
    raise SystemExit(main())
