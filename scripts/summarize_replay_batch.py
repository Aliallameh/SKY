"""Summarize a directory of Skyscouter replay runs.

The script reads one run directory per video and writes batch-level CSV, JSON,
Markdown, interval, and review-candidate artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize Skyscouter replay batch outputs")
    p.add_argument("--root", required=True, help="Batch output root containing one run dir per video")
    return p.parse_args()


def pct(n: int, d: int) -> float:
    return 0.0 if not d else round(100.0 * n / d, 2)


def quantile(values: List[float], frac: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[int(frac * (len(values) - 1))]


def read_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_intervals(
    rows: List[Dict[str, str]],
    key_fn: Callable[[Dict[str, str]], str],
    want_fn: Callable[[Dict[str, str]], bool],
) -> List[Tuple[int, int, str]]:
    out: List[Tuple[int, int, str]] = []
    start: int | None = None
    last: int | None = None
    label: str | None = None
    for row in rows:
        fid = int(row["frame_index"])
        val = key_fn(row)
        want = want_fn(row)
        if want:
            if start is None:
                start = fid
                label = val
            elif label != val or last is None or fid != last + 1:
                out.append((start, last if last is not None else start, label or ""))
                start = fid
                label = val
            last = fid
        elif start is not None:
            out.append((start, last if last is not None else start, label or ""))
            start = None
            last = None
            label = None
    if start is not None:
        out.append((start, last if last is not None else start, label or ""))
    return out


def counter_text(counter: Counter[str]) -> str:
    return "; ".join(f"{key}:{value}" for key, value in counter.most_common())


def safe_float(value: str | None) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def summarize_run(run_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    diag_path = run_dir / "diagnostics.csv"
    manifest = read_manifest(run_dir / "manifest.json")
    if not diag_path.exists() or diag_path.stat().st_size == 0:
        return {"video": run_dir.name, "status": "missing_or_empty"}, [], []

    rows = list(csv.DictReader(diag_path.open(encoding="utf-8")))
    frames = len(rows)
    target_rows = [r for r in rows if r.get("message_type") == "TARGET_STATE"]
    confidences = [v for v in (safe_float(r.get("confidence")) for r in target_rows) if v is not None]
    latencies = [v for v in (safe_float(r.get("latency_ms")) for r in rows) if v is not None]
    widths = [v for v in (safe_float(r.get("w")) for r in target_rows) if v is not None]
    heights = [v for v in (safe_float(r.get("h")) for r in target_rows) if v is not None]
    states = Counter(r.get("lock_state") or "NONE" for r in rows)
    labels = Counter(r.get("class_label") or "NO_TARGET" for r in rows)
    semantic = Counter(r.get("semantic_label") or "NO_TARGET" for r in rows)

    locked = states.get("LOCKED", 0) + states.get("STRIKE_READY", 0)
    tracking_or_better = states.get("TRACKING", 0) + states.get("LOCKED", 0) + states.get("STRIKE_READY", 0)
    stale = sum(1 for r in target_rows if r.get("time_since_update") not in ("", "0", None))

    summary = {
        "video": run_dir.name,
        "status": manifest.get("status", "unknown"),
        "frames": frames,
        "target_rows": len(target_rows),
        "target_coverage_pct": pct(len(target_rows), frames),
        "no_target_frames": frames - len(target_rows),
        "tracking_or_better_frames": tracking_or_better,
        "tracking_or_better_pct": pct(tracking_or_better, frames),
        "locked_frames": locked,
        "locked_pct": pct(locked, frames),
        "guidance_valid_frames": sum(1 for r in rows if r.get("guidance_valid") == "1"),
        "drone_label_frames": labels.get("drone", 0),
        "drone_label_pct": pct(labels.get("drone", 0), frames),
        "airplane_label_frames": labels.get("airplane", 0),
        "airplane_label_pct": pct(labels.get("airplane", 0), frames),
        "unknown_airborne_label_frames": labels.get("unknown_airborne", 0),
        "unknown_airborne_label_pct": pct(labels.get("unknown_airborne", 0), frames),
        "confidence_mean": round(statistics.mean(confidences), 3) if confidences else "",
        "confidence_median": round(statistics.median(confidences), 3) if confidences else "",
        "confidence_p90": round(quantile(confidences, 0.90), 3) if confidences else "",
        "confidence_max": round(max(confidences), 3) if confidences else "",
        "bbox_w_median_px": round(statistics.median(widths), 2) if widths else "",
        "bbox_h_median_px": round(statistics.median(heights), 2) if heights else "",
        "latency_median_ms": round(statistics.median(latencies), 2) if latencies else "",
        "latency_p95_ms": round(quantile(latencies, 0.95), 2) if latencies else "",
        "stale_tracker_frames": stale,
        "detections_total": manifest.get("detections_total", ""),
        "tracks_created": manifest.get("tracks_created", ""),
        "raw_label_counts": counter_text(labels),
        "semantic_label_counts": counter_text(semantic),
        "lock_state_counts": counter_text(states),
        "annotated_mp4": str(run_dir / "annotated.mp4"),
        "diagnostics_csv": str(diag_path),
    }

    interval_rows: List[Dict[str, Any]] = []
    for start, end, label in build_intervals(
        rows,
        lambda r: r.get("lock_state") or "",
        lambda r: r.get("lock_state") in ("TRACKING", "LOCKED", "STRIKE_READY"),
    ):
        interval_rows.append({
            "video": run_dir.name,
            "start_frame": start,
            "end_frame": end,
            "duration_frames": end - start + 1,
            "kind": "tracking_or_better",
            "label": label,
        })
    for start, end, label in build_intervals(
        rows,
        lambda r: r.get("class_label") or "NO_TARGET",
        lambda r: bool(r.get("class_label")),
    ):
        if end - start + 1 >= 10:
            interval_rows.append({
                "video": run_dir.name,
                "start_frame": start,
                "end_frame": end,
                "duration_frames": end - start + 1,
                "kind": "raw_label_run",
                "label": label,
            })

    review_rows = build_review_candidates(run_dir.name, rows, target_rows)
    return summary, interval_rows, review_rows


def build_review_candidates(
    video_name: str,
    rows: List[Dict[str, str]],
    target_rows: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    candidates: List[Tuple[int, str]] = []
    for start, end, _label in build_intervals(
        rows,
        lambda r: r.get("lock_state") or "",
        lambda r: r.get("lock_state") in ("TRACKING", "LOCKED", "STRIKE_READY"),
    ):
        candidates.append((start + (end - start) // 2, "mid_tracking_interval"))

    scored: List[Tuple[float, int, str]] = []
    for row in target_rows:
        conf = safe_float(row.get("confidence"))
        if conf is not None:
            scored.append((conf, int(row["frame_index"]), row.get("class_label") or ""))
    for _conf, fid, label in sorted(scored, reverse=True)[:10]:
        candidates.append((fid, f"high_conf_{label}"))

    seen: set[int] = set()
    out: List[Dict[str, Any]] = []
    for fid, reason in candidates:
        if fid in seen or not (0 <= fid < len(rows)):
            continue
        seen.add(fid)
        row = rows[fid]
        out.append({
            "video": video_name,
            "frame_index": fid,
            "reason": reason,
            "lock_state": row.get("lock_state", ""),
            "class_label": row.get("class_label", ""),
            "semantic_label": row.get("semantic_label", ""),
            "confidence": row.get("confidence", ""),
            "x": row.get("x", ""),
            "y": row.get("y", ""),
            "w": row.get("w", ""),
            "h": row.get("h", ""),
        })
    return out


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verdict(row: Dict[str, Any]) -> str:
    if row.get("status") == "missing_or_empty":
        return "missing/empty"
    locked = float(row.get("locked_pct") or 0)
    target = float(row.get("target_coverage_pct") or 0)
    airplane = float(row.get("airplane_label_pct") or 0)
    drone = float(row.get("drone_label_pct") or 0)
    if locked > 20:
        return "best lock candidate"
    if target > 50 and airplane > drone:
        return "tracking exists; semantic gate blocks lock"
    if target < 5:
        return "mostly missed/no target at this threshold"
    return "review"


def write_markdown(root: Path, summary_rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Fresh Video Batch Summary",
        "",
        f"Run root: `{root}`",
        "",
        (
            "Important: these are proxy metrics because the fresh videos do not "
            "yet have human GT boxes. True accuracy requires annotating sampled "
            "frames from `review_candidates.csv` or a denser GT packet."
        ),
        "",
        (
            "| Video | Target coverage | Tracking+ | Locked | Drone label | "
            "Airplane label | Median conf | Median latency | Verdict |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        if row.get("status") == "missing_or_empty":
            lines.append(f"| {row['video']} | - | - | - | - | - | - | - | missing/empty |")
            continue
        lines.append(
            f"| {row['video']} | {row['target_coverage_pct']}% | "
            f"{row['tracking_or_better_pct']}% | {row['locked_pct']}% | "
            f"{row['drone_label_pct']}% | {row['airplane_label_pct']}% | "
            f"{row['confidence_median']} | {row['latency_median_ms']} ms | "
            f"{verdict(row)} |"
        )
    lines.extend([
        "",
        "Artifacts:",
        "- `batch_summary.csv` / `batch_summary.json`: per-video metrics.",
        "- `batch_intervals.csv`: tracking and label intervals.",
        "- `review_candidates.csv`: frames to annotate first for true accuracy.",
    ])
    (root / "BATCH_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    root = Path(parse_args().root)
    summary_rows: List[Dict[str, Any]] = []
    interval_rows: List[Dict[str, Any]] = []
    review_rows: List[Dict[str, Any]] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("Video_")):
        summary, intervals, review = summarize_run(run_dir)
        summary_rows.append(summary)
        interval_rows.extend(intervals)
        review_rows.extend(review)

    write_csv(root / "batch_summary.csv", summary_rows)
    (root / "batch_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    write_csv(root / "batch_intervals.csv", interval_rows)
    write_csv(root / "review_candidates.csv", review_rows)
    write_markdown(root, summary_rows)

    print(root / "BATCH_SUMMARY.md")
    print(root / "batch_summary.csv")
    print(root / "review_candidates.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
