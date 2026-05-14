"""Compare two gimbal-follow runs side-by-side.

Reads each run's gimbal_follow_commands.jsonl + manifest.json and prints a
single table of metrics relevant for control-loop tuning:

  - effective pipeline rate
  - lock-state distribution
  - valid-command yield
  - command-magnitude distribution and clamp saturation rate
  - sign-matrix correctness (target quadrant -> command quadrant)
  - mean |pixel error| during TRACKING/LOCKED (convergence proxy)
  - overshoot count: consecutive valid commands where pixel error flips sign

Usage:
    python scripts/dev/compare_gimbal_runs.py <baseline_run_dir> <new_run_dir>
"""
from __future__ import annotations

import collections
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _read_run(run_dir: Path) -> Dict[str, Any]:
    rows = [json.loads(x) for x in (run_dir / "gimbal_follow_commands.jsonl").read_text().splitlines() if x.strip()]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg = manifest.get("config", {})
    gf = cfg.get("gimbal_follow", {})
    return {"dir": run_dir, "rows": rows, "manifest": manifest, "cfg": gf}


def _sign(x: float) -> str:
    return "+" if x > 0 else ("-" if x < 0 else "0")


def _signs_match(target_sign: str, cmd_sign: str) -> bool:
    return cmd_sign == "0" or cmd_sign == target_sign


def _metrics(run: Dict[str, Any]) -> Dict[str, Any]:
    rows = run["rows"]
    manifest = run["manifest"]
    cfg = run["cfg"]

    started = datetime.fromisoformat(manifest["started_utc"])
    ended = datetime.fromisoformat(manifest["ended_utc"])
    dur_s = max(0.001, (ended - started).total_seconds())

    valid = [r for r in rows if r["valid"]]
    n_valid = len(valid)
    n_rows = len(rows)

    lock = collections.Counter(r["source_lock_state"] for r in rows)

    yaws = [r["yaw_cmd"] for r in valid]
    pitchs = [r["pitch_cmd"] for r in valid]
    max_yaw = int(cfg.get("max_yaw_cmd", 0)) or 1
    max_pitch = int(cfg.get("max_pitch_cmd", 0)) or 1
    sat_yaw = sum(1 for v in yaws if abs(v) == max_yaw)
    sat_pitch = sum(1 for v in pitchs if abs(v) == max_pitch)

    sign_counts = collections.Counter()
    sign_correct = 0
    for r in valid:
        px = r["pixel_error_px"]
        kx, ky = _sign(px[0]), _sign(px[1])
        sy, sp = _sign(r["yaw_cmd"]), _sign(r["pitch_cmd"])
        sign_counts[(kx, ky, sy, sp)] += 1
        if _signs_match(kx, sy) and _signs_match(ky, sp):
            sign_correct += 1

    # Mean |pixel error| over valid commands (proxy for how close to center we got)
    if valid:
        mean_abs_px_x = sum(abs(r["pixel_error_px"][0]) for r in valid) / n_valid
        mean_abs_px_y = sum(abs(r["pixel_error_px"][1]) for r in valid) / n_valid
    else:
        mean_abs_px_x = mean_abs_px_y = float("nan")

    # Overshoot detection: in a run of consecutive valid frames, if the
    # pixel-error sign flips between adjacent frames, the gimbal slewed past
    # the target. Count these flips per axis.
    flips_x = flips_y = 0
    for i in range(1, n_valid):
        prev_px = valid[i - 1]["pixel_error_px"]
        cur_px = valid[i]["pixel_error_px"]
        if prev_px[0] * cur_px[0] < 0:
            flips_x += 1
        if prev_px[1] * cur_px[1] < 0:
            flips_y += 1

    return {
        "name": run["dir"].name,
        "kp_yaw": float(cfg.get("kp_yaw", 0.0)),
        "kp_pitch": float(cfg.get("kp_pitch", 0.0)),
        "max_yaw_cmd": max_yaw,
        "max_pitch_cmd": max_pitch,
        "deadband_x": float(cfg.get("deadband_px_x", 0.0)),
        "deadband_y": float(cfg.get("deadband_px_y", 0.0)),
        "frames": manifest.get("frame_count", 0),
        "detections": manifest.get("detections_total", 0),
        "tracks_created": manifest.get("tracks_created", 0),
        "duration_s": dur_s,
        "effective_fps": manifest.get("frame_count", 0) / dur_s,
        "n_rows": n_rows,
        "n_valid": n_valid,
        "lock": dict(lock),
        "yaw_min": min(yaws) if yaws else 0,
        "yaw_max": max(yaws) if yaws else 0,
        "pitch_min": min(pitchs) if pitchs else 0,
        "pitch_max": max(pitchs) if pitchs else 0,
        "sat_yaw_pct": 100.0 * sat_yaw / max(1, n_valid),
        "sat_pitch_pct": 100.0 * sat_pitch / max(1, n_valid),
        "sign_correct_pct": 100.0 * sign_correct / max(1, n_valid),
        "sign_quadrants": dict(sign_counts),
        "mean_abs_px_x": mean_abs_px_x,
        "mean_abs_px_y": mean_abs_px_y,
        "flips_x": flips_x,
        "flips_y": flips_y,
    }


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.2f}"
    return str(v)


def _diff(baseline: Any, new: Any) -> str:
    if not isinstance(baseline, (int, float)) or not isinstance(new, (int, float)):
        return ""
    if baseline == 0:
        return f"  (+{new})" if new != 0 else ""
    delta = new - baseline
    pct = 100.0 * delta / abs(baseline)
    sign = "+" if delta >= 0 else ""
    return f"  ({sign}{delta:.2f}, {sign}{pct:.0f}%)"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2

    baseline = _read_run(Path(sys.argv[1]))
    new = _read_run(Path(sys.argv[2]))
    bm = _metrics(baseline)
    nm = _metrics(new)

    print()
    print("=" * 78)
    print(f"BASELINE : {bm['name']}")
    print(f"           kp=({bm['kp_yaw']}, {bm['kp_pitch']})  clamp=(±{bm['max_yaw_cmd']}, ±{bm['max_pitch_cmd']})")
    print(f"NEW      : {nm['name']}")
    print(f"           kp=({nm['kp_yaw']}, {nm['kp_pitch']})  clamp=(±{nm['max_yaw_cmd']}, ±{nm['max_pitch_cmd']})")
    print("=" * 78)

    fields = [
        ("effective_fps",       "Effective pipeline rate (fps)",   "higher is better"),
        ("detections",          "Detections total",                "higher is better"),
        ("tracks_created",      "Tracks created",                  "higher with the same scene"),
        ("n_valid",             "Valid gimbal commands",           "higher with same scene"),
        ("sign_correct_pct",    "Sign-matrix correctness (%)",     "want 100"),
        ("mean_abs_px_x",       "Mean |pixel error| X (px)",       "LOWER = better convergence"),
        ("mean_abs_px_y",       "Mean |pixel error| Y (px)",       "LOWER = better convergence"),
        ("sat_yaw_pct",         "Yaw  saturation (%)",             "watch — high means clamp limits us"),
        ("sat_pitch_pct",       "Pitch saturation (%)",            "watch — high means clamp limits us"),
        ("flips_x",             "Pixel-error sign flips X",        "LOWER = less overshoot"),
        ("flips_y",             "Pixel-error sign flips Y",        "LOWER = less overshoot"),
    ]
    print(f"{'Metric':<32s} {'Baseline':>14s} {'New':>14s}  Change")
    print("-" * 78)
    for k, label, _ in fields:
        b = bm[k]
        n = nm[k]
        print(f"{label:<32s} {_fmt(b):>14s} {_fmt(n):>14s}{_diff(b, n)}")

    print()
    print("--- Lock-state distribution ---")
    all_states = sorted(set(bm["lock"]) | set(nm["lock"]))
    print(f"{'state':<14s} {'baseline':>10s} {'new':>10s}")
    for st in all_states:
        print(f"  {st:<12s} {bm['lock'].get(st, 0):>10d} {nm['lock'].get(st, 0):>10d}")

    print()
    print("--- Sign matrix (target quadrant -> command quadrant) ---")
    all_keys = sorted(set(bm["sign_quadrants"]) | set(nm["sign_quadrants"]))
    print(f"{'(px_x,px_y)->(yaw,pitch)':<28s} {'baseline':>10s} {'new':>10s}")
    for k in all_keys:
        label = f"({k[0]},{k[1]}) -> ({k[2]:>2s},{k[3]:>2s})"
        print(f"  {label:<26s} {bm['sign_quadrants'].get(k, 0):>10d} {nm['sign_quadrants'].get(k, 0):>10d}")

    print()
    print("Interpretation hints:")
    print("  - If mean |pixel error| dropped AND sign flips stayed low: kp bump helped.")
    print("  - If mean |pixel error| dropped but sign flips grew: kp introducing overshoot.")
    print("  - If saturation grew a lot: clamp now limiting; raise max_*_cmd or accept it.")
    print("  - If sign_correct stayed at 100%: direction logic intact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
