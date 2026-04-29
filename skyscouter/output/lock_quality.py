"""
Lock-quality computation.

PRD FR-LOCK-002: lock_quality is a composite stability score, separate from
class confidence. It captures whether the track is *stable enough to commit
guidance to*, independent of whether the detector is confident about the
class.

Implementation: weighted combination of
    - bbox jitter (lower = better)
    - track hit ratio over a window (higher = better)
    - kinematic smoothness (lower acceleration variance = better)

Returns a value in [0, 1]. This is a heuristic for Sprint 0; refinement in
Sprint 1 once we have labeled data.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Tuple

from ..tracking.base_tracker import Track


def compute_lock_quality(
    track: Track,
    bbox_jitter_window: int = 8,
) -> float:
    """
    Compute a lock-quality score in [0, 1] for the given track.
    """
    history: Deque[Tuple[float, float, float]] = track.center_history
    if len(history) < 2:
        # Not enough data — minimal quality
        return 0.0

    if _touches_frame_boundary(track):
        return 0.0

    # 1. Hit ratio component: fraction of frames the track was matched
    #    in its lifetime.
    hits = max(1, track.hits)
    age = max(1, track.age_frames)
    hit_ratio = min(1.0, hits / age)

    # 2. Bbox jitter component: low jitter = high quality.
    #    Use the std-dev of recent center positions, normalized by bbox size.
    pts = list(history)[-bbox_jitter_window:]
    if len(pts) < 2:
        jitter_score = 0.5
    else:
        # Mean
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        # Variance
        vx = sum((p[0] - mx) ** 2 for p in pts) / n
        vy = sum((p[1] - my) ** 2 for p in pts) / n
        std = (vx + vy) ** 0.5
        # Normalize by bbox diagonal — a track that jitters by less than
        # one bbox diagonal per window is "stable".
        diag = (track.detection.w ** 2 + track.detection.h ** 2) ** 0.5
        if diag <= 0:
            jitter_score = 0.0
        else:
            r = std / diag
            # 0 jitter -> 1.0; 1 diagonal -> ~0.0
            jitter_score = max(0.0, min(1.0, 1.0 - r))

    # 3. Kinematic smoothness component: low velocity variance = high quality.
    if len(pts) < 3:
        smooth_score = 0.5
    else:
        vels: List[float] = []
        for i in range(1, len(pts)):
            x0, y0, t0 = pts[i - 1]
            x1, y1, t1 = pts[i]
            dt = max(1e-6, t1 - t0)
            v = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / dt
            vels.append(v)
        if not vels:
            smooth_score = 0.5
        else:
            mv = sum(vels) / len(vels)
            vv = sum((v - mv) ** 2 for v in vels) / len(vels)
            sv = vv ** 0.5
            # Normalize variance by mean (coefficient of variation)
            cv = sv / max(1e-6, mv) if mv > 0 else 1.0
            smooth_score = max(0.0, min(1.0, 1.0 - min(1.0, cv)))

    # Weighted combination — weights tuned later from labeled data
    w_hit = 0.35
    w_jit = 0.40
    w_smo = 0.25
    quality = (
        w_hit * hit_ratio
        + w_jit * jitter_score
        + w_smo * smooth_score
    )
    return max(0.0, min(1.0, quality))


def _touches_frame_boundary(track: Track) -> bool:
    d = track.detection
    if d.x <= 1.0 or d.y <= 1.0:
        return True
    return False
