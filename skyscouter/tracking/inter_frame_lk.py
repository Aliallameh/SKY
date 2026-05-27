"""Inter-frame Lucas-Kanade tracker running at full camera FPS.

WHY THIS EXISTS
---------------
The YOLO pipeline processes ~5-7 frames per second. Between consecutive
YOLO frames the gap is 140-180 ms. At a drone speed of 10 m/s and a
typical focal length the target center can move 30-50 px in that gap —
far too large for reliable sparse optical flow.

The reference "click-to-track" code the user found works because it runs
LK on every raw camera frame (~33 ms, 5-10 px per step). That is the only
reason it "never loses" the target. This module replicates that cadence
autonomously, without any human click.

ARCHITECTURE
------------
Camera reader thread  ──► feed_frame(bgr)  ──► internal queue (maxsize=4)
InterFrameLKThread    ◄── drains queue     ──► runs LK  ──► _latest_bbox
YOLO thread           ──► reset_from_yolo(gray, det)   (on confirmed detection)
YOLO thread           ◄── get_latest()                  (in coast / predict mode)

The YOLO tracker calls:
  • reset_from_yolo(gray, det)  — every time a detection is confirmed, to
                                   resync the LK reference point to the YOLO box
  • get_latest()                — when no YOLO detection is found, to get the
                                   inter-frame-tracked position as a replacement
  • clear()                     — when the track is lost / cleared

The pipeline calls:
  • feed_frame(bgr)             — for every raw camera frame from the reader
                                   thread, regardless of whether YOLO runs on it
"""
from __future__ import annotations

import queue
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

from ..perception.base_detector import Detection

# ------------------------------------------------------------------
# Module-level LK config (shared by thread and fallback predictor)
# ------------------------------------------------------------------

_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=4,   # One more level than the old per-YOLO LK — handles larger
                  # residual motion during fast camera pans
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
)

_MIN_TRACKED_POINTS = 3   # Much lower than old lk_min_points=6/8 because we
                           # refresh features often and run every 33ms
_BACK_ERR_THRESHOLD  = 2.5  # pixels — forward-backward consistency threshold
_REFRESH_BELOW       = 10   # refresh feature set when tracked points drop below this
_MAX_CORNERS         = 40   # fewer than old 80; enough for a tiny target patch


# ------------------------------------------------------------------
# Feature extractor (shared utility)
# ------------------------------------------------------------------

def find_features_in_region(gray: np.ndarray, d: Detection) -> np.ndarray:
    """Shi-Tomasi corner detection in an EXPANDED region around a detection.

    The search area is padded 35% in each direction so that features just
    outside the raw YOLO bbox (which is often tight) are captured.  This
    is critical for tiny targets (0.08-0.22% of frame) where the bbox
    interior may have very few texture pixels.

    Returns an (N, 1, 2) float32 array, possibly empty.
    """
    h_img, w_img = gray.shape[:2]

    pad_x = max(8, int(0.35 * max(1.0, d.w)))
    pad_y = max(8, int(0.35 * max(1.0, d.h)))
    x1 = int(np.clip(round(d.x) - pad_x,     0,          max(0, w_img - 2)))
    y1 = int(np.clip(round(d.y) - pad_y,     0,          max(0, h_img - 2)))
    x2 = int(np.clip(round(d.x + d.w) + pad_x, x1 + 1,  max(1, w_img - 1)))
    y2 = int(np.clip(round(d.y + d.h) + pad_y, y1 + 1,  max(1, h_img - 1)))

    mask = np.zeros_like(gray)
    mask[y1:y2, x1:x2] = 255

    # Adaptive quality level: tiny targets have weaker gradients
    target_px = (x2 - x1) * (y2 - y1)
    quality_level = 0.004 if target_px < 5000 else 0.008

    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=_MAX_CORNERS,
        qualityLevel=quality_level,
        minDistance=3,    # tighter packing than old minDistance=4
        blockSize=5,
        mask=mask,
    )
    if pts is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    return pts.astype(np.float32)


# ------------------------------------------------------------------
# Forward-backward LK with consistency check
# ------------------------------------------------------------------

def lk_step_with_backcheck(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_pts: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Run LK forward + backward and return only consistent points.

    Returns (old_good, new_good, delta_median) or (None, None, None).
    The caller decides whether enough points survived.
    """
    # Forward
    next_pts, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None, **_LK_PARAMS
    )
    if next_pts is None or st_fwd is None:
        return None, None, None

    # Backward (forward-backward consistency)
    prev_pts_back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
        curr_gray, prev_gray, next_pts, None, **_LK_PARAMS
    )

    keep = st_fwd.reshape(-1).astype(bool)
    if prev_pts_back is not None and st_bwd is not None:
        back_err = np.abs(
            prev_pts_back.reshape(-1, 2) - prev_pts.reshape(-1, 2)
        ).max(axis=1)
        keep = keep & st_bwd.reshape(-1).astype(bool) & (back_err < _BACK_ERR_THRESHOLD)

    old_good = prev_pts.reshape(-1, 2)[keep]
    new_good = next_pts.reshape(-1, 2)[keep]

    if len(new_good) < _MIN_TRACKED_POINTS:
        return old_good, new_good, None   # caller sees len < threshold

    delta = np.median(new_good - old_good, axis=0)
    return old_good, new_good, delta


# ------------------------------------------------------------------
# Inter-frame LK background thread
# ------------------------------------------------------------------

class InterFrameLKThread:
    """Runs LK optical flow on every camera frame in a background thread.

    Thread-safety contract
    ----------------------
    All shared state is protected by ``self._lock``.  The YOLO thread
    (caller of reset_from_yolo / clear / get_latest) and the LK thread
    both acquire ``_lock`` around shared-state access.  ``feed_frame`` is
    intentionally lock-free — it only touches the thread-safe ``queue.Queue``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue(maxsize=4)
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # --- shared state (protected by _lock) ---
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts:  Optional[np.ndarray] = None   # (N, 1, 2) float32
        self._ref_det:   Optional[Detection]  = None   # current tracked position
        self._latest:    Optional[Tuple[float, float, float, float, float, int]] = None
        #                                         cx   cy   w    h    quality  n_pts

        # Pending requests from YOLO thread
        self._reset_req: Optional[Tuple[np.ndarray, Detection]] = None
        self._clear_req: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            name="skyscouter-interframe-lk",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # API — camera reader thread (called on EVERY decoded frame, ~30 fps)
    # ------------------------------------------------------------------

    def feed_frame(self, frame_bgr: np.ndarray) -> None:
        """Non-blocking.  Converts to gray; drops oldest frame if queue is full."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        try:
            self._queue.put_nowait(gray)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(gray)
            except queue.Full:
                pass   # still full — just skip this frame

    # ------------------------------------------------------------------
    # API — YOLO thread (called at YOLO rate, ~5-7 fps)
    # ------------------------------------------------------------------

    def reset_from_yolo(self, gray: np.ndarray, detection: Detection) -> None:
        """Resync LK reference point to a confirmed YOLO detection."""
        with self._lock:
            self._reset_req = (gray.copy(), detection)
            self._clear_req = False

    def clear(self) -> None:
        """Stop tracking (track lost/cleared)."""
        with self._lock:
            self._clear_req  = True
            self._reset_req  = None
            self._latest     = None

    def get_latest(self) -> Optional[Tuple[float, float, float, float, float, int]]:
        """Return ``(cx, cy, w, h, quality, n_points)`` or ``None``.

        None means no active track or LK has not yet produced a result.
        Quality is in [0, 1]; n_points is the number of tracked corners.
        """
        with self._lock:
            return self._latest

    # ------------------------------------------------------------------
    # Background thread body
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while self._running:
            try:
                gray = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            with self._lock:

                # ---- handle YOLO-thread requests ----
                if self._clear_req:
                    self._prev_gray = None
                    self._prev_pts  = None
                    self._ref_det   = None
                    self._clear_req = False
                    continue

                if self._reset_req is not None:
                    init_gray, init_det = self._reset_req
                    self._reset_req = None
                    self._ref_det   = init_det
                    self._prev_pts  = find_features_in_region(init_gray, init_det)
                    # Use the CURRENT frame as prev so next iteration tracks
                    # from now (not from the YOLO frame which may be stale)
                    self._prev_gray = gray.copy()
                    self._latest = (
                        init_det.cx, init_det.cy,
                        init_det.w,  init_det.h,
                        1.0, int(len(self._prev_pts)),
                    )
                    continue

                # ---- normal LK step ----
                if (
                    self._prev_gray is None
                    or self._prev_pts is None
                    or self._ref_det  is None
                    or len(self._prev_pts) < _MIN_TRACKED_POINTS
                ):
                    # No active track — just update prev_gray so we're ready
                    # to track immediately when reset_from_yolo() is called
                    self._prev_gray = gray.copy()
                    continue

                old_good, new_good, delta = lk_step_with_backcheck(
                    self._prev_gray, gray, self._prev_pts
                )

                if delta is None or new_good is None or len(new_good) < _MIN_TRACKED_POINTS:
                    # Too few points survived — try to refresh features from
                    # the last known reference position
                    if self._ref_det is not None:
                        refreshed = find_features_in_region(gray, self._ref_det)
                        if len(refreshed) >= _MIN_TRACKED_POINTS:
                            self._prev_pts  = refreshed
                            self._prev_gray = gray.copy()
                    continue

                # Compute quality score
                residual     = np.linalg.norm(
                    (new_good - old_good) - delta[None, :], axis=1
                )
                residual_med = float(np.median(residual)) if len(residual) else 99.0
                pt_score     = min(1.0, len(new_good) / 20.0)
                res_score    = max(0.0, 1.0 - residual_med / 8.0)
                quality      = float(np.clip(0.6 * pt_score + 0.4 * res_score, 0.0, 1.0))

                # Accumulate delta onto reference detection
                new_cx = float(self._ref_det.cx + delta[0])
                new_cy = float(self._ref_det.cy + delta[1])
                self._latest = (
                    new_cx, new_cy,
                    self._ref_det.w, self._ref_det.h,
                    quality, len(new_good),
                )

                # Advance reference detection for next iteration
                self._ref_det = Detection(
                    x=new_cx - self._ref_det.w / 2.0,
                    y=new_cy - self._ref_det.h / 2.0,
                    w=self._ref_det.w,
                    h=self._ref_det.h,
                    confidence=self._ref_det.confidence,
                    class_id=self._ref_det.class_id,
                    class_label=self._ref_det.class_label,
                    source="lk_inter_frame",
                )

                # Periodically refresh feature set to replace lost corners
                if len(new_good) < _REFRESH_BELOW:
                    refreshed = find_features_in_region(gray, self._ref_det)
                    self._prev_pts = (
                        refreshed if len(refreshed) >= _MIN_TRACKED_POINTS
                        else new_good.reshape(-1, 1, 2).astype(np.float32)
                    )
                else:
                    self._prev_pts = new_good.reshape(-1, 1, 2).astype(np.float32)

                self._prev_gray = gray.copy()
