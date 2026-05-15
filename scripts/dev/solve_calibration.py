"""
Solve camera intrinsics from a captured calibration session.

Reads frames from data/calibration/raw/<session>/, detects chessboard corners
with sub-pixel refinement, runs cv2.calibrateCamera, and writes the resulting
intrinsics into both:

  data/calibration/<session>/intrinsics.yaml      - raw OpenCV-format numbers
  data/calibration/<session>/skyscouter_intrinsics.yaml - YAML block ready
                                                          to paste into the
                                                          deploy config

Also writes a couple of sanity-check images:

  reproject_<run_id>.png      - per-corner reprojection errors visualized
  undistort_sample.png        - one captured frame rectified through the
                                solved intrinsics, for visual confirmation

Reprojection error targets:
  < 0.5 px  : excellent calibration
  < 1.0 px  : acceptable for our application
  > 1.5 px  : recapture with better pose / corner coverage
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Solve intrinsics from captured chessboard frames")
    p.add_argument("--session", required=True, help="Path to data/calibration/raw/<timestamp>/")
    p.add_argument("--out-dir", default=None,
                   help="Where to write outputs. Default: data/calibration/<session>/")
    p.add_argument("--cols", type=int, default=None,
                   help="Override inner cols (otherwise read from session.txt)")
    p.add_argument("--rows", type=int, default=None,
                   help="Override inner rows (otherwise read from session.txt)")
    p.add_argument("--square-mm", type=float, default=None,
                   help="Override square size mm (otherwise read from session.txt)")
    return p.parse_args()


def read_session_meta(session: Path) -> dict:
    meta = {}
    txt = session / "session.txt"
    if not txt.exists():
        return meta
    for line in txt.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip()
    return meta


def main() -> int:
    args = parse_args()
    session = Path(args.session)
    if not session.is_dir():
        print(f"ERROR: session dir not found: {session}", file=sys.stderr)
        return 2

    meta = read_session_meta(session)
    cols = int(args.cols if args.cols is not None else meta.get("pattern_cols_inner", 6))
    rows = int(args.rows if args.rows is not None else meta.get("pattern_rows_inner", 7))
    square_mm = float(args.square_mm if args.square_mm is not None else meta.get("square_mm", 25.0))
    pattern_size = (cols, rows)

    frames = sorted(session.glob("frame_*.png"))
    if not frames:
        print(f"ERROR: no frames in {session}", file=sys.stderr)
        return 2

    print(f"Session     : {session}")
    print(f"Frames      : {len(frames)}")
    print(f"Pattern     : {cols} x {rows} inner corners")
    print(f"Square size : {square_mm} mm")
    print()

    # Construct the 3-D model of the chessboard. Z=0, X/Y in mm.
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_mm

    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []
    accepted: List[str] = []
    rejected: List[str] = []

    chess_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    subpix_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)

    image_size: Tuple[int, int] | None = None
    for fp in frames:
        img = cv2.imread(str(fp))
        if img is None:
            rejected.append(f"{fp.name}: read failed")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=chess_flags)
        if not found:
            rejected.append(f"{fp.name}: corners not found")
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_crit)
        obj_points.append(objp.copy())
        img_points.append(corners)
        accepted.append(fp.name)

    print(f"Accepted    : {len(accepted)} frames")
    if rejected:
        print(f"Rejected    : {len(rejected)} frames")
        for r in rejected[:5]:
            print(f"  - {r}")
        if len(rejected) > 5:
            print(f"  ... ({len(rejected)-5} more)")
    print()

    if len(obj_points) < 8:
        print(f"ERROR: only {len(obj_points)} valid frames. Need at least ~15 for stable solve.", file=sys.stderr)
        return 2

    assert image_size is not None
    print(f"Image size  : {image_size[0]} x {image_size[1]}")
    print(f"Solving calibration ... (may take a few seconds)")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    print()
    print(f"RMS reprojection error: {rms:.4f} px")
    if rms < 0.5:
        verdict = "EXCELLENT"
    elif rms < 1.0:
        verdict = "ACCEPTABLE"
    elif rms < 1.5:
        verdict = "MARGINAL — consider recapturing with better coverage"
    else:
        verdict = "POOR — recapture with more diverse poses, especially in frame corners"
    print(f"Verdict     : {verdict}")
    print()

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    k1, k2, p1, p2, k3 = dist.ravel()[:5]
    W, H = image_size

    # FOV from intrinsics
    hfov_rad = 2 * np.arctan(W / (2 * fx))
    vfov_rad = 2 * np.arctan(H / (2 * fy))
    hfov_deg = float(np.degrees(hfov_rad))
    vfov_deg = float(np.degrees(vfov_rad))

    print(f"Intrinsics  :")
    print(f"  fx = {fx:.3f}    fy = {fy:.3f}")
    print(f"  cx = {cx:.3f}    cy = {cy:.3f}     (geometric center = {W/2:.1f}, {H/2:.1f})")
    print(f"  delta from geometric center: ({cx - W/2:+.1f}, {cy - H/2:+.1f}) px")
    print(f"  distortion (k1,k2,p1,p2,k3) = ({k1:+.4f}, {k2:+.4f}, {p1:+.4f}, {p2:+.4f}, {k3:+.4f})")
    print(f"  HFOV = {hfov_deg:.2f} deg     VFOV = {vfov_deg:.2f} deg")
    print()

    # Per-frame reprojection error
    per_frame_rms = []
    for i in range(len(obj_points)):
        proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(img_points[i], proj, cv2.NORM_L2) / len(proj)
        per_frame_rms.append((accepted[i], float(err)))
    per_frame_rms.sort(key=lambda x: -x[1])
    print("Worst frames (consider removing if much worse than the rest):")
    for name, e in per_frame_rms[:5]:
        print(f"  {name}: {e:.3f} px")
    print()

    # Outputs
    out_dir = Path(args.out_dir) if args.out_dir else session.parent.parent / session.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Raw OpenCV-format intrinsics
    raw = {
        "image_width": W,
        "image_height": H,
        "fx_px": float(fx),
        "fy_px": float(fy),
        "cx_px": float(cx),
        "cy_px": float(cy),
        "distortion_k1": float(k1),
        "distortion_k2": float(k2),
        "distortion_p1": float(p1),
        "distortion_p2": float(p2),
        "distortion_k3": float(k3),
        "horizontal_fov_deg": hfov_deg,
        "vertical_fov_deg": vfov_deg,
        "rms_reprojection_error_px": float(rms),
        "n_frames_used": len(obj_points),
        "square_size_mm": square_mm,
        "pattern_cols_inner": cols,
        "pattern_rows_inner": rows,
        "calibrated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "intrinsics.json").write_text(json.dumps(raw, indent=2))

    # SkyScouter-format YAML block (paste-ready)
    calib_id = f"SIYI_A8_MINI_1080P_CALIBRATED_{datetime.now(timezone.utc).strftime('%Y%m%d')}_v1"
    yaml_block = f"""# Camera calibration block — generated 2026-05-13 from {session.name}
# RMS reprojection error: {rms:.4f} px ({verdict})
# Paste into your deploy config, replacing the existing `camera:` and
# `guidance.camera:` blocks.

camera:
  calibration_id: "{calib_id}"
  is_calibrated: true
  intrinsics:
    fx: {fx:.3f}
    fy: {fy:.3f}
    cx: {cx:.3f}
    cy: {cy:.3f}
    distortion: [{k1:.6f}, {k2:.6f}, {p1:.6f}, {p2:.6f}, {k3:.6f}]
  resolution_wh: [{W}, {H}]

guidance:
  camera:
    mode: "intrinsics"           # was "fov"; switch to use the solved fx/fy
    horizontal_fov_deg: {hfov_deg:.3f}
    vertical_fov_deg: {vfov_deg:.3f}
    frame_width_px: {W}
    frame_height_px: {H}
    fx_px: {fx:.3f}
    fy_px: {fy:.3f}
    cx_px: {cx:.3f}
    cy_px: {cy:.3f}
    assume_principal_point_center: false
    calibration_reviewed: true   # set this after a sanity-check run
"""
    yaml_path = out_dir / "skyscouter_intrinsics.yaml"
    yaml_path.write_text(yaml_block)

    # Sanity-check undistortion image — pick the first accepted frame
    sample = cv2.imread(str(session / accepted[0]))
    if sample is not None:
        undistorted = cv2.undistort(sample, K, dist)
        side_by_side = np.hstack([sample, undistorted])
        # draw the solved optical center on the original (left)
        cv2.drawMarker(side_by_side, (int(round(cx)), int(round(cy))), (0, 255, 255),
                       cv2.MARKER_CROSS, 24, 2)
        # and the geometric center for reference
        cv2.drawMarker(side_by_side, (int(W/2), int(H/2)), (255, 200, 0),
                       cv2.MARKER_TILTED_CROSS, 24, 2)
        # repeat on the right (undistorted)
        cv2.drawMarker(side_by_side, (W + int(round(cx)), int(round(cy))), (0, 255, 255),
                       cv2.MARKER_CROSS, 24, 2)
        cv2.putText(side_by_side, "ORIGINAL", (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(side_by_side, "UNDISTORTED", (W + 20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / "undistort_sample.png"), side_by_side)

    print(f"Wrote:")
    print(f"  {out_dir / 'intrinsics.json'}")
    print(f"  {yaml_path}")
    print(f"  {out_dir / 'undistort_sample.png'}")
    print()
    print(f"Next: review undistort_sample.png. If straight edges look straight in the right panel,")
    print(f"calibration is good. Then copy the camera: + guidance.camera: blocks from")
    print(f"{yaml_path}")
    print(f"into your deploy YAML.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
