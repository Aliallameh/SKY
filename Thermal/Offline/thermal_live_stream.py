"""
thermal_live_stream.py

This file handles only:
    - video reading
    - real-time playback
    - multithreading
    - live display
    - recording output video

It does NOT implement detection algorithms.

Detection is delegated to:

    thermal_detector_algorithms.detect_frame(frame, algorithm_config)

This keeps the system modular.
"""

import argparse
import json
import os
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import numpy as np
from Thermal_detector import detect_frame, get_last_debug_images

@dataclass
class DetectionResult:
    """
    Detection result produced by the detector thread.

    boxes:
        List of Box objects returned by the algorithm script.

    source_frame_idx:
        Frame index used by the detector.

    processing_ms:
        How long the algorithm took on that frame.
    """
    boxes: list
    source_frame_idx: int
    processing_ms: float


def draw_boxes(frame, result, current_frame_idx):
    """
    Draw latest available detection boxes on the current live frame.

    Important:
        The result may come from an older frame than the current displayed frame.

    This is intentional for live-stream behavior:
        - video remains real-time
        - detector runs in background
        - latest completed result is shown on current frame
    """
    output = frame.copy()

    if result is None:
        cv2.putText(
            output,
            f"Live frame: {current_frame_idx} | Waiting for detector...",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    for box in result.boxes:
        x, y, w, h = box.bbox

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (80, 80, 255),
            1,
        )

        cv2.putText(
            output,
            f"{box.label}: {box.score:.1f}",
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (80, 80, 255),
            1,
            cv2.LINE_AA,
        )

    lag_frames = current_frame_idx - result.source_frame_idx

    cv2.putText(
        output,
        (
            f"Live frame: {current_frame_idx} | "
            f"Detection frame: {result.source_frame_idx} | "
            f"Lag: {lag_frames} frames | "
            f"Proc: {result.processing_ms:.1f} ms"
        ),
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output

def make_debug_view(original_frame, annotated_frame, debug_images):
    """
    Create a 4x2 debug visualization:

        row 1: original                  blurred background
        row 2: subtracted/local contrast threshold mask
        row 3: morphology mask           raw contour boxes
        row 4: filtered candidates       final annotated
    """

    h, w = original_frame.shape[:2]

    original = original_frame.copy()
    annotated = annotated_frame.copy()

    blurred = debug_images.get("blurred")
    subtracted = debug_images.get("subtracted")
    threshold_mask = debug_images.get("threshold_mask")
    morphology_mask = debug_images.get("morphology_mask")
    raw_contours = debug_images.get("raw_contours")
    filtered_candidates = debug_images.get("filtered_candidates")

    def to_bgr_or_blank(img):
        if img is None:
            return np.zeros_like(original)

        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img.copy()

    blurred_bgr = to_bgr_or_blank(blurred)
    subtracted_bgr = to_bgr_or_blank(subtracted)
    threshold_bgr = to_bgr_or_blank(threshold_mask)
    morphology_bgr = to_bgr_or_blank(morphology_mask)
    raw_contours_bgr = to_bgr_or_blank(raw_contours)
    filtered_candidates_bgr = to_bgr_or_blank(filtered_candidates)

    blurred_bgr = cv2.resize(blurred_bgr, (w, h))
    subtracted_bgr = cv2.resize(subtracted_bgr, (w, h))
    threshold_bgr = cv2.resize(threshold_bgr, (w, h))
    morphology_bgr = cv2.resize(morphology_bgr, (w, h))
    raw_contours_bgr = cv2.resize(raw_contours_bgr, (w, h))
    filtered_candidates_bgr = cv2.resize(filtered_candidates_bgr, (w, h))
    annotated = cv2.resize(annotated, (w, h))

    cv2.putText(original, "1 Original", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(blurred_bgr, "2 Blurred background", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(subtracted_bgr, "3 Subtracted/local contrast", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(threshold_bgr, "4 Threshold mask", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(morphology_bgr, "5 Morphology mask", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(raw_contours_bgr, "6 Raw contour boxes", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(filtered_candidates_bgr, "7 Filtered candidates", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(annotated, "8 Final annotated", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 2 rows x 4 columns layout
    row1 = np.hstack([
        original,
        blurred_bgr,
        subtracted_bgr,
        threshold_bgr,
    ])

    row2 = np.hstack([
        morphology_bgr,
        raw_contours_bgr,
        filtered_candidates_bgr,
        annotated,
    ])

    debug_view = np.vstack([row1, row2])

    return debug_view

def run_live_stream(args):
    """
    Main live-stream loop.

    Main thread:
        - reads frames
        - keeps original FPS timing
        - displays current frame
        - records output video

    Detector thread:
        - receives newest available frame
        - calls detect_frame()
        - publishes latest detection result

    Stale frame policy:
        If the detector is slower than the video FPS, old unprocessed frames
        are overwritten. This prevents growing delay.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30.0

    frame_period = 1.0 / fps

    basename = os.path.splitext(os.path.basename(args.input))[0]
    output_video = os.path.join(args.output_dir, f"{basename}_live_detected.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {output_video}")

    print(f"Input: {args.input}")
    print(f"Resolution: {width}x{height}, FPS={fps}")
    print(f"Output: {output_video}")
    print("Mode: modular live stream")
    print("Live script: video/thread/display/recording")
    print("Algorithm script: frame in -> boxes out")

    shared = {
        "latest_frame": None,
        "latest_frame_idx": -1,
        "latest_result": None,
    }

    lock = threading.Lock()
    stop_event = threading.Event()

    def detector_worker():
        """
        Background detector thread.

        It always processes the newest available frame.
        """
        last_processed_idx = -1

        print("[Detector] Thread started.")

        while not stop_event.is_set():
            with lock:
                frame = shared["latest_frame"]
                frame_idx = shared["latest_frame_idx"]

                if frame is None or frame_idx == last_processed_idx:
                    frame_to_process = None
                    idx_to_process = None
                else:
                    frame_to_process = frame.copy()
                    idx_to_process = frame_idx

            if frame_to_process is None:
                time.sleep(0.001)
                continue

            try:
                start_time = time.perf_counter()

                boxes = detect_frame(
                    frame_to_process,
                    args.algorithm,
                )

                processing_ms = (time.perf_counter() - start_time) * 1000.0

                result = DetectionResult(
                    boxes=boxes,
                    source_frame_idx=idx_to_process,
                    processing_ms=processing_ms,
                )

                with lock:
                    shared["latest_result"] = result

                last_processed_idx = idx_to_process

            except Exception as e:
                print("[Detector] ERROR:", repr(e))
                stop_event.set()
                break

        print("[Detector] Thread stopped.")

    worker = threading.Thread(target=detector_worker, daemon=True)
    worker.start()

    frame_idx = 0
    playback_start_time = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            with lock:
                shared["latest_frame"] = frame.copy()
                shared["latest_frame_idx"] = frame_idx
                result = shared["latest_result"]

            annotated = draw_boxes(frame, result, frame_idx)

            writer.write(annotated)

            if args.show:
                debug_images = get_last_debug_images()

                if getattr(args, "debug_four_view", False):
                    display_frame = make_debug_view(
                        original_frame=frame,
                        annotated_frame=annotated,
                        debug_images=debug_images,
                    )
                else:
                    display_frame = annotated

                if args.display_scale != 1.0:
                    display_frame = cv2.resize(
                        display_frame,
                        None,
                        fx=args.display_scale,
                        fy=args.display_scale,
                        interpolation=cv2.INTER_AREA,
                    )

                cv2.imshow("Thermal Live Stream", display_frame)

            target_time = playback_start_time + frame_idx * frame_period
            now = time.perf_counter()
            wait_time = target_time - now

            if wait_time > 0:
                wait_ms = max(1, int(wait_time * 1000))
            else:
                wait_ms = 1

            key = cv2.waitKey(wait_ms) & 0xFF

            if key == 27 or key == ord("q"):
                break

            frame_idx += 1

    finally:
        stop_event.set()
        worker.join(timeout=1.0)

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Done. Played {frame_idx} frames.")
    print(f"  Debug view: {output_video}")


def run_offline_save(args):
    """
    Offline quick-processing mode.

    This mode quickly processes the input video and saves ONE combined 2x2 video:

        top-left:     original video
        top-right:    blurred local background
        bottom-left:  subtracted local contrast
        bottom-right: annotated detection

    It can also start from a specific time in the video using:

        "start_time_sec": 12.5

    Optional:

        "end_time_sec": 30.0

    If end_time_sec is null, missing, or <= start_time_sec, it processes until the end.
    """

    os.makedirs(args.output_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    # ------------------------------------------------------------------
    # Start/end time support
    # ------------------------------------------------------------------
    start_time_sec = float(getattr(args, "start_time_sec", 0.0) or 0.0)
    end_time_sec = getattr(args, "end_time_sec", None)

    if end_time_sec is not None:
        end_time_sec = float(end_time_sec)

    start_frame = max(0, int(round(start_time_sec * fps)))

    if total_frames > 0:
        start_frame = min(start_frame, total_frames - 1)

    if end_time_sec is not None and end_time_sec > start_time_sec:
        end_frame = int(round(end_time_sec * fps))

        if total_frames > 0:
            end_frame = min(end_frame, total_frames)
    else:
        end_frame = total_frames if total_frames > 0 else None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # ------------------------------------------------------------------
    # Output is one 2x2 video, same as live debug view
    # ------------------------------------------------------------------
    basename = os.path.splitext(os.path.basename(args.input))[0]

    debug_output_video = os.path.join(
        args.output_dir,
        f"{basename}_offline_debug_view.mp4"
    )

    final_output_video = os.path.join(
        args.output_dir,
        f"{basename}_offline_final_annotated.mp4"
    )

    # If mp4v fails on your Windows/OpenCV setup, switch this to MJPG + .avi.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Debug video: 2 rows x 4 columns
    output_width = width * 4
    output_height = height * 2

    debug_writer = cv2.VideoWriter(
        debug_output_video,
        fourcc,
        fps,
        (output_width, output_height),
    )

    # Final annotated video: original full resolution
    final_writer = cv2.VideoWriter(
        final_output_video,
        fourcc,
        fps,
        (width, height),
    )

    if not debug_writer.isOpened():
        raise RuntimeError(f"Could not open debug output video writer: {debug_output_video}")

    if not final_writer.isOpened():
        raise RuntimeError(f"Could not open final output video writer: {final_output_video}")

    print(f"Input: {args.input}")
    print(f"Resolution: {width}x{height}, FPS={fps}")
    print(f"Total frames: {total_frames}")
    print(f"Start time: {start_time_sec:.2f} s")
    print(f"Start frame: {start_frame}")

    if end_frame is not None:
        print(f"End frame: {end_frame}")
    else:
        print("End frame: end of video")

    print("Mode: offline quick save")
    print("Saving debug video:")
    print(f"  {debug_output_video}")
    print("Saving full-scale final annotated video:")
    print(f"  {final_output_video}")

    frame_idx = start_frame
    processed_count = 0
    start_all = time.perf_counter()

    try:
        while True:
            if end_frame is not None and frame_idx >= end_frame:
                break

            ret, frame = cap.read()

            if not ret:
                break

            start_time = time.perf_counter()

            # ----------------------------------------------------------
            # Run detector on current frame
            # ----------------------------------------------------------
            boxes = detect_frame(
                frame,
                args.algorithm,
            )

            processing_ms = (time.perf_counter() - start_time) * 1000.0

            result = DetectionResult(
                boxes=boxes,
                source_frame_idx=frame_idx,
                processing_ms=processing_ms,
            )

            # ----------------------------------------------------------
            # Create annotated frame
            # ----------------------------------------------------------
            annotated = draw_boxes(frame, result, frame_idx)

            # ----------------------------------------------------------
            # Create the same 2x2 view as live display
            # ----------------------------------------------------------
            debug_images = get_last_debug_images()

            debug_view = make_debug_view(
                original_frame=frame,
                annotated_frame=annotated,
                debug_images=debug_images,
            )

            # Make extra sure output size matches writer size
            if debug_view.shape[1] != output_width or debug_view.shape[0] != output_height:
                debug_view = cv2.resize(
                    debug_view,
                    (output_width, output_height),
                    interpolation=cv2.INTER_AREA,
                )

            debug_writer.write(debug_view)

            # Save view 8 separately at original/full resolution
            final_writer.write(annotated)

            # ----------------------------------------------------------
            # Optional preview while rendering
            # ----------------------------------------------------------
            if args.show:
                display_frame = debug_view

                if args.display_scale != 1.0:
                    display_frame = cv2.resize(
                        display_frame,
                        None,
                        fx=args.display_scale,
                        fy=args.display_scale,
                        interpolation=cv2.INTER_AREA,
                    )

                cv2.imshow("Thermal Offline Four View", display_frame)

                key = cv2.waitKey(1) & 0xFF

                if key == 27 or key == ord("q"):
                    break

            if processed_count % 50 == 0:
                if total_frames > 0:
                    progress = 100.0 * frame_idx / total_frames
                    print(
                        f"Processed source frame {frame_idx}/{total_frames} "
                        f"({progress:.1f}%) | "
                        f"Detector: {processing_ms:.1f} ms"
                    )
                else:
                    print(
                        f"Processed source frame {frame_idx} | "
                        f"Detector: {processing_ms:.1f} ms"
                    )

            frame_idx += 1
            processed_count += 1

    finally:
        cap.release()
        debug_writer.release()
        final_writer.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_all
    effective_fps = processed_count / elapsed if elapsed > 0 else 0.0

    print("Mode: offline quick save")
    print("Saving debug video:")
    print(f"  {debug_output_video}")
    print("Saving full-scale final annotated video:")
    print(f"  {final_output_video}")

def load_config():
    """
    Load JSON config.

    The live-stream script only needs:
        input
        output_dir
        show
        display_scale
        algorithm
    """
    parser = argparse.ArgumentParser(
        description="Modular thermal live stream runner."
    )

    parser.add_argument(
        "--config",
        default="thermal_detector_config.json",
        help="Path to JSON config file",
    )

    cli_args = parser.parse_args()

    with open(cli_args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    required_keys = [
        "mode",
        "input",
        "output_dir",
        "show",
        "display_scale",
        "debug_four_view",
        "save_four_view",
        "algorithm",
    ]

    missing = [key for key in required_keys if key not in config]

    if missing:
        raise ValueError(f"Missing config keys: {missing}")

    return SimpleNamespace(**config)


if __name__ == "__main__":
    args = load_config()

    mode = args.mode.lower().strip()

    if mode == "live":
        run_live_stream(args)

    elif mode == "offline":
        run_offline_save(args)

    else:
        raise ValueError(
            f"Unknown mode: {args.mode}. Use either 'live' or 'offline'."
        )