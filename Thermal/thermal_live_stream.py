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

def make_four_view(original_frame, annotated_frame, debug_images):
    """
    Create a 2x2 visualization:

        top-left:     original video
        top-right:    blurred local background
        bottom-left:  subtracted local-contrast image
        bottom-right: annotated detection video

    This is mainly for tuning detector parameters.
    """

    h, w = original_frame.shape[:2]

    # Original
    original = original_frame.copy()

    # Blurred background
    blurred = debug_images.get("blurred")

    if blurred is None:
        blurred_bgr = np.zeros_like(original)
    else:
        blurred_bgr = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

    # Subtracted/local contrast image
    subtracted = debug_images.get("subtracted")

    if subtracted is None:
        subtracted_bgr = np.zeros_like(original)
    else:
        subtracted_bgr = cv2.cvtColor(subtracted, cv2.COLOR_GRAY2BGR)

    # Annotated
    annotated = annotated_frame.copy()

    # Make sure all are same size
    blurred_bgr = cv2.resize(blurred_bgr, (w, h))
    subtracted_bgr = cv2.resize(subtracted_bgr, (w, h))
    annotated = cv2.resize(annotated, (w, h))

    # Labels
    cv2.putText(original, "Original", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(blurred_bgr, "Blurred local background", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(subtracted_bgr, "Subtracted local contrast", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(annotated, "Annotated detection", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    top = np.hstack([original, blurred_bgr])
    bottom = np.hstack([subtracted_bgr, annotated])

    four_view = np.vstack([top, bottom])

    return four_view

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
                    display_frame = make_four_view(
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
    print(f"Saved output video to: {output_video}")


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
        "input",
        "output_dir",
        "show",
        "display_scale",
        "debug_four_view",
        "algorithm",
    ]

    missing = [key for key in required_keys if key not in config]

    if missing:
        raise ValueError(f"Missing config keys: {missing}")

    return SimpleNamespace(**config)


if __name__ == "__main__":
    args = load_config()
    run_live_stream(args)