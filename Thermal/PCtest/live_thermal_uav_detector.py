"""
live_thermal_uav_detector.py

Live USB thermal camera detection.

Features:
    - Reads thermal USB camera stream
    - Splits/preprocesses the two-view thermal image
    - Runs thermal_detector.detect_frame()
    - Shows live OpenCV window with UAV boxes
    - Records annotated detection video
    - Safely saves video when stopped with q, ESC, or Ctrl+C

Run:
    python live_thermal_uav_detector.py --camera 0 --config thermal_detector_config.json --debug

Try other camera indices:
    python live_thermal_uav_detector.py --camera 1 --config thermal_detector_config.json --debug
    python live_thermal_uav_detector.py --camera 2 --config thermal_detector_config.json --debug
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from Thermal_detector import detect_frame, get_last_debug_images
from thermal_preprocess import preprocess_frame


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def get_config_value(config, key, default):
    if config is None:
        return default

    if isinstance(config, dict):
        return config.get(key, default)

    return getattr(config, key, default)


def draw_boxes(frame, boxes):
    output = frame.copy()

    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    for box in boxes:
        x, y, w, h = box.bbox

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

        text = f"{box.label}: {box.score:.1f}"

        cv2.putText(
            output,
            text,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return output

def to_bgr_or_blank(img, reference):
    """
    Convert debug image to BGR.
    If image is missing, create black image with same size as reference.
    """
    if img is None:
        return np.zeros_like(reference)

    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img.copy()


def put_title(img, title):
    output = img.copy()

    cv2.putText(
        output,
        title,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


def make_debug_view(original_frame, annotated_frame, debug_images, mode="fast6"):
    """
    Create debug recording frame.

    mode="full8":
        1 Original
        2 Blurred background
        3 Subtracted/local contrast
        4 Threshold mask
        5 Morphology mask
        6 Raw contour boxes
        7 Filtered candidates
        8 Final annotated

    mode="fast6":
        1 Original
        2 Blurred background
        3 Subtracted/local contrast
        4 Threshold mask
        5 Filtered candidates
        6 Final annotated

    fast6 skips morphology and raw contour views to reduce output size/work.
    """

    h, w = original_frame.shape[:2]

    reference = original_frame.copy()

    if len(reference.shape) == 2:
        reference = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)

    annotated = annotated_frame.copy()

    if len(annotated.shape) == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

    original = reference

    blurred = to_bgr_or_blank(debug_images.get("blurred"), reference)
    subtracted = to_bgr_or_blank(debug_images.get("subtracted"), reference)
    threshold = to_bgr_or_blank(debug_images.get("threshold_mask"), reference)
    morphology = to_bgr_or_blank(debug_images.get("morphology_mask"), reference)
    raw_contours = to_bgr_or_blank(debug_images.get("raw_contours"), reference)
    filtered = to_bgr_or_blank(debug_images.get("filtered_candidates"), reference)

    # Force every panel to same size as final processed frame
    panels = {
        "original": cv2.resize(original, (w, h)),
        "blurred": cv2.resize(blurred, (w, h)),
        "subtracted": cv2.resize(subtracted, (w, h)),
        "threshold": cv2.resize(threshold, (w, h)),
        "morphology": cv2.resize(morphology, (w, h)),
        "raw_contours": cv2.resize(raw_contours, (w, h)),
        "filtered": cv2.resize(filtered, (w, h)),
        "annotated": cv2.resize(annotated, (w, h)),
    }

    if mode == "full8":
        p1 = put_title(panels["original"], "1 Original")
        p2 = put_title(panels["blurred"], "2 Blurred background")
        p3 = put_title(panels["subtracted"], "3 Subtracted/local contrast")
        p4 = put_title(panels["threshold"], "4 Threshold mask")
        p5 = put_title(panels["morphology"], "5 Morphology mask")
        p6 = put_title(panels["raw_contours"], "6 Raw contour boxes")
        p7 = put_title(panels["filtered"], "7 Filtered candidates")
        p8 = put_title(panels["annotated"], "8 Final annotated")

        row1 = np.hstack([p1, p2, p3, p4])
        row2 = np.hstack([p5, p6, p7, p8])

        return np.vstack([row1, row2])

    if mode == "fast6":
        p1 = put_title(panels["original"], "1 Original")
        p2 = put_title(panels["blurred"], "2 Blurred background")
        p3 = put_title(panels["subtracted"], "3 Subtracted/local contrast")
        p4 = put_title(panels["threshold"], "4 Threshold mask")
        p5 = put_title(panels["filtered"], "5 Filtered candidates")
        p6 = put_title(panels["annotated"], "6 Final annotated")

        row1 = np.hstack([p1, p2, p3])
        row2 = np.hstack([p4, p5, p6])

        return np.vstack([row1, row2])

    raise ValueError("debug_record_mode must be either 'fast6' or 'full8'")

def resize_to_same_height(img, target_h):
    h, w = img.shape[:2]

    if h == target_h:
        return img

    scale = target_h / float(h)
    new_w = int(w * scale)

    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def make_live_debug_view(raw_frame, processed_frame, annotated_frame):
    raw_display = raw_frame.copy()
    processed_display = processed_frame.copy()
    annotated_display = annotated_frame.copy()

    if len(raw_display.shape) == 2:
        raw_display = cv2.cvtColor(raw_display, cv2.COLOR_GRAY2BGR)

    if len(processed_display.shape) == 2:
        processed_display = cv2.cvtColor(processed_display, cv2.COLOR_GRAY2BGR)

    if len(annotated_display.shape) == 2:
        annotated_display = cv2.cvtColor(annotated_display, cv2.COLOR_GRAY2BGR)

    target_h = 360

    raw_display = resize_to_same_height(raw_display, target_h)
    processed_display = resize_to_same_height(processed_display, target_h)
    annotated_display = resize_to_same_height(annotated_display, target_h)

    cv2.putText(
        raw_display,
        "Raw camera frame",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        processed_display,
        "Selected thermal view",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        annotated_display,
        "Detection result",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return np.hstack([raw_display, processed_display, annotated_display])


def open_camera(camera_index, width=None, height=None, fps=None):
    print(f"[INFO] Trying camera index {camera_index} with CAP_DSHOW...")

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("[WARNING] CAP_DSHOW failed. Trying default backend...")
        cap.release()
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))

    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, int(fps))

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print("[INFO] Camera opened.")
    print(f"[INFO] Resolution: {actual_width:.0f} x {actual_height:.0f}")
    print(f"[INFO] FPS reported by camera: {actual_fps:.2f}")

    return cap


def create_video_writer(output_dir, frame_shape, fps):
    """
    Create VideoWriter after we know the actual processed frame size.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"live_detection_{timestamp}.mp4"

    h, w = frame_shape[:2]

    if fps is None or fps <= 1:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (w, h),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {output_path}")

    print(f"[INFO] Recording annotated video to: {output_path}")

    return writer, output_path
def create_named_video_writer(output_dir, filename_prefix, frame_shape, fps):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{filename_prefix}_{timestamp}.mp4"

    h, w = frame_shape[:2]

    if fps is None or fps <= 1:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (w, h),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {output_path}")

    print(f"[INFO] Recording video to: {output_path}")

    return writer, output_path

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="thermal_detector_config.json",
        help="Path to JSON config file.",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera index. Overrides config camera.index.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show raw frame, selected frame, and detection result.",
    )

    parser.add_argument(
        "--display-scale",
        type=float,
        default=1.0,
        help="Scale OpenCV display window.",
    )

    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable video recording.",
    )

    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    full_config = load_config(config_path)

    camera_config = full_config.get("camera", {})
    preprocess_config = full_config.get("preprocess", {})
    algorithm_config = full_config.get("algorithm", full_config)
    recording_config = full_config.get("recording", {})

    camera_index = args.camera

    if camera_index is None:
        camera_index = get_config_value(camera_config, "index", 0)

    width = get_config_value(camera_config, "width", None)
    height = get_config_value(camera_config, "height", None)
    fps_setting = get_config_value(camera_config, "fps", None)

    output_dir = get_config_value(recording_config, "output_dir", "outputs")
    record_enabled = get_config_value(recording_config, "enabled", True)

    record_final = get_config_value(recording_config, "record_final", True)
    record_debug = get_config_value(recording_config, "record_debug", False)
    debug_record_mode = get_config_value(recording_config, "debug_record_mode", "fast6")

    if args.no_record:
        record_enabled = False
        record_final = False
        record_debug = False

    cap = None
    final_writer = None
    final_output_path = None

    debug_writer = None
    debug_output_path = None

    try:
        cap = open_camera(
            camera_index=camera_index,
            width=width,
            height=height,
            fps=fps_setting,
        )

        camera_fps = cap.get(cv2.CAP_PROP_FPS)

        if camera_fps is None or camera_fps <= 1:
            camera_fps = get_config_value(recording_config, "fps", 30.0)

        print("[INFO] Press q or ESC to stop.")
        print("[INFO] Press d to toggle debug view.")
        print("[INFO] Ctrl+C also stops and saves the recording.")

        prev_time = time.time()
        debug_enabled = args.debug

        while True:
            ret, raw_frame = cap.read()

            if not ret or raw_frame is None:
                print("[WARNING] Failed to read frame.")
                continue

            processed_frame = preprocess_frame(
                raw_frame,
                preprocess_config,
            )

            boxes = detect_frame(
                processed_frame,
                algorithm_config,
            )

            annotated = draw_boxes(
                processed_frame,
                boxes,
            )

            now = time.time()
            dt = now - prev_time
            prev_time = now

            live_fps = 1.0 / dt if dt > 1e-6 else 0.0

            cv2.putText(
                annotated,
                f"FPS: {live_fps:.1f} | Boxes: {len(boxes)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )



            # ------------------------------------------------------------
# Record final annotated one-view video
# ------------------------------------------------------------
            if record_enabled and record_final and final_writer is None:
                final_writer, final_output_path = create_named_video_writer(
                    output_dir=output_dir,
                    filename_prefix="final_annotated",
                    frame_shape=annotated.shape,
                    fps=camera_fps,
                )

            if final_writer is not None:
                final_writer.write(annotated)

            # ------------------------------------------------------------
            # Record debug multi-view video
            # ------------------------------------------------------------
            if record_enabled and record_debug:
                debug_images = get_last_debug_images()

                debug_record_frame = make_debug_view(
                    original_frame=processed_frame,
                    annotated_frame=annotated,
                    debug_images=debug_images,
                    mode=debug_record_mode,
                )

                if debug_writer is None:
                    debug_writer, debug_output_path = create_named_video_writer(
                        output_dir=output_dir,
                        filename_prefix=f"debug_{debug_record_mode}",
                        frame_shape=debug_record_frame.shape,
                        fps=camera_fps,
                    )

                debug_writer.write(debug_record_frame)



            if final_writer is not None:
                final_writer.release()
                print(f"[INFO] Final annotated recording saved to: {final_output_path}")

            if debug_writer is not None:
                debug_writer.release()
                print(f"[INFO] Debug recording saved to: {debug_output_path}")

            if debug_enabled:
                display = make_live_debug_view(
                    raw_frame,
                    processed_frame,
                    annotated,
                )
            else:
                display = annotated

            if args.display_scale != 1.0:
                display = cv2.resize(
                    display,
                    None,
                    fx=args.display_scale,
                    fy=args.display_scale,
                    interpolation=cv2.INTER_AREA,
                )

            cv2.imshow("Live Thermal UAV Detection", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                print("[INFO] Stop requested by user.")
                break

            elif key == ord("d"):
                debug_enabled = not debug_enabled
                print(f"[INFO] Debug view: {debug_enabled}")

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received. Stopping safely...")

    finally:
        if cap is not None:
            cap.release()

        if writer is not None:
            writer.release()
            print(f"[INFO] Recording saved to: {output_path}")

        cv2.destroyAllWindows()
        print("[INFO] Camera, video writer, and windows closed.")


if __name__ == "__main__":
    main()