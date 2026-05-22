'''
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install opencv-python numpy

.\.venv\Scripts\Activate.ps1

python thermal_detector.py --config thermal_detector_config.json


'''

import json
from types import SimpleNamespace

import cv2
import numpy as np
import argparse
import os
from dataclasses import dataclass


@dataclass
class Detection:
    bbox: tuple  # x, y, w, h
    center: tuple  # cx, cy
    area: float
    score: float


class KalmanTrack:
    def __init__(self, initial_center):
        self.kf = cv2.KalmanFilter(4, 2)

        # State: [x, y, vx, vy]
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement: [x, y]
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 3.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        x, y = initial_center
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)

        self.missed = 0
        self.hits = 1
        self.age = 1
        self.last_center = initial_center
        self.last_bbox = None

    def predict(self):
        pred = self.kf.predict()
        self.age += 1
        return int(pred[0]), int(pred[1])

    def update(self, detection):
        cx, cy = detection.center
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        corrected = self.kf.correct(measurement)

        self.last_center = (int(corrected[0]), int(corrected[1]))
        self.last_bbox = detection.bbox
        self.missed = 0
        self.hits += 1

    def mark_missed(self):
        self.missed += 1


def preprocess_frame(frame, clahe_clip=2.0, clahe_tile=8, blur_ksize=3):
    """
    Converts MP4 frame to grayscale thermal intensity image.
    Even if the video looks grayscale, OpenCV usually reads MP4 as BGR.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip,
        tileGridSize=(clahe_tile, clahe_tile)
    )
    gray_eq = clahe.apply(gray)

    if blur_ksize > 1:
        gray_eq = cv2.GaussianBlur(gray_eq, (blur_ksize, blur_ksize), 0)

    return gray, gray_eq


def top_hat_detector(gray_eq, kernel_size=15):
    """
    White top-hat enhances small bright targets and suppresses large background.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    tophat = cv2.morphologyEx(gray_eq, cv2.MORPH_TOPHAT, kernel)

    _, mask = cv2.threshold(
        tophat,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return tophat, mask


def cfar_local_contrast_detector(gray_eq, window_size=31, threshold_k=3.0):
    """
    Simple CFAR/local contrast detector.

    score = (pixel - local_mean) / local_std

    Bright pixels that are much stronger than their local neighborhood are kept.
    """
    gray_f = gray_eq.astype(np.float32)

    local_mean = cv2.blur(gray_f, (window_size, window_size))
    local_sq_mean = cv2.blur(gray_f * gray_f, (window_size, window_size))

    local_var = local_sq_mean - local_mean * local_mean
    local_var = np.maximum(local_var, 1e-6)
    local_std = np.sqrt(local_var)

    score = (gray_f - local_mean) / (local_std + 1e-6)

    mask = (score > threshold_k).astype(np.uint8) * 255

    # Normalize score image only for visualization/debugging
    score_vis = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX)
    score_vis = score_vis.astype(np.uint8)

    return score, score_vis, mask


def clean_mask(mask, open_size=3, dilate_size=3):
    """
    Removes tiny noise and slightly connects fragmented target pixels.
    """
    if open_size > 1:
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_size, open_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    if dilate_size > 1:
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilate_size, dilate_size)
        )
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)

    return mask


def extract_detections(mask, gray_eq,
                       min_area=3,
                       max_area=600,
                       min_aspect=0.25,
                       max_aspect=4.0,
                       min_mean_intensity=80):
    """
    Converts binary mask into candidate UAV detections.
    """
    detections = []

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if w <= 0 or h <= 0:
            continue

        aspect = w / float(h)

        if aspect < min_aspect or aspect > max_aspect:
            continue

        roi = gray_eq[y:y + h, x:x + w]
        mean_intensity = float(np.mean(roi))

        if mean_intensity < min_mean_intensity:
            continue

        cx = x + w // 2
        cy = y + h // 2

        # Simple score: brighter and larger candidates rank higher
        score = mean_intensity + 0.1 * area

        detections.append(
            Detection(
                bbox=(x, y, w, h),
                center=(cx, cy),
                area=area,
                score=score
            )
        )

    detections.sort(key=lambda d: d.score, reverse=True)
    return detections


def associate_detection_to_track(track, detections, max_distance=60):
    """
    Associates the nearest detection to the predicted Kalman position.
    """
    if track is None or len(detections) == 0:
        return None

    px, py = track.predict()

    best_det = None
    best_dist = float("inf")

    for det in detections:
        cx, cy = det.center
        dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)

        if dist < best_dist:
            best_dist = dist
            best_det = det

    if best_dist <= max_distance:
        return best_det

    return None


def draw_results(frame, detections, track, frame_idx):
    output = frame.copy()

    # Draw all candidates in thin boxes
    for det in detections:
        x, y, w, h = det.bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), (80, 80, 255), 1)
        cv2.circle(output, det.center, 2, (80, 80, 255), -1)

    # Draw confirmed/current track
    if track is not None and track.last_bbox is not None:
        x, y, w, h = track.last_bbox
        cx, cy = track.last_center

        if track.hits >= 3 and track.missed <= 5:
            label = "UAV candidate track"
        else:
            label = "tentative track"

        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(output, (cx, cy), 4, (0, 255, 0), -1)
        cv2.putText(
            output,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    cv2.putText(
        output,
        f"Frame: {frame_idx}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return output


def run_detector(args):
    os.makedirs(args.output_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30.0

    basename = os.path.splitext(os.path.basename(args.input))[0]
    output_video = os.path.join(args.output_dir, f"{basename}_thermal_detected.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {output_video}")

    print(f"Input: {args.input}")
    print(f"Resolution: {width}x{height}, FPS={fps}")
    print(f"Output: {output_video}")

    track = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray_raw, gray_eq = preprocess_frame(
            frame,
            clahe_clip=args.clahe_clip,
            clahe_tile=args.clahe_tile,
            blur_ksize=args.blur
        )

        tophat, mask_tophat = top_hat_detector(
            gray_eq,
            kernel_size=args.tophat_kernel
        )

        score, score_vis, mask_cfar = cfar_local_contrast_detector(
            gray_eq,
            window_size=args.cfar_window,
            threshold_k=args.cfar_k
        )

        if args.combine == "and":
            combined_mask = cv2.bitwise_and(mask_tophat, mask_cfar)
        else:
            combined_mask = cv2.bitwise_or(mask_tophat, mask_cfar)

        combined_mask = clean_mask(
            combined_mask,
            open_size=args.open_size,
            dilate_size=args.dilate_size
        )

        detections = extract_detections(
            combined_mask,
            gray_eq,
            min_area=args.min_area,
            max_area=args.max_area,
            min_aspect=args.min_aspect,
            max_aspect=args.max_aspect,
            min_mean_intensity=args.min_intensity
        )

        if track is None:
            if len(detections) > 0:
                track = KalmanTrack(detections[0].center)
                track.update(detections[0])
        else:
            matched = associate_detection_to_track(
                track,
                detections,
                max_distance=args.max_assoc_distance
            )

            if matched is not None:
                track.update(matched)
            else:
                track.mark_missed()

            if track.missed > args.max_missed:
                track = None

        annotated = draw_results(frame, detections, track, frame_idx)

        # Always record annotated output video
        writer.write(annotated)

        # Optionally show live
        if args.show:
            display_frame = annotated

            if args.display_scale != 1.0:
                display_frame = cv2.resize(
                    display_frame,
                    None,
                    fx=args.display_scale,
                    fy=args.display_scale,
                    interpolation=cv2.INTER_AREA
                )

            cv2.imshow("Thermal UAV Detection - Annotated", display_frame)

            if args.show_debug:
                cv2.imshow("Combined Mask", combined_mask)
                cv2.imshow("Tophat", tophat)
                cv2.imshow("CFAR Score", score_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Done. Processed {frame_idx} frames.")
    print(f"Saved annotated video to: {output_video}")

def load_config():
    parser = argparse.ArgumentParser(
        description="Classical non-CNN thermal UAV detector using a JSON config file."
    )

    parser.add_argument(
        "--config",
        default="thermal_config.json",
        help="Path to JSON config file"
    )

    cli_args = parser.parse_args()

    with open(cli_args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    required_keys = [
        "input",
        "output_dir",
        "show",
        "show_debug",
        "display_scale",
        "clahe_clip",
        "clahe_tile",
        "blur",
        "tophat_kernel",
        "cfar_window",
        "cfar_k",
        "combine",
        "open_size",
        "dilate_size",
        "min_area",
        "max_area",
        "min_aspect",
        "max_aspect",
        "min_intensity",
        "max_assoc_distance",
        "max_missed"
    ]

    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")

    if config["combine"] not in ["and", "or"]:
        raise ValueError("Config value 'combine' must be either 'and' or 'or'.")

    return SimpleNamespace(**config)


if __name__ == "__main__":
    args = load_config()
    run_detector(args)