'''
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install opencv-python numpy

python thermal_detector.py `
  --input ./ThermalVideos/Video1_Raw_top.mp4 `
  --output-dir detector_results `
  --combine and `
  --cfar-window 31 `
  --cfar-k 3.0 `
  --tophat-kernel 15 `
  --clahe-clip 2.0 `
  --clahe-tile 8 `
  --blur 3 `
  --min-area 3 `
  --max-area 600 `
  --min-aspect 0.25 `
  --max-aspect 4.0 `
  --min-intensity 80 `
  --max-assoc-distance 60 `
  --max-missed 10


'''

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

        # Combine top-hat and CFAR masks.
        # AND is stricter, OR is more sensitive.
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
        writer.write(annotated)

        if args.show:
            cv2.imshow("Thermal UAV Detection", annotated)
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classical non-CNN thermal UAV detector using CLAHE, top-hat, CFAR/local contrast, contour filtering, and Kalman tracking."
    )

    parser.add_argument("--input", required=True, help="Input thermal MP4 video")
    parser.add_argument("--output-dir", default="thermal_detector_output", help="Output folder")
    parser.add_argument("--show", action="store_true", help="Show live debug windows")

    # Preprocessing
    parser.add_argument("--clahe-clip", type=float, default=2.0)
    parser.add_argument("--clahe-tile", type=int, default=8)
    parser.add_argument("--blur", type=int, default=3, help="Gaussian blur kernel size. Use odd number: 1, 3, 5")

    # Top-hat
    parser.add_argument("--tophat-kernel", type=int, default=15)

    # CFAR/local contrast
    parser.add_argument("--cfar-window", type=int, default=31)
    parser.add_argument("--cfar-k", type=float, default=3.0)

    # Mask combination
    parser.add_argument("--combine", choices=["and", "or"], default="and")

    # Morphology
    parser.add_argument("--open-size", type=int, default=3)
    parser.add_argument("--dilate-size", type=int, default=3)

    # Contour filtering
    parser.add_argument("--min-area", type=float, default=3)
    parser.add_argument("--max-area", type=float, default=600)
    parser.add_argument("--min-aspect", type=float, default=0.25)
    parser.add_argument("--max-aspect", type=float, default=4.0)
    parser.add_argument("--min-intensity", type=float, default=80)

    # Tracking
    parser.add_argument("--max-assoc-distance", type=float, default=60)
    parser.add_argument("--max-missed", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_detector(args)