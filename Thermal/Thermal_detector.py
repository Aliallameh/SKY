'''
Windows:

requirements:

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install opencv-python numpy

running:

python thermal_detector.py --config thermal_detector_config.json


Pipeline:

Main/display thread:
    - Reads the video at its original FPS
    - Shows the current live frame
    - Records the displayed annotated video

Detector thread:
    - Processes the newest available frame in the background
    - Drops stale/unprocessed frames automatically if processing is slower than video FPS
    - Updates the latest detection result

Display behavior:

    Current live frame + latest completed detection result

This simulates a real camera/Jetson livestream better than slowing down
the video to wait for detection.
'''

import argparse
import json
import os
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import numpy as np


@dataclass
class Detection:
    """
    Container for one candidate detection.

    bbox:
        Bounding box in OpenCV format: (x, y, w, h)

    center:
        Center point of the candidate: (cx, cy)

    area:
        Contour area in pixels.

    score:
        Simple ranking score. Currently based on brightness and area.
    """
    bbox: tuple
    center: tuple
    area: float
    score: float


@dataclass
class OverlayResult:
    """
    Container for the latest completed detector result.

    This result may come from an older frame than the frame currently being displayed.

    Example:
        Live frame: 120
        Detection source frame: 118
        Lag: 2 frames

    This is expected in live-stream mode.
    """
    detections: list
    track_bbox: object
    track_center: object
    track_hits: int
    track_missed: int
    source_frame_idx: int
    processing_ms: float


class KalmanTrack:
    """
    Simple constant-velocity Kalman tracker.

    State:
        [x, y, vx, vy]

    Measurement:
        [x, y]

    Purpose:
        - Smooth noisy candidate detections
        - Keep a tentative UAV track alive for a few missed frames
        - Reduce flickering between frames
    """

    def __init__(self, initial_center):
        self.kf = cv2.KalmanFilter(4, 2)

        # State transition matrix.
        # x_new = x + vx
        # y_new = y + vy
        # vx_new = vx
        # vy_new = vy
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Measurement matrix.
        # We only measure position: x, y.
        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float32,
        )

        # Process noise:
        # Higher value = tracker allows faster/random motion.
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # Measurement noise:
        # Higher value = tracker trusts detections less.
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 3.0

        # Initial state uncertainty.
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        x, y = initial_center
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)

        self.missed = 0
        self.hits = 1
        self.age = 1
        self.last_center = initial_center
        self.last_bbox = None

    def predict(self):
        """
        Predict the next track center using the Kalman model.
        """
        pred = self.kf.predict()
        self.age += 1
        return int(pred[0]), int(pred[1])

    def update(self, detection):
        """
        Correct the Kalman track using a new detection.
        """
        cx, cy = detection.center
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        corrected = self.kf.correct(measurement)

        self.last_center = (int(corrected[0]), int(corrected[1]))
        self.last_bbox = detection.bbox
        self.missed = 0
        self.hits += 1

    def mark_missed(self):
        """
        Mark that no detection was associated with this track for this detector step.
        """
        self.missed += 1


def preprocess_frame(frame, clahe_clip=2.0, clahe_tile=8, blur_ksize=3):
    """
    Preprocess one thermal video frame.

    Steps:
        1. Convert MP4 frame to grayscale thermal intensity image.
           Even if the video looks grayscale, OpenCV usually reads MP4 as BGR.
        2. Apply CLAHE to improve local contrast.
        3. Apply light Gaussian blur to reduce tiny noise.

    Notes:
        - Do not use too much blur because the UAV may be very small.
        - For tiny UAVs, blur_ksize = 1 or 3 is safer than 5.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip,
        tileGridSize=(clahe_tile, clahe_tile),
    )
    gray_eq = clahe.apply(gray)

    if blur_ksize > 1:
        # Gaussian kernel size must be odd.
        if blur_ksize % 2 == 0:
            blur_ksize += 1

        gray_eq = cv2.GaussianBlur(gray_eq, (blur_ksize, blur_ksize), 0)

    return gray, gray_eq


def top_hat_detector(gray_eq, kernel_size=15):
    """
    White top-hat detector.

    Intuition:
        White top-hat enhances small bright targets and suppresses
        slowly varying large background structures.

    This is useful for thermal UAV spotting because small UAVs often appear
    as bright compact blobs against a darker or smoother background.

    kernel_size:
        Controls the approximate object/background scale separation.

        Smaller kernel:
            - More sensitive to very tiny objects
            - More false positives

        Larger kernel:
            - More suppressive
            - May weaken very tiny UAVs
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    tophat = cv2.morphologyEx(gray_eq, cv2.MORPH_TOPHAT, kernel)

    # Otsu chooses a threshold automatically from the top-hat response.
    _, mask = cv2.threshold(
        tophat,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return tophat, mask


def cfar_local_contrast_detector(gray_eq, window_size=31, threshold_k=3.0):
    """
    Simple CFAR / local contrast detector.

    CFAR-like idea:
        Instead of asking whether a pixel is globally bright, ask whether
        it is significantly brighter than its local neighborhood.

    Score:
        score = (pixel - local_mean) / local_std

    If score > threshold_k:
        pixel is considered locally bright.

    window_size:
        Size of the local neighborhood used to compute mean and std.

        Smaller:
            - More local
            - More sensitive
            - More noisy

        Larger:
            - More stable background estimate
            - Less noisy
            - May miss targets in clutter

    threshold_k:
        Detection strictness.

        Lower:
            - More sensitive
            - More false positives

        Higher:
            - Cleaner
            - May miss weak UAVs
    """
    if window_size % 2 == 0:
        window_size += 1

    gray_f = gray_eq.astype(np.float32)

    local_mean = cv2.blur(gray_f, (window_size, window_size))
    local_sq_mean = cv2.blur(gray_f * gray_f, (window_size, window_size))

    local_var = local_sq_mean - local_mean * local_mean
    local_var = np.maximum(local_var, 1e-6)
    local_std = np.sqrt(local_var)

    score = (gray_f - local_mean) / (local_std + 1e-6)

    mask = (score > threshold_k).astype(np.uint8) * 255

    # Normalized score image is useful for debugging/visualization.
    score_vis = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX)
    score_vis = score_vis.astype(np.uint8)

    return score, score_vis, mask


def clean_mask(mask, open_size=3, dilate_size=3):
    """
    Clean the binary candidate mask.

    Opening:
        Removes tiny isolated noise.

    Dilation:
        Slightly expands/joins fragmented candidate pixels.

    Be careful:
        Too much opening can delete tiny UAVs.
        Too much dilation can merge clutter into large blobs.
    """
    if open_size > 1:
        if open_size % 2 == 0:
            open_size += 1

        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_size, open_size),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    if dilate_size > 1:
        if dilate_size % 2 == 0:
            dilate_size += 1

        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilate_size, dilate_size),
        )
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)

    return mask


def extract_detections(
    mask,
    gray_eq,
    min_area=3,
    max_area=600,
    min_aspect=0.25,
    max_aspect=4.0,
    min_mean_intensity=80,
):
    """
    Convert the binary mask into candidate UAV detections.

    Filtering criteria:
        area:
            Rejects extremely tiny noise and very large blobs.

        aspect ratio:
            Rejects very long horizontal/vertical artifacts.

        mean intensity:
            Rejects candidates that are not bright enough.

    Output:
        List of Detection objects sorted by score.
    """
    detections = []

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
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

        roi = gray_eq[y : y + h, x : x + w]
        mean_intensity = float(np.mean(roi))

        if mean_intensity < min_mean_intensity:
            continue

        cx = x + w // 2
        cy = y + h // 2

        # Simple score:
        # Brighter and slightly larger candidates rank higher.
        score = mean_intensity + 0.1 * area

        detections.append(
            Detection(
                bbox=(x, y, w, h),
                center=(cx, cy),
                area=area,
                score=score,
            )
        )

    detections.sort(key=lambda d: d.score, reverse=True)
    return detections


def associate_detection_to_track(track, detections, max_distance=60):
    """
    Associate the nearest detection to the predicted Kalman position.

    max_distance:
        Maximum pixel distance between predicted track center and detection center.

        Lower:
            - Less likely to jump to false targets
            - May lose fast-moving UAV

        Higher:
            - Can handle faster motion
            - More likely to jump to wrong object
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


def process_frame_for_detection(frame, args, track):
    """
    Full detector pipeline for one frame.

    This function runs in the detector thread, not in the display thread.

    Pipeline:
        frame
        -> grayscale
        -> CLAHE
        -> blur
        -> top-hat detector
        -> CFAR/local contrast detector
        -> combine masks
        -> clean mask
        -> contour filtering
        -> Kalman tracking

    Returns:
        detections:
            Raw red-box candidates.

        track:
            Updated Kalman track object.

        track_bbox / track_center:
            Latest green-box tracked candidate.

        processing_ms:
            How long this detector step took.
    """
    start_time = time.perf_counter()

    _, gray_eq = preprocess_frame(
        frame,
        clahe_clip=args.clahe_clip,
        clahe_tile=args.clahe_tile,
        blur_ksize=args.blur,
    )

    _, mask_tophat = top_hat_detector(
        gray_eq,
        kernel_size=args.tophat_kernel,
    )

    _, _, mask_cfar = cfar_local_contrast_detector(
        gray_eq,
        window_size=args.cfar_window,
        threshold_k=args.cfar_k,
    )

    # Combine top-hat and CFAR masks.
    #
    # AND:
    #     Stricter.
    #     Candidate must pass both top-hat and CFAR.
    #
    # OR:
    #     More sensitive.
    #     Candidate can pass either detector.
    if args.combine == "and":
        combined_mask = cv2.bitwise_and(mask_tophat, mask_cfar)
    else:
        combined_mask = cv2.bitwise_or(mask_tophat, mask_cfar)

    combined_mask = clean_mask(
        combined_mask,
        open_size=args.open_size,
        dilate_size=args.dilate_size,
    )

    detections = extract_detections(
        combined_mask,
        gray_eq,
        min_area=args.min_area,
        max_area=args.max_area,
        min_aspect=args.min_aspect,
        max_aspect=args.max_aspect,
        min_mean_intensity=args.min_intensity,
    )

    # Tracking logic:
    #
    # If there is no track, initialize from the highest-score detection.
    # If there is a track, associate the nearest detection to it.
    if track is None:
        if len(detections) > 0:
            track = KalmanTrack(detections[0].center)
            track.update(detections[0])
    else:
        matched = associate_detection_to_track(
            track,
            detections,
            max_distance=args.max_assoc_distance,
        )

        if matched is not None:
            track.update(matched)
        else:
            track.mark_missed()

        if track.missed > args.max_missed:
            track = None

    processing_ms = (time.perf_counter() - start_time) * 1000.0

    track_bbox = None
    track_center = None
    track_hits = 0
    track_missed = 0

    if track is not None:
        track_bbox = track.last_bbox
        track_center = track.last_center
        track_hits = track.hits
        track_missed = track.missed

    return (
        detections,
        track,
        track_bbox,
        track_center,
        track_hits,
        track_missed,
        processing_ms,
    )


def draw_live_results(frame, result, current_frame_idx):
    """
    Draw latest available detection result on the current live frame.

    Important:
        The detection result may belong to an older frame than the current one.

    That is intentional:
        The video display behaves like a real live stream.
        The detector result arrives whenever processing finishes.

    Red boxes:
        Raw candidate detections.

    Green box:
        Kalman-tracked candidate.

    On-screen lag:
        current live frame index - detection source frame index

    If lag is small:
        Detector is keeping up.

    If lag is large:
        Detector is too slow for true real-time operation.
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

    # Red boxes: latest raw candidate detections.
    for det in result.detections:
        x, y, w, h = det.bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), (80, 80, 255), 1)
        cv2.circle(output, det.center, 2, (80, 80, 255), -1)

    # Green box: latest Kalman-tracked candidate.
    if result.track_bbox is not None and result.track_center is not None:
        x, y, w, h = result.track_bbox
        cx, cy = result.track_center

        if result.track_hits >= 3 and result.track_missed <= 5:
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


def run_detector(args):
    """
    Main live-stream simulation.

    Main thread:
        - Reads frames from video.
        - Maintains original video playback timing.
        - Displays current live frame.
        - Writes annotated output video.

    Detector thread:
        - Receives the newest available frame.
        - Processes in the background.
        - Stores latest completed result.

    Stale frame policy:
        If the detector is slow, the newest frame overwrites older unprocessed frames.
        This prevents delay from growing forever.

    This is the correct behavior for live camera-like operation.
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
    output_video = os.path.join(
        args.output_dir,
        f"{basename}_thermal_detected_live.mp4",
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {output_video}")

    print(f"Input: {args.input}")
    print(f"Resolution: {width}x{height}, FPS={fps}")
    print(f"Output: {output_video}")
    print("Mode: live-stream simulation")
    print("Main thread: real-time video display/write")
    print("Detector thread: newest-frame processing")

    # Shared state between main/display thread and detector thread.
    #
    # latest_frame:
    #     Newest frame made available to detector.
    #
    # latest_frame_idx:
    #     Frame index of latest_frame.
    #
    # latest_result:
    #     Most recent completed detection result.
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

        This thread always processes the newest available frame.

        If the main thread produces frames faster than this thread can process,
        older frames are overwritten. That is intentional for live-stream behavior.
        """
        track = None
        last_processed_idx = -1

        while not stop_event.is_set():
            with lock:
                frame = shared["latest_frame"]
                frame_idx = shared["latest_frame_idx"]

                if frame is None or frame_idx == last_processed_idx:
                    frame_to_process = None
                    idx_to_process = None
                else:
                    # Copy frame so main thread can safely continue.
                    frame_to_process = frame.copy()
                    idx_to_process = frame_idx

            if frame_to_process is None:
                time.sleep(0.001)
                continue

            (
                detections,
                track,
                track_bbox,
                track_center,
                track_hits,
                track_missed,
                processing_ms,
            ) = process_frame_for_detection(frame_to_process, args, track)

            result = OverlayResult(
                detections=detections,
                track_bbox=track_bbox,
                track_center=track_center,
                track_hits=track_hits,
                track_missed=track_missed,
                source_frame_idx=idx_to_process,
                processing_ms=processing_ms,
            )

            with lock:
                shared["latest_result"] = result

            last_processed_idx = idx_to_process

    worker = threading.Thread(target=detector_worker, daemon=True)
    worker.start()

    frame_idx = 0
    playback_start_time = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Provide newest live frame to detector.
            # If detector is slow, older unprocessed frames are overwritten.
            with lock:
                shared["latest_frame"] = frame.copy()
                shared["latest_frame_idx"] = frame_idx
                result = shared["latest_result"]

            annotated = draw_live_results(frame, result, frame_idx)

            # Record exactly what the live operator sees.
            writer.write(annotated)

            if args.show:
                display_frame = annotated

                if args.display_scale != 1.0:
                    display_frame = cv2.resize(
                        display_frame,
                        None,
                        fx=args.display_scale,
                        fy=args.display_scale,
                        interpolation=cv2.INTER_AREA,
                    )

                cv2.imshow("Thermal UAV Detection - Live Stream", display_frame)

            # Keep video playback synchronized to original FPS.
            #
            # If processing/display is faster than the original video FPS:
            #     wait a little.
            #
            # If processing/display is already late:
            #     do not wait.
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

    print(f"Done. Played {frame_idx} live frames.")
    print(f"Saved live annotated video to: {output_video}")


def load_config():
    """
    Load detector parameters from a JSON config file.

    The only command-line argument is:

        --config thermal_detector_config.json

    All tuning parameters are stored in the JSON file to avoid long commands.
    """
    parser = argparse.ArgumentParser(
        description="Classical non-CNN thermal UAV detector using a JSON config file."
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
        "max_missed",
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