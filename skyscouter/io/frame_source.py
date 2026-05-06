"""
Frame source abstraction.

Any frame source produces a stream of `Frame` objects. Adding a live camera
later is a new subclass; the rest of the pipeline does not change.

Per PRD FR-CAM-002, every frame carries a timestamp from a single time base.
For bench replay on a video file, that time base is derived from the video's
own timeline (frame_index / fps) plus the wall-clock at session start.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class Frame:
    """A single image with its timestamps and metadata."""
    image_bgr: np.ndarray          # H x W x 3, uint8, BGR (OpenCV convention)
    frame_index: int               # zero-based frame index from source
    capture_time_s: float          # seconds since session start (monotonic)
    timestamp_utc: str             # ISO-8601 UTC at capture
    source_id: str                 # identifier of the source (filename, etc.)
    width: int
    height: int


class BaseFrameSource(ABC):
    """Base class for any source of frames."""

    @abstractmethod
    def __iter__(self) -> Iterator[Frame]:
        ...

    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """Return (width, height) of frames produced by this source."""
        ...

    @abstractmethod
    def get_fps(self) -> Optional[float]:
        """Return frame rate if known, else None."""
        ...

    @abstractmethod
    def get_source_id(self) -> str:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class VideoFileSource(BaseFrameSource):
    """Reads frames from a video file using OpenCV."""

    def __init__(self, video_path: str, frame_stride: int = 1, strict: bool = True):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")

        self._path = video_path
        self._stride = frame_stride
        self._strict = strict
        self._source_id = Path(video_path).stem

        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or None
        if self._fps is not None and self._fps <= 0:
            self._fps = None

        # Session start in wall-clock (UTC) — anchors all timestamps.
        self._session_utc_start = datetime.now(timezone.utc)

    def get_resolution(self) -> Tuple[int, int]:
        return (self._width, self._height)

    def get_fps(self) -> Optional[float]:
        return self._fps

    def get_source_id(self) -> str:
        return self._source_id

    def __iter__(self) -> Iterator[Frame]:
        frame_idx = 0
        emitted = 0
        while True:
            ok, img = self._cap.read()
            if not ok:
                # End of stream. cv2 doesn't distinguish clean EOF from error,
                # but we don't fabricate frames — just stop.
                break

            if frame_idx % self._stride == 0:
                # Capture time: prefer video PTS; fall back to frame_idx / fps.
                if self._fps:
                    capture_time_s = frame_idx / self._fps
                else:
                    capture_time_s = float(frame_idx)

                # UTC: session start + capture offset (deterministic, replayable)
                utc = self._session_utc_start.timestamp() + capture_time_s
                timestamp_utc = datetime.fromtimestamp(utc, tz=timezone.utc).isoformat(
                    timespec="milliseconds"
                ).replace("+00:00", "Z")

                yield Frame(
                    image_bgr=img,
                    frame_index=frame_idx,
                    capture_time_s=capture_time_s,
                    timestamp_utc=timestamp_utc,
                    source_id=self._source_id,
                    width=self._width,
                    height=self._height,
                )
                emitted += 1
            frame_idx += 1

        if emitted == 0 and self._strict:
            raise RuntimeError(
                f"No frames could be read from {self._path}. "
                f"Check codec compatibility (try ffmpeg -i to inspect)."
            )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class ImageFolderSource(BaseFrameSource):
    """Reads frames from a folder of images sorted by filename."""

    SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(self, folder_path: str, frame_stride: int = 1, fps: float = 30.0):
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        if fps <= 0:
            raise ValueError("fps must be > 0 for image folder source")

        self._folder = folder_path
        self._stride = frame_stride
        self._assumed_fps = fps
        self._source_id = Path(folder_path).name

        self._files = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if Path(f).suffix.lower() in self.SUPPORTED_EXT
        ])
        if not self._files:
            raise RuntimeError(f"No images found in {folder_path}")

        # Determine resolution from first image
        first = cv2.imread(self._files[0])
        if first is None:
            raise IOError(f"Could not read first image: {self._files[0]}")
        self._height, self._width = first.shape[:2]

        self._session_utc_start = datetime.now(timezone.utc)

    def get_resolution(self) -> Tuple[int, int]:
        return (self._width, self._height)

    def get_fps(self) -> Optional[float]:
        return self._assumed_fps

    def get_source_id(self) -> str:
        return self._source_id

    def __iter__(self) -> Iterator[Frame]:
        for idx, path in enumerate(self._files):
            if idx % self._stride != 0:
                continue
            img = cv2.imread(path)
            if img is None:
                continue  # silently skip unreadable images
            capture_time_s = idx / self._assumed_fps
            utc = self._session_utc_start.timestamp() + capture_time_s
            timestamp_utc = datetime.fromtimestamp(utc, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ).replace("+00:00", "Z")
            yield Frame(
                image_bgr=img,
                frame_index=idx,
                capture_time_s=capture_time_s,
                timestamp_utc=timestamp_utc,
                source_id=self._source_id,
                width=self._width,
                height=self._height,
            )

    def close(self) -> None:
        pass


class OpenCVCameraSource(BaseFrameSource):
    """Reads live frames from a USB/V4L2/OpenCV camera."""

    def __init__(
        self,
        device: Union[int, str] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        fourcc: Optional[str] = None,
        frame_stride: int = 1,
        strict: bool = True,
        backend: Optional[str] = None,
        buffer_size: int = 1,
    ):
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        self._device = device
        self._stride = frame_stride
        self._strict = strict
        self._source_id = f"opencv_camera:{device}"
        self._session_utc_start = datetime.now(timezone.utc)
        self._session_monotonic_start = time.monotonic()

        backend_flag = cv2.CAP_ANY
        if backend:
            name = backend.strip().upper()
            if name == "V4L2":
                backend_flag = cv2.CAP_V4L2
            elif name == "DSHOW":
                backend_flag = cv2.CAP_DSHOW
            elif name == "MSMF":
                backend_flag = cv2.CAP_MSMF
            elif name not in {"ANY", "AUTO"}:
                raise ValueError(f"Unsupported OpenCV camera backend: {backend}")

        cap_device: Union[int, str] = int(device) if isinstance(device, str) and device.isdigit() else device
        self._cap = cv2.VideoCapture(cap_device, backend_flag)
        if not self._cap.isOpened():
            raise IOError(f"Could not open camera source: {device}")

        if width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        if fps is not None:
            self._cap.set(cv2.CAP_PROP_FPS, float(fps))
        if fourcc:
            code = fourcc[:4]
            if len(code) != 4:
                raise ValueError("fourcc must be four characters, for example MJPG")
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*code))
        if buffer_size > 0:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, int(buffer_size))

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or fps
        if self._fps is not None and self._fps <= 0:
            self._fps = fps

        if self._width <= 0 or self._height <= 0:
            ok, img = self._cap.read()
            if not ok or img is None:
                self.close()
                raise RuntimeError(f"Camera opened but produced no first frame: {device}")
            self._height, self._width = img.shape[:2]
            self._first_frame = img
        else:
            self._first_frame = None

    def get_resolution(self) -> Tuple[int, int]:
        return (self._width, self._height)

    def get_fps(self) -> Optional[float]:
        return self._fps

    def get_source_id(self) -> str:
        return self._source_id

    def __iter__(self) -> Iterator[Frame]:
        frame_idx = 0
        emitted = 0
        first = self._first_frame
        self._first_frame = None

        while True:
            if first is not None:
                ok, img = True, first
                first = None
            else:
                ok, img = self._cap.read()
            if not ok or img is None:
                if emitted == 0 and self._strict:
                    raise RuntimeError(f"No frames could be read from camera source: {self._device}")
                break

            if frame_idx % self._stride == 0:
                capture_time_s = time.monotonic() - self._session_monotonic_start
                utc = self._session_utc_start.timestamp() + capture_time_s
                timestamp_utc = datetime.fromtimestamp(utc, tz=timezone.utc).isoformat(
                    timespec="milliseconds"
                ).replace("+00:00", "Z")
                height, width = img.shape[:2]
                yield Frame(
                    image_bgr=img,
                    frame_index=frame_idx,
                    capture_time_s=capture_time_s,
                    timestamp_utc=timestamp_utc,
                    source_id=self._source_id,
                    width=width,
                    height=height,
                )
                emitted += 1
            frame_idx += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def build_frame_source(source_cfg: dict, video_path: Optional[str] = None) -> BaseFrameSource:
    """Factory that builds a frame source from config."""
    src_type = source_cfg.get("type", "video_file")
    stride = int(source_cfg.get("frame_stride", 1))
    strict = bool(source_cfg.get("strict", True))

    if src_type == "video_file":
        if not video_path:
            raise ValueError("video_path is required for source.type='video_file'")
        return VideoFileSource(video_path, frame_stride=stride, strict=strict)

    if src_type == "image_folder":
        if not video_path:
            raise ValueError("a folder path must be passed via --video for source.type='image_folder'")
        fps = float(source_cfg.get("assumed_fps", 30.0))
        return ImageFolderSource(video_path, frame_stride=stride, fps=fps)

    if src_type in {"live_camera", "opencv_camera"}:
        return OpenCVCameraSource(
            device=source_cfg.get("device", 0),
            width=source_cfg.get("width"),
            height=source_cfg.get("height"),
            fps=source_cfg.get("fps"),
            fourcc=source_cfg.get("fourcc"),
            frame_stride=stride,
            strict=strict,
            backend=source_cfg.get("backend"),
            buffer_size=int(source_cfg.get("buffer_size", 1)),
        )

    raise ValueError(f"Unknown source.type: {src_type}")
