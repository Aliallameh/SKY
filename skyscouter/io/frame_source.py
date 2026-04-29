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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional, Tuple

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

    if src_type == "live_camera":
        raise NotImplementedError(
            "Live camera ingest is not implemented in Sprint 0. "
            "Plug in a new BaseFrameSource subclass when the camera lands."
        )

    raise ValueError(f"Unknown source.type: {src_type}")
