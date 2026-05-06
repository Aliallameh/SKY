"""Live USB/V4L2 camera frame source for Jetson deployment."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Iterator, Optional, Tuple

import cv2

from .frame_source import BaseFrameSource, Frame


class LiveCameraSource(BaseFrameSource):
    """Reads frames from the verified Jetson USB camera path.

    The first SkyScouter live camera is the USB "4K ZOOM CAMERA" on
    `/dev/video0`, exposed to OpenCV as device index 0. It must be opened as
    MJPG for realtime work; YUYV is too slow at the target resolutions.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        fourcc: str = "MJPG",
        backend: str = "v4l2",
        source_id: str = "jetson_4k_zoom_camera",
        max_frames: Optional[int] = None,
        warmup_frames: int = 10,
        frame_stride: int = 1,
        strict: bool = True,
    ):
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        if max_frames is not None and int(max_frames) < 1:
            raise ValueError("max_frames must be >= 1 when provided")
        if warmup_frames < 0:
            raise ValueError("warmup_frames must be >= 0")
        if len(fourcc) != 4:
            raise ValueError("fourcc must be exactly four characters, for example MJPG")

        self._device_index = int(device_index)
        self._requested_width = int(width)
        self._requested_height = int(height)
        self._requested_fps = float(fps)
        self._requested_fourcc = str(fourcc).upper()
        self._backend_name = str(backend).lower()
        self._source_id = source_id
        self._max_frames = int(max_frames) if max_frames is not None else None
        self._warmup_frames = int(warmup_frames)
        self._frame_stride = int(frame_stride)
        self._strict = bool(strict)
        self._session_monotonic_start = time.monotonic()

        backend_flag = self._backend_flag(self._backend_name)
        self._cap = cv2.VideoCapture(self._device_index, backend_flag)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open live camera device index {self._device_index} "
                f"with backend={self._backend_name!r}. Expected Jetson USB camera at /dev/video0."
            )

        # Order matters on V4L2 cameras. Force MJPG before resolution/FPS so
        # OpenCV does not negotiate the much slower YUYV mode.
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self._requested_fourcc))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._requested_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested_height)
        self._cap.set(cv2.CAP_PROP_FPS, self._requested_fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS) or self._requested_fps)
        self._negotiated_fourcc = self._decode_fourcc(self._cap.get(cv2.CAP_PROP_FOURCC))

        if self._width <= 0 or self._height <= 0:
            self.close()
            raise RuntimeError("Live camera opened but did not report a valid resolution.")

        if self._strict:
            self._validate_negotiated_settings()

    @staticmethod
    def _backend_flag(backend: str) -> int:
        if backend in {"v4l2", "video4linux", "video4linux2"}:
            return cv2.CAP_V4L2
        if backend in {"auto", "any"}:
            return cv2.CAP_ANY
        raise ValueError(f"Unsupported live camera backend: {backend!r}. Use 'v4l2' on Jetson.")

    @staticmethod
    def _decode_fourcc(raw_value: float) -> str:
        value = int(raw_value)
        chars = [chr((value >> 8 * i) & 0xFF) for i in range(4)]
        text = "".join(chars)
        return text if text.strip("\x00") else "UNKNOWN"

    def _validate_negotiated_settings(self) -> None:
        problems = []
        if self._negotiated_fourcc != self._requested_fourcc:
            problems.append(f"fourcc={self._negotiated_fourcc!r}, expected {self._requested_fourcc!r}")
        if self._width != self._requested_width or self._height != self._requested_height:
            problems.append(
                f"resolution={self._width}x{self._height}, "
                f"expected {self._requested_width}x{self._requested_height}"
            )
        if self._fps > 0 and abs(self._fps - self._requested_fps) > 1.0:
            problems.append(f"fps={self._fps:.2f}, expected {self._requested_fps:.2f}")
        if problems:
            self.close()
            raise RuntimeError(
                "Live camera negotiated unexpected settings: "
                + "; ".join(problems)
                + ". Check v4l2-ctl --device=/dev/video0 --list-formats-ext."
            )

    def get_resolution(self) -> Tuple[int, int]:
        return (self._width, self._height)

    def get_fps(self) -> Optional[float]:
        return self._fps

    def get_source_id(self) -> str:
        return self._source_id

    def get_negotiated_settings(self) -> dict:
        return {
            "device_index": self._device_index,
            "backend": self._backend_name,
            "fourcc": self._negotiated_fourcc,
            "width": self._width,
            "height": self._height,
            "fps": self._fps,
            "warmup_frames": self._warmup_frames,
            "max_frames": self._max_frames,
        }

    def __iter__(self) -> Iterator[Frame]:
        raw_read_count = 0
        emitted_count = 0

        while raw_read_count < self._warmup_frames:
            ok, _ = self._cap.read()
            if not ok:
                if self._strict:
                    raise RuntimeError(
                        f"Live camera failed during warmup after {raw_read_count} frames."
                    )
                return
            raw_read_count += 1

        source_frame_index = 0
        while True:
            if self._max_frames is not None and emitted_count >= self._max_frames:
                break

            ok, img = self._cap.read()
            if not ok or img is None:
                if emitted_count == 0 and self._strict:
                    raise RuntimeError("Live camera produced no frames after warmup.")
                break

            if source_frame_index % self._frame_stride == 0:
                capture_time_s = time.monotonic() - self._session_monotonic_start
                timestamp_utc = datetime.now(timezone.utc).isoformat(
                    timespec="milliseconds"
                ).replace("+00:00", "Z")
                frame_height, frame_width = img.shape[:2]
                yield Frame(
                    image_bgr=img,
                    frame_index=emitted_count,
                    capture_time_s=capture_time_s,
                    timestamp_utc=timestamp_utc,
                    source_id=self._source_id,
                    width=frame_width,
                    height=frame_height,
                )
                emitted_count += 1

            source_frame_index += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
