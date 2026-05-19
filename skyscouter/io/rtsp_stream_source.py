"""Low-latency RTSP/IP camera frame source."""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Iterator, Optional, Tuple

import cv2

from .frame_source import BaseFrameSource, Frame


class RtspStreamSource(BaseFrameSource):
    """Reads the newest frame from a live RTSP stream.

    Network cameras keep producing frames even while inference is busy. A normal
    blocking ``VideoCapture.read()`` loop can lag behind by seconds because it
    drains buffered packets. This source uses a reader thread that continuously
    grabs frames and only exposes the newest decoded frame to the pipeline.
    """

    def __init__(
        self,
        url: str,
        *,
        source_id: str = "rtsp_stream",
        expected_width: Optional[int] = None,
        expected_height: Optional[int] = None,
        fps: Optional[float] = None,
        transport: str = "tcp",
        open_timeout_s: float = 10.0,
        read_timeout_s: float = 5.0,
        reconnect_delay_s: float = 1.0,
        first_frame_timeout_s: float = 15.0,
        max_frames: Optional[int] = None,
        frame_stride: int = 1,
        strict: bool = True,
    ) -> None:
        if not url:
            raise ValueError("RTSP source requires a non-empty url")
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        if max_frames is not None and int(max_frames) < 1:
            raise ValueError("max_frames must be >= 1 when provided")
        if transport.lower() not in {"tcp", "udp"}:
            raise ValueError("RTSP transport must be 'tcp' or 'udp'")

        self._url = str(url)
        self._source_id = str(source_id)
        self._expected_width = int(expected_width) if expected_width is not None else None
        self._expected_height = int(expected_height) if expected_height is not None else None
        self._fps = float(fps) if fps is not None else None
        self._transport = str(transport).lower()
        self._open_timeout_s = float(open_timeout_s)
        self._read_timeout_s = float(read_timeout_s)
        self._reconnect_delay_s = float(reconnect_delay_s)
        self._first_frame_timeout_s = float(first_frame_timeout_s)
        self._max_frames = int(max_frames) if max_frames is not None else None
        self._frame_stride = int(frame_stride)
        self._strict = bool(strict)
        self._session_monotonic_start = time.monotonic()

        self._condition = threading.Condition()
        self._closed = False
        self._reader_error: Optional[str] = None
        self._latest_frame = None
        self._latest_sequence = 0
        self._raw_read_count = 0
        self._width = self._expected_width or 0
        self._height = self._expected_height or 0
        self._last_reconnect_utc: Optional[str] = None

        self._reader = threading.Thread(
            target=self._reader_loop,
            name=f"skyscouter-rtsp-reader-{self._source_id}",
            daemon=True,
        )
        self._reader.start()
        self._wait_for_first_frame()

    def get_resolution(self) -> Tuple[int, int]:
        return (self._width, self._height)

    def get_fps(self) -> Optional[float]:
        return self._fps

    def get_source_id(self) -> str:
        return self._source_id

    def get_negotiated_settings(self) -> dict:
        return {
            "type": "rtsp_stream",
            "url": self._redacted_url(),
            "source_id": self._source_id,
            "transport": self._transport,
            "width": self._width,
            "height": self._height,
            "fps": self._fps,
            "frame_stride": self._frame_stride,
            "max_frames": self._max_frames,
            "raw_frames_read": self._raw_read_count,
            "last_reconnect_utc": self._last_reconnect_utc,
        }

    def __iter__(self) -> Iterator[Frame]:
        emitted_count = 0
        last_sequence = 0

        while True:
            if self._max_frames is not None and emitted_count >= self._max_frames:
                break

            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed or self._latest_sequence != last_sequence,
                    timeout=self._read_timeout_s,
                )
                if self._closed:
                    break
                if self._latest_sequence == last_sequence:
                    if self._strict:
                        raise RuntimeError(
                            f"RTSP stream {self._source_id!r} produced no new frames "
                            f"for {self._read_timeout_s:.1f}s"
                        )
                    continue
                frame = self._latest_frame
                sequence = self._latest_sequence
                reader_error = self._reader_error

            if frame is None:
                if reader_error and self._strict:
                    raise RuntimeError(reader_error)
                continue

            last_sequence = sequence
            if (sequence - 1) % self._frame_stride != 0:
                continue

            capture_time_s = time.monotonic() - self._session_monotonic_start
            timestamp_utc = datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            ).replace("+00:00", "Z")
            frame_height, frame_width = frame.shape[:2]
            yield Frame(
                image_bgr=frame.copy(),
                frame_index=emitted_count,
                capture_time_s=capture_time_s,
                timestamp_utc=timestamp_utc,
                source_id=self._source_id,
                width=frame_width,
                height=frame_height,
            )
            emitted_count += 1

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._reader.is_alive():
            self._reader.join(timeout=2.0)

    def _wait_for_first_frame(self) -> None:
        deadline = time.monotonic() + self._first_frame_timeout_s
        with self._condition:
            while self._latest_frame is None and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=min(0.25, remaining))

            if self._latest_frame is None:
                self._closed = True
                error = self._reader_error or (
                    f"Timed out waiting {self._first_frame_timeout_s:.1f}s for first RTSP frame "
                    f"from {self._redacted_url()}"
                )
                if self._strict:
                    raise RuntimeError(error)

    def _reader_loop(self) -> None:
        while True:
            with self._condition:
                if self._closed:
                    return

            cap = self._open_capture()
            if cap is None:
                self._sleep_before_reconnect()
                continue

            try:
                while True:
                    with self._condition:
                        if self._closed:
                            return

                    ok, frame = cap.read()
                    if not ok or frame is None:
                        self._record_reader_error("RTSP read failed; reconnecting")
                        break

                    height, width = frame.shape[:2]
                    if self._width <= 0 or self._height <= 0:
                        self._width = int(width)
                        self._height = int(height)

                    self._raw_read_count += 1
                    with self._condition:
                        self._latest_frame = frame
                        self._latest_sequence += 1
                        self._reader_error = None
                        self._condition.notify_all()
            finally:
                cap.release()

            self._sleep_before_reconnect()

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        previous_options = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = self._ffmpeg_capture_options()
        try:
            cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        finally:
            if previous_options is None:
                os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            else:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = previous_options

        if not cap.isOpened():
            self._record_reader_error(f"Could not open RTSP stream: {self._redacted_url()}")
            cap.release()
            return None

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self._expected_width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._expected_width)
        if self._expected_height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._expected_height)
        if self._fps is not None:
            cap.set(cv2.CAP_PROP_FPS, self._fps)

        self._last_reconnect_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return cap

    def _ffmpeg_capture_options(self) -> str:
        timeout_us = max(1, int(self._open_timeout_s * 1_000_000))
        return "|".join(
            [
                f"rtsp_transport;{self._transport}",
                f"stimeout;{timeout_us}",
                "fflags;nobuffer",
                "flags;low_delay",
                "max_delay;500000",
                "reorder_queue_size;0",
            ]
        )

    def _record_reader_error(self, message: str) -> None:
        with self._condition:
            self._reader_error = message
            self._condition.notify_all()

    def _sleep_before_reconnect(self) -> None:
        deadline = time.monotonic() + self._reconnect_delay_s
        while time.monotonic() < deadline:
            with self._condition:
                if self._closed:
                    return
            time.sleep(0.05)

    def _redacted_url(self) -> str:
        if "@" not in self._url:
            return self._url
        scheme, rest = self._url.split("://", 1) if "://" in self._url else ("", self._url)
        host_part = rest.split("@", 1)[1]
        return f"{scheme}://<credentials>@{host_part}" if scheme else f"<credentials>@{host_part}"
