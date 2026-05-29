"""Low-latency RTSP/IP camera frame source."""
from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Iterator, Optional, Tuple

import cv2
import numpy as np

from .frame_source import BaseFrameSource, Frame

# ---------------------------------------------------------------------------
# Optional GStreamer nvv4l2decoder hardware-decode path
#
# The pip opencv-python wheel has GStreamer: NO, so we cannot use
# cv2.CAP_GSTREAMER.  Instead we drive GStreamer directly through
# gi.repository.Gst (PyGObject), which JetPack ships as part of the
# system GStreamer runtime.  The _GstNvdecCapture wrapper presents the
# same minimal interface as cv2.VideoCapture so _reader_loop() is
# unchanged.  We fall back to the FFmpeg path transparently if anything
# in the GStreamer stack is missing.
# ---------------------------------------------------------------------------

_GST_NVDEC_AVAILABLE: Optional[bool] = None


def _check_gst_nvdec() -> bool:
    """Return True if gi.Gst + the nvvideo4linux2 plugin are present."""
    global _GST_NVDEC_AVAILABLE
    if os.environ.get("SKY_NO_GST"):
        _GST_NVDEC_AVAILABLE = False
        return False
    if _GST_NVDEC_AVAILABLE is not None:
        return _GST_NVDEC_AVAILABLE
    try:
        import gi  # type: ignore[import]
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst  # type: ignore[import]
        if not Gst.is_initialized():
            Gst.init(None)
        _GST_NVDEC_AVAILABLE = (
            Gst.Registry.get().find_plugin("nvvideo4linux2") is not None
        )
    except Exception:
        _GST_NVDEC_AVAILABLE = False
    return _GST_NVDEC_AVAILABLE


class _GstNvdecCapture:
    """Hardware RTSP decoder using GStreamer nvv4l2decoder.

    Pipeline::

        rtspsrc → rtph264depay → h264parse → nvv4l2decoder
                → nvvidconv → BGRx → videoconvert → BGR
                → appsink (max-buffers=1, drop=true)

    The nvv4l2decoder element runs on the Jetson hardware video decoder,
    freeing the CPU cores that FFmpeg would otherwise use for software
    H.264 decode.

    Public interface matches cv2.VideoCapture (isOpened / read / release)
    so that RtspStreamSource._reader_loop() works with both backends.
    """

    _PULL_TIMEOUT_NS = 500_000_000  # 500 ms — short enough that close()
                                    # wakes within ~0.5 s; stream at 30 fps
                                    # delivers a new frame every 33 ms.

    def __init__(self, url: str, transport: str, open_timeout_s: float) -> None:
        import gi  # type: ignore[import]
        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
        from gi.repository import Gst, GstApp  # type: ignore[import]  # noqa: F401
        # GstApp must be imported so PyGObject registers the GstAppSink subclass
        # and exposes try_pull_sample() on the element returned by get_by_name().
        if not Gst.is_initialized():
            Gst.init(None)
        self._Gst = Gst

        # SIYI A8 Mini streams H.265/HEVC despite the URL ending in .264.
        # The SDP explicitly negotiates encoding-name=H265 over RTP.
        # rtph265depay + nvv4l2decoder handles HEVC natively in hardware.
        # h265parse is NOT needed (not installed) — nvv4l2decoder accepts
        # the raw HEVC bitstream from rtph265depay directly.
        pipeline_str = (
            f"rtspsrc location={url} protocols={transport} latency=0 ! "
            "rtph265depay ! "
            "nvv4l2decoder ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=sink max-buffers=1 drop=true sync=false"
        )
        self._pipeline = Gst.parse_launch(pipeline_str)
        self._sink = self._pipeline.get_by_name("sink")
        self._opened = False
        self._width = 0
        self._height = 0

        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self._pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(
                "GStreamer pipeline failed to start (FAILURE on set_state)"
            )

        # ASYNC is normal for network sources — the pipeline will complete
        # the transition independently; we will wait for frames via appsink.
        change, _cur, _pending = self._pipeline.get_state(
            int(open_timeout_s * Gst.SECOND)
        )
        if change == Gst.StateChangeReturn.FAILURE:
            self._pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(
                f"GStreamer pipeline state-change FAILURE (result={change})"
            )
        self._opened = True

    # ------------------------------------------------------------------
    # cv2.VideoCapture-compatible interface
    # ------------------------------------------------------------------

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._opened:
            return False, None

        Gst = self._Gst
        sample = self._sink.try_pull_sample(self._PULL_TIMEOUT_NS)
        if sample is None:
            self._drain_bus()
            # Distinguish between two cases:
            #   ok=True,  frame=None → transient timeout; pipeline still alive,
            #                          caller should retry without reconnecting.
            #   ok=False, frame=None → pipeline posted EOS/ERROR; caller must
            #                          release and reconnect.
            return self._opened, None

        caps = sample.get_caps()
        s = caps.get_structure(0)
        w = int(s.get_value("width"))
        h = int(s.get_value("height"))
        if self._width == 0:
            self._width = w
            self._height = h

        buf = sample.get_buffer()
        ok, map_info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return False, None
        try:
            # BGR: 3 bytes per pixel
            frame = (
                np.frombuffer(map_info.data, dtype=np.uint8)
                .reshape(h, w, 3)
                .copy()
            )
        finally:
            buf.unmap(map_info)
        return True, frame

    def release(self) -> None:
        self._opened = False
        if self._pipeline is not None:
            self._pipeline.set_state(self._Gst.State.NULL)
            self._pipeline = None  # type: ignore[assignment]

    # ------------------------------------------------------------------

    def _drain_bus(self) -> None:
        """Mark opened=False if the pipeline posted an error or EOS."""
        Gst = self._Gst
        bus = self._pipeline.get_bus()
        msg = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
        if msg is not None:
            self._opened = False


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
        self._capture_backend: str = "unknown"
        self._reconnect_count = 0

        # Optional frame observer — called on every decoded frame before the
        # latest-frame buffer is updated.  Used by InterFrameLKThread to
        # receive all camera frames at full fps, not just the frames that
        # reach the YOLO detector.  Registered via register_frame_observer().
        self._frame_observer_cb = None

        self._reader = threading.Thread(
            target=self._reader_loop,
            name=f"skyscouter-rtsp-reader-{self._source_id}",
            daemon=True,
        )
        self._reader.start()
        self._wait_for_first_frame()

    def register_frame_observer(self, cb) -> None:
        """Register a callback invoked on every decoded frame (all camera fps).

        The callback receives a single argument: the BGR numpy frame.  It is
        called from the reader thread so it must be non-blocking (e.g. queue a
        copy, never do heavy computation inline).  Only one observer at a time.
        """
        self._frame_observer_cb = cb

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
            "capture_backend": self._capture_backend,
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
        # On a FIELD cold-boot the Jetson is usually up before the SIYI camera /
        # air-unit has finished powering on and is serving a decodable stream.
        # While we wait, the only console output is the NvMM decoder's own init
        # lines (".. NvMMLiteBlockCreate .."), so a not-yet-ready camera looks
        # exactly like a frozen hang.  Emit a heartbeat every few seconds so the
        # operator can SEE we are waiting for the camera (not stuck), and give a
        # clear, actionable message if the grace period really does expire.
        start = time.monotonic()
        deadline = start + self._first_frame_timeout_s
        next_log = start + 5.0
        with self._condition:
            while self._latest_frame is None and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=min(0.25, remaining))
                now = time.monotonic()
                if now >= next_log and self._latest_frame is None and not self._closed:
                    print(
                        f"[rtsp] waiting for first frame from {self._redacted_url()} "
                        f"({now - start:.0f}/{self._first_frame_timeout_s:.0f}s, "
                        f"backend={self._capture_backend}, reconnects={self._reconnect_count}) "
                        "— camera may still be powering up",
                        file=sys.stderr, flush=True,
                    )
                    next_log = now + 5.0

            if self._latest_frame is None:
                self._closed = True
                error = self._reader_error or (
                    f"Timed out waiting {self._first_frame_timeout_s:.1f}s for first RTSP frame "
                    f"from {self._redacted_url()} — camera not streaming yet? Check the "
                    f"camera power/link; a cold field boot can take longer than this budget "
                    f"(raise source.first_frame_timeout_s)."
                )
                if self._strict:
                    raise RuntimeError(error)

    def _reader_loop(self) -> None:
        # Safety limit for consecutive empty reads (ok=True but frame=None).
        # Each pull has a 500 ms timeout so 20 consecutive empties ≈ 10 s.
        # This guards against a GStreamer pipeline that stays "alive" (no
        # EOS/ERROR on the bus) but stops delivering frames.
        _EMPTY_READ_LIMIT = 20

        attempt = 0
        while True:
            with self._condition:
                if self._closed:
                    return

            # The first iteration is the initial connect; every iteration after
            # is a reconnect.  Surface reconnects to the operator — during a
            # field cold-boot the camera may refuse several times before it is
            # ready, and silent retries look identical to a frozen hang.
            attempt += 1
            if attempt > 1:
                self._reconnect_count += 1
                print(
                    f"[rtsp] reconnect attempt {self._reconnect_count} to "
                    f"{self._redacted_url()} (backend={self._capture_backend})",
                    file=sys.stderr, flush=True,
                )

            cap = self._open_capture()
            if cap is None:
                self._sleep_before_reconnect()
                continue

            consecutive_empty = 0
            try:
                while True:
                    with self._condition:
                        if self._closed:
                            return

                    ok, frame = cap.read()

                    if not ok:
                        # Real pipeline error (GStreamer EOS/ERROR or FFmpeg
                        # stream failure) — release and reconnect.
                        self._record_reader_error("RTSP read failed; reconnecting")
                        break

                    if frame is None:
                        # GStreamer backend only: try_pull_sample timed out but
                        # the pipeline is still alive (startup latency or brief
                        # network congestion).  Keep reading without reconnecting.
                        consecutive_empty += 1
                        if consecutive_empty >= _EMPTY_READ_LIMIT:
                            self._record_reader_error(
                                "RTSP stream produced no frames for 10 s; reconnecting"
                            )
                            break
                        continue

                    consecutive_empty = 0
                    height, width = frame.shape[:2]
                    if self._width <= 0 or self._height <= 0:
                        self._width = int(width)
                        self._height = int(height)

                    # Notify the inter-frame LK thread (or any other observer)
                    # on EVERY decoded camera frame — before the latest-frame
                    # buffer is updated.  The callback must be non-blocking.
                    if self._frame_observer_cb is not None:
                        try:
                            self._frame_observer_cb(frame)
                        except Exception:
                            pass

                    self._raw_read_count += 1
                    with self._condition:
                        self._latest_frame = frame
                        self._latest_sequence += 1
                        self._reader_error = None
                        self._condition.notify_all()
            finally:
                cap.release()

            self._sleep_before_reconnect()

    def _open_capture(self) -> Optional[Any]:
        # ------------------------------------------------------------------
        # Attempt 1: GStreamer nvv4l2decoder (hardware H.264 decode)
        # ------------------------------------------------------------------
        if _check_gst_nvdec():
            try:
                cap: Any = _GstNvdecCapture(
                    self._url, self._transport, self._open_timeout_s
                )
                self._capture_backend = "gst_nvdec"
                self._last_reconnect_utc = datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                )
                return cap
            except Exception as exc:
                self._record_reader_error(
                    f"GStreamer nvdec open failed ({exc}); falling back to FFmpeg"
                )

        # ------------------------------------------------------------------
        # Attempt 2: OpenCV FFmpeg (software decode)
        # ------------------------------------------------------------------
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
            self._record_reader_error(
                f"Could not open RTSP stream: {self._redacted_url()}"
            )
            cap.release()
            return None

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self._expected_width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._expected_width)
        if self._expected_height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._expected_height)
        if self._fps is not None:
            cap.set(cv2.CAP_PROP_FPS, self._fps)

        self._capture_backend = "ffmpeg"
        self._last_reconnect_utc = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
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
