"""Live operator view for processed SkyScouter frames."""
from __future__ import annotations

import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import cv2
import numpy as np


class _SharedJpegFrame:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.jpeg: Optional[bytes] = None
        self.sequence = 0
        self.closed = False

    def update(self, jpeg: bytes) -> None:
        with self.condition:
            self.jpeg = jpeg
            self.sequence += 1
            self.condition.notify_all()

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()


class _OperatorViewHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        shared_frame: _SharedJpegFrame,
        title: str,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.shared_frame = shared_frame
        self.title = title


class _MjpegHandler(BaseHTTPRequestHandler):
    server: _OperatorViewHTTPServer

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"", "/", "/index.html"}:
            self._write_index()
            return
        if path == "/healthz":
            self._write_text("ok\n", content_type="text/plain")
            return
        if path == "/latest.jpg":
            self._write_latest_jpeg()
            return
        if path == "/stream.mjpg":
            self._write_mjpeg_stream()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _write_index(self) -> None:
        title = self.server.title
        body = (
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            f"<title>{title}</title>"
            "<style>html,body{margin:0;background:#050505;color:#eee;"
            "font-family:sans-serif;height:100%;overflow:hidden}"
            "img{display:block;width:100vw;height:100vh;object-fit:contain}"
            ".bar{position:fixed;left:0;right:0;bottom:0;padding:6px 10px;"
            "background:rgba(0,0,0,.65);font-size:13px}</style></head>"
            "<body><img src=\"/stream.mjpg\" alt=\"SkyScouter live stream\">"
            f"<div class=\"bar\">{title} - live annotated ML view</div>"
            "</body></html>"
        ).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _write_text(self, text: str, *, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _write_latest_jpeg(self) -> None:
        with self.server.shared_frame.condition:
            jpeg = self.server.shared_frame.jpeg
        if jpeg is None:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "no frame yet")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(jpeg)

    def _write_mjpeg_stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        last_sequence = -1
        while True:
            with self.server.shared_frame.condition:
                self.server.shared_frame.condition.wait_for(
                    lambda: (
                        self.server.shared_frame.closed
                        or self.server.shared_frame.sequence != last_sequence
                    ),
                    timeout=1.0,
                )
                if self.server.shared_frame.closed:
                    break
                jpeg = self.server.shared_frame.jpeg
                last_sequence = self.server.shared_frame.sequence

            if jpeg is None:
                continue

            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                break


class LiveOperatorView:
    """Publishes annotated frames to an operator-visible live view."""

    VALID_MODES = {"mjpeg", "window", "both"}

    def __init__(
        self,
        *,
        mode: str = "mjpeg",
        host: str = "0.0.0.0",
        port: int = 8090,
        jpeg_quality: int = 75,
        max_width: Optional[int] = 1280,
        window_name: str = "SkyScouter Operator View",
        wait_key_ms: int = 1,
    ) -> None:
        mode = str(mode).lower().strip()
        if mode not in self.VALID_MODES:
            raise ValueError(f"operator view mode must be one of {sorted(self.VALID_MODES)}")
        if not 1 <= int(jpeg_quality) <= 100:
            raise ValueError("operator view jpeg_quality must be between 1 and 100")
        if int(port) < 0 or int(port) > 65535:
            raise ValueError("operator view port must be between 0 and 65535")
        if max_width is not None and int(max_width) < 1:
            raise ValueError("operator view max_width must be >= 1 when provided")

        self.mode = mode
        self.host = str(host)
        self.port = int(port)
        self.jpeg_quality = int(jpeg_quality)
        self.max_width = int(max_width) if max_width is not None else None
        self.window_name = str(window_name)
        self.wait_key_ms = int(wait_key_ms)
        self.stop_requested = False
        self._closed = False

        self._stream_enabled = self.mode in {"mjpeg", "both"}
        self._window_enabled = self.mode in {"window", "both"}
        self._shared_frame: Optional[_SharedJpegFrame] = None
        self._server: Optional[_OperatorViewHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self.url: Optional[str] = None
        self.remote_url_hint: Optional[str] = None

        try:
            if self._stream_enabled:
                self._start_mjpeg_server()
            if self._window_enabled:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        except Exception:
            self.close()
            raise

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LiveOperatorView":
        return cls(
            mode=str(cfg.get("mode", "mjpeg")),
            host=str(cfg.get("host", "0.0.0.0")),
            port=int(cfg.get("port", 8090)),
            jpeg_quality=int(cfg.get("jpeg_quality", 75)),
            max_width=cfg.get("max_width", 1280),
            window_name=str(cfg.get("window_name", "SkyScouter Operator View")),
            wait_key_ms=int(cfg.get("wait_key_ms", 1)),
        )

    def publish(self, image_bgr: np.ndarray) -> None:
        if self._closed:
            return
        frame = self._resize_for_view(image_bgr)

        if self._stream_enabled and self._shared_frame is not None:
            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if ok:
                self._shared_frame.update(encoded.tobytes())

        if self._window_enabled:
            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(max(1, self.wait_key_ms)) & 0xFF
            if key in (27, ord("q")):
                self.stop_requested = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._shared_frame is not None:
            self._shared_frame.close()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
        if self._window_enabled:
            cv2.destroyWindow(self.window_name)

    def _start_mjpeg_server(self) -> None:
        self._shared_frame = _SharedJpegFrame()
        self._server = _OperatorViewHTTPServer(
            (self.host, self.port),
            _MjpegHandler,
            self._shared_frame,
            self.window_name,
        )
        actual_host, actual_port = self._server.server_address[:2]
        self.port = int(actual_port)
        url_host = "127.0.0.1" if actual_host in {"", "0.0.0.0"} else str(actual_host)
        self.url = f"http://{url_host}:{self.port}/"
        self.remote_url_hint = (
            f"http://<jetson-ip>:{self.port}/"
            if actual_host in {"", "0.0.0.0"}
            else self.url
        )
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="skyscouter-operator-view",
            daemon=True,
        )
        self._server_thread.start()

    def _resize_for_view(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.max_width is None:
            return image_bgr
        height, width = image_bgr.shape[:2]
        if width <= self.max_width:
            return image_bgr
        scale = self.max_width / float(width)
        view_height = max(1, int(round(height * scale)))
        return cv2.resize(image_bgr, (self.max_width, view_height), interpolation=cv2.INTER_AREA)
