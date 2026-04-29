"""
Run logger.

Produces a manifest (JSON) and a structured log (text) for every run.
Together they make a run fully replayable per PRD NFR-OBS-001.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class RunLogger:
    """One per pipeline run. Writes a manifest and a log file."""

    def __init__(self, output_dir: str, run_id: Optional[str] = None):
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._run_id = run_id or self._make_run_id()
        self._manifest: Dict[str, Any] = {
            "run_id": self._run_id,
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "argv": list(sys.argv),
            "env": {
                k: v for k, v in os.environ.items()
                if k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")
            },
            "config": None,
            "video_path": None,
            "frame_count": 0,
            "detections_total": 0,
            "tracks_created": 0,
            "ended_utc": None,
            "status": "running",
            "error": None,
        }

        self._log_path = self._dir / "run.log"
        self._manifest_path = self._dir / "manifest.json"

        self._logger = logging.getLogger(f"skyscouter.run.{self._run_id}")
        self._logger.setLevel(logging.INFO)
        # Avoid duplicate handlers if the logger was reused
        for h in list(self._logger.handlers):
            self._logger.removeHandler(h)

        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        fh = logging.FileHandler(self._log_path, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        self._logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        self._logger.addHandler(ch)

        self._logger.info(f"Run started: {self._run_id}")

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def output_dir(self) -> Path:
        return self._dir

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)

    def set_config(self, cfg: Dict[str, Any]) -> None:
        self._manifest["config"] = cfg

    def set_video_path(self, path: Optional[str]) -> None:
        self._manifest["video_path"] = path

    def increment_frame(self, n: int = 1) -> None:
        self._manifest["frame_count"] = int(self._manifest["frame_count"]) + n

    def increment_detections(self, n: int) -> None:
        self._manifest["detections_total"] = int(self._manifest["detections_total"]) + n

    def increment_tracks(self, n: int) -> None:
        self._manifest["tracks_created"] = int(self._manifest["tracks_created"]) + n

    def finalize(self, status: str = "completed", error: Optional[str] = None) -> None:
        self._manifest["ended_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._manifest["status"] = status
        self._manifest["error"] = error
        with self._manifest_path.open("w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2, ensure_ascii=False)
        self._logger.info(f"Run finished: status={status}")
        # Flush handlers
        for h in list(self._logger.handlers):
            try:
                h.flush()
            except Exception:
                pass

    @staticmethod
    def _make_run_id() -> str:
        return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
