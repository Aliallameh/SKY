"""Writes target_state messages to a JSONL file, one per processed frame."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, TextIO

from ..schemas import TargetState


class TargetStateJsonlWriter:
    """Persists TargetState messages as JSONL for offline replay (PRD NFR-OBS-001)."""

    def __init__(self, output_path: str):
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: Optional[TextIO] = None

    def __enter__(self) -> "TargetStateJsonlWriter":
        self._fh = self._path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write(self, state: TargetState) -> None:
        if self._fh is None:
            raise RuntimeError("Writer is not open. Use as context manager.")
        json.dump(state.to_dict(), self._fh, ensure_ascii=False)
        self._fh.write("\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
