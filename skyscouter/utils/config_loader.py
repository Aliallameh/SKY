"""Config loader. Reads YAML, validates lightly, returns a plain dict."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


REQUIRED_TOP_LEVEL = ["schema_version", "source", "detector", "tracker", "lock", "output"]


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping (YAML dict).")
    missing = [k for k in REQUIRED_TOP_LEVEL if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required top-level keys: {missing}")
    return cfg
