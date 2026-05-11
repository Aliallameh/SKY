"""Factory for optional gimbal follow control."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .follow_controller import GimbalFollowController


def build_gimbal_follow_controller(
    cfg: Dict[str, Any],
    *,
    output_dir: Path,
    run_id: Optional[str],
) -> Optional[GimbalFollowController]:
    follow_cfg = dict(cfg.get("gimbal_follow", {}))
    if not bool(follow_cfg.get("enabled", False)):
        return None
    return GimbalFollowController(
        cfg=follow_cfg,
        output_dir=output_dir,
        run_id=run_id,
    )

