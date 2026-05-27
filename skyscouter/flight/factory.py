"""Factory for the optional MAVLink flight-control link."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .mavlink_link import MavlinkFlightLink


def build_mavlink_flight_link(
    cfg: Dict[str, Any],
    *,
    output_dir: Path,
    run_id: Optional[str],
) -> Optional[MavlinkFlightLink]:
    """Build a MavlinkFlightLink from a top-level config dict.

    Returns None when flight_control is absent or disabled.  Same pattern as
    build_gimbal_follow_controller -- the pipeline can be wired the same way
    for both sinks.
    """
    flight_cfg = dict(cfg.get("flight_control", {}))
    if not bool(flight_cfg.get("enabled", False)):
        return None
    link = MavlinkFlightLink(
        cfg=flight_cfg,
        output_dir=output_dir,
        run_id=run_id,
    )
    link.start()
    return link
