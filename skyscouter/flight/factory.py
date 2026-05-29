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
    ok = link.start()
    # In LIVE mode (real serial + real FC commands), a failed connect must be a
    # hard abort — silently continuing without a FC link means the drone never
    # takes off while the operator assumes it will.  The pipeline catches this
    # RuntimeError, logs "Pipeline failed: ..." and exits cleanly.
    # Dry-run mode never opens serial, so start() always succeeds there.
    if not ok and not link.dry_run:
        raise RuntimeError(
            f"FC LIVE mode: could not connect to flight controller — "
            f"{link.status_text()}. "
            "Check the USB cable, FC power, and that /dev/ttyACM0 appears in "
            "`ls /dev/ttyACM*` before running."
        )
    return link
