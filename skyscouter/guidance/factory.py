"""Factory for visual bearing guidance components."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from .bearing import BearingGuidanceComputer
from .controller import YawRateController


def build_guidance_computer(
    guidance_cfg: Dict[str, Any],
    *,
    run_id: Optional[str] = None,
    acceptable_class_labels: Optional[Iterable[str]] = None,
    frame_width_px: Optional[int] = None,
    frame_height_px: Optional[int] = None,
) -> BearingGuidanceComputer:
    camera_cfg = dict(guidance_cfg["camera"])
    validity_cfg = dict(guidance_cfg["validity"])
    filtering_cfg = dict(guidance_cfg["filtering"])
    controller_cfg = dict(guidance_cfg["controller"])
    controller = YawRateController(
        enabled=bool(controller_cfg.get("enabled", True)),
        mode=str(controller_cfg.get("mode", "yaw_p")),
        kp_yaw=float(controller_cfg["kp_yaw"]),
        deadband_deg=float(controller_cfg["deadband_deg"]),
        max_yaw_rate_deg_s=float(controller_cfg["max_yaw_rate_deg_s"]),
        max_delta_yaw_rate_deg_s=(
            None
            if controller_cfg.get("max_delta_yaw_rate_deg_s") is None
            else float(controller_cfg["max_delta_yaw_rate_deg_s"])
        ),
    )
    return BearingGuidanceComputer(
        camera_cfg=camera_cfg,
        validity_cfg=validity_cfg,
        filtering_cfg=filtering_cfg,
        controller=controller,
        run_id=run_id,
        acceptable_class_labels=acceptable_class_labels,
        frame_width_px=frame_width_px,
        frame_height_px=frame_height_px,
    )
