"""Factory for mocked bridge components."""
from __future__ import annotations

from typing import Any, Dict, Optional

from .mock_guidance_bridge import MockGuidanceBridge


def build_mock_guidance_bridge(
    mock_bridge_cfg: Dict[str, Any],
    *,
    guidance_camera_cfg: Dict[str, Any],
    calibration_id: str,
    run_id: Optional[str],
) -> MockGuidanceBridge:
    return MockGuidanceBridge(
        run_id=run_id,
        calibration_id=calibration_id,
        calibration_reviewed=bool(guidance_camera_cfg.get("calibration_reviewed", False)),
        require_reviewed_calibration=bool(
            mock_bridge_cfg.get("require_reviewed_calibration", True)
        ),
        allowed_lock_states=mock_bridge_cfg.get("allowed_lock_states", ["LOCKED", "STRIKE_READY"]),
        require_guidance_hint_valid=bool(
            mock_bridge_cfg.get("require_guidance_hint_valid", True)
        ),
        max_abs_yaw_rate_deg_s=float(mock_bridge_cfg["max_abs_yaw_rate_deg_s"]),
    )
