"""Deterministic pinhole camera model for image-plane bearing estimates."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


BBox = Tuple[float, float, float, float]


def bbox_to_center(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
    """Return the center of an xyxy bbox."""
    _validate_bbox((x1, y1, x2, y2))
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


@dataclass(frozen=True)
class PinholeCameraModel:
    """
    Minimal pinhole camera model.

    Coordinate convention:
      - positive bearing/yaw error means target is right of optical center;
      - positive elevation error means target is above optical center.

    If vertical FOV is unavailable in FOV mode, fy is set equal to fx. This is
    a conservative bench-replay approximation that avoids inventing calibration
    and should be replaced by explicit intrinsics for flight use.
    """
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float

    def __post_init__(self) -> None:
        for name, value in (
            ("fx_px", self.fx_px),
            ("fy_px", self.fy_px),
            ("cx_px", self.cx_px),
            ("cy_px", self.cy_px),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.fx_px <= 0.0 or self.fy_px <= 0.0:
            raise ValueError("fx_px and fy_px must be > 0")

    @classmethod
    def from_config(
        cls,
        camera_cfg: dict,
        *,
        frame_width_px: Optional[int] = None,
        frame_height_px: Optional[int] = None,
    ) -> "PinholeCameraModel":
        mode = str(camera_cfg.get("mode", "fov")).lower().strip()
        width_value = camera_cfg.get("frame_width_px")
        height_value = camera_cfg.get("frame_height_px")
        width = _positive_dimension(
            frame_width_px if width_value is None else width_value,
            "frame_width_px",
        )
        height = _positive_dimension(
            frame_height_px if height_value is None else height_value,
            "frame_height_px",
        )

        if mode == "intrinsics":
            fx = _positive_float(camera_cfg.get("fx_px"), "fx_px")
            fy = _positive_float(camera_cfg.get("fy_px"), "fy_px")
        elif mode == "fov":
            hfov_deg = _positive_float(camera_cfg.get("horizontal_fov_deg"), "horizontal_fov_deg")
            hfov_rad = math.radians(hfov_deg)
            fx = width / (2.0 * math.tan(hfov_rad * 0.5))
            vfov = camera_cfg.get("vertical_fov_deg")
            if vfov is None:
                fy = fx
            else:
                vfov_rad = math.radians(_positive_float(vfov, "vertical_fov_deg"))
                fy = height / (2.0 * math.tan(vfov_rad * 0.5))
        else:
            raise ValueError("guidance.camera.mode must be 'intrinsics' or 'fov'")

        assume_center = bool(camera_cfg.get("assume_principal_point_center", True))
        cx_cfg = camera_cfg.get("cx_px")
        cy_cfg = camera_cfg.get("cy_px")
        if cx_cfg is None:
            if not assume_center:
                raise ValueError("cx_px is required when assume_principal_point_center=false")
            cx = width * 0.5
        else:
            cx = _finite_float(cx_cfg, "cx_px")
        if cy_cfg is None:
            if not assume_center:
                raise ValueError("cy_px is required when assume_principal_point_center=false")
            cy = height * 0.5
        else:
            cy = _finite_float(cy_cfg, "cy_px")

        return cls(fx_px=fx, fy_px=fy, cx_px=cx, cy_px=cy)

    def pixel_to_normalized_ray(self, u: float, v: float) -> Tuple[float, float, float]:
        """Return an unnormalized camera ray where z=1."""
        _validate_pixel(u, v)
        return ((float(u) - self.cx_px) / self.fx_px, (float(v) - self.cy_px) / self.fy_px, 1.0)

    def pixel_error_to_angles(
        self,
        u: float,
        v: float,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[float, float]:
        """
        Convert a pixel location to bearing/elevation errors in radians.

        The frame dimensions are validated to catch bad pipeline state. The
        active principal point remains the calibrated cx/cy for this model.
        """
        _positive_dimension(frame_width, "frame_width")
        _positive_dimension(frame_height, "frame_height")
        _validate_pixel(u, v)
        bearing_rad = math.atan((float(u) - self.cx_px) / self.fx_px)
        elevation_rad = -math.atan((float(v) - self.cy_px) / self.fy_px)
        return bearing_rad, elevation_rad

    def bbox_to_bearing_elevation(
        self,
        bbox: Iterable[float],
        frame_shape: Tuple[int, int],
    ) -> Tuple[float, float]:
        x1, y1, x2, y2 = _bbox_tuple(bbox)
        u, v = bbox_to_center(x1, y1, x2, y2)
        height, width = frame_shape
        return self.pixel_error_to_angles(u, v, width, height)

    def frame_center(self) -> List[float]:
        return [float(self.cx_px), float(self.cy_px)]


def _bbox_tuple(bbox: Iterable[float]) -> BBox:
    vals = tuple(float(v) for v in bbox)
    if len(vals) != 4:
        raise ValueError("bbox must contain four xyxy values")
    _validate_bbox(vals)
    return vals  # type: ignore[return-value]


def _validate_bbox(bbox: BBox) -> None:
    x1, y1, x2, y2 = bbox
    if not all(math.isfinite(v) for v in bbox):
        raise ValueError("bbox values must be finite")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive width and height")


def _validate_pixel(u: float, v: float) -> None:
    if not math.isfinite(float(u)) or not math.isfinite(float(v)):
        raise ValueError("pixel coordinates must be finite")


def _positive_dimension(value: object, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required")
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be > 0")
    return out


def _positive_float(value: object, name: str) -> float:
    out = _finite_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return out


def _finite_float(value: object, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out
