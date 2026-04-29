"""
Ego-motion compensation.

PRD FR-TRK-003: tracker shall consume IMU data to compensate for camera
ego-motion before track association.

Architectural note (matches the "decoupled IMU" decision from the v3.2 review):
this module is the seam. Today it returns an identity transform — i.e., does
nothing — but the tracker calls `warp_point(cx, cy, dt)` for every tracked
center before doing association. When a real IMU lands (either dedicated on
the camera mount, or via the flight controller stream), only the
EgoMotionCompensator subclass changes. The tracker code does not.

This keeps the tracker decoupled from the IMU source. The "tight coupling
danger" we discussed is mitigated by isolating the IMU behind this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple


class EgoMotionCompensator(ABC):
    """Abstract base. Subclass to integrate gyro/IMU-driven warping."""

    @abstractmethod
    def warp_point(
        self,
        cx: float,
        cy: float,
        dt_seconds: float,
    ) -> Tuple[float, float]:
        """
        Given an image-space point at the previous frame, return the
        predicted image-space point in the current frame after compensating
        for camera rotation/translation between the two frames.

        For the identity stub, returns (cx, cy) unchanged.
        """
        ...


class IdentityEgoMotion(EgoMotionCompensator):
    """No-op compensator. Used until a real IMU source is wired in."""

    def warp_point(self, cx: float, cy: float, dt_seconds: float) -> Tuple[float, float]:
        return (cx, cy)


# Placeholder for the real implementation; ships in Sprint 3.
#
# class GyroDrivenEgoMotion(EgoMotionCompensator):
#     def __init__(self, imu_source: ImuSource, intrinsics: CameraIntrinsics):
#         ...
#     def warp_point(self, cx, cy, dt_seconds):
#         # 1. Read gyro angular rates over [t-dt, t]
#         # 2. Integrate to relative rotation matrix
#         # 3. Project (cx,cy) ray via intrinsics, rotate, project back
#         ...
