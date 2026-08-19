from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class CameraFrame:
    color: Any
    depth_mm: Any
    intrinsics: CameraIntrinsics
    timestamp: float = field(default_factory=time)


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]


@dataclass
class OrientationEstimate:
    angle_deg: float = 0.0
    valid: bool = False
    confidence: float = 0.0
    period_deg: float = 180.0
    reason: str = ""


@dataclass
class TargetResult:
    detection: Detection
    height_mm: float
    camera_xyz_mm: tuple[float, float, float]
    base_pose: list[float]
    angle_deg: float = 0.0
    angle_valid: bool = True
    angle_confidence: float = 1.0
    angle_reason: str = ""
    grasp_pixel: tuple[int, int] | None = None
    pose_point_count: int = 0
