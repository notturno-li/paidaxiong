from __future__ import annotations

from math import cos, radians, sin
from pathlib import Path

import numpy as np

from .types import CameraIntrinsics


def euler_zyx_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = map(radians, [rx_deg, ry_deg, rz_deg])
    rx_m = np.array([[1, 0, 0], [0, cos(rx), -sin(rx)], [0, sin(rx), cos(rx)]], dtype=float)
    ry_m = np.array([[cos(ry), 0, sin(ry)], [0, 1, 0], [-sin(ry), 0, cos(ry)]], dtype=float)
    rz_m = np.array([[cos(rz), -sin(rz), 0], [sin(rz), cos(rz), 0], [0, 0, 1]], dtype=float)
    return rz_m @ ry_m @ rx_m


def pose_to_matrix(pose: list[float] | tuple[float, ...]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = euler_zyx_to_matrix(float(pose[3]), float(pose[4]), float(pose[5]))
    matrix[:3, 3] = [float(pose[0]), float(pose[1]), float(pose[2])]
    return matrix


def matrix_to_pose_xyz_keep_rpy(matrix: np.ndarray, rpy: list[float] | tuple[float, float, float]) -> list[float]:
    return [float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3]), float(rpy[0]), float(rpy[1]), float(rpy[2])]


class CoordinateTransformer:
    def __init__(self, config: dict):
        self.config = config
        self.t_camera_to_gripper = self._load_hand_eye_matrix()

    def _load_hand_eye_matrix(self) -> np.ndarray:
        raw = self.config["calibration"].get("transform_camera_to_gripper")
        if raw is not None:
            try:
                arr = np.array(raw, dtype=float)
                if arr.shape == (4, 4):
                    return arr
            except Exception:
                pass
        fallback = np.eye(4)
        yaml_path = Path(self.config["calibration"].get("hand_eye_yaml", ""))
        if not yaml_path.exists():
            yaml_path = Path(__file__).resolve().parents[2] / yaml_path
        if not yaml_path.exists():
            return fallback
        text = yaml_path.read_text(encoding="utf-8", errors="ignore")
        marker = "data: ["
        if marker not in text:
            return fallback
        data_part = text.split(marker, 1)[1].split("]", 1)[0]
        values = [float(item.strip()) for item in data_part.replace("\n", " ").split(",") if item.strip()]
        if len(values) != 16:
            return fallback
        return np.array(values, dtype=float).reshape(4, 4)

    def pixel_to_camera(self, u: int, v: int, depth_mm: float, intrinsics: CameraIntrinsics) -> tuple[float, float, float]:
        z = float(depth_mm)
        x = (float(u) - intrinsics.cx) * z / intrinsics.fx
        y = (float(v) - intrinsics.cy) * z / intrinsics.fy
        return x, y, z

    def camera_to_base_matrix(self, flange_pose: list[float]) -> np.ndarray:
        t_base_to_gripper = pose_to_matrix(flange_pose)
        return t_base_to_gripper @ self.t_camera_to_gripper

    def camera_to_base_pose(self, camera_xyz_mm: tuple[float, float, float], flange_pose: list[float], target_rpy: list[float] | None = None) -> list[float]:
        t_base_to_gripper = pose_to_matrix(flange_pose)
        point_camera = np.array([[camera_xyz_mm[0]], [camera_xyz_mm[1]], [camera_xyz_mm[2]], [1.0]])
        point_base = t_base_to_gripper @ self.t_camera_to_gripper @ point_camera
        pose_matrix = np.eye(4)
        pose_matrix[:3, 3] = point_base[:3, 0]
        rpy = target_rpy if target_rpy is not None else [float(flange_pose[3]), float(flange_pose[4]), float(flange_pose[5])]
        return matrix_to_pose_xyz_keep_rpy(pose_matrix, rpy)
