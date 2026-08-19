"""Shared metadata and validation helpers for hand-eye calibration data."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


METADATA_NAME = "camera_intrinsics.yaml"


class CalibrationMetadataError(RuntimeError):
    pass


def metadata_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / METADATA_NAME


def build_metadata(intr: Any, width: int, height: int, fps: int, serial_number: str | None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stream": "color",
        "image_size": [int(width), int(height)],
        "fps": int(fps),
        "source": "realsense_device",
        "serial_number": serial_number,
        "camera_matrix": {
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "cx": float(intr.ppx),
            "cy": float(intr.ppy),
        },
        "dist_coeffs": [float(value) for value in intr.coeffs],
    }


def write_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    metadata_file = Path(path)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.write_text(
        yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def load_metadata(path: str | Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    metadata_file = Path(path)
    if not metadata_file.is_file():
        raise RuntimeError(
            f"缺少本批次相机内参文件: {metadata_file}; "
            "请归档旧 runs/calib_data 后重新运行 calib_collect_auto.py"
        )
    raw = yaml.safe_load(metadata_file.read_text(encoding="utf-8")) or {}
    if raw.get("image_size") != [int(width), int(height)]:
        raise RuntimeError(
            f"标定内参分辨率 {raw.get('image_size')} 与手眼图像 {width}x{height} 不一致"
        )
    matrix = raw.get("camera_matrix") or {}
    try:
        camera_matrix = np.array(
            [[float(matrix["fx"]), 0.0, float(matrix["cx"])],
             [0.0, float(matrix["fy"]), float(matrix["cy"])],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.asarray(raw["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"标定内参文件格式错误: {metadata_file}: {exc}") from exc
    if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0 or not np.isfinite(camera_matrix).all():
        raise RuntimeError(f"标定内参数值无效: {metadata_file}")
    return camera_matrix, dist_coeffs, raw


def validate_existing_images(image_dir: str | Path, width: int, height: int) -> None:
    directory = Path(image_dir)
    for image_path in sorted(directory.glob("*.jpg")):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"无法读取已有标定图像: {image_path}")
        actual_height, actual_width = image.shape[:2]
        if (actual_width, actual_height) != (int(width), int(height)):
            raise RuntimeError(
                f"已有标定图像 {image_path.name} 是 {actual_width}x{actual_height}，"
                f"当前要求 {width}x{height}; 请先在手眼工具中选择归档旧数据"
            )


def validate_device_against_metadata(
    metadata: dict[str, Any], current: dict[str, Any], *, tolerance_px: float = 0.2
) -> None:
    saved_serial = metadata.get("serial_number")
    current_serial = current.get("serial_number")
    if saved_serial and current_serial and saved_serial != current_serial:
        raise CalibrationMetadataError(
            f"标定数据相机序列号 {saved_serial} 与当前相机 {current_serial} 不一致"
        )
    saved = metadata.get("camera_matrix") or {}
    for key in ("fx", "fy", "cx", "cy"):
        if key in saved and abs(float(saved[key]) - float(current["camera_matrix"][key])) > tolerance_px:
            raise CalibrationMetadataError(
                f"当前设备内参 {key} 与采集时不一致，请重新采集手眼数据"
            )
    saved_dist = np.asarray(metadata.get("dist_coeffs") or [], dtype=float)
    current_dist = np.asarray(current.get("dist_coeffs") or [], dtype=float)
    if saved_dist.size and current_dist.size and saved_dist.shape == current_dist.shape:
        if not np.allclose(saved_dist, current_dist, atol=1e-4, rtol=1e-5):
            raise CalibrationMetadataError(
                "当前设备畸变参数与采集时不一致，请重新采集手眼数据"
            )
