from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "mode": "simulation",
    "camera": {"width": 640, "height": 480, "fps": 30, "min_valid_depth_mm": 150, "max_valid_depth_mm": 900},
    "model": {"weights": "models/fruit_best.pt", "fallback_weights": "models/yolov8s.pt", "conf_threshold": 0.45, "class_names": ["apple", "banana", "grape", "strawberry"]},
    "robot": {
        "ip": "192.168.5.1", "dashboard_port": 29999, "connect_timeout_s": 2.0,
        "user": 0, "tool": 0, "speed_percent": 25, "accel_percent": 25,
        "safe_z_mm": 180.0, "grasp_clearance_mm": 18.0, "suction_tool_do": 1,
        "min_grasp_z_mm": 60.0, "descend_speed_percent": 8, "descend_accel_percent": 10,
        "grasp_x_offset_mm": 0.0, "grasp_y_offset_mm": 0.0, "grasp_z_offset_mm": 0.0,
        "suction_settle_s": 0.5,
        "suction_release_settle_s": 0.3,
        "motion_done_timeout_s": 30.0,
        "motion_done_poll_s": 0.1,
        "motion_done_xyz_tol_mm": 2.0,
        "motion_done_rpy_tol_deg": 3.0,
        "suction_io_type": "tool_do",
        "home_pose": [260.0, 0.0, 180.0, 180.0, 0.0, 0.0],
        "fixed_test_pose": [240.0, -80.0, 160.0, 180.0, 0.0, 0.0],
        "bins": {
            "apple": [210.0, 155.0, 130.0, 180.0, 0.0, 0.0],
            "banana": [280.0, 155.0, 130.0, 180.0, 0.0, 0.0],
            "strawberry": [210.0, 225.0, 130.0, 180.0, 0.0, 0.0],
            "grape": [280.0, 225.0, 130.0, 180.0, 0.0, 0.0],
        },
    },
    "calibration": {
        "hand_eye_yaml": "configs/20260610.yaml",
        "camera_matrix": {"fx": 615.0, "fy": 615.0, "cx": 320.0, "cy": 240.0},
        "transform_camera_to_gripper": [
            [0.7218742278, 0.6919286206, -0.0115057948, -72.7322748581],
            [-0.6919621455, 0.7219324330, 0.0013969538, -38.3275375922],
            [0.0092729987, 0.0069531495, 0.9999328303, 24.3234046605],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    "height": {
        "table_depth_mm": 420.0, "roi_ratio": 0.35, "min_height_mm": 15.0, "max_height_mm": 60.0, "smoothing_window": 5,
        "plane_inlier_mm": 8.0, "plane_ransac_iters": 120, "plane_sample_points": 4000,
        "plane_top_percentile": 70.0,
    },
    "workflow": {"auto_max_objects": 4, "auto_empty_frames_to_finish": 8, "command_json_log": True},
}


def deep_update(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path is None:
        path = Path(__file__).resolve().parents[1] / "configs" / "competition.yaml"
    path = Path(path)
    if not path.exists():
        return config
    try:
        import yaml
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            deep_update(config, loaded)
    except Exception:
        pass
    return config
