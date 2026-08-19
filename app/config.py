from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "mode": "simulation",
    "competition": {
        "name": "2026 睿抗机器人大赛机器视觉系统创新赛",
        "level": "provincial",
        "profile": "provincial_7_tasks",
    },
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
    "command_transport": {
        "mode": "preview",
        "host": "192.168.5.1",
        "port": 10000,
        "connect_timeout_s": 2.0,
        "reply_timeout_s": 3.0,
        "terminator": "\n",
    },
    "vision_studio": {
        "mode": "disabled",
        "encoding": "utf-8",
        "connect_timeout_s": 2.0,
        "reply_timeout_s": 3.0,
        "send_terminator": "",
        "receive_terminator": "",
        "max_reply_bytes": 65536,
        "serial": {
            "port": None,
            "baudrate": 115200,
            "data_bits": 8,
            "parity": "none",
            "stop_bits": 1,
        },
        "tcp": {"host": None, "port": None},
        "protocol": {
            "send_before_receive": True,
            "exchange_command": None,
            "delimiter": ",",
            "response_fields": [],
            "global_values": {},
        },
        "simulation_result": {"ok": True},
    },
    "modules": {
        "auto_dependencies": True,
        "default_recipe": "national_auto"
    },
    "recipes": {},
    "workflow": {"auto_max_objects": 4, "auto_empty_frames_to_finish": 8, "command_json_log": True},
    "orientation": {
        "enabled": True,
        "angle_frame": "base",
        "require_valid": False,
        "bbox_expand_ratio": 0.12,
        "min_depth_points": 30,
        "min_object_points": 25,
        "min_height_above_plane_mm": 3.0,
        "max_height_above_plane_mm": 600.0,
        "min_anisotropy": 0.20,
        "classes": {},
    },
    "assembly": {"target_mode": "fixed", "frame_pose_base": None, "slots": {}},
}


def deep_update(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    for key, value in extra.items():
        if isinstance(value, dict) and value.get("__replace__") is True:
            # Keep the marker through nested extends resolution. load_config removes
            # it only after the resolved document has replaced DEFAULT_CONFIG too.
            base[key] = deepcopy(value)
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_yaml_with_extends(path: Path, loading: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    loading = set() if loading is None else loading
    if path in loading:
        chain = " -> ".join(str(item) for item in (*loading, path))
        raise RuntimeError(f"配置 extends 存在循环引用: {chain}")
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("未安装 PyYAML，无法读取比赛配置") from exc

    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise RuntimeError(f"配置文件顶层必须是映射: {path}")

    parent_name = loaded.pop("extends", None)
    if not parent_name:
        return loaded
    parent_path = Path(parent_name)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    loading.add(path)
    try:
        parent = _load_yaml_with_extends(parent_path, loading)
    finally:
        loading.remove(path)
    return deep_update(parent, loaded)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path is None:
        path = Path(__file__).resolve().parents[1] / "configs" / "competition.yaml"
    path = Path(path)
    if not path.exists():
        return config
    try:
        deep_update(config, _load_yaml_with_extends(path))
    except (FileNotFoundError, RuntimeError):
        raise
    except Exception as exc:
        raise RuntimeError(f"读取配置失败: {path}: {exc}") from exc
    _strip_replace_markers(config)
    config["_config_path"] = str(path.resolve())
    return config


def _strip_replace_markers(value: Any) -> None:
    if isinstance(value, dict):
        value.pop("__replace__", None)
        for child in value.values():
            _strip_replace_markers(child)
    elif isinstance(value, list):
        for child in value:
            _strip_replace_markers(child)


def update_yaml_config(
    path: str | Path,
    updates: dict[str, Any],
    *,
    extends: str | None = None,
) -> Path:
    """Merge and atomically persist a small field override configuration."""
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("未安装 PyYAML，无法保存现场配置") from exc

    destination = Path(path)
    existing: dict[str, Any] = {}
    if destination.exists():
        loaded = yaml.safe_load(destination.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise RuntimeError(f"现场配置文件顶层必须是映射: {destination}")
        existing = loaded
    if extends is not None:
        existing["extends"] = extends
    deep_update(existing, deepcopy(updates))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(existing, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination.resolve()
