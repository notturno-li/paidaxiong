from __future__ import annotations

import importlib.util
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import load_config
from app.core.action_sequence import ActionSequencePlanner
from app.core.model_profiles import load_model_profile
from app.core.vision_studio_client import DobotVisionStudioClient
from app.core.vision_task import VisionTaskProcessor
from app.modules import build_standard_registry


EXPECTED_ROBOT_IP = "192.168.5.1"
EXPECTED_DASHBOARD_PORT = 29999


@dataclass(frozen=True)
class PreflightCheck:
    status: str
    name: str
    detail: str


@dataclass(frozen=True)
class PreflightReport:
    checks: tuple[PreflightCheck, ...]

    @property
    def failures(self) -> int:
        return sum(check.status == "FAIL" for check in self.checks)

    @property
    def warnings(self) -> int:
        return sum(check.status == "WARN" for check in self.checks)

    @property
    def passed(self) -> int:
        return sum(check.status == "PASS" for check in self.checks)

    @property
    def ready(self) -> bool:
        return self.failures == 0


def run_preflight(
    config_path: str | Path,
    *,
    hardware: bool = False,
    root: str | Path | None = None,
) -> PreflightReport:
    config_path = Path(config_path).resolve()
    root_path = Path(root).resolve() if root else config_path.parent.parent
    checks: list[PreflightCheck] = []

    required_modules = {
        "numpy": "numpy",
        "OpenCV": "cv2",
        "PyQt5": "PyQt5",
        "PyYAML": "yaml",
        "Ultralytics": "ultralytics",
        "RealSense": "pyrealsense2",
        "PySerial": "serial",
    }
    missing = [label for label, module in required_modules.items() if importlib.util.find_spec(module) is None]
    _add(
        checks,
        not missing,
        "Python 依赖",
        "全部已安装" if not missing else "缺少: " + ", ".join(missing),
    )

    try:
        config = load_config(config_path)
        checks.append(PreflightCheck("PASS", "配置加载", str(config_path)))
    except Exception as exc:
        checks.append(PreflightCheck("FAIL", "配置加载", str(exc)))
        return PreflightReport(tuple(checks))

    robot = config.get("robot", {})
    endpoint_ok = (
        str(robot.get("ip")) == EXPECTED_ROBOT_IP
        and int(robot.get("dashboard_port", 0)) == EXPECTED_DASHBOARD_PORT
    )
    _add(
        checks,
        endpoint_ok,
        "机械臂固定端点",
        f"{robot.get('ip')}:{robot.get('dashboard_port')}；要求 {EXPECTED_ROBOT_IP}:{EXPECTED_DASHBOARD_PORT}",
    )
    _check_numeric_safety(checks, robot)
    _check_assembly_configuration(checks, config)
    _check_files(checks, config, root_path)
    _check_model_configuration(checks, config, root_path)
    _check_calibration(checks, config)
    _check_profiles(checks, config)
    _check_action_sequences(checks, config)
    _check_recipes(checks, config)
    _check_cycle(checks, config)
    _check_dvs_configuration(checks, config, hardware)
    if hardware:
        _probe_robot_dashboard(checks, robot)
    else:
        checks.append(
            PreflightCheck("WARN", "硬件连通性", "离线自检未连接机械臂和 DVS；比赛接线后再运行硬件自检")
        )
    return PreflightReport(tuple(checks))


def _add(checks: list[PreflightCheck], ok: bool, name: str, detail: str) -> None:
    checks.append(PreflightCheck("PASS" if ok else "FAIL", name, detail))


def _check_numeric_safety(checks: list[PreflightCheck], robot: dict[str, Any]) -> None:
    try:
        minimum_z = float(robot.get("min_grasp_z_mm"))
        safe_z = float(robot.get("safe_z_mm"))
        speed = float(robot.get("speed_percent"))
        descend = float(robot.get("descend_speed_percent"))
        home = [float(value) for value in robot.get("home_pose", [])]
        ok = (
            minimum_z > 0
            and safe_z >= minimum_z
            and 0 < speed <= 100
            and 0 < descend <= speed
            and len(home) == 6
            and home[2] >= minimum_z
        )
        detail = (
            f"minZ={minimum_z:g}, safeZ={safe_z:g}, speed={speed:g}%, "
            f"descend={descend:g}%, homeZ={home[2]:g}"
        )
    except (TypeError, ValueError, IndexError) as exc:
        ok, detail = False, f"安全参数格式错误: {exc}"
    _add(checks, ok, "机器人安全参数", detail)


def _check_assembly_configuration(checks: list[PreflightCheck], config: dict[str, Any]) -> None:
    robot = config.get("robot", {})
    workflow = config.get("workflow", {})
    try:
        minimum_z = float(robot.get("min_grasp_z_mm"))
        photo = robot.get("photo_pose")
        photo_required = bool(workflow.get("photo_pose_before_detection", False))
        photo_ok = (
            not photo_required
            or isinstance(photo, (list, tuple))
            and len(photo) == 6
            and float(photo[2]) >= minimum_z
        )
        _add(
            checks,
            photo_ok,
            "拍照位姿",
            "未启用拍照位运动" if not photo_required else str(photo),
        )
    except (TypeError, ValueError, IndexError) as exc:
        _add(checks, False, "拍照位姿", f"格式错误: {exc}")

    classes = config.get("model", {}).get("class_names", [])
    assembly = config.get("assembly", {})
    mode = str(assembly.get("target_mode", "fixed")).lower()
    bins = robot.get("bins", {})
    slots = assembly.get("slots", {})
    failures: list[str] = []
    if mode not in {"fixed", "tray_frame"}:
        failures.append(f"不支持的 target_mode={mode}")
    if not isinstance(bins, dict):
        bins = {}
    if not isinstance(slots, dict):
        slots = {}
    if mode == "tray_frame":
        frame = assembly.get("frame_pose_base")
        if not isinstance(frame, (list, tuple)) or len(frame) != 6:
            failures.append("tray_frame 缺少有效 frame_pose_base")
    for label in classes if isinstance(classes, list) else []:
        slot = slots.get(label, {}) if isinstance(slots.get(label, {}), dict) else {}
        raw = slot.get("tcp_pose")
        if raw is None and mode == "tray_frame":
            raw = slot.get("pose_tray")
        if raw is None:
            raw = bins.get(label)
        if isinstance(raw, dict):
            raw = raw.get("pose") or raw.get("tcp_pose")
        if not isinstance(raw, (list, tuple)) or len(raw) != 6:
            failures.append(f"{label} 缺少6轴装配位姿")
    _add(
        checks,
        not failures,
        "装配槽位配置",
        "全部类别均有有效目标位姿" if not failures else "; ".join(failures),
    )


def _resolve_project_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _check_files(checks: list[PreflightCheck], config: dict[str, Any], root: Path) -> None:
    model = config.get("model", {})
    for label, key in (("主模型", "weights"), ("备用模型", "fallback_weights")):
        path = _resolve_project_path(root, model.get(key, ""))
        _add(checks, path.is_file() and path.stat().st_size > 0, label, str(path))
    calibration_path = _resolve_project_path(
        root, config.get("calibration", {}).get("hand_eye_yaml", "")
    )
    _add(
        checks,
        calibration_path.is_file() and calibration_path.stat().st_size > 0,
        "手眼标定文件",
        str(calibration_path),
    )


def _check_model_configuration(
    checks: list[PreflightCheck], config: dict[str, Any], root: Path
) -> None:
    model = config.get("model", {})
    classes = model.get("class_names")
    classes_ok = (
        isinstance(classes, list)
        and bool(classes)
        and all(str(value).strip() for value in classes)
        and len(classes) == len(set(str(value).casefold() for value in classes))
    )
    _add(
        checks,
        classes_ok,
        "模型类别配置",
        ", ".join(str(value) for value in classes) if classes_ok else "类别不能为空或重复",
    )

    weights = _resolve_project_path(root, model.get("weights", "")).resolve()
    field_models = (root / "models" / "field_models").resolve()
    try:
        weights.relative_to(field_models)
        is_field_model = True
    except ValueError:
        is_field_model = False
    if not is_field_model:
        checks.append(PreflightCheck("PASS", "现场模型 Profile", "当前使用内置/旧版模型"))
        return
    try:
        profile = load_model_profile(weights.parent / "model_profile.yaml", root)
        if tuple(str(value) for value in classes) != profile.class_names:
            raise RuntimeError("现场配置类别顺序与模型 Profile 不一致")
        if weights != profile.weights_path.resolve():
            raise RuntimeError("现场配置权重与模型 Profile 不一致")
        checks.append(PreflightCheck("PASS", "现场模型 Profile", profile.name))
    except Exception as exc:
        checks.append(PreflightCheck("FAIL", "现场模型 Profile", str(exc)))


def _check_calibration(checks: list[PreflightCheck], config: dict[str, Any]) -> None:
    matrix = config.get("calibration", {}).get("transform_camera_to_gripper")
    ok = (
        isinstance(matrix, list)
        and len(matrix) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in matrix)
    )
    _add(checks, ok, "手眼矩阵", "4x4 矩阵" if ok else "必须是 4x4 数值矩阵")


def _check_profiles(checks: list[PreflightCheck], config: dict[str, Any]) -> None:
    settings = config.get("vision_tasks", {})
    profiles = settings.get("profiles", {})
    active = str(settings.get("active_profile") or "")
    try:
        if active not in profiles:
            raise RuntimeError(f"活动 Profile 不存在: {active}")
        VisionTaskProcessor(profiles[active])
        checks.append(PreflightCheck("PASS", "活动视觉任务", active))
    except Exception as exc:
        checks.append(PreflightCheck("FAIL", "活动视觉任务", str(exc)))


def _check_action_sequences(checks: list[PreflightCheck], config: dict[str, Any]) -> None:
    settings = config.get("action_sequences", {})
    try:
        planner = ActionSequencePlanner(settings, config.get("robot", {}))
        cycle = config.get("dvs_cycle", {})
        names = [
            str(settings.get("active_profile") or ""),
            str(cycle.get("before_each_sequence") or ""),
            str(cycle.get("after_each_sequence") or ""),
        ]
        planned: list[str] = []
        for name in dict.fromkeys(item for item in names if item):
            profile = settings.get("profiles", {}).get(name, {})
            if profile.get("steps"):
                planner.plan(name)
                planned.append(name)
            elif name in names[1:]:
                raise RuntimeError(f"连续任务引用了空动作序列: {name}")
        detail = "已验证: " + ", ".join(planned) if planned else "当前未启用固定动作序列"
        checks.append(PreflightCheck("PASS", "固定动作预检", detail))
    except Exception as exc:
        checks.append(PreflightCheck("FAIL", "固定动作预检", str(exc)))


def _check_recipes(checks: list[PreflightCheck], config: dict[str, Any]) -> None:
    registry = build_standard_registry()
    recipes = config.get("recipes", {})
    failures: list[str] = []
    for recipe_id, recipe in recipes.items():
        try:
            modules = recipe.get("modules", []) if isinstance(recipe, dict) else recipe
            registry.resolve(modules, include_dependencies=True)
        except Exception as exc:
            failures.append(f"{recipe_id}: {exc}")
    _add(
        checks,
        not failures,
        "模块配方",
        f"{len(recipes)} 个配方有效" if not failures else "；".join(failures),
    )


def _check_cycle(checks: list[PreflightCheck], config: dict[str, Any]) -> None:
    cycle = config.get("dvs_cycle", {})
    motion = bool(cycle.get("execute_motion")) or bool(cycle.get("before_each_sequence")) or bool(
        cycle.get("after_each_sequence")
    )
    ok = (
        1 <= int(cycle.get("max_items", 0)) <= 100
        and float(cycle.get("delay_s", -1)) >= 0
        and str(cycle.get("processor", ""))
        in {"none", "status", "measurement", "status_measurement"}
        and (not motion or str(cycle.get("on_error")) == "stop")
    )
    _add(
        checks,
        ok,
        "连续任务安全策略",
        f"max={cycle.get('max_items')}, processor={cycle.get('processor')}, motion={motion}, on_error={cycle.get('on_error')}",
    )


def _check_dvs_configuration(
    checks: list[PreflightCheck], config: dict[str, Any], hardware: bool
) -> None:
    dvs = config.get("vision_studio", {})
    mode = str(dvs.get("mode", "disabled"))
    if mode == "disabled":
        checks.append(PreflightCheck("WARN", "DVS 通信", "尚未配置；拿到现场 IP/端口/COM 后用通信配置填写"))
        return
    if mode == "serial":
        endpoint = str(dvs.get("serial", {}).get("port") or "")
        valid = bool(endpoint) and int(dvs.get("serial", {}).get("baudrate", 0)) > 0
    else:
        tcp = dvs.get("tcp", {})
        host = str(tcp.get("host") or "")
        port = tcp.get("port")
        endpoint = f"{host}:{port}"
        valid = bool(host or mode == "tcp_server") and isinstance(port, int) and 1 <= port <= 65535
    if not valid:
        checks.append(PreflightCheck("FAIL", "DVS 通信配置", f"{mode} 端点不完整: {endpoint}"))
        return
    checks.append(PreflightCheck("PASS", "DVS 通信配置", f"{mode} {endpoint}"))
    if not hardware:
        return
    client = DobotVisionStudioClient(config, simulation=False)
    try:
        client.connect()
        checks.append(PreflightCheck("PASS", "DVS 硬件连接", f"已连接 {mode} {endpoint}，未发送业务报文"))
    except Exception as exc:
        checks.append(PreflightCheck("FAIL", "DVS 硬件连接", str(exc)))
    finally:
        client.close()


def _probe_robot_dashboard(checks: list[PreflightCheck], robot: dict[str, Any]) -> None:
    endpoint = (str(robot.get("ip")), int(robot.get("dashboard_port")))
    try:
        connection = socket.create_connection(endpoint, timeout=float(robot.get("connect_timeout_s", 2)))
        connection.close()
        checks.append(PreflightCheck("PASS", "机械臂网络", f"{endpoint[0]}:{endpoint[1]} TCP 可达；未使能、未运动"))
    except OSError as exc:
        checks.append(PreflightCheck("FAIL", "机械臂网络", str(exc)))
