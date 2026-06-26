from __future__ import annotations

import re
import socket
import threading
import time
from dataclasses import dataclass


@dataclass
class RobotReply:
    error_id: int
    values: list[str]
    raw: str


class DobotClient:
    def __init__(self, config: dict, simulation: bool = False):
        self.config = config
        self.simulation = simulation
        self.sock: socket.socket | None = None
        self.connected = False
        self.enabled = False
        self.current_pose = list(config["robot"]["home_pose"])
        self.sent_commands: list[str] = []
        self.tool_do_states: dict[int, int] = {}
        self._comm_lock = threading.Lock()
        self.cancel_event = threading.Event()

    def connect(self) -> RobotReply:
        if self.simulation:
            self.connected = True
            return RobotReply(0, [], "0,{},Connect(simulation);")
        self.close()
        robot_cfg = self.config["robot"]
        self.sock = socket.create_connection((robot_cfg["ip"], int(robot_cfg["dashboard_port"])), timeout=float(robot_cfg["connect_timeout_s"]))
        self.connected = True
        return RobotReply(0, [], "0,{},Connect();")

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None
        self.connected = False
        self.enabled = False

    def send(self, command: str) -> RobotReply:
        with self._comm_lock:
            self.sent_commands.append(command)
            if self.simulation:
                lower = command.lower()
                if lower.startswith("enablerobot"):
                    self.enabled = True
                if lower.startswith("disablerobot") or lower.startswith("emergencystop"):
                    self.enabled = False
                if lower.startswith(("stop", "pause")):
                    return RobotReply(0, [], f"0,{{}},{command};")
                if lower.startswith(("movl", "movj")):
                    pose = self._parse_pose(command)
                    if pose:
                        self.current_pose = pose
                    return RobotReply(0, [str(len(self.sent_commands))], f"0,{{{len(self.sent_commands)}}},{command};")
                if lower.startswith("tooldoinstant"):
                    parsed = self._parse_tool_do(command)
                    if parsed is not None:
                        index, status = parsed
                        self.tool_do_states[index] = status
                    return RobotReply(0, [str(len(self.sent_commands))], f"0,{{{len(self.sent_commands)}}},{command};")
                if lower.startswith("gettooldo"):
                    index = self._parse_single_index(command)
                    value = self.tool_do_states.get(index, 0) if index is not None else 0
                    return RobotReply(0, [str(value)], f"0,{{{value}}},{command};")
                if lower.startswith("getpose"):
                    values = [f"{value:.3f}" for value in self.current_pose]
                    return RobotReply(0, values, f"0,{{{','.join(values)}}},{command};")
                if lower.startswith("getangle"):
                    values = ["0.000", "-12.000", "38.000", "0.000", "52.000", "0.000"]
                    return RobotReply(0, values, f"0,{{{','.join(values)}}},{command};")
                return RobotReply(0, [], f"0,{{}},{command};")
            if not self.connected or self.sock is None:
                raise RuntimeError("机器人未连接")
            try:
                self.sock.sendall((command + "\n").encode("utf-8"))
                raw = self.sock.recv(4096).decode("utf-8", errors="ignore").strip()
                return self._parse_reply(raw)
            except (OSError, socket.timeout):
                self.close()
                raise

    def _parse_reply(self, raw: str) -> RobotReply:
        match = re.match(r"\s*(-?\d+)\s*,\s*\{([^}]*)\}", raw)
        if not match:
            return RobotReply(-99999, [], raw)
        values = [item.strip() for item in match.group(2).split(",") if item.strip()]
        return RobotReply(int(match.group(1)), values, raw)

    def _parse_pose(self, command: str) -> list[float] | None:
        match = re.search(r"pose\s*=\s*\{([^}]*)\}", command)
        if not match:
            return None
        values = [float(item.strip()) for item in match.group(1).split(",")]
        return values if len(values) == 6 else None

    def _parse_tool_do(self, command: str) -> tuple[int, int] | None:
        match = re.search(r"ToolDOInstant\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", command, re.IGNORECASE)
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    def _parse_single_index(self, command: str) -> int | None:
        match = re.search(r"\(\s*(\d+)\s*\)", command)
        if not match:
            return None
        return int(match.group(1))

    def require_ok(self, reply: RobotReply, action: str) -> RobotReply:
        if reply.error_id != 0:
            raise RuntimeError(f"{action}失败: {reply.raw}")
        return reply

    def request_control(self) -> RobotReply:
        return self.require_ok(self.send("RequestControl()"), "请求控制权")

    def enable(self) -> RobotReply:
        if not self.simulation and bool(self.config["robot"].get("request_control_before_enable", True)):
            self.request_control()
        reply = self.require_ok(self.send("EnableRobot()"), "机器人使能")
        self.enabled = True
        return reply

    def get_pose(self) -> list[float]:
        user = int(self.config["robot"].get("user", 0))
        tool = int(self.config["robot"].get("tool", 0))
        reply = self.require_ok(self.send(f"GetPose(user={user},tool={tool})"), "获取位姿")
        if len(reply.values) >= 6:
            self.current_pose = [float(value) for value in reply.values[:6]]
        return list(self.current_pose)

    def get_angles(self) -> list[float]:
        reply = self.require_ok(self.send("GetAngle()"), "获取关节角")
        if len(reply.values) >= 6:
            return [float(value) for value in reply.values[:6]]
        raise RuntimeError(f"获取关节角返回值异常: {reply.raw}")

    def robot_mode(self) -> int:
        reply = self.require_ok(self.send("RobotMode()"), "获取机器人状态")
        if len(reply.values) >= 1:
            return int(float(reply.values[0]))
        raise RuntimeError(f"获取机器人状态返回值异常: {reply.raw}")

    def movl(self, pose: list[float], speed: int | None = None, accel: int | None = None) -> RobotReply:
        robot_cfg = self.config["robot"]
        speed = int(speed or robot_cfg["speed_percent"])
        accel = int(accel or robot_cfg["accel_percent"])
        user = int(robot_cfg.get("user", 0))
        tool = int(robot_cfg.get("tool", 0))
        pose_text = ",".join(f"{value:.3f}" for value in pose)
        return self.require_ok(self.send(f"MovL(pose={{{pose_text}}},user={user},tool={tool},a={accel},v={speed},cp=0)"), "直线运动")

    def movj(self, pose: list[float], speed: int | None = None, accel: int | None = None) -> RobotReply:
        robot_cfg = self.config["robot"]
        speed = int(speed or robot_cfg["speed_percent"])
        accel = int(accel or robot_cfg["accel_percent"])
        user = int(robot_cfg.get("user", 0))
        tool = int(robot_cfg.get("tool", 0))
        pose_text = ",".join(f"{value:.3f}" for value in pose)
        return self.require_ok(self.send(f"MovJ(pose={{{pose_text}}},user={user},tool={tool},a={accel},v={speed},cp=0)"), "关节运动")

    def wait_until_pose(self, target_pose: list[float]) -> None:
        robot_cfg = self.config["robot"]
        timeout_s = float(robot_cfg.get("motion_done_timeout_s", 30.0))
        poll_s = float(robot_cfg.get("motion_done_poll_s", 0.1))
        xyz_tol = float(robot_cfg.get("motion_done_xyz_tol_mm", 2.0))
        rpy_tol = float(robot_cfg.get("motion_done_rpy_tol_deg", 3.0))
        deadline = time.time() + timeout_s
        last_pose: list[float] | None = None
        last_mode: int | None = None
        while time.time() < deadline:
            if self.cancel_event.is_set():
                raise RuntimeError("已急停，取消等待机械臂到位")
            try:
                last_mode = self.robot_mode()
                if last_mode != 5:
                    time.sleep(max(0.02, poll_s))
                    continue
                last_pose = self.get_pose()
                xyz_ok = all(abs(last_pose[i] - target_pose[i]) <= xyz_tol for i in range(3))
                rpy_ok = all(self._angle_delta(last_pose[i], target_pose[i]) <= rpy_tol for i in range(3, 6))
                if xyz_ok and rpy_ok:
                    return
            except Exception:
                pass
            time.sleep(max(0.02, poll_s))
        raise RuntimeError(
            "等待机械臂到位超时："
            f"target={[round(v, 2) for v in target_pose]} "
            f"last={[round(v, 2) for v in last_pose] if last_pose else None} "
            f"mode={last_mode}"
        )

    @staticmethod
    def _angle_delta(actual: float, target: float) -> float:
        return abs(((actual - target + 180.0) % 360.0) - 180.0)

    def suction(self, on: bool) -> RobotReply:
        robot_cfg = self.config["robot"]
        status = int(robot_cfg.get("suction_on_level", 1) if on else robot_cfg.get("suction_off_level", 0))
        io_type = str(robot_cfg.get("suction_io_type", "both")).lower()
        if io_type not in {"tool_do", "tool", "both"}:
            raise RuntimeError(f"ES01 吸盘应使用末端 ToolDOInstant，当前配置为: {io_type}")
        index = int(robot_cfg.get("suction_tool_do", 1))
        reply = self.require_ok(self.send(f"ToolDOInstant({index},{status})"), "末端吸盘控制")
        settle_s = float(robot_cfg.get("suction_state_timeout_s", 0.5))
        poll_s = float(robot_cfg.get("suction_state_poll_s", 0.05))
        deadline = time.time() + max(0.0, settle_s)
        last_value: int | None = None
        while time.time() < deadline:
            if self.cancel_event.is_set():
                raise RuntimeError("已急停，取消吸盘等待")
            state = self.get_tool_do(index)
            last_value = state
            if state == status:
                break
            time.sleep(max(0.01, poll_s))
        if last_value is not None and last_value != status:
            raise RuntimeError(f"末端吸盘状态未切换到目标值：index={index} target={status} actual={last_value}")
        return reply

    def get_tool_do(self, index: int) -> int:
        reply = self.require_ok(self.send(f"GetToolDO({int(index)})"), "读取末端吸盘状态")
        if len(reply.values) >= 1:
            try:
                return int(float(reply.values[0]))
            except ValueError:
                pass
        raise RuntimeError(f"读取末端吸盘状态失败: {reply.raw}")

    def stop_motion(self) -> RobotReply:
        self.cancel_event.set()
        return self.require_ok(self.send("Stop()"), "停止运动")

    def pause_motion(self) -> RobotReply:
        return self.require_ok(self.send("Pause()"), "暂停运动")

    def emergency_stop(self) -> RobotReply:
        self.cancel_event.set()
        reply = self.require_ok(self.send("EmergencyStop(1)"), "紧急停止")
        self.enabled = False
        return reply

    def disable(self) -> RobotReply:
        reply = self.require_ok(self.send("DisableRobot()"), "机器人下使能")
        self.enabled = False
        return reply

    def reset_cancel(self) -> None:
        self.cancel_event.clear()

    def build_grasp_sequence(self, grasp_pose: list[float], label: str) -> list[tuple[str, list[float] | bool]]:
        robot_cfg = self.config["robot"]
        safe_z = float(robot_cfg["safe_z_mm"])
        clearance = float(robot_cfg["grasp_clearance_mm"])
        suction_settle_s = float(robot_cfg.get("suction_settle_s", 0.5))
        suction_release_settle_s = float(robot_cfg.get("suction_release_settle_s", 0.3))
        bin_pose = list(robot_cfg["bins"][label])
        above_grasp = list(grasp_pose)
        above_grasp[2] = max(safe_z, grasp_pose[2] + clearance)
        above_bin = list(bin_pose)
        above_bin[2] = max(safe_z, bin_pose[2] + clearance)
        # 【防撞下降】：先全速移动到抓取点正上方(above_grasp)，最后一段垂直下降改用 movl_slow 慢速逼近，
        # 避免全速直冲扎到物料/桌面造成机械臂轴碰撞。下降速度由 robot.descend_speed_percent 控制。
        return [
            ("movj", above_grasp),
            ("movl_slow", grasp_pose),
            ("suction", True),
            ("wait", suction_settle_s),
            ("movl", above_grasp),
            ("movj", above_bin),
            ("movl_slow", bin_pose),
            ("suction", False),
            ("wait", suction_release_settle_s),
            ("movl", above_bin),
            ("movj", list(robot_cfg["home_pose"])),
        ]
