from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PlannedActionSequence:
    profile: str
    steps: tuple[tuple[str, Any], ...]


class ActionSequencePlanner:
    MOVE_ACTIONS = {"movj", "movl", "movl_slow"}
    VALID_ACTIONS = MOVE_ACTIONS | {"suction", "wait", "home"}

    def __init__(self, settings: Mapping[str, Any], robot_config: Mapping[str, Any]):
        self.settings = dict(settings)
        self.robot_config = dict(robot_config)
        self.validate_settings()

    def validate_settings(self) -> None:
        pose_bank = self.settings.get("pose_bank", {})
        profiles = self.settings.get("profiles", {})
        if not isinstance(pose_bank, Mapping):
            raise RuntimeError("action_sequences.pose_bank 必须是映射")
        for name, pose in pose_bank.items():
            self._validate_pose(pose, f"pose_bank.{name}")
        if not isinstance(profiles, Mapping):
            raise RuntimeError("action_sequences.profiles 必须是映射")
        for profile_name, profile in profiles.items():
            if not isinstance(profile, Mapping):
                raise RuntimeError(f"动作序列 {profile_name} 必须是映射")
            steps = profile.get("steps", [])
            if not isinstance(steps, list):
                raise RuntimeError(f"动作序列 {profile_name}.steps 必须是列表")
            for index, step in enumerate(steps, 1):
                self._validate_step(step, f"{profile_name}.steps[{index}]")

    def plan(self, profile_name: str | None = None) -> PlannedActionSequence:
        profile_name = str(profile_name or self.settings.get("active_profile") or "")
        profiles = self.settings.get("profiles", {})
        if not profile_name:
            raise RuntimeError("尚未配置 action_sequences.active_profile")
        if profile_name not in profiles:
            raise RuntimeError(f"动作序列 Profile 不存在: {profile_name}")
        raw_steps = profiles[profile_name].get("steps", [])
        if not raw_steps:
            raise RuntimeError(f"动作序列 {profile_name} 没有步骤")
        planned: list[tuple[str, Any]] = []
        for index, step in enumerate(raw_steps, 1):
            action = str(step["action"]).lower()
            if action in self.MOVE_ACTIONS:
                pose = self._resolve_pose(step, profile_name, index)
                self._check_safe_z(pose, profile_name, index)
                planned.append((action, pose))
            elif action == "home":
                pose = self._validate_pose(self.robot_config.get("home_pose"), "robot.home_pose")
                self._check_safe_z(pose, profile_name, index)
                planned.append(("movj", pose))
            elif action == "suction":
                planned.append(("suction", bool(step["value"])))
            elif action == "wait":
                seconds = float(step["seconds"])
                if not 0 <= seconds <= 30:
                    raise RuntimeError(f"动作序列 {profile_name} 第 {index} 步等待时间必须在 0~30 秒")
                planned.append(("wait", seconds))
        return PlannedActionSequence(profile_name, tuple(planned))

    def _resolve_pose(self, step: Mapping[str, Any], profile: str, index: int) -> list[float]:
        if step.get("pose") is not None:
            return self._validate_pose(step["pose"], f"{profile}.steps[{index}].pose")
        reference = str(step.get("pose_ref") or "")
        if not reference:
            raise RuntimeError(f"动作序列 {profile} 第 {index} 步缺少 pose 或 pose_ref")
        pose_bank = self.settings.get("pose_bank", {})
        if reference not in pose_bank:
            raise RuntimeError(f"动作序列 {profile} 第 {index} 步引用的位姿不存在: {reference}")
        return self._validate_pose(pose_bank[reference], f"pose_bank.{reference}")

    def _check_safe_z(self, pose: list[float], profile: str, index: int) -> None:
        minimum_z = float(self.robot_config.get("min_grasp_z_mm", 60.0))
        if pose[2] < minimum_z:
            raise RuntimeError(
                f"动作序列 {profile} 第 {index} 步 Z={pose[2]:g}mm 低于安全下限 {minimum_z:g}mm"
            )

    def _validate_step(self, step: Any, location: str) -> None:
        if not isinstance(step, Mapping):
            raise RuntimeError(f"{location} 必须是映射")
        action = str(step.get("action") or "").lower()
        if action not in self.VALID_ACTIONS:
            raise RuntimeError(f"{location}.action 不支持: {action or '<empty>'}")
        if action in self.MOVE_ACTIONS:
            has_pose = step.get("pose") is not None
            has_reference = bool(step.get("pose_ref"))
            if has_pose == has_reference:
                raise RuntimeError(f"{location} 必须且只能配置 pose 或 pose_ref")
            if has_pose:
                self._validate_pose(step["pose"], f"{location}.pose")
        elif action == "suction" and not isinstance(step.get("value"), bool):
            raise RuntimeError(f"{location}.value 必须是 true 或 false")
        elif action == "wait":
            try:
                float(step["seconds"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"{location}.seconds 必须是数值") from exc

    @staticmethod
    def _validate_pose(value: Any, location: str) -> list[float]:
        if not isinstance(value, (list, tuple)) or len(value) != 6:
            raise RuntimeError(f"{location} 必须包含 [x, y, z, rx, ry, rz] 6 个值")
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{location} 的位姿值必须是数值") from exc
