from __future__ import annotations

import json
import threading
from pathlib import Path
from time import strftime

import numpy as np

from .camera import RealSenseCamera, save_capture
from .action_sequence import ActionSequencePlanner, PlannedActionSequence
from .command_transport import CommandReply, JsonCommandTransport
from .detector import FruitDetector
from .height import HeightEstimator
from .object_pose import ObjectPoseEstimator
from .robot_client import DobotClient
from .transform import CoordinateTransformer
from .types import Detection, TargetResult
from .vision_studio_client import DobotVisionStudioClient, VisionStudioReply
from .vision_task import VisionTaskProcessor, VisionTaskResult
from app.sim.sim_camera import SimulatedCamera


class CompetitionWorkflow:
    def __init__(self, config: dict, logger=print):
        self.config = config
        self.log = logger
        self.simulation = config.get("mode") == "simulation"
        self.camera = SimulatedCamera(config) if self.simulation else RealSenseCamera(config)
        self.detector = FruitDetector(config, simulation=self.simulation)
        self.height_estimator = HeightEstimator(config)
        self.object_pose_estimator = ObjectPoseEstimator(config)
        self.transformer = CoordinateTransformer(config)
        self.command_transport = JsonCommandTransport(config, simulation=self.simulation)
        self.vision_studio = DobotVisionStudioClient(config, simulation=self.simulation)
        self.robot = DobotClient(config, simulation=self.simulation)
        self.frame = None
        self.detection_frame = None
        self._frame_lock = threading.Lock()
        self.detections: list[Detection] = []
        self.current_target: TargetResult | None = None
        self.shared_data: dict[str, object] = {}
        self.vision_studio_data = None
        self.vision_task_result: VisionTaskResult | None = None
        self.camera_started = False

    def _action_sequence_planner(self) -> ActionSequencePlanner:
        return ActionSequencePlanner(
            self.config.get("action_sequences", {}),
            self.config.get("robot", {}),
        )

    def plan_action_sequence(self, profile_name: str | None = None) -> PlannedActionSequence:
        plan = self._action_sequence_planner().plan(profile_name)
        self.shared_data["planned_action_sequence"] = plan
        self.log(f"动作序列预检完成: {plan.profile}，共 {len(plan.steps)} 步")
        for index, (action, value) in enumerate(plan.steps, 1):
            self.log(f"  {index}. {action} {value}")
        return plan

    def execute_action_sequence(self, profile_name: str | None = None) -> PlannedActionSequence:
        plan = self.shared_data.get("planned_action_sequence")
        if (
            not isinstance(plan, PlannedActionSequence)
            or (profile_name is not None and plan.profile != profile_name)
        ):
            plan = self.plan_action_sequence(profile_name)
        self._ensure_robot_ready()
        try:
            self._execute_sequence(list(plan.steps))
        except Exception:
            if self.robot.connected:
                try:
                    self.robot.suction(False)
                except Exception:
                    pass
                try:
                    self.robot.stop_motion()
                except Exception:
                    pass
            raise
        finally:
            self.shared_data.pop("planned_action_sequence", None)
        self.log(f"动作序列执行完成: {plan.profile}")
        return plan

    def discover_vision_studio(self) -> dict:
        result = self.vision_studio.discover()
        self.log("DobotVisionStudio 通信扫描: " + json.dumps(result, ensure_ascii=False))
        return result

    def connect_vision_studio(self) -> None:
        self.vision_studio.connect()
        self.log(f"DobotVisionStudio 已连接：mode={self.vision_studio.mode}")

    def set_vision_studio_globals(self) -> dict[str, object]:
        protocol = self.config.get("vision_studio", {}).get("protocol", {})
        values = protocol.get("global_values", {})
        if not isinstance(values, dict) or not values:
            raise RuntimeError("尚未配置 vision_studio.protocol.global_values")
        for name, value in values.items():
            self.vision_studio.set_global_value(str(name), value)
            self.log(f"DVS 全局变量已设置: {name}={value}")
        self.shared_data["vision_studio_globals"] = dict(values)
        return dict(values)

    def exchange_vision_studio(self) -> VisionStudioReply:
        reply = self.vision_studio.exchange()
        self.vision_studio_data = reply.data
        self.vision_task_result = None
        self.shared_data["vision_studio"] = reply.data
        self.shared_data.pop("vision_task", None)
        self.shared_data.pop("planned_pick_place", None)
        self.log(
            "DobotVisionStudio 返回: "
            + json.dumps(reply.data, ensure_ascii=False, default=str)
        )
        return reply

    def _vision_task_processor(self) -> VisionTaskProcessor:
        task_cfg = self.config.get("vision_tasks", {})
        active_name = task_cfg.get("active_profile")
        profiles = task_cfg.get("profiles", {})
        if not active_name:
            raise RuntimeError("尚未配置 vision_tasks.active_profile")
        if active_name not in profiles:
            raise RuntimeError(f"视觉任务配置不存在: {active_name}")
        return VisionTaskProcessor(profiles[active_name])

    def normalize_vision_result(self) -> VisionTaskResult:
        if self.vision_studio_data is None:
            raise RuntimeError("尚未获取 DobotVisionStudio 结果")
        self.shared_data.pop("planned_pick_place", None)
        self.vision_task_result = self._vision_task_processor().normalize(self.vision_studio_data)
        self.shared_data["vision_task"] = self.vision_task_result
        self.log("DVS 字段已标准化: " + json.dumps(self.vision_task_result.fields, ensure_ascii=False))
        return self.vision_task_result

    def evaluate_measurements(self) -> VisionTaskResult:
        if self.vision_task_result is None:
            self.normalize_vision_result()
        self.vision_task_result = self._vision_task_processor().evaluate_measurements(self.vision_task_result)
        self.shared_data["vision_task"] = self.vision_task_result
        detail = "；".join(self.vision_task_result.violations) or "全部测量值在公差内"
        self.log(f"尺寸判定: {self.vision_task_result.decision} ({detail})")
        return self.vision_task_result

    def evaluate_quality_status(self) -> VisionTaskResult:
        if self.vision_task_result is None:
            self.normalize_vision_result()
        self.vision_task_result = self._vision_task_processor().evaluate_status(self.vision_task_result)
        self.shared_data["vision_task"] = self.vision_task_result
        self.log(f"质量状态判定: {self.vision_task_result.decision}")
        return self.vision_task_result

    def validate_identifiers(self) -> VisionTaskResult:
        if self.vision_task_result is None:
            self.normalize_vision_result()
        self.vision_task_result = self._vision_task_processor().validate_identifiers(self.vision_task_result)
        self.shared_data["vision_task"] = self.vision_task_result
        self.log("识别字段: " + json.dumps(self.vision_task_result.identifiers, ensure_ascii=False))
        return self.vision_task_result

    def route_vision_result(self) -> VisionTaskResult:
        if self.vision_task_result is None:
            self.normalize_vision_result()
        self.vision_task_result = self._vision_task_processor().resolve_route(self.vision_task_result)
        self.shared_data["vision_task"] = self.vision_task_result
        self.log(f"结果路由: {self.vision_task_result.route}")
        return self.vision_task_result

    def plan_result_pick_place(self) -> tuple[list[float], list[float]]:
        if self.vision_task_result is None:
            raise RuntimeError("尚未标准化 DobotVisionStudio 结果")
        processor = self._vision_task_processor()
        pick_pose = processor.resolve_pick_pose(self.vision_task_result)
        placement_key = self.vision_task_result.route or "__default__"
        counters = self.shared_data.setdefault("placement_counts", {})
        if not isinstance(counters, dict):
            raise RuntimeError("placement_counts 状态无效")
        placement_index = int(counters.get(placement_key, 0))
        place_pose = processor.resolve_place_pose(self.vision_task_result, placement_index)
        minimum_z = float(self.config["robot"].get("min_grasp_z_mm", 60.0))
        if pick_pose[2] < minimum_z or place_pose[2] < minimum_z:
            raise RuntimeError(
                f"DVS 动作坐标低于安全下限 {minimum_z:g}mm: pick_z={pick_pose[2]:g}, place_z={place_pose[2]:g}"
            )
        self.shared_data["planned_pick_place"] = {
            "pick_pose": list(pick_pose),
            "place_pose": list(place_pose),
            "placement_key": placement_key,
            "placement_index": placement_index,
        }
        self.log(
            f"DVS 动作预览: pick={[round(v, 2) for v in pick_pose]} "
            f"place={[round(v, 2) for v in place_pose]}"
        )
        return pick_pose, place_pose

    def execute_result_pick_place(self) -> tuple[list[float], list[float]]:
        planned = self.shared_data.get("planned_pick_place")
        if not isinstance(planned, dict):
            self.plan_result_pick_place()
            planned = self.shared_data.get("planned_pick_place")
        if not isinstance(planned, dict):
            raise RuntimeError("DVS 动作预览数据未生成")
        pick_pose = list(planned.get("pick_pose", []))
        place_pose = list(planned.get("place_pose", []))
        if len(pick_pose) != 6 or len(place_pose) != 6:
            raise RuntimeError("DVS 动作预览数据无效，请重新运行坐标预览模块")
        self._ensure_robot_ready()
        try:
            self._execute_sequence(self.robot.build_pick_place_sequence(pick_pose, place_pose))
        except Exception:
            if self.robot.connected:
                try:
                    self.robot.suction(False)
                except Exception:
                    pass
                try:
                    self.robot.stop_motion()
                except Exception:
                    pass
            raise
        finally:
            self.shared_data.pop("planned_pick_place", None)
        self.shared_data["last_pick_place"] = {
            "pick_pose": list(pick_pose),
            "place_pose": list(place_pose),
        }
        placement_key = str(planned.get("placement_key", "__default__"))
        placement_index = int(planned.get("placement_index", 0))
        counters = self.shared_data.setdefault("placement_counts", {})
        if isinstance(counters, dict):
            counters[placement_key] = placement_index + 1
        self.log(
            f"DVS 结果抓放完成: pick={[round(v, 2) for v in pick_pose]} "
            f"place={[round(v, 2) for v in place_pose]}"
        )
        return pick_pose, place_pose

    def reset_placement_counters(self) -> None:
        self.shared_data["placement_counts"] = {}
        self.shared_data.pop("planned_pick_place", None)
        self.log("顺序槽位/堆叠计数已清零")

    def run_dvs_cycle(self) -> dict[str, object]:
        settings = self.config.get("dvs_cycle", {})
        max_items = int(settings.get("max_items", 8))
        delay_s = float(settings.get("delay_s", 0.2))
        processor = str(settings.get("processor", "none")).lower()
        validate_identifiers = bool(settings.get("validate_identifiers", False))
        route_enabled = bool(settings.get("route", False))
        execute_motion = bool(settings.get("execute_motion", False))
        before_sequence = str(settings.get("before_each_sequence") or "")
        after_sequence = str(settings.get("after_each_sequence") or "")
        on_error = str(settings.get("on_error", "stop")).lower()
        if not 1 <= max_items <= 100:
            raise RuntimeError("dvs_cycle.max_items 必须在 1~100 之间")
        if delay_s < 0:
            raise RuntimeError("dvs_cycle.delay_s 不能小于 0")
        if processor not in {"none", "status", "measurement", "status_measurement"}:
            raise RuntimeError(f"不支持的 DVS 循环处理方式: {processor}")
        if on_error not in {"stop", "skip"}:
            raise RuntimeError(f"不支持的 DVS 循环错误策略: {on_error}")
        has_robot_motion = execute_motion or bool(before_sequence) or bool(after_sequence)
        if has_robot_motion and on_error != "stop":
            raise RuntimeError("DVS 循环包含机械臂动作时，on_error 必须为 stop")
        planner = self._action_sequence_planner()
        if before_sequence:
            planner.plan(before_sequence)
        if after_sequence:
            planner.plan(after_sequence)

        completed = 0
        attempts = 0
        failures = 0
        stopped_empty = False
        cancelled = False
        decision_counts: dict[str, int] = {}
        route_counts: dict[str, int] = {}
        records: list[dict[str, object]] = []

        def current_stats() -> dict[str, object]:
            return {
                "completed": completed,
                "attempts": attempts,
                "failures": failures,
                "stopped_empty": stopped_empty,
                "cancelled": cancelled,
                "decision_counts": dict(decision_counts),
                "route_counts": dict(route_counts),
                "records": list(records),
            }
        self.log(
            f"DVS 连续任务开始: max={max_items} processor={processor} "
            f"route={route_enabled} motion={execute_motion} "
            f"before={before_sequence or '-'} after={after_sequence or '-'}"
        )
        if bool(settings.get("reset_placement_counters_on_start", True)):
            self.reset_placement_counters()
        for index in range(max_items):
            if self.robot.cancel_event.is_set():
                cancelled = True
                break
            attempts += 1
            try:
                if before_sequence:
                    self.execute_action_sequence(before_sequence)
                self.exchange_vision_studio()
                result = self.normalize_vision_result()
                if self._is_dvs_cycle_empty(result.fields, settings.get("empty", {})):
                    stopped_empty = True
                    self.log(f"DVS 连续任务收到空料信号，第 {index + 1} 次结束")
                    break
                if processor == "status":
                    result = self.evaluate_quality_status()
                elif processor == "measurement":
                    result = self.evaluate_measurements()
                elif processor == "status_measurement":
                    result = self.evaluate_quality_status()
                    result = self.evaluate_measurements()
                if validate_identifiers:
                    result = self.validate_identifiers()
                if route_enabled:
                    result = self.route_vision_result()
                if execute_motion:
                    self.plan_result_pick_place()
                    self.execute_result_pick_place()
                if after_sequence:
                    self.execute_action_sequence(after_sequence)
                completed += 1
                if result.decision:
                    decision_counts[result.decision] = decision_counts.get(result.decision, 0) + 1
                if result.route:
                    route_counts[result.route] = route_counts.get(result.route, 0) + 1
                records.append(
                    {
                        "index": completed,
                        "decision": result.decision,
                        "route": result.route,
                        "identifiers": dict(result.identifiers),
                        "violations": list(result.violations),
                        "fields": dict(result.fields),
                    }
                )
                self.log(
                    f"DVS 连续任务进度: {completed}/{max_items} "
                    f"decision={result.decision or '-'} route={result.route or '-'}"
                )
            except Exception as exc:
                failures += 1
                self.log(f"DVS 连续任务第 {index + 1} 次失败: {exc}")
                if has_robot_motion and self.robot.connected:
                    try:
                        self.robot.suction(False)
                    except Exception as stop_exc:
                        self.log(f"连续任务异常后关闭吸盘失败: {stop_exc}")
                    try:
                        self.robot.stop_motion()
                    except Exception as stop_exc:
                        self.log(f"连续任务异常后停止运动失败: {stop_exc}")
                if on_error == "stop":
                    stats = current_stats()
                    self.shared_data["dvs_cycle"] = stats
                    raise RuntimeError(f"DVS 连续任务已停止: {exc}") from exc
            if index + 1 < max_items and delay_s > 0:
                if self.robot.cancel_event.wait(delay_s):
                    cancelled = True
                    break

        stats = current_stats()
        self.shared_data["dvs_cycle"] = stats
        self.log(
            f"DVS 连续任务结束: completed={completed} attempts={attempts} "
            f"failures={failures} empty={stopped_empty} cancelled={cancelled}"
        )
        return stats

    @staticmethod
    def _is_dvs_cycle_empty(fields: dict[str, object], empty_config: object) -> bool:
        if not isinstance(empty_config, dict):
            return False
        field_name = empty_config.get("field")
        if not field_name or str(field_name) not in fields:
            return False
        target = str(fields[str(field_name)]).strip().casefold()
        values = empty_config.get("values", [])
        if not isinstance(values, (list, tuple, set)):
            values = [values]
        return any(str(value).strip().casefold() == target for value in values)

    def start_camera(self) -> None:
        if not self.camera_started:
            self.camera.start()
            self.camera_started = True
        self.frame = self.camera.read()
        self.log("相机已启动：RGB/Depth 双流就绪" + ("（模拟模式）" if self.simulation else ""))

    def read_frame(self):
        with self._frame_lock:
            self.frame = self.camera.read()
            return self.frame

    def calibrate_table(self) -> bool:
        """在空桌面（或物体很少）时标定桌面平面。

        标定结果存在基坐标系，机械臂之后移动无需重标。返回是否成功拟合平面。
        """
        if self.frame is None:
            self.read_frame()
        if self.robot.connected:
            current_pose = self.robot.get_pose()
        else:
            current_pose = self.robot.current_pose
        cam_to_base = self.transformer.camera_to_base_matrix(current_pose)
        self.height_estimator.calibrate_table(self.frame.depth_mm, self.frame.intrinsics, cam_to_base)
        ok = self.height_estimator.has_plane()
        if ok:
            a, b, c, d = self.height_estimator.plane_base
            self.log(f"桌面标定完成（基坐标系平面）：法向=({a:.3f},{b:.3f},{c:.3f}) d={d:.1f}")
        else:
            self.log("桌面标定失败：未能拟合平面，将退回标量高度估计")
        return ok

    def save_current_frame(self) -> dict[str, Path]:
        if self.frame is None:
            self.read_frame()
        output_dir = Path(__file__).resolve().parents[2] / "runs" / "captures" / strftime("%Y%m%d")
        paths = save_capture(self.frame, output_dir)
        self.log(f"图像已保存：{paths['rgb'].name} / {paths['depth'].name}")
        return paths

    def detect_once(self) -> list[Detection]:
        if self.frame is None:
            self.read_frame()
        self.detection_frame = self.frame.color.copy()
        self.detections = self.detector.detect(self.frame.color)
        if not self.detections:
            self.current_target = None
            self.log("未识别到有效目标")
            return []
        target = self.detections[0]
        current_pose = self.robot.get_pose() if self.robot.connected else self.robot.current_pose
        self.current_target = self._build_target_result(target, current_pose)
        cx, cy = self.current_target.detection.center
        depth = self._depth_at(cx, cy)
        base_pose = self.current_target.base_pose
        object_height = self.current_target.height_mm
        angle_deg = self.current_target.angle_deg
        camera_xyz = self.current_target.camera_xyz_mm
        self.log(
            f"识别成功：{target.label} conf={target.confidence:.2f} pixel={target.center} "
            f"height={object_height:.1f}mm angle={angle_deg:.1f}° "
            f"angle_valid={self.current_target.angle_valid} "
            f"grasp_pixel={self.current_target.grasp_pixel}"
        )
        self.log(f"  拍照位姿：{[round(v,2) for v in current_pose]}  深度：{depth:.1f}mm  相机坐标：({camera_xyz[0]:.1f},{camera_xyz[1]:.1f},{camera_xyz[2]:.1f})")
        self.log(f"  基坐标解算：{[round(v,2) for v in base_pose]}")
        self.log(f"  抓取点最终Z={base_pose[2]:.1f}mm（桌面平面={'已使用' if self.height_estimator.has_plane() else '未使用'}）")
        # 【深度合理性检查】：如果深度异常(过浅、过深)，原始Z会胡跳。直接打印帮你定位是【深度问题】还是【标定问题】
        try:
            bx, by = self.config["height"].get("table_depth_mm", 0), 0
            depth_ok = "✓正常" if 200 < depth < 600 else f"⚠异常(应在200~600之间)"
            # ROI 内深度统计：暴露是不是只读到一两个噪点
            roi = self.frame.depth_mm[max(0,cy-3):cy+4, max(0,cx-3):cx+4]
            valid = roi[np.isfinite(roi) & (roi > 0)]
            self.log(f"  深度ROI: 取样{valid.size}/{roi.size}像素  中值={depth:.1f}mm {depth_ok}  范围[{float(valid.min()) if valid.size else 0:.1f},{float(valid.max()) if valid.size else 0:.1f}]")
        except Exception:
            pass
        return self.detections

    def _build_target_result(self, target: Detection, current_pose: list[float]) -> TargetResult:
        if self.frame is None:
            raise RuntimeError("没有相机帧，无法计算目标位姿")
        cam_to_base = self.transformer.camera_to_base_matrix(current_pose)
        object_height = self.height_estimator.estimate(
            self.frame.depth_mm, target, self.frame.intrinsics, cam_to_base
        )
        estimate = self.object_pose_estimator.estimate(
            self.frame.depth_mm,
            target,
            self.frame.intrinsics,
            cam_to_base,
            self.height_estimator.plane_base,
        )
        orientation = estimate.orientation
        require_valid = bool(self.config.get("orientation", {}).get("require_valid", False))
        if require_valid and not orientation.valid:
            raise RuntimeError(f"{target.label}角度估计无效：{orientation.reason or '未知原因'}")
        grasp_pixel = estimate.grasp_pixel if estimate.base_xyz_mm is not None else target.center
        cx, cy = grasp_pixel
        depth = self._depth_at(cx, cy)
        camera_xyz = self.transformer.pixel_to_camera(cx, cy, depth, self.frame.intrinsics)
        base_pose = self.transformer.camera_to_base_pose(camera_xyz, current_pose, [180.0, 0.0, 0.0])
        base_pose = self._apply_grasp_offsets(base_pose)
        object_height = float(object_height)
        raw_base_z = float(base_pose[2])
        grasp_z_offset = float(self.config["robot"].get("grasp_z_offset_mm", 0.0))
        min_grasp_z = float(self.config["robot"].get("min_grasp_z_mm", 60.0))
        target_z = raw_base_z + grasp_z_offset
        if self.height_estimator.has_plane():
            plane = self.height_estimator.plane_base
            if plane is not None and abs(float(plane[2])) > 1e-6:
                a, b, c, d = plane
                table_z = -(a * base_pose[0] + b * base_pose[1] + d) / c
                target_z = table_z + object_height + grasp_z_offset
        base_pose[2] = max(min_grasp_z, target_z)
        base_pose = self._apply_object_orientation(base_pose, target.label, orientation.angle_deg)
        return TargetResult(
            target,
            object_height,
            camera_xyz,
            base_pose,
            orientation.angle_deg,
            orientation.valid,
            orientation.confidence,
            orientation.reason,
            grasp_pixel,
            estimate.point_count,
        )

    def select_auto_target(self) -> Detection | None:
        if self.frame is None:
            self.read_frame()
        self.detection_frame = self.frame.color.copy()
        detections = self.detector.detect(self.frame.color)
        self.detections = detections
        if not detections:
            self.current_target = None
            self.log("未识别到有效目标")
            return None
        current_pose = self.robot.get_pose() if self.robot.connected else self.robot.current_pose
        cam_to_base = self.transformer.camera_to_base_matrix(current_pose)
        target = max(
            detections,
            key=lambda det: (
                self.height_estimator.estimate(self.frame.depth_mm, det, self.frame.intrinsics, cam_to_base),
                det.confidence,
            ),
        )
        self.current_target = None
        return target

    def _depth_at(self, u: int, v: int) -> float:
        if self.frame is None:
            raise RuntimeError("没有相机帧")
        height, width = self.frame.depth_mm.shape[:2]
        x1, x2 = max(0, u - 3), min(width, u + 4)
        y1, y2 = max(0, v - 3), min(height, v + 4)
        roi = self.frame.depth_mm[y1:y2, x1:x2]
        valid = roi[np.isfinite(roi) & (roi > 0)]
        if valid.size == 0:
            return float(self.config["height"]["table_depth_mm"])
        return float(np.median(valid))

    def estimate_object_angle(self, detection: Detection) -> float:
        """Return the current base-frame angle for compatibility callers."""
        if self.frame is None:
            return 0.0
        current_pose = self.robot.get_pose() if self.robot.connected else self.robot.current_pose
        cam_to_base = self.transformer.camera_to_base_matrix(current_pose)
        self.height_estimator.estimate(self.frame.depth_mm, detection, self.frame.intrinsics, cam_to_base)
        estimate = self.object_pose_estimator.estimate(
            self.frame.depth_mm,
            detection,
            self.frame.intrinsics,
            cam_to_base,
            self.height_estimator.plane_base,
        )
        return float(estimate.orientation.angle_deg)

    def connect_robot(self) -> None:
        try:
            self.robot.connect()
            self.robot.enable()
            self.robot.get_pose()
        except Exception:
            self.robot.close()
            raise
        self.log("机器人已连接并使能" + ("（模拟模式）" if self.simulation else ""))

    def fixed_point_test(self) -> None:
        self._ensure_robot_ready()
        grasp_pose = list(self.config["robot"]["fixed_test_pose"])
        bins = self.config["robot"].get("bins", {})
        slots = self.config.get("assembly", {}).get("slots", {})
        labels = list(bins) + [label for label in slots if label not in bins]
        if not labels:
            raise RuntimeError("未配置任何置物盒坐标，无法完成定点抓取测试")
        target_label = labels[0]
        sequence = self.robot.build_grasp_sequence(grasp_pose, target_label)
        self.log(f"定点抓取测试开始：固定抓取点={grasp_pose}，目标置物盒={target_label}")
        self._execute_sequence(sequence)
        self.log(f"定点抓取测试完成：已从固定点抓取并放置到 {target_label} 置物盒")

    def calculate_grasp(self) -> TargetResult:
        if self.current_target is None:
            self.detect_once()
        if self.current_target is None:
            raise RuntimeError("无有效目标，无法计算抓取坐标")
        payload = {
            "cmd": "grasp_target",
            "label": self.current_target.detection.label,
            "pixel": list(self.current_target.detection.center),
            "height_mm": round(self.current_target.height_mm, 2),
            "base_pose": [round(value, 3) for value in self.current_target.base_pose],
        }
        self.log("抓取坐标JSON: " + json.dumps(payload, ensure_ascii=False))
        return self.current_target

    def build_tcp_command_payload(self) -> dict:
        target = self.calculate_grasp()
        label = target.detection.label
        if not self.robot.has_target_pose(label):
            raise RuntimeError(f"未配置 {label} 的料盒坐标")
        sequence = self.robot.build_grasp_sequence(target.base_pose, label)
        commands = []
        for action, value in sequence:
            item = {"action": action}
            if isinstance(value, list):
                item["pose"] = [round(float(v), 3) for v in value]
                item["tcp"] = self._tcp_preview(action, value)
            else:
                item["value"] = value
                item["tcp"] = self._tcp_preview(action, value)
            commands.append(item)
        return {
            "cmd": "tcp_grasp_sequence",
            "label": label,
            "pixel": list(target.detection.center),
            "height_mm": round(target.height_mm, 2),
            "base_pose": [round(float(v), 3) for v in target.base_pose],
            "commands": commands,
        }

    def send_grasp_command(self) -> tuple[dict, CommandReply]:
        payload = self.build_tcp_command_payload()
        reply = self.command_transport.send(payload)
        self.log(
            f"指令下发完成：mode={reply.mode} reply={reply.raw}"
        )
        return payload, reply

    def _tcp_preview(self, action: str, value) -> str | None:
        robot_cfg = self.config["robot"]
        if action in {"movj", "movl", "movl_slow"} and isinstance(value, list):
            speed = int(robot_cfg.get("speed_percent", 25))
            accel = int(robot_cfg.get("accel_percent", 25))
            if action == "movl_slow":
                speed = int(robot_cfg.get("descend_speed_percent", 8))
                accel = int(robot_cfg.get("descend_accel_percent", 10))
            user = int(robot_cfg.get("user", 0))
            tool = int(robot_cfg.get("tool", 0))
            pose_text = ",".join(f"{float(v):.3f}" for v in value)
            cmd = "MovJ" if action == "movj" else "MovL"
            return f"{cmd}(pose={{{pose_text}}},user={user},tool={tool},a={accel},v={speed},cp=0)"
        if action == "suction":
            status = int(robot_cfg.get("suction_on_level", 1) if bool(value) else robot_cfg.get("suction_off_level", 0))
            index = int(robot_cfg.get("suction_tool_do", 1))
            return f"ToolDOInstant({index},{status})"
        if action == "wait":
            return f"Wait({float(value):.3f}s)"
        return None

    def execute_grasp(self) -> None:
        self._ensure_robot_ready()
        if self.current_target is None:
            self._move_to_photo_pose_if_configured()
            self.read_frame()
        target = self.calculate_grasp()
        if not self.robot.has_target_pose(target.detection.label):
            raise RuntimeError(f"未配置 {target.detection.label} 的料盒坐标")
        sequence = self.robot.build_grasp_sequence(target.base_pose, target.detection.label)
        self._execute_sequence(sequence)
        self.log(f"单物料分拣完成：{target.detection.label}")
        self.current_target = None
        self.detections = []

    def execute_grasp_lift(self) -> TargetResult:
        """Run the task-book single cycle: detect, grasp, lift, then stop."""
        self._ensure_robot_ready()
        self.current_target = None
        self.detections = []
        self._move_to_photo_pose_if_configured()
        self.read_frame()
        self.detect_once()
        target = self.current_target
        if target is None:
            raise RuntimeError("未识别到可抓取物料")
        safe_z = float(self.config["robot"].get("safe_z_mm", 180.0))
        clearance = float(self.config["robot"].get("grasp_clearance_mm", 18.0))
        above = list(target.base_pose)
        above[2] = max(safe_z, target.base_pose[2] + clearance)
        self._execute_sequence(
            [
                ("movj", above),
                ("movl_slow", list(target.base_pose)),
                ("suction", True),
                ("wait", float(self.config["robot"].get("suction_settle_s", 0.5))),
                ("movl", above),
            ]
        )
        self._mark_simulated_removed(target.detection)
        self.log(f"单次运行完成：{target.detection.label} 已抓取并抬起")
        return target

    def _execute_sequence(self, sequence: list[tuple[str, list[float] | bool]]) -> None:
        for action, value in sequence:
            if action == "movj":
                self.robot.movj(value)
                self.robot.wait_until_pose(value)
            elif action == "movl":
                self.robot.movl(value)
                self.robot.wait_until_pose(value)
            elif action == "movl_slow":
                # 慢速垂直逼近：用更低的速度下降，防止冲击造成轴碰撞
                descend_speed = int(self.config["robot"].get("descend_speed_percent", 8))
                descend_accel = int(self.config["robot"].get("descend_accel_percent", 10))
                self.robot.movl(value, speed=descend_speed, accel=descend_accel)
                self.robot.wait_until_pose(value)
            elif action == "suction":
                self.robot.suction(bool(value))
                self.log("吸盘得电吸取" if value else "吸盘断电释放")
            elif action == "wait":
                import time

                time.sleep(float(value))
                self.log(f"等待 {float(value):.1f}s")
            self.log(f"执行：{action} {value}")

    def auto_run(self) -> int:
        self._ensure_robot_ready()
        completed = 0
        max_objects = int(self.config["workflow"]["auto_max_objects"])
        empty_frames = 0
        while completed < max_objects:
            self._move_to_photo_pose_if_configured()
            self.read_frame()
            target = self.select_auto_target()
            if target is None:
                empty_frames += 1
                if empty_frames >= int(self.config["workflow"]["auto_empty_frames_to_finish"]):
                    break
                continue
            empty_frames = 0
            self.current_target = None
            self._execute_target(target)
            completed += 1
        self.current_target = None
        self.detections = []
        self.log(f"自动分拣结束：完成 {completed} 个目标")
        return completed

    def _move_to_photo_pose_if_configured(self) -> None:
        workflow_cfg = self.config.get("workflow", {})
        if not bool(workflow_cfg.get("photo_pose_before_detection", False)):
            return
        pose = self.config.get("robot", {}).get("photo_pose")
        if not isinstance(pose, (list, tuple)) or len(pose) != 6:
            raise RuntimeError("已启用拍照点运动，但 robot.photo_pose 不是 6 轴位姿")
        photo_pose = [float(value) for value in pose]
        minimum_z = float(self.config.get("robot", {}).get("min_grasp_z_mm", 60.0))
        if photo_pose[2] < minimum_z:
            raise RuntimeError(f"拍照点 Z={photo_pose[2]:g} 低于安全下限 {minimum_z:g}mm")
        self.robot.movj(photo_pose)
        self.robot.wait_until_pose(photo_pose)
        self.log(f"机械臂已到拍照点: {[round(value, 2) for value in photo_pose]}")

    def _execute_target(self, target: Detection) -> None:
        current_pose = self.robot.get_pose() if self.robot.connected else self.robot.current_pose
        self.current_target = self._build_target_result(target, current_pose)
        object_height = self.current_target.height_mm
        base_pose = self.current_target.base_pose
        angle_deg = self.current_target.angle_deg
        self.log(
            f"识别成功：{target.label} conf={target.confidence:.2f} pixel={target.center} "
            f"height={object_height:.1f}mm angle={angle_deg:.1f}° "
            f"angle_valid={self.current_target.angle_valid} "
            f"grasp_pixel={self.current_target.grasp_pixel}"
        )
        target_label = target.label
        if not self.robot.has_target_pose(target_label):
            raise RuntimeError(f"未配置 {target_label} 的料盒坐标")
        sequence = self.robot.build_grasp_sequence(base_pose, target_label)
        self._execute_sequence(sequence)
        self._mark_simulated_removed(target)
        self.log(f"单物料分拣完成：{target_label}")

    def _apply_grasp_offsets(self, base_pose: list[float]) -> list[float]:
        robot_cfg = self.config["robot"]
        base_pose[0] += float(robot_cfg.get("grasp_x_offset_mm", 0.0))
        base_pose[1] += float(robot_cfg.get("grasp_y_offset_mm", 0.0))
        return base_pose

    def _apply_object_orientation(
        self, base_pose: list[float], label: str, image_angle_deg: float
    ) -> list[float]:
        settings = self.config.get("orientation", {})
        if not bool(settings.get("enabled", False)):
            return base_pose
        if label in settings.get("symmetric_labels", []):
            return base_pose
        if str(settings.get("angle_frame", "image")).lower() == "base":
            sign = 1.0
        else:
            sign = float(settings.get("image_to_tool_sign", 1.0))
        offset = float(settings.get("tool_yaw_offset_deg", 0.0))
        base_pose[5] = ((float(base_pose[5]) + sign * image_angle_deg + offset + 180.0) % 360.0) - 180.0
        return base_pose

    def _mark_simulated_removed(self, detection: Detection) -> None:
        self.detector.mark_simulated_removed(detection)
        marker = getattr(self.camera, "mark_simulated_removed", None)
        if callable(marker):
            marker(detection.bbox)

    def return_home(self) -> None:
        self._ensure_robot_ready()
        home_pose = list(self.config["robot"]["home_pose"])
        self.robot.movj(home_pose)
        self.robot.wait_until_pose(home_pose)
        self.log("机器人已返回安全点")

    def _ensure_robot_ready(self) -> None:
        if not self.robot.connected:
            self.connect_robot()
        if not self.robot.enabled:
            self.robot.enable()

    def emergency_stop(self) -> None:
        errors: list[str] = []
        self.robot.cancel_event.set()
        try:
            if self.robot.connected:
                try:
                    self.robot.suction(False)
                except Exception as exc:
                    errors.append(f"关闭吸盘失败: {exc}")
                try:
                    mode = str(self.config["robot"].get("emergency_stop_mode", "stop")).lower()
                    if mode == "emergency_stop":
                        self.robot.emergency_stop()
                        self.log("急停触发：已发送 EmergencyStop(1)，机器人会下使能并报警")
                    else:
                        self.robot.stop_motion()
                        self.log("急停触发：已发送 Stop()，运动队列已请求停止")
                except Exception as exc:
                    errors.append(f"停止运动失败: {exc}")
        finally:
            self.robot.enabled = False
            if errors:
                self.log("；".join(errors))
