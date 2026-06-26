from __future__ import annotations

import json
import threading
from pathlib import Path
from time import strftime

import numpy as np

from .camera import RealSenseCamera, save_capture
from .detector import FruitDetector
from .height import HeightEstimator
from .robot_client import DobotClient
from .transform import CoordinateTransformer
from .types import Detection, TargetResult
from app.sim.sim_camera import SimulatedCamera


class CompetitionWorkflow:
    def __init__(self, config: dict, logger=print):
        self.config = config
        self.log = logger
        self.simulation = config.get("mode") == "simulation"
        self.camera = SimulatedCamera(config) if self.simulation else RealSenseCamera(config)
        self.detector = FruitDetector(config, simulation=self.simulation)
        self.height_estimator = HeightEstimator(config)
        self.transformer = CoordinateTransformer(config)
        self.robot = DobotClient(config, simulation=self.simulation)
        self.frame = None
        self._frame_lock = threading.Lock()
        self.detections: list[Detection] = []
        self.current_target: TargetResult | None = None

    def start_camera(self) -> None:
        self.camera.start()
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
        self.detections = self.detector.detect(self.frame.color)
        if not self.detections:
            self.current_target = None
            self.log("未识别到有效目标")
            return []
        target = self.detections[0]
        cx, cy = target.center
        # 眼在手：必须用拍照瞬间的末端位姿做坐标转换
        if self.robot.connected:
            current_pose = self.robot.get_pose()
        else:
            current_pose = self.robot.current_pose
        cam_to_base = self.transformer.camera_to_base_matrix(current_pose)
        object_height = self.height_estimator.estimate(self.frame.depth_mm, target, self.frame.intrinsics, cam_to_base)
        depth = self._depth_at(cx, cy)
        camera_xyz = self.transformer.pixel_to_camera(cx, cy, depth, self.frame.intrinsics)
        base_pose = self.transformer.camera_to_base_pose(camera_xyz, current_pose, [180.0, 0.0, 0.0])
        base_pose = self._apply_grasp_offsets(base_pose)
        # 【抓取高度】：若已标定桌面平面，则用“桌面平面 Z + 识别到的物体高度”来算吸取点；
        # 否则回退到当前深度点对应的 base_pose[2]。这样 15mm 这种视觉高度会真正参与下发。
        raw_base_z = float(base_pose[2])
        grasp_z_offset = float(self.config["robot"].get("grasp_z_offset_mm", 0.0))
        # 【防撞下限】：抓取点 Z 不得低于配置的安全地板高度，避免高度/标定误差导致机械臂向下扎进桌面造成轴碰撞。
        min_grasp_z = float(self.config["robot"].get("min_grasp_z_mm", 60.0))
        target_z = raw_base_z + grasp_z_offset
        table_z = None
        if self.height_estimator.has_plane():
            plane = self.height_estimator.plane_base
            if plane is not None:
                a, b, c, d = plane
                if abs(c) > 1e-6:
                    table_z = -(a * base_pose[0] + b * base_pose[1] + d) / c
                    target_z = table_z + object_height + grasp_z_offset
        base_pose[2] = max(min_grasp_z, target_z)
        z_note = "  ⚠地板兜底已生效" if target_z < min_grasp_z else ""
        self.current_target = TargetResult(target, object_height, camera_xyz, base_pose)
        self.log(f"识别成功：{target.label} conf={target.confidence:.2f} pixel={target.center} height={object_height:.1f}mm")
        self.log(f"  拍照位姿：{[round(v,2) for v in current_pose]}  深度：{depth:.1f}mm  相机坐标：({camera_xyz[0]:.1f},{camera_xyz[1]:.1f},{camera_xyz[2]:.1f})")
        self.log(f"  基坐标解算：{[round(v,2) for v in base_pose]}")
        if table_z is not None:
            self.log(
                f"  Z分解：桌面Z={table_z:.1f}mm + 物体高={object_height:.1f}mm + offset={grasp_z_offset:+.1f} "
                f"→ 目标Z={target_z:.1f}mm  最终下发={base_pose[2]:.1f}mm{z_note}"
            )
        else:
            self.log(f"  Z分解：原始顶面={raw_base_z:.1f}mm + offset={grasp_z_offset:+.1f} → 目标Z={target_z:.1f}mm  最终下发={base_pose[2]:.1f}mm{z_note}")
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

    def select_auto_target(self) -> Detection | None:
        if self.frame is None:
            self.read_frame()
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
        if not bins:
            raise RuntimeError("未配置任何置物盒坐标，无法完成定点抓取测试")
        target_label = next(iter(bins.keys()))
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

    def execute_grasp(self) -> None:
        self._ensure_robot_ready()
        target = self.calculate_grasp()
        if target.detection.label not in self.config["robot"]["bins"]:
            raise RuntimeError(f"未配置 {target.detection.label} 的料盒坐标")
        sequence = self.robot.build_grasp_sequence(target.base_pose, target.detection.label)
        self._execute_sequence(sequence)
        self.log(f"单物料分拣完成：{target.detection.label}")

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
        self.log(f"自动分拣结束：完成 {completed} 个目标")
        return completed

    def _execute_target(self, target: Detection) -> None:
        current_pose = self.robot.get_pose() if self.robot.connected else self.robot.current_pose
        cam_to_base = self.transformer.camera_to_base_matrix(current_pose)
        object_height = self.height_estimator.estimate(self.frame.depth_mm, target, self.frame.intrinsics, cam_to_base)
        cx, cy = target.center
        depth = self._depth_at(cx, cy)
        camera_xyz = self.transformer.pixel_to_camera(cx, cy, depth, self.frame.intrinsics)
        base_pose = self.transformer.camera_to_base_pose(camera_xyz, current_pose, [180.0, 0.0, 0.0])
        base_pose = self._apply_grasp_offsets(base_pose)
        raw_base_z = float(base_pose[2])
        grasp_z_offset = float(self.config["robot"].get("grasp_z_offset_mm", 0.0))
        min_grasp_z = float(self.config["robot"].get("min_grasp_z_mm", 60.0))
        target_z = raw_base_z + grasp_z_offset
        table_z = None
        if self.height_estimator.has_plane():
            plane = self.height_estimator.plane_base
            if plane is not None:
                a, b, c, d = plane
                if abs(c) > 1e-6:
                    table_z = -(a * base_pose[0] + b * base_pose[1] + d) / c
                    target_z = table_z + object_height + grasp_z_offset
        base_pose[2] = max(min_grasp_z, target_z)
        self.current_target = TargetResult(target, object_height, camera_xyz, base_pose)
        self.log(f"识别成功：{target.label} conf={target.confidence:.2f} pixel={target.center} height={object_height:.1f}mm")
        if table_z is not None:
            self.log(
                f"  Z分解：桌面Z={table_z:.1f}mm + 物体高={object_height:.1f}mm + offset={grasp_z_offset:+.1f} "
                f"→ 目标Z={target_z:.1f}mm  最终下发={base_pose[2]:.1f}mm"
            )
        target_label = target.label
        if target_label not in self.config["robot"]["bins"]:
            raise RuntimeError(f"未配置 {target_label} 的料盒坐标")
        sequence = self.robot.build_grasp_sequence(base_pose, target_label)
        self._execute_sequence(sequence)
        self.log(f"单物料分拣完成：{target_label}")

    def _apply_grasp_offsets(self, base_pose: list[float]) -> list[float]:
        robot_cfg = self.config["robot"]
        base_pose[0] += float(robot_cfg.get("grasp_x_offset_mm", 0.0))
        base_pose[1] += float(robot_cfg.get("grasp_y_offset_mm", 0.0))
        base_pose[2] += float(robot_cfg.get("grasp_z_offset_mm", 0.0))
        return base_pose

    def _ensure_robot_ready(self) -> None:
        if not self.robot.connected:
            self.connect_robot()
        if not self.robot.enabled:
            self.robot.enable()

    def emergency_stop(self) -> None:
        errors: list[str] = []
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
