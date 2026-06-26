from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from app.config import load_config
from app.core.workflow import CompetitionWorkflow

try:
    import cv2
    from PyQt5.QtCore import QObject, QThread, Qt, QTimer, pyqtSignal
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    IMPORT_ERROR = exc
    cv2 = None
    QObject = QThread = pyqtSignal = None
    QTimer = Qt = QImage = QPixmap = QApplication = QGridLayout = QGroupBox = QLabel = (
        None
    )
    QMessageBox = QPushButton = QScrollArea = QSizePolicy = QTextEdit = QVBoxLayout = (
        QHBoxLayout
    ) = QDialog = QLineEdit = QFormLayout = None

    class QWidget:  # type: ignore[no-redef]
        pass
else:
    IMPORT_ERROR = None


class WorkflowWorker(QObject):
    log = pyqtSignal(str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, action):
        super().__init__()
        self.action = action

    def run(self):
        try:
            self.finished.emit(self.action())
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.workflow = CompetitionWorkflow(self.config, logger=self.append_log)
        self.last_frame_time = None
        self.current_fps = 0.0
        self.camera_ready = False
        self.tcp_ready = False
        self.robot_ready = False
        self.suction_on = False
        self.running_mode = (
            "模拟" if self.config.get("mode") == "simulation" else "手动"
        )
        self.recording = False
        self.video_writer = None
        self.video_record_path = None
        self.current_joint_angles: list[float] | None = None
        self._last_robot_poll = 0.0
        self._last_io_poll = 0.0
        self.worker_thread: QThread | None = None
        self.worker: WorkflowWorker | None = None
        self.task_labels: list[QLabel] = []
        self.status_indicators: dict[str, QLabel] = {}
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.refresh_footer)
        self.clock_timer.start(1000)
        self.init_ui()
        self.refresh_all_status()

    def workflow_log(self, message: str) -> None:
        self.append_log(message)

    def init_ui(self) -> None:
        self.setWindowTitle("2026睿抗机器人大赛黑龙江赛区 - 机器视觉分拣系统")
        self.resize(1540, 920)
        self.setMinimumSize(1280, 780)
        self.setStyleSheet(
            """
            QWidget { background:#f4f7fb; color:#1f2937; font-family:'Microsoft YaHei','Segoe UI'; font-size:14px; }
            QGroupBox { background:#ffffff; border:1px solid #d6dee9; border-radius:10px; margin-top:18px; padding:12px; font-weight:700; }
            QGroupBox::title { subcontrol-origin: margin; left:14px; padding:0 8px; color:#0f4c81; }
            QPushButton { background:#e8f1fb; border:1px solid #a9c3df; border-radius:8px; padding:8px 10px; font-weight:600; }
            QPushButton:hover { background:#d6e9fb; }
            QPushButton:pressed { background:#b8d7f5; }
            QTextEdit { background:#0f172a; color:#d1fae5; border-radius:8px; font-family:Consolas,'Microsoft YaHei'; }
            QScrollArea { border:0; background:transparent; }
            QScrollBar:vertical { background:#e5edf6; width:12px; border-radius:6px; }
            QScrollBar::handle:vertical { background:#8fb2d4; border-radius:6px; min-height:36px; }
            QScrollBar:horizontal { background:#e5edf6; height:12px; border-radius:6px; }
            QScrollBar::handle:horizontal { background:#8fb2d4; border-radius:6px; min-width:36px; }
            QLabel#videoLabel { background:#111827; color:#d1d5db; border:2px solid #334155; border-radius:8px; font-size:20px; }
            QLabel#metricCard { background:#eef6ff; border:1px solid #c9ddf2; border-radius:8px; padding:8px; font-size:15px; }
            QLabel#footer { background:#102033; color:#f8fafc; padding:8px 12px; border-radius:6px; font-weight:600; }
            """
        )

        root = QVBoxLayout()
        top = QHBoxLayout()
        top.addWidget(self.build_video_area(), stretch=8)
        top.addWidget(self.build_control_area(), stretch=4)
        root.addLayout(top, stretch=7)
        root.addWidget(self.build_status_area(), stretch=3)
        self.footer_label = QLabel()
        self.footer_label.setObjectName("footer")
        root.addWidget(self.footer_label)
        self.setLayout(root)
        self.append_log("系统初始化完成：非全屏窗口模式，等待打开相机")

    def build_video_area(self) -> QGroupBox:
        video_box = QGroupBox("A区 双摄监控")
        layout = QGridLayout()
        self.rgb_label = QLabel("RGB画面待启动")
        self.depth_label = QLabel("Depth画面待启动")
        for label in (self.rgb_label, self.depth_label):
            label.setObjectName("videoLabel")
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(430, 300)
            label.setScaledContents(False)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(0, 1)
        layout.addWidget(self.rgb_label, 0, 0)
        layout.addWidget(self.depth_label, 0, 1)
        video_box.setLayout(layout)
        return video_box

    def build_control_area(self) -> QGroupBox:
        control_box = QGroupBox("B区 控制台")
        control_box.setMinimumWidth(480)
        control_box.setMaximumWidth(540)
        outer = QVBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(10)
        grid_layout.setVerticalSpacing(8)
        grid_layout.setContentsMargins(4, 4, 10, 4)
        tasks = [
            ("任务1 系统唤醒", [("打开相机", self.on_start_camera)]),
            (
                "任务2 图像采集",
                [
                    ("保存图像", self.on_save_image),
                    ("开始录制", self.on_toggle_recording),
                ],
            ),
            ("任务3 单目标识别", [("单次识别", self.on_detect_once)]),
            ("任务4 固定点位", [("定点测试", self.on_fixed_test)]),
            (
                "任务5 坐标握手",
                [
                    ("机器人连接", self.on_connect_robot),
                    ("计算抓取坐标", self.on_calculate_grasp),
                    ("指令下发", self.on_send_command),
                ],
            ),
            ("任务6 单物料分拣", [("执行抓取", self.on_execute_grasp)]),
            ("任务7 连续分拣", [("一键自动运行", self.on_auto_run)]),
        ]
        for idx, (title, buttons) in enumerate(tasks):
            group = QGroupBox(title)
            group.setMinimumHeight(92)
            group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button_layout = QGridLayout()
            button_layout.setContentsMargins(8, 14, 8, 8)
            for btn_idx, (text, slot) in enumerate(buttons):
                btn = QPushButton(text)
                btn.setMinimumHeight(34)
                btn.clicked.connect(slot)
                if text == "开始录制":
                    self.record_button = btn
                button_layout.addWidget(btn, btn_idx, 0)
            group.setLayout(button_layout)
            grid_layout.addWidget(group, idx // 2, idx % 2)

        function_group = QGroupBox("功能区")
        function_group.setMinimumHeight(168)
        function_grid = QGridLayout()
        function_grid.setContentsMargins(8, 16, 8, 8)
        function_buttons = [
            ("清空日志", self.on_clear_log),
            ("标定桌面", self.on_calibrate_table),
            ("参数设置", self.on_parameter_settings),
            ("相机参数", self.on_camera_parameters),
            ("校准标定", self.on_calibration_tools),
            ("复位机器人", self.on_reset_robot),
        ]
        for idx, (text, slot) in enumerate(function_buttons):
            btn = QPushButton(text)
            btn.setMinimumHeight(36)
            btn.clicked.connect(slot)
            function_grid.addWidget(btn, idx // 2, idx % 2)
        function_group.setLayout(function_grid)
        grid_layout.addWidget(function_group, 4, 0, 1, 2)
        grid_layout.setRowStretch(5, 1)
        content.setLayout(grid_layout)
        scroll.setWidget(content)
        outer.addWidget(scroll, stretch=1)

        self.stop_button = QPushButton("急停 STOP")
        self.stop_button.setMinimumHeight(64)
        self.stop_button.setStyleSheet(
            "QPushButton { background:#dc2626; color:white; border:3px solid #7f1d1d; border-radius:12px; font-size:26px; font-weight:900; }"
            "QPushButton:hover { background:#b91c1c; }"
        )
        self.stop_button.clicked.connect(self.on_emergency_stop)
        outer.addWidget(self.stop_button)
        control_box.setLayout(outer)
        return control_box

    def build_status_area(self) -> QGroupBox:
        area = QGroupBox("C区 状态与日志")
        root = QHBoxLayout()
        progress = self.build_progress_box()
        progress.setMinimumWidth(220)
        progress.setMaximumWidth(280)
        right = QWidget()
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.addWidget(self.build_recognition_box(), 0, 0)
        grid.addWidget(self.build_base_box(), 0, 1)
        grid.addWidget(self.build_system_status_box(), 0, 2)
        grid.addWidget(self.build_robot_data_box(), 1, 0, 1, 2)
        grid.addWidget(self.build_log_box(), 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 3)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 2)
        right.setLayout(grid)
        root.addWidget(progress)
        root.addWidget(right, stretch=1)
        area.setLayout(root)
        return area

    def build_recognition_box(self) -> QGroupBox:
        box = QGroupBox("识别结果")
        layout = QVBoxLayout()
        self.recognition_label = QLabel("类别：-\n置信度：-\n像素坐标：-\n高度：-")
        self.recognition_label.setObjectName("metricCard")
        layout.addWidget(self.recognition_label)
        box.setLayout(layout)
        return box

    def build_base_box(self) -> QGroupBox:
        box = QGroupBox("基坐标系坐标")
        layout = QVBoxLayout()
        self.base_coord_label = QLabel("X：- mm\nY：- mm\nZ：- mm\nRx/Ry/Rz：-")
        self.base_coord_label.setObjectName("metricCard")
        layout.addWidget(self.base_coord_label)
        box.setLayout(layout)
        return box

    def build_system_status_box(self) -> QGroupBox:
        box = QGroupBox("系统状态")
        layout = QGridLayout()
        items = ["相机状态", "TCP状态", "机器人状态", "吸盘状态", "运行模式"]
        for row, name in enumerate(items):
            layout.addWidget(QLabel(name), row, 0)
            indicator = QLabel("● 未就绪")
            indicator.setStyleSheet("color:#dc2626;font-weight:800;font-size:16px;")
            self.status_indicators[name] = indicator
            layout.addWidget(indicator, row, 1)
        box.setLayout(layout)
        return box

    def build_progress_box(self) -> QGroupBox:
        box = QGroupBox("任务进度")
        layout = QVBoxLayout()
        labels = [
            "1 系统唤醒：待执行",
            "2 图像采集：待执行",
            "3 识别测高：待执行",
            "4 定点运动：待执行",
            "5 坐标握手：待执行",
            "6 单物料分拣：待执行",
            "7 连续分拣：待执行",
        ]
        for text in labels:
            label = QLabel(text)
            label.setObjectName("metricCard")
            self.task_labels.append(label)
            layout.addWidget(label)
        box.setLayout(layout)
        return box

    def build_robot_data_box(self) -> QGroupBox:
        box = QGroupBox("机器人关节角度 / TCP位姿")
        layout = QVBoxLayout()
        self.joint_label = QLabel("J1：-°  J2：-°  J3：-°\nJ4：-°  J5：-°  J6：-°")
        self.joint_label.setObjectName("metricCard")
        self.tcp_pose_label = QLabel("TCP X/Y/Z：-\nTCP Rx/Ry/Rz：-")
        self.tcp_pose_label.setObjectName("metricCard")
        layout.addWidget(self.joint_label)
        layout.addWidget(self.tcp_pose_label)
        box.setLayout(layout)
        return box

    def build_log_box(self) -> QGroupBox:
        box = QGroupBox("日志窗口")
        layout = QVBoxLayout()
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setLineWrapMode(QTextEdit.NoWrap)
        self.log_box.setMinimumHeight(110)
        self.log_box.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.log_box.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        layout.addWidget(self.log_box)
        box.setLayout(layout)
        return box

    def append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        if hasattr(self, "log_box"):
            self.log_box.append(f"[{timestamp}] {message}")
        else:
            print(f"[{timestamp}] {message}")

    def guard(self, action):
        try:
            action()
            self.refresh_all_status()
        except Exception as exc:
            self.tcp_ready = self.workflow.robot.connected
            self.robot_ready = self.workflow.robot.enabled
            self.append_log("错误：" + str(exc))
            QMessageBox.warning(self, "执行失败", str(exc))
            self.refresh_all_status()

    def _confirm_pose_source(self, action_name: str) -> bool:
        """眼在手：坐标/测高都依赖拍照瞬间的真实末端位姿。

        模拟模式或已连接机械臂时直接放行；真机但未连接时，说明当前会用默认
        home_pose 代替真实位姿（结果只有在机械臂确实停在 home_pose 时才正确），
        让用户决定是否继续。返回 True 表示继续，False 表示中止。
        """
        if self.config.get("mode") == "simulation":
            return True
        if self.workflow.robot.connected:
            return True
        home = self.config["robot"].get("home_pose")
        reply = QMessageBox.question(
            self,
            "机械臂未连接",
            f"{action_name}依赖拍照瞬间的真实末端位姿，但机械臂尚未连接。\n\n"
            f"当前将用默认 home_pose（{home}）代替真实位姿，\n"
            "结果只有在机械臂确实停在该位姿时才正确，否则坐标会偏。\n\n"
            "建议先点“机器人连接”。是否仍要继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            self.append_log(f"{action_name}已取消：机械臂未连接，避免误用 home_pose")
            return False
        self.append_log(
            f"警告：{action_name}在未连接机械臂下执行，使用默认 home_pose，坐标可能不准"
        )
        return True

    def on_start_camera(self):
        def run():
            self.workflow.start_camera()
            self.camera_ready = True
            self.update_task(1, "完成")
            self.timer.start(33)
            self.append_log("任务1完成：相机双流启动")

        self.guard(run)

    def on_save_image(self):
        def run():
            self.workflow.save_current_frame()
            self.update_task(2, "完成")

        self.guard(run)

    def on_toggle_recording(self):
        def run():
            if not self.recording:
                if self.workflow.frame is None:
                    self.workflow.read_frame()
                frame = self.workflow.frame
                height, width = frame.color.shape[:2]
                output_dir = (
                    Path(__file__).resolve().parents[1]
                    / "runs"
                    / "videos"
                    / datetime.now().strftime("%Y%m%d")
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                self.video_record_path = (
                    output_dir / f"rgb_{datetime.now().strftime('%H%M%S')}.mp4"
                )
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps = max(15.0, min(30.0, float(self.config["camera"].get("fps", 30))))
                self.video_writer = cv2.VideoWriter(
                    str(self.video_record_path), fourcc, fps, (width, height)
                )
                if not self.video_writer.isOpened():
                    self.video_writer = None
                    raise RuntimeError(
                        "无法创建视频文件，请检查写入权限或 OpenCV 编码器"
                    )
                self.recording = True
                self.record_button.setText("停止录制")
                self.record_button.setStyleSheet(
                    "background:#f59e0b;color:#111827;border:1px solid #92400e;border-radius:8px;font-weight:800;"
                )
                self.append_log(f"开始录制RGB视频：{self.video_record_path}")
            else:
                stopped_path = self.video_record_path
                self.stop_recording()
                self.record_button.setText("开始录制")
                self.record_button.setStyleSheet("")
                self.append_log(f"停止录制RGB视频：{stopped_path}")
                self.update_task(2, "图像/视频已采集")

        self.guard(run)

    def stop_recording(self) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
        self.video_writer = None
        self.recording = False

    def on_detect_once(self):
        def run():
            if not self._confirm_pose_source("单次识别"):
                return
            self.workflow.detect_once()
            self.refresh_target_info()
            self.update_task(3, "完成")

        self.guard(run)

    def on_connect_robot(self):
        def run():
            self.tcp_ready = False
            self.robot_ready = False
            self.workflow.connect_robot()
            self.tcp_ready = self.workflow.robot.connected
            self.robot_ready = self.workflow.robot.enabled
            self.update_robot_pose_labels(force=True)
            self.append_log("机器人 TCP 已连接，使能完成")

        self.guard(run)

    def on_fixed_test(self):
        if self.worker_thread is not None and self.worker_thread.isRunning():
            self.append_log("当前已有后台任务在运行，请稍后再试")
            return

        self.workflow.robot.reset_cancel()
        self.running_mode = "手动"
        self.refresh_all_status()
        self.append_log("定点测试已启动：后台执行，GUI 保持响应")

        self.worker_thread = QThread(self)
        self.worker = WorkflowWorker(self.workflow.fixed_point_test)
        self.workflow.log = self.worker.log.emit
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self.on_fixed_test_finished)
        self.worker.failed.connect(self.on_fixed_test_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self.clear_worker)
        self.worker_thread.start()

    def on_fixed_test_finished(self, _result) -> None:
        self.update_robot_pose_labels(force=True)
        self.refresh_all_status()
        self.update_task(4, "完成")
        self.append_log("定点测试完成")

    def on_fixed_test_failed(self, message: str) -> None:
        self.robot_ready = self.workflow.robot.enabled
        self.refresh_all_status()
        self.append_log("定点测试错误：" + message)
        QMessageBox.warning(self, "定点测试失败", message)

    def on_calculate_grasp(self):
        def run():
            if self.workflow.current_target is None and not self._confirm_pose_source(
                "计算抓取坐标"
            ):
                return
            self.workflow.calculate_grasp()
            self.refresh_target_info()
            self.update_task(5, "完成")

        self.guard(run)

    def on_send_command(self):
        def run():
            if self.workflow.current_target is None and not self._confirm_pose_source(
                "指令下发"
            ):
                return
            target = self.workflow.calculate_grasp()
            self.refresh_target_info()
            self.append_log(
                "指令下发预览：将发送抓取位姿 "
                + str([round(v, 3) for v in target.base_pose])
            )
            self.update_task(5, "已下发")

        self.guard(run)

    def on_execute_grasp(self):
        if self.worker_thread is not None and self.worker_thread.isRunning():
            self.append_log("当前已有后台任务在运行，请稍后再试")
            return

        self.workflow.robot.reset_cancel()
        self.running_mode = "手动"
        self.refresh_all_status()
        self.append_log("单物料分拣已启动：后台执行，GUI 保持响应")

        self.worker_thread = QThread(self)
        self.worker = WorkflowWorker(self.workflow.execute_grasp)
        self.workflow.log = self.worker.log.emit
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self.on_execute_grasp_finished)
        self.worker.failed.connect(self.on_execute_grasp_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self.clear_worker)
        self.worker_thread.start()

    def on_execute_grasp_finished(self, _result) -> None:
        self.suction_on = False
        self.update_robot_pose_labels(force=True)
        self.refresh_target_info()
        self.refresh_all_status()
        self.update_task(6, "完成")
        self.append_log("单物料分拣完成")

    def on_execute_grasp_failed(self, message: str) -> None:
        self.suction_on = False
        self.robot_ready = self.workflow.robot.enabled
        self.refresh_all_status()
        self.append_log("单物料分拣错误：" + message)
        QMessageBox.warning(self, "单物料分拣失败", message)

    def on_auto_run(self):
        if self.worker_thread is not None and self.worker_thread.isRunning():
            self.append_log("自动分拣正在运行，请等待当前任务结束")
            return

        self.workflow.robot.reset_cancel()
        self.running_mode = "自动"
        self.refresh_all_status()
        self.append_log("自动分拣已启动：后台执行，GUI 保持响应")

        self.worker_thread = QThread(self)
        self.worker = WorkflowWorker(self.workflow.auto_run)
        self.workflow.log = self.worker.log.emit
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self.on_auto_run_finished)
        self.worker.failed.connect(self.on_auto_run_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self.clear_worker)
        self.worker_thread.start()

    def on_auto_run_finished(self, count: int) -> None:
        self.suction_on = False
        self.running_mode = "手动" if self.config.get("mode") != "simulation" else "模拟"
        self.update_robot_pose_labels(force=True)
        self.update_task(7, f"完成 {count} 个")
        self.refresh_target_info()
        self.refresh_all_status()
        self.append_log(f"自动分拣完成：{count} 个目标")
        QMessageBox.information(self, "分拣完成", f"自动分拣完成：{count} 个目标")

    def on_auto_run_failed(self, message: str) -> None:
        self.suction_on = False
        self.running_mode = "手动" if self.config.get("mode") != "simulation" else "模拟"
        self.robot_ready = self.workflow.robot.enabled
        self.refresh_all_status()
        self.append_log("自动分拣错误：" + message)
        QMessageBox.warning(self, "自动分拣失败", message)

    def clear_worker(self) -> None:
        self.workflow.log = self.workflow_log
        self.worker = None
        self.worker_thread = None

    def on_emergency_stop(self):
        try:
            self.workflow.emergency_stop()
        except Exception as exc:
            self.append_log("急停错误：" + str(exc))
        finally:
            self.workflow.robot.cancel_event.set()
            self.suction_on = False
            self.robot_ready = self.workflow.robot.enabled
            self.running_mode = "急停"
            self.refresh_all_status()
            self.append_log("! 急停 STOP 已触发")

    def on_clear_log(self):
        self.log_box.clear()
        self.append_log("日志已清空")

    def on_parameter_settings(self):
        self.show_config_dialog(
            "参数设置",
            {
                "运行模式": str(self.config.get("mode")),
                "机器人IP": str(self.config["robot"]["ip"]),
                "速度比例": str(self.config["robot"]["speed_percent"]),
                "加速度比例": str(self.config["robot"]["accel_percent"]),
            },
        )

    def on_camera_parameters(self):
        intr = self.workflow.frame.intrinsics if self.workflow.frame else None
        data = {
            "分辨率": f"{self.config['camera']['width']} x {self.config['camera']['height']}",
            "帧率设置": str(self.config["camera"]["fps"]),
            "fx/fy": f"{intr.fx:.2f} / {intr.fy:.2f}" if intr else "未启动",
            "cx/cy": f"{intr.cx:.2f} / {intr.cy:.2f}" if intr else "未启动",
        }
        self.show_config_dialog("相机参数", data)

    def on_calibrate_table(self):
        def run():
            if not self.camera_ready:
                raise RuntimeError("请先打开相机再标定桌面")
            if not self._confirm_pose_source("标定桌面"):
                return
            self.workflow.read_frame()
            ok = self.workflow.calibrate_table()
            if ok:
                self.append_log(
                    "桌面标定完成：平面已存入基坐标系，机械臂移动后无需重标"
                )
                QMessageBox.information(
                    self,
                    "标定桌面",
                    "桌面标定成功。\n\n平面已存入机器人基坐标系，之后移动机械臂无需重新标定。\n仅在挪动桌面或更换相机安装位置后需要重标。",
                )
            else:
                self.append_log("桌面标定失败：未能拟合平面，已退回标量高度估计")
                QMessageBox.warning(
                    self,
                    "标定桌面",
                    "未能拟合出桌面平面（可能桌面有效深度点太少或物体过多）。\n请清空桌面后重试，当前将退回标量高度估计。",
                )

        self.guard(run)

    def on_calibration_tools(self):
        matrix = self.workflow.transformer.t_camera_to_gripper
        QMessageBox.information(
            self,
            "校准标定",
            "当前已加载手眼矩阵：\n"
            + np.array2string(matrix, precision=4, suppress_small=True),
        )
        self.append_log(
            "校准标定：已显示当前手眼矩阵，采集/求解请使用 scripts/calib_collect.py 与 scripts/calib_solve.py"
        )

    def on_reset_robot(self):
        def run():
            self.workflow.robot.movj(list(self.config["robot"]["home_pose"]))
            self.update_robot_pose_labels(force=True)
            self.append_log("机器人已复位到 home_pose")

        self.guard(run)

    def show_config_dialog(self, title: str, data: dict[str, str]) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        layout = QFormLayout()
        for key, value in data.items():
            field = QLineEdit(value)
            field.setReadOnly(True)
            layout.addRow(key, field)
        dialog.setLayout(layout)
        dialog.resize(420, 220)
        dialog.exec_()

    def update_task(self, index: int, status: str) -> None:
        if 1 <= index <= len(self.task_labels):
            text = self.task_labels[index - 1].text().split("：", 1)[0]
            self.task_labels[index - 1].setText(f"{text}：{status}")
            self.task_labels[index - 1].setStyleSheet(
                "background:#dcfce7;border:1px solid #86efac;border-radius:8px;padding:8px;color:#166534;font-weight:700;"
            )

    def refresh_target_info(self) -> None:
        target = self.workflow.current_target
        if target is None:
            self.recognition_label.setText("类别：-\n置信度：-\n像素坐标：-\n高度：-")
            self.base_coord_label.setText("X：- mm\nY：- mm\nZ：- mm\nRx/Ry/Rz：-")
            return
        det = target.detection
        self.recognition_label.setText(
            f"类别：{det.label}\n置信度：{det.confidence:.2f}\n像素坐标：{det.center}\n高度：{target.height_mm:.1f} mm"
        )
        pose = target.base_pose
        self.base_coord_label.setText(
            f"X：{pose[0]:.1f} mm\nY：{pose[1]:.1f} mm\nZ：{pose[2]:.1f} mm\nRx/Ry/Rz：{pose[3]:.1f}, {pose[4]:.1f}, {pose[5]:.1f}"
        )

    def update_robot_pose_labels(self, force: bool = False) -> None:
        now = perf_counter()
        robot = self.workflow.robot
        if force or now - self._last_robot_poll >= 0.5:
            self._last_robot_poll = now
            if robot.simulation:
                self.current_joint_angles = [0.0, -12.0, 38.0, 0.0, 52.0, 0.0]
            else:
                try:
                    robot.current_pose = robot.get_pose()
                except Exception:
                    pass
                try:
                    if robot.robot_mode() == 5:
                        self.current_joint_angles = robot.get_angles()
                except Exception:
                    pass
        if force or now - self._last_io_poll >= 0.2:
            self._last_io_poll = now
            try:
                tool_index = int(self.config["robot"].get("suction_tool_do", 1))
                on_level = int(self.config["robot"].get("suction_on_level", 1))
                self.suction_on = robot.get_tool_do(tool_index) == on_level
            except Exception:
                pass

        pose = robot.current_pose
        self.tcp_pose_label.setText(
            f"TCP X/Y/Z：{pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f}\nTCP Rx/Ry/Rz：{pose[3]:.1f}, {pose[4]:.1f}, {pose[5]:.1f}"
        )
        if self.current_joint_angles is not None:
            angles = self.current_joint_angles
            self.joint_label.setText(
                f"J1：{angles[0]:.1f}°  J2：{angles[1]:.1f}°  J3：{angles[2]:.1f}°\nJ4：{angles[3]:.1f}°  J5：{angles[4]:.1f}°  J6：{angles[5]:.1f}°"
            )
        else:
            self.joint_label.setText(
                "J1：读取失败  J2：读取失败  J3：读取失败\nJ4：读取失败  J5：读取失败  J6：读取失败"
            )

    def refresh_all_status(self) -> None:
        self.set_indicator(
            "相机状态", self.camera_ready, "运行中" if self.camera_ready else "未启动"
        )
        self.set_indicator(
            "TCP状态",
            self.tcp_ready or self.workflow.robot.connected,
            "已连接" if (self.tcp_ready or self.workflow.robot.connected) else "未连接",
        )
        self.set_indicator(
            "机器人状态",
            self.robot_ready or self.workflow.robot.enabled,
            "已使能" if (self.robot_ready or self.workflow.robot.enabled) else "未使能",
        )
        self.set_indicator(
            "吸盘状态", self.suction_on, "吸附中" if self.suction_on else "关闭"
        )
        mode_ok = self.running_mode not in {"急停"}
        self.set_indicator("运行模式", mode_ok, self.running_mode)
        self.refresh_footer()

    def set_indicator(self, name: str, ok: bool, text: str) -> None:
        label = self.status_indicators.get(name)
        if not label:
            return
        color = "#16a34a" if ok else "#dc2626"
        label.setText("● " + text)
        label.setStyleSheet(f"color:{color};font-weight:900;font-size:16px;")

    def refresh_footer(self) -> None:
        if not hasattr(self, "footer_label"):
            return
        ready = self.camera_ready and (
            self.workflow.robot.connected or self.config.get("mode") == "simulation"
        )
        ready_text = "系统就绪" if ready else "系统未就绪"
        address = (
            f"{self.config['robot']['ip']}:{self.config['robot']['dashboard_port']}"
        )
        comm = "已连接" if self.workflow.robot.connected else "未连接"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.footer_label.setText(
            f"{ready_text}    通信地址：{address}    通信状态：{comm}    当前时间：{now}"
        )

    def update_frame(self) -> None:
        try:
            now = perf_counter()
            if self.last_frame_time is not None:
                delta = max(1e-6, now - self.last_frame_time)
                instant_fps = 1.0 / delta
                self.current_fps = (
                    instant_fps
                    if self.current_fps == 0
                    else self.current_fps * 0.85 + instant_fps * 0.15
                )
            self.last_frame_time = now
            frame = self.workflow.read_frame()
            color = frame.color.copy()
            for det in self.workflow.detections:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(color, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    color,
                    f"{det.label} {det.confidence:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            self.draw_fps(color, f"RGB FPS {self.current_fps:.1f}")
            if self.recording and self.video_writer is not None:
                self.video_writer.write(frame.color)
            self.rgb_label.setPixmap(self.to_pixmap(color, self.rgb_label, is_bgr=True))
            depth = frame.depth_mm
            valid = depth[np.isfinite(depth) & (depth > 0)]
            max_depth = float(np.percentile(valid, 95)) if valid.size else 1000.0
            depth_vis = np.clip(depth / max_depth * 255, 0, 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            self.draw_fps(depth_color, f"DEPTH FPS {self.current_fps:.1f}")
            self.depth_label.setPixmap(
                self.to_pixmap(depth_color, self.depth_label, is_bgr=True)
            )
            self.refresh_target_info()
            self.update_robot_pose_labels(force=True)
            self.refresh_all_status()
            self.refresh_footer()
        except Exception as exc:
            self.append_log("画面刷新失败：" + str(exc))
            self.timer.stop()

    def draw_fps(self, image: np.ndarray, text: str) -> None:
        h, w = image.shape[:2]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        x = max(8, w - tw - 18)
        y = 28
        cv2.rectangle(image, (x - 8, y - th - 8), (w - 6, y + 8), (0, 0, 0), -1)
        cv2.putText(
            image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
        )

    def closeEvent(self, event):
        self.stop_recording()
        try:
            self.workflow.camera.stop()
        except Exception:
            pass
        event.accept()

    def to_pixmap(
        self, image: np.ndarray, target_label: QLabel, is_bgr: bool = True
    ) -> QPixmap:
        if is_bgr:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        qimage = QImage(
            image.data, width, height, image.strides[0], QImage.Format_RGB888
        )
        target_size = target_label.contentsRect().size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            target_size = target_label.size()
        return QPixmap.fromImage(qimage).scaled(
            target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )


def main() -> int:
    if QApplication is None or cv2 is None:
        print("启动失败：缺少 GUI 运行依赖。")
        print("请在当前 Python 环境安装：pip install PyQt5 opencv-python numpy PyYAML")
        if IMPORT_ERROR is not None:
            print(f"具体错误：{IMPORT_ERROR}")
        return 1
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.showMaximized()
        return app.exec_()
    except Exception as exc:
        print(f"启动失败：{exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
