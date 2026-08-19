from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from app.main import IMPORT_ERROR, MainWindow, WorkflowWorker, cv2
from app.core.task_output_receiver import TaskOutputMessage, TcpTaskOutputReceiver

try:
    from PyQt5.QtCore import QThread, QTimer, Qt
    from PyQt5.QtGui import QColor, QFont, QPainter, QPen
    from PyQt5.QtWidgets import (
        QApplication,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
    )
except ImportError:
    QApplication = None


ROOT = Path(__file__).resolve().parents[1]
TODAY_CONFIG = ROOT / "configs" / "today.yaml"


class TodayMainWindow(MainWindow):
    """Scoring-oriented GUI for the 2026-08-18 national task."""

    TABLE_HEADERS = ["序号", "物料类别", "置信度", "X/mm", "Y/mm", "角度/°", "高度/mm"]

    def __init__(self):
        self.monitor_fields: dict[str, QLineEdit] = {}
        self._output_mtimes: dict[Path, int] = {}
        self._last_table_signature: tuple | None = None
        self._task_tcp_last_status: tuple | None = None
        super().__init__(TODAY_CONFIG)
        self.task_output_receiver = TcpTaskOutputReceiver(self.config)
        self.task_output_timer = QTimer(self)
        self.task_output_timer.timeout.connect(self._drain_task_tcp)
        self.task_output_timer.start(100)
        self.task_output_receiver.start()
        self.setWindowTitle("RAICOM")
        self.append_log("系统正在初始化")

    def init_ui(self) -> None:
        self.setWindowTitle("RAICOM")
        self.resize(1500, 900)
        self.setMinimumSize(1280, 760)
        self.setStyleSheet(
            """
            QWidget { background:#eef2f5; color:#17202a; font-family:'Microsoft YaHei','Segoe UI'; font-size:13px; }
            QGroupBox { background:#ffffff; border:1px solid #b9c3cc; border-radius:6px; margin-top:15px; padding:9px; font-weight:700; }
            QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 5px; }
            QPushButton { background:#e6edf3; border:1px solid #8fa3b5; border-radius:4px; padding:8px; font-weight:700; }
            QPushButton:hover { background:#d4e3ee; }
            QLabel#videoLabel { background:#111820; color:#dbe4ea; border:1px solid #52606d; min-width:300px; min-height:220px; }
            QLineEdit { background:#f8fafb; border:1px solid #c7d0d8; padding:5px; }
            QTableWidget { background:#ffffff; gridline-color:#cbd5dd; }
            QTextEdit { background:#111820; color:#d9f5e5; font-family:Consolas,'Microsoft YaHei'; }
            QLabel#footer { background:#263746; color:#ffffff; padding:6px; }
            """
        )

        root = QVBoxLayout(self)
        title = QLabel("RAICOM 机器视觉系统创新赛")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:22px;font-weight:800;background:#ffffff;padding:8px;border:1px solid #b9c3cc;")
        root.addWidget(title)
        upper = QHBoxLayout()
        upper.addWidget(self.build_video_area(), 7)
        upper.addWidget(self.build_control_area(), 2)
        upper.addWidget(self.build_monitor_area(), 4)
        root.addLayout(upper, 7)
        root.addWidget(self.build_result_table(), 3)
        self.footer_label = QLabel()
        self.footer_label.setObjectName("footer")
        root.addWidget(self.footer_label)
        self.append_log("系统初始化完成，等待启动相机")

    def build_video_area(self) -> QGroupBox:
        box = QGroupBox("图像区")
        layout = QGridLayout(box)
        self.rgb_label = self._video_label("RGB 彩色图")
        self.depth_label = self._video_label("RGB 深度图")
        self.yolo_label = self._video_label("YOLO 推理图")
        layout.addWidget(QLabel("RGB 彩色图"), 0, 0)
        layout.addWidget(QLabel("RGB 深度图"), 0, 1)
        layout.addWidget(self.rgb_label, 1, 0)
        layout.addWidget(self.depth_label, 1, 1)
        layout.addWidget(QLabel("YOLO 推理图"), 2, 0, 1, 2)
        layout.addWidget(self.yolo_label, 3, 0, 1, 2)
        layout.setRowStretch(1, 1)
        layout.setRowStretch(3, 1)
        return box

    def _video_label(self, text: str) -> QLabel:
        label = QLabel(text + "待启动")
        label.setObjectName("videoLabel")
        label.setAlignment(Qt.AlignCenter)
        return label

    def build_control_area(self) -> QGroupBox:
        box = QGroupBox("功能区")
        layout = QVBoxLayout(box)
        self.task_labels = []
        controls = [
            ("启动相机", self.on_start_camera),
            ("单次运行", self.on_single_run),
            ("自动运行", self.on_auto_run),
        ]
        for index, (text, callback) in enumerate(controls, 1):
            button = QPushButton(text)
            button.setMinimumHeight(40)
            button.clicked.connect(callback)
            layout.addWidget(button)
            state = QLabel(f"{index} {text}：待执行")
            self.task_labels.append(state)

        connect = QPushButton("连接机器人")
        connect.clicked.connect(self.on_connect_robot)
        save = QPushButton("保存采集图像")
        save.clicked.connect(self.on_save_image)
        calibrate = QPushButton("标定桌面")
        calibrate.clicked.connect(self.on_calibrate_table)
        release = QPushButton("释放吸盘")
        release.clicked.connect(self.on_release_suction)
        auxiliary = QGridLayout()
        for index, button in enumerate((connect, save, calibrate, release)):
            auxiliary.addWidget(button, index // 2, index % 2)
        layout.addLayout(auxiliary)
        stop = QPushButton("急停 STOP")
        stop.setMinimumHeight(48)
        stop.setStyleSheet("background:#c62828;color:#ffffff;font-size:20px;border:2px solid #7f1d1d;")
        stop.clicked.connect(self.on_emergency_stop)
        layout.addWidget(stop)

        self.status_indicators = {}
        status_form = QFormLayout()
        for name in ("相机状态", "TCP状态", "机器人状态", "吸盘状态", "运行模式"):
            label = QLabel("● 未就绪")
            self.status_indicators[name] = label
            status_form.addRow(name, label)
        layout.addLayout(status_form)
        layout.addStretch(1)
        return box

    def build_monitor_area(self) -> QGroupBox:
        box = QGroupBox("监控区")
        layout = QVBoxLayout(box)
        self.task_tcp_status_label = QLabel("模块 B 输入：正在初始化")
        self.task_tcp_status_label.setStyleSheet("color:#92400e;font-weight:700;")
        layout.addWidget(self.task_tcp_status_label)
        form = QFormLayout()
        configured = self.config.get("task_outputs", {}).get("monitor_fields", {})
        fields = list(configured.items()) if isinstance(configured, dict) else []
        if not fields:
            fields = [
                ("gear", "齿轮识别"),
                ("broken", "缺齿识别"),
                ("ocr", "字符识别"),
                ("qr", "二维码识别"),
                ("compare", "信息对比"),
                ("defect", "缺陷检测"),
                ("measurement", "尺寸测量"),
            ]
        for key, title in fields:
            editor = QLineEdit("-")
            editor.setReadOnly(True)
            self.monitor_fields[key] = editor
            form.addRow(title, editor)
        layout.addLayout(form)

        self.recognition_label = QLabel("类别：-\n置信度：-\n角度：-\n高度：-")
        self.base_coord_label = QLabel("X：- mm\nY：- mm\nZ：- mm")
        self.joint_label = QLabel("J1--J6：-")
        self.tcp_pose_label = QLabel("TCP：-")
        layout.addWidget(self.recognition_label)
        layout.addWidget(self.base_coord_label)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(150)
        layout.addWidget(self.log_box)
        return box

    def build_result_table(self) -> QGroupBox:
        box = QGroupBox("3D 识别结果")
        layout = QVBoxLayout(box)
        self.result_table = QTableWidget(0, len(self.TABLE_HEADERS))
        self.result_table.setHorizontalHeaderLabels(self.TABLE_HEADERS)
        self.result_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.result_table)
        return box

    def on_single_run(self) -> None:
        if self.worker_thread is not None and self.worker_thread.isRunning():
            self.append_log("当前已有任务在运行，请稍后再试")
            return
        self.workflow.robot.reset_cancel()
        self.running_mode = "单次"
        self.worker_thread = QThread(self)
        self.worker = WorkflowWorker(self.workflow.execute_grasp_lift)
        self.workflow.log = self.worker.log.emit
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self.on_single_run_finished)
        self.worker.failed.connect(self.on_single_run_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self.clear_worker)
        self.worker_thread.start()

    def on_single_run_finished(self, _result) -> None:
        self.suction_on = True
        self.running_mode = "物料已抬起"
        self.update_task(2, "抓取抬起完成")
        self.refresh_target_info()
        self.refresh_all_status()
        self.append_log("单次运行完成：物料已抓取并抬起，等待后续操作")

    def on_single_run_failed(self, message: str) -> None:
        self.running_mode = "模拟" if self.config.get("mode") == "simulation" else "手动"
        self.append_log("单次运行失败：" + message)
        QMessageBox.warning(self, "单次运行失败", message)

    def on_release_suction(self) -> None:
        try:
            if self.workflow.robot.connected:
                self.workflow.robot.suction(False)
            self.suction_on = False
            self.running_mode = "模拟" if self.config.get("mode") == "simulation" else "手动"
            self.refresh_all_status()
            self.append_log("吸盘已释放")
        except Exception as exc:
            self.append_log("释放吸盘失败：" + str(exc))
            QMessageBox.warning(self, "释放失败", str(exc))

    def on_auto_run_finished(self, count: int) -> None:
        self.suction_on = False
        self.running_mode = "模拟" if self.config.get("mode") == "simulation" else "手动"
        self.update_task(3, f"完成 {count} 个")
        self.refresh_all_status()
        self.append_log(f"自动运行完成：{count} 个目标")
        QMessageBox.information(self, "任务完成", f"自动运行完成：{count} 个目标")

    def update_frame(self) -> None:
        try:
            now = perf_counter()
            if self.last_frame_time is not None:
                delta = max(1e-6, now - self.last_frame_time)
                instant_fps = 1.0 / delta
                self.current_fps = instant_fps if self.current_fps == 0 else self.current_fps * 0.85 + instant_fps * 0.15
            self.last_frame_time = now
            frame = self.workflow.read_frame()
            self.rgb_label.setPixmap(self.to_pixmap(frame.color, self.rgb_label, is_bgr=True))

            depth = frame.depth_mm
            valid = depth[np.isfinite(depth) & (depth > 0)]
            max_depth = float(np.percentile(valid, 95)) if valid.size else 1000.0
            depth_vis = np.clip(depth / max_depth * 255.0, 0, 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            self.depth_label.setPixmap(self.to_pixmap(depth_color, self.depth_label, is_bgr=True))
            inference_image = self.workflow.detection_frame
            if inference_image is None:
                inference_image = frame.color
            self.yolo_label.setPixmap(self._detection_pixmap(inference_image, self.yolo_label))
            self.refresh_target_info()
            self._refresh_task_outputs()
            self.update_robot_pose_labels(force=False)
            self.refresh_all_status()
        except Exception as exc:
            self.append_log("画面刷新失败：" + str(exc))
            self.timer.stop()

    def _detection_pixmap(self, bgr: np.ndarray, label: QLabel):
        pixmap = self.to_pixmap(bgr, label, is_bgr=True)
        source_h, source_w = bgr.shape[:2]
        scale = min(pixmap.width() / max(1, source_w), pixmap.height() / max(1, source_h))
        painter = QPainter(pixmap)
        painter.setPen(QPen(QColor("#22c55e"), 2))
        painter.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        for detection in self.workflow.detections:
            x1, y1, x2, y2 = detection.bbox
            painter.drawRect(int(x1 * scale), int(y1 * scale), int((x2 - x1) * scale), int((y2 - y1) * scale))
            painter.drawText(int(x1 * scale), max(16, int(y1 * scale) - 3), f"{detection.label} {detection.confidence:.2f}")
        painter.end()
        return pixmap

    def refresh_target_info(self) -> None:
        target = self.workflow.current_target
        if target is None:
            return
        detection = target.detection
        self.recognition_label.setText(
            f"类别：{detection.label}\n置信度：{detection.confidence:.2f}\n"
            f"角度：{target.angle_deg:.1f}° ({'有效' if target.angle_valid else '无效'})\n"
            f"角度置信度：{target.angle_confidence:.2f}\n高度：{target.height_mm:.1f} mm"
        )
        pose = target.base_pose
        self.base_coord_label.setText(f"X：{pose[0]:.2f} mm\nY：{pose[1]:.2f} mm\nZ：{pose[2]:.2f} mm")
        signature = (detection.label, round(pose[0], 2), round(pose[1], 2), round(target.angle_deg, 1), round(target.height_mm, 1))
        if signature == self._last_table_signature:
            return
        self._last_table_signature = signature
        row = self.result_table.rowCount()
        self.result_table.insertRow(row)
        values = [str(row + 1), detection.label, f"{detection.confidence:.2f}", f"{pose[0]:.2f}", f"{pose[1]:.2f}", f"{target.angle_deg:.1f}", f"{target.height_mm:.1f}"]
        for column, value in enumerate(values):
            self.result_table.setItem(row, column, QTableWidgetItem(value))

    def _refresh_task_outputs(self) -> None:
        config = self.config.get("task_outputs", {})
        tcp_config = config.get("tcp", {})
        if isinstance(tcp_config, dict) and tcp_config.get("enabled") and not config.get("file_fallback_enabled", False):
            return
        directory = Path(str(config.get("directory", "runs/vision_outputs")))
        if not directory.is_absolute():
            directory = ROOT / directory
        files = config.get("files", {})
        for key, filename in files.items():
            path = directory / str(filename)
            try:
                stamp = path.stat().st_mtime_ns
            except OSError:
                continue
            if self._output_mtimes.get(path) == stamp:
                continue
            self._output_mtimes[path] = stamp
            raw = path.read_bytes()
            try:
                text = raw.decode("utf-8-sig").strip()
            except UnicodeDecodeError:
                text = raw.decode("gb18030", errors="replace").strip()
            self._apply_task_output(TaskOutputMessage(str(key), text))

    def _drain_task_tcp(self) -> None:
        for event in self.task_output_receiver.drain():
            event_type = event[0]
            if event_type == "data":
                self._apply_task_output(event[1])
                continue
            state = str(event[1])
            detail = str(event[2]) if len(event) > 2 else ""
            status = (state, detail)
            if status == self._task_tcp_last_status:
                continue
            self._task_tcp_last_status = status
            self._set_task_tcp_status(state, detail)

    def _set_task_tcp_status(self, state: str, detail: str = "") -> None:
        receiver = self.task_output_receiver
        endpoint = f"{receiver.host}:{receiver.port or '待配置'}"
        if state == "connected":
            self.tcp_ready = True
            text, color = f"模块 B TCP：已连接 {endpoint}", "#15803d"
            self.append_log(text)
        elif state == "waiting_port":
            self.tcp_ready = False
            text, color = f"模块 B TCP：等待在配置中填写端口（{receiver.host}）", "#b45309"
            self.append_log(text)
        elif state == "disabled":
            self.tcp_ready = False
            text, color = "模块 B 输入：TCP 未启用，保留文本文件兼容模式", "#64748b"
        else:
            self.tcp_ready = False
            text, color = f"模块 B TCP：重连中 {endpoint}", "#b91c1c"
            if detail:
                self.append_log(f"{text}；{detail}")
        self.task_tcp_status_label.setText(text)
        self.task_tcp_status_label.setStyleSheet(f"color:{color};font-weight:700;")
        self.refresh_all_status()

    def _apply_task_output(self, message: TaskOutputMessage) -> None:
        key = message.key
        compact = re.sub(r"\s+", " ", message.text).strip()
        if key == "gear":
            self.monitor_fields["gear"].setText(compact or "-")
            self.monitor_fields["broken"].setText("NG" if "NG" in compact.upper() else "-")
        elif key == "recognition":
            parts = [part.strip() for part in re.split(r"[,;|]", compact) if part.strip()]
            match, ocr, qr = self.parse_recognition_parts(parts)
            self.monitor_fields["ocr"].setText(ocr)
            self.monitor_fields["qr"].setText(qr)
            self.monitor_fields["compare"].setText(match)
        elif key in self.monitor_fields:
            self.monitor_fields[key].setText(compact or "-")

    @staticmethod
    def parse_recognition_parts(parts: list[str]) -> tuple[str, str, str]:
        if parts and parts[0].upper() in {"OK", "NG"}:
            return (
                parts[0].upper(),
                parts[1] if len(parts) > 1 else "-",
                parts[2] if len(parts) > 2 else "-",
            )
        match = next((part.upper() for part in parts if part.upper() in {"OK", "NG"}), "-")
        payload = [part for part in parts if part.upper() not in {"OK", "NG"}]
        return match, payload[0] if payload else "-", payload[1] if len(payload) > 1 else "-"

    def refresh_footer(self) -> None:
        if not hasattr(self, "footer_label"):
            return
        mode = "模拟" if self.config.get("mode") == "simulation" else "真机"
        camera = "已启动" if self.camera_ready else "未启动"
        robot = "已连接" if self.workflow.robot.connected else "未连接"
        self.footer_label.setText(f"模式：{mode}    相机：{camera}    机器人：{robot}    {datetime.now():%Y-%m-%d %H:%M:%S}")

    def closeEvent(self, event) -> None:
        self.task_output_timer.stop()
        self.task_output_receiver.stop()
        super().closeEvent(event)


def main() -> int:
    if QApplication is None or cv2 is None:
        print("启动失败：缺少 PyQt5、OpenCV 或 NumPy。")
        if IMPORT_ERROR is not None:
            print(f"具体错误：{IMPORT_ERROR}")
        return 1
    app = QApplication(sys.argv)
    window = TodayMainWindow()
    window.showMaximized()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
