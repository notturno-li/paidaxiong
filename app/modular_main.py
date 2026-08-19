from __future__ import annotations

import csv
import json
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from time import strftime

from app.config import deep_update, update_yaml_config
from app.core.detector import FruitDetector
from app.core.model_profiles import model_config_update
from app.core.preflight import run_preflight
from app.core.vision_studio_client import DobotVisionStudioClient
from app.main import IMPORT_ERROR, MainWindow, WorkflowWorker, cv2
from app.modules import ModuleExecutor, build_standard_registry

try:
    from app.action_sequence_dialog import ActionSequenceDialog
    from app.field_config_dialog import FieldConfigDialog
    from app.model_manager_dialog import ModelManagerDialog
    from app.task_wizard import TaskWizardDialog
    from PyQt5.QtCore import QSize, QThread, QTimer, Qt
    from PyQt5.QtGui import QColor, QIcon, QPainter, QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    QApplication = None
    ActionSequenceDialog = None
    FieldConfigDialog = None
    ModelManagerDialog = None
    TaskWizardDialog = None


ROOT = Path(__file__).resolve().parents[1]
NATIONAL_CONFIG = ROOT / "configs" / "national.yaml"
FIELD_CONFIG = ROOT / "configs" / "field.yaml"


class NationalMainWindow(MainWindow):
    QUEUE_STATE_STYLES = {
        "pending": ("#9ca3af", "#ffffff", "待执行"),
        "running": ("#d97706", "#fffbeb", "执行中"),
        "success": ("#16a34a", "#f0fdf4", "执行成功"),
        "failed": ("#dc2626", "#fef2f2", "执行失败"),
    }

    def __init__(self):
        self.registry = build_standard_registry()
        self.executor: ModuleExecutor | None = None
        self.run_buttons: list[QPushButton] = []
        self.reload_recipe_button: QPushButton | None = None
        self._active_module_ids: list[str] = []
        self._status_icons: dict[str, QIcon] = {}
        super().__init__(FIELD_CONFIG if FIELD_CONFIG.exists() else NATIONAL_CONFIG)
        self.reload_feedback_timer = QTimer(self)
        self.reload_feedback_timer.setSingleShot(True)
        self.reload_feedback_timer.timeout.connect(self.restore_reload_button_feedback)
        self.executor = ModuleExecutor(self.workflow, self.registry, logger=self.append_log)
        self.populate_module_library()
        self.populate_recipes()
        profile = self.config.get("competition", {}).get("profile", "national_modular")
        self.append_log(f"国赛模块化控制台已加载：profile={profile}")

    def build_control_area(self) -> QGroupBox:
        box = QGroupBox("B区 模块编排台")
        box.setMinimumWidth(480)
        box.setMaximumWidth(560)
        root = QVBoxLayout()
        root.setSpacing(4)

        header = QHBoxLayout()
        header.addWidget(QLabel("现场配方"))
        self.recipe_combo = QComboBox()
        self.recipe_combo.currentIndexChanged.connect(self.load_selected_recipe)
        header.addWidget(self.recipe_combo, stretch=1)
        wizard_button = QPushButton("任务向导")
        wizard_button.setToolTip("按国赛任务书填写表格，自动生成 Profile、循环参数和执行队列")
        wizard_button.clicked.connect(self.open_task_wizard)
        header.addWidget(wizard_button)
        field_config_button = QPushButton("通信/高级")
        field_config_button.setToolTip("配置 DobotVisionStudio 通信及高级参数")
        field_config_button.clicked.connect(self.open_field_config)
        header.addWidget(field_config_button)
        root.addLayout(header)

        lists = QGridLayout()
        lists.addWidget(QLabel("功能模块库"), 0, 0)
        lists.addWidget(QLabel("执行队列"), 0, 1)
        self.module_list = QListWidget()
        self.module_list.setMinimumHeight(55)
        self.module_list.itemDoubleClicked.connect(lambda _item: self.add_selected_module())
        self.queue_list = QListWidget()
        self.queue_list.setMinimumHeight(55)
        self.queue_list.setIconSize(QSize(12, 12))
        self.queue_list.itemDoubleClicked.connect(lambda _item: self.run_current_module())
        lists.addWidget(self.module_list, 1, 0)
        lists.addWidget(self.queue_list, 1, 1)
        lists.setRowStretch(0, 0)
        lists.setRowStretch(1, 1)
        root.addLayout(lists)

        edit_grid = QGridLayout()
        edit_actions = [
            ("加入队列", self.add_selected_module),
            ("移出队列", self.remove_selected_module),
            ("上移", lambda: self.move_queue_item(-1)),
            ("下移", lambda: self.move_queue_item(1)),
            ("清空", self.clear_queue),
            ("重载配方", self.reload_selected_recipe),
            ("保存配方", self.save_current_recipe),
            ("完整结果", self.show_full_result),
            ("动作编辑器", self.open_action_sequence_editor),
            ("模型管理", self.open_model_manager),
            ("赛前自检", self.run_preflight_check),
            ("导出结果", self.export_results),
        ]
        for index, (text, slot) in enumerate(edit_actions):
            button = QPushButton(text)
            button.setFixedHeight(32)
            button.clicked.connect(slot)
            if text == "重载配方":
                self.reload_recipe_button = button
                button.setToolTip("把当前下拉框中的配方恢复到执行队列")
            edit_grid.addWidget(button, index // 4, index % 4)
        root.addLayout(edit_grid)

        run_grid = QGridLayout()
        self.run_one_button = QPushButton("运行当前模块")
        self.run_one_button.clicked.connect(self.run_current_module)
        self.run_queue_button = QPushButton("运行组合流程")
        self.run_queue_button.clicked.connect(self.run_queue)
        self.run_one_button.setFixedHeight(38)
        self.run_queue_button.setFixedHeight(38)
        run_grid.addWidget(self.run_one_button, 0, 0)
        run_grid.addWidget(self.run_queue_button, 0, 1)
        root.addLayout(run_grid)
        self.run_buttons.extend((self.run_one_button, self.run_queue_button))

        self.stop_button = QPushButton("急停 STOP")
        self.stop_button.setFixedHeight(56)
        self.stop_button.setStyleSheet(
            "QPushButton { background:#dc2626; color:white; border:3px solid #7f1d1d; "
            "border-radius:8px; font-size:24px; font-weight:900; }"
            "QPushButton:hover { background:#b91c1c; }"
        )
        self.stop_button.clicked.connect(self.on_emergency_stop)
        root.addWidget(self.stop_button)
        box.setLayout(root)
        return box

    def build_status_area(self) -> QGroupBox:
        area = QGroupBox("C区 状态与日志")
        root = QHBoxLayout()
        progress = self.build_progress_box()
        progress.setMinimumWidth(220)
        progress.setMaximumWidth(280)
        right = QWidget()
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        recognition = self.build_recognition_box()
        coordinates = self.build_base_box()
        system = self.build_system_status_box()
        for box in (recognition, coordinates, system):
            box.setMinimumHeight(132)
        grid.addWidget(recognition, 0, 0)
        grid.addWidget(coordinates, 0, 1)
        grid.addWidget(system, 0, 2)
        grid.addWidget(self.build_robot_data_box(), 1, 0, 1, 2)
        grid.addWidget(self.build_log_box(), 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 3)
        grid.setRowStretch(0, 2)
        grid.setRowStretch(1, 2)
        right.setLayout(grid)
        root.addWidget(progress)
        root.addWidget(right, stretch=1)
        area.setLayout(root)
        return area

    def build_progress_box(self) -> QGroupBox:
        box = QGroupBox("模块执行状态")
        layout = QVBoxLayout()
        self.module_status_list = QListWidget()
        self.module_status_list.setMinimumWidth(230)
        self.module_status_list.setMinimumHeight(150)
        self.module_status_list.setIconSize(QSize(12, 12))
        layout.addWidget(self.module_status_list)
        box.setLayout(layout)
        return box

    def build_system_status_box(self) -> QGroupBox:
        box = QGroupBox("系统状态")
        layout = QGridLayout()
        layout.setContentsMargins(8, 14, 8, 4)
        layout.setVerticalSpacing(0)
        items = ["相机状态", "TCP状态", "机器人状态", "DVS状态", "吸盘状态", "运行模式"]
        for row, name in enumerate(items):
            title = QLabel(name)
            title.setStyleSheet("font-size:12px;")
            layout.addWidget(title, row, 0)
            indicator = QLabel("● 未就绪")
            indicator.setStyleSheet("color:#dc2626;font-weight:800;font-size:12px;")
            self.status_indicators[name] = indicator
            layout.addWidget(indicator, row, 1)
        box.setLayout(layout)
        return box

    def build_recognition_box(self) -> QGroupBox:
        box = QGroupBox("识别结果")
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 14, 8, 4)
        self.recognition_label = QLabel("类别：-\n置信度：-\n像素坐标：-\n高度：-")
        self.recognition_label.setObjectName("metricCard")
        self.recognition_label.setStyleSheet("font-size:12px;padding:4px;")
        self.recognition_label.setWordWrap(True)
        layout.addWidget(self.recognition_label)
        box.setLayout(layout)
        return box

    def refresh_target_info(self) -> None:
        result = self.workflow.vision_task_result
        raw_data = self.workflow.vision_studio_data
        if result is None and not isinstance(raw_data, dict):
            super().refresh_target_info()
            return

        fields = result.fields if result is not None else raw_data
        decision = result.decision if result is not None else None
        route = result.route if result is not None else None
        identifiers = result.identifiers if result is not None else {}
        def short(value, limit=14):
            text = str(value)
            return text if len(text) <= limit else text[: limit - 3] + "..."

        field_text = ", ".join(
            f"{short(key, 8)}={short(value, 8)}" for key, value in list(fields.items())[:2]
        )
        identifier_text = ", ".join(
            f"{short(key, 8)}={short(value, 10)}" for key, value in list(identifiers.items())[:1]
        ) or "-"
        lines = [
            f"DVS：{decision or '-'} / {route or '-'}",
            f"识别：{identifier_text}",
            f"字段：{field_text or '-'}",
        ]
        if result is not None and result.violations:
            lines.append("越界：" + short(result.violations[0], 22))
        self.recognition_label.setText("\n".join(lines))

        action = self.workflow.shared_data.get("last_pick_place") or self.workflow.shared_data.get(
            "planned_pick_place"
        )
        if isinstance(action, dict):
            pick = action.get("pick_pose", [])
            place = action.get("place_pose", [])
            if len(pick) == 6 and len(place) == 6:
                self.base_coord_label.setText(
                    f"抓取XYZ：{pick[0]:.1f}, {pick[1]:.1f}, {pick[2]:.1f}\n"
                    f"放置XYZ：{place[0]:.1f}, {place[1]:.1f}, {place[2]:.1f}\n"
                    f"抓取姿态：{pick[3]:.1f}, {pick[4]:.1f}, {pick[5]:.1f}\n"
                    f"放置姿态：{place[3]:.1f}, {place[4]:.1f}, {place[5]:.1f}"
                )
                return
        if all(name in fields for name in ("x", "y", "z")):
            self.base_coord_label.setText(
                f"DVS X：{fields['x']} mm\nDVS Y：{fields['y']} mm\n"
                f"DVS Z：{fields['z']} mm\n目标路线：{route or '-'}"
            )
        else:
            self.base_coord_label.setText("DVS动作坐标：待配置/待执行")

    def build_base_box(self) -> QGroupBox:
        box = QGroupBox("基坐标系坐标")
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 14, 8, 4)
        self.base_coord_label = QLabel("X：- mm\nY：- mm\nZ：- mm\nRx/Ry/Rz：-")
        self.base_coord_label.setObjectName("metricCard")
        self.base_coord_label.setStyleSheet("font-size:12px;padding:4px;")
        layout.addWidget(self.base_coord_label)
        box.setLayout(layout)
        return box

    def refresh_all_status(self) -> None:
        super().refresh_all_status()
        dvs = self.workflow.vision_studio
        if dvs.connected:
            text = f"已连接 ({dvs.mode})"
        elif dvs.mode == "disabled":
            text = "未配置"
        else:
            text = f"待连接 ({dvs.mode})"
        self.set_indicator("DVS状态", dvs.connected, text)

    def set_indicator(self, name: str, ok: bool, text: str) -> None:
        label = self.status_indicators.get(name)
        if not label:
            return
        color = "#16a34a" if ok else "#dc2626"
        label.setText("● " + text)
        label.setStyleSheet(f"color:{color};font-weight:900;font-size:12px;")

    def populate_module_library(self) -> None:
        self.module_list.clear()
        for module in self.registry.all():
            item = QListWidgetItem(f"[{module.group}] {module.name}")
            item.setData(Qt.UserRole, module.module_id)
            dependencies = ", ".join(module.dependencies) if module.dependencies else "无"
            item.setToolTip(f"{module.module_id}\n依赖：{dependencies}")
            self.module_list.addItem(item)
        if self.module_list.count():
            self.module_list.setCurrentRow(0)

    def open_field_config(self) -> None:
        if self.worker_thread is not None and self.worker_thread.isRunning():
            QMessageBox.warning(self, "任务运行中", "请先停止或等待当前模块流程完成")
            return
        try:
            discovery = self.workflow.vision_studio.discover()
            serial_ports = discovery.get("serial_ports", [])
        except Exception as exc:
            serial_ports = []
            self.append_log(f"串口扫描失败，仍可手动填写：{exc}")
        assert FieldConfigDialog is not None
        dialog = FieldConfigDialog(self.config, serial_ports=serial_ports, parent=self)
        if dialog.exec_() != dialog.Accepted:
            return
        try:
            path = update_yaml_config(
                FIELD_CONFIG,
                dialog.updates,
                extends=NATIONAL_CONFIG.name,
            )
            self.workflow.vision_studio.close()
            deep_update(self.config, dialog.updates)
            self.workflow.vision_studio = DobotVisionStudioClient(
                self.config,
                simulation=self.workflow.simulation,
            )
            self.workflow.vision_studio_data = None
            self.workflow.vision_task_result = None
            self.workflow.shared_data.pop("vision_studio", None)
            self.workflow.shared_data.pop("vision_task", None)
            self.workflow.shared_data.pop("planned_pick_place", None)
            self.workflow.shared_data.pop("planned_action_sequence", None)
            self.workflow.shared_data.pop("placement_counts", None)
            self.refresh_target_info()
            self.refresh_all_status()
            active = self.config["vision_tasks"]["active_profile"]
            self.append_log(f"现场配置已保存并应用：{path}，任务 Profile={active}")
        except Exception as exc:
            QMessageBox.warning(self, "保存现场配置失败", str(exc))

    def open_model_manager(self) -> None:
        if self.worker_thread is not None and self.worker_thread.isRunning():
            QMessageBox.warning(self, "任务运行中", "请先停止或等待当前模块流程完成")
            return
        assert ModelManagerDialog is not None
        dialog = ModelManagerDialog(
            ROOT,
            current_weights=self.config.get("model", {}).get("weights"),
            parent=self,
        )
        if dialog.exec_() != dialog.Accepted or dialog.selected_profile is None:
            return
        profile = dialog.selected_profile
        updates = model_config_update(profile)
        candidate = deepcopy(self.config)
        deep_update(candidate, updates)
        try:
            # Validate the replacement completely before touching the running workflow.
            detector = FruitDetector(candidate, simulation=self.workflow.simulation)
            path = update_yaml_config(
                FIELD_CONFIG,
                updates,
                extends=NATIONAL_CONFIG.name,
            )
            deep_update(self.config, updates)
            detector.config = self.config
            self.workflow.detector = detector
            self.workflow.detections = []
            self.workflow.current_target = None
            if self.executor is not None:
                for module_id in ("vision.detect", "target.calculate"):
                    self.executor.completed.discard(module_id)
            self.refresh_target_info()
            self.append_log(
                f"模型已热加载：{profile.name}；类别={list(profile.class_names)}；"
                f"conf={profile.conf_threshold:g}；配置={path}"
            )
            QMessageBox.information(
                self,
                "模型已应用",
                f"{profile.name}\n类别：{', '.join(profile.class_names)}\n"
                f"置信度：{profile.conf_threshold:g}",
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "模型加载失败",
                "旧模型和现场配置均已保留。\n\n" + str(exc),
            )

    def open_task_wizard(self) -> None:
        if self.worker_thread is not None and self.worker_thread.isRunning():
            QMessageBox.warning(self, "任务运行中", "请先停止或等待当前模块流程完成")
            return
        assert TaskWizardDialog is not None
        dialog = TaskWizardDialog(self.config, parent=self)
        if dialog.exec_() != dialog.Accepted:
            return
        result = dialog.result
        try:
            path = update_yaml_config(FIELD_CONFIG, result.updates, extends=NATIONAL_CONFIG.name)
            self.workflow.vision_studio.close()
            deep_update(self.config, result.updates)
            self.workflow.vision_studio = DobotVisionStudioClient(
                self.config,
                simulation=self.workflow.simulation,
            )
            self.workflow.vision_studio_data = None
            self.workflow.vision_task_result = None
            for key in (
                "vision_studio",
                "vision_task",
                "planned_pick_place",
                "planned_action_sequence",
                "placement_counts",
                "dvs_cycle",
            ):
                self.workflow.shared_data.pop(key, None)
            self.populate_recipes()
            index = self.recipe_combo.findData(result.recipe_id)
            if index >= 0:
                self.recipe_combo.setCurrentIndex(index)
            self.refresh_target_info()
            self.refresh_all_status()
            self.append_log(
                f"零代码任务已生成：{result.recipe_name}，Profile={result.profile_name}，"
                f"队列={list(result.module_ids)}，保存到 {path}"
            )
            QMessageBox.information(
                self,
                "现场配方已就绪",
                f"已生成并装载：{result.recipe_name}\n"
                "请先保持机器人动作关闭完成通信和字段验证，再进行低速单件动作。",
            )
        except Exception as exc:
            QMessageBox.warning(self, "生成现场配方失败", str(exc))

    def open_action_sequence_editor(self) -> None:
        if self.worker_thread is not None and self.worker_thread.isRunning():
            QMessageBox.warning(self, "任务运行中", "请先停止或等待当前模块流程完成")
            return
        assert ActionSequenceDialog is not None
        dialog = ActionSequenceDialog(self.config, parent=self)
        if dialog.exec_() != dialog.Accepted:
            return
        try:
            path = update_yaml_config(
                FIELD_CONFIG,
                dialog.updates,
                extends=NATIONAL_CONFIG.name,
            )
            deep_update(self.config, dialog.updates)
            self.workflow.shared_data.pop("planned_action_sequence", None)
            active = self.config["action_sequences"]["active_profile"]
            self.append_log(f"固定动作序列已保存并应用：{active} -> {path}")
            QMessageBox.information(
                self,
                "动作序列已保存",
                "请先运行“预览并校验固定动作序列”，确认步骤和位姿后再执行。",
            )
        except Exception as exc:
            QMessageBox.warning(self, "保存动作序列失败", str(exc))

    def run_preflight_check(self) -> None:
        if self.worker_thread is not None and self.worker_thread.isRunning():
            QMessageBox.warning(self, "任务运行中", "请先停止或等待当前模块流程完成")
            return
        answer = QMessageBox.question(
            self,
            "赛前自检范围",
            "是否同时探测机械臂 Dashboard 和 DVS 硬件连接？\n"
            "探测只建立连接，不使能机器人、不发送业务或运动命令。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        hardware = answer == QMessageBox.Yes
        report = run_preflight(FIELD_CONFIG, hardware=hardware, root=ROOT)
        labels = {"PASS": "[通过]", "WARN": "[待确认]", "FAIL": "[阻止]"}
        lines = [
            f"{labels[check.status]} {check.name}: {check.detail}"
            for check in report.checks
        ]
        lines.extend(
            [
                "",
                f"汇总：通过 {report.passed}，待确认 {report.warnings}，阻止 {report.failures}",
                "结论：" + ("未发现阻止项" if report.ready else "存在阻止项，禁止启动机械臂动作"),
            ]
        )
        dialog = QDialog(self)
        dialog.setWindowTitle("赛前自检结果")
        dialog.resize(860, 620)
        layout = QVBoxLayout(dialog)
        viewer = QPlainTextEdit()
        viewer.setReadOnly(True)
        viewer.setPlainText("\n".join(lines))
        layout.addWidget(viewer)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Close).setText("关闭")
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec_()
        self.append_log(
            f"赛前自检完成：pass={report.passed} warn={report.warnings} fail={report.failures}"
        )

    def populate_recipes(self) -> None:
        self.recipe_combo.blockSignals(True)
        self.recipe_combo.clear()
        recipes = self.config.get("recipes", {})
        for recipe_id, recipe in recipes.items():
            name = recipe.get("name", recipe_id) if isinstance(recipe, dict) else recipe_id
            self.recipe_combo.addItem(str(name), recipe_id)
        default_id = self.config.get("modules", {}).get("default_recipe")
        default_index = self.recipe_combo.findData(default_id)
        self.recipe_combo.setCurrentIndex(default_index if default_index >= 0 else 0)
        self.recipe_combo.blockSignals(False)
        self.load_selected_recipe()

    def load_selected_recipe(self) -> None:
        recipe_id = self.recipe_combo.currentData()
        if not recipe_id:
            return
        recipe = self.config.get("recipes", {}).get(recipe_id, {})
        module_ids = recipe.get("modules", []) if isinstance(recipe, dict) else recipe
        self.queue_list.clear()
        for module_id in module_ids:
            self.add_module_id(str(module_id))
        self.refresh_queue_status()

    def reload_selected_recipe(self) -> None:
        recipe_id = self.recipe_combo.currentData()
        if not recipe_id:
            QMessageBox.warning(self, "无可用配方", "请先选择一个现场配方")
            return
        self.load_selected_recipe()
        recipe = self.config.get("recipes", {}).get(recipe_id, {})
        recipe_name = recipe.get("name", recipe_id) if isinstance(recipe, dict) else recipe_id
        self.append_log(f"配方已重载：{recipe_name}，共 {self.queue_list.count()} 个模块")
        if self.reload_recipe_button is not None:
            self.reload_recipe_button.setText("重载完成")
            self.reload_recipe_button.setStyleSheet(
                "QPushButton { background:#dcfce7; color:#166534; font-weight:700; }"
            )
            self.reload_feedback_timer.start(1200)

    def restore_reload_button_feedback(self) -> None:
        if self.reload_recipe_button is not None:
            self.reload_recipe_button.setText("重载配方")
            self.reload_recipe_button.setStyleSheet("")

    def add_module_id(self, module_id: str) -> None:
        module = self.registry.get(module_id)
        item = QListWidgetItem()
        item.setData(Qt.UserRole, module.module_id)
        self.set_queue_item_state(item, "pending")
        self.queue_list.addItem(item)

    def set_queue_item_state(self, item: QListWidgetItem, state: str) -> None:
        color, background, description = self.QUEUE_STATE_STYLES[state]
        module = self.registry.get(str(item.data(Qt.UserRole)))
        item.setData(Qt.UserRole + 1, state)
        item.setText(module.name)
        item.setIcon(self.status_icon(color))
        item.setForeground(QColor("#1f2937"))
        item.setBackground(QColor(background))
        item.setToolTip(description)

    def status_icon(self, color: str) -> QIcon:
        if color not in self._status_icons:
            pixmap = QPixmap(12, 12)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(color))
            painter.drawEllipse(1, 1, 10, 10)
            painter.end()
            self._status_icons[color] = QIcon(pixmap)
        return self._status_icons[color]

    def set_module_execution_state(self, module_id: str, state: str) -> None:
        display_state = "success" if state == "skipped" else state
        for index in range(self.queue_list.count()):
            item = self.queue_list.item(index)
            if str(item.data(Qt.UserRole)) == module_id:
                self.set_queue_item_state(item, display_state)

        labels = {
            "pending": ("待执行", "#9ca3af"),
            "running": ("执行中", "#d97706"),
            "success": ("成功", "#16a34a"),
            "skipped": ("已就绪", "#16a34a"),
            "failed": ("失败", "#dc2626"),
        }
        prefix, color = labels[state]
        for index in range(self.module_status_list.count()):
            status_item = self.module_status_list.item(index)
            if str(status_item.data(Qt.UserRole)) == module_id:
                module = self.registry.get(module_id)
                status_item.setText(f"{prefix}  {module.name}")
                status_item.setIcon(self.status_icon(color))
                status_item.setForeground(QColor("#1f2937"))


    def add_selected_module(self) -> None:
        item = self.module_list.currentItem()
        if item is not None:
            self.add_module_id(str(item.data(Qt.UserRole)))
            self.queue_list.setCurrentRow(self.queue_list.count() - 1)
            self.refresh_queue_status()

    def remove_selected_module(self) -> None:
        row = self.queue_list.currentRow()
        if row >= 0:
            self.queue_list.takeItem(row)
            self.refresh_queue_status()

    def move_queue_item(self, direction: int) -> None:
        row = self.queue_list.currentRow()
        target = row + direction
        if row < 0 or target < 0 or target >= self.queue_list.count():
            return
        item = self.queue_list.takeItem(row)
        self.queue_list.insertItem(target, item)
        self.queue_list.setCurrentRow(target)

    def clear_queue(self) -> None:
        self.queue_list.clear()
        self.refresh_queue_status()

    def save_current_recipe(self) -> None:
        module_ids = self.queue_module_ids()
        if not module_ids:
            QMessageBox.warning(self, "队列为空", "请先把需要的模块加入执行队列")
            return
        default_name = "现场配方 " + strftime("%H%M%S")
        name, accepted = QInputDialog.getText(self, "保存现场配方", "配方显示名称", text=default_name)
        name = name.strip()
        if not accepted:
            return
        if not name:
            QMessageBox.warning(self, "名称为空", "配方名称不能为空")
            return
        recipe_id = "field_" + strftime("%Y%m%d_%H%M%S")
        suffix = 1
        while recipe_id in self.config.get("recipes", {}):
            suffix += 1
            recipe_id = "field_" + strftime("%Y%m%d_%H%M%S") + f"_{suffix}"
        recipe = {"name": name, "modules": module_ids}
        updates = {"recipes": {recipe_id: recipe}}
        try:
            path = update_yaml_config(FIELD_CONFIG, updates, extends=NATIONAL_CONFIG.name)
            deep_update(self.config, updates)
            self.populate_recipes()
            index = self.recipe_combo.findData(recipe_id)
            if index >= 0:
                self.recipe_combo.setCurrentIndex(index)
            self.append_log(f"现场配方已保存：{name} ({recipe_id}) → {path}")
        except Exception as exc:
            QMessageBox.warning(self, "保存配方失败", str(exc))

    def show_full_result(self) -> None:
        payload = self.result_payload()
        if all(value in (None, {}) for value in payload.values()):
            QMessageBox.information(self, "完整结果", "尚未获取 DVS 结果")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("DVS 完整结果")
        dialog.resize(760, 560)
        layout = QVBoxLayout(dialog)
        viewer = QPlainTextEdit()
        viewer.setReadOnly(True)
        viewer.setLineWrapMode(QPlainTextEdit.NoWrap)
        viewer.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        layout.addWidget(viewer)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Close).setText("关闭")
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec_()

    def result_payload(self) -> dict:
        result = self.workflow.vision_task_result
        return {
            "vision_studio_raw": self.workflow.vision_studio_data,
            "vision_task": asdict(result) if result is not None else None,
            "planned_pick_place": self.workflow.shared_data.get("planned_pick_place"),
            "last_pick_place": self.workflow.shared_data.get("last_pick_place"),
            "placement_counts": self.workflow.shared_data.get("placement_counts", {}),
            "dvs_cycle": self.workflow.shared_data.get("dvs_cycle"),
        }

    def export_results(self) -> None:
        payload = self.result_payload()
        if all(value in (None, {}) for value in payload.values()):
            QMessageBox.information(self, "导出结果", "尚未获取 DVS 结果")
            return
        output_dir = ROOT / "runs" / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = strftime("%Y%m%d_%H%M%S")
        json_path = output_dir / f"dvs_result_{stamp}.json"
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        exported = [json_path]
        cycle = payload.get("dvs_cycle")
        records = cycle.get("records", []) if isinstance(cycle, dict) else []
        if records:
            csv_path = output_dir / f"dvs_result_{stamp}.csv"
            with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "index",
                        "decision",
                        "route",
                        "identifiers",
                        "violations",
                        "fields",
                    ],
                )
                writer.writeheader()
                for record in records:
                    writer.writerow(
                        {
                            "index": record.get("index"),
                            "decision": record.get("decision"),
                            "route": record.get("route"),
                            "identifiers": json.dumps(
                                record.get("identifiers", {}), ensure_ascii=False, default=str
                            ),
                            "violations": "；".join(
                                str(item) for item in record.get("violations", [])
                            ),
                            "fields": json.dumps(
                                record.get("fields", {}), ensure_ascii=False, default=str
                            ),
                        }
                    )
            exported.append(csv_path)
        self.append_log("结果已导出：" + "，".join(str(path) for path in exported))
        QMessageBox.information(
            self,
            "导出完成",
            "已保存：\n" + "\n".join(str(path) for path in exported),
        )

    def queue_module_ids(self) -> list[str]:
        return [
            str(self.queue_list.item(index).data(Qt.UserRole))
            for index in range(self.queue_list.count())
        ]

    def run_current_module(self) -> None:
        item = self.queue_list.currentItem() or self.module_list.currentItem()
        if item is None:
            QMessageBox.warning(self, "模块未选择", "请先选择一个功能模块")
            return
        self.start_module_execution([str(item.data(Qt.UserRole))])

    def run_queue(self) -> None:
        self.start_module_execution(self.queue_module_ids())

    def start_module_execution(self, module_ids: list[str]) -> None:
        if self.worker_thread is not None and self.worker_thread.isRunning():
            self.append_log("已有模块流程正在运行")
            return
        if not module_ids:
            QMessageBox.warning(self, "队列为空", "请先加入需要执行的功能模块")
            return
        assert self.executor is not None
        plan = self.registry.resolve(
            module_ids,
            include_dependencies=bool(self.config.get("modules", {}).get("auto_dependencies", True)),
        )
        if any(module.moves_robot for module in plan) and bool(
            self.config.get("modules", {}).get("confirm_robot_motion", True)
        ):
            answer = QMessageBox.question(
                self,
                "确认机器人动作",
                "当前组合包含机械臂运动。请确认工作区无人、无障碍且急停可用。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        self.workflow.robot.reset_cancel()
        self._active_module_ids = list(module_ids)
        self.running_mode = "模块运行"
        self.set_run_controls_enabled(False)
        self.module_status_list.clear()
        for module in plan:
            status_item = QListWidgetItem()
            status_item.setData(Qt.UserRole, module.module_id)
            self.module_status_list.addItem(status_item)
            self.set_module_execution_state(module.module_id, "pending")

        for index in range(self.queue_list.count()):
            self.set_queue_item_state(self.queue_list.item(index), "pending")

        auto_dependencies = bool(self.config.get("modules", {}).get("auto_dependencies", True))
        self.worker_thread = QThread(self)
        self.worker = WorkflowWorker(
            lambda: self.executor.execute(
                module_ids,
                auto_dependencies=auto_dependencies,
                progress=self.worker.progress.emit,
            )
        )
        self.executor.log = self.worker.log.emit
        self.workflow.log = self.worker.log.emit
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.progress.connect(self.set_module_execution_state)
        self.worker.finished.connect(self.on_modules_finished)
        self.worker.failed.connect(self.on_modules_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self.clear_worker)
        self.worker_thread.start()

    def on_modules_finished(self, results) -> None:
        completed_ids = {result.module_id for result in results}
        for module_id in completed_ids:
            self.set_module_execution_state(module_id, "success")
        if "camera.start" in self.executor.completed:
            self.camera_ready = True
            self.timer.start(33)
        self.tcp_ready = self.workflow.robot.connected
        self.robot_ready = self.workflow.robot.enabled
        self.running_mode = "模拟" if self.config.get("mode") == "simulation" else "手动"
        self.set_run_controls_enabled(True)
        self.refresh_target_info()
        self.refresh_all_status()
        self.append_log(f"组合流程完成：本次执行 {len(results)} 个模块")
        self.refresh_queue_status()

    def on_modules_failed(self, message: str) -> None:
        queue_has_failure = any(
            self.queue_list.item(index).data(Qt.UserRole + 1) == "failed"
            for index in range(self.queue_list.count())
        )
        if not queue_has_failure:
            for module_id in self._active_module_ids:
                matching_items = [
                    self.queue_list.item(index)
                    for index in range(self.queue_list.count())
                    if str(self.queue_list.item(index).data(Qt.UserRole)) == module_id
                ]
                if matching_items and not all(
                    item.data(Qt.UserRole + 1) == "success" for item in matching_items
                ):
                    for item in matching_items:
                        self.set_queue_item_state(item, "failed")
                    break
        self.running_mode = "故障"
        self.tcp_ready = self.workflow.robot.connected
        self.robot_ready = self.workflow.robot.enabled
        self.set_run_controls_enabled(True)
        self.refresh_all_status()
        self.append_log("模块流程失败：" + message)
        QMessageBox.warning(self, "模块执行失败", message)

    def clear_worker(self) -> None:
        super().clear_worker()
        if self.executor is not None:
            self.executor.log = self.append_log

    def set_run_controls_enabled(self, enabled: bool) -> None:
        for button in self.run_buttons:
            button.setEnabled(enabled)

    def refresh_queue_status(self) -> None:
        if not hasattr(self, "module_status_list"):
            return
        self.module_status_list.clear()
        for module_id in self.queue_module_ids():
            module = self.registry.get(module_id)
            state = "success" if self.executor and module_id in self.executor.completed else "pending"
            item = QListWidgetItem()
            item.setData(Qt.UserRole, module.module_id)
            self.module_status_list.addItem(item)
            self.set_module_execution_state(module_id, state)


def main() -> int:
    if QApplication is None or cv2 is None:
        print("启动失败：缺少 GUI 运行依赖。")
        print("请安装 requirements_competition.txt 中的依赖。")
        if IMPORT_ERROR is not None:
            print(f"具体错误：{IMPORT_ERROR}")
        return 1
    app = QApplication(sys.argv)
    window = NationalMainWindow()
    window.showMaximized()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
