from __future__ import annotations

from copy import deepcopy
from typing import Any

import yaml
from app.core.vision_task import VisionTaskProcessor
from app.core.action_sequence import ActionSequencePlanner
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


def escape_control_text(value: Any) -> str:
    return (
        str(value or "")
        .replace("\\", "\\\\")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )


def unescape_control_text(value: str) -> str:
    output: list[str] = []
    index = 0
    escapes = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\"}
    while index < len(value):
        if value[index] == "\\" and index + 1 < len(value) and value[index + 1] in escapes:
            output.append(escapes[value[index + 1]])
            index += 2
        else:
            output.append(value[index])
            index += 1
    return "".join(output)


class FieldConfigDialog(QDialog):
    def __init__(self, config: dict, serial_ports: list[dict] | None = None, parent=None):
        super().__init__(parent)
        self.config = config
        self.serial_ports = serial_ports or []
        self._updates: dict[str, Any] | None = None
        self.setWindowTitle("DVS 现场通信与任务配置")
        self.setMinimumSize(720, 620)
        self._build_ui()
        self._load_values()

    @property
    def updates(self) -> dict[str, Any]:
        if self._updates is None:
            raise RuntimeError("现场配置尚未确认")
        return self._updates

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self._build_transport_tab(), "通信")
        tabs.addTab(self._build_task_tab(), "任务字段（高级）")
        tabs.addTab(self._build_cycle_tab(), "连续任务")
        tabs.addTab(self._build_action_sequence_tab(), "动作序列（高级）")
        root.addWidget(tabs)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("保存并立即应用")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _build_transport_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.mode_combo = QComboBox()
        for mode, name in (
            ("disabled", "未配置"),
            ("serial", "串口"),
            ("tcp_client", "TCP 客户端"),
            ("tcp_server", "TCP 服务端"),
        ):
            self.mode_combo.addItem(name, mode)
        self.serial_port_combo = QComboBox()
        self.serial_port_combo.setEditable(True)
        for item in self.serial_ports:
            device = str(item.get("device", ""))
            description = str(item.get("description", ""))
            if device:
                self.serial_port_combo.addItem(
                    f"{device} - {description}" if description else device,
                    device,
                )
        self.baudrate_edit = QLineEdit()
        self.tcp_host_edit = QLineEdit()
        self.tcp_port_edit = QLineEdit()
        self.command_edit = QLineEdit()
        self.send_before_check = QCheckBox("本程序先发送触发/查询字符串")
        self.send_terminator_edit = QLineEdit()
        self.receive_terminator_edit = QLineEdit()
        self.delimiter_edit = QLineEdit()
        self.fields_edit = QLineEdit()
        self.fields_edit.setPlaceholderText("status, x, y；JSON 返回时留空")
        self.global_values_edit = QLineEdit()
        self.global_values_edit.setPlaceholderText("task=1; threshold=0.8")
        form.addRow("通信模式", self.mode_combo)
        form.addRow("串口", self.serial_port_combo)
        form.addRow("波特率", self.baudrate_edit)
        form.addRow("TCP IP/监听地址", self.tcp_host_edit)
        form.addRow("TCP 端口", self.tcp_port_edit)
        form.addRow("触发字符串", self.command_edit)
        form.addRow("触发方式", self.send_before_check)
        form.addRow("发送结束符", self.send_terminator_edit)
        form.addRow("接收结束符", self.receive_terminator_edit)
        form.addRow("字段分隔符", self.delimiter_edit)
        form.addRow("返回字段顺序", self.fields_edit)
        form.addRow("DVS 全局变量", self.global_values_edit)
        note = QLabel(
            "结束符可写 \\n、\\r\\n；空白分隔写 whitespace。"
            "端点参数与机械臂 192.168.5.1:29999 完全独立。"
        )
        note.setWordWrap(True)
        form.addRow(note)
        return page

    def _build_task_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.profile_combo = QComboBox()
        self.profile_combo.setEditable(True)
        profiles = self.config.get("vision_tasks", {}).get("profiles", {})
        self.profile_combo.addItems(str(name) for name in profiles)
        self.profile_combo.activated[str].connect(self._load_profile_text)
        form.addRow("活动任务 Profile", self.profile_combo)
        layout.addLayout(form)
        note = QLabel(
            "高级模式：可直接编辑完整 Profile。正常比赛请使用主界面的“任务向导”，"
            "无需接触 YAML。内容保存前会进行 YAML 和数据结构校验。"
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.profile_editor = QPlainTextEdit()
        self.profile_editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self.profile_editor, stretch=1)
        return page

    def _build_cycle_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.cycle_max_items = QSpinBox()
        self.cycle_max_items.setRange(1, 100)
        self.cycle_delay = QDoubleSpinBox()
        self.cycle_delay.setRange(0.0, 30.0)
        self.cycle_delay.setDecimals(2)
        self.cycle_delay.setSuffix(" s")
        self.cycle_processor = QComboBox()
        self.cycle_processor.addItem("只标准化字段", "none")
        self.cycle_processor.addItem("缺陷/状态 OK-NG 判定", "status")
        self.cycle_processor.addItem("尺寸公差判定", "measurement")
        self.cycle_processor.addItem("状态 + 尺寸联合判定", "status_measurement")
        self.cycle_identifiers = QCheckBox("校验 OCR/条码/二维码字段")
        self.cycle_route = QCheckBox("按结果选择路线")
        self.cycle_motion = QCheckBox("执行 DVS 坐标抓放/装配")
        self.cycle_reset_places = QCheckBox("每次启动时从第一个槽位/第一层开始")
        self.cycle_before_sequence = QComboBox()
        self.cycle_before_sequence.setEditable(True)
        self.cycle_after_sequence = QComboBox()
        self.cycle_after_sequence.setEditable(True)
        sequence_names = list(self.config.get("action_sequences", {}).get("profiles", {}))
        self.cycle_before_sequence.addItem("", "")
        self.cycle_after_sequence.addItem("", "")
        self.cycle_before_sequence.addItems(sequence_names)
        self.cycle_after_sequence.addItems(sequence_names)
        self.cycle_empty_field = QLineEdit()
        self.cycle_empty_field.setPlaceholderText("例如 status；不需要空料信号时留空")
        self.cycle_empty_values = QLineEdit()
        self.cycle_empty_values.setPlaceholderText("EMPTY, NONE, NO_TARGET")
        self.cycle_on_error = QComboBox()
        self.cycle_on_error.addItem("立即停止", "stop")
        self.cycle_on_error.addItem("记录并跳过本次", "skip")
        form.addRow("最多处理件数", self.cycle_max_items)
        form.addRow("每次间隔", self.cycle_delay)
        form.addRow("结果处理方式", self.cycle_processor)
        form.addRow("识别字段", self.cycle_identifiers)
        form.addRow("条件路由", self.cycle_route)
        form.addRow("机器人动作", self.cycle_motion)
        form.addRow("每件检测前动作", self.cycle_before_sequence)
        form.addRow("每件完成后动作", self.cycle_after_sequence)
        form.addRow("放置计数", self.cycle_reset_places)
        form.addRow("空料状态字段", self.cycle_empty_field)
        form.addRow("空料状态值", self.cycle_empty_values)
        form.addRow("单次出错策略", self.cycle_on_error)
        note = QLabel("启用机器人动作时，错误策略必须为“立即停止”。急停会同时终止通信循环。")
        note.setWordWrap(True)
        form.addRow(note)
        return page

    def _build_action_sequence_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.sequence_profile_combo = QComboBox()
        self.sequence_profile_combo.setEditable(True)
        sequence_profiles = self.config.get("action_sequences", {}).get("profiles", {})
        self.sequence_profile_combo.addItems(str(name) for name in sequence_profiles)
        self.sequence_profile_combo.activated[str].connect(self._load_sequence_profile_text)
        form.addRow("活动动作 Profile", self.sequence_profile_combo)
        layout.addLayout(form)
        pose_label = QLabel("位姿库（必须是基坐标系 6 轴位姿）")
        layout.addWidget(pose_label)
        self.pose_bank_editor = QPlainTextEdit()
        self.pose_bank_editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.pose_bank_editor.setMaximumHeight(150)
        layout.addWidget(self.pose_bank_editor)
        sequence_label = QLabel(
            "高级模式动作步骤；正常比赛请使用主界面的“动作编辑器”。"
            "保存前检查结构，运行预检时检查引用和安全 Z。"
        )
        sequence_label.setWordWrap(True)
        layout.addWidget(sequence_label)
        self.sequence_editor = QPlainTextEdit()
        self.sequence_editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self.sequence_editor, stretch=1)
        return page

    def _load_values(self) -> None:
        dvs = self.config.get("vision_studio", {})
        mode_index = self.mode_combo.findData(str(dvs.get("mode", "disabled")))
        self.mode_combo.setCurrentIndex(max(0, mode_index))
        serial = dvs.get("serial", {})
        configured_port = str(serial.get("port") or "")
        port_index = self.serial_port_combo.findData(configured_port)
        if port_index >= 0:
            self.serial_port_combo.setCurrentIndex(port_index)
        else:
            self.serial_port_combo.setEditText(configured_port)
        self.baudrate_edit.setText(str(serial.get("baudrate", 115200)))
        tcp = dvs.get("tcp", {})
        self.tcp_host_edit.setText(str(tcp.get("host") or ""))
        self.tcp_port_edit.setText(str(tcp.get("port") or ""))
        protocol = dvs.get("protocol", {})
        self.command_edit.setText(str(protocol.get("exchange_command") or ""))
        self.send_before_check.setChecked(bool(protocol.get("send_before_receive", True)))
        self.send_terminator_edit.setText(escape_control_text(dvs.get("send_terminator", "")))
        self.receive_terminator_edit.setText(escape_control_text(dvs.get("receive_terminator", "")))
        self.delimiter_edit.setText(str(protocol.get("delimiter", ",")))
        self.fields_edit.setText(", ".join(str(item) for item in protocol.get("response_fields", [])))
        self.global_values_edit.setText(
            "; ".join(f"{name}={value}" for name, value in protocol.get("global_values", {}).items())
        )
        active = str(self.config.get("vision_tasks", {}).get("active_profile", "generic"))
        profile_index = self.profile_combo.findText(active)
        if profile_index >= 0:
            self.profile_combo.setCurrentIndex(profile_index)
        else:
            self.profile_combo.setEditText(active)
        self._load_profile_text(active)
        cycle = self.config.get("dvs_cycle", {})
        self.cycle_max_items.setValue(int(cycle.get("max_items", 8)))
        self.cycle_delay.setValue(float(cycle.get("delay_s", 0.2)))
        processor_index = self.cycle_processor.findData(str(cycle.get("processor", "none")))
        self.cycle_processor.setCurrentIndex(max(0, processor_index))
        self.cycle_identifiers.setChecked(bool(cycle.get("validate_identifiers", False)))
        self.cycle_route.setChecked(bool(cycle.get("route", False)))
        self.cycle_motion.setChecked(bool(cycle.get("execute_motion", False)))
        self.cycle_before_sequence.setCurrentText(str(cycle.get("before_each_sequence") or ""))
        self.cycle_after_sequence.setCurrentText(str(cycle.get("after_each_sequence") or ""))
        self.cycle_reset_places.setChecked(bool(cycle.get("reset_placement_counters_on_start", True)))
        empty = cycle.get("empty", {})
        self.cycle_empty_field.setText(str(empty.get("field") or ""))
        self.cycle_empty_values.setText(
            ", ".join(str(item) for item in empty.get("values", ["EMPTY", "NONE", "NO_TARGET"]))
        )
        error_index = self.cycle_on_error.findData(str(cycle.get("on_error", "stop")))
        self.cycle_on_error.setCurrentIndex(max(0, error_index))
        action_sequences = self.config.get("action_sequences", {})
        active_sequence = str(action_sequences.get("active_profile", "custom"))
        sequence_index = self.sequence_profile_combo.findText(active_sequence)
        if sequence_index >= 0:
            self.sequence_profile_combo.setCurrentIndex(sequence_index)
        else:
            self.sequence_profile_combo.setEditText(active_sequence)
        self.pose_bank_editor.setPlainText(
            yaml.safe_dump(
                action_sequences.get("pose_bank", {}),
                allow_unicode=True,
                sort_keys=False,
            ).strip()
            + "\n"
        )
        self._load_sequence_profile_text(active_sequence)

    def _load_profile_text(self, name: str) -> None:
        profiles = self.config.get("vision_tasks", {}).get("profiles", {})
        profile = profiles.get(name)
        if profile is None:
            profile = deepcopy(profiles.get("generic", {}))
        self.profile_editor.setPlainText(
            yaml.safe_dump(profile, allow_unicode=True, sort_keys=False).strip() + "\n"
        )

    def _load_sequence_profile_text(self, name: str) -> None:
        profiles = self.config.get("action_sequences", {}).get("profiles", {})
        profile = profiles.get(name)
        if profile is None:
            profile = deepcopy(profiles.get("custom", {"steps": []}))
        self.sequence_editor.setPlainText(
            yaml.safe_dump(profile, allow_unicode=True, sort_keys=False).strip() + "\n"
        )

    def _selected_serial_port(self) -> str:
        index = self.serial_port_combo.currentIndex()
        data = self.serial_port_combo.itemData(index) if index >= 0 else None
        line_edit = self.serial_port_combo.lineEdit()
        if data and line_edit is not None and not line_edit.isModified():
            return str(data)
        text = self.serial_port_combo.currentText().strip()
        return text.split(" - ", 1)[0].strip()

    def _validate_and_accept(self) -> None:
        try:
            mode = str(self.mode_combo.currentData())
            serial_port = self._selected_serial_port() or None
            baudrate = int(self.baudrate_edit.text().strip())
            if baudrate <= 0:
                raise ValueError("波特率必须大于 0")
            tcp_host = self.tcp_host_edit.text().strip() or None
            tcp_port_text = self.tcp_port_edit.text().strip()
            tcp_port = int(tcp_port_text) if tcp_port_text else None
            if tcp_port is not None and not 1 <= tcp_port <= 65535:
                raise ValueError("TCP 端口必须在 1~65535 之间")
            if mode == "serial" and not serial_port:
                raise ValueError("串口模式必须选择或填写 COM 口")
            if mode == "tcp_client" and (not tcp_host or tcp_port is None):
                raise ValueError("TCP 客户端模式必须填写目标 IP 和端口")
            if mode == "tcp_server" and tcp_port is None:
                raise ValueError("TCP 服务端模式必须填写监听端口")
            command = self.command_edit.text()
            if self.send_before_check.isChecked() and mode != "disabled" and not command:
                raise ValueError("主动触发模式必须填写触发字符串；DVS 主动上报时请取消勾选")
            field_names = [item.strip() for item in self.fields_edit.text().split(",") if item.strip()]
            if len(set(field_names)) != len(field_names):
                raise ValueError("返回字段名不能重复")
            global_values: dict[str, Any] = {}
            for assignment in self.global_values_edit.text().split(";"):
                assignment = assignment.strip()
                if not assignment:
                    continue
                if "=" not in assignment:
                    raise ValueError(f"DVS 全局变量格式应为 name=value: {assignment}")
                name, raw_value = assignment.split("=", 1)
                name = name.strip()
                if not name:
                    raise ValueError("DVS 全局变量名不能为空")
                global_values[name] = yaml.safe_load(raw_value.strip())
            profile_name = self.profile_combo.currentText().strip()
            if not profile_name:
                raise ValueError("活动任务 Profile 名称不能为空")
            profile = yaml.safe_load(self.profile_editor.toPlainText()) or {}
            if not isinstance(profile, dict):
                raise ValueError("任务 Profile 顶层必须是 YAML 映射")
            VisionTaskProcessor(profile).validate_settings()
            sequence_name = self.sequence_profile_combo.currentText().strip()
            if not sequence_name:
                raise ValueError("活动动作 Profile 名称不能为空")
            pose_bank = yaml.safe_load(self.pose_bank_editor.toPlainText()) or {}
            sequence_profile = yaml.safe_load(self.sequence_editor.toPlainText()) or {}
            if not isinstance(pose_bank, dict):
                raise ValueError("动作位姿库顶层必须是 YAML 映射")
            if not isinstance(sequence_profile, dict):
                raise ValueError("动作 Profile 顶层必须是 YAML 映射")
            ActionSequencePlanner(
                {
                    "active_profile": sequence_name,
                    "pose_bank": pose_bank,
                    "profiles": {sequence_name: sequence_profile},
                },
                self.config.get("robot", {}),
            ).validate_settings()
            on_error = str(self.cycle_on_error.currentData())
            before_sequence = self.cycle_before_sequence.currentText().strip() or None
            after_sequence = self.cycle_after_sequence.currentText().strip() or None
            if (self.cycle_motion.isChecked() or before_sequence or after_sequence) and on_error != "stop":
                raise ValueError("连续任务包含机器人动作时，单次出错策略必须为“立即停止”")
            empty_values = [
                item.strip() for item in self.cycle_empty_values.text().split(",") if item.strip()
            ]
            self._updates = {
                "vision_studio": {
                    "mode": mode,
                    "send_terminator": unescape_control_text(self.send_terminator_edit.text()),
                    "receive_terminator": unescape_control_text(self.receive_terminator_edit.text()),
                    "serial": {"port": serial_port, "baudrate": baudrate},
                    "tcp": {"host": tcp_host, "port": tcp_port},
                    "protocol": {
                        "send_before_receive": self.send_before_check.isChecked(),
                        "exchange_command": command or None,
                        "delimiter": self.delimiter_edit.text() or ",",
                        "response_fields": field_names,
                        "global_values": global_values,
                    },
                },
                "vision_tasks": {
                    "active_profile": profile_name,
                    "profiles": {profile_name: profile},
                },
                "dvs_cycle": {
                    "max_items": self.cycle_max_items.value(),
                    "delay_s": self.cycle_delay.value(),
                    "processor": str(self.cycle_processor.currentData()),
                    "validate_identifiers": self.cycle_identifiers.isChecked(),
                    "route": self.cycle_route.isChecked(),
                    "execute_motion": self.cycle_motion.isChecked(),
                    "before_each_sequence": before_sequence,
                    "after_each_sequence": after_sequence,
                    "reset_placement_counters_on_start": self.cycle_reset_places.isChecked(),
                    "empty": {
                        "field": self.cycle_empty_field.text().strip() or None,
                        "values": empty_values,
                    },
                    "on_error": on_error,
                },
                "action_sequences": {
                    "active_profile": sequence_name,
                    "pose_bank": pose_bank,
                    "profiles": {sequence_name: sequence_profile},
                },
            }
        except (TypeError, ValueError, RuntimeError, yaml.YAMLError) as exc:
            QMessageBox.warning(self, "现场配置无效", str(exc))
            return
        self.accept()
