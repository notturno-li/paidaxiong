from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from app.core.action_sequence import ActionSequencePlanner
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


ACTION_LABELS = {
    "movj": "关节运动 MovJ",
    "movl": "直线运动 MovL",
    "movl_slow": "低速下降 MovL",
    "suction": "吸盘开关",
    "wait": "等待",
    "home": "返回安全点",
}


class ActionSequenceDialog(QDialog):
    """Structured editor for fixed robot sequences and taught poses."""

    def __init__(self, config: dict[str, Any], parent=None):
        super().__init__(parent)
        self.config = config
        self._updates: dict[str, Any] | None = None
        self.setWindowTitle("固定动作零代码编辑器")
        self.resize(930, 680)
        self.setMinimumSize(800, 600)
        self._build_ui()
        self._load_all()

    @property
    def updates(self) -> dict[str, Any]:
        if self._updates is None:
            raise RuntimeError("动作序列尚未确认")
        return self._updates

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        form = QFormLayout()
        self.template_combo = QComboBox()
        profiles = self.config.get("action_sequences", {}).get("profiles", {})
        for name in profiles:
            self.template_combo.addItem(str(name), str(name))
        self.template_combo.currentIndexChanged.connect(self._load_selected_profile)
        self.name_edit = QLineEdit()
        form.addRow("从已有模板加载", self.template_combo)
        form.addRow("保存为动作 Profile", self.name_edit)
        root.addLayout(form)

        root.addWidget(QLabel("示教位姿库（机器人基坐标系）"))
        self.pose_table = QTableWidget(0, 7)
        self.pose_table.setHorizontalHeaderLabels(["位姿名", "X", "Y", "Z", "Rx", "Ry", "Rz"])
        self.pose_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.pose_table.setSelectionBehavior(QTableWidget.SelectRows)
        root.addWidget(self.pose_table, stretch=2)
        pose_buttons = QHBoxLayout()
        add_pose = QPushButton("增加位姿")
        remove_pose = QPushButton("删除位姿")
        add_pose.clicked.connect(lambda: self._append_pose(["", "", "", "", "", "", ""]))
        remove_pose.clicked.connect(lambda: self._remove_row(self.pose_table))
        pose_buttons.addWidget(add_pose)
        pose_buttons.addWidget(remove_pose)
        pose_buttons.addStretch(1)
        root.addLayout(pose_buttons)

        instruction = QLabel(
            "动作参数填写规则：运动步骤填位姿名；吸盘填 开/关；等待填秒数；返回安全点无需参数。"
        )
        instruction.setWordWrap(True)
        root.addWidget(instruction)
        self.step_table = QTableWidget(0, 2)
        self.step_table.setHorizontalHeaderLabels(["动作类型", "位姿名 / 开关 / 秒数"])
        self.step_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.step_table.setSelectionBehavior(QTableWidget.SelectRows)
        root.addWidget(self.step_table, stretch=2)
        step_buttons = QHBoxLayout()
        add_step = QPushButton("增加步骤")
        remove_step = QPushButton("删除步骤")
        up_step = QPushButton("上移")
        down_step = QPushButton("下移")
        add_step.clicked.connect(lambda: self._append_step("movj", ""))
        remove_step.clicked.connect(lambda: self._remove_row(self.step_table))
        up_step.clicked.connect(lambda: self._move_step(-1))
        down_step.clicked.connect(lambda: self._move_step(1))
        for button in (add_step, remove_step, up_step, down_step):
            step_buttons.addWidget(button)
        step_buttons.addStretch(1)
        root.addLayout(step_buttons)

        note = QLabel(
            "保存时会检查动作、引用、六轴格式和最低安全 Z。建议把接近点与工作点分开示教，"
            "执行前先在模块库运行“预览并校验固定动作序列”。"
        )
        note.setWordWrap(True)
        root.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("保存并立即应用")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _load_all(self) -> None:
        settings = self.config.get("action_sequences", {})
        self.pose_table.setRowCount(0)
        for name, pose in settings.get("pose_bank", {}).items():
            self._append_pose([name, *pose])
        active = str(settings.get("active_profile") or "custom")
        index = self.template_combo.findData(active)
        self.template_combo.setCurrentIndex(index if index >= 0 else 0)
        self._load_selected_profile()

    def _load_selected_profile(self) -> None:
        if not hasattr(self, "step_table"):
            return
        name = str(self.template_combo.currentData() or "custom")
        self.name_edit.setText(name)
        profile = deepcopy(
            self.config.get("action_sequences", {}).get("profiles", {}).get(name, {"steps": []})
        )
        self.step_table.setRowCount(0)
        for step in profile.get("steps", []):
            action = str(step.get("action") or "")
            if action in {"movj", "movl", "movl_slow"}:
                argument = str(step.get("pose_ref") or "")
                if not argument and step.get("pose") is not None:
                    argument = ", ".join(str(value) for value in step["pose"])
            elif action == "suction":
                argument = "开" if step.get("value") else "关"
            elif action == "wait":
                argument = str(step.get("seconds", ""))
            else:
                argument = ""
            self._append_step(action, argument)

    def _append_pose(self, values: list[Any]) -> None:
        row = self.pose_table.rowCount()
        self.pose_table.insertRow(row)
        for column, value in enumerate(values):
            self.pose_table.setItem(row, column, QTableWidgetItem(str(value)))

    def _append_step(self, action: str, argument: str) -> None:
        row = self.step_table.rowCount()
        self.step_table.insertRow(row)
        combo = QComboBox()
        for action_id, label in ACTION_LABELS.items():
            combo.addItem(label, action_id)
        index = combo.findData(action)
        combo.setCurrentIndex(index if index >= 0 else 0)
        self.step_table.setCellWidget(row, 0, combo)
        self.step_table.setItem(row, 1, QTableWidgetItem(argument))

    @staticmethod
    def _remove_row(table: QTableWidget) -> None:
        row = table.currentRow()
        if row >= 0:
            table.removeRow(row)

    def _move_step(self, direction: int) -> None:
        row = self.step_table.currentRow()
        target = row + direction
        if row < 0 or target < 0 or target >= self.step_table.rowCount():
            return
        action, argument = self._step_values(row)
        target_action, target_argument = self._step_values(target)
        self._set_step(row, target_action, target_argument)
        self._set_step(target, action, argument)
        self.step_table.setCurrentCell(target, 1)

    def _set_step(self, row: int, action: str, argument: str) -> None:
        combo = self.step_table.cellWidget(row, 0)
        assert isinstance(combo, QComboBox)
        combo.setCurrentIndex(combo.findData(action))
        self.step_table.setItem(row, 1, QTableWidgetItem(argument))

    def _step_values(self, row: int) -> tuple[str, str]:
        combo = self.step_table.cellWidget(row, 0)
        assert isinstance(combo, QComboBox)
        item = self.step_table.item(row, 1)
        return str(combo.currentData()), item.text().strip() if item else ""

    @staticmethod
    def _parse_inline_pose(text: str) -> list[float] | None:
        values = [value for value in re.split(r"[\s,，]+", text.strip()) if value]
        if len(values) != 6:
            return None
        try:
            return [float(value) for value in values]
        except ValueError:
            return None

    def _validate_and_accept(self) -> None:
        try:
            name = self.name_edit.text().strip()
            if not name:
                raise ValueError("动作 Profile 名称不能为空")
            pose_bank: dict[str, list[float]] = {}
            for row in range(self.pose_table.rowCount()):
                values = []
                for column in range(7):
                    item = self.pose_table.item(row, column)
                    values.append(item.text().strip() if item else "")
                if not any(values):
                    continue
                if not values[0]:
                    raise ValueError(f"位姿库第 {row + 1} 行缺少位姿名")
                if values[0] in pose_bank:
                    raise ValueError(f"位姿名重复: {values[0]}")
                pose_bank[values[0]] = [float(value) for value in values[1:]]
            steps: list[dict[str, Any]] = []
            for row in range(self.step_table.rowCount()):
                action, argument = self._step_values(row)
                step: dict[str, Any] = {"action": action}
                if action in {"movj", "movl", "movl_slow"}:
                    if not argument:
                        raise ValueError(f"第 {row + 1} 步运动缺少位姿名")
                    inline_pose = self._parse_inline_pose(argument)
                    if inline_pose is not None:
                        step["pose"] = inline_pose
                    else:
                        step["pose_ref"] = argument
                elif action == "suction":
                    normalized = argument.strip().casefold()
                    if normalized in {"开", "on", "true", "1"}:
                        step["value"] = True
                    elif normalized in {"关", "off", "false", "0"}:
                        step["value"] = False
                    else:
                        raise ValueError(f"第 {row + 1} 步吸盘参数只能填 开 或 关")
                elif action == "wait":
                    step["seconds"] = float(argument)
                steps.append(step)
            if not steps:
                raise ValueError("动作序列至少要有一个步骤")
            settings = {
                "active_profile": name,
                "pose_bank": pose_bank,
                "profiles": {name: {"steps": steps}},
            }
            planner = ActionSequencePlanner(settings, self.config.get("robot", {}))
            planner.plan(name)
            self._updates = {"action_sequences": settings}
        except (TypeError, ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "动作序列无效", str(exc))
            return
        self.accept()
