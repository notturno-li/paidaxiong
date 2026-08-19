from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from app.core.preset_builder import TASK_TYPES, BuiltTaskPreset, build_task_preset
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


DEFAULTS: dict[str, dict[str, Any]] = {
    "communication": {"fields": [], "aliases": []},
    "defect_sort": {
        "fields": ["status", "x", "y", "z"],
        "aliases": [("status", "result, quality, state"), ("x", "robot_x"), ("y", "robot_y"), ("z", "robot_z")],
        "routes": [("OK", "ok"), ("NG", "reject")],
    },
    "category_sort": {
        "fields": ["label", "x", "y", "z"],
        "aliases": [("label", "color, class, type"), ("x", "robot_x"), ("y", "robot_y"), ("z", "robot_z")],
        "route_field": "label",
    },
    "measurement": {
        "fields": ["width", "length"],
        "aliases": [("width", "w, width_mm"), ("length", "l, length_mm"), ("diameter", "d, diameter_mm")],
        "measurements": [("width", "", ""), ("length", "", "")],
    },
    "measurement_sort": {
        "fields": ["width", "length"],
        "aliases": [("width", "w, width_mm"), ("length", "l, length_mm"), ("diameter", "d, diameter_mm")],
        "measurements": [("width", "", ""), ("length", "", "")],
        "routes": [("OK", "ok"), ("NG", "reject")],
    },
    "recognition": {
        "fields": ["ocr"],
        "aliases": [("ocr", "text, characters"), ("barcode", "bar_code, code1d"), ("qrcode", "qr, code2d")],
        "identifiers": "ocr",
        "identifier_rules": [("ocr", "equals", "")],
    },
    "recognition_sort": {
        "fields": ["ocr"],
        "aliases": [("ocr", "text, characters"), ("barcode", "bar_code, code1d"), ("qrcode", "qr, code2d")],
        "identifiers": "ocr",
        "identifier_rules": [("ocr", "equals", "")],
        "routes": [("OK", "ok"), ("NG", "reject")],
    },
    "guided_assembly": {
        "fields": ["part_x", "part_y", "part_z", "hole_x", "hole_y", "hole_z"],
        "aliases": [("part_x", "x, pick_x"), ("part_y", "y, pick_y"), ("part_z", "z, pick_z"), ("hole_x", "target_x, place_x"), ("hole_y", "target_y, place_y"), ("hole_z", "target_z, place_z")],
        "pick_fields": "part_x, part_y, part_z",
        "place_fields": "hole_x, hole_y, hole_z",
        "motion": True,
    },
    "fixed_inspection": {
        "fields": ["status", "x", "y", "z"],
        "aliases": [("status", "result, quality, state"), ("x", "robot_x"), ("y", "robot_y"), ("z", "robot_z")],
        "routes": [("OK", "ok"), ("NG", "reject")],
        "before": "inspection_feed_template",
        "motion": True,
    },
    "combined": {
        "fields": ["status", "label", "width", "ocr", "x", "y", "z"],
        "aliases": [
            ("status", "result, quality, state"),
            ("label", "color, class, type"),
            ("width", "w, width_mm"),
            ("ocr", "text, characters"),
            ("x", "robot_x"),
            ("y", "robot_y"),
            ("z", "robot_z"),
        ],
        "measurements": [("width", "", "")],
        "identifiers": "ocr",
        "identifier_rules": [("ocr", "equals", "")],
    },
}


class TaskWizardDialog(QDialog):
    """Build a national-competition task without exposing Python or YAML."""

    def __init__(self, config: dict[str, Any], parent=None):
        super().__init__(parent)
        self.config = config
        self._result: BuiltTaskPreset | None = None
        self.setWindowTitle("国赛零代码任务向导")
        self.resize(940, 700)
        self.setMinimumSize(820, 620)
        self._build_ui()
        self._load_task_type()

    @property
    def result(self) -> BuiltTaskPreset:
        if self._result is None:
            raise RuntimeError("任务向导尚未生成配置")
        return self._result

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        banner = QLabel("按任务书填写字段和示教位姿；生成后会创建永久现场配方并装载推荐模块队列。")
        banner.setWordWrap(True)
        root.addWidget(banner)
        tabs = QTabWidget()
        tabs.addTab(self._build_basic_tab(), "1 题型与循环")
        tabs.addTab(self._build_fields_tab(), "2 字段与判定")
        tabs.addTab(self._build_routes_tab(), "3 路线与放置")
        tabs.addTab(self._build_motion_tab(), "4 抓取与装配")
        root.addWidget(tabs, stretch=1)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("生成并应用现场配方")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self._build_result)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _build_basic_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.task_type_combo = QComboBox()
        for task_id, settings in TASK_TYPES.items():
            self.task_type_combo.addItem(settings["name"], task_id)
        self.task_type_combo.currentIndexChanged.connect(self._load_task_type)
        self.profile_name_edit = QLineEdit()
        self.recipe_name_edit = QLineEdit()
        self.response_fields_edit = QLineEdit()
        self.response_fields_edit.setPlaceholderText("按 DVS 报文顺序填写；JSON 报文可留空")
        self.use_cycle_check = QCheckBox("使用连续任务引擎（单件也建议勾选）")
        self.use_cycle_check.setChecked(True)
        self.processor_combo = QComboBox()
        self.processor_combo.addItem("只标准化字段", "none")
        self.processor_combo.addItem("状态 OK/NG 判定", "status")
        self.processor_combo.addItem("尺寸公差判定", "measurement")
        self.processor_combo.addItem("状态 + 尺寸联合判定", "status_measurement")
        self.identifiers_check = QCheckBox("校验 OCR/条码/二维码字段及规则")
        self.route_check = QCheckBox("按判定值或指定字段选择路线")
        self.max_items_spin = QSpinBox()
        self.max_items_spin.setRange(1, 100)
        self.max_items_spin.setValue(8)
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0, 30)
        self.delay_spin.setDecimals(2)
        self.delay_spin.setValue(0.2)
        self.delay_spin.setSuffix(" s")
        self.empty_field_edit = QLineEdit()
        self.empty_field_edit.setPlaceholderText("没有空料状态时留空")
        self.empty_values_edit = QLineEdit("EMPTY, NONE, NO_TARGET")
        self.before_combo = self._sequence_combo()
        self.after_combo = self._sequence_combo()
        form.addRow("任务类型", self.task_type_combo)
        form.addRow("任务 Profile 名称", self.profile_name_edit)
        form.addRow("现场配方显示名称", self.recipe_name_edit)
        form.addRow("DVS 返回字段顺序", self.response_fields_edit)
        form.addRow("执行方式", self.use_cycle_check)
        form.addRow("结果处理", self.processor_combo)
        form.addRow("识别核验", self.identifiers_check)
        form.addRow("条件路由", self.route_check)
        form.addRow("最多处理件数", self.max_items_spin)
        form.addRow("每件间隔", self.delay_spin)
        form.addRow("空料状态字段", self.empty_field_edit)
        form.addRow("空料状态值", self.empty_values_edit)
        form.addRow("每件检测前固定动作", self.before_combo)
        form.addRow("每件完成后固定动作", self.after_combo)
        note = QLabel("DVS 的 IP、端口或 COM 口仍在“通信配置”中填写；向导不会猜测未知端点。")
        note.setWordWrap(True)
        form.addRow(note)
        return page

    def _build_fields_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        alias_box = QGroupBox("字段别名：把 DVS 实际字段统一成程序标准字段")
        alias_layout = QVBoxLayout(alias_box)
        self.alias_table = self._table(["标准字段", "DVS 实际字段/别名（逗号分隔）"])
        alias_layout.addWidget(self.alias_table)
        alias_layout.addLayout(self._row_buttons(self.alias_table, 2))
        layout.addWidget(alias_box, stretch=2)

        status_box = QGroupBox("状态判定")
        status_form = QFormLayout(status_box)
        self.status_field_edit = QLineEdit("status")
        self.ok_values_edit = QLineEdit("OK, PASS, 1, true")
        self.ng_values_edit = QLineEdit("NG, FAIL, 0, false")
        status_form.addRow("状态字段", self.status_field_edit)
        status_form.addRow("判为 OK 的值", self.ok_values_edit)
        status_form.addRow("判为 NG 的值", self.ng_values_edit)
        layout.addWidget(status_box)

        bottom = QHBoxLayout()
        measurement_box = QGroupBox("尺寸公差")
        measurement_layout = QVBoxLayout(measurement_box)
        self.measurement_table = self._table(["字段", "最小值", "最大值"])
        measurement_layout.addWidget(self.measurement_table)
        measurement_layout.addLayout(self._row_buttons(self.measurement_table, 3))
        bottom.addWidget(measurement_box, stretch=2)
        identifier_box = QGroupBox("识别结果")
        identifier_layout = QVBoxLayout(identifier_box)
        identifier_layout.addWidget(QLabel("必需识别字段（逗号分隔）"))
        self.identifiers_edit = QLineEdit()
        self.identifiers_edit.setPlaceholderText("例如 ocr, barcode")
        identifier_layout.addWidget(self.identifiers_edit)
        rule_note = QLabel("核验方式：equals 精确值；allowed 允许列表；regex 正则")
        rule_note.setWordWrap(True)
        identifier_layout.addWidget(rule_note)
        self.identifier_rules_table = self._table(["字段", "方式", "期望值"])
        self.identifier_rules_table.setMaximumHeight(105)
        identifier_layout.addWidget(self.identifier_rules_table)
        identifier_layout.addLayout(self._row_buttons(self.identifier_rules_table, 3))
        bottom.addWidget(identifier_box, stretch=1)
        layout.addLayout(bottom, stretch=1)
        return page

    def _build_routes_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        route_header = QFormLayout()
        self.route_field_edit = QLineEdit("decision")
        route_header.addRow("用于分流的字段", self.route_field_edit)
        layout.addLayout(route_header)
        self.route_table = self._table(["DVS 值/判定值", "动作路线名"])
        layout.addWidget(self.route_table, stretch=1)
        layout.addLayout(self._row_buttons(self.route_table, 2))

        place_header = QFormLayout()
        self.placement_mode_combo = QComboBox()
        self.placement_mode_combo.addItem("每条路线一个固定点", "fixed")
        self.placement_mode_combo.addItem("同一路线多个顺序槽位", "slots")
        self.placement_mode_combo.addItem("按层高自动堆叠", "stack")
        self.layer_height_spin = QDoubleSpinBox()
        self.layer_height_spin.setRange(0.1, 500)
        self.layer_height_spin.setValue(20)
        self.layer_height_spin.setSuffix(" mm")
        self.max_layers_spin = QSpinBox()
        self.max_layers_spin.setRange(1, 100)
        self.max_layers_spin.setValue(3)
        place_header.addRow("放置方式", self.placement_mode_combo)
        place_header.addRow("堆叠层高", self.layer_height_spin)
        place_header.addRow("最大层数", self.max_layers_spin)
        layout.addLayout(place_header)
        self.placement_table = self._table(["路线名", "X", "Y", "Z", "Rx", "Ry", "Rz"])
        layout.addWidget(self.placement_table, stretch=2)
        layout.addLayout(self._row_buttons(self.placement_table, 7))
        note = QLabel("槽位模式可重复填写同一路线；顺序即放置顺序。所有静态 Z 必须不低于机器人安全下限。")
        note.setWordWrap(True)
        layout.addWidget(note)
        return page

    def _build_motion_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.motion_check = QCheckBox("启用机械臂抓放/装配动作")
        layout.addWidget(self.motion_check)
        pick = QGroupBox("抓取位姿")
        pick_form = QFormLayout(pick)
        self.pick_fields_edit = QLineEdit("x, y, z")
        self.pick_defaults_edit = QLineEdit("0, 0, 0, 180, 0, 0")
        self.pick_offsets_edit = QLineEdit("0, 0, 0, 0, 0, 0")
        self.fixed_pick_check = QCheckBox("使用固定抓取位姿（忽略抓取字段）")
        pick_form.addRow("DVS 抓取字段", self.pick_fields_edit)
        pick_form.addRow("默认/固定六轴位姿", self.pick_defaults_edit)
        pick_form.addRow("六轴补偿", self.pick_offsets_edit)
        pick_form.addRow(self.fixed_pick_check)
        layout.addWidget(pick)
        place = QGroupBox("动态放置/装配位姿（无条件路由时使用）")
        place_form = QFormLayout(place)
        self.place_fields_edit = QLineEdit()
        self.place_defaults_edit = QLineEdit("0, 0, 0, 180, 0, 0")
        self.place_offsets_edit = QLineEdit("0, 0, 0, 0, 0, 0")
        self.fixed_place_check = QCheckBox("使用固定放置位姿（忽略放置字段）")
        place_form.addRow("DVS 放置字段", self.place_fields_edit)
        place_form.addRow("默认/固定六轴位姿", self.place_defaults_edit)
        place_form.addRow("六轴补偿", self.place_offsets_edit)
        place_form.addRow(self.fixed_place_check)
        layout.addWidget(place)
        note = QLabel("所有位姿均为机器人基坐标系 [X,Y,Z,Rx,Ry,Rz]。首次运行先取消机器人动作，只验证 DVS 返回和路线。")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
        return page

    def _sequence_combo(self) -> QComboBox:
        combo = QComboBox()
        combo.addItem("不执行", "")
        for name in self.config.get("action_sequences", {}).get("profiles", {}):
            combo.addItem(str(name), str(name))
        return combo

    @staticmethod
    def _table(headers: list[str]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        return table

    def _row_buttons(self, table: QTableWidget, columns: int) -> QHBoxLayout:
        row = QHBoxLayout()
        add_button = QPushButton("增加一行")
        remove_button = QPushButton("删除选中行")
        add_button.clicked.connect(lambda: self._append_row(table, [""] * columns))
        remove_button.clicked.connect(lambda: self._remove_current_row(table))
        row.addWidget(add_button)
        row.addWidget(remove_button)
        row.addStretch(1)
        return row

    @staticmethod
    def _append_row(table: QTableWidget, values: list[Any]) -> None:
        row = table.rowCount()
        table.insertRow(row)
        for column, value in enumerate(values):
            table.setItem(row, column, QTableWidgetItem(str(value)))

    @staticmethod
    def _remove_current_row(table: QTableWidget) -> None:
        row = table.currentRow()
        if row >= 0:
            table.removeRow(row)

    def _set_rows(self, table: QTableWidget, rows: list[tuple[Any, ...]]) -> None:
        table.setRowCount(0)
        for row in rows:
            self._append_row(table, list(row))

    def _load_task_type(self) -> None:
        if not hasattr(self, "profile_name_edit"):
            return
        task_type = str(self.task_type_combo.currentData())
        defaults = deepcopy(DEFAULTS[task_type])
        label = TASK_TYPES[task_type]["name"]
        self.profile_name_edit.setText("field_" + task_type)
        self.recipe_name_edit.setText("现场-" + label)
        self.response_fields_edit.setText(", ".join(defaults.get("fields", [])))
        self._set_rows(self.alias_table, defaults.get("aliases", []))
        self._set_rows(self.measurement_table, defaults.get("measurements", []))
        self._set_rows(self.identifier_rules_table, defaults.get("identifier_rules", []))
        self._set_rows(self.route_table, defaults.get("routes", []))
        self._set_rows(self.placement_table, [])
        self.identifiers_edit.setText(defaults.get("identifiers", ""))
        self.route_field_edit.setText(defaults.get("route_field", "decision"))
        self.pick_fields_edit.setText(defaults.get("pick_fields", "x, y, z"))
        self.place_fields_edit.setText(defaults.get("place_fields", ""))
        self.motion_check.setChecked(bool(defaults.get("motion", False)))
        template = TASK_TYPES[task_type]
        processor_index = self.processor_combo.findData(str(template.get("processor", "none")))
        self.processor_combo.setCurrentIndex(max(0, processor_index))
        self.identifiers_check.setChecked(bool(template.get("identifiers", False)))
        self.route_check.setChecked(bool(template.get("route", False)))
        self.before_combo.setCurrentIndex(max(0, self.before_combo.findData(defaults.get("before", ""))))
        self.after_combo.setCurrentIndex(0)

    @staticmethod
    def _cells(table: QTableWidget) -> list[list[str]]:
        rows: list[list[str]] = []
        for row in range(table.rowCount()):
            values = []
            for column in range(table.columnCount()):
                item = table.item(row, column)
                values.append(item.text().strip() if item else "")
            if any(values):
                rows.append(values)
        return rows

    @staticmethod
    def _split(text: str) -> list[str]:
        return [part.strip() for part in re.split(r"[,;|，；]+", text) if part.strip()]

    @staticmethod
    def _pose(text: str) -> list[float]:
        values = [part for part in re.split(r"[\s,，]+", text.strip()) if part]
        if len(values) != 6:
            raise ValueError("六轴位姿必须填写 X,Y,Z,Rx,Ry,Rz 六个数值")
        return [float(value) for value in values]

    @staticmethod
    def _scalar(text: str) -> Any:
        value = text.strip()
        lowered = value.casefold()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        try:
            return float(value) if "." in value else int(value)
        except ValueError:
            return value

    def _pose_section(self, fields_edit: QLineEdit, defaults_edit: QLineEdit, offsets_edit: QLineEdit, fixed: QCheckBox) -> dict[str, Any]:
        defaults = self._pose(defaults_edit.text())
        section: dict[str, Any]
        if fixed.isChecked():
            section = {"pose": defaults}
        else:
            fields = self._split(fields_edit.text())
            section = {"fields": fields, "defaults": defaults} if fields else {}
        if section:
            section["offsets"] = self._pose(offsets_edit.text())
        return section

    def _build_result(self) -> None:
        try:
            aliases = {
                canonical: self._split(candidates)
                for canonical, candidates in self._cells(self.alias_table)
                if canonical
            }
            measurements: dict[str, dict[str, Any]] = {}
            for field, minimum, maximum in self._cells(self.measurement_table):
                if field:
                    measurements[field] = {"min": minimum or None, "max": maximum or None}
            route_map = {
                self._scalar(value): route
                for value, route in self._cells(self.route_table)
                if value and route
            }
            placements = []
            for row in self._cells(self.placement_table):
                if len(row) == 7 and row[0] and any(row[1:]):
                    placements.append({"route": row[0], "pose": [float(value) for value in row[1:]]})
            identifier_rules: dict[str, dict[str, Any]] = {}
            for field, mode, expected in self._cells(self.identifier_rules_table):
                if not field or not mode or not expected:
                    continue
                parsed_expected: Any = expected
                if mode.strip().lower() == "allowed":
                    parsed_expected = [
                        self._scalar(value) for value in self._split(expected)
                    ]
                identifier_rules[field] = {
                    "mode": mode.strip().lower(),
                    "expected": parsed_expected,
                }
            spec = {
                "task_type": str(self.task_type_combo.currentData()),
                "profile_name": self.profile_name_edit.text().strip(),
                "recipe_name": self.recipe_name_edit.text().strip(),
                "response_fields": self._split(self.response_fields_edit.text()),
                "aliases": aliases,
                "status": {
                    "field": self.status_field_edit.text().strip(),
                    "ok_values": [self._scalar(value) for value in self._split(self.ok_values_edit.text())],
                    "ng_values": [self._scalar(value) for value in self._split(self.ng_values_edit.text())],
                },
                "measurement_limits": measurements,
                "identifier_fields": self._split(self.identifiers_edit.text()),
                "identifier_rules": identifier_rules,
                "processor": str(self.processor_combo.currentData()),
                "validate_identifiers": self.identifiers_check.isChecked(),
                "route_enabled": self.route_check.isChecked(),
                "route_field": self.route_field_edit.text().strip(),
                "route_map": route_map,
                "execute_motion": self.motion_check.isChecked(),
                "pick": self._pose_section(self.pick_fields_edit, self.pick_defaults_edit, self.pick_offsets_edit, self.fixed_pick_check),
                "place": self._pose_section(self.place_fields_edit, self.place_defaults_edit, self.place_offsets_edit, self.fixed_place_check),
                "placement_mode": str(self.placement_mode_combo.currentData()),
                "placements": placements,
                "layer_height_mm": self.layer_height_spin.value(),
                "max_layers": self.max_layers_spin.value(),
                "use_cycle": self.use_cycle_check.isChecked(),
                "max_items": self.max_items_spin.value(),
                "delay_s": self.delay_spin.value(),
                "empty_field": self.empty_field_edit.text().strip(),
                "empty_values": self._split(self.empty_values_edit.text()),
                "before_each_sequence": str(self.before_combo.currentData() or ""),
                "after_each_sequence": str(self.after_combo.currentData() or ""),
                "on_error": "stop",
            }
            self._result = build_task_preset(spec, self.config.get("robot", {}))
        except (TypeError, ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "任务参数无效", str(exc))
            return
        self.accept()
