from __future__ import annotations

from pathlib import Path

from app.core.model_profiles import (
    ModelProfile,
    import_model_bundle,
    scan_model_profiles,
)

try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFileDialog,
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
except ImportError:
    QDialog = object  # type: ignore[misc,assignment]


class ImportModelDialog(QDialog):
    def __init__(self, source_path: Path, parent=None):
        super().__init__(parent)
        self.source_path = source_path
        self.setWindowTitle("导入外部 YOLO 模型")
        self.setMinimumWidth(520)
        root = QVBoxLayout(self)
        form = QFormLayout()
        source = QLabel(str(source_path))
        source.setWordWrap(True)
        self.name_edit = QLineEdit(source_path.stem)
        self.classes_edit = QLineEdit()
        self.classes_edit.setPlaceholderText("按训练顺序填写，例如: part_a, part_b")
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.01, 1.0)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(0.45)
        form.addRow("权重文件", source)
        form.addRow("模型名称", self.name_edit)
        form.addRow("类别顺序", self.classes_edit)
        form.addRow("置信度阈值", self.threshold)
        root.addLayout(form)
        note = QLabel("类别顺序必须与训练时 data.yaml 完全一致，错误顺序会导致分拣路线错配。")
        note.setWordWrap(True)
        note.setStyleSheet("color:#9a3412;font-weight:600;")
        root.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    @property
    def values(self):
        classes = [
            value.strip()
            for value in self.classes_edit.text().replace("，", ",").split(",")
            if value.strip()
        ]
        return self.name_edit.text().strip(), classes, self.threshold.value()

    def _validate(self) -> None:
        name, classes, _threshold = self.values
        if not name or not classes:
            QMessageBox.warning(self, "信息不完整", "请填写模型名称和至少一个类别")
            return
        self.accept()


class ModelManagerDialog(QDialog):
    def __init__(
        self,
        project_root: str | Path,
        current_weights: str | Path | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.project_root = Path(project_root).resolve()
        self.models_root = self.project_root / "models" / "field_models"
        self.current_weights = self._resolve_current(current_weights)
        self.profiles: list[ModelProfile] = []
        self.selected_profile: ModelProfile | None = None
        self.setWindowTitle("比赛模型管理")
        self.resize(1040, 560)

        root = QVBoxLayout(self)
        summary = QLabel(
            "扫描现场训练和导入模型。应用前会先加载验证；任务运行期间不能切换。"
        )
        summary.setWordWrap(True)
        root.addWidget(summary)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["当前", "模型名称", "类别", "训练时间", "图片数", "指标", "权重路径"]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        self.table.doubleClicked.connect(self._accept_selected)
        root.addWidget(self.table)
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color:#b91c1c;")
        root.addWidget(self.error_label)

        actions = QHBoxLayout()
        refresh = QPushButton("刷新列表")
        refresh.clicked.connect(self.refresh)
        import_button = QPushButton("导入外部 .pt")
        import_button.clicked.connect(self.import_external)
        apply_button = QPushButton("应用所选模型")
        apply_button.setDefault(True)
        apply_button.clicked.connect(self._accept_selected)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.reject)
        actions.addWidget(refresh)
        actions.addWidget(import_button)
        actions.addStretch(1)
        actions.addWidget(apply_button)
        actions.addWidget(close_button)
        root.addLayout(actions)
        self.refresh()

    def refresh(self) -> None:
        self.profiles, errors = scan_model_profiles(self.models_root, self.project_root)
        self.table.setRowCount(len(self.profiles))
        current_row = -1
        for row, profile in enumerate(self.profiles):
            is_current = self.current_weights is not None and (
                profile.weights_path.resolve() == self.current_weights
            )
            if is_current:
                current_row = row
            metrics = profile.training.get("metrics") or {}
            metric_text = _metric_summary(metrics)
            values = [
                "使用中" if is_current else "",
                profile.name,
                ", ".join(profile.class_names),
                profile.trained_at or "外部导入",
                str(profile.image_count) if profile.image_count else "-",
                metric_text,
                profile.weights_value,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in (0, 4):
                    item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, column, item)
        if self.profiles:
            self.table.selectRow(current_row if current_row >= 0 else 0)
        self.error_label.setText(
            (f"忽略了 {len(errors)} 个损坏的模型配置：" + "；".join(errors[:2]))
            if errors
            else ""
        )

    def import_external(self) -> None:
        filename, _filter = QFileDialog.getOpenFileName(
            self, "选择 YOLO 权重", str(self.project_root), "YOLO 权重 (*.pt)"
        )
        if not filename:
            return
        dialog = ImportModelDialog(Path(filename), self)
        if dialog.exec_() != dialog.Accepted:
            return
        try:
            name, classes, threshold = dialog.values
            profile = import_model_bundle(
                filename,
                self.project_root,
                name=name,
                class_names=classes,
                conf_threshold=threshold,
            )
            self.refresh()
            for row, item in enumerate(self.profiles):
                if item.profile_path == profile.profile_path:
                    self.table.selectRow(row)
                    break
        except Exception as exc:
            QMessageBox.warning(self, "导入失败", str(exc))

    def _accept_selected(self, *_args) -> None:
        row = self.table.currentRow()
        if not 0 <= row < len(self.profiles):
            QMessageBox.warning(self, "未选择模型", "请先选择一个模型")
            return
        self.selected_profile = self.profiles[row]
        self.accept()

    def _resolve_current(self, value: str | Path | None) -> Path | None:
        if not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()


def _metric_summary(metrics) -> str:
    if not isinstance(metrics, dict) or not metrics:
        return "-"
    preferred = (
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/precision(B)",
        "metrics/recall(B)",
    )
    output = []
    for key in preferred:
        if key in metrics:
            try:
                output.append(f"{key.split('/')[-1]}={float(metrics[key]):.3f}")
            except (TypeError, ValueError):
                pass
    return "  ".join(output[:2]) or "已记录"
