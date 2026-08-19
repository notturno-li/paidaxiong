from __future__ import annotations

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault(
    "QT_QPA_PLATFORM", "windows" if "--windows" in sys.argv else "offscreen"
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt5.QtWidgets import QApplication, QPushButton

from app.core.model_profiles import import_model_bundle
from app.model_manager_dialog import ModelManagerDialog
from app.modular_main import NationalMainWindow


def main() -> int:
    output = ROOT / "runs" / "ui_checks"
    output.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])

    window = NationalMainWindow()
    window.resize(1280, 780)
    window.show()
    app.processEvents()
    button_names = {button.text() for button in window.findChildren(QPushButton)}
    required = {"模型管理", "任务向导", "通信/高级", "急停 STOP"}
    missing = required - button_names
    if missing:
        raise RuntimeError("main GUI is missing buttons: " + ", ".join(sorted(missing)))
    if not window.grab().save(str(output / "competition_gui_1280x780.png")):
        raise RuntimeError("failed to save competition GUI screenshot")
    window.resize(1280, 1000)
    app.processEvents()
    if not window.grab().save(str(output / "competition_gui_1280x1000.png")):
        raise RuntimeError("failed to save tall competition GUI screenshot")
    reload_button = next(
        button for button in window.findChildren(QPushButton) if button.text() == "重载配方"
    )
    reload_button.click()
    app.processEvents()
    if reload_button.text() != "重载完成":
        raise RuntimeError("recipe reload did not provide visible feedback")
    window.close()

    with TemporaryDirectory() as directory:
        project_root = Path(directory)
        source = project_root / "sample.pt"
        source.write_bytes(b"layout-check-weight")
        profile = import_model_bundle(
            source,
            project_root,
            name="现场零件模型",
            class_names=["part_a", "part_b", "defect"],
            conf_threshold=0.4,
        )
        dialog = ModelManagerDialog(project_root, profile.weights_value)
        dialog.show()
        app.processEvents()
        if dialog.table.rowCount() != 1:
            raise RuntimeError("model manager did not list the model profile")
        if not dialog.grab().save(str(output / "model_manager.png")):
            raise RuntimeError("failed to save model manager screenshot")
        dialog.close()
    print(f"GUI_LAYOUT_OK {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
