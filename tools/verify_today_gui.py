from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt5.QtWidgets import QApplication, QPushButton

from app.today_main import TodayMainWindow


def main() -> int:
    output = ROOT / "runs" / "ui_checks" / "today_gui_1280x760.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    window = TodayMainWindow()
    window.resize(1280, 760)
    window.show()
    app.processEvents()

    buttons = {button.text() for button in window.findChildren(QPushButton)}
    missing = {"启动相机", "单次运行", "自动运行", "急停 STOP"} - buttons
    if missing:
        raise RuntimeError("今日 GUI 缺少按钮: " + ", ".join(sorted(missing)))
    if window.result_table.columnCount() != 7:
        raise RuntimeError("3D 结果表应为 7 列（含序号）")
    if len(window.monitor_fields) != 7:
        raise RuntimeError("监控区应包含 7 个题面字段")
    visible_buttons = [button for button in window.findChildren(QPushButton) if button.isVisible()]
    for button in visible_buttons:
        if button.width() < 60 or button.height() < 24:
            raise RuntimeError(f"按钮被裁切: {button.text()} {button.geometry()}")
    for index, left in enumerate(visible_buttons):
        for right in visible_buttons[index + 1 :]:
            if left.parent() is right.parent() and left.geometry().intersects(right.geometry()):
                raise RuntimeError(f"按钮重叠: {left.text()} / {right.text()}")
    for widget in (window.rgb_label, window.depth_label, window.yolo_label, window.result_table):
        if not widget.isVisible() or widget.width() < 100 or widget.height() < 60:
            raise RuntimeError(f"控件不可见或尺寸过小: {widget.objectName() or type(widget).__name__}")
    if not window.grab().save(str(output)):
        raise RuntimeError("无法保存今日 GUI 截图")
    by_text = {button.text(): button for button in visible_buttons}
    by_text["启动相机"].click()
    app.processEvents()
    by_text["单次运行"].click()
    deadline = time.monotonic() + 5.0
    while window.worker_thread is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    if window.worker_thread is not None:
        raise RuntimeError("单次运行在 5 秒内未结束")
    if window.result_table.rowCount() != 1:
        raise RuntimeError("单次运行未写入 3D 结果表")
    if len(window.workflow.detector.detect(window.workflow.frame.color)) != 4:
        raise RuntimeError("单次运行后模拟物料未正确移除")
    refresh_deadline = time.monotonic() + 0.15
    while time.monotonic() < refresh_deadline:
        app.processEvents()
        time.sleep(0.01)
    active_output = output.with_name("today_gui_active_1280x760.png")
    if not window.grab().save(str(active_output)):
        raise RuntimeError("无法保存运行态 GUI 截图")
    window.on_release_suction()
    window.close()
    print(f"TODAY_GUI_OK {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
