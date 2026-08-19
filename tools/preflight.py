from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.preflight import run_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description="国赛程序一键赛前自检（不发送机器人运动命令）")
    parser.add_argument("--config", default=str(ROOT / "configs" / "field.yaml"))
    parser.add_argument(
        "--hardware",
        action="store_true",
        help="额外建立机械臂 Dashboard 和 DVS 连接；不使能机器人、不发送业务或运动命令",
    )
    args = parser.parse_args()
    report = run_preflight(args.config, hardware=args.hardware, root=ROOT)
    icons = {"PASS": "[通过]", "WARN": "[待确认]", "FAIL": "[阻止]"}
    for check in report.checks:
        print(f"{icons[check.status]} {check.name}: {check.detail}")
    print(
        f"\n汇总：通过 {report.passed}，待确认 {report.warnings}，阻止 {report.failures}。"
    )
    print("结论：" + ("未发现阻止项" if report.ready else "存在阻止项，禁止启动机械臂动作"))
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
