from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.core.robot_client import DobotClient


def main() -> int:
    parser = argparse.ArgumentParser(description="末端吸盘 ToolDOInstant 专用测试")
    parser.add_argument("--ip", default=None, help="机器人IP，默认读取 configs/competition.yaml")
    parser.add_argument("--index", type=int, default=None, help="末端 DO 编号，默认读取 suction_tool_do")
    parser.add_argument("--hold", type=float, default=2.0, help="每次吸附/释放保持秒数")
    parser.add_argument("--repeat", type=int, default=1, help="循环次数")
    parser.add_argument("--skip-enable", action="store_true", help="跳过 EnableRobot()")
    args = parser.parse_args()

    config = load_config(ROOT / "configs" / "competition.yaml")
    config["mode"] = "hardware"
    config["robot"]["suction_io_type"] = "tool_do"
    if args.ip:
        config["robot"]["ip"] = args.ip
    if args.index is not None:
        config["robot"]["suction_tool_do"] = args.index

    client = DobotClient(config, simulation=False)
    print(f"连接机器人 {config['robot']['ip']}:{config['robot']['dashboard_port']} ...")
    client.connect()
    print("连接成功")
    if not args.skip_enable:
        print("EnableRobot:", client.enable().raw)

    index = int(config["robot"].get("suction_tool_do", 1))
    print(f"使用末端 ToolDOInstant({index}, level)")
    for i in range(max(1, args.repeat)):
        print(f"[{i+1}/{max(1, args.repeat)}] ON :", client.suction(True).raw)
        time.sleep(args.hold)
        print(f"[{i+1}/{max(1, args.repeat)}] OFF:", client.suction(False).raw)
        time.sleep(args.hold)

    client.close()
    print("测试完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
