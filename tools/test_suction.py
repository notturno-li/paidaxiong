from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.core.robot_client import DobotClient


def main() -> int:
    parser = argparse.ArgumentParser(description="测试吸盘IO：末端 ToolDOInstant")
    parser.add_argument("--io-type", choices=["tool_do"], default=None)
    parser.add_argument("--index", type=int, default=None, help="覆盖 suction_tool_do")
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()

    config = load_config(ROOT / "configs" / "competition.yaml")
    config["mode"] = "hardware"
    if args.io_type:
        config["robot"]["suction_io_type"] = args.io_type
    if args.index is not None:
        config["robot"]["suction_tool_do"] = args.index

    client = DobotClient(config, simulation=False)
    client.connect()
    print("Suction ON:", client.suction(True).raw)
    time.sleep(args.seconds)
    print("Suction OFF:", client.suction(False).raw)
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
