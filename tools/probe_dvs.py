from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.core.vision_studio_client import DobotVisionStudioClient


def main() -> int:
    parser = argparse.ArgumentParser(description="DobotVisionStudio 串口/TCP 现场通信探针")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "field.yaml"),
        help="配置文件，默认 configs/field.yaml",
    )
    parser.add_argument("--discover", action="store_true", help="列出串口和当前 TCP 配置")
    parser.add_argument("--connect", action="store_true", help="只测试建立连接")
    parser.add_argument("--exchange", action="store_true", help="发送/接收一次或多次报文")
    parser.add_argument("--message", default=None, help="覆盖配置中的触发/查询字符串")
    parser.add_argument("--count", type=int, default=1, help="交换次数，默认 1")
    parser.add_argument(
        "--set-global",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="发送 DVS SetGlobalValue，可重复指定",
    )
    parser.add_argument("--simulation", action="store_true", help="不访问硬件，使用 simulation_result")
    args = parser.parse_args()
    if args.count < 1 or args.count > 100:
        parser.error("--count 必须在 1~100 之间")
    if not any((args.discover, args.connect, args.exchange, args.set_global)):
        args.discover = True

    config = load_config(args.config)
    if args.simulation:
        config["mode"] = "simulation"
    client = DobotVisionStudioClient(config, simulation=args.simulation)
    try:
        if args.discover:
            print(json.dumps(client.discover(), ensure_ascii=False, indent=2))
        if args.connect or args.set_global or args.exchange:
            client.connect()
            print(f"CONNECTED mode={client.mode}")
        for assignment in args.set_global:
            if "=" not in assignment:
                raise RuntimeError(f"--set-global 格式应为 NAME=VALUE: {assignment}")
            name, value = assignment.split("=", 1)
            client.set_global_value(name.strip(), value)
            print(f"SET_GLOBAL {name.strip()}={value}")
        if args.exchange:
            for index in range(args.count):
                reply = client.exchange(args.message)
                print(f"EXCHANGE {index + 1}/{args.count} transport={reply.transport}")
                print("RAW " + reply.raw)
                print("PARSED " + json.dumps(reply.data, ensure_ascii=False, default=str))
    except Exception as exc:
        print(f"DVS_PROBE_ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
