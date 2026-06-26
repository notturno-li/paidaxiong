from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from app.config import load_config
from app.core.robot_client import DobotClient


def format_number(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if "." in text else f"{text}.0"


def render_sequence(values: Iterable[float], indent: int) -> list[str]:
    prefix = " " * indent
    return [f"{prefix}- {format_number(value)}" for value in values]


def replace_sequence_block(text: str, key: str, values: list[float]) -> str:
    lines = text.splitlines()
    key_pattern = re.compile(rf"^(\s*){re.escape(key)}:\s*$")

    for index, line in enumerate(lines):
        match = key_pattern.match(line)
        if not match:
            continue

        key_indent = len(match.group(1))
        value_indent = key_indent + 2
        end = index + 1
        while end < len(lines):
            current = lines[end]
            if not current.strip():
                end += 1
                continue
            current_indent = len(current) - len(current.lstrip(" "))
            if current_indent > key_indent:
                end += 1
                continue
            break

        new_lines = lines[: index + 1] + render_sequence(values, value_indent) + lines[end:]
        return "\n".join(new_lines) + ("\n" if text.endswith("\n") else "")

    raise KeyError(f"未找到配置项: {key}")


def save_pose_to_yaml(path: Path, robot_section: dict[str, object], updates: dict[str, list[float]]) -> None:
    original = path.read_text(encoding="utf-8")
    updated = original

    if "home_pose" in updates:
        updated = replace_sequence_block(updated, "home_pose", updates["home_pose"])
    if "fixed_test_pose" in updates:
        updated = replace_sequence_block(updated, "fixed_test_pose", updates["fixed_test_pose"])

    bins = robot_section.get("bins", {})
    if isinstance(bins, dict):
        for label, pose in updates.items():
            if label in {"home_pose", "fixed_test_pose"}:
                continue
            updated = replace_sequence_block(updated, label, pose)

    path.write_text(updated, encoding="utf-8")


def read_robot_pose(client: DobotClient) -> list[float]:
    pose = client.get_pose()
    if len(pose) < 6:
        raise RuntimeError(f"GetPose 返回值异常: {pose}")
    return [float(value) for value in pose[:6]]


def main() -> int:
    parser = argparse.ArgumentParser(description="交互式标定 competition.yaml 里的抓取位和盒子位")
    parser.add_argument("--config", default=str(ROOT / "configs" / "competition.yaml"), help="配置文件路径")
    parser.add_argument("--ip", default=None, help="机器人 IP，覆盖配置")
    parser.add_argument("--port", type=int, default=None, help="机器人 TCP 端口，覆盖配置")
    parser.add_argument("--skip-home", action="store_true", help="跳过 home_pose")
    parser.add_argument("--skip-fixed", action="store_true", help="跳过 fixed_test_pose")
    parser.add_argument("--labels", nargs="*", default=None, help="只标定指定的盒子标签，默认按现有 bins 顺序")
    parser.add_argument("--dry-run", action="store_true", help="只打印不写回文件")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    config["mode"] = "hardware"
    if args.ip:
        config["robot"]["ip"] = args.ip
    if args.port:
        config["robot"]["dashboard_port"] = args.port

    robot_cfg = config["robot"]
    bin_cfg = robot_cfg.get("bins", {})
    if not isinstance(bin_cfg, dict) or not bin_cfg:
        raise RuntimeError("competition.yaml 里没有 bins，无法标定置物盒")

    labels = list(args.labels) if args.labels else list(bin_cfg.keys())
    labels = [label for label in labels if label in bin_cfg]
    if not labels:
        raise RuntimeError("没有可标定的盒子标签")

    tasks: list[tuple[str, str]] = []
    if not args.skip_home:
        tasks.append(("home_pose", "回零/安全停靠位"))
    if not args.skip_fixed:
        tasks.append(("fixed_test_pose", "定点测试抓取位"))
    tasks.extend((label, f"{label} 盒子位") for label in labels)

    print(f"目标配置: {config_path}")
    print(f"机器人: {robot_cfg['ip']}:{robot_cfg['dashboard_port']}")
    print("操作：手动把机械臂移到目标位后按回车，脚本会立即读取 GetPose() 并写回 competition.yaml。")

    client = DobotClient(config, simulation=False)
    try:
        client.connect()
        try:
            client.request_control()
        except Exception:
            pass
        try:
            client.enable()
        except Exception:
            pass

        updates: dict[str, list[float]] = {}
        for key, title in tasks:
            current = robot_cfg.get("bins", {}).get(key) if key not in {"home_pose", "fixed_test_pose"} else robot_cfg.get(key)
            print()
            print(f"[{title}]")
            if current:
                print(f"当前值: {current}")
            input("把机械臂移动到位后按回车读取...")
            pose = read_robot_pose(client)
            print(f"读取到: {[round(v, 3) for v in pose]}")
            updates[key] = pose

        print()
        print("本次结果预览：")
        print(yaml.safe_dump({"robot": updates}, allow_unicode=True, sort_keys=False, default_flow_style=False))

        if args.dry_run:
            print("dry-run: 未写回文件")
            return 0

        save_pose_to_yaml(config_path, robot_cfg, updates)
        print(f"已写回: {config_path}")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
