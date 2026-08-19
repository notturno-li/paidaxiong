from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "runs" / "calib_data"
IMAGE_DIR = DATA_DIR / "images"
POSE_DIR = DATA_DIR / "poses"


def numbered_stems(directory: Path, suffix: str) -> set[str]:
    if not directory.is_dir():
        return set()
    return {path.stem for path in directory.glob(f"*{suffix}") if path.stem.isdigit()}


def data_summary() -> tuple[set[str], set[str]]:
    images = numbered_stems(IMAGE_DIR, ".jpg")
    poses = numbered_stems(POSE_DIR, ".txt")
    print(
        f"当前数据：图像 {len(images)} 张，位姿 {len(poses)} 组，"
        f"可配对 {len(images & poses)} 组"
    )
    return images, poses


def run_script(name: str) -> int:
    command = [sys.executable, "-X", "utf8", str(ROOT / "scripts" / name)]
    return subprocess.run(command, cwd=str(ROOT), check=False).returncode


def dependencies_available(*names: str) -> bool:
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    if not missing:
        return True
    print("缺少依赖：" + ", ".join(missing))
    print("请先运行 install_offline.bat。")
    return False


def collect() -> None:
    print("\n[自动采集]")
    print("标定板必须固定不动，相机固定在机械臂末端。")
    print("手动移动机械臂，停稳且出现绿色角点后按 S；按 Q 退出。")
    print("建议采集 15~20 组，姿态和倾角应有明显变化。")
    print("此功能只发送 GetPose() 读取位姿，不会控制机械臂运动。\n")
    if dependencies_available("cv2", "numpy", "yaml", "pyrealsense2"):
        run_script("calib_collect_auto.py")


def diagnose() -> None:
    images, poses = data_summary()
    if not images & poses:
        print("没有可诊断的成对数据，请先采集。")
        return
    if dependencies_available("cv2", "numpy", "yaml"):
        run_script("calib_diagnose.py")


def solve() -> None:
    images, poses = data_summary()
    paired = images & poses
    missing_poses = images - poses
    if missing_poses:
        print("以下图像缺少同编号机器人位姿，已阻止求解：")
        print(", ".join(sorted(missing_poses, key=int)))
        return
    if len(paired) < 5:
        print(f"只有 {len(paired)} 组数据，至少需要 5 组，建议 15~20 组。")
        return
    if len(paired) < 15:
        answer = input("样本少于 15 组，精度可能不足，仍继续？[y/N] ").strip().lower()
        if answer != "y":
            return
    answer = input("求解成功后将更新 configs/competition.yaml，确认？[y/N] ").strip().lower()
    if answer != "y":
        return
    if dependencies_available("cv2", "numpy", "yaml"):
        run_script("calib_solve.py")


def intrinsic() -> None:
    print("RealSense 原厂镜头通常不需要重做内参；此项只用于可选的外部核验。")
    print("手眼采集会另存设备当前流内参，求解不会读取这里生成的内参文件。")
    if dependencies_available("cv2", "numpy", "yaml", "pyrealsense2"):
        run_script("calib_intrinsic.py")


def archive_data() -> None:
    if not DATA_DIR.exists():
        print("当前没有旧的手眼标定采集数据。")
        return
    data_summary()
    answer = input("将旧数据归档后开始新一轮？数据不会被删除。[y/N] ").strip().lower()
    if answer != "y":
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = ROOT / "runs" / f"calib_archive_{stamp}"
    suffix = 1
    while destination.exists():
        suffix += 1
        destination = ROOT / "runs" / f"calib_archive_{stamp}_{suffix}"
    shutil.move(str(DATA_DIR), str(destination))
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    POSE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"旧数据已归档到：{destination}")


def open_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(str(DATA_DIR))  # type: ignore[attr-defined]
    else:
        print(DATA_DIR)


def main() -> int:
    actions = {
        "1": collect,
        "2": diagnose,
        "3": solve,
        "4": intrinsic,
        "5": archive_data,
        "6": open_data_dir,
    }
    while True:
        print("\n" + "=" * 60)
        print("手眼标定现场工具")
        print("=" * 60)
        print("推荐顺序：5 归档旧数据 > 1 自动采集 > 2 诊断 > 3 求解")
        print("RealSense 原厂内参会随手眼数据自动保存；选项 4 仅用于外部核验。\n")
        print("[1] 自动采集图像和机器人位姿")
        print("[2] 诊断采集数据和欧拉角约定")
        print("[3] 求解手眼矩阵并写入比赛配置")
        print("[4] 相机内参外部核验（可选）")
        print("[5] 归档旧采集数据")
        print("[6] 打开采集数据目录")
        print("[Q] 退出")
        choice = input("\n请选择：").strip().lower()
        if choice == "q":
            return 0
        action = actions.get(choice)
        if action is None:
            print("无效选项，请重新输入。")
            continue
        try:
            action()
        except KeyboardInterrupt:
            print("\n已取消当前操作。")
        except Exception as exc:
            print(f"操作失败：{exc}")


if __name__ == "__main__":
    raise SystemExit(main())
