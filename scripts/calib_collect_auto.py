"""
手眼标定自动采集脚本（自动读取机器人位姿）
=========================================
操作流程：
1. 确保机器人已通过 TCP 连接（29999 端口），并已使能
2. 手动把机械臂挪到一个姿态，让标定板清晰出现在画面里
3. 看到绿色网格说明识别到棋盘格，按 S 自动保存图片+自动读取机器人 XYZ Rx Ry Rz
4. 换姿态重复，采集 15~20 组（姿态差异要大，尽量覆盖不同倾斜方向）
5. 按 Q 退出，然后运行 calib_solve.py 求解手眼矩阵

保存位置：
  图片 → runs/calib_data/images/{n}.jpg
  位姿 → runs/calib_data/poses/{n}.txt  (格式: X Y Z Rx Ry Rz，单位 mm/度)
"""
from __future__ import annotations

import os
import re
import socket
import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

# ── 配置区（按实际修改） ────────────────────────────────────────────────────
ROBOT_IP            = "192.168.5.1"
ROBOT_PORT          = 29999
ROBOT_TIMEOUT_S     = 3.0

CHECKERBOARD        = (8, 11)   # 内角点 (宽, 高)，10x12方格的标定板
CAMERA_WIDTH        = 1280
CAMERA_HEIGHT       = 720
CAMERA_FPS          = 30

SAVE_DIR_IMG        = "runs/calib_data/images"
SAVE_DIR_POSE       = "runs/calib_data/poses"
# ────────────────────────────────────────────────────────────────────────────


def tcp_send(sock: socket.socket, cmd: str) -> str:
    sock.sendall((cmd + "\n").encode("utf-8"))
    return sock.recv(4096).decode("utf-8", errors="ignore").strip()


def get_robot_pose(sock: socket.socket) -> list[float] | None:
    """通过 GetPose() 指令读取末端笛卡尔位姿，返回 [X, Y, Z, Rx, Ry, Rz]。"""
    raw = tcp_send(sock, "GetPose()")
    # 返回格式：ErrorID,{X,Y,Z,Rx,Ry,Rz},GetPose();
    match = re.match(r"\s*0\s*,\s*\{([^}]+)\}", raw)
    if not match:
        print(f"[GetPose 失败] 返回: {raw}")
        return None
    try:
        values = [float(v.strip()) for v in match.group(1).split(",")]
        if len(values) >= 6:
            return values[:6]
    except ValueError:
        pass
    return None


def main() -> int:
    # ── 创建保存目录 ────────────────────────────────────────────────────────
    Path(SAVE_DIR_IMG).mkdir(parents=True, exist_ok=True)
    Path(SAVE_DIR_POSE).mkdir(parents=True, exist_ok=True)

    # ── 连接机器人 ──────────────────────────────────────────────────────────
    print(f"正在连接机器人 {ROBOT_IP}:{ROBOT_PORT} ...")
    try:
        sock = socket.create_connection((ROBOT_IP, ROBOT_PORT), timeout=ROBOT_TIMEOUT_S)
        print("机器人连接成功")
    except OSError as e:
        print(f"连接失败：{e}")
        print("将以【无机器人模式】运行，按 S 时只保存图片，不记录位姿。")
        sock = None

    # ── 启动相机 ────────────────────────────────────────────────────────────
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)
    pipeline.start(config)

    img_count = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print("\n[操作说明]")
    print("  手动挪动机械臂 → 看到绿色网格 → 按 S 自动采集")
    print("  按 Q 退出\n")

    try:
        while True:
            frames      = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img         = np.asanyarray(color_frame.get_data())
            gray        = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            display_img = img.copy()

            ret, corners = cv2.findChessboardCorners(
                gray, CHECKERBOARD,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE,
            )

            if ret:
                corners_sp = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(display_img, CHECKERBOARD, corners_sp, ret)
                cv2.putText(display_img, f"[OK] Press S to save #{img_count}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                cv2.putText(display_img, "Searching board...", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

            cv2.putText(display_img, f"Saved: {img_count - 1} poses", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Hand-Eye Calibration - S:save Q:quit", display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("退出采集。")
                break

            if key == ord('s') and ret:
                # 1) 先读机器人位姿（减少相机等待时间）
                pose_str = None
                if sock:
                    pose = get_robot_pose(sock)
                    if pose is not None:
                        pose_str = " ".join(f"{v:.6f}" for v in pose)
                        print(f"[位姿] {pose_str}")
                    else:
                        print("⚠️  GetPose 读取失败，本组跳过（请检查机器人连接）")
                        continue

                # 2) 保存干净原图（不带绿线）
                img_path  = os.path.join(SAVE_DIR_IMG,  f"{img_count}.jpg")
                pose_path = os.path.join(SAVE_DIR_POSE, f"{img_count}.txt")
                cv2.imwrite(img_path, img)

                # 3) 保存位姿 txt
                if pose_str:
                    Path(pose_path).write_text(pose_str, encoding="utf-8")
                    print(f"✅ 第 {img_count} 组已保存 → {img_path}  |  {pose_path}")
                else:
                    # 无机器人模式：只存图，位姿留空
                    Path(pose_path).write_text("0.000 0.000 0.000 0.000 0.000 0.000", encoding="utf-8")
                    print(f"✅ 图片已保存（无机器人，位姿为占位零值）→ {img_path}")

                img_count += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        if sock:
            sock.close()
        print(f"\n采集结束，共 {img_count - 1} 组。运行 calib_solve.py 求解手眼矩阵。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
