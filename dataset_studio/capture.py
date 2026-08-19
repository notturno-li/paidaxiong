from __future__ import annotations

import threading
from typing import Any

import cv2
import numpy as np


class CameraCapture:
    def __init__(self, mode: str = "realsense", index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.mode = mode
        self.index = int(index)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._pipeline: Any = None
        self._opencv: Any = None
        self._rs: Any = None
        self._mutex = threading.Lock()

    def capture(self) -> np.ndarray:
        with self._mutex:
            if self.mode == "realsense":
                return self._capture_realsense()
            if self.mode == "opencv":
                return self._capture_opencv()
            raise RuntimeError(f"不支持的相机模式: {self.mode}")

    def preview_jpeg(self) -> bytes:
        frame = self.capture()
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("相机预览编码失败")
        return encoded.tobytes()

    def _capture_realsense(self) -> np.ndarray:
        if self._pipeline is None:
            try:
                import pyrealsense2 as rs
            except Exception as exc:
                raise RuntimeError("未安装 pyrealsense2，不能使用 RealSense 采集") from exc
            self._rs = rs
            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(
                rs.stream.color,
                self.width,
                self.height,
                rs.format.bgr8,
                self.fps,
            )
            self._pipeline.start(config)
            for _index in range(8):
                self._pipeline.wait_for_frames()
        frames = self._pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("RealSense 未返回彩色帧")
        return np.asanyarray(color.get_data()).copy()

    def _capture_opencv(self) -> np.ndarray:
        if self._opencv is None:
            backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
            self._opencv = cv2.VideoCapture(self.index, backend)
            self._opencv.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._opencv.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._opencv.set(cv2.CAP_PROP_FPS, self.fps)
            if not self._opencv.isOpened():
                self._opencv.release()
                self._opencv = None
                raise RuntimeError(f"无法打开 USB 相机 index={self.index}")
            for _index in range(5):
                self._opencv.read()
        ok, frame = self._opencv.read()
        if not ok or frame is None:
            raise RuntimeError("USB 相机读取失败")
        return frame

    def close(self) -> None:
        with self._mutex:
            if self._pipeline is not None:
                self._pipeline.stop()
            if self._opencv is not None:
                self._opencv.release()
            self._pipeline = None
            self._opencv = None
