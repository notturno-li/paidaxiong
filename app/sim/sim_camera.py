from __future__ import annotations

from time import time

import numpy as np

from app.core.types import CameraFrame, CameraIntrinsics


class SimulatedCamera:
    def __init__(self, config: dict):
        self.config = config
        cam_cfg = config["camera"]
        cal_cfg = config["calibration"]["camera_matrix"]
        self.width = int(cam_cfg["width"])
        self.height = int(cam_cfg["height"])
        self.frame_index = 0
        self.intrinsics = CameraIntrinsics(float(cal_cfg["fx"]), float(cal_cfg["fy"]), float(cal_cfg["cx"]), float(cal_cfg["cy"]))
        self.labels = list(config["model"]["class_names"])
        self.removed_bboxes: set[tuple[int, int, int, int]] = set()

    def start(self) -> None:
        self.frame_index = 0

    def read(self) -> CameraFrame:
        import cv2
        self.frame_index += 1
        color = np.full((self.height, self.width, 3), (35, 45, 50), dtype=np.uint8)
        depth = np.full((self.height, self.width), float(self.config["height"]["table_depth_mm"]), dtype=np.float32)
        simulation_objects = self.config.get("simulation", {}).get("objects", [])
        colors = [(80, 80, 220), (70, 190, 70), (220, 120, 40), (170, 70, 170), (40, 170, 210), (120, 120, 50), (70, 210, 210)]
        if isinstance(simulation_objects, list) and simulation_objects:
            for idx, item in enumerate(simulation_objects):
                if not isinstance(item, dict):
                    continue
                box = item.get("bbox", [])
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                x1, y1, x2, y2 = (int(value) for value in box)
                if (x1, y1, x2, y2) in self.removed_bboxes:
                    continue
                x1, x2 = max(0, x1), min(self.width - 1, x2)
                y1, y2 = max(0, y1), min(self.height - 1, y2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                radius_x, radius_y = max(10, (x2 - x1) // 2 - 4), max(10, (y2 - y1) // 2 - 4)
                cv2.rectangle(color, (x1, y1), (x2, y2), (235, 235, 235), -1)
                cv2.ellipse(color, (cx, cy), (radius_x, radius_y), 0, 0, 360, colors[idx % len(colors)], -1)
                cv2.putText(color, str(idx + 1), (x1 + 5, min(self.height - 8, y2 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                mask = np.zeros((self.height, self.width), dtype=np.uint8)
                cv2.ellipse(mask, (cx, cy), (radius_x, radius_y), 0, 0, 360, 255, -1)
                object_height = float(item.get("height_mm", 30.0))
                depth[mask > 0] = float(self.config["height"]["table_depth_mm"]) - object_height
        else:
            centers = [(180, 180), (315, 165), (440, 260), (240, 315)]
            radii = [32, 30, 28, 31]
            heights = [35.0, 48.0, 26.0, 55.0]
            colors = [(0, 0, 220), (0, 220, 220), (40, 40, 230), (160, 60, 160)]
            for idx, (center, radius, height, bgr) in enumerate(zip(centers, radii, heights, colors)):
                cx = center[0] + int(6 * np.sin((self.frame_index + idx * 9) / 20.0))
                cy = center[1]
                cv2.circle(color, (cx, cy), radius + 7, (235, 235, 235), -1)
                cv2.circle(color, (cx, cy), radius, bgr, -1)
                cv2.putText(color, self.labels[idx][0].upper(), (cx - 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                mask = np.zeros((self.height, self.width), dtype=np.uint8)
                cv2.circle(mask, (cx, cy), radius + 5, 255, -1)
                depth[mask > 0] = float(self.config["height"]["table_depth_mm"]) - height
        cv2.putText(color, "SIMULATION MODE", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 210, 255), 2)
        return CameraFrame(color=color, depth_mm=depth, intrinsics=self.intrinsics, timestamp=time())

    def stop(self) -> None:
        pass

    def mark_simulated_removed(self, bbox) -> None:
        self.removed_bboxes.add(tuple(int(value) for value in bbox))
