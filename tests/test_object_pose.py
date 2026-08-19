from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from app.core.object_pose import ObjectPoseEstimator
from app.core.robot_client import DobotClient
from app.core.types import CameraIntrinsics, Detection
from scripts.teach_competition_poses import save_pose_to_yaml


class ObjectPoseEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "camera": {"min_valid_depth_mm": 100, "max_valid_depth_mm": 1500},
            "height": {"table_depth_mm": 1000.0},
            "orientation": {
                "enabled": True,
                "bbox_expand_ratio": 0.1,
                "min_depth_points": 20,
                "min_object_points": 20,
                "min_height_above_plane_mm": 3.0,
                "max_height_above_plane_mm": 100.0,
                "min_anisotropy": 0.2,
                "classes": {
                    "rect": {"mode": "pca", "period_deg": 180},
                    "square": {"mode": "symmetric", "period_deg": 90},
                },
            },
        }
        self.intrinsics = CameraIntrinsics(500.0, 500.0, 100.0, 100.0)
        self.cam_to_base = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 1000.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def test_rotated_depth_rectangle_yields_base_frame_angle(self):
        depth = np.full((200, 200), 1000.0, dtype=np.float32)
        box = cv2.boxPoints(((100, 100), (80, 30), 30.0)).astype(np.int32)
        cv2.fillConvexPoly(depth, box, 950.0)
        detection = Detection("rect", 0.99, (50, 60, 150, 140), (100, 100))
        result = ObjectPoseEstimator(self.config).estimate(
            depth, detection, self.intrinsics, self.cam_to_base, (0.0, 0.0, 1.0, 0.0)
        )
        self.assertTrue(result.orientation.valid, result.orientation.reason)
        self.assertAlmostEqual(result.orientation.angle_deg, -30.0, delta=3.0)
        self.assertGreater(result.orientation.confidence, 0.5)
        self.assertLess(abs(result.grasp_pixel[0] - 100), 5)
        self.assertLess(abs(result.grasp_pixel[1] - 100), 5)

    def test_symmetric_class_does_not_invent_an_orientation(self):
        depth = np.full((200, 200), 1000.0, dtype=np.float32)
        detection = Detection("square", 0.99, (70, 70, 130, 130), (100, 100))
        result = ObjectPoseEstimator(self.config).estimate(
            depth, detection, self.intrinsics, self.cam_to_base, (0.0, 0.0, 1.0, 0.0)
        )
        self.assertTrue(result.orientation.valid)
        self.assertEqual(result.orientation.angle_deg, 0.0)
        self.assertEqual(result.orientation.period_deg, 90.0)

    def test_round_pca_contour_is_rejected_as_ambiguous(self):
        depth = np.full((200, 200), 1000.0, dtype=np.float32)
        cv2.circle(depth, (100, 100), 35, 950.0, -1)
        detection = Detection("rect", 0.99, (60, 60, 140, 140), (100, 100))
        result = ObjectPoseEstimator(self.config).estimate(
            depth, detection, self.intrinsics, self.cam_to_base, (0.0, 0.0, 1.0, 0.0)
        )
        self.assertFalse(result.orientation.valid)
        self.assertIn("主方向", result.orientation.reason)


class AssemblyConfigurationTests(unittest.TestCase):
    def test_structured_slot_overrides_legacy_bin_and_clearance(self):
        config = {
            "mode": "simulation",
            "robot": {
                "safe_z_mm": 180.0,
                "grasp_clearance_mm": 18.0,
                "suction_settle_s": 0.0,
                "suction_release_settle_s": 0.0,
                "home_pose": [0, 0, 200, 180, 0, 0],
                "bins": {"rect": [200, 100, 110, 180, 0, 0]},
            },
            "assembly": {
                "slots": {
                    "rect": {
                        "tcp_pose": [210, 120, 115, 180, 0, 15],
                        "approach_clearance_mm": 40,
                    }
                }
            },
        }
        robot = DobotClient(config, simulation=True)
        sequence = robot.build_grasp_sequence([100, 50, 100, 180, 0, 20], "rect")
        self.assertEqual(robot.resolve_bin_pose("rect"), [210.0, 120.0, 115.0, 180.0, 0.0, 15.0])
        self.assertEqual(sequence[5][1][2], 180.0)

    def test_teaching_updates_inline_photo_and_bin_poses(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "today.yaml"
            path.write_text(
                "robot:\n"
                "  photo_pose: [1, 2, 3, 180, 0, 0]\n"
                "  bins:\n"
                "    rect: [4, 5, 6, 180, 0, 0]\n",
                encoding="utf-8",
            )
            save_pose_to_yaml(
                path,
                {"bins": {"rect": [4, 5, 6, 180, 0, 0]}},
                {
                    "photo_pose": [10, 20, 300, 180, 0, 5],
                    "rect": [210, 120, 115, 180, 0, 15],
                },
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn("photo_pose: [10.0, 20.0, 300.0, 180.0, 0.0, 5.0]", text)
            self.assertIn("rect: [210.0, 120.0, 115.0, 180.0, 0.0, 15.0]", text)

    def test_tray_frame_composes_all_slot_positions(self):
        config = {
            "mode": "simulation",
            "robot": {
                "home_pose": [0, 0, 200, 180, 0, 0],
                "bins": {},
            },
            "assembly": {
                "target_mode": "tray_frame",
                "frame_pose_base": [100, 200, 20, 0, 0, 90],
                "slots": {"rect": {"pose_tray": [10, 0, 95, 180, 0, 0]}},
            },
        }
        pose = DobotClient(config, simulation=True).resolve_bin_pose("rect")
        self.assertAlmostEqual(pose[0], 100.0, places=5)
        self.assertAlmostEqual(pose[1], 210.0, places=5)
        self.assertAlmostEqual(pose[2], 115.0, places=5)


if __name__ == "__main__":
    unittest.main()
