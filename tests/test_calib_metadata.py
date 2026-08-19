from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from calib_metadata import (  # noqa: E402
    load_metadata,
    validate_device_against_metadata,
    validate_existing_images,
    write_metadata,
)


def sample_metadata(*, width: int = 640, height: int = 480, serial: str = "ABC") -> dict:
    return {
        "schema_version": 1,
        "stream": "color",
        "image_size": [width, height],
        "fps": 30,
        "source": "realsense_device",
        "serial_number": serial,
        "camera_matrix": {"fx": 615.0, "fy": 616.0, "cx": 320.0, "cy": 240.0},
        "dist_coeffs": [0.1, -0.2, 0.0, 0.0, 0.03],
    }


class CalibrationMetadataTests(unittest.TestCase):
    def test_loads_matching_capture_intrinsics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera_intrinsics.yaml"
            write_metadata(path, sample_metadata())

            matrix, distortion, metadata = load_metadata(path, 640, 480)

            self.assertEqual(metadata["serial_number"], "ABC")
            self.assertEqual(matrix.shape, (3, 3))
            self.assertEqual(distortion.shape, (5, 1))
            self.assertAlmostEqual(matrix[0, 0], 615.0)

    def test_rejects_intrinsics_from_another_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera_intrinsics.yaml"
            write_metadata(path, sample_metadata(width=1280, height=720))

            with self.assertRaisesRegex(RuntimeError, "分辨率"):
                load_metadata(path, 640, 480)

    def test_rejects_existing_image_from_another_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "1.jpg"
            cv2.imwrite(str(image_path), np.zeros((720, 1280, 3), dtype=np.uint8))

            with self.assertRaisesRegex(RuntimeError, "1280x720"):
                validate_existing_images(directory, 640, 480)

    def test_rejects_another_camera_serial_number(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "序列号"):
            validate_device_against_metadata(
                sample_metadata(serial="ABC"), sample_metadata(serial="XYZ")
            )


if __name__ == "__main__":
    unittest.main()
