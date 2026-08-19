from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees

import numpy as np

from .types import CameraIntrinsics, Detection, OrientationEstimate


@dataclass
class ObjectPoseEstimate:
    orientation: OrientationEstimate
    grasp_pixel: tuple[int, int]
    base_xyz_mm: tuple[float, float, float] | None = None
    point_count: int = 0


class ObjectPoseEstimator:
    """Estimate a tabletop object's planar pose from an RGB-D detection.

    The detector supplies only a coarse box. Depth points are deprojected and
    transformed into the robot base frame before segmentation and PCA, so the
    reported yaw is already expressed in the frame used by the robot TCP.
    """

    def __init__(self, config: dict):
        self.config = config

    def estimate(
        self,
        depth_mm: np.ndarray,
        detection: Detection,
        intrinsics: CameraIntrinsics,
        cam_to_base: np.ndarray,
        plane_base: tuple[float, float, float, float] | None,
    ) -> ObjectPoseEstimate:
        settings = self.config.get("orientation", {})
        if not bool(settings.get("enabled", True)):
            base_xyz = self._pixel_base_xyz(depth_mm, detection.center, intrinsics, cam_to_base)
            return ObjectPoseEstimate(
                OrientationEstimate(0.0, True, 1.0, 360.0, "角度估计已禁用"),
                detection.center,
                base_xyz,
                0,
            )
        class_settings = self._class_settings(detection.label)
        mode = str(class_settings.get("mode", "pca")).lower()
        period = float(class_settings.get("period_deg", class_settings.get("symmetry_period_deg", 180.0)))
        if period <= 0.0 or period > 360.0:
            period = 180.0
        if mode in {"ignore", "symmetric"}:
            center = detection.center
            base_xyz = self._pixel_base_xyz(depth_mm, center, intrinsics, cam_to_base)
            return ObjectPoseEstimate(
                OrientationEstimate(0.0, True, 1.0, period, "该类别旋转对称，忽略平面角度"),
                center,
                base_xyz,
                0,
            )
        if depth_mm.ndim != 2:
            return self._invalid(detection, "深度图不是单通道")

        bounds = self._expanded_bounds(depth_mm.shape[:2], detection.bbox, settings)
        x1, y1, x2, y2 = bounds
        roi_depth = depth_mm[y1:y2, x1:x2].astype(np.float64, copy=False)
        valid = np.isfinite(roi_depth) & (roi_depth > 0.0)
        camera_cfg = self.config.get("camera", {})
        min_depth = float(camera_cfg.get("min_valid_depth_mm", 0.0) or 0.0)
        max_depth = float(camera_cfg.get("max_valid_depth_mm", 0.0) or 0.0)
        if min_depth > 0:
            valid &= roi_depth >= min_depth
        if max_depth > 0:
            valid &= roi_depth <= max_depth
        if int(valid.sum()) < int(settings.get("min_depth_points", 30)):
            return self._invalid(detection, "有效深度点不足")

        ys, xs = np.nonzero(valid)
        z = roi_depth[ys, xs]
        u = xs.astype(np.float64) + x1
        v = ys.astype(np.float64) + y1
        points_cam = np.column_stack(
            (
                (u - intrinsics.cx) * z / intrinsics.fx,
                (v - intrinsics.cy) * z / intrinsics.fy,
                z,
            )
        )
        points_base = self._transform_points(points_cam, cam_to_base)
        object_mask = self._object_mask(
            points_base,
            z,
            plane_base,
            mode,
            settings,
        )
        object_mask = self._largest_relevant_component(
            object_mask,
            ys,
            xs,
            roi_depth.shape,
            (detection.center[0] - x1, detection.center[1] - y1),
            settings,
        )
        if int(object_mask.sum()) < int(settings.get("min_object_points", 25)):
            return self._invalid(detection, "目标深度轮廓点不足")

        object_points = points_base[object_mask]
        object_pixels = np.column_stack((u[object_mask], v[object_mask]))
        grasp_pixel = self._safe_grasp_pixel(
            object_pixels,
            (x1, y1, x2, y2),
            settings,
        )
        orientation = self._orientation_from_points(
            object_points,
            mode,
            period,
            class_settings,
            settings,
        )
        base_xyz = self._pixel_base_xyz(depth_mm, grasp_pixel, intrinsics, cam_to_base)
        return ObjectPoseEstimate(
            orientation=orientation,
            grasp_pixel=grasp_pixel,
            base_xyz_mm=base_xyz,
            point_count=int(object_points.shape[0]),
        )

    def _class_settings(self, label: str) -> dict:
        settings = self.config.get("orientation", {})
        classes = settings.get("classes", {})
        item = classes.get(label, {}) if isinstance(classes, dict) else {}
        if not isinstance(item, dict):
            item = {}
        if not item and label in settings.get("symmetric_labels", []):
            return {"mode": "symmetric", "period_deg": 360.0}
        if not item and "圆柱" in label:
            return {"mode": "ignore", "period_deg": 360.0}
        return item

    @staticmethod
    def _expanded_bounds(shape, bbox, settings):
        h, w = shape
        x1, y1, x2, y2 = (int(value) for value in bbox)
        expand = max(0.0, float(settings.get("bbox_expand_ratio", 0.12)))
        dx = int(round((x2 - x1) * expand))
        dy = int(round((y2 - y1) * expand))
        return max(0, x1 - dx), max(0, y1 - dy), min(w, x2 + dx), min(h, y2 + dy)

    def _object_mask(self, points_base, camera_depth, plane_base, mode, settings):
        if plane_base is not None:
            normal = np.asarray(plane_base[:3], dtype=float)
            signed_height = points_base @ normal + float(plane_base[3])
            min_height = float(settings.get("min_height_above_plane_mm", 3.0))
            max_height = float(settings.get("max_height_above_plane_mm", 600.0))
            return (signed_height >= min_height) & (signed_height <= max_height)

        # Without a fitted plane, use the configured camera-depth table value as
        # a conservative fallback. This is less accurate for a tilted camera.
        table_depth = float(self.config.get("height", {}).get("table_depth_mm", 0.0))
        delta = float(settings.get("fallback_depth_delta_mm", 3.0))
        heights = table_depth - np.asarray(camera_depth, dtype=float)
        return heights >= delta

    def _orientation_from_points(self, points, mode, period, class_settings, settings):
        xy = np.asarray(points[:, :2], dtype=float)
        centered = xy - xy.mean(axis=0)
        covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)
        minor = max(float(eigenvalues[order[0]]), 1e-9)
        major = max(float(eigenvalues[order[1]]), 1e-9)
        anisotropy = (major - minor) / major
        min_anisotropy = float(settings.get("min_anisotropy", 0.20))
        if anisotropy < min_anisotropy:
            return OrientationEstimate(
                valid=False,
                confidence=max(0.0, anisotropy / max(min_anisotropy, 1e-6)),
                period_deg=period,
                reason="轮廓没有足够明显的主方向",
            )
        vector = eigenvectors[:, order[1]]
        if mode in {"directed", "directed_contour"}:
            projections = centered @ vector
            skew = float(np.mean((projections - projections.mean()) ** 3))
            if abs(skew) < float(settings.get("min_direction_skew", 1.0)):
                return OrientationEstimate(
                    valid=False,
                    confidence=anisotropy,
                    period_deg=period,
                    reason="轮廓无法稳定区分头尾方向",
                )
            if skew < 0.0:
                vector = -vector
            if float(class_settings.get("directed_sign", 1.0)) < 0.0:
                vector = -vector
        angle = self._canonical_angle(degrees(atan2(float(vector[1]), float(vector[0]))), period)
        return OrientationEstimate(angle, True, min(1.0, anisotropy), period, "")

    @staticmethod
    def _largest_relevant_component(object_mask, ys, xs, shape, center, settings):
        if not object_mask.any():
            return object_mask
        try:
            import cv2

            image = np.zeros(shape, dtype=np.uint8)
            image[ys[object_mask], xs[object_mask]] = 255
            kernel_size = max(1, int(settings.get("morphology_kernel_px", 3)))
            if kernel_size > 1:
                kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
                image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(image, 8)
            if count <= 1:
                return object_mask
            cx, cy = int(center[0]), int(center[1])
            cx = max(0, min(shape[1] - 1, cx))
            cy = max(0, min(shape[0] - 1, cy))
            selected = int(labels[cy, cx])
            if selected == 0:
                selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            return labels[ys, xs] == selected
        except Exception:
            return object_mask

    @staticmethod
    def _canonical_angle(angle, period):
        if period >= 360.0:
            return ((angle + 180.0) % 360.0) - 180.0
        return ((angle + period / 2.0) % period) - period / 2.0

    def _safe_grasp_pixel(self, pixels, bounds, settings):
        if pixels.size == 0:
            return ((bounds[0] + bounds[2]) // 2, (bounds[1] + bounds[3]) // 2)
        try:
            import cv2

            x1, y1, x2, y2 = bounds
            mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            local = np.rint(pixels).astype(int)
            local[:, 0] -= x1
            local[:, 1] -= y1
            local[:, 0] = np.clip(local[:, 0], 0, mask.shape[1] - 1)
            local[:, 1] = np.clip(local[:, 1], 0, mask.shape[0] - 1)
            mask[local[:, 1], local[:, 0]] = 255
            kernel_size = max(1, int(settings.get("grasp_erode_kernel_px", 3)))
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            eroded = cv2.erode(mask, kernel)
            source = eroded if int(eroded.any()) else mask
            distances = cv2.distanceTransform(source, cv2.DIST_L2, 3)
            maximum = float(distances.max())
            candidates = np.argwhere(distances >= maximum * 0.98)
            centroid = np.array([pixels[:, 1].mean() - y1, pixels[:, 0].mean() - x1])
            yy, xx = candidates[int(np.argmin(np.sum((candidates - centroid) ** 2, axis=1)))]
            return int(xx + x1), int(yy + y1)
        except Exception:
            center = pixels.mean(axis=0)
            return int(round(center[0])), int(round(center[1]))

    @staticmethod
    def _transform_points(points_cam, cam_to_base):
        homo = np.column_stack((points_cam, np.ones(points_cam.shape[0])))
        return (np.asarray(cam_to_base, dtype=float) @ homo.T).T[:, :3]

    def _pixel_base_xyz(self, depth_mm, pixel, intrinsics, cam_to_base):
        u, v = (int(pixel[0]), int(pixel[1]))
        h, w = depth_mm.shape[:2]
        x1, x2 = max(0, u - 2), min(w, u + 3)
        y1, y2 = max(0, v - 2), min(h, v + 3)
        values = depth_mm[y1:y2, x1:x2]
        valid = values[np.isfinite(values) & (values > 0)]
        if valid.size == 0:
            return None
        z = float(np.median(valid))
        point_cam = np.array([[(u - intrinsics.cx) * z / intrinsics.fx, (v - intrinsics.cy) * z / intrinsics.fy, z]])
        point_base = self._transform_points(point_cam, cam_to_base)[0]
        return tuple(float(value) for value in point_base)

    @staticmethod
    def _invalid(detection, reason):
        return ObjectPoseEstimate(
            OrientationEstimate(valid=False, reason=reason),
            detection.center,
            None,
            0,
        )
