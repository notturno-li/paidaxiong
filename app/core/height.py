from __future__ import annotations

import numpy as np

from .types import CameraIntrinsics, Detection


class HeightEstimator:
    def __init__(self, config: dict):
        self.config = config
        self.table_depth_mm = float(config["height"]["table_depth_mm"])
        self.plane_base: tuple[float, float, float, float] | None = None

    def calibrate_table(
        self,
        depth_mm: np.ndarray,
        intrinsics: CameraIntrinsics | None = None,
        cam_to_base: np.ndarray | None = None,
    ) -> float:
        valid = depth_mm[np.isfinite(depth_mm) & (depth_mm > 0)]
        if valid.size:
            self.table_depth_mm = float(np.median(valid))
        if intrinsics is not None and cam_to_base is not None:
            plane = self._fit_plane_base(depth_mm, intrinsics, cam_to_base)
            if plane is not None:
                self.plane_base = plane
        return self.table_depth_mm

    def has_plane(self) -> bool:
        return self.plane_base is not None

    def _deproject(self, depth_mm: np.ndarray, intrinsics: CameraIntrinsics):
        h, w = depth_mm.shape[:2]
        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        z = depth_mm.astype(np.float64)
        mask = np.isfinite(z) & (z > 0)
        cam = self.config.get("camera", {})
        zmin = float(cam.get("min_valid_depth_mm", 0) or 0)
        zmax = float(cam.get("max_valid_depth_mm", 0) or 0)
        if zmin > 0:
            mask &= z >= zmin
        if zmax > 0:
            mask &= z <= zmax
        zz = z[mask]
        u = us[mask].astype(np.float64)
        v = vs[mask].astype(np.float64)
        x = (u - intrinsics.cx) * zz / intrinsics.fx
        y = (v - intrinsics.cy) * zz / intrinsics.fy
        return np.stack([x, y, zz], axis=1), mask

    @staticmethod
    def _transform_points(points_cam: np.ndarray, cam_to_base: np.ndarray) -> np.ndarray:
        n = points_cam.shape[0]
        homo = np.hstack([points_cam, np.ones((n, 1))])
        return (cam_to_base @ homo.T).T[:, :3]

    def _fit_plane_base(self, depth_mm, intrinsics, cam_to_base):
        points_cam, _ = self._deproject(depth_mm, intrinsics)
        if points_cam.shape[0] < 50:
            return None
        points = self._transform_points(points_cam, cam_to_base)
        cfg = self.config["height"]
        max_pts = int(cfg.get("plane_sample_points", 4000))
        rng = np.random.default_rng(0)
        if points.shape[0] > max_pts:
            sample = points[rng.choice(points.shape[0], max_pts, replace=False)]
        else:
            sample = points
        threshold = float(cfg.get("plane_inlier_mm", 8.0))
        iterations = int(cfg.get("plane_ransac_iters", 120))
        n = sample.shape[0]
        best_inliers = None
        best_count = 0
        for _ in range(iterations):
            tri = sample[rng.choice(n, 3, replace=False)]
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm
            d = -float(normal @ tri[0])
            dist = np.abs(sample @ normal + d)
            count = int((dist < threshold).sum())
            if count > best_count:
                best_count = count
                best_inliers = dist < threshold
        if best_inliers is None or best_count < max(50, int(0.2 * n)):
            return None
        inlier_pts = sample[best_inliers]
        centroid = inlier_pts.mean(axis=0)
        _, _, vh = np.linalg.svd(inlier_pts - centroid)
        normal = vh[-1]
        normal = normal / np.linalg.norm(normal)
        d = -float(normal @ centroid)
        if normal[2] < 0:
            normal = -normal
            d = -d
        return float(normal[0]), float(normal[1]), float(normal[2]), float(d)

    def estimate(self, depth_mm, detection, intrinsics=None, cam_to_base=None):
        rx1, rx2, ry1, ry2 = self._roi_bounds(depth_mm, detection)
        roi = depth_mm[ry1:ry2, rx1:rx2]
        valid_mask = np.isfinite(roi) & (roi > 0)
        if not valid_mask.any():
            return 0.0
        min_h = float(self.config["height"].get("min_height_mm", 15.0))
        max_h = float(self.config["height"].get("max_height_mm", 60.0))
        if intrinsics is not None and cam_to_base is not None:
            if self.plane_base is None:
                self.plane_base = self._fit_plane_base(depth_mm, intrinsics, cam_to_base)
            if self.plane_base is not None:
                height = self._estimate_with_plane(roi, valid_mask, (rx1, ry1), intrinsics, cam_to_base)
                if height is not None:
                    return float(np.clip(height, min_h, max_h))
        valid = roi[valid_mask]
        top_depth = float(np.percentile(valid, 35))
        object_height = self.table_depth_mm - top_depth
        return float(np.clip(object_height, min_h, max_h))

    def _roi_bounds(self, depth_mm, detection):
        x1, y1, x2, y2 = detection.bbox
        roi_ratio = float(self.config["height"].get("roi_ratio", 0.35))
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        cx, cy = detection.center
        half_w = max(2, int(width * roi_ratio / 2))
        half_h = max(2, int(height * roi_ratio / 2))
        rx1 = max(0, cx - half_w)
        rx2 = min(depth_mm.shape[1], cx + half_w)
        ry1 = max(0, cy - half_h)
        ry2 = min(depth_mm.shape[0], cy + half_h)
        return rx1, rx2, ry1, ry2

    def _estimate_with_plane(self, roi, valid_mask, origin, intrinsics, cam_to_base):
        a, b, c, d = self.plane_base
        rx1, ry1 = origin
        ys, xs = np.nonzero(valid_mask)
        z = roi[ys, xs].astype(np.float64)
        u = (xs + rx1).astype(np.float64)
        v = (ys + ry1).astype(np.float64)
        x = (u - intrinsics.cx) * z / intrinsics.fx
        y = (v - intrinsics.cy) * z / intrinsics.fy
        pts_cam = np.stack([x, y, z], axis=1)
        pts_base = self._transform_points(pts_cam, cam_to_base)
        dist = pts_base @ np.array([a, b, c]) + d
        if dist.size == 0:
            return None
        pct = float(self.config["height"].get("plane_top_percentile", 70.0))
        return float(np.percentile(dist, pct))
