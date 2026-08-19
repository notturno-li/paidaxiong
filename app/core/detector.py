from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import Detection


class FruitDetector:
    def __init__(self, config: dict, simulation: bool = False):
        self.config = config
        self.simulation = simulation
        self.model = None
        self._simulated_removed: set[tuple[int, int, int, int]] = set()
        self.class_names = list(config["model"]["class_names"])
        if not simulation:
            self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("未安装 ultralytics，无法加载 YOLO 模型") from exc
        weights = self._resolve_weights()
        if not weights.is_file():
            fallback_value = self.config["model"].get("fallback_weights")
            fallback = self._resolve_path(fallback_value) if fallback_value else None
            if fallback is None or not fallback.is_file():
                raise RuntimeError(
                    f"YOLO 模型不存在: {weights}；备用模型也不可用: {fallback}"
                )
            weights = fallback
        self.model = YOLO(str(weights))
        model_names = getattr(self.model, "names", {})
        if isinstance(model_names, dict):
            native_names = [str(model_names[index]) for index in sorted(model_names)]
        else:
            native_names = [str(name) for name in model_names]
        configured_names = [str(name) for name in self.config["model"].get("class_names", [])]
        if bool(self.config["model"].get("strict_class_names", False)) and native_names != configured_names:
            raise RuntimeError(
                "YOLO 模型类别顺序与配置不一致；为防止物料被送入错误组装位，已拒绝启动。"
                f" model={native_names} config={configured_names}"
            )
        if native_names:
            self.class_names = native_names

    def _resolve_weights(self) -> Path:
        pattern = str(self.config["model"].get("weights_glob") or "").strip()
        if pattern:
            pattern_path = Path(pattern)
            if pattern_path.is_absolute():
                root = pattern_path.anchor
                relative_pattern = str(pattern_path)[len(root) :].lstrip("/\\")
                candidates = list(Path(root).glob(relative_pattern))
            else:
                config_path = self.config.get("_config_path")
                root = Path(config_path).resolve().parent.parent if config_path else Path(__file__).resolve().parents[2]
                candidates = list(root.glob(pattern))
            files = [path.resolve() for path in candidates if path.is_file()]
            if files:
                return max(files, key=lambda path: path.stat().st_mtime_ns)
        return self._resolve_path(self.config["model"].get("weights"))

    def _resolve_path(self, value) -> Path:
        path = Path(str(value or ""))
        if path.is_absolute():
            return path
        config_path = self.config.get("_config_path")
        root = Path(config_path).resolve().parent.parent if config_path else Path(__file__).resolve().parents[2]
        return (root / path).resolve()

    def detect(self, color_image: np.ndarray) -> list[Detection]:
        if self.simulation or self.model is None:
            return self._detect_simulated(color_image)
        conf_threshold = float(self.config["model"]["conf_threshold"])
        results = self.model.predict(color_image, conf=conf_threshold, verbose=False)
        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0].cpu().item())
                label = str(names.get(cls_id, cls_id))
                confidence = float(box.conf[0].cpu().item())
                x1, y1, x2, y2 = map(int, xyxy)
                detections.append(Detection(label=label, confidence=confidence, bbox=(x1, y1, x2, y2), center=((x1 + x2) // 2, (y1 + y2) // 2)))
        return sorted(detections, key=lambda item: item.confidence, reverse=True)

    def _detect_simulated(self, color_image: np.ndarray) -> list[Detection]:
        height, width = color_image.shape[:2]
        simulation_objects = self.config.get("simulation", {}).get("objects", [])
        if isinstance(simulation_objects, list) and simulation_objects:
            detections: list[Detection] = []
            for item in simulation_objects:
                if not isinstance(item, dict):
                    continue
                box = item.get("bbox", [])
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                x1, y1, x2, y2 = (int(value) for value in box)
                x1, x2 = max(0, x1), min(width - 1, x2)
                y1, y2 = max(0, y1), min(height - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                key = (x1, y1, x2, y2)
                if key in self._simulated_removed:
                    continue
                detections.append(
                    Detection(
                        label=str(item.get("label", "目标")),
                        confidence=float(item.get("confidence", 0.9)),
                        bbox=(x1, y1, x2, y2),
                        center=((x1 + x2) // 2, (y1 + y2) // 2),
                    )
                )
            return sorted(detections, key=lambda item: item.confidence, reverse=True)
        boxes = [
            ("apple", 0.94, (140, 140, 220, 220)),
            ("banana", 0.91, (275, 125, 355, 205)),
            ("strawberry", 0.88, (400, 220, 480, 300)),
            ("grape", 0.86, (200, 275, 280, 355)),
        ]
        detections = []
        for label, conf, (x1, y1, x2, y2) in boxes:
            x1, x2 = max(0, x1), min(width - 1, x2)
            y1, y2 = max(0, y1), min(height - 1, y2)
            detections.append(Detection(label=label, confidence=conf, bbox=(x1, y1, x2, y2), center=((x1 + x2) // 2, (y1 + y2) // 2)))
        return detections

    def mark_simulated_removed(self, detection: Detection) -> None:
        if self.simulation:
            self._simulated_removed.add(tuple(int(value) for value in detection.bbox))
