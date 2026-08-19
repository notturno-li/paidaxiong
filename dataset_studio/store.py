from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np


def safe_project_id(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_-]+", "_", value.strip()).strip("_-").lower()
    return text or "field_dataset"


def normalize_classes(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        raise ValueError("类别必须是列表")
    classes = [str(value).strip() for value in values if str(value).strip()]
    if not classes:
        raise ValueError("至少需要一个类别")
    if len(classes) != len(set(name.casefold() for name in classes)):
        raise ValueError("类别名称不能重复")
    if len(classes) > 100:
        raise ValueError("类别数量不能超过 100")
    return classes


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    file_name: str
    width: int
    height: int
    captured_at: str
    source: str
    annotated: bool
    box_count: int
    version: int


class ProjectStore:
    def __init__(self, root: str | Path, project_id: str, initial_classes: list[str] | None = None):
        self.root = Path(root).resolve()
        self.project_id = safe_project_id(project_id)
        self.project_dir = self.root / self.project_id
        self.images_dir = self.project_dir / "images"
        self.annotations_dir = self.project_dir / "annotations"
        self.labels_dir = self.project_dir / "labels"
        self.project_file = self.project_dir / "project.json"
        self._mutex = threading.RLock()
        for directory in (self.images_dir, self.annotations_dir, self.labels_dir):
            directory.mkdir(parents=True, exist_ok=True)
        if not self.project_file.exists():
            self._write_json_atomic(
                self.project_file,
                {
                    "project_id": self.project_id,
                    "name": self.project_id,
                    "classes": normalize_classes(initial_classes or ["object"]),
                    "created_at": self._timestamp(),
                    "updated_at": self._timestamp(),
                },
            )
        self.load_project()

    def load_project(self) -> dict[str, Any]:
        with self._mutex:
            value = json.loads(self.project_file.read_text(encoding="utf-8"))
            value["classes"] = normalize_classes(value.get("classes", []))
            return value

    def update_project(self, name: str, classes: list[str]) -> dict[str, Any]:
        classes = normalize_classes(classes)
        with self._mutex:
            current = self.load_project()
            old_classes = current["classes"]
            if self.annotation_count() and len(classes) < len(old_classes):
                raise ValueError("已有标注时不能删除类别；可改名或在末尾增加类别")
            current.update(
                {
                    "name": str(name).strip() or current.get("name") or self.project_id,
                    "classes": classes,
                    "updated_at": self._timestamp(),
                }
            )
            self._write_json_atomic(self.project_file, current)
            return current

    def add_image_bytes(self, payload: bytes, source: str = "upload") -> ImageRecord:
        if not payload:
            raise ValueError("图片内容为空")
        array = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3:
            raise ValueError("无法解码图片，只支持常见 JPG/PNG/BMP 格式")
        height, width = image.shape[:2]
        if width < 32 or height < 32:
            raise ValueError("图片尺寸过小")
        return self.add_image(image, source=source)

    def add_image(self, image: np.ndarray, source: str = "camera") -> ImageRecord:
        if image is None or image.ndim != 3:
            raise ValueError("相机未返回有效彩色图像")
        height, width = image.shape[:2]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        image_id = f"{stamp}_{uuid4().hex[:8]}"
        destination = self.images_dir / f"{image_id}.jpg"
        temporary = destination.with_suffix(".jpg.tmp")
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("图片编码失败")
        temporary.write_bytes(encoded.tobytes())
        temporary.replace(destination)
        metadata = {
            "image_id": image_id,
            "file_name": destination.name,
            "width": int(width),
            "height": int(height),
            "captured_at": self._timestamp(),
            "source": source,
        }
        self._write_json_atomic(self.images_dir / f"{image_id}.json", metadata)
        return ImageRecord(**metadata, annotated=False, box_count=0, version=0)

    def image_path(self, image_id: str) -> Path:
        self._validate_image_id(image_id)
        path = self.images_dir / f"{image_id}.jpg"
        if not path.is_file():
            raise FileNotFoundError(f"图片不存在: {image_id}")
        return path

    def list_images(self) -> list[ImageRecord]:
        records: list[ImageRecord] = []
        with self._mutex:
            for metadata_path in sorted(self.images_dir.glob("*.json")):
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                annotation = self._read_annotation(metadata["image_id"])
                records.append(
                    ImageRecord(
                        image_id=metadata["image_id"],
                        file_name=metadata["file_name"],
                        width=int(metadata["width"]),
                        height=int(metadata["height"]),
                        captured_at=str(metadata.get("captured_at", "")),
                        source=str(metadata.get("source", "unknown")),
                        annotated=bool(annotation.get("completed", False)),
                        box_count=len(annotation.get("boxes", [])),
                        version=int(annotation.get("version", 0)),
                    )
                )
        return records

    def get_annotation(self, image_id: str) -> dict[str, Any]:
        self.image_path(image_id)
        return self._read_annotation(image_id)

    def save_annotation(
        self,
        image_id: str,
        boxes: list[dict[str, Any]],
        expected_version: int,
        client_id: str,
        completed: bool = True,
    ) -> dict[str, Any]:
        metadata = self._read_image_metadata(image_id)
        classes = self.load_project()["classes"]
        cleaned = [self._validate_box(box, len(classes), index) for index, box in enumerate(boxes, 1)]
        with self._mutex:
            current = self._read_annotation(image_id)
            if int(current.get("version", 0)) != int(expected_version):
                raise VersionConflict(
                    f"标注已被其他队友更新：当前版本 {current.get('version', 0)}，"
                    f"你的版本 {expected_version}"
                )
            annotation = {
                "image_id": image_id,
                "width": metadata["width"],
                "height": metadata["height"],
                "completed": bool(completed),
                "boxes": cleaned,
                "version": int(expected_version) + 1,
                "updated_by": str(client_id)[:80],
                "updated_at": self._timestamp(),
            }
            self._write_json_atomic(self.annotations_dir / f"{image_id}.json", annotation)
            lines = []
            for box in cleaned:
                center_x = box["x"] + box["width"] / 2
                center_y = box["y"] + box["height"] / 2
                lines.append(
                    f"{box['class_id']} {center_x:.8f} {center_y:.8f} "
                    f"{box['width']:.8f} {box['height']:.8f}"
                )
            label_path = self.labels_dir / f"{image_id}.txt"
            temporary = label_path.with_suffix(".txt.tmp")
            temporary.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="ascii")
            temporary.replace(label_path)
            return annotation

    def delete_image(self, image_id: str) -> None:
        self._validate_image_id(image_id)
        with self._mutex:
            for path in (
                self.images_dir / f"{image_id}.jpg",
                self.images_dir / f"{image_id}.json",
                self.annotations_dir / f"{image_id}.json",
                self.labels_dir / f"{image_id}.txt",
            ):
                if path.exists():
                    path.unlink()

    def annotation_count(self) -> int:
        return sum(1 for item in self.list_images() if item.annotated)

    def summary(self) -> dict[str, Any]:
        images = self.list_images()
        project = self.load_project()
        per_class = {name: 0 for name in project["classes"]}
        for record in images:
            if not record.annotated:
                continue
            annotation = self._read_annotation(record.image_id)
            for box in annotation.get("boxes", []):
                class_id = int(box["class_id"])
                if 0 <= class_id < len(project["classes"]):
                    name = project["classes"][class_id]
                    per_class[name] += 1
        return {
            "total": len(images),
            "annotated": sum(record.annotated for record in images),
            "pending": sum(not record.annotated for record in images),
            "negative": sum(record.annotated and record.box_count == 0 for record in images),
            "boxes": sum(record.box_count for record in images),
            "per_class": per_class,
        }

    def _read_image_metadata(self, image_id: str) -> dict[str, Any]:
        self._validate_image_id(image_id)
        path = self.images_dir / f"{image_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"图片元数据不存在: {image_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _read_annotation(self, image_id: str) -> dict[str, Any]:
        path = self.annotations_dir / f"{image_id}.json"
        if not path.is_file():
            return {
                "image_id": image_id,
                "completed": False,
                "boxes": [],
                "version": 0,
            }
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _validate_box(box: Any, class_count: int, index: int) -> dict[str, Any]:
        if not isinstance(box, dict):
            raise ValueError(f"第 {index} 个标注框格式无效")
        try:
            class_id = int(box["class_id"])
            x = float(box["x"])
            y = float(box["y"])
            width = float(box["width"])
            height = float(box["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"第 {index} 个标注框缺少数值字段") from exc
        if not 0 <= class_id < class_count:
            raise ValueError(f"第 {index} 个标注框类别编号越界")
        if width <= 0 or height <= 0:
            raise ValueError(f"第 {index} 个标注框宽高必须大于 0")
        if x < 0 or y < 0 or x + width > 1.000001 or y + height > 1.000001:
            raise ValueError(f"第 {index} 个标注框超出图片范围")
        return {
            "class_id": class_id,
            "x": max(0.0, min(1.0, x)),
            "y": max(0.0, min(1.0, y)),
            "width": max(0.0, min(1.0, width)),
            "height": max(0.0, min(1.0, height)),
        }

    @staticmethod
    def _validate_image_id(image_id: str) -> None:
        if not re.fullmatch(r"[0-9A-Za-z_-]+", str(image_id)):
            raise ValueError("图片 ID 无效")

    @staticmethod
    def _write_json_atomic(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _timestamp() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class VersionConflict(RuntimeError):
    pass


class ImageLockManager:
    def __init__(self, timeout_s: float = 600.0):
        self.timeout_s = float(timeout_s)
        self._locks: dict[str, tuple[str, float]] = {}
        self._mutex = threading.Lock()

    def acquire(self, image_id: str, client_id: str) -> bool:
        now = time.monotonic()
        with self._mutex:
            self._purge(now)
            current = self._locks.get(image_id)
            if current and current[0] != client_id:
                return False
            self._locks[image_id] = (client_id, now)
            return True

    def release(self, image_id: str, client_id: str) -> None:
        with self._mutex:
            current = self._locks.get(image_id)
            if current and current[0] == client_id:
                self._locks.pop(image_id, None)

    def owner(self, image_id: str) -> str | None:
        now = time.monotonic()
        with self._mutex:
            self._purge(now)
            current = self._locks.get(image_id)
            return current[0] if current else None

    def snapshot(self) -> dict[str, str]:
        now = time.monotonic()
        with self._mutex:
            self._purge(now)
            return {image_id: owner for image_id, (owner, _stamp) in self._locks.items()}

    def _purge(self, now: float) -> None:
        expired = [
            image_id
            for image_id, (_owner, stamp) in self._locks.items()
            if now - stamp > self.timeout_s
        ]
        for image_id in expired:
            self._locks.pop(image_id, None)

