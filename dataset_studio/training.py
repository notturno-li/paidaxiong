from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dataset_studio.store import ProjectStore


@dataclass(frozen=True)
class PreparedDataset:
    run_id: str
    build_dir: Path
    data_yaml: Path
    train_count: int
    val_count: int
    class_counts: dict[str, int]


def prepare_dataset(
    store: ProjectStore,
    run_id: str,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> PreparedDataset:
    if not 0.05 <= float(val_ratio) <= 0.5:
        raise ValueError("验证集比例必须在 0.05~0.5 之间")
    records = [record for record in store.list_images() if record.annotated]
    if len(records) < 2:
        raise ValueError("至少需要 2 张已完成标注的图片才能训练")
    project = store.load_project()
    classes = project["classes"]
    class_counts = {name: 0 for name in classes}
    for record in records:
        annotation = store.get_annotation(record.image_id)
        for box in annotation.get("boxes", []):
            class_counts[classes[int(box["class_id"])]] += 1
    missing = [name for name, count in class_counts.items() if count == 0]
    if missing:
        raise ValueError("以下类别没有任何标注框: " + ", ".join(missing))

    shuffled = list(records)
    random.Random(int(seed)).shuffle(shuffled)
    val_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * float(val_ratio))))
    val_ids = {record.image_id for record in shuffled[:val_count]}
    build_dir = store.project_dir / "builds" / run_id
    for split in ("train", "val"):
        (build_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (build_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    split_manifest: dict[str, list[str]] = {"train": [], "val": []}
    for record in records:
        split = "val" if record.image_id in val_ids else "train"
        shutil.copy2(store.image_path(record.image_id), build_dir / "images" / split / record.file_name)
        source_label = store.labels_dir / f"{record.image_id}.txt"
        if not source_label.is_file():
            raise RuntimeError(f"已完成图片缺少 YOLO 标签: {record.image_id}")
        shutil.copy2(source_label, build_dir / "labels" / split / source_label.name)
        split_manifest[split].append(record.image_id)

    data_yaml = build_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(build_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {index: name for index, name in enumerate(classes)},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (build_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "classes": classes,
                "class_counts": class_counts,
                "splits": split_manifest,
                "created_at": _timestamp(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return PreparedDataset(
        run_id=run_id,
        build_dir=build_dir,
        data_yaml=data_yaml,
        train_count=len(records) - val_count,
        val_count=val_count,
        class_counts=class_counts,
    )


class TrainingManager:
    def __init__(self, store: ProjectStore, project_root: str | Path, models_root: str | Path):
        self.store = store
        self.project_root = Path(project_root).resolve()
        self.models_root = Path(models_root).resolve()
        self._process: subprocess.Popen[Any] | None = None
        self._run_dir: Path | None = None
        self._log_stream: Any = None
        self._mutex = threading.RLock()

    def available_base_models(self) -> list[dict[str, str]]:
        candidates: list[Path] = []
        for pattern in ("models/*.pt", "scripts/*.pt"):
            candidates.extend(self.project_root.glob(pattern))
        output = []
        for path in sorted(set(path.resolve() for path in candidates if path.is_file())):
            output.append(
                {
                    "name": path.name,
                    "path": path.relative_to(self.project_root).as_posix(),
                }
            )
        return output

    def start(self, settings: dict[str, Any]) -> dict[str, Any]:
        with self._mutex:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("已有训练任务正在运行")
            epochs = _bounded_int(settings.get("epochs", 100), "epochs", 1, 1000)
            imgsz = _bounded_int(settings.get("imgsz", 640), "imgsz", 128, 2048)
            batch = _bounded_int(settings.get("batch", 8), "batch", 1, 256)
            patience = _bounded_int(settings.get("patience", 30), "patience", 0, 500)
            val_ratio = float(settings.get("val_ratio", 0.2))
            seed = int(settings.get("seed", 42))
            device = str(settings.get("device", "")).strip()
            base_model = self._resolve_base_model(str(settings.get("base_model") or ""))
            run_id = time.strftime("train_%Y%m%d_%H%M%S")
            prepared = prepare_dataset(self.store, run_id, val_ratio=val_ratio, seed=seed)
            run_dir = self.store.project_dir / "training" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            request = {
                "run_id": run_id,
                "project_id": self.store.project_id,
                "project_name": self.store.load_project()["name"],
                "classes": self.store.load_project()["classes"],
                "data_yaml": str(prepared.data_yaml),
                "base_model": str(base_model),
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "patience": patience,
                "device": device,
                "seed": seed,
                "train_count": prepared.train_count,
                "val_count": prepared.val_count,
                "class_counts": prepared.class_counts,
                "project_root": str(self.project_root),
                "models_root": str(self.models_root),
                "run_dir": str(run_dir),
            }
            request_path = run_dir / "request.json"
            request_path.write_text(
                json.dumps(request, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _write_status(
                run_dir,
                {
                    "state": "starting",
                    "run_id": run_id,
                    "message": "数据集已生成，正在启动训练",
                    "progress": 0,
                    "started_at": _timestamp(),
                },
            )
            log_path = run_dir / "train.log"
            self._log_stream = log_path.open("w", encoding="utf-8", buffering=1)
            command = [
                sys.executable,
                "-m",
                "dataset_studio.train_worker",
                "--request",
                str(request_path),
            ]
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._process = subprocess.Popen(
                command,
                cwd=str(self.project_root),
                stdout=self._log_stream,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
            )
            self._run_dir = run_dir
            return self.status()

    def status(self) -> dict[str, Any]:
        with self._mutex:
            if self._run_dir is None:
                latest = self._latest_run_dir()
                if latest is None:
                    return {"state": "idle", "message": "尚未开始训练", "log_tail": ""}
                self._run_dir = latest
            status_path = self._run_dir / "status.json"
            status = (
                json.loads(status_path.read_text(encoding="utf-8"))
                if status_path.is_file()
                else {"state": "unknown", "message": "训练状态文件不存在"}
            )
            process = self._process
            if process is not None:
                code = process.poll()
                if code is not None and status.get("state") in {"starting", "running"}:
                    status.update(
                        {
                            "state": "failed",
                            "message": f"训练进程异常退出，code={code}",
                            "finished_at": _timestamp(),
                        }
                    )
                    _write_status(self._run_dir, status)
                if code is not None and self._log_stream is not None:
                    self._log_stream.close()
                    self._log_stream = None
            status["log_tail"] = self._read_log_tail(self._run_dir / "train.log")
            return status

    def cancel(self) -> dict[str, Any]:
        with self._mutex:
            if self._process is None or self._process.poll() is not None:
                raise RuntimeError("当前没有正在运行的训练任务")
            self._process.terminate()
            if self._run_dir is not None:
                status = {
                    "state": "cancelled",
                    "run_id": self._run_dir.name,
                    "message": "训练已由用户停止",
                    "finished_at": _timestamp(),
                }
                _write_status(self._run_dir, status)
            return self.status()

    def _resolve_base_model(self, value: str) -> Path:
        if not value:
            raise ValueError("必须选择基础模型")
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        candidate = candidate.resolve()
        if candidate.suffix.lower() != ".pt" or not candidate.is_file():
            raise ValueError(f"基础模型不存在: {candidate}")
        allowed_roots = [self.project_root / "models", self.project_root / "scripts"]
        if not any(_is_relative_to(candidate, root.resolve()) for root in allowed_roots):
            raise ValueError("基础模型必须位于项目的 models 或 scripts 目录")
        return candidate

    def _latest_run_dir(self) -> Path | None:
        root = self.store.project_dir / "training"
        values = sorted((path for path in root.glob("train_*") if path.is_dir()), reverse=True)
        return values[0] if values else None

    @staticmethod
    def _read_log_tail(path: Path, max_chars: int = 16000) -> str:
        if not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[-max_chars:]


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    parsed = int(value)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} 必须在 {minimum}~{maximum} 之间")
    return parsed


def _write_status(run_dir: Path, value: dict[str, Any]) -> None:
    destination = run_dir / "status.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(destination)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
