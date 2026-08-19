from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml


class ModelProfileError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelProfile:
    profile_path: Path
    profile_id: str
    name: str
    weights_path: Path
    weights_value: str
    class_names: tuple[str, ...]
    conf_threshold: float
    trained_at: str
    training: dict[str, Any]

    @property
    def image_count(self) -> int:
        return int(self.training.get("train_count", 0)) + int(
            self.training.get("val_count", 0)
        )


def load_model_profile(path: str | Path, project_root: str | Path) -> ModelProfile:
    profile_path = Path(path).resolve()
    root = Path(project_root).resolve()
    if not profile_path.is_file():
        raise ModelProfileError(f"模型配置不存在: {profile_path}")
    try:
        raw = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ModelProfileError(f"模型配置读取失败: {profile_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ModelProfileError(f"模型配置顶层必须是映射: {profile_path}")

    classes = _normalize_classes(raw.get("class_names"))
    try:
        threshold = float(raw.get("conf_threshold", 0.45))
    except (TypeError, ValueError) as exc:
        raise ModelProfileError("conf_threshold 必须是数值") from exc
    if not 0.0 < threshold <= 1.0:
        raise ModelProfileError("conf_threshold 必须在 0~1 之间")

    weights_value = str(raw.get("weights") or "").strip()
    if not weights_value:
        raise ModelProfileError("模型配置缺少 weights")
    weights_path = Path(weights_value)
    if not weights_path.is_absolute():
        root_candidate = (root / weights_path).resolve()
        local_candidate = (profile_path.parent / weights_path).resolve()
        weights_path = root_candidate if root_candidate.is_file() else local_candidate
    else:
        weights_path = weights_path.resolve()
    if weights_path.suffix.lower() != ".pt" or not weights_path.is_file():
        raise ModelProfileError(f"模型权重不存在或不是 .pt: {weights_path}")
    if weights_path.stat().st_size <= 0:
        raise ModelProfileError(f"模型权重为空: {weights_path}")

    training = raw.get("training") or {}
    if not isinstance(training, dict):
        raise ModelProfileError("training 必须是映射")
    return ModelProfile(
        profile_path=profile_path,
        profile_id=str(raw.get("id") or profile_path.parent.name),
        name=str(raw.get("name") or profile_path.parent.name),
        weights_path=weights_path,
        weights_value=_project_path_value(weights_path, root),
        class_names=tuple(classes),
        conf_threshold=threshold,
        trained_at=str(raw.get("trained_at") or ""),
        training=dict(training),
    )


def scan_model_profiles(
    models_root: str | Path, project_root: str | Path
) -> tuple[list[ModelProfile], list[str]]:
    root = Path(models_root)
    profiles: list[ModelProfile] = []
    errors: list[str] = []
    if not root.exists():
        return profiles, errors
    for path in sorted(root.rglob("model_profile.yaml")):
        try:
            profiles.append(load_model_profile(path, project_root))
        except ModelProfileError as exc:
            errors.append(str(exc))
    profiles.sort(key=lambda item: (item.trained_at, item.name), reverse=True)
    return profiles, errors


def import_model_bundle(
    source: str | Path,
    project_root: str | Path,
    *,
    name: str,
    class_names: list[str] | tuple[str, ...],
    conf_threshold: float = 0.45,
) -> ModelProfile:
    source_path = Path(source).resolve()
    root = Path(project_root).resolve()
    if source_path.suffix.lower() != ".pt" or not source_path.is_file():
        raise ModelProfileError("请选择有效的 .pt 权重文件")
    if source_path.stat().st_size <= 0:
        raise ModelProfileError("权重文件为空")
    clean_name = str(name).strip()
    if not clean_name:
        raise ModelProfileError("模型名称不能为空")
    classes = _normalize_classes(class_names)
    threshold = float(conf_threshold)
    if not 0.0 < threshold <= 1.0:
        raise ModelProfileError("置信度阈值必须在 0~1 之间")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    bundle_id = f"imported_{stamp}"
    bundle_dir = root / "models" / "field_models" / bundle_id
    if bundle_dir.exists():
        bundle_id += "_" + uuid4().hex[:6]
        bundle_dir = root / "models" / "field_models" / bundle_id
    bundle_dir.mkdir(parents=True, exist_ok=False)
    destination = bundle_dir / "best.pt"
    try:
        shutil.copy2(source_path, destination)
        metadata = {
            "profile_version": 1,
            "id": bundle_id,
            "name": clean_name,
            "weights": _project_path_value(destination, root),
            "class_names": classes,
            "conf_threshold": threshold,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "training": {
                "source": "external_import",
                "original_file": source_path.name,
                "train_count": 0,
                "val_count": 0,
            },
        }
        profile_path = bundle_dir / "model_profile.yaml"
        profile_path.write_text(
            yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        (bundle_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return load_model_profile(profile_path, root)
    except Exception:
        shutil.rmtree(str(bundle_dir), ignore_errors=True)
        raise


def model_config_update(profile: ModelProfile) -> dict[str, Any]:
    return {
        "model": {
            "weights": profile.weights_value,
            "class_names": list(profile.class_names),
            "conf_threshold": profile.conf_threshold,
        }
    }


def _normalize_classes(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        raise ModelProfileError("class_names 必须是列表")
    classes = [str(value).strip() for value in values if str(value).strip()]
    if not classes:
        raise ModelProfileError("至少需要一个类别")
    if len(classes) != len(set(value.casefold() for value in classes)):
        raise ModelProfileError("类别名称不能重复")
    return classes


def _project_path_value(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())
