from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

from dataset_studio.training import _write_status


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def run(request_path: Path) -> Path:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    run_dir = Path(request["run_dir"])
    _write_status(
        run_dir,
        {
            "state": "running",
            "run_id": request["run_id"],
            "message": "Ultralytics YOLO 正在训练",
            "progress": None,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    from ultralytics import YOLO

    model = YOLO(request["base_model"])
    train_args: dict[str, Any] = {
        "data": request["data_yaml"],
        "epochs": request["epochs"],
        "imgsz": request["imgsz"],
        "batch": request["batch"],
        "patience": request["patience"],
        "seed": request["seed"],
        "workers": 0,
        "project": str(run_dir / "ultralytics"),
        "name": "train",
        "exist_ok": True,
        "verbose": True,
    }
    if request.get("device"):
        train_args["device"] = request["device"]
    result = model.train(**train_args)
    trainer = model.trainer
    best_path = Path(str(trainer.best))
    last_path = Path(str(trainer.last))
    if not best_path.is_file():
        raise RuntimeError(f"训练结束但未找到 best.pt: {best_path}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    bundle_id = f"{request['project_id']}_{stamp}"
    bundle_dir = Path(request["models_root"]) / bundle_id
    bundle_dir.mkdir(parents=True, exist_ok=False)
    bundle_best = bundle_dir / "best.pt"
    shutil.copy2(best_path, bundle_best)
    if last_path.is_file():
        shutil.copy2(last_path, bundle_dir / "last.pt")
    project_root = Path(request["project_root"])
    relative_weights = bundle_best.relative_to(project_root).as_posix()
    metrics = _json_safe(getattr(result, "results_dict", {}))
    metadata = {
        "profile_version": 1,
        "id": bundle_id,
        "name": f"{request['project_name']} {stamp}",
        "weights": relative_weights,
        "class_names": request["classes"],
        "conf_threshold": 0.45,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "training": {
            "base_model": request["base_model"],
            "epochs": request["epochs"],
            "imgsz": request["imgsz"],
            "batch": request["batch"],
            "train_count": request["train_count"],
            "val_count": request["val_count"],
            "class_counts": request["class_counts"],
            "metrics": metrics,
        },
    }
    (bundle_dir / "model_profile.yaml").write_text(
        yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (bundle_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    _write_status(
        run_dir,
        {
            "state": "completed",
            "run_id": request["run_id"],
            "message": "训练完成，模型包已生成",
            "progress": 100,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model_profile": str(bundle_dir / "model_profile.yaml"),
            "weights": str(bundle_best),
            "metrics": metrics,
        },
    )
    return bundle_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request_path = Path(args.request).resolve()
    try:
        bundle = run(request_path)
        print(f"TRAINING_COMPLETE {bundle}", flush=True)
        return 0
    except Exception as exc:
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            run_dir = Path(request["run_dir"])
            _write_status(
                run_dir,
                {
                    "state": "failed",
                    "run_id": request.get("run_id"),
                    "message": str(exc),
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "traceback": traceback.format_exc(),
                },
            )
        except Exception:
            pass
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
