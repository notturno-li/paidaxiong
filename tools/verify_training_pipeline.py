from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.model_profiles import load_model_profile
from dataset_studio.store import ProjectStore
from dataset_studio.training import TrainingManager


def main() -> int:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    store = ProjectStore(ROOT / "runs" / "training_smoke", f"smoke_{stamp}", ["object"])
    for source in (ROOT / "models" / "t1.jpg", ROOT / "models" / "t2.jpg"):
        record = store.add_image_bytes(source.read_bytes(), source="smoke_test")
        store.save_annotation(
            record.image_id,
            [{"class_id": 0, "x": 0.2, "y": 0.2, "width": 0.5, "height": 0.5}],
            expected_version=0,
            client_id="smoke-test",
        )
    manager = TrainingManager(
        store,
        ROOT,
        ROOT / "runs" / "training_smoke_models",
    )
    status = manager.start(
        {
            "base_model": "models/yolov8s.pt",
            "epochs": 1,
            "imgsz": 128,
            "batch": 1,
            "patience": 0,
            "val_ratio": 0.5,
            "device": "cpu",
            "seed": 42,
        }
    )
    deadline = time.monotonic() + 300
    while status.get("state") in {"starting", "running"} and time.monotonic() < deadline:
        time.sleep(1)
        status = manager.status()
    if status.get("state") != "completed":
        print(status.get("log_tail", ""))
        raise RuntimeError(f"training smoke test failed: {status}")
    profile = load_model_profile(status["model_profile"], ROOT)
    if profile.class_names != ("object",) or not profile.weights_path.is_file():
        raise RuntimeError("generated model profile is invalid")
    print(f"TRAINING_PIPELINE_OK {profile.profile_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
