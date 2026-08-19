from __future__ import annotations

import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import yaml

from app.core.model_profiles import (
    import_model_bundle,
    load_model_profile,
    model_config_update,
    scan_model_profiles,
)
from dataset_studio.server import StudioApplication, make_handler
from dataset_studio.store import ImageLockManager, ProjectStore, VersionConflict
from dataset_studio.training import TrainingManager, prepare_dataset


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = ProjectStore(self.root / "runs", "Field Test", ["part_a", "part_b"])
        self.image = np.full((80, 120, 3), 127, dtype=np.uint8)

    def tearDown(self):
        self.temporary.cleanup()

    def test_project_image_annotation_and_yolo_output(self):
        record = self.store.add_image(self.image, source="test")
        saved = self.store.save_annotation(
            record.image_id,
            [{"class_id": 1, "x": 0.1, "y": 0.2, "width": 0.4, "height": 0.5}],
            expected_version=0,
            client_id="alice",
        )
        self.assertEqual(saved["version"], 1)
        self.assertEqual(self.store.summary()["per_class"], {"part_a": 0, "part_b": 1})
        label = (self.store.labels_dir / f"{record.image_id}.txt").read_text("ascii")
        self.assertEqual(label, "1 0.30000000 0.45000000 0.40000000 0.50000000\n")

    def test_negative_sample_writes_empty_label(self):
        record = self.store.add_image(self.image)
        self.store.save_annotation(record.image_id, [], 0, "alice", completed=True)
        self.assertEqual((self.store.labels_dir / f"{record.image_id}.txt").read_text(), "")
        self.assertEqual(self.store.summary()["negative"], 1)

    def test_invalid_box_and_version_conflict_are_rejected(self):
        record = self.store.add_image(self.image)
        with self.assertRaisesRegex(ValueError, "超出图片范围"):
            self.store.save_annotation(
                record.image_id,
                [{"class_id": 0, "x": 0.8, "y": 0.1, "width": 0.3, "height": 0.2}],
                0,
                "alice",
            )
        self.store.save_annotation(record.image_id, [], 0, "alice")
        with self.assertRaises(VersionConflict):
            self.store.save_annotation(record.image_id, [], 0, "bob")

    def test_classes_cannot_be_removed_after_annotation(self):
        record = self.store.add_image(self.image)
        self.store.save_annotation(record.image_id, [], 0, "alice")
        with self.assertRaisesRegex(ValueError, "不能删除类别"):
            self.store.update_project("test", ["part_a"])

    def test_image_lock_has_owner_and_release(self):
        locks = ImageLockManager(timeout_s=60)
        self.assertTrue(locks.acquire("image", "alice"))
        self.assertFalse(locks.acquire("image", "bob"))
        self.assertEqual(locks.owner("image"), "alice")
        locks.release("image", "bob")
        self.assertEqual(locks.owner("image"), "alice")
        locks.release("image", "alice")
        self.assertIsNone(locks.owner("image"))

    def test_prepare_dataset_uses_completed_images_and_all_classes(self):
        records = [self.store.add_image(self.image) for _ in range(3)]
        self.store.save_annotation(
            records[0].image_id,
            [{"class_id": 0, "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}],
            0,
            "a",
        )
        self.store.save_annotation(
            records[1].image_id,
            [{"class_id": 1, "x": 0.2, "y": 0.2, "width": 0.3, "height": 0.3}],
            0,
            "a",
        )
        self.store.save_annotation(records[2].image_id, [], 0, "a")
        prepared = prepare_dataset(self.store, "unit", val_ratio=0.34, seed=7)
        data = yaml.safe_load(prepared.data_yaml.read_text(encoding="utf-8"))
        self.assertEqual(prepared.train_count + prepared.val_count, 3)
        self.assertEqual(data["names"], {0: "part_a", 1: "part_b"})
        self.assertEqual(len(list((prepared.build_dir / "images" / "val").glob("*.jpg"))), 1)

    def test_prepare_dataset_rejects_unseen_class(self):
        for _index in range(2):
            record = self.store.add_image(self.image)
            self.store.save_annotation(
                record.image_id,
                [{"class_id": 0, "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}],
                0,
                "a",
            )
        with self.assertRaisesRegex(ValueError, "part_b"):
            prepare_dataset(self.store, "missing")


class ModelProfileTests(unittest.TestCase):
    def test_import_scan_and_config_update(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pt"
            source.write_bytes(b"fake-yolo-weight")
            profile = import_model_bundle(
                source,
                root,
                name="现场零件",
                class_names=["ok", "ng"],
                conf_threshold=0.4,
            )
            scanned, errors = scan_model_profiles(root / "models" / "field_models", root)
            self.assertFalse(errors)
            self.assertEqual(scanned, [profile])
            self.assertEqual(model_config_update(profile)["model"]["class_names"], ["ok", "ng"])
            self.assertTrue(profile.weights_path.is_file())
            self.assertTrue((profile.profile_path.parent / "metadata.json").is_file())

    def test_invalid_profile_is_reported_without_hiding_valid_profiles(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = root / "models" / "field_models" / "bad"
            invalid.mkdir(parents=True)
            (invalid / "model_profile.yaml").write_text("class_names: []\n", encoding="utf-8")
            profiles, errors = scan_model_profiles(root / "models" / "field_models", root)
            self.assertEqual(profiles, [])
            self.assertEqual(len(errors), 1)

    def test_training_manager_rejects_model_outside_project_model_roots(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = ProjectStore(root / "runs", "test", ["item"])
            outside = root / "outside.pt"
            outside.write_bytes(b"x")
            manager = TrainingManager(store, root, root / "models" / "field_models")
            with self.assertRaisesRegex(ValueError, "models 或 scripts"):
                manager._resolve_base_model(str(outside))


class _CameraStub:
    mode = "test"
    index = 0

    def capture(self):
        return np.full((64, 96, 3), 100, dtype=np.uint8)

    def preview_jpeg(self):
        ok, encoded = cv2.imencode(".jpg", self.capture())
        assert ok
        return encoded.tobytes()


class _TrainerStub:
    def available_base_models(self):
        return []

    def status(self):
        return {"state": "idle", "message": "test"}

    def start(self, _settings):
        return self.status()

    def cancel(self):
        return self.status()


class HttpApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.store = ProjectStore(Path(self.temporary.name), "api", ["item"])
        self.record = self.store.add_image(np.full((64, 96, 3), 80, dtype=np.uint8))
        application = StudioApplication(self.store, _CameraStub(), _TrainerStub(), "127.0.0.1", 0)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(application))
        application.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temporary.cleanup()

    def _post(self, path, value):
        payload = json.dumps(value).encode("utf-8")
        request = Request(
            self.base + path,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=3) as response:
            return json.loads(response.read().decode("utf-8"))

    def test_state_next_lock_and_conflict(self):
        with urlopen(self.base + "/api/state", timeout=3) as response:
            state = json.loads(response.read().decode("utf-8"))["data"]
        self.assertEqual(state["summary"]["total"], 1)
        selected = self._post("/api/next", {"client_id": "alice", "only_pending": True})["data"]
        self.assertEqual(selected["image_url"], f"/api/image/{self.record.image_id}")
        with self.assertRaises(HTTPError) as caught:
            self._post(
                "/api/lock",
                {"image_id": self.record.image_id, "client_id": "bob"},
            )
        self.assertEqual(caught.exception.code, 409)

    def test_javascript_has_executable_content_type_on_windows(self):
        with urlopen(self.base + "/static/app.js", timeout=3) as response:
            self.assertEqual(response.headers.get_content_type(), "application/javascript")
            self.assertIn(b'"use strict"', response.read(40))

    def test_save_annotation_releases_lock(self):
        self._post("/api/lock", {"image_id": self.record.image_id, "client_id": "alice"})
        result = self._post(
            f"/api/annotation/{self.record.image_id}",
            {"client_id": "alice", "version": 0, "completed": True, "boxes": []},
        )
        self.assertEqual(result["data"]["version"], 1)
        acquired = self._post(
            "/api/lock", {"image_id": self.record.image_id, "client_id": "bob"}
        )
        self.assertTrue(acquired["ok"])


if __name__ == "__main__":
    unittest.main()
