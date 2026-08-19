from __future__ import annotations

import argparse
import cgi
import json
import mimetypes
import socket
import sys
import threading
import webbrowser
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from dataset_studio.capture import CameraCapture
from dataset_studio.store import ImageLockManager, ProjectStore, VersionConflict
from dataset_studio.training import TrainingManager


STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_BODY_BYTES = 60 * 1024 * 1024
STATIC_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
}


class StudioApplication:
    def __init__(
        self,
        store: ProjectStore,
        camera: CameraCapture,
        trainer: TrainingManager,
        host: str,
        port: int,
    ):
        self.store = store
        self.camera = camera
        self.trainer = trainer
        self.locks = ImageLockManager(timeout_s=600)
        self.host = host
        self.port = int(port)

    def state(self) -> dict[str, Any]:
        locks = self.locks.snapshot()
        images = []
        for record in self.store.list_images():
            value = asdict(record)
            value["locked_by"] = locks.get(record.image_id)
            value["image_url"] = f"/api/image/{record.image_id}"
            images.append(value)
        return {
            "project": self.store.load_project(),
            "summary": self.store.summary(),
            "images": images,
            "camera": {"mode": self.camera.mode, "index": self.camera.index},
            "base_models": self.trainer.available_base_models(),
            "training": self.trainer.status(),
            "server_urls": lan_urls(self.port),
        }


def make_handler(application: StudioApplication):
    class StudioHandler(BaseHTTPRequestHandler):
        server_version = "DatasetStudio/1.0"

        def do_GET(self) -> None:  # noqa: N802
            try:
                self._handle_get()
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass
            except Exception as exc:
                self._error(exc)

        def do_POST(self) -> None:  # noqa: N802
            try:
                self._handle_post()
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass
            except VersionConflict as exc:
                self._json({"ok": False, "error": str(exc)}, HTTPStatus.CONFLICT)
            except (ValueError, FileNotFoundError) as exc:
                self._json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self._error(exc)

        def _handle_get(self) -> None:
            path = unquote(urlparse(self.path).path)
            if path == "/":
                self._serve_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            elif path == "/favicon.ico":
                self._bytes(b"", "image/x-icon", status=HTTPStatus.NO_CONTENT)
            elif path.startswith("/static/"):
                relative = path[len("/static/") :]
                if not relative or "/" in relative or "\\" in relative:
                    raise FileNotFoundError("静态资源不存在")
                self._serve_file(STATIC_DIR / relative)
            elif path == "/api/state":
                self._json({"ok": True, "data": application.state()})
            elif path.startswith("/api/image/"):
                image_id = path.rsplit("/", 1)[-1]
                self._serve_file(application.store.image_path(image_id), "image/jpeg")
            elif path.startswith("/api/annotation/"):
                image_id = path.rsplit("/", 1)[-1]
                self._json({"ok": True, "data": application.store.get_annotation(image_id)})
            elif path == "/api/camera/preview":
                self._bytes(
                    application.camera.preview_jpeg(),
                    "image/jpeg",
                    extra_headers={"Cache-Control": "no-store"},
                )
            elif path == "/api/training/status":
                self._json({"ok": True, "data": application.trainer.status()})
            else:
                self._json({"ok": False, "error": "接口不存在"}, HTTPStatus.NOT_FOUND)

        def _handle_post(self) -> None:
            path = unquote(urlparse(self.path).path)
            if path == "/api/project":
                body = self._read_json()
                project = application.store.update_project(body.get("name", ""), body.get("classes", []))
                self._json({"ok": True, "data": project})
            elif path == "/api/capture":
                frame = application.camera.capture()
                record = application.store.add_image(frame, source=application.camera.mode)
                self._json({"ok": True, "data": asdict(record)})
            elif path == "/api/upload":
                files = self._read_uploads()
                records = [
                    asdict(application.store.add_image_bytes(payload, source="upload"))
                    for _name, payload in files
                ]
                self._json({"ok": True, "data": records})
            elif path == "/api/lock":
                body = self._read_json()
                image_id = str(body.get("image_id") or "")
                client_id = _client_id(body)
                application.store.image_path(image_id)
                acquired = application.locks.acquire(image_id, client_id)
                if not acquired:
                    self._json(
                        {
                            "ok": False,
                            "error": "该图片正在被另一位队友标注",
                            "locked_by": application.locks.owner(image_id),
                        },
                        HTTPStatus.CONFLICT,
                    )
                else:
                    self._json({"ok": True, "data": {"image_id": image_id, "owner": client_id}})
            elif path == "/api/unlock":
                body = self._read_json()
                application.locks.release(str(body.get("image_id") or ""), _client_id(body))
                self._json({"ok": True})
            elif path == "/api/next":
                body = self._read_json()
                client_id = _client_id(body)
                only_pending = bool(body.get("only_pending", True))
                selected = None
                for record in application.store.list_images():
                    if only_pending and record.annotated:
                        continue
                    if application.locks.acquire(record.image_id, client_id):
                        selected = asdict(record)
                        selected["image_url"] = f"/api/image/{record.image_id}"
                        break
                self._json({"ok": True, "data": selected})
            elif path.startswith("/api/annotation/"):
                image_id = path.rsplit("/", 1)[-1]
                body = self._read_json()
                client_id = _client_id(body)
                if not application.locks.acquire(image_id, client_id):
                    raise VersionConflict("当前图片锁属于另一位队友，不能覆盖")
                annotation = application.store.save_annotation(
                    image_id,
                    body.get("boxes", []),
                    int(body.get("version", 0)),
                    client_id,
                    completed=bool(body.get("completed", True)),
                )
                application.locks.release(image_id, client_id)
                self._json({"ok": True, "data": annotation})
            elif path == "/api/delete":
                body = self._read_json()
                if body.get("confirm") is not True:
                    raise ValueError("删除图片必须明确确认")
                image_id = str(body.get("image_id") or "")
                client_id = _client_id(body)
                if not application.locks.acquire(image_id, client_id):
                    raise VersionConflict("当前图片锁属于另一位队友，不能删除")
                application.store.delete_image(image_id)
                application.locks.release(image_id, client_id)
                self._json({"ok": True})
            elif path == "/api/training/start":
                self._json({"ok": True, "data": application.trainer.start(self._read_json())})
            elif path == "/api/training/cancel":
                self._read_json()
                self._json({"ok": True, "data": application.trainer.cancel()})
            else:
                self._json({"ok": False, "error": "接口不存在"}, HTTPStatus.NOT_FOUND)

        def _read_json(self) -> dict[str, Any]:
            content_type = self.headers.get("Content-Type", "")
            if "application/json" not in content_type:
                raise ValueError("请求必须使用 application/json")
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > MAX_BODY_BYTES:
                raise ValueError("请求内容过大")
            raw = self.rfile.read(length)
            value = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(value, dict):
                raise ValueError("JSON 顶层必须是对象")
            return value

        def _read_uploads(self) -> list[tuple[str, bytes]]:
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                raise ValueError("上传接口需要 multipart/form-data")
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ValueError("上传内容为空或超过 60MB")
            environment = {
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": str(length),
            }
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ=environment,
                keep_blank_values=False,
            )
            items = form["images"] if "images" in form else []
            if not isinstance(items, list):
                items = [items]
            output = []
            for item in items:
                if getattr(item, "file", None) is None:
                    continue
                payload = item.file.read()
                if payload:
                    output.append((str(getattr(item, "filename", "image")), payload))
            if not output:
                raise ValueError("未选择有效图片")
            return output

        def _serve_file(self, path: Path, content_type: str | None = None) -> None:
            if not path.is_file():
                raise FileNotFoundError(str(path))
            guessed = (
                content_type
                or STATIC_CONTENT_TYPES.get(path.suffix.lower())
                or mimetypes.guess_type(path.name)[0]
                or "application/octet-stream"
            )
            self._bytes(path.read_bytes(), guessed)

        def _json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            payload = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
            self._bytes(payload, "application/json; charset=utf-8", status=status)

        def _bytes(
            self,
            payload: bytes,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status.value)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store" if content_type.startswith("application/json") else "private, max-age=60")
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(payload)

        def _error(self, exc: Exception) -> None:
            print(f"HTTP {self.command} {self.path} failed: {exc}", file=sys.stderr)
            status = HTTPStatus.NOT_FOUND if isinstance(exc, FileNotFoundError) else HTTPStatus.INTERNAL_SERVER_ERROR
            self._json({"ok": False, "error": str(exc)}, status)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[{self.log_date_time_string()}] {self.client_address[0]} {format % args}")

    return StudioHandler


def _client_id(body: dict[str, Any]) -> str:
    value = str(body.get("client_id") or "").strip()
    if not value:
        raise ValueError("缺少标注客户端 ID")
    return value[:80]


def lan_urls(port: int) -> list[str]:
    addresses = {"127.0.0.1"}
    try:
        addresses.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        addresses.add(probe.getsockname()[0])
        probe.close()
    except OSError:
        pass
    ordered = sorted(address for address in addresses if ":" not in address)
    return [f"http://{address}:{port}" for address in ordered]


def main() -> int:
    parser = argparse.ArgumentParser(description="离线局域网 YOLO 数据集一条龙工作台")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--project", default="field_dataset")
    parser.add_argument("--classes", default="object")
    parser.add_argument("--camera", choices=["realsense", "opencv"], default="realsense")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--config", default=str(ROOT / "configs" / "field.yaml"))
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port 必须在 1~65535")
    config = load_config(args.config)
    camera_cfg = config.get("camera", {})
    store = ProjectStore(
        ROOT / "runs" / "dataset_studio",
        args.project,
        initial_classes=[value.strip() for value in args.classes.split(",") if value.strip()],
    )
    camera = CameraCapture(
        args.camera,
        index=args.camera_index,
        width=int(camera_cfg.get("width", 640)),
        height=int(camera_cfg.get("height", 480)),
        fps=int(camera_cfg.get("fps", 30)),
    )
    trainer = TrainingManager(store, ROOT, ROOT / "models" / "field_models")
    application = StudioApplication(store, camera, trainer, args.host, args.port)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(application))
    server.daemon_threads = True
    urls = lan_urls(args.port)
    print("DATASET_STUDIO_READY", flush=True)
    for url in urls:
        print("  " + url, flush=True)
    print("队友访问同一局域网 IP；首次访问请允许 Windows 防火墙的专用网络访问。", flush=True)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("正在关闭 Dataset Studio...", flush=True)
    finally:
        server.server_close()
        camera.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
