from __future__ import annotations

import json
import queue
import re
import socket
import threading
from dataclasses import dataclass
from typing import Any


TASK_KEYS = ("gear", "recognition", "defect", "measurement")


@dataclass(frozen=True)
class TaskOutputMessage:
    key: str
    text: str


class TcpTaskOutputReceiver:
    """Receive DVS module-B results without blocking the Qt GUI thread.

    The default wire format is one UTF-8 line per message. A line may be a JSON
    object, ``key=value``, ``key:value`` or ``key|value``. Unprefixed lines use
    ``default_key`` from the configuration.
    """

    def __init__(self, config: dict[str, Any], logger=None):
        settings = config.get("task_outputs", {}).get("tcp", {})
        self.settings = settings if isinstance(settings, dict) else {}
        self.logger = logger
        self.events: queue.Queue[tuple[Any, ...]] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.settings.get("enabled", False))

    @property
    def host(self) -> str:
        return str(self.settings.get("host") or "192.168.5.4").strip()

    @property
    def port(self) -> int | None:
        value = self.settings.get("port")
        try:
            port = int(value)
        except (TypeError, ValueError):
            return None
        return port if 1 <= port <= 65535 else None

    def start(self) -> None:
        if not self.enabled:
            self.events.put(("status", "disabled"))
            return
        if self.port is None:
            self.events.put(("status", "waiting_port"))
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="dvs-task-output-tcp", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        sock = self._socket
        self._socket = None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=1.5)
        self._thread = None

    def drain(self) -> list[tuple[str, Any]]:
        items: list[tuple[str, Any]] = []
        while True:
            try:
                items.append(self.events.get_nowait())
            except queue.Empty:
                return items

    def _run(self) -> None:
        connect_timeout = self._float_setting("connect_timeout_s", 1.5)
        receive_timeout = self._float_setting("receive_timeout_s", 0.25)
        reconnect_delay = self._float_setting("reconnect_interval_s", 1.0)
        max_bytes = max(1024, self._int_setting("max_buffer_bytes", 262144))
        while not self._stop.is_set():
            sock: socket.socket | None = None
            try:
                sock = socket.create_connection((self.host, self.port or 0), timeout=connect_timeout)
                sock.settimeout(receive_timeout)
                self._socket = sock
                self.events.put(("status", "connected"))
                self._receive(sock, max_bytes)
            except (OSError, ValueError) as exc:
                if not self._stop.is_set():
                    self.events.put(("status", "error", str(exc)))
            finally:
                self._socket = None
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
                if not self._stop.is_set():
                    self.events.put(("status", "disconnected"))
                    self._stop.wait(reconnect_delay)

    def _receive(self, sock: socket.socket, max_bytes: int) -> None:
        delimiter = self._delimiter()
        buffer = b""
        while not self._stop.is_set():
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                if buffer.strip():
                    self._emit_frame(buffer)
                return
            buffer += chunk
            if len(buffer) > max_bytes:
                self.events.put(("status", "error", "TCP 报文超过最大缓存，已丢弃当前缓存"))
                buffer = b""
            while delimiter in buffer:
                frame, buffer = buffer.split(delimiter, 1)
                if frame.strip():
                    self._emit_frame(frame)
        if buffer.strip():
            self._emit_frame(buffer)

    def _emit_frame(self, frame: bytes) -> None:
        encoding = str(self.settings.get("encoding") or "utf-8")
        try:
            text = frame.decode(encoding).lstrip("\ufeff").strip()
        except (LookupError, UnicodeDecodeError):
            text = frame.decode("gb18030", errors="replace").strip()
        try:
            messages = parse_task_output_payload(text, self.settings)
        except Exception as exc:
            # A malformed DVS frame must not terminate the receiver thread.
            self.events.put(("status", "error", f"DVS 报文解析失败: {exc}"))
            return
        for message in messages:
            self.events.put(("data", message))

    def _delimiter(self) -> bytes:
        value = self.settings.get("delimiter", "\\n")
        if value in ("", None):
            return b"\n"
        if value == "\\n":
            return b"\n"
        if value == "\\r\\n":
            return b"\r\n"
        return str(value).encode(str(self.settings.get("encoding") or "utf-8"))

    def _float_setting(self, key: str, default: float) -> float:
        try:
            return max(0.01, float(self.settings.get(key, default)))
        except (TypeError, ValueError):
            return default

    def _int_setting(self, key: str, default: int) -> int:
        try:
            return int(self.settings.get(key, default))
        except (TypeError, ValueError):
            return default


def parse_task_output_payload(payload: str, settings: dict[str, Any] | None = None) -> list[TaskOutputMessage]:
    """Normalize common DVS TCP payloads into the four configured task keys."""

    settings = settings or {}
    text = str(payload or "").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return _messages_from_mapping(value, settings)

    aliases = _aliases(settings)
    messages: list[TaskOutputMessage] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines or [text]:
        match = re.match(r"^\s*([^:=|]+?)\s*[:=|]\s*(.+?)\s*$", line)
        if match:
            key = _resolve_key(match.group(1), aliases)
            if key:
                messages.append(TaskOutputMessage(key, match.group(2).strip()))
                continue
        default_key = str(settings.get("default_key") or "").strip()
        if default_key in TASK_KEYS:
            messages.append(TaskOutputMessage(default_key, line))
    return messages


def _messages_from_mapping(value: dict[str, Any], settings: dict[str, Any]) -> list[TaskOutputMessage]:
    aliases = _aliases(settings)
    messages: list[TaskOutputMessage] = []
    for raw_key, raw_value in value.items():
        key = _resolve_key(str(raw_key), aliases)
        if key and raw_value is not None:
            if isinstance(raw_value, (dict, list)):
                raw_value = json.dumps(raw_value, ensure_ascii=False, separators=(",", ":"))
            messages.append(TaskOutputMessage(key, str(raw_value).strip()))
    return messages


def _aliases(settings: dict[str, Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    configured = settings.get("field_aliases", {})
    for key in TASK_KEYS:
        values = [key, key.lower()]
        if isinstance(configured, dict):
            extra = configured.get(key, [])
            values.extend(extra if isinstance(extra, list) else [extra])
        for value in values:
            normalized = _normalize_key(str(value))
            if normalized:
                aliases[normalized] = key
    return aliases


def _resolve_key(value: str, aliases: dict[str, str]) -> str | None:
    return aliases.get(_normalize_key(value))


def _normalize_key(value: str) -> str:
    # ``str.removesuffix`` is unavailable on the Python 3.8 runtime used on
    # the field PC, so keep this normalization compatible with 3.8.
    normalized = value.strip().lower()
    if normalized.endswith(".txt"):
        normalized = normalized[:-4]
    return re.sub(r"[\s_./\\-]+", "", normalized)
