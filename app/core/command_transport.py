from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CommandReply:
    ok: bool
    mode: str
    raw: str
    data: Any = None


class JsonCommandTransport:
    """Send one newline-delimited JSON command to the robot-side task server."""

    def __init__(self, config: dict, simulation: bool = False):
        self.config = config
        self.simulation = simulation

    def send(self, payload: dict) -> CommandReply:
        transport = self.config.get("command_transport", {})
        mode = str(transport.get("mode", "preview")).lower()
        message = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if self.simulation:
            return CommandReply(True, "simulation", message, {"ok": True})
        if mode == "preview":
            return CommandReply(True, "preview", message, {"ok": True})
        if mode != "tcp":
            raise RuntimeError(f"不支持的指令传输模式: {mode}")

        host = str(transport.get("host") or self.config["robot"]["ip"])
        port = int(transport.get("port", 10000))
        connect_timeout = float(transport.get("connect_timeout_s", 2.0))
        reply_timeout = float(transport.get("reply_timeout_s", 3.0))
        terminator = str(transport.get("terminator", "\n"))
        try:
            with socket.create_connection((host, port), timeout=connect_timeout) as sock:
                sock.settimeout(reply_timeout)
                sock.sendall((message + terminator).encode("utf-8"))
                raw = sock.recv(8192).decode("utf-8", errors="replace").strip()
        except (OSError, socket.timeout) as exc:
            raise RuntimeError(f"JSON TCP 下发失败 {host}:{port}: {exc}") from exc
        if not raw:
            raise RuntimeError("机器人任务服务器未返回握手应答")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = raw
        if not self._is_ok(data):
            raise RuntimeError(f"机器人任务服务器拒绝指令: {raw}")
        return CommandReply(True, "tcp", raw, data)

    @staticmethod
    def _is_ok(data: Any) -> bool:
        if isinstance(data, dict):
            if "ok" in data:
                return bool(data["ok"])
            status = str(data.get("status", "")).lower()
            return status in {"ok", "accepted", "ready", "success"}
        return str(data).strip().lower() in {"ok", "accepted", "ready", "success", "0"}
