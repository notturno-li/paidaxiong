from __future__ import annotations

import json
import socket
import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VisionStudioReply:
    transport: str
    raw: str
    data: Any


class DobotVisionStudioClient:
    """Protocol-neutral transport for DobotVisionStudio communication devices."""

    def __init__(self, config: dict, simulation: bool = False):
        self.config = config
        self.simulation = simulation
        self.sock: socket.socket | None = None
        self.listener: socket.socket | None = None
        self.serial_port = None
        self.connected = False
        self._lock = threading.Lock()

    @property
    def settings(self) -> dict:
        return self.config.get("vision_studio", {})

    @property
    def mode(self) -> str:
        if self.simulation:
            return "simulation"
        return str(self.settings.get("mode", "disabled")).lower()

    def discover(self) -> dict[str, Any]:
        ports: list[dict[str, str]] = []
        try:
            from serial.tools import list_ports

            ports = [
                {
                    "device": str(item.device),
                    "description": str(item.description or ""),
                    "hwid": str(item.hwid or ""),
                }
                for item in list_ports.comports()
            ]
        except ImportError:
            pass
        tcp = self.settings.get("tcp", {})
        return {
            "configured_mode": self.mode,
            "serial_ports": ports,
            "tcp_host": tcp.get("host"),
            "tcp_port": tcp.get("port"),
        }

    def connect(self) -> None:
        if self.connected:
            return
        if self.mode == "simulation":
            self.connected = True
            return
        if self.mode == "disabled":
            raise RuntimeError("DobotVisionStudio 通信尚未配置，请先选择串口或 TCP 模式")
        if self.mode == "serial":
            self._connect_serial()
        elif self.mode == "tcp_client":
            self._connect_tcp_client()
        elif self.mode == "tcp_server":
            self._connect_tcp_server()
        else:
            raise RuntimeError(f"不支持的 DobotVisionStudio 通信模式: {self.mode}")
        self.connected = True

    def _connect_serial(self) -> None:
        serial_cfg = self.settings.get("serial", {})
        port = serial_cfg.get("port")
        if not port:
            raise RuntimeError("DobotVisionStudio 串口号未配置，请先运行串口扫描")
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("未安装 pyserial，无法使用 DobotVisionStudio 串口通信") from exc

        parity = {
            "none": serial.PARITY_NONE,
            "odd": serial.PARITY_ODD,
            "even": serial.PARITY_EVEN,
            "mark": serial.PARITY_MARK,
            "space": serial.PARITY_SPACE,
        }
        stop_bits = {
            1.0: serial.STOPBITS_ONE,
            1.5: serial.STOPBITS_ONE_POINT_FIVE,
            2.0: serial.STOPBITS_TWO,
        }
        parity_name = str(serial_cfg.get("parity", "none")).lower()
        stop_value = float(serial_cfg.get("stop_bits", 1))
        if parity_name not in parity:
            raise RuntimeError(f"DobotVisionStudio 串口校验位无效: {parity_name}")
        if stop_value not in stop_bits:
            raise RuntimeError(f"DobotVisionStudio 串口停止位无效: {stop_value}")
        self.serial_port = serial.Serial(
            port=str(port),
            baudrate=int(serial_cfg.get("baudrate", 115200)),
            bytesize=int(serial_cfg.get("data_bits", 8)),
            parity=parity[parity_name],
            stopbits=stop_bits[stop_value],
            timeout=float(self.settings.get("reply_timeout_s", 3.0)),
            write_timeout=float(self.settings.get("connect_timeout_s", 2.0)),
        )

    def _connect_tcp_client(self) -> None:
        tcp = self.settings.get("tcp", {})
        host = tcp.get("host")
        port = tcp.get("port")
        if not host or port in {None, ""}:
            raise RuntimeError("DobotVisionStudio TCP 客户端目标 IP/端口尚未配置")
        self.sock = socket.create_connection(
            (str(host), int(port)),
            timeout=float(self.settings.get("connect_timeout_s", 2.0)),
        )
        self.sock.settimeout(float(self.settings.get("reply_timeout_s", 3.0)))

    def _connect_tcp_server(self) -> None:
        tcp = self.settings.get("tcp", {})
        host = str(tcp.get("host") or "0.0.0.0")
        port = tcp.get("port")
        if port in {None, ""}:
            raise RuntimeError("DobotVisionStudio TCP 服务端本机端口尚未配置")
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.settimeout(float(self.settings.get("connect_timeout_s", 2.0)))
        self.listener.bind((host, int(port)))
        self.listener.listen(1)
        try:
            self.sock, _address = self.listener.accept()
            self.sock.settimeout(float(self.settings.get("reply_timeout_s", 3.0)))
        except (OSError, socket.timeout) as exc:
            self.close()
            raise RuntimeError(f"等待 DobotVisionStudio TCP 客户端连接超时: {exc}") from exc

    def close(self) -> None:
        if self.serial_port is not None:
            try:
                self.serial_port.close()
            except Exception:
                pass
        self.serial_port = None
        for sock in (self.sock, self.listener):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self.sock = None
        self.listener = None
        self.connected = False

    def exchange(self, message: str | None = None) -> VisionStudioReply:
        protocol = self.settings.get("protocol", {})
        send_before_receive = bool(protocol.get("send_before_receive", True))
        message = str(message if message is not None else protocol.get("exchange_command") or "")
        if send_before_receive and not message:
            raise RuntimeError("DobotVisionStudio 触发/查询字符串尚未配置")
        with self._lock:
            self.connect()
            if self.mode == "simulation":
                raw = json.dumps(
                    self.settings.get("simulation_result", {"ok": True}),
                    ensure_ascii=False,
                )
            else:
                try:
                    if send_before_receive:
                        self._send_text(message)
                    raw = self._receive_text()
                except Exception:
                    self.close()
                    raise
        return VisionStudioReply(self.mode, raw, self._parse_reply(raw))

    def set_global_value(self, name: str, value: Any) -> None:
        if not name:
            raise ValueError("DobotVisionStudio 全局变量名不能为空")
        with self._lock:
            self.connect()
            if self.mode == "simulation":
                return
            self._send_text(f"SetGlobalValue:{name}={value}")

    def _send_text(self, message: str) -> None:
        encoding = str(self.settings.get("encoding", "utf-8"))
        terminator = str(self.settings.get("send_terminator", ""))
        payload = (message + terminator).encode(encoding)
        if self.serial_port is not None:
            self.serial_port.write(payload)
            self.serial_port.flush()
            return
        if self.sock is None:
            raise RuntimeError("DobotVisionStudio 通信连接未建立")
        self.sock.sendall(payload)

    def _receive_text(self) -> str:
        encoding = str(self.settings.get("encoding", "utf-8"))
        terminator = str(self.settings.get("receive_terminator", ""))
        max_bytes = int(self.settings.get("max_reply_bytes", 65536))
        expected = terminator.encode(encoding)
        if self.serial_port is not None:
            raw = (
                self.serial_port.read_until(expected=expected, size=max_bytes)
                if expected
                else self.serial_port.read(max_bytes)
            )
        elif self.sock is not None:
            chunks = bytearray()
            while len(chunks) < max_bytes:
                block = self.sock.recv(min(4096, max_bytes - len(chunks)))
                if not block:
                    break
                chunks.extend(block)
                if not expected or expected in chunks:
                    break
            raw = bytes(chunks)
        else:
            raise RuntimeError("DobotVisionStudio 通信连接未建立")
        if not raw:
            raise RuntimeError("DobotVisionStudio 未返回数据")
        text = raw.decode(encoding, errors="replace")
        if terminator and text.endswith(terminator):
            text = text[: -len(terminator)]
        return text.strip()

    def _parse_reply(self, raw: str) -> Any:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        protocol = self.settings.get("protocol", {})
        fields = protocol.get("response_fields", [])
        delimiter = str(protocol.get("delimiter", ","))
        if fields:
            if delimiter.strip().lower() in {"", "whitespace", "space", "\\s"}:
                values = raw.split()
            else:
                values = raw.split(delimiter)
            if len(values) != len(fields):
                raise RuntimeError(
                    f"DobotVisionStudio 返回字段数不匹配: expected={len(fields)} actual={len(values)} raw={raw}"
                )
            return {
                str(name): self._coerce_value(value)
                for name, value in zip(fields, values)
            }
        return raw

    @staticmethod
    def _coerce_value(value: str) -> Any:
        text = value.strip()
        try:
            return int(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return text
