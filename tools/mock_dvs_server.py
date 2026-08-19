from __future__ import annotations

import argparse
import socket
import sys


def decode_escapes(value: str) -> str:
    output: list[str] = []
    escapes = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\"}
    index = 0
    while index < len(value):
        if value[index] == "\\" and index + 1 < len(value) and value[index + 1] in escapes:
            output.append(escapes[value[index + 1]])
            index += 2
        else:
            output.append(value[index])
            index += 1
    return "".join(output)


def receive_message(connection: socket.socket, terminator: bytes, max_bytes: int = 65536) -> bytes:
    data = bytearray()
    while len(data) < max_bytes:
        block = connection.recv(min(4096, max_bytes - len(data)))
        if not block:
            break
        data.extend(block)
        if not terminator or terminator in data:
            break
    return bytes(data)


def main() -> int:
    parser = argparse.ArgumentParser(description="本机 DobotVisionStudio TCP 模拟服务端")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=19001, help="监听端口，默认 19001")
    parser.add_argument("--count", type=int, default=1, help="响应次数，默认 1")
    parser.add_argument(
        "--response",
        action="append",
        default=[],
        help="返回文本，可重复指定；默认 OK,300,0,100",
    )
    parser.add_argument("--receive-terminator", default=r"\n", help=r"触发结束符，默认 \n")
    parser.add_argument("--send-terminator", default=r"\n", help=r"返回结束符，默认 \n")
    parser.add_argument("--proactive", action="store_true", help="不等待触发，连接后主动发送")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port 必须在 1~65535 之间")
    if not 1 <= args.count <= 100:
        parser.error("--count 必须在 1~100 之间")
    responses = args.response or ["OK,300,0,100"]
    receive_terminator = decode_escapes(args.receive_terminator).encode("utf-8")
    send_terminator = decode_escapes(args.send_terminator).encode("utf-8")

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.host, args.port))
    listener.listen(1)
    print(f"MOCK_DVS_READY {args.host}:{args.port}", flush=True)
    try:
        connection, address = listener.accept()
        print(f"CLIENT {address[0]}:{address[1]}", flush=True)
        with connection:
            connection.settimeout(30.0)
            for index in range(args.count):
                if not args.proactive:
                    request = receive_message(connection, receive_terminator)
                    if not request:
                        print("CLIENT_CLOSED", flush=True)
                        break
                    print(
                        f"REQUEST {index + 1}/{args.count} "
                        + request.decode("utf-8", errors="replace").strip(),
                        flush=True,
                    )
                response = responses[index % len(responses)]
                connection.sendall(response.encode("utf-8") + send_terminator)
                print(f"RESPONSE {index + 1}/{args.count} {response}", flush=True)
    except (OSError, socket.timeout) as exc:
        print(f"MOCK_DVS_ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        listener.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
