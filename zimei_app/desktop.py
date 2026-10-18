"""太易 · 紫梅桌面 —— 本地 GUI 外壳启动器（零第三方依赖）

以系统 Edge 的"应用模式"（--app）打开独立桌面窗口，
服务端由同目录 server.py 承载（默认 127.0.0.1:8771）。

用法：
    python desktop.py [--port 8771] [--width 1280] [--height 840]
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PORT = 8771
DEFAULT_HOST = "127.0.0.1"


def is_listening(host: str, port: int, timeout: float = 0.6) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def find_edge() -> str | None:
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    env_edge = os.environ.get("LMSL_EDGE", "")
    if env_edge and os.path.exists(env_edge):
        return env_edge
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def ensure_server(host: str, port: int) -> None:
    if is_listening(host, port):
        return
    py = sys.executable or "python"
    server = os.path.join(_HERE, "server.py")
    flags = 0
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        flags |= subprocess.CREATE_NO_WINDOW
    subprocess.Popen([py, server, "--host", host, "--port", str(port)],
                     cwd=_HERE, creationflags=flags)
    for _ in range(40):
        time.sleep(0.4)
        if is_listening(host, port):
            return
    raise RuntimeError(f"宿主服务启动失败：{host}:{port}")


def main() -> int:
    ap = argparse.ArgumentParser(description="太易 · 紫梅桌面本地 GUI 外壳")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=840)
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/"
    try:
        ensure_server(args.host, args.port)
    except RuntimeError as exc:
        print(f"[desktop] {exc}", file=sys.stderr)
        return 1

    edge = find_edge()
    if edge is None:
        import webbrowser
        webbrowser.open(url)
        print(f"[desktop] 未找到 Edge，已用默认浏览器打开 {url}", file=sys.stderr)
        return 0

    subprocess.Popen([
        edge,
        "--app=" + url,
        f"--window-size={args.width},{args.height}",
        "--window-position=120,80",
        "--disable-features=Translate",
    ])
    print(f"[desktop] 太易·紫梅桌面已启动（{url}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
