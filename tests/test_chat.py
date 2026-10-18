"""tds.chat 原生对话内核单测：模式切换 / 单轮 / 对照 / 压测 / 渲染 / CLI。

不连真实模型：用本地微型 HTTP 服务模拟 OpenAI 兼容 SSE 后端，
配合 FakeAdapter 驱动调度链路，覆盖 native 与 sdk 两种模式、
正常 / 报错 / HTTP 500 三条分支，以及单流与并发压测。
"""
from __future__ import annotations

import http.server
import json
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tds import TDSRuntime  # noqa: E402
from tds.chat import (  # noqa: E402
    ChatError,
    StreamClient,
    TDSChat,
    main,
    render_compare,
    render_stress,
)

OFFLINE_BACKEND = "http://127.0.0.1:9"


# --------------------------------------------------------------------------
# 测试替身：本地 SSE 后端 + 假适配器
# --------------------------------------------------------------------------


class _SseHandler(http.server.BaseHTTPRequestHandler):
    """模拟 OpenAI 兼容后端：ok / error / http500 三种模式。"""

    mode = "ok"

    def log_message(self, *args):  # 静音访问日志
        return

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith("/api/version"):
            self._send_json(200, {"version": "fake"})
        elif self.path.endswith("/v1/models"):
            self._send_json(200, {"data": [{"id": "fake-model"}]})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        if _SseHandler.mode == "http500":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def emit(obj):
            self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8"))
            self.wfile.flush()

        if _SseHandler.mode == "error":
            emit({"error": "model exploded"})
            return
        emit({"choices": [{"delta": {"content": "你"}}]})
        emit({"choices": [{"delta": {"content": "好"}}]})
        emit({"choices": [{"delta": {}, "finish_reason": "stop"}],
              "usage": {"completion_tokens": 2}})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class FakeVram:
    total_mb = 12288.0
    used_mb = 4096.0
    free_mb = 8192.0
    usage_mb = 4096.0

    @property
    def usage_ratio(self):
        return 0.33


class FakeAdapter:
    """满足 BaseAdapter 契约的替身。"""

    def scan_usage(self):
        return {}

    def load(self, *args, **kwargs):
        return True

    def unload(self, *args, **kwargs):
        return True

    def prefetch(self, *args, **kwargs):
        return True

    def vram_state(self):
        return FakeVram()


# --------------------------------------------------------------------------
# 测试用例
# --------------------------------------------------------------------------


class StreamClientTest(unittest.TestCase):
    def setUp(self):
        _SseHandler.mode = "ok"
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _SseHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = "http://127.0.0.1:{}".format(self.server.server_address[1])

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_ping_online(self):
        online, note = StreamClient(self.base).ping()
        self.assertTrue(online)
        self.assertIn("/api/version", note)

    def test_ping_offline(self):
        online, _ = StreamClient(OFFLINE_BACKEND).ping()
        self.assertFalse(online)

    def test_stream_ok(self):
        chunks = list(StreamClient(self.base).stream("fake", [{"role": "user", "content": "hi"}]))
        text = "".join(c.get("text", "") for c in chunks)
        self.assertEqual(text, "你好")
        self.assertTrue(any(c.get("usage") for c in chunks))

    def test_stream_error(self):
        _SseHandler.mode = "error"
        chunks = list(StreamClient(self.base).stream("fake", [{"role": "user", "content": "hi"}]))
        self.assertTrue(any(c.get("error") for c in chunks))

    def test_stream_http500(self):
        _SseHandler.mode = "http500"
        with self.assertRaises(ChatError):
            list(StreamClient(self.base).stream("fake", [{"role": "user", "content": "hi"}]))

    def test_stream_offline_raises(self):
        with self.assertRaises(ChatError):
            list(StreamClient(OFFLINE_BACKEND).stream("fake", [{"role": "user", "content": "hi"}]))


class TDSChatTest(unittest.TestCase):
    def setUp(self):
        _SseHandler.mode = "ok"
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _SseHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = "http://127.0.0.1:{}".format(self.server.server_address[1])
        self.runtime = TDSRuntime()
        self.runtime.register_adapter(FakeAdapter())

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_mode_switch(self):
        chat = TDSChat(self.base, model="fake", runtime=self.runtime)
        self.assertEqual(chat.mode, "native")
        self.runtime.attach()
        self.assertEqual(chat.mode, "sdk")
        chat.detach_scheduler()
        self.assertEqual(chat.mode, "native")

    def test_send_native(self):
        chat = TDSChat(self.base, model="fake")
        turn = chat.send("你好")
        self.assertEqual(turn.error, "")
        self.assertEqual(turn.content, "你好")
        self.assertEqual(turn.stats.mode, "native")
        self.assertEqual(turn.stats.dispatch_ms, 0.0)
        self.assertEqual(len(chat.history), 2)

    def test_send_sdk_records_trace(self):
        self.runtime.attach()
        chat = TDSChat(self.base, model="fake", runtime=self.runtime)
        turn = chat.send("写一段调度测试")
        self.assertEqual(turn.stats.mode, "sdk")
        self.assertIn("ok", turn.stats.trace_line)
        self.assertGreater(turn.stats.dispatch_ms, 0.0)
        self.assertEqual(turn.stats.vram_mb, 4096.0)

    def test_send_error_isolation(self):
        chat = TDSChat(self.base, model="fake")
        chat.client.stream = lambda *a, **k: iter([{"error": "bad model"}])
        turn = chat.send("hi")
        self.assertIn("bad model", turn.error)
        self.assertEqual(chat.history, [])  # 失败轮次不留在历史里

    def test_send_requires_model(self):
        chat = TDSChat(self.base)
        with self.assertRaises(ChatError):
            chat.send("hi")

    def test_compare_modes(self):
        self.runtime.attach()
        chat = TDSChat(self.base, model="fake", runtime=self.runtime)
        result = chat.compare(["a", "b"])
        self.assertEqual(result.prompts, 2)
        self.assertEqual(result.native["failed"], 0)
        self.assertEqual(result.sdk["failed"], 0)
        self.assertIn("avg_first_ms", result.delta)
        self.assertTrue(self.runtime.is_attached())  # 对照后模式已还原
        self.assertIn("性能对照", render_compare(result))

    def test_stress_single(self):
        chat = TDSChat(self.base, model="fake")
        result = chat.stress(num_tasks=3, concurrency=1)
        self.assertEqual(result.mode, "single")
        self.assertEqual(result.failed, 0)
        self.assertEqual(result.tasks, 3)
        self.assertEqual(len(result.details), 3)

    def test_stress_concurrent(self):
        chat = TDSChat(self.base, model="fake")
        result = chat.stress(num_tasks=4, concurrency=2)
        self.assertEqual(result.mode, "concurrent")
        self.assertEqual(result.failed, 0)
        self.assertIn("压测结果", render_stress(result))

    def test_cli_offline_returns_1(self):
        self.assertEqual(main(["--base-url", OFFLINE_BACKEND, "--model", "fake"]), 1)

    def test_cli_requires_model(self):
        self.assertEqual(main(["--base-url", self.base]), 1)


if __name__ == "__main__":
    unittest.main()
