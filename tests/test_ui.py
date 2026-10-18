"""tds.ui 单测：会话模型、附件校验、HTTP 接口行为（不依赖真实模型后端）。

后端一律指向 ``http://127.0.0.1:9``（不可达），以验证"后端离线时面板仍可加载、
接口按约定降级"的路径；真实模型链路由 tools 下的端到端脚本覆盖。
"""
from __future__ import annotations

import base64
import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tds.ui import UiServer  # noqa: E402
from tds.ui.server import _parse_data_uri  # noqa: E402
from tds.ui.session import (  # noqa: E402
    MAX_IMAGE_BYTES,
    SessionStore,
    make_attachment,
)

OFFLINE_BACKEND = "http://127.0.0.1:9"


def _png_bytes() -> bytes:
    """最小合法 1x1 PNG，用于附件测试。"""
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8AAAwAB/QFvGQAAAABJRU5ErkJggg=="
    )


class TestSession(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SessionStore()

    def test_create_and_require(self) -> None:
        session = self.store.create(model="m1", system="sys")
        self.assertEqual(session.model, "m1")
        self.assertEqual(session.system, "sys")
        self.assertIs(self.store.require(session.id), session)
        self.assertEqual(len(self.store), 1)

    def test_require_missing_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.store.require("nope")

    def test_title_from_first_message(self) -> None:
        """首次提问生成标题：18 字以内原样保留，超出则截断加省略号。"""
        short = "帮我看看这张图里有什么颜色和形状"
        session = self.store.create()
        session.add_user(short)
        self.assertEqual(session.title, short)

        long_text = "请逐条分析这张图片里出现的全部颜色、几何形状、相对位置关系以及拍摄场景"
        session2 = self.store.create()
        session2.add_user(long_text)
        self.assertEqual(session2.title, long_text[:18] + "…")
        self.assertEqual(len(session2.title), 19)

    def test_history_skips_error_messages(self) -> None:
        session = self.store.create()
        session.add_user("你好")
        bad = session.add_assistant("", {"model": "m"})
        bad.error = "连接失败"
        session.add_assistant("世界上有两种人", {"model": "m"})
        history = session.history_for_model()
        self.assertEqual(len(history), 2)
        self.assertTrue(all("连接失败" not in json.dumps(h, ensure_ascii=False) for h in history))

    def test_delete_and_clear(self) -> None:
        session = self.store.create()
        session.add_user("hi")
        session.clear()
        self.assertEqual(session.messages, [])
        self.assertTrue(self.store.delete(session.id))
        self.assertFalse(self.store.delete(session.id))
        self.assertEqual(len(self.store), 0)

    def test_summary_sorting(self) -> None:
        """列表按最近更新倒序；显式时间戳规避 Windows 时钟粒度影响。"""
        first = self.store.create()
        second = self.store.create()
        first.updated_at = 100.0
        second.updated_at = 200.0
        rows = self.store.list()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["id"], second.id)
        self.assertEqual(rows[1]["id"], first.id)
        self.assertEqual(rows[0]["count"], 0)
        self.assertEqual(set(rows[0]) , {"id", "title", "model", "count", "updated_at"})


class TestAttachments(unittest.TestCase):
    def test_make_attachment_ok(self) -> None:
        att = make_attachment("a.png", "image/png", _png_bytes())
        self.assertEqual(att.mime, "image/png")
        self.assertTrue(att.data_uri.startswith("data:image/png;base64,"))

    def test_reject_wrong_mime(self) -> None:
        with self.assertRaises(ValueError):
            make_attachment("a.txt", "text/plain", b"hello")

    def test_reject_oversize(self) -> None:
        with self.assertRaises(ValueError):
            make_attachment("big.png", "image/png", b"\x00" * (MAX_IMAGE_BYTES + 1))

    def test_parse_data_uri(self) -> None:
        uri = "data:image/png;base64," + base64.b64encode(_png_bytes()).decode("ascii")
        att = _parse_data_uri(uri, "x.png")
        self.assertEqual(att.name, "x.png")
        self.assertEqual(att.mime, "image/png")

    def test_parse_data_uri_bad_payload(self) -> None:
        with self.assertRaises(ValueError):
            _parse_data_uri("data:image/png;base64,!!!!")

    def test_multimodal_payload_shape(self) -> None:
        session = SessionStore().create()
        msg = session.add_user("看图", [make_attachment("a.png", "image/png", _png_bytes())])
        payload = msg.to_openai()
        self.assertEqual(payload["role"], "user")
        self.assertIsInstance(payload["content"], list)
        kinds = [p["type"] for p in payload["content"]]
        self.assertEqual(kinds, ["text", "image_url"])


class TestHttpApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = UiServer(port=0, backend_url=OFFLINE_BACKEND, verbose=False)
        cls.httpd = cls.server.build()
        cls.port = cls.httpd.server_address[1]
        cls.base = "http://127.0.0.1:{}".format(cls.port)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _request(self, path: str, payload=None, method: str = "GET"):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8")

    def test_panel_served(self) -> None:
        status, body = self._request("/")
        self.assertEqual(status, 200)
        self.assertIn("TDS 对话测试台", body)
        self.assertIn('id="messages"', body)

    def test_panel_has_no_cdn(self) -> None:
        _status, body = self._request("/")
        self.assertNotIn("http://cdn", body)
        self.assertNotIn("https://cdn", body)
        self.assertNotIn("unpkg.com", body)

    def test_status_when_backend_offline(self) -> None:
        status, body = self._request("/api/status")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertFalse(payload["backend"]["online"])
        self.assertIn("sessions", payload)

    def test_models_degrade_when_backend_offline(self) -> None:
        status, body = self._request("/api/models")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["models"], [])

    def test_session_lifecycle(self) -> None:
        status, body = self._request("/api/sessions", {"model": "m"}, "POST")
        self.assertEqual(status, 200)
        sid = json.loads(body)["session"]["id"]

        status, body = self._request("/api/sessions/" + sid)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["session"]["id"], sid)

        status, body = self._request(
            "/api/sessions/{}/params".format(sid),
            {"temperature": 1.2, "top_p": 0.5, "max_tokens": 128, "system": "sys"},
            "POST",
        )
        session = json.loads(body)["session"]
        self.assertEqual(session["temperature"], 1.2)
        self.assertEqual(session["top_p"], 0.5)
        self.assertEqual(session["max_tokens"], 128)

        status, body = self._request("/api/sessions/{}/clear".format(sid), None, "POST")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["session"]["messages"], [])

        status, _body = self._request("/api/sessions/" + sid, None, "DELETE")
        self.assertEqual(status, 200)

    def test_params_clamped(self) -> None:
        _status, body = self._request("/api/sessions", {}, "POST")
        sid = json.loads(body)["session"]["id"]
        _status, body = self._request(
            "/api/sessions/{}/params".format(sid),
            {"temperature": 99, "top_p": -3},
            "POST",
        )
        session = json.loads(body)["session"]
        self.assertEqual(session["temperature"], 2.0)
        self.assertEqual(session["top_p"], 0.01)

    def test_chat_unknown_session_404(self) -> None:
        status, body = self._request(
            "/api/chat", {"session_id": "missing", "content": "hi"}, "POST"
        )
        self.assertEqual(status, 404)
        self.assertIn("会话不存在", json.loads(body)["error"])

    def test_chat_empty_message_400(self) -> None:
        _status, body = self._request("/api/sessions", {"model": "m"}, "POST")
        sid = json.loads(body)["session"]["id"]
        status, body = self._request(
            "/api/chat", {"session_id": sid, "content": "   ", "images": []}, "POST"
        )
        self.assertEqual(status, 400)

    def test_unknown_route_404(self) -> None:
        status, _body = self._request("/api/nope")
        self.assertEqual(status, 404)

    def test_asset_traversal_blocked(self) -> None:
        status, _body = self._request("/assets/../server.py")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
