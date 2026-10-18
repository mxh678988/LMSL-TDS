"""测试台会话落盘 / 恢复单元测试（标准库 unittest，零第三方依赖）。"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tds.ui.session import (
    SNAPSHOT_VERSION,
    ChatSession,
    SessionStore,
    make_attachment,
)

PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6360000002000100f3a4b6a60000000049454e44ae426082"
)


def build_store() -> SessionStore:
    store = SessionStore()
    session = store.create(model="qwen2.5vl:7b", system="你是测试助手")
    session.temperature = 0.0
    session.max_tokens = 256
    session.add_user("这是什么图？", [make_attachment("a.png", "image/png", PNG_1X1)])
    session.add_assistant("一只猫", stats={"latency_ms": 812.5, "tokens": 4})
    session.add_user("再总结一次")
    session.add_assistant("猫", stats={"latency_ms": 120.0, "tokens": 1})
    return store


class TestSessionStoreSnapshot(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_save_creates_json_with_version(self):
        store = build_store()
        path = store.save(self.root / "snap.json")
        self.assertTrue(Path(path).exists())
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertEqual(raw["version"], SNAPSHOT_VERSION)
        self.assertIn("saved_at", raw)
        self.assertEqual(len(raw["sessions"]), 1)
        self.assertEqual(len(raw["sessions"][0]["messages"]), 4)

    def test_roundtrip_restores_fields_and_images(self):
        store = build_store()
        path = store.save(self.root / "snap.json")

        restored = SessionStore()
        count = restored.load(path)
        self.assertEqual(count, 1)
        self.assertEqual(len(restored), 1)

        sid = store.list()[0]["id"]
        before = store.get(sid)
        after = restored.get(sid)
        self.assertIsNotNone(after)
        self.assertEqual(after.model, "qwen2.5vl:7b")
        self.assertEqual(after.system, "你是测试助手")
        self.assertEqual(after.temperature, 0.0)
        self.assertEqual(after.max_tokens, 256)
        self.assertEqual(after.title, before.title)
        self.assertEqual(len(after.messages), 4)
        self.assertEqual(after.messages[0].images[0].data_uri, before.messages[0].images[0].data_uri)
        self.assertEqual(after.messages[0].images[0].size, len(PNG_1X1))
        self.assertEqual(after.messages[1].stats["tokens"], 4)
        self.assertEqual(restored.list()[0]["id"], sid)

    def test_roundtrip_preserves_openai_protocol_shape(self):
        store = build_store()
        path = store.save(self.root / "snap.json")
        restored = SessionStore()
        restored.load(path)
        sid = store.list()[0]["id"]
        history = restored.get(sid).history_for_model()
        self.assertEqual(history[0]["role"], "user")
        self.assertIsInstance(history[0]["content"], list)  # 带图 → content 数组
        self.assertEqual(history[0]["content"][1]["type"], "image_url")
        self.assertEqual(history[1]["content"], "一只猫")

    def test_with_content_false_strips_image_bytes(self):
        store = build_store()
        path = store.save(self.root / "meta.json", with_content=False)
        restored = SessionStore()
        restored.load(path)
        sid = store.list()[0]["id"]
        img = restored.get(sid).messages[0].images[0]
        self.assertEqual(img.data, "")
        self.assertEqual(img.size, len(PNG_1X1))
        self.assertEqual(img.mime, "image/png")

    def test_suffix_autofilled_and_no_tmp_left(self):
        store = build_store()
        path = Path(store.save(self.root / "nosuffix"))
        self.assertEqual(path.suffix, ".json")
        self.assertTrue(path.exists())
        leftovers = list(path.parent.glob("*.tmp"))
        self.assertEqual(leftovers, [])

    def test_nested_directory_is_created(self):
        store = build_store()
        path = Path(store.save(self.root / "deep" / "nested" / "snap.json"))
        self.assertTrue(path.exists())

    def test_load_replaces_or_merges(self):
        path = build_store().save(self.root / "snap.json")

        store = SessionStore()
        store.create(model="qwen2.5:3b")
        self.assertEqual(len(store), 1)
        self.assertEqual(store.load(path), 1)
        self.assertEqual(len(store), 1)  # 默认替换
        self.assertEqual(store.list()[0]["model"], "qwen2.5vl:7b")

        store.create(model="qwen2.5:3b")  # 追加一条不同 id 的会话
        self.assertEqual(len(store), 2)
        self.assertEqual(store.load(path, merge=True), 1)
        self.assertEqual(len(store), 2)  # 保留新会话，快照同 id 覆盖

    def test_from_dict_rejects_non_object(self):
        store = SessionStore()
        with self.assertRaises(ValueError):
            store.from_dict([1, 2, 3])

    def test_load_missing_file_raises(self):
        with self.assertRaises(OSError):
            SessionStore().load(self.root / "nope.json")

    def test_tolerant_to_partial_records(self):
        store = SessionStore()
        restored = store.from_dict({
            "version": SNAPSHOT_VERSION,
            "sessions": [{
                "model": "m",
                "messages": [{"role": "user"}, {"role": "assistant", "images": [{"mime": "image/png"}]}],
            }],
        })
        self.assertEqual(restored, 1)
        session = ChatSession.from_dict({"messages": []})
        self.assertTrue(session.id)
        self.assertEqual(session.messages, [])

    def test_empty_store_snapshot_roundtrip(self):
        store = SessionStore()
        path = store.save(self.root / "empty.json")
        self.assertEqual(SessionStore().load(path), 0)


if __name__ == "__main__":
    unittest.main()
