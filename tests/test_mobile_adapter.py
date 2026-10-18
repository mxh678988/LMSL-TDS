"""MobileAdapter 单测：虚拟显存、注册/扫描、load/unload/prefetch、模拟模式。"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tds.config import TDSConfig  # noqa: E402
from tds.core.memory import VramState  # noqa: E402
from tds.adapters.mobile_adapter import MobileAdapter  # noqa: E402


class TestMobileAdapterVirtual(unittest.TestCase):
    def setUp(self):
        self.adapter = MobileAdapter(backend="ncnn")

    def test_vram_state_virtual(self):
        """虚拟显存状态正确。"""
        vs = self.adapter.vram_state()
        self.assertIsInstance(vs, VramState)
        self.assertEqual(vs.total_mb, 2048)
        self.assertEqual(vs.used_mb, 0)
        self.assertEqual(vs.free_mb, 2048)

    def test_register_and_lookup(self):
        """登记和查找模型句柄。"""
        mock_model = {"model_name": "mobilenet.onnx"}
        self.adapter.register_model("mobilenet:0", mock_model)
        self.assertEqual(self.adapter.model_for("mobilenet:0"), mock_model)
        self.assertIsNone(self.adapter.model_for("nonexistent"))

    def test_register_by_name(self):
        """按模型名反向登记。"""
        mock_model = {"model_name": "vit.onnx"}
        self.adapter.register_model("vit:0", mock_model)
        self.adapter.register_model_by_name("vit.onnx", mock_model)
        block = self.adapter._name_to_block.get("vit.onnx")
        self.assertIsNotNone(block)
        self.assertEqual(self.adapter.model_for(block), mock_model)

    def test_scan_empty_graph(self):
        """空图返回空列表。"""
        result = self.adapter.scan_usage(None)
        self.assertEqual(result, [])

        result = self.adapter.scan_usage("")
        self.assertEqual(result, [])

        result = self.adapter.scan_usage({})
        self.assertEqual(result, [])

    def test_scan_json_graph(self):
        """JSON 执行图正确解析。"""
        graph = {
            "graph": {
                "0": {"class_type": "MobileCkpt", "model_name": "mobilenet.onnx", "block_type": "ckpt"},
                "1": {"class_type": "MobileClip", "model_name": "clip_vit.onnx", "block_type": "clip"},
            },
            "edges": [],
        }
        result = self.adapter.scan_usage(graph)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["block_id"], "ckpt:mobilenet.onnx")
        self.assertEqual(result[1]["block_id"], "clip:clip_vit.onnx")

    def test_scan_csv_graph(self):
        """CSV 文本图正确解析。"""
        text = "mobilenet.onnx,ckpt,0\nclip_vit.onnx,clip,1"
        result = self.adapter.scan_usage(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["block_id"], "ckpt:mobilenet.onnx")

    def test_scan_invalid_graph(self):
        """无效输入返回空列表。"""
        result = self.adapter.scan_usage([1, 2, 3])  # 不支持的列表
        self.assertEqual(result, [])

        result = self.adapter.scan_usage(b"not a string")  # 字节
        self.assertEqual(result, [])


class TestMobileAdapterSimulate(unittest.TestCase):
    def setUp(self):
        self.adapter = MobileAdapter(backend="onnxrt")

    def test_simulate_load_and_unload(self):
        """模拟加载/卸载流程。"""
        self.assertTrue(self.adapter.simulate_load("model1.onnx", 512))
        vs = self.adapter.vram_state()
        self.assertEqual(vs.used_mb, 512)
        self.assertEqual(vs.free_mb, 2048 - 512)

        self.assertTrue(self.adapter.simulate_load("model2.onnx", 768))
        vs = self.adapter.vram_state()
        self.assertEqual(vs.used_mb, 1280)
        self.assertEqual(vs.free_mb, 2048 - 1280)

        self.assertTrue(self.adapter.simulate_unload("model1.onnx"))
        vs = self.adapter.vram_state()
        self.assertEqual(vs.used_mb, 768)
        self.assertEqual(vs.free_mb, 2048 - 768)

    def test_simulate_unload_nonexistent(self):
        """卸载不存在的模型返回 False。"""
        result = self.adapter.simulate_unload("nonexistent.onnx")
        self.assertFalse(result)

    def test_prefetch_calls_load(self):
        """prefetch 委托给 load。"""
        self.assertTrue(self.adapter.simulate_load("prefetch_model.onnx", 256))
        vs = self.adapter.vram_state()
        self.assertEqual(vs.used_mb, 256)


class TestMobileAdapterConfig(unittest.TestCase):
    def test_custom_config(self):
        """自定义 TDSConfig 被正确持有。"""
        config = TDSConfig(vram_budget_mb=4096)
        adapter = MobileAdapter(config=config, backend="mnn")
        self.assertEqual(adapter.cfg.vram_budget_mb, 4096)
        self.assertEqual(adapter.backend, "mnn")


if __name__ == "__main__":
    unittest.main()
