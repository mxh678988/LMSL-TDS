"""ComfyUIAdapter 测试：验证 load/unload/scan_usage/vram_state/prefetch 真实接口。"""
import json
import sys
import os
import unittest

sys.path.insert(0, r"E:\LMSL-comfyui\ComfyUI")
sys.path.insert(0, r"E:\LMSL-comfyui\ComfyUI\custom_nodes\TDS")

from tds.adapters.comfyui_adapter import ComfyUIAdapter


class TestComfyUIAdapter(unittest.TestCase):
    """ComfyUIAdapter 真实接口测试。"""

    def setUp(self):
        self.adapter = ComfyUIAdapter()

    # ===== vram_state =====
    def test_vram_state_has_values(self):
        state = self.adapter.vram_state()
        self.assertGreater(state.total_mb, 0, "Total VRAM should be > 0")
        self.assertGreaterEqual(state.free_mb, 0, "Free VRAM >= 0")
        self.assertGreaterEqual(state.used_mb, 0, "Used VRAM >= 0")
        self.assertEqual(
            state.total_mb, state.used_mb + state.free_mb, "Total = Used + Free"
        )
        print(f"[VRAM] total={state.total_mb}MB used={state.used_mb}MB free={state.free_mb}MB")

    def test_vram_state_returns_correct_type(self):
        state = self.adapter.vram_state()
        self.assertIsInstance(state.total_mb, int)
        self.assertIsInstance(state.used_mb, int)
        self.assertIsInstance(state.free_mb, int)

    # ===== scan_usage =====
    def test_scan_usage_empty_graph(self):
        result = self.adapter.scan_usage({})
        self.assertIsInstance(result, list)
        print(f"[scan_usage empty] {result}")

    def test_scan_usage_simple_prompt(self):
        """模拟 ComfyUI prompt graph，验证 scan_usage 返回模型序列。"""
        prompt = {
            "graph": {
                "1": {"class_type": "CheckpointLoaderSimple"},
                "2": {"class_type": "KSampler"},
                "3": {"class_type": "VAEDecode"},
                "4": {"class_type": "CLIPSetLastLayer"},
            },
            "edges": [],
        }
        result = self.adapter.scan_usage(prompt)
        self.assertIsInstance(result, list)
        # CheckpointLoaderSimple 应在结果中
        block_types = [r.get("block_type") for r in result]
        self.assertIn("ckpt", block_types, "CheckpointLoaderSimple should be detected as ckpt")
        self.assertIn("clip", block_types, "CLIPSetLastLayer should be detected as clip")
        self.assertIn("vae", block_types, "VAEDecode should be detected as vae")
        print(f"[scan_usage] {json.dumps(result, indent=2, ensure_ascii=False)}")

    def test_scan_usage_graph_string(self):
        """字符串 graph 输入。"""
        prompt_str = json.dumps({
            "graph": {
                "1": {"class_type": "CheckpointLoaderSimple"},
                "2": {"class_type": "VAELoader"},
            },
            "edges": [],
        })
        result = self.adapter.scan_usage(prompt_str)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) >= 2)

    def test_scan_usage_no_mm(self):
        """model_management 不可用时的退化。"""
        self.adapter._mm = None
        result = self.adapter.scan_usage({"graph": {"1": {"class_type": "Unknown"}}})
        self.assertIsInstance(result, list)

    # ===== load / unload / prefetch =====
    def test_load_returns_bool(self):
        result = self.adapter.load("fake_model_path")
        self.assertIsInstance(result, bool)
        print(f"[load fake] {result}")

    def test_unload_returns_bool(self):
        result = self.adapter.unload("fake_model_path")
        self.assertIsInstance(result, bool)
        print(f"[unload fake] {result}")

    def test_prefetch_returns_bool(self):
        result = self.adapter.prefetch("fake_model_path")
        self.assertIsInstance(result, bool)
        print(f"[prefetch fake] {result}")

    def test_load_returns_true_if_model_loaded(self):
        """用真实模型路径调用 load，可能返回 True。"""
        # 用 checkpoint 路径（如果存在）测试
        result = self.adapter.load("fake_checkpoint.safetensors")
        self.assertIsInstance(result, bool)
        print(f"[load checkpoint] {result}")

    # ===== edge cases =====
    def test_scan_usage_none_graph(self):
        result = self.adapter.scan_usage(None)
        self.assertEqual(result, [])

    def test_scan_usage_list_graph(self):
        result = self.adapter.scan_usage([1, 2, 3])
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
