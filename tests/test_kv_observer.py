# -*- coding: utf-8 -*-
r"""Phase 2 A 段 · KV 可观测层单元测试（标准库 unittest，零第三方依赖）。

覆盖：数据模型 / dtype 解析 / 包装器转发与记录 / 异常隔离 /
报告聚合 / 开关与环境变量 / bridge 集成。

运行：
    E:\LMSL-comfyui\python\python.exe -m unittest tests.test_kv_observer -v
"""
import os
import sys
import types
import unittest
from unittest import mock

# ---- 注入 TDS 根目录，保证 `import tds.*` 可用 ----
_TDS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TDS_ROOT not in sys.path:
    sys.path.insert(0, _TDS_ROOT)

from tds.kv import observer as obs  # noqa: E402


class FakeTensor:
    """最小张量替身：仅提供 numel / element_size。"""

    def __init__(self, numel, esize):
        self._numel = numel
        self._esize = esize

    def numel(self):
        return self._numel

    def element_size(self):
        return self._esize


class FakeFixedKV:
    """FixedKV 替身（含 key/value 属性）。"""

    def __init__(self, key, value):
        self.key = key
        self.value = value


def make_self(head_dim=128, num_kv_heads=32, fixed_kv=False, name="Llama2_"):
    return types.SimpleNamespace(
        config=types.SimpleNamespace(head_dim=head_dim, num_key_value_heads=num_kv_heads),
        fixed_kv=fixed_kv,
        __class__=type(name, (), {}),
    )


class KVObserverBase(unittest.TestCase):
    """每个用例前后重置观测状态与开关。"""

    def setUp(self):
        obs.reset_report()
        obs._KV_ENABLED = True

    def tearDown(self):
        obs.reset_report()
        obs._KV_ENABLED = True
        obs._KV_STATE["patched"] = False


# --------------------------------------------------------------------------
# 1. 数据模型
# --------------------------------------------------------------------------

class TestDataModel(KVObserverBase):

    def test_layer_record_defaults(self):
        rec = obs.KVLayerRecord(
            layer_id=0, capacity_tokens=512, dtype="torch.float16",
            device="cuda:0", batch=1, num_kv_heads=32, head_dim=128,
            bytes_allocated=1024,
        )
        self.assertEqual(rec.layer_id, 0)
        self.assertEqual(rec.use_count, 0)
        self.assertEqual(rec.last_used_step, 0)
        self.assertFalse(rec.is_fixed_kv)
        self.assertIsNone(rec.page_id)

    def test_report_defaults_and_summary_text(self):
        rep = obs.KVObservationReport()
        self.assertEqual(rep.layers, [])
        self.assertEqual(rep.total_bytes_allocated, 0)
        self.assertEqual(rep.total_capacity_tokens, 0)
        self.assertEqual(rep.layers_by_device, {})
        self.assertEqual(rep.layers_by_dtype, {})
        self.assertEqual(rep.layers_by_fixed_kv, {})
        lines = rep.summary_lines
        self.assertTrue(any("0 层" in ln for ln in lines))
        self.assertTrue(any("总容量" in ln for ln in lines))
        self.assertTrue(any("总分配" in ln for ln in lines))


# --------------------------------------------------------------------------
# 2. dtype 解析
# --------------------------------------------------------------------------

class TestDtypeBytes(KVObserverBase):

    def test_dtype_bytes_mapping(self):
        cases = [
            ("torch.float16", 2), ("torch.bfloat16", 2), ("torch.float32", 4),
            ("torch.float64", 8), ("float16", 2), ("int8", 1),
            ("half", 2), ("unknown_dtype", 0),
        ]
        for dtype, expect in cases:
            with self.subTest(dtype=dtype):
                self.assertEqual(obs._dtype_bytes(dtype), expect)

    def test_tensor_bytes(self):
        self.assertEqual(obs._tensor_bytes(FakeTensor(1000, 2)), 2000)
        self.assertEqual(obs._tensor_bytes(object()), 0)
        self.assertEqual(obs._tensor_bytes(None), 0)

    def test_layer_bytes_measured_and_formula(self):
        layer = FakeFixedKV(FakeTensor(512, 2), FakeTensor(512, 2))
        self.assertEqual(obs._layer_bytes(layer, 1, 512, 128, 32, "torch.float16"), 2048)

        expect = 2 * 128 * 32 * 2 * 512
        self.assertEqual(
            obs._layer_bytes("not_a_tensor", 1, 512, 128, 32, "torch.float16"), expect)

        tup = (FakeTensor(256, 2), FakeTensor(256, 2), 0)
        self.assertEqual(
            obs._layer_bytes(tup, 1, 256, 128, 32, "torch.float16"), 1024)


# --------------------------------------------------------------------------
# 3. 包装器：转发与返回
# --------------------------------------------------------------------------

class TestWrapperForwarding(KVObserverBase):

    def test_wrapper_forwards_call_and_returns_result(self):
        """核心回归：原始调用必须被真实执行，返回值原样透传。"""
        sentinel = ["layerA", "layerB"]
        original = mock.MagicMock(return_value=sentinel)
        wrapper = obs._make_kv_observer_wrapper(original)
        s = make_self()

        out = wrapper(s, 2, 512, "cuda:0", "torch.float16")

        original.assert_called_once_with(s, 2, 512, "cuda:0", "torch.float16")
        self.assertIs(out, sentinel)

    def test_wrapper_records_layers(self):
        original = mock.MagicMock(
            return_value=[types.SimpleNamespace(), types.SimpleNamespace()])
        wrapper = obs._make_kv_observer_wrapper(original)
        s = make_self(num_kv_heads=32, head_dim=128)

        wrapper(s, 1, 512, "cuda:0", "torch.float16")

        rep = obs.get_report()
        self.assertIsNotNone(rep)
        self.assertEqual(len(rep.layers), 2)
        self.assertEqual(rep.layers_by_device, {"cuda:0": 2})
        self.assertEqual(rep.layers_by_dtype, {"torch.float16": 2})
        self.assertEqual(rep.total_capacity_tokens, 512 * 2)
        expected_per_layer = 2 * 128 * 32 * 2 * 512
        self.assertEqual(rep.total_bytes_allocated, expected_per_layer * 2)
        self.assertEqual(rep.layers[0].bytes_allocated, expected_per_layer)

    def test_wrapper_marks_fixed_kv(self):
        fixed = FakeFixedKV(FakeTensor(512, 2), FakeTensor(512, 2))
        original = mock.MagicMock(return_value=[fixed])
        wrapper = obs._make_kv_observer_wrapper(original)
        wrapper(make_self(fixed_kv=False), 1, 512, "cuda:0", "torch.float16")

        rep = obs.get_report()
        self.assertTrue(rep.layers[0].is_fixed_kv)
        self.assertEqual(rep.layers_by_fixed_kv, {True: 1})

    def test_wrapper_empty_layers(self):
        original = mock.MagicMock(return_value=[])
        wrapper = obs._make_kv_observer_wrapper(original)
        out = wrapper(make_self(), 1, 512, "cpu", "torch.float32")

        self.assertEqual(out, [])
        rep = obs.get_report()
        self.assertEqual(rep.layers, [])
        self.assertEqual(rep.total_bytes_allocated, 0)
        self.assertEqual(rep.total_capacity_tokens, 0)


# --------------------------------------------------------------------------
# 4. 异常隔离（零侵入底线）
# --------------------------------------------------------------------------

class TestIsolation(KVObserverBase):

    def test_observation_failure_does_not_break_host(self):
        sentinel = ["ok"]
        original = mock.MagicMock(return_value=sentinel)
        wrapper = obs._make_kv_observer_wrapper(original)

        with mock.patch.object(obs, "_record", side_effect=RuntimeError("boom")), \
             mock.patch.object(obs.LOG, "exception"):
            out = wrapper(make_self(), 1, 512, "cuda:0", "torch.float16")

        self.assertIs(out, sentinel)
        original.assert_called_once()

    def test_host_exception_propagates(self):
        original = mock.MagicMock(side_effect=ValueError("host error"))
        wrapper = obs._make_kv_observer_wrapper(original)

        with self.assertRaises(ValueError):
            wrapper(make_self(), 1, 512, "cuda:0", "torch.float16")

    def test_disabled_skips_recording(self):
        obs._KV_ENABLED = False
        original = mock.MagicMock(return_value=["x"])
        wrapper = obs._make_kv_observer_wrapper(original)

        out = wrapper(make_self(), 1, 512, "cuda:0", "torch.float16")

        self.assertEqual(out, ["x"])
        self.assertEqual(obs.get_report().layers, [])


# --------------------------------------------------------------------------
# 5. 报告聚合
# --------------------------------------------------------------------------

class TestReport(KVObserverBase):

    def test_report_aggregates_multiple_devices(self):
        original = mock.MagicMock(return_value=[types.SimpleNamespace()])
        wrapper = obs._make_kv_observer_wrapper(original)
        s = make_self()

        wrapper(s, 1, 256, "cuda:0", "torch.float16")
        wrapper(s, 1, 256, "cuda:1", "torch.float32")
        wrapper(s, 1, 256, "cuda:0", "torch.float16")

        rep = obs.get_report()
        self.assertEqual(rep.layers_by_device, {"cuda:0": 2, "cuda:1": 1})
        self.assertEqual(rep.layers_by_dtype, {"torch.float16": 2, "torch.float32": 1})
        self.assertEqual(rep.total_capacity_tokens, 256 * 3)

    def test_reset_report_clears_data(self):
        obs._record(make_self(), 1, 128, "cuda:0", "torch.float16", ["a"])
        self.assertEqual(len(obs.get_report().layers), 1)

        obs.reset_report()
        self.assertEqual(obs.get_report().layers, [])
        self.assertEqual(obs.get_report().total_bytes_allocated, 0)

    def test_get_summary_text(self):
        # 从未记录 -> None
        obs._KV_STATE["report"] = None
        self.assertEqual(obs.get_summary_text(), "无观测数据")

        # 已重置 -> 空报告
        obs.reset_report()
        self.assertIn("KV 观测报告", obs.get_summary_text())


# --------------------------------------------------------------------------
# 6. 开关与环境变量
# --------------------------------------------------------------------------

class TestSwitch(KVObserverBase):

    def test_enable_disable_triggers_patch_hooks(self):
        with mock.patch.object(obs, "_ensure_wrapped") as m_wrap, \
             mock.patch.object(obs, "_ensure_unwrapped") as m_unwrap:
            obs.enable()
            self.assertTrue(obs.is_enabled())
            m_wrap.assert_called_once()

            obs.disable()
            self.assertFalse(obs.is_enabled())
            m_unwrap.assert_called_once()

    def test_ensure_wrapped_idempotent_on_patched_flag(self):
        obs._KV_STATE["patched"] = True
        try:
            obs._ensure_wrapped()  # 已 patched，应直接返回
        finally:
            obs._KV_STATE["patched"] = False

    def test_env_enabled_parsing(self):
        cases = [("off", False), ("0", False), ("false", False),
                 ("disable", False), ("observe", True), ("page", True),
                 ("ON", True), ("", True)]
        for value, expect in cases:
            with self.subTest(value=value):
                with mock.patch.dict(os.environ, {"TDS_KV": value}, clear=False):
                    self.assertEqual(obs._env_enabled(), expect)

    def test_env_default_is_enabled(self):
        env = dict(os.environ)
        env.pop("TDS_KV", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(obs._env_enabled())


# --------------------------------------------------------------------------
# 7. bridge 集成
# --------------------------------------------------------------------------

class TestBridgeIntegration(KVObserverBase):

    def test_bridge_activates_kv_engine(self):
        from tds.integrations import comfyui_bridge as bridge_mod
        from tds.kv import activate

        bridge_mod._KV_ENGINE_ACTIVE = False
        try:
            with mock.patch.dict(os.environ, {"TDS_KV": "observe"}):
                with mock.patch("tds.kv.observer.enable") as m_enable:
                    bridge_mod._try_activate_kv()
                    m_enable.assert_called_once()
                    self.assertTrue(bridge_mod._KV_ENGINE_ACTIVE)

                # 幂等：再次调用不应重复开闸
                with mock.patch("tds.kv.observer.enable") as m_enable2:
                    bridge_mod._try_activate_kv()
                    m_enable2.assert_not_called()
        finally:
            bridge_mod._KV_ENGINE_ACTIVE = False
            activate("off")


if __name__ == "__main__":
    unittest.main(verbosity=2)
