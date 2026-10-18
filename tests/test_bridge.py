"""桥接层单元测试（零第三方依赖，纯离线）。

覆盖：域判定 / 八卦加权等价性 / 四象压力门控 / 预取门控 / attach-detach 回退。
"""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tds.config import BAGUA_KEEP_STEPS, TDSConfig
from tds.core.memory import VramState
from tds.core.scheduler import PressureState
from tds.integrations.comfyui_bridge import (
    ComfyBridge,
    domain_of_key,
    find_smart_offload,
)


def _state(ratio: float) -> VramState:
    total = 10000
    return VramState(total_mb=total, used_mb=int(total * ratio), free_mb=total - int(total * ratio))


class FakeSmartOffload(types.ModuleType):
    """lmsl-smart-offload 的最小可信替身（接口取自真实源码）。"""

    def __init__(self, keep_steps: int = 1, distances=None):
        super().__init__("lmsl-smart-offload.smart_offload")
        self._distances = dict(distances or {})
        self.prefetch_calls = 0
        self.dist_calls = 0
        self.config = types.SimpleNamespace(keep_steps=keep_steps, prefetch_enabled=True)

    def _next_use_distance(self, key):
        self.dist_calls += 1
        return self._distances.get(key)

    def _maybe_prefetch(self):
        self.prefetch_calls += 1


class TestDomainMapping(unittest.TestCase):
    def test_video_models(self):
        self.assertEqual(domain_of_key("WanVideoModelLoader|wan2.1_t2v.safetensors"), "shipin")
        self.assertEqual(domain_of_key("LTXVConditioning|ltx-video.safetensors"), "shipin")

    def test_image_models(self):
        self.assertEqual(domain_of_key("CheckpointLoaderSimple|flux1-dev.safetensors"), "tuxiang")
        self.assertEqual(domain_of_key("LoraLoader|lora_style.safetensors"), "tuxiang")
        self.assertEqual(domain_of_key("ControlNetLoader|controlnet_pose.safetensors"), "tuxiang")

    def test_text_components(self):
        self.assertEqual(domain_of_key("CLIPLoader|t5xxl.safetensors"), "duihua")

    def test_unknown_falls_back_to_qita(self):
        self.assertEqual(domain_of_key("SomeWeirdNode|mystery.bin"), "qita")
        self.assertEqual(BAGUA_KEEP_STEPS["qita"], 1)

    def test_cache_is_consistent(self):
        b = ComfyBridge(vram_probe=lambda: None)
        self.assertEqual(b.domain_of("CheckpointLoaderSimple|flux.safetensors"), "tuxiang")
        self.assertEqual(b.domain_of("CheckpointLoaderSimple|flux.safetensors"), "tuxiang")
        self.assertEqual(len(b._domain_cache), 1)


class TestWeightedEquivalence(unittest.TestCase):
    """核心不变量：dist_eff <= base  <=>  dist <= ks_domain（gain=1）。"""

    def setUp(self):
        self.base_state = _state(0.30)  # LOW
        self.bridge = ComfyBridge(vram_probe=lambda: self.base_state)

    def test_equivalence_all_domains(self):
        keys = {
            "duihua": "CLIPLoader|t5xxl.safetensors",
            "tuxiang": "CheckpointLoaderSimple|flux1-dev.safetensors",
            "shipin": "WanVideoModelLoader|wan2.1.safetensors",
            "qita": "Unknown|mystery.bin",
        }
        for base in (1, 2, 3):
            for dom, key in keys.items():
                ks = BAGUA_KEEP_STEPS[dom]
                for dist in range(-2, 8):
                    dist_eff = self.bridge.effective_distance(dist, key, base=base)
                    self.assertEqual(
                        dist_eff <= base, dist <= ks,
                        f"dom={dom} base={base} dist={dist} ks={ks} eff={dist_eff}")

    def test_neutral_when_ks_equals_base(self):
        """base 与域基准一致时，返回值必须与原始实现逐位一致（零漂移）。"""
        for dom, key in {
            "duihua": "CLIPLoader|t5xxl.safetensors",
            "qita": "Unknown|mystery.bin",
        }.items():
            base = BAGUA_KEEP_STEPS[dom]
            for dist in range(0, 5):
                self.assertEqual(self.bridge.effective_distance(dist, key, base=base), dist)
        self.assertEqual(self.bridge.stats.get("keep.neutral", 0), 10)

    def test_gain_zero_is_passthrough(self):
        b = ComfyBridge(vram_probe=lambda: None, gain=0.0)
        key = "CLIPLoader|t5xxl.safetensors"   # duihua ks=3
        for dist in range(0, 5):
            self.assertEqual(b.effective_distance(dist, key, base=1), dist)


class TestFourSymbolsGate(unittest.TestCase):
    def _bridge(self, ratio, **kw):
        b = ComfyBridge(vram_probe=lambda: _state(ratio), **kw)
        mod = FakeSmartOffload(keep_steps=1, distances={"CLIPLoader|t5.safetensors": 2})
        b.attach(mod)
        return b, mod

    def test_low_pressure_protects(self):
        b, mod = self._bridge(0.30)
        self.assertEqual(b.pressure_state(), PressureState.LOW)
        # duihua ks=3, base=1 -> 有效距离 0 <= 1 -> 保护
        self.assertEqual(mod._next_use_distance("CLIPLoader|t5.safetensors"), 0)

    def test_critical_pressure_protects_but_blocks_prefetch(self):
        b, mod = self._bridge(0.88)
        self.assertEqual(b.pressure_state(), PressureState.CRITICAL)
        self.assertEqual(mod._next_use_distance("CLIPLoader|t5.safetensors"), 0)
        mod._maybe_prefetch()
        self.assertEqual(mod.prefetch_calls, 0)
        self.assertEqual(b.stats.get("prefetch.blocked_pressure"), 1)

    def test_high_pressure_gives_up_protection(self):
        b, mod = self._bridge(0.95)
        self.assertEqual(b.pressure_state(), PressureState.HIGH)
        self.assertIsNone(mod._next_use_distance("CLIPLoader|t5.safetensors"))
        self.assertEqual(b.stats.get("pressure_skip"), 1)
        mod._maybe_prefetch()
        self.assertEqual(mod.prefetch_calls, 0)

    def test_frozen_releases_everything(self):
        b, mod = self._bridge(0.99)
        self.assertEqual(b.pressure_state(), PressureState.FROZEN)
        self.assertIsNone(mod._next_use_distance("CLIPLoader|t5.safetensors"))

    def test_no_future_use_passthrough(self):
        b, mod = self._bridge(0.30)
        self.assertIsNone(mod._next_use_distance("not-in-plan"))
        self.assertEqual(b.stats.get("dist.no_future"), 1)

    def test_prefetch_allowed_only_when_gate_open_and_low(self):
        b, mod = self._bridge(0.30, config=TDSConfig(prefetch_enabled=True))
        mod._maybe_prefetch()
        self.assertEqual(mod.prefetch_calls, 1)
        self.assertEqual(b.stats.get("prefetch.allowed"), 1)

    def test_prefetch_default_gate_blocks(self):
        b, mod = self._bridge(0.30)   # TDSConfig.prefetch_enabled 默认 False
        mod._maybe_prefetch()
        self.assertEqual(mod.prefetch_calls, 0)
        self.assertEqual(b.stats.get("prefetch.blocked_gate"), 1)

    def test_vram_probe_failure_degrades_to_low(self):
        def boom():
            raise RuntimeError("no cuda")
        b = ComfyBridge(vram_probe=boom)
        mod = FakeSmartOffload(distances={"CheckpointLoaderSimple|flux.safetensors": 1})
        b.attach(mod)
        self.assertEqual(b.pressure_state(), PressureState.LOW)
        self.assertIsNotNone(mod._next_use_distance("CheckpointLoaderSimple|flux.safetensors"))


class TestRespectsWorkflowNodeConfig(unittest.TestCase):
    def test_base_follows_plugin_config(self):
        """工作流内 SmartResidentScheduler 把 keep_steps 设为 3 时，以 3 为基准。"""
        b = ComfyBridge(vram_probe=lambda: None)
        mod = FakeSmartOffload(keep_steps=3)
        b.attach(mod)
        self.assertEqual(b.base_keep_steps(), 3)

    def test_detach_restores_original(self):
        b = ComfyBridge(vram_probe=lambda: None)
        mod = FakeSmartOffload(keep_steps=3)
        b.attach(mod)
        self.assertIn("_next_use_distance", mod.__dict__)
        self.assertIn("_maybe_prefetch", mod.__dict__)
        b.detach()
        # 实例级临时注入被清除，回落至原始类方法（同一函数对象）
        self.assertNotIn("_next_use_distance", mod.__dict__)
        self.assertNotIn("_maybe_prefetch", mod.__dict__)
        self.assertIs(mod._next_use_distance.__func__, FakeSmartOffload._next_use_distance)
        self.assertEqual(mod._next_use_distance("CLIPLoader|t5.safetensors"), None)
        self.assertIsNone(b.module)

    def test_attach_is_idempotent(self):
        b = ComfyBridge(vram_probe=lambda: None)
        mod = FakeSmartOffload()
        b.attach(mod)
        first = mod._next_use_distance
        b.attach(mod)
        self.assertIs(mod._next_use_distance, first)
        self.assertEqual(b.stats.get("attach"), 1)

    def test_attach_rejects_incompatible_module(self):
        b = ComfyBridge(vram_probe=lambda: None)
        self.assertFalse(b.attach(types.ModuleType("not_smart_offload")))

    def test_snapshot(self):
        b = ComfyBridge(vram_probe=lambda: None)
        b.attach(FakeSmartOffload())
        snap = b.snapshot()
        self.assertTrue(snap["attached"])
        self.assertEqual(snap["base_keep_steps"], 1)
        self.assertFalse(snap["prefetch_gate"])


class TestModuleDiscovery(unittest.TestCase):
    def test_find_via_sys_modules(self):
        pkg = types.ModuleType("lmsl-smart-offload")
        pkg.smart_offload = FakeSmartOffload()
        sys.modules["lmsl-smart-offload"] = pkg
        try:
            self.assertIs(find_smart_offload(), pkg.smart_offload)
        finally:
            sys.modules.pop("lmsl-smart-offload", None)

    def test_no_host_returns_none(self):
        for name in ("lmsl-smart-offload", "lmsl_smart_offload"):
            sys.modules.pop(name, None)
        self.assertIsNone(find_smart_offload())


if __name__ == "__main__":
    unittest.main(verbosity=2)
