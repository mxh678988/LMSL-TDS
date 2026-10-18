"""基础单元测试（标准库 unittest，零第三方依赖）。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tds.config import BAGUA_DOMAINS, BAGUA_KEEP_STEPS, TDSConfig, YANG, YIN
from tds.core.classifier import BaguaClassifier
from tds.core.memory import MemoryManager, VramState
from tds.core.planner import Planner
from tds.core.scheduler import Scheduler, PressureState
from tds.core.storage import TieredStore, Block, Tier
from tds.adapters.base import BaseAdapter
from tds.adapters import ComfyUIAdapter, LlamaCppAdapter, MobileAdapter


class MockAdapter(BaseAdapter):
    """离线适配器：记录调用，不触碰真实 GPU。"""

    def __init__(self, state=None):
        self._state = state or VramState(total_mb=12288, used_mb=4000, free_mb=8288)
        self.calls = []

    def scan_usage(self, graph):
        return ["m1", "m2"]

    def load(self, block_id):
        self.calls.append(("load", block_id))
        return True

    def unload(self, block_id):
        self.calls.append(("unload", block_id))
        return True

    def prefetch(self, block_id):
        self.calls.append(("prefetch", block_id))
        return True

    def vram_state(self):
        return self._state


class TestClassifier(unittest.TestCase):
    def test_eight_domains_defined(self):
        self.assertEqual(len(BAGUA_DOMAINS), 8)

    def test_image_domain(self):
        p = BaguaClassifier().classify("帮我生成一张图片")
        self.assertEqual(p.primary, "tuxiang")

    def test_code_domain(self):
        p = BaguaClassifier().classify("这段 python 代码报错了")
        self.assertEqual(p.primary, "daima")

    def test_fallback(self):
        p = BaguaClassifier().classify("嗯")
        self.assertEqual(p.primary, "qita")


class TestMemory(unittest.TestCase):
    def test_pressure(self):
        cfg = TDSConfig(pressure_ratio=0.9)
        mm = MemoryManager(cfg)
        state = VramState(total_mb=10000, used_mb=9500)
        self.assertTrue(mm.under_pressure(state))
        state2 = VramState(total_mb=10000, used_mb=5000)
        self.assertFalse(mm.under_pressure(state2))

    def test_budget(self):
        cfg = TDSConfig(vram_budget_mb=11000, reserve_mb=1500)
        self.assertEqual(MemoryManager(cfg).effective_budget(), 9500)


class TestStorage(unittest.TestCase):
    def test_tier_transitions(self):
        store = TieredStore()
        b = Block(id="layer0", size_mb=512, tier=Tier.SSD)
        store.add(b)
        store.promote("layer0")
        self.assertEqual(store.get("layer0").tier, Tier.RAM)
        store.promote("layer0")
        self.assertEqual(store.get("layer0").tier, Tier.VRAM)
        store.evict("layer0")
        self.assertEqual(store.get("layer0").tier, Tier.SSD)

    def test_pinned_not_evicted(self):
        store = TieredStore()
        b = Block(id="sys", size_mb=64, tier=Tier.VRAM, pinned=True)
        store.add(b)
        store.evict("sys")
        self.assertEqual(store.get("sys").tier, Tier.VRAM)


class TestPlanner(unittest.TestCase):
    def test_rank_by_next_use(self):
        planner = Planner(TDSConfig())
        ranked = planner.rank_by_next_use(["a", "b", "c"], {"a": 0, "b": 0, "c": 0})
        self.assertEqual(len(ranked), 3)


class TestScheduler(unittest.TestCase):
    def test_pressure_states(self):
        """四象压力策略：低/临界/高/冻结"""
        sched = Scheduler()
        low_state = VramState(total_mb=12288, used_mb=4000, free_mb=8288)
        critical_state = VramState(total_mb=12288, used_mb=11000, free_mb=1288)
        high_state = VramState(total_mb=12288, used_mb=11500, free_mb=788)

        self.assertEqual(sched.pressure_state(low_state), PressureState.LOW)
        self.assertEqual(sched.pressure_state(critical_state), PressureState.CRITICAL)
        self.assertEqual(sched.pressure_state(high_state), PressureState.HIGH)

    def test_pressure_actions(self):
        """四象行为模板：低=预取，临界=保活，高=跳过"""
        sched = Scheduler()
        self.assertEqual(sched.pressure_action(PressureState.LOW, []), "prefetch")
        self.assertEqual(sched.pressure_action(PressureState.CRITICAL, []), "hold")
        self.assertEqual(sched.pressure_action(PressureState.HIGH, []), "skip")
        self.assertEqual(sched.pressure_action(PressureState.FROZEN, []), "evict_all")

    def test_bagua_keep_steps(self):
        """八卦加权动态 keep_steps：对话域=3，工具域=1"""
        sched = Scheduler()
        self.assertEqual(sched.keep_steps_for_domain("duihua"), 3)
        self.assertEqual(sched.keep_steps_for_domain("gongju"), 1)
        self.assertEqual(sched.keep_steps_for_domain("qita"), 1)

    def test_yin_yang_tag_dynamic(self):
        """阴阳标记：基于八卦加权 keep_steps，而非硬编码"""
        sched = Scheduler()
        b_yang = Block(id="dialogue", size_mb=512, domain="duihua")
        b_yang.last_used_step = sched.step  # 当前步，距离0
        b_yin = Block(id="cached", size_mb=512, domain="duihua")
        b_yin.last_used_step = sched.step - 10  # 距离10，远超 keep_steps=3

        self.assertEqual(sched.yin_yang_tag(b_yang), YANG)
        self.assertEqual(sched.yin_yang_tag(b_yin), YIN)

    def test_classifier_integration(self):
        """分类器集成测试：分类→八卦加权→阴阳标记"""
        sched = Scheduler()
        profile = sched.on_task("帮我生成一张图片，然后用代码处理")
        self.assertEqual(profile.primary, "tuxiang")

        b = Block(id="img_gen", size_mb=1024, domain=profile.primary)
        b.last_used_step = sched.step
        self.assertEqual(sched.yin_yang_tag(b), YANG)


class TestAdapters(unittest.TestCase):
    def test_all_adapters_importable(self):
        """适配层可导入（此前的 ..memory 错误引用会导致 ImportError）"""
        for cls in (ComfyUIAdapter, LlamaCppAdapter, MobileAdapter):
            self.assertTrue(issubclass(cls, BaseAdapter))

    def test_comfyui_vram_state_readonly(self):
        """显存快照只读；无 GPU 时返回 0 而非抛异常"""
        state = ComfyUIAdapter().vram_state()
        self.assertGreaterEqual(state.total_mb, 0)
        self.assertGreaterEqual(state.used_mb, 0)


class TestPlannerPlan(unittest.TestCase):
    def test_yang_pinned_yin_demoted(self):
        """阴阳分层：阳块驻留，阴块下沉。prefetch 受总闸门控（问题6）"""
        cfg = TDSConfig()
        planner = Planner(cfg)
        sched = Scheduler(cfg)
        b_yang = Block(id="hot", size_mb=512, domain="duihua", last_used_step=10)
        b_yin = Block(id="cold", size_mb=512, domain="duihua", last_used_step=0)

        # duihua 域 keep_steps=3：distance 0 → 阳；distance 10 → 阴
        plan = planner.plan(None, blocks=[b_yang, b_yin], pressure="low",
                            current_step=10, keep_steps_fn=sched.keep_steps_for_domain)
        actions = {(it.block_id, it.action) for it in plan.items}
        self.assertIn(("hot", "pin"), actions)
        self.assertIn(("cold", "demote"), actions)
        # 问题6：默认 prefetch_enabled=False，计划层不产出 prefetch
        self.assertNotIn(("hot", "prefetch"), actions)

    def test_prefetch_only_when_enabled(self):
        """问题6：prefetch 总闸开启后，低压力阳块才产出 prefetch"""
        cfg = TDSConfig(prefetch_enabled=True)
        planner = Planner(cfg)
        sched = Scheduler(cfg)
        b_yang = Block(id="hot", size_mb=512, domain="duihua", last_used_step=10)
        plan = planner.plan(None, blocks=[b_yang], pressure="low",
                            current_step=10, keep_steps_fn=sched.keep_steps_for_domain)
        actions = {(it.block_id, it.action) for it in plan.items}
        self.assertIn(("hot", "pin"), actions)
        self.assertIn(("hot", "prefetch"), actions)
        # 非低压力窗口不预取
        plan_high = planner.plan(None, blocks=[b_yang], pressure="high",
                                 current_step=10, keep_steps_fn=sched.keep_steps_for_domain)
        actions_high = {(it.block_id, it.action) for it in plan_high.items}
        self.assertNotIn(("hot", "prefetch"), actions_high)

    def test_frozen_evicts_all(self):
        """冻结：除常驻块外全部淘汰（降级到最慢层）"""
        planner = Planner(TDSConfig())
        b = Block(id="x", size_mb=512, domain="tuxiang", last_used_step=9)
        pinned = Block(id="sys", size_mb=64, domain="qian", pinned=True)
        plan = planner.plan(None, blocks=[b, pinned], pressure="frozen", current_step=9)
        actions = {(it.block_id, it.action) for it in plan.items}
        self.assertIn(("x", "evict"), actions)
        self.assertIn(("sys", "pin"), actions)


class TestEndToEnd(unittest.TestCase):
    def _sched(self, state=None):
        sched = Scheduler(adapter=MockAdapter(state))
        b_hot = Block(id="ctx", size_mb=1024, domain="duihua", tier=Tier.RAM)
        b_cold = Block(id="old", size_mb=1024, domain="gongju", tier=Tier.VRAM)
        b_keep = Block(id="sys", size_mb=64, domain="qian", tier=Tier.VRAM, pinned=True)
        sched.store.add(b_hot)
        sched.store.add(b_cold)
        sched.store.add(b_keep)
        sched.step = 5
        b_hot.last_used_step = 5
        b_cold.last_used_step = 0
        sched._blocks = [b_hot, b_cold, b_keep]
        return sched

    def test_run_task_full_chain(self):
        """全链路：分类 → 八卦加权计划 → 四象约束执行 → 指标"""
        sched = self._sched()
        res = sched.run_task("帮我生成一张图片", blocks=sched._blocks)

        self.assertEqual(res["primary"], "tuxiang")
        self.assertEqual(res["pressure"], PressureState.LOW)
        self.assertEqual(res["action"], "prefetch")
        self.assertTrue(res["applied"])
        self.assertGreater(res["metrics"]["counters"]["task.tuxiang"], 0)
        self.assertIn(("load", "ctx"), sched.adapter.calls)
        self.assertIn(("unload", "old"), sched.adapter.calls)

    def test_apply_release_only_under_pressure(self):
        """高压力：禁止驻留/预取，但必须放行释放类动作（下沉/淘汰）"""
        heavy = VramState(total_mb=12288, used_mb=11700, free_mb=588)
        sched = self._sched(heavy)
        res = sched.run_task("帮我生成一张图片", blocks=sched._blocks)

        self.assertEqual(res["pressure"], PressureState.HIGH)
        self.assertGreater(res["metrics"]["counters"]["pressure_skip"], 0)
        actions = {a for a, _ in sched.adapter.calls}
        self.assertNotIn("load", actions)       # 不驻留
        self.assertNotIn("prefetch", actions)   # 不预取
        self.assertIn("unload", actions)        # 但释放动作照常执行

    def test_frozen_state_detected(self):
        """四象·冻结：水位 >= frozen_ratio"""
        sched = Scheduler()
        frozen = VramState(total_mb=12288, used_mb=12100, free_mb=188)
        self.assertEqual(sched.pressure_state(frozen), PressureState.FROZEN)
        self.assertEqual(sched.pressure_action(PressureState.FROZEN, []), "evict_all")


if __name__ == "__main__":
    unittest.main()
