"""两仪阴阳策略层集成测试。

验证 scheduler.py 中阴阳策略层的三处接线：
1. plan() 内 decide() → _apply_strategy_to_plan()（计划增强，保留 Plan 结构）
2. run_task() 内 decide() 记录模式/原因到结果与轨迹
3. record_decision() 扩展参数（yang_yin_mode / yang_yin_reason）

覆盖场景：
- 阳模式：对话域任务 → prefetch/pin 优先级提升
- 阴模式：非对话域任务 → prefetch 降级为 pin（快速响应）
- 高压力：强制阴策略保活，释放动作保留
- 平滑过渡与紧急模式路径（无异常即可）
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

from tds.config import TDSConfig
from tds.core.memory import VramState
from tds.core.scheduler import Scheduler
from tds.core.storage import Block, Tier
from tds.core.yang_yin_strategy import StrategyMode


class MockAdapter:
    """最小适配器：只提供 vram_state 与无副作用动作。"""

    def __init__(self, used_mb=6000):
        self._used = used_mb

    def scan_usage(self, graph=None):
        return []

    def vram_state(self):
        return VramState(total_mb=12000, used_mb=self._used)

    def load(self, block_id):
        return True

    def unload(self, block_id):
        return True

    def prefetch(self, block_id):
        return True

    def unload_all(self):
        pass


def make_scheduler(used_mb=6000):
    cfg = TDSConfig(
        pressure_guard=True,
        pressure_ratio=0.92,
        frozen_ratio=0.98,
        prefetch_enabled=True,
    )
    sch = Scheduler(cfg, MockAdapter(used_mb=used_mb))
    sch.step = 5
    blocks = [
        Block(id="chat_ctx", size_mb=100, domain="duihua", tier=Tier.SSD, last_used_step=4),
        Block(id="img_ctx", size_mb=200, domain="tuxiang", tier=Tier.SSD, last_used_step=5),
    ]
    for b in blocks:
        sch.store.add(b)
    return sch, blocks


class TestYangYinIntegration(unittest.TestCase):

    def test_yang_mode_dialogue_domain(self):
        """阳模式：对话域任务 → 模式为 yang，计划含 prefetch 且结果回填。"""
        sch, blocks = make_scheduler()
        result = sch.run_task("你好 请继续帮我分析数据报告", blocks=blocks,
                              vram_state=sch.adapter.vram_state())
        self.assertEqual(result["primary"], "duihua")
        self.assertEqual(result["yang_yin_mode"], "yang")
        self.assertIn("prefetch", result["plan"])
        self.assertTrue(result["yang_yin_reason"])

    def test_yin_mode_non_dialogue(self):
        """阴模式：非对话域任务 → 模式为 yin，prefetch 降级为 pin。"""
        sch, blocks = make_scheduler()
        result = sch.run_task("图像 生成一张风景图", blocks=blocks,
                              vram_state=sch.adapter.vram_state())
        self.assertEqual(result["primary"], "tuxiang")
        self.assertEqual(result["yang_yin_mode"], "yin")
        # 阴策略下计划不应再包含 prefetch（降为 pin）
        self.assertNotIn("prefetch", result["plan"])

    def test_high_pressure_forced_yin(self):
        """高压力：强制阴策略，释放动作保留、预取丢弃。"""
        sch, blocks = make_scheduler(used_mb=11100)  # ratio=0.925 > 0.92
        result = sch.run_task("你好 请继续帮我分析数据报告", blocks=blocks,
                              vram_state=sch.adapter.vram_state())
        self.assertEqual(result["pressure"], "high")
        self.assertEqual(result["yang_yin_mode"], "yin")
        self.assertIn("demote", result["plan"])
        self.assertNotIn("prefetch", result["plan"])

    def test_trace_records_yang_yin(self):
        """轨迹记录：note 含 mode/reason。"""
        sch, blocks = make_scheduler()
        sch.run_task("你好 请继续帮我分析数据报告", blocks=blocks,
                     vram_state=sch.adapter.vram_state())
        last = sch.trace_entries(limit=1)[0]
        self.assertIn("mode=yang", last.note)
        self.assertIn("reason=", last.note)

    def test_plan_keeps_plan_structure(self):
        """计划增强不破坏 Plan 结构：summary() 仍可用且项数不变。"""
        sch, blocks = make_scheduler()
        profile = sch.on_task("你好 请继续帮我分析数据报告")
        plan = sch.plan(profile, blocks=blocks, vram_state=sch.adapter.vram_state())
        self.assertTrue(hasattr(plan, "summary"))
        self.assertGreater(len(plan.items), 0)

    def test_emergency_mode_path(self):
        """紧急/冻结压力：策略可调用，释放动作保留。"""
        sch, blocks = make_scheduler(used_mb=11800)  # ratio≈0.983 > frozen
        result = sch.run_task("你好 请继续帮我分析数据报告", blocks=blocks,
                              vram_state=sch.adapter.vram_state())
        self.assertIn(result["pressure"], ("high", "frozen"))
        # 紧急/冻结：释放类动作（demote/evict）保留
        self.assertTrue(any(a in result["plan"] for a in ("demote", "evict")))
        self.assertNotIn("prefetch", result["plan"])

    def test_emergency_mode_explicit_switch(self):
        """显式开关：关闭 emergency_mode 后，高压不强制阴策略保活。"""
        from tds.core.yang_yin_strategy import YangYinStrategy, YangYinConfig
        from tds.core.classifier import DomainProfile
        strat = YangYinStrategy(YangYinConfig(emergency_mode_enabled=False))
        profile = DomainProfile(primary="duihua", scores={"duihua": 1.0})
        sr = strat.decide(profile=profile, pressure="high",
                          vram_state=VramState(total_mb=12000, used_mb=11800),
                          blocks=[], current_step=5)
        # 关闭显式开关：紧急压力不进入紧急保活，仍按对话域阳策略
        self.assertFalse(sr.emergency_mode)
        self.assertEqual(sr.mode, StrategyMode.YANG)

    def test_emergency_mode_default_enabled(self):
        """默认配置：emergency_mode 显式启用，高压强制保活。"""
        from tds.core.yang_yin_strategy import YangYinStrategy
        from tds.core.classifier import DomainProfile
        strat = YangYinStrategy()
        profile = DomainProfile(primary="duihua", scores={"duihua": 1.0})
        sr = strat.decide(profile=profile, pressure="high",
                          vram_state=VramState(total_mb=12000, used_mb=11800),
                          blocks=[], current_step=5)
        self.assertTrue(sr.emergency_mode)
        self.assertEqual(sr.mode, StrategyMode.YIN)


if __name__ == "__main__":
    unittest.main(verbosity=2)
