"""决策轨迹单元测试（标准库 unittest，零第三方依赖）。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tds.config import TDSConfig
from tds.core.memory import VramState
from tds.core.scheduler import Scheduler, PressureState
from tds.core.storage import Block, Tier
from tds.core.trace import DEFAULT_TRACE_MAXLEN, DecisionTrace


class TestDecisionTrace(unittest.TestCase):
    def test_record_and_read(self):
        tr = DecisionTrace(maxlen=10)
        e = tr.record(1, "duihua", "low", "prefetch", plan="Plan(pin=1)", applied=True,
                      scores={"duihua": 0.8}, latency_ms=12.5)
        self.assertEqual(e.step, 1)
        self.assertEqual(e.primary, "duihua")
        self.assertEqual(len(tr), 1)
        self.assertEqual(tr.total, 1)
        self.assertEqual(tr.dropped, 0)
        row = tr.to_list()[0]
        self.assertEqual(row["action"], "prefetch")
        self.assertTrue(row["applied"])
        self.assertAlmostEqual(row["latency_ms"], 12.5, places=3)

    def test_ring_buffer_evicts_oldest(self):
        tr = DecisionTrace(maxlen=3)
        for i in range(1, 6):
            tr.record(i, "daima", "low", "pin", latency_ms=float(i))
        self.assertEqual(len(tr), 3)
        self.assertEqual(tr.total, 5)
        self.assertEqual(tr.dropped, 2)
        self.assertEqual([e.step for e in tr.entries()], [3, 4, 5])
        self.assertEqual(tr.last().step, 5)

    def test_entries_limit_takes_tail(self):
        tr = DecisionTrace(maxlen=10)
        for i in range(1, 6):
            tr.record(i, "qita", "low", "hold")
        self.assertEqual([e.step for e in tr.entries(2)], [4, 5])

    def test_scores_are_copied_not_aliased(self):
        tr = DecisionTrace()
        scores = {"tuxiang": 0.5}
        entry = tr.record(1, "tuxiang", "low", "prefetch", scores=scores)
        scores["tuxiang"] = 0.9
        self.assertEqual(entry.scores, {"tuxiang": 0.5})

    def test_aggregates_and_summary(self):
        tr = DecisionTrace(maxlen=10)
        tr.record(1, "duihua", "low", "prefetch", applied=True, latency_ms=10.0)
        tr.record(2, "duihua", "high", "skip", applied=True, latency_ms=20.0)
        tr.record(3, "shipin", "high", "skip", applied=False, latency_ms=30.0)
        self.assertEqual(tr.action_counts(), {"prefetch": 1, "skip": 2})
        self.assertEqual(tr.pressure_counts(), {"low": 1, "high": 2})
        self.assertEqual(tr.domain_counts(), {"duihua": 2, "shipin": 1})
        summary = tr.summary()
        self.assertEqual(summary["size"], 3)
        self.assertEqual(summary["maxlen"], 10)
        self.assertAlmostEqual(summary["applied_rate"], 2 / 3, places=6)
        self.assertAlmostEqual(summary["latency_ms"]["avg"], 20.0, places=3)
        self.assertAlmostEqual(summary["latency_ms"]["max"], 30.0, places=3)

    def test_empty_summary_is_safe(self):
        summary = DecisionTrace().summary()
        self.assertEqual(summary["size"], 0)
        self.assertEqual(summary["applied_rate"], 0.0)
        self.assertEqual(summary["latency_ms"], {"avg": 0.0, "max": 0.0})
        self.assertIsNone(DecisionTrace().last())

    def test_clear_keeps_total_counter(self):
        tr = DecisionTrace(maxlen=5)
        tr.record(1, "qita", "low", "hold")
        tr.clear()
        self.assertEqual(len(tr), 0)
        self.assertIsNone(tr.last())
        self.assertEqual(tr.total, 1)

    def test_default_and_min_maxlen(self):
        self.assertEqual(DecisionTrace().maxlen, DEFAULT_TRACE_MAXLEN)
        self.assertEqual(DecisionTrace(maxlen=0).maxlen, 1)
        self.assertEqual(DecisionTrace(maxlen=-5).maxlen, 1)

    def test_line_view_is_single_line(self):
        tr = DecisionTrace()
        tr.record(7, "daima", "critical", "hold", plan="Plan(pin=2)", applied=True,
                  latency_ms=3.5, note="保活")
        line = tr.lines()[0]
        self.assertNotIn("\n", line)
        self.assertIn("#7", line)
        self.assertIn("daima", line)
        self.assertIn("保活", line)


class TestSchedulerTrace(unittest.TestCase):
    def _scheduler(self, **overrides):
        cfg = TDSConfig(**overrides) if overrides else TDSConfig()
        return Scheduler(config=cfg)

    def test_run_task_records_entry(self):
        sch = self._scheduler()
        result = sch.run_task("帮我写一段 Python 代码", vram_state=None)
        self.assertEqual(len(sch.trace), 1)
        entry = sch.trace.last()
        self.assertEqual(entry.step, result["step"])
        self.assertEqual(entry.primary, result["primary"])
        self.assertEqual(entry.pressure, result["pressure"])
        self.assertEqual(entry.action, result["action"])
        self.assertEqual(entry.plan, result["plan"])
        self.assertEqual(entry.applied, result["applied"])
        self.assertGreaterEqual(entry.latency_ms, 0.0)

    def test_trace_maxlen_from_config(self):
        sch = self._scheduler(trace_maxlen=2)
        self.assertEqual(sch.trace.maxlen, 2)
        for i in range(4):
            sch.run_task("第 {} 轮对话".format(i))
        self.assertEqual(len(sch.trace), 2)
        self.assertEqual(sch.trace.total, 4)
        self.assertEqual(sch.trace.dropped, 2)

    def test_run_task_with_blocks_and_pressure(self):
        sch = Scheduler(config=TDSConfig())
        hot = Block(id="b1", size_mb=512, domain="duihua", tier=Tier.VRAM, last_used_step=0)
        cold = Block(id="b2", size_mb=512, domain="qita", tier=Tier.VRAM, last_used_step=-50)
        state = VramState(total_mb=12288, used_mb=12288, free_mb=0)  # 100% → 冻结
        result = sch.run_task("生成一段视频", blocks=[hot, cold], vram_state=state)
        self.assertEqual(result["pressure"], PressureState.FROZEN)
        self.assertEqual(result["action"], "evict_all")
        self.assertIn("evict", result["plan"])
        self.assertEqual(sch.trace.last().pressure, PressureState.FROZEN)

    def test_manual_record_decision(self):
        sch = self._scheduler()
        entry = sch.record_decision(primary="jiansuo", pressure="low", action="prefetch",
                                    plan="Plan(prefetch=3)", applied=True, latency_ms=1.0,
                                    note="自定义链路")
        self.assertEqual(entry.step, sch.step)
        self.assertEqual(sch.trace_entries()[-1].note, "自定义链路")

    def test_trace_report_and_lines(self):
        sch = self._scheduler()
        sch.run_task("你好")
        sch.run_task("再写点代码")
        report = sch.trace_report()
        self.assertEqual(report["size"], 2)
        self.assertEqual(sum(report["by_action"].values()), 2)
        self.assertEqual(len(sch.trace_lines()), 2)

    def test_metrics_report_unchanged(self):
        sch = self._scheduler()
        sch.run_task("你好")
        report = sch.report()
        self.assertIn("counters", report)
        self.assertIn("gauges", report)
        self.assertIn("cache_hit_rate", report)
        self.assertIn("task.", " ".join(report["counters"].keys()))

    def test_trace_is_isolated_per_scheduler(self):
        a, b = self._scheduler(), self._scheduler()
        a.run_task("你好")
        self.assertEqual(len(a.trace), 1)
        self.assertEqual(len(b.trace), 0)


if __name__ == "__main__":
    unittest.main()
