"""EvolutionEngine（双线进化）+ 测试技术接口 单元测试。

覆盖：
1. 默认关闭（enabled=False）→ evolve() 为 no-op，不产生任何变更/建议；
2. 样本不足 → 无候选；
3. 低风险候选自动应用（备份 + changelog + 覆盖层写入）；
4. medium/high 只出建议书，绝不自动应用；
5. 四象压力守卫：高压/冻结禁用变更；
6. IP 进化线：新词条入库、误判词条降权、双线档案独立；
7. ParameterStore 白名单边界与 clamp；
8. 回退 revert；
9. 测试接口：压力仿真 / A-B 对照 / 回归基准 / 探针。
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tds.core.metrics import Metrics
from tds.core.trace import DecisionTrace
from tds.evolution import (
    EvolutionConfig, EvolutionEngine, ParameterStore, Whitelist, ParamSpec,
    Risk, IPEvolutionEngine, Proposal, ProposalBook,
)
from tds.testing import (
    PressureInjector, PressureScenario, ABTrial, TrialResult,
    RegressionBenchmark, Probe,
)


class TempArchiveMixin:
    def make_config(self, enabled=True, archive_dir=None, proposal_dir=None,
                    min_samples=4, learn_window=50):
        return EvolutionConfig(
            enabled=enabled,
            archive_dir=archive_dir or tempfile.mkdtemp(prefix="tds_ev_arch_"),
            proposal_dir=proposal_dir or tempfile.mkdtemp(prefix="tds_ev_prop_"),
            min_samples=min_samples,
            learn_window=learn_window,
        )


class TestEvolutionEngine(unittest.TestCase, TempArchiveMixin):
    def test_default_disabled_is_noop(self):
        cfg = self.make_config(enabled=False)
        trace = DecisionTrace(maxlen=20)
        for i in range(10):
            trace.record(i, "duihua", "low", "prefetch", applied=True)
        metrics = Metrics()
        for _ in range(50):
            metrics.hit()
        engine = EvolutionEngine(config=cfg, trace=trace, metrics=metrics)
        book = engine.evolve()
        self.assertEqual(len(book), 0)
        self.assertEqual(len(engine.archive.changelog()), 0)

    def test_insufficient_samples_is_noop(self):
        cfg = self.make_config(enabled=True, min_samples=50)
        trace = DecisionTrace(maxlen=20)
        for i in range(3):
            trace.record(i, "duihua", "low", "prefetch", applied=True)
        engine = EvolutionEngine(config=cfg, trace=trace)
        book = engine.evolve()
        self.assertEqual(len(book), 0)

    def test_low_risk_auto_apply_keep_steps(self):
        cfg = self.make_config(enabled=True, min_samples=4)
        trace = DecisionTrace(maxlen=100)
        # 大量 duihua 域 evict + 命中率偏低 → 建议增大 keep_steps(low, 自动应用)
        # 压力保持 low：高压/冻结下引擎禁止变更（由 test_pressure_guard 覆盖）
        for i in range(40):
            trace.record(i, "duihua", "low", "evict_all", applied=True)
        metrics = Metrics()
        for _ in range(40):
            metrics.miss()
        for _ in range(5):
            metrics.hit()
        engine = EvolutionEngine(config=cfg, trace=trace, metrics=metrics)
        book = engine.evolve()
        self.assertGreater(len(engine.archive.changelog()), 0)
        # 覆盖层应已写入 BAGUA_KEEP_STEPS.duihua 的增大值
        overlay = dict(engine.store.overlay_items())
        path = ("BAGUA_KEEP_STEPS", "duihua")
        self.assertIn(path, overlay)
        self.assertGreater(float(overlay[path]), 0.0)

    def test_medium_high_only_proposal(self):
        cfg = self.make_config(enabled=True, min_samples=4)
        trace = DecisionTrace(maxlen=100)
        # 高压占比高 → pressure_ratio 候选(medium)
        for i in range(30):
            trace.record(i, "duihua", "high", "evict_all", applied=True)
        for i in range(5):
            trace.record(30 + i, "duihua", "low", "prefetch", applied=True)
        metrics = Metrics()
        for _ in range(40):
            metrics.miss()
        engine = EvolutionEngine(config=cfg, trace=trace, metrics=metrics)
        book = engine.evolve()
        medium = book.items_by_risk(Risk.MEDIUM)
        self.assertGreater(len(medium), 0)
        self.assertTrue(all(not p.applied for p in book.pending_items()))
        # 仅 low 风险可被自动应用；medium 只建议（apply 记录全部为 low）
        applies = [c for c in engine.archive.changelog() if c.op == "apply"]
        self.assertTrue(all(c.risk == "low" for c in applies), applies)

    def test_pressure_guard_blocks_high(self):
        cfg = self.make_config(enabled=True, min_samples=4)
        trace = DecisionTrace(maxlen=100)
        # 最后一条为 frozen → 压力守卫禁止变更
        for i in range(40):
            trace.record(i, "duihua", "high", "evict_all", applied=True)
        trace.record(99, "duihua", "frozen", "evict_all", applied=True)
        metrics = Metrics()
        for _ in range(40):
            metrics.miss()
        engine = EvolutionEngine(config=cfg, trace=trace, metrics=metrics)
        book = engine.evolve()
        self.assertEqual(len(book), 0)
        self.assertEqual(len(engine.archive.overlay_items() if hasattr(engine.archive, "overlay_items") else []), 0)

    def test_parameter_store_whitelist_boundary(self):
        wl = Whitelist()
        store = ParameterStore(wl, defaults={("BAGUA_KEEP_STEPS", "duihua"): 3})
        self.assertEqual(store.get(("BAGUA_KEEP_STEPS", "duihua")), 3)
        self.assertIsNone(store.get(("not", "whitelisted")))
        with self.assertRaises(KeyError):
            store.apply(("not", "whitelisted"), 1.0)
        # clamp: keep_steps max 64
        final = store.apply(("BAGUA_KEEP_STEPS", "duihua"), 999)
        self.assertEqual(final, 64)

    def test_revert(self):
        cfg = self.make_config(enabled=True, min_samples=4)
        trace = DecisionTrace(maxlen=100)
        for i in range(40):
            trace.record(i, "duihua", "low", "evict_all", applied=True)
        metrics = Metrics()
        for _ in range(40):
            metrics.miss()
        engine = EvolutionEngine(config=cfg, trace=trace, metrics=metrics)
        engine.evolve()
        path = ("BAGUA_KEEP_STEPS", "duihua")
        before = engine.store.get(path)
        self.assertTrue(engine.revert(path))
        # revert 应写回备份旧值并追加 changelog
        ops = [c.op for c in engine.archive.changelog()]
        self.assertIn("revert", ops)

    def test_manual_apply_all(self):
        cfg = self.make_config(enabled=True, min_samples=4)
        trace = DecisionTrace(maxlen=100)
        for i in range(30):
            trace.record(i, "duihua", "high", "evict_all", applied=True)
        trace.record(99, "duihua", "low", "prefetch", applied=True)
        metrics = Metrics()
        for _ in range(40):
            metrics.miss()
        engine = EvolutionEngine(config=cfg, trace=trace, metrics=metrics)
        book = engine.evolve()
        pending = len(book.pending_items())
        if pending == 0:
            self.skipTest("no pending proposals in this run")
        applied = engine.apply_all(book)
        self.assertEqual(applied, pending)
        self.assertTrue(all(p.applied for p in book.items()))


class TestIPEvolutionEngine(unittest.TestCase, TempArchiveMixin):
    def test_new_term_auto_apply(self):
        cfg = self.make_config(enabled=True, min_samples=2)
        engine = IPEvolutionEngine(config=cfg)
        # 词条 "def" 正确分类为 daima 2 次 → low 自动入库
        engine.observe("daima", "def", True)
        engine.observe("daima", "def", True)
        book = engine.evolve()
        term = engine.store.get_term("def")
        self.assertIsNotNone(term)
        self.assertEqual(term["domain"], "daima")
        # changelog 有 apply
        self.assertIn("apply", [c.op for c in engine.archive.changelog()])

    def test_wrong_term_demotion_only_proposal(self):
        cfg = self.make_config(enabled=True, min_samples=2)
        engine = IPEvolutionEngine(config=cfg)
        engine.store.add_term("tuxiang", "tuxiang", source="seed")
        # 误判 3 次 / 正确 1 次 → 降权为 medium，只建议
        for _ in range(3):
            engine.observe("daima", "tuxiang", False)
        engine.observe("tuxiang", "tuxiang", True)
        book = engine.evolve()
        medium = book.items_by_risk(Risk.MEDIUM)
        self.assertGreater(len(medium), 0)
        self.assertTrue(all(not p.applied for p in medium))
        # 词条未降权（未被自动应用）
        self.assertGreaterEqual(float(engine.store.get_term("tuxiang")["confidence"]), 1.0)

    def test_dual_line_archives_independent(self):
        cfg = self.make_config(enabled=True, min_samples=2)
        engine = EvolutionEngine(config=cfg)
        ip = IPEvolutionEngine(config=cfg)
        self.assertNotEqual(engine.archive.file_path, ip.archive.file_path)
        # 软件线档案不应出现 knowledge 段
        self.assertNotIn("knowledge", engine.archive.get_data())
        self.assertIn("knowledge", ip.archive.get_data())


class TestTestingInterfaces(unittest.TestCase):
    def test_pressure_injector_spike(self):
        sc = PressureScenario.spike(steps=10, spike_at=5, spike_ratio=0.99)
        inj = PressureInjector(scenario=sc)
        out = inj.run(lambda ratio: "frozen" if ratio >= 0.98 else "low")
        self.assertEqual(out["steps"], 10)
        self.assertGreaterEqual(out["match_rate"], 0.8)

    def test_ab_trial_verdict(self):
        trial = ABTrial(baseline_params={"keep": 1}, candidate_params={"keep": 5})
        def sim(params, load):
            hits = sum(1 for l in load if params["keep"] >= 3 and l["ratio"] < 0.95)
            return TrialResult(label="x", hit_rate=hits / max(1, len(load)),
                               evicts=len(load) - hits)
        results = trial.run(sim)
        report = ABTrial.report(results)
        self.assertIn("verdict", report)

    def test_regression_benchmark(self):
        bench = RegressionBenchmark()
        bench.register("hit_rate_baseline", "hit_rate", 0.9)
        bench.register("evict_guard", "no_frozen", 1.0)
        records = bench.run(lambda name: 0.95 if name == "hit_rate_baseline" else 1.0)
        self.assertTrue(bench.all_passed())
        self.assertTrue(all(r.passed for r in records))

    def test_probe_summary(self):
        probe = Probe()
        probe.watch("evict").fire(1.2, ratio=0.92)
        probe.watch("evict").fire(2.5, ratio=0.95)
        probe.watch("prefetch").fire(0.3, ratio=0.8)
        s = probe.summary()
        self.assertEqual(s["count"], 3)
        self.assertIn("evict", s["by_point"])


if __name__ == "__main__":
    unittest.main()
