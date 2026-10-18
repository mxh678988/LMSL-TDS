"""TDS 控制面板单测：/api/tds/config 读改、白名单校验、/api/tds/state、trace.resize。

不依赖真实模型后端；直接构造 UiServer + Scheduler 实例调用方法层，
避免起 HTTP 线程，保证零外部依赖且可离线运行。
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tds.core.scheduler import Scheduler  # noqa: E402
from tds.core.trace import DecisionTrace  # noqa: E402
from tds.ui import UiServer  # noqa: E402


class _FakeVram:
    total_mb = 10788
    used_mb = 10100
    free_mb = 688
    fragmentation_mb = 0

    @property
    def usage_ratio(self) -> float:
        return self.used_mb / self.total_mb


class _FakeAdapter:
    def vram_state(self):
        return _FakeVram()


class TestTdsPanel(unittest.TestCase):
    def _panel(self, with_scheduler: bool = True) -> UiServer:
        srv = UiServer(backend_url="http://127.0.0.1:9")
        if with_scheduler:
            srv.scheduler = Scheduler()
        return srv

    def test_config_unmounted(self) -> None:
        panel = self._panel(with_scheduler=False)
        cfg = panel.tds_config()
        self.assertFalse(cfg["ok"])
        self.assertFalse(cfg["mounted"])
        self.assertEqual(cfg["config"], {})

    def test_config_read(self) -> None:
        panel = self._panel()
        cfg = panel.tds_config()
        self.assertTrue(cfg["ok"])
        self.assertTrue(cfg["mounted"])
        conf = cfg["config"]
        for key in ("vram_budget_mb", "reserve_mb", "keep_steps",
                     "prefetch_enabled", "pressure_guard", "pressure_ratio",
                     "frozen_ratio", "trace_maxlen"):
            self.assertIn(key, conf)

    def test_config_update_valid(self) -> None:
        panel = self._panel()
        res = panel.tds_update(
            {"keep_steps": 3, "prefetch_enabled": True, "pressure_ratio": 0.85}
        )
        self.assertTrue(res["ok"])
        sched = panel.scheduler
        self.assertEqual(sched.cfg.keep_steps, 3)
        self.assertTrue(sched.cfg.prefetch_enabled)
        self.assertAlmostEqual(sched.cfg.pressure_ratio, 0.85)

    def test_config_update_rejects_unknown_and_oob(self) -> None:
        panel = self._panel()
        res = panel.tds_update({"no_such_param": 1, "keep_steps": -5, "vram_budget_mb": 100000})
        self.assertFalse(res["ok"])
        self.assertTrue(any("不支持参数" in e for e in res["errors"]))
        self.assertTrue(any("不能小于" in e for e in res["errors"]))
        self.assertTrue(any("不能大于" in e for e in res["errors"]))
        # 合法字段不受非法字段牵连
        self.assertNotIn("keep_steps", res["updated"])

    def test_config_update_bool_parsing(self) -> None:
        panel = self._panel()
        res = panel.tds_update({"prefetch_enabled": "true"})
        self.assertTrue(res["ok"])
        self.assertTrue(panel.scheduler.cfg.prefetch_enabled)

    def test_trace_resize(self) -> None:
        tr = DecisionTrace(maxlen=5)
        for i in range(10):
            tr.record(step=i, primary="qita", pressure="low", action="pin")
        self.assertEqual(len(tr.entries()), 5)
        tr.resize(3)
        self.assertEqual(len(tr.entries()), 3)
        # 收缩后保留的是最近的 3 条
        self.assertEqual([e.step for e in tr.entries()], [7, 8, 9])
        tr.resize(100)
        self.assertEqual(len(tr.entries()), 3)
        tr.resize(5)
        self.assertEqual(tr.maxlen, 5)

    def test_config_update_trace_maxlen_effective(self) -> None:
        panel = self._panel()
        for i in range(25):
            panel.scheduler.run_task("你好，帮我规划下今天的工作")
        self.assertEqual(len(panel.scheduler.trace.entries()), 25)
        res = panel.tds_update({"trace_maxlen": 10})
        self.assertTrue(res["ok"])
        self.assertEqual(panel.scheduler.trace.maxlen, 10)
        self.assertEqual(len(panel.scheduler.trace.entries()), 10)

    def test_state_unmounted(self) -> None:
        panel = self._panel(with_scheduler=False)
        st = panel.tds_state()
        self.assertFalse(st["ok"])
        self.assertFalse(st["mounted"])

    def test_state_with_adapter_pressure(self) -> None:
        panel = self._panel()
        panel.scheduler.adapter = _FakeAdapter()
        st = panel.tds_state()
        self.assertTrue(st["ok"])
        self.assertEqual(st["pressure"], "high")  # 10100/10788=0.936 >= 0.92
        self.assertIsNotNone(st["vram"])
        self.assertIn("metrics", st)
        self.assertIn("trace", st)
        self.assertIn("yang_yin", st)
        self.assertEqual(st["step"], 0)

    def test_state_tracks_yang_yin(self) -> None:
        panel = self._panel()
        panel.scheduler.run_task("你好，帮我总结一下这份文档")
        panel.scheduler.run_task("渲染一张风景图")
        st = panel.tds_state()
        self.assertTrue(st["ok"])
        self.assertGreaterEqual(sum(st["yang_yin"].values()), 2)
        self.assertGreaterEqual(st["step"], 2)


if __name__ == "__main__":
    unittest.main()
