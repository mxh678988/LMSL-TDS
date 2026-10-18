"""太极紫梅系列 · 测试技术接口包（自研增量验证设施）。

设计初衷（分支设想）
--------------------
在正式调度闭环之外，为"太极紫梅"独立软件预埋一套**可插拔的测试技术接口**，
用于持续验证自研增量（八卦域语义调度 / 紫梅动态淘汰 / 进化引擎）确实带来
收益，而非空谈。包含四类能力：

1. 压力仿真注入（PressureInjector）：
   离线/在线向调度层注入可控压力序列，复现高压/冻结场景，验证四象保护逻辑。
2. A/B 参数对照（ABTrial）：
   对同一负载分别跑"基线参数"与"候选参数"，输出命中率/换页数/耗时对比。
3. 回归基准（RegressionBenchmark）：
   固化代表性负载与期望下限，任何增量改动不得突破基准（防止回退）。
4. 可观测性探针（Probe）：
   挂载在关键决策点，记录决策/执行耗时/资源水位，供 UI 与进化引擎消费。

约束：零第三方依赖（仅标准库），不 import torch，不触碰宿主，纯接口+仿真。
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

# ---------- 1. 压力仿真注入 ----------

PRESSURE_LEVELS = ("low", "critical", "high", "frozen")


@dataclass
class PressureScenario:
    """一段可控的压力序列：每步给出显存水位比例与期望档位。"""
    name: str = "default"
    steps: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def sine_wave(cls, name: str = "sine", steps: int = 120,
                  min_ratio: float = 0.70, max_ratio: float = 0.99,
                  seed: int = 42) -> "PressureScenario":
        """正弦压力波：模拟周期性负载（训练/渲染高峰）。"""
        rng = random.Random(seed)
        steps_list: List[Dict[str, Any]] = []
        for i in range(steps):
            t = i / max(1, steps - 1)
            ratio = min_ratio + (max_ratio - min_ratio) * (0.5 + 0.5 * __import__("math").sin(2 * 3.14159 * t * 2 + rng.uniform(-0.1, 0.1)))
            steps_list.append({"step": i, "ratio": round(ratio, 4)})
        return cls(name=name, steps=steps_list)

    @classmethod
    def spike(cls, name: str = "spike", steps: int = 60,
              spike_at: int = 30, spike_ratio: float = 0.99,
              base_ratio: float = 0.75) -> "PressureScenario":
        """尖峰压力：模拟突发高负载（爆显存场景）。"""
        steps_list = []
        for i in range(steps):
            ratio = spike_ratio if i == spike_at else base_ratio
            steps_list.append({"step": i, "ratio": round(ratio, 4)})
        return cls(name=name, steps=steps_list)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "steps": self.steps}


class PressureInjector:
    """压力仿真注入器：向被测调度函数注入可控压力序列。"""

    def __init__(self, scenario: Optional[PressureScenario] = None,
                 seed: int = 7):
        self.scenario = scenario or PressureScenario.sine_wave()
        self.seed = seed
        self._idx = 0
        self.log: List[Dict[str, Any]] = []

    def next_ratio(self) -> float:
        """取下一帧压力水位比例（循环播放）。"""
        if not self.scenario.steps:
            return 0.75
        item = self.scenario.steps[self._idx % len(self.scenario.steps)]
        self._idx += 1
        return float(item["ratio"])

    def run(self, fn: Callable[[float], Any], expect_pressure: Optional[str] = None) -> Dict[str, Any]:
        """按场景驱动被测函数，记录每步档位与期望档位比对。"""
        results = []
        correct = 0
        start = time.time()
        for item in self.scenario.steps:
            ratio = float(item["ratio"])
            out = fn(ratio)
            actual = out if isinstance(out, str) else (out.get("pressure") if isinstance(out, dict) else "")
            results.append({"step": item["step"], "ratio": ratio,
                            "actual": actual, "expect": expect_pressure})
            if expect_pressure is None or actual == expect_pressure:
                correct += 1
        return {
            "scenario": self.scenario.name,
            "steps": len(results),
            "match_rate": round(correct / max(1, len(results)), 4),
            "elapsed_ms": round((time.time() - start) * 1000, 3),
            "results": results,
        }

    def save(self, file_path: str) -> str:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump({"scenario": self.scenario.to_dict(),
                       "log": self.log[-500:]}, fh, ensure_ascii=False, indent=2)
        return file_path


# ---------- 2. A/B 参数对照 ----------

@dataclass
class TrialResult:
    """单组参数在一份负载上的运行结果。"""
    label: str
    hit_rate: float = 0.0
    evicts: int = 0
    elapsed_ms: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "hit_rate": self.hit_rate,
                "evicts": self.evicts, "elapsed_ms": self.elapsed_ms,
                "extra": dict(self.extra)}


class ABTrial:
    """A/B 参数对照：同一负载，基线 vs 候选，输出公平对比。"""

    def __init__(self, baseline_params: Dict[str, Any],
                 candidate_params: Dict[str, Any],
                 load: Optional[Sequence[Dict[str, Any]]] = None,
                 seed: int = 2026):
        self.baseline = baseline_params
        self.candidate = candidate_params
        self.load = load
        self.seed = seed

    def run(self, simulate: Callable[[Dict[str, Any], Sequence[Dict[str, Any]]], TrialResult]) -> List[TrialResult]:
        """对基线/候选各跑一遍 simulate，返回对比结果。"""
        load = self.load if self.load is not None else self._default_load()
        out = []
        for label, params in (("baseline", self.baseline), ("candidate", self.candidate)):
            r = simulate(params, load)
            r.label = label
            out.append(r)
        return out

    def _default_load(self) -> List[Dict[str, Any]]:
        """生成确定性负载（固定种子，可复现）。"""
        rng = random.Random(self.seed)
        domains = ["qian", "duihua", "jiansuo", "daima", "tuxiang", "shipin", "gongju", "qita"]
        load = []
        for i in range(300):
            load.append({
                "step": i,
                "page_id": "p{}".format(rng.randint(0, 200)),
                "domain": domains[rng.randint(0, 7)],
                "ratio": round(rng.uniform(0.7, 0.98), 4),
            })
        return load

    @staticmethod
    def report(results: List[TrialResult]) -> Dict[str, Any]:
        if len(results) != 2:
            return {"error": "need baseline+candidate"}
        b, c = results[0], results[1]
        return {
            "baseline": b.to_dict(),
            "candidate": c.to_dict(),
            "delta_hit_rate": round(c.hit_rate - b.hit_rate, 4),
            "delta_evicts": c.evicts - b.evicts,
            "delta_elapsed_ms": round(c.elapsed_ms - b.elapsed_ms, 3),
            "verdict": "candidate_wins" if c.hit_rate > b.hit_rate else (
                "baseline_wins" if c.hit_rate < b.hit_rate else "tie"),
        }


# ---------- 3. 回归基准 ----------

@dataclass
class BenchRecord:
    """一条回归基准：负载名 + 期望下限 + 实际结果。"""
    name: str
    metric: str
    min_value: float
    actual: float
    passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "metric": self.metric,
                "min_value": self.min_value, "actual": self.actual,
                "passed": self.passed}


class RegressionBenchmark:
    """回归基准：固化代表性负载下限，防止增量改动导致回退。"""

    def __init__(self, baseline_file: Optional[str] = None):
        self._records: List[BenchRecord] = []
        self.baseline_file = baseline_file

    def register(self, name: str, metric: str, min_value: float) -> None:
        self._records.append(BenchRecord(name=name, metric=metric,
                                         min_value=min_value, actual=0.0))

    def run(self, fn: Callable[[str], float]) -> List[BenchRecord]:
        for rec in self._records:
            rec.actual = float(fn(rec.name))
            rec.passed = rec.actual >= rec.min_value
        return list(self._records)

    def all_passed(self) -> bool:
        return all(r.passed for r in self._records)

    def save(self, file_path: Optional[str] = None) -> str:
        target = file_path or self.baseline_file
        if not target:
            return ""
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump({"records": [r.to_dict() for r in self._records],
                       "all_passed": self.all_passed()},
                      fh, ensure_ascii=False, indent=2)
        return target


# ---------- 4. 可观测性探针 ----------

@dataclass
class ProbeEvent:
    """一条探针事件：决策点 + 耗时 + 水位 + 载荷。"""
    point: str
    elapsed_ms: float = 0.0
    ratio: float = 0.0
    payload: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"point": self.point, "elapsed_ms": round(self.elapsed_ms, 3),
                "ratio": self.ratio, "payload": dict(self.payload),
                "ts": round(self.ts, 3)}


class Probe:
    """可观测性探针：挂载关键决策点，收集耗时/水位/载荷，供 UI 与进化引擎消费。"""

    def __init__(self, max_events: int = 2000):
        self._events: List[ProbeEvent] = []
        self.max_events = max_events

    def watch(self, point: str) -> "Probe":
        """返回自身以支持链式：probe.watch("evict").fire(...)"""
        self._last_point = point
        return self

    def fire(self, elapsed_ms: float, ratio: float = 0.0,
             payload: Optional[Dict[str, Any]] = None) -> ProbeEvent:
        ev = ProbeEvent(point=getattr(self, "_last_point", "unknown"),
                        elapsed_ms=elapsed_ms, ratio=ratio,
                        payload=payload or {})
        self._events.append(ev)
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events:]
        return ev

    def events(self) -> List[ProbeEvent]:
        return list(self._events)

    def summary(self) -> Dict[str, Any]:
        if not self._events:
            return {"count": 0}
        by_point: Dict[str, List[float]] = {}
        for e in self._events:
            by_point.setdefault(e.point, []).append(e.elapsed_ms)
        agg = {}
        for pt, times in by_point.items():
            agg[pt] = {
                "count": len(times),
                "avg_ms": round(sum(times) / len(times), 3),
                "max_ms": round(max(times), 3),
                "p95_ms": round(sorted(times)[int(len(times) * 0.95) - 1] if len(times) > 20 else max(times), 3),
            }
        return {"count": len(self._events), "by_point": agg}
