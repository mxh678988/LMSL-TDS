"""指标埋点：命中率 / TTFT / 显存峰值。"""
import time
from collections import defaultdict
from typing import Dict, Optional


class Metrics:
    """轻量指标收集器（零依赖）。"""

    def __init__(self):
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self._ttft_start: Optional[float] = None

    def record(self, key: str, value: float = 1.0) -> None:
        self.counters[key] += value

    def gauge(self, key: str, value: float) -> None:
        self.gauges[key] = value

    # ---- 缓存命中 ----
    def hit(self) -> None:
        self.record("cache.hit")

    def miss(self) -> None:
        self.record("cache.miss")

    def hit_rate(self) -> float:
        h, m = self.counters.get("cache.hit", 0), self.counters.get("cache.miss", 0)
        total = h + m
        return h / total if total else 0.0

    # ---- 首字延迟 TTFT ----
    def start_ttft(self) -> None:
        self._ttft_start = time.perf_counter()

    def end_ttft(self) -> Optional[float]:
        if self._ttft_start is None:
            return None
        dt = (time.perf_counter() - self._ttft_start) * 1000.0
        self.record("ttft_ms.sum", dt)
        return dt

    def report(self) -> dict:
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "cache_hit_rate": self.hit_rate(),
        }
