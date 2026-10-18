"""调度决策轨迹：零依赖的决策时间线。

设计约束
--------
1. 仅依赖标准库，保持 core 平台无关（可随可售 IP 一起交付）；
2. 固定容量环形缓冲，长跑不增长、不泄漏；
3. 只记录"决策元数据"（域 / 压力 / 动作 / 计划摘要 / 时延），
   不落任何宿主态正文，便于跨宿主复用与事后归因。

用途：为调度层提供可追溯的执行时间线，支撑
  - 抖动与回退归因（哪一步做了 evict / 为何跳过预取）
  - 压力触发的频次统计（四象分布）
  - 与测试台报告叠加，形成"输入 → 决策 → 结果"的完整链路
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

DEFAULT_TRACE_MAXLEN = 200


@dataclass
class TraceEntry:
    """一次调度决策的快照。"""

    step: int
    primary: str                 # 八卦主域
    pressure: str                # 四象压力状态
    action: str                  # 四象动作模板：prefetch / hold / skip / evict_all
    plan: str = ""               # 计划摘要（Plan(...)）
    applied: bool = False        # 适配层是否真正执行
    scores: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    note: str = ""
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "primary": self.primary,
            "pressure": self.pressure,
            "action": self.action,
            "plan": self.plan,
            "applied": self.applied,
            "scores": dict(self.scores),
            "latency_ms": round(self.latency_ms, 3),
            "note": self.note,
            "ts": self.ts,
        }

    def line(self) -> str:
        """单行文本视图，便于日志与报告排版。"""
        flag = "ok" if self.applied else "--"
        extra = " {}".format(self.note) if self.note else ""
        return "#{:<4d} [{}/{}] {:<10s} {:<16s} {:>7.2f}ms {}{}".format(
            self.step, self.primary, self.pressure, self.action,
            self.plan or "Plan(empty)", self.latency_ms, flag, extra,
        )


class DecisionTrace:
    """决策轨迹环形缓冲（零依赖）。"""

    def __init__(self, maxlen: int = DEFAULT_TRACE_MAXLEN):
        self.maxlen = max(1, int(maxlen))
        self._entries: Deque[TraceEntry] = deque(maxlen=self.maxlen)
        self._total = 0

    # ---- 写入 ----
    def record(
        self,
        step: int,
        primary: str,
        pressure: str,
        action: str,
        plan: str = "",
        applied: bool = False,
        scores: Optional[Dict[str, float]] = None,
        latency_ms: float = 0.0,
        note: str = "",
    ) -> TraceEntry:
        entry = TraceEntry(
            step=int(step),
            primary=str(primary),
            pressure=str(pressure),
            action=str(action),
            plan=str(plan),
            applied=bool(applied),
            scores=dict(scores or {}),
            latency_ms=float(latency_ms),
            note=str(note),
        )
        self._entries.append(entry)
        self._total += 1
        return entry

    # ---- 容量调整 ----
    def resize(self, maxlen: int) -> None:
        """调整环形缓冲容量；收缩时保留最近 maxlen 条，扩容保留全部。"""
        newlen = max(1, int(maxlen))
        if newlen == self.maxlen:
            return
        rows = list(self._entries)
        self.maxlen = newlen
        self._entries = deque(rows[-newlen:], maxlen=newlen)

    # ---- 读取 ----
    def entries(self, limit: Optional[int] = None) -> List[TraceEntry]:
        rows = list(self._entries)
        if limit is not None and limit >= 0:
            rows = rows[-limit:]
        return rows

    def to_list(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.entries(limit)]

    def lines(self, limit: Optional[int] = None) -> List[str]:
        return [e.line() for e in self.entries(limit)]

    def last(self) -> Optional[TraceEntry]:
        return self._entries[-1] if self._entries else None

    def __len__(self) -> int:
        return len(self._entries)

    # ---- 聚合 ----
    @property
    def total(self) -> int:
        """累计记录数（含已被环形缓冲淘汰的）。"""
        return self._total

    @property
    def dropped(self) -> int:
        return max(0, self._total - len(self._entries))

    def action_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in self._entries:
            counts[e.action] = counts.get(e.action, 0) + 1
        return counts

    def pressure_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in self._entries:
            counts[e.pressure] = counts.get(e.pressure, 0) + 1
        return counts

    def domain_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in self._entries:
            counts[e.primary] = counts.get(e.primary, 0) + 1
        return counts

    def summary(self) -> Dict[str, Any]:
        rows = list(self._entries)
        latencies = [e.latency_ms for e in rows]
        return {
            "size": len(rows),
            "maxlen": self.maxlen,
            "total": self._total,
            "dropped": self.dropped,
            "applied_rate": (
                sum(1 for e in rows if e.applied) / len(rows) if rows else 0.0
            ),
            "latency_ms": {
                "avg": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
                "max": round(max(latencies), 3) if latencies else 0.0,
            },
            "by_action": self.action_counts(),
            "by_pressure": self.pressure_counts(),
            "by_domain": self.domain_counts(),
        }

    def clear(self) -> None:
        self._entries.clear()
