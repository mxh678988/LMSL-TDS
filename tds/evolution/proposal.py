"""建议书（ProposalBook）：进化引擎的"只建议不自动应用"出口。

risk=medium/high 的候选变更只会进入建议书，绝不自动应用。
建议书可落盘为 JSON（供人工审阅），也可渲染为可读文本。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .whitelist import Risk, risk_rank

Path = Tuple[str, ...]


@dataclass
class Proposal:
    """一条进化候选变更。"""
    param: str                 # 点号路径
    old_value: Any = None
    new_value: Any = None
    risk: Risk = Risk.LOW
    reason: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    applied: bool = False
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": round(self.ts, 3),
            "param": self.param,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "risk": self.risk.value,
            "reason": self.reason,
            "evidence": dict(self.evidence),
            "applied": self.applied,
        }

    def line(self) -> str:
        state = "已应用" if self.applied else "待审阅"
        return "[{}] {}: {} -> {} （risk={}, {}) {} {}".format(
            state, self.param, self.old_value, self.new_value,
            self.risk.value, self.reason, self.evidence, "")


class ProposalBook:
    """建议书集合：按风险分组，支持落盘与文本渲染。"""

    def __init__(self, file_path: Optional[str] = None):
        self._items: List[Proposal] = []
        self.file_path = file_path

    def add(self, proposal: Proposal) -> None:
        self._items.append(proposal)

    def extend(self, proposals: List[Proposal]) -> None:
        self._items.extend(proposals)

    def items(self) -> List[Proposal]:
        return list(self._items)

    def items_by_risk(self, risk: Risk) -> List[Proposal]:
        return [p for p in self._items if p.risk == risk]

    def applied_items(self) -> List[Proposal]:
        return [p for p in self._items if p.applied]

    def pending_items(self) -> List[Proposal]:
        return [p for p in self._items if not p.applied]

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()

    # ---- 持久化 ----
    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at": time.time(),
            "total": len(self._items),
            "pending": len(self.pending_items()),
            "proposals": [p.to_dict() for p in self._items],
        }

    def save(self, file_path: Optional[str] = None) -> str:
        target = file_path or self.file_path
        if not target:
            return ""
        d = os.path.dirname(target)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
        return target

    def render(self) -> str:
        lines = ["===== EvolutionEngine 建议书 ====="]
        for risk in (Risk.HIGH, Risk.MEDIUM, Risk.LOW):
            group = self.items_by_risk(risk)
            if not group:
                continue
            lines.append(f"--- {risk.value.upper()}（{len(group)}）---")
            for p in group:
                state = "已应用" if p.applied else "待审阅"
                lines.append("- [{}] {}: {} -> {} | {} | evidence={}".format(
                    state, p.param, p.old_value, p.new_value,
                    p.reason, p.evidence))
        return "\n".join(lines)
