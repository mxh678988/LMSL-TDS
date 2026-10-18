"""知识产权进化引擎（IP EvolutionEngine）—— 知识资产的独立进化线。

设计哲学（太极·知识产权进化）
=============================
    "天地定位，山泽通气" —— 调度层的"知识"（八卦域分类规则、分类词典、
    域知识权重）不是静态版权，而是可自生长的知识资产：

        阴阳增删知识  →  高置信新词条（阳）加入词典 / 失效词条（阴）降权
        三才存储      →  词典/权重/元数据分层存放于 IP 档案
        四象阶段控制  →  高压/冻结下禁止知识变更（与总体进化一致）
        八卦域独立学习→  每个域单独统计词条命中与误判，互不干扰

与总体进化线（engine.py）的关系
------------------------------
- 总体进化：管"运行规则参数"（BAGUA_KEEP_STEPS / pressure_ratio / 阴阳权重）
- 知识产权进化：管"分类知识"（分类词典关键词表、域知识权重、域别名）
- 双线档案独立（software_archive.json vs ip_archive.json）、模块独立、
  策略独立，绝不混用。

默认关闭：enabled=False 时 evolve() 为 no-op，不改变任何分类行为。
零第三方依赖（仅标准库）。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .archive import EvolutionArchive, dotted, get_path, set_path
from .config import EvolutionConfig
from .proposal import Proposal, ProposalBook
from .whitelist import Risk

PRESSURE_BLOCKED = ("high", "frozen")

# IP 档案结构骨架（实际以档案内容为准）
IP_SKELETON_SECTION = "knowledge"


@dataclass
class KnowledgeEntry:
    """一条知识资产：分类词条或域知识权重。"""
    key: str                 # 词条原文或权重路径
    domain: str              # 所属八卦域
    kind: str = "term"       # "term" | "weight" | "alias"
    value: Any = None        # term: 1.0（初始置信）, weight: 数值
    source: str = ""         # 来源（人工/进化/归档）
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "domain": self.domain, "kind": self.kind,
            "value": self.value, "source": self.source, "ts": round(self.ts, 3),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KnowledgeEntry":
        return cls(
            key=str(d.get("key", "")), domain=str(d.get("domain", "qita")),
            kind=str(d.get("kind", "term")), value=d.get("value"),
            source=str(d.get("source", "")), ts=float(d.get("ts", 0.0)),
        )


class KnowledgeStore:
    """IP 知识存取门面：统一读写 IP 档案中的知识资产。"""

    def __init__(self, archive: EvolutionArchive):
        self.archive = archive
        data = self.archive.get_data()
        if IP_SKELETON_SECTION not in data:
            data[IP_SKELETON_SECTION] = {"terms": {}, "weights": {}, "aliases": {}}
            self.archive.save()
        self._section = data[IP_SKELETON_SECTION]

    def _terms(self) -> Dict[str, Any]:
        t = self._section.get("terms")
        if not isinstance(t, dict):
            t = {}
            self._section["terms"] = t
        return t

    def _weights(self) -> Dict[str, Any]:
        w = self._section.get("weights")
        if not isinstance(w, dict):
            w = {}
            self._section["weights"] = w
        return w

    def _aliases(self) -> Dict[str, Any]:
        a = self._section.get("aliases")
        if not isinstance(a, dict):
            a = {}
            self._section["aliases"] = a
        return a

    # ---- 词条 ----
    def add_term(self, term: str, domain: str, confidence: float = 1.0,
                 source: str = "evolution") -> None:
        t = self._terms()
        existing = t.get(term)
        if existing is None:
            t[term] = {"domain": domain, "confidence": confidence,
                       "source": source, "ts": time.time()}
        else:
            # 已有词条：提高置信（上限 1.0），来源标记保留
            existing["confidence"] = min(1.0, float(existing.get("confidence", 1.0)) + 0.05)
            existing["ts"] = time.time()
        self.archive.save()

    def get_term(self, term: str) -> Optional[Dict[str, Any]]:
        return self._terms().get(term)

    def all_terms(self) -> Dict[str, Any]:
        return dict(self._terms())

    def demote_term(self, term: str, delta: float = 0.2) -> bool:
        """降权词条（阴阳增删知识之"阴"：低频/误判词条降权）。"""
        t = self._terms().get(term)
        if t is None:
            return False
        conf = float(t.get("confidence", 1.0)) - delta
        t["confidence"] = max(0.0, conf)
        if t["confidence"] <= 0.0:
            del self._terms()[term]
        self.archive.save()
        return True

    # ---- 域知识权重 ----
    def set_weight(self, domain: str, weight: float, source: str = "evolution") -> None:
        w = self._weights()
        w[domain] = {"weight": weight, "source": source, "ts": time.time()}
        self.archive.save()

    def get_weight(self, domain: str) -> Optional[float]:
        w = self._weights().get(domain)
        return None if w is None else float(w.get("weight", 1.0))

    # ---- 域别名 ----
    def add_alias(self, alias: str, domain: str, source: str = "evolution") -> None:
        self._aliases()[alias] = {"domain": domain, "source": source, "ts": time.time()}
        self.archive.save()

    def get_alias(self, alias: str) -> Optional[str]:
        a = self._aliases().get(alias)
        return None if a is None else str(a.get("domain", "qita"))

    # ---- 全量 ----
    def snapshot(self) -> Dict[str, Any]:
        return {
            "terms": dict(self._terms()),
            "weights": dict(self._weights()),
            "aliases": dict(self._aliases()),
        }


class IPEvolutionEngine:
    """知识产权进化引擎（零依赖）。

    输入
    ----
    - ``observations``：分类观测样本列表，每项形如
        {"domain": "daima", "term": "def", "correct": True/False}
      由宿主（classifier 消费方）在运行时投喂。
    - ``archive``：IP 线独立档案（ip_archive.json）。

    输出
    ----
    - 新词条加入建议（low：自动应用；medium/high：建议书）
    - 误判词条降权/移出建议
    - 域知识权重调整建议
    """

    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        archive: Optional[EvolutionArchive] = None,
        proposal_dir: Optional[str] = None,
    ):
        self.config = config or EvolutionConfig()
        if archive is None:
            archive = EvolutionArchive(
                os.path.join(self.config.archive_dir, "ip_archive.json"),
                line="ip",
                backup_keep=self.config.backup_keep,
            )
        self.archive = archive
        self.store = KnowledgeStore(archive)
        self.proposal_dir = proposal_dir or self.config.proposal_dir
        self._observations: List[Dict[str, Any]] = []
        self.config.ensure_dirs()

    # ---- 观测投喂 ----
    def observe(self, domain: str, term: str, correct: bool,
                source: str = "host") -> None:
        self._observations.append({
            "domain": domain, "term": term, "correct": bool(correct),
            "source": source, "ts": time.time(),
        })
        # 内存环形：最多保留 learn_window*4 条
        max_keep = max(64, self.config.learn_window * 4)
        if len(self._observations) > max_keep:
            self._observations = self._observations[-max_keep:]

    def observations(self) -> List[Dict[str, Any]]:
        return list(self._observations)

    # ---- 候选生成 ----
    def _candidates(self, obs: List[Dict[str, Any]]) -> List[Proposal]:
        cands: List[Proposal] = []
        # 统计：词条 → (correct, wrong)
        stats: Dict[str, Dict[str, Any]] = {}
        for o in obs:
            term = str(o.get("term", "")).strip()
            if not term:
                continue
            st = stats.setdefault(term, {"correct": 0, "wrong": 0, "domain": o.get("domain", "qita")})
            if o.get("correct"):
                st["correct"] += 1
            else:
                st["wrong"] += 1
                st["domain"] = o.get("domain", st["domain"])
        for term, st in stats.items():
            total = st["correct"] + st["wrong"]
            if total < 2:
                continue
            # 高频正确且尚未入库 → 建议新增词条（low）
            existing = self.store.get_term(term)
            if existing is None and st["correct"] >= 2 and st["wrong"] == 0:
                cands.append(Proposal(
                    param="knowledge.term.{}".format(term),
                    old_value=None,
                    new_value=st["domain"],
                    risk=Risk.LOW,
                    reason="观测到词条 '{}' 被正确分类为 {}（{} 次），建议加入分类词典".format(
                        term, st["domain"], st["correct"]),
                    evidence={"correct": st["correct"], "wrong": st["wrong"]},
                ))
            # 误判率 >40% 且已有词条 → 降权/移出建议（medium）
            elif existing is not None and st["wrong"] >= 2 and st["wrong"] / total > 0.4:
                cands.append(Proposal(
                    param="knowledge.term.{}".format(term),
                    old_value=existing.get("confidence", 1.0),
                    new_value=max(0.0, float(existing.get("confidence", 1.0)) - 0.2),
                    risk=Risk.MEDIUM,
                    reason="词条 '{}' 误判率 {:.0%}，建议降权（当前置信 {:.2f}）".format(
                        term, st["wrong"] / total, float(existing.get("confidence", 1.0))),
                    evidence={"correct": st["correct"], "wrong": st["wrong"]},
                ))
        return cands

    # ---- 主入口 ----
    def evolve(self) -> ProposalBook:
        """执行一轮 IP 进化。默认关闭时 no-op。"""
        book = ProposalBook(self._proposal_file())
        if not self.config.enabled:
            return book
        obs = self._observations
        if len(obs) < self.config.min_samples:
            return book
        # 四象压力守卫（IP 线同样遵守：高压/冻结禁变更）
        # 注：IP 线无 trace 数据源，由宿主注入 pressure 状态（可选）
        if self.config.pressure_guard and self._pressure_blocked():
            self.archive.append_changelog(op="skip", path="*", note="pressure_guard")
            return book
        for p in self._candidates(obs):
            if p.risk == Risk.LOW and self.config.auto_apply_low:
                self._apply(p)
            else:
                book.add(p)
                self.archive.append_changelog(
                    op="suggest", path=p.param, old=p.old_value, new=p.new_value,
                    risk=p.risk.value, note=p.reason,
                )
        if self.proposal_dir:
            book.save()
        return book

    def _apply(self, proposal: Proposal) -> None:
        """低风险自动应用：新词条入库（备份旧状态写 changelog）。"""
        key = proposal.param[len("knowledge.term."):]
        if proposal.old_value is None:
            self.store.add_term(key, str(proposal.new_value), source="evolution")
            self.archive.backup_and_change(
                "knowledge.term.{}".format(key), None, proposal.new_value,
                risk="low", note=proposal.reason,
            )
        else:
            self.store.demote_term(key, delta=0.2)
            self.archive.backup_and_change(
                "knowledge.term.{}".format(key), proposal.old_value,
                proposal.new_value, risk="low", note=proposal.reason,
            )
        proposal.applied = True

    def apply_all(self, book: ProposalBook) -> int:
        """人工审阅后批量应用 medium/high 建议。返回应用条数。"""
        n = 0
        for p in book.pending_items():
            if p.risk == Risk.LOW:
                continue
            key = p.param[len("knowledge.term."):]
            if p.old_value is None:
                self.store.add_term(key, str(p.new_value), source="manual")
                self.archive.backup_and_change(
                    "knowledge.term.{}".format(key), None, p.new_value,
                    risk=p.risk.value, note=p.reason,
                )
            else:
                self.store.demote_term(key, delta=0.4)
                self.archive.backup_and_change(
                    "knowledge.term.{}".format(key), p.old_value, p.new_value,
                    risk=p.risk.value, note=p.reason,
                )
            p.applied = True
            n += 1
        return n

    # ---- 压力守卫（宿主可注入） ----
    def _pressure_blocked(self) -> bool:
        data = self.archive.get_data()
        return bool(data.get("pressure_blocked", False))

    def set_pressure_blocked(self, blocked: bool) -> None:
        data = self.archive.get_data()
        data["pressure_blocked"] = bool(blocked)
        self.archive.save()

    # ---- 辅助 ----
    def _proposal_file(self) -> Optional[str]:
        if not self.proposal_dir:
            return None
        return os.path.join(self.proposal_dir, "ip_proposals.json")

    def snapshot(self) -> Dict[str, Any]:
        return {
            "observations": self._observations[-200:],
            "knowledge": self.store.snapshot(),
        }
