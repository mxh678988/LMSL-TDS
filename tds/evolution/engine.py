"""软件总体进化引擎（EvolutionEngine）—— 运行规则的自学习进化线。

设计哲学（太极·总体进化）
=========================
    "道生一，一生二，二生三，三生万物" —— 调度规则不是人工写死的，
    而是由决策轨迹（决策之迹）与性能指标（成效之数）持续自我演化：

        采集（阴）  →  决策轨迹 DecisionTrace + 性能指标 Metrics
        学习（阳）  →  从数据中提炼"运行规律"信号，形成参数候选变更
        应用（变）  →  低风险候选自动应用（备份+changelog），
                      中/高风险只出建议书供人工审阅
        沉淀（藏）  →  每次变更写入进化档案，可审计、可回退

职责边界
--------
1. 只动"运行规则参数"（白名单 whitelist.DEFAULT_WHITELIST）：
   BAGUA_KEEP_STEPS / pressure_ratio / frozen_ratio /
   yang_prefetch_advance_steps / 阴阳权重等；
2. 知识资产（分类词典/域知识权重）由 ip_engine（知识产权进化线）独立管理，
   本引擎不越界；
3. 默认关闭（enabled=False）：不接入任何 hook、不改变既有调度行为；
4. 高压/冻结（四象）下即使 enabled=True 也禁止任何变更；
5. 零第三方依赖（仅标准库）。

学习信号示例
------------
- 某域淘汰频繁且命中率下降  →  建议增大该域 keep_steps（low，自动应用）
- 全局命中率长期 >0.99      →  建议缩小保活窗口（low）
- 高压档位占比显著上升      →  建议调整压力阈值（medium，只建议）
- 出现冻结档位              →  建议收紧冻结阈值（high，只建议）
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.metrics import Metrics
from ..core.trace import DecisionTrace
from .archive import EvolutionArchive, dotted, get_path, set_path
from .config import EvolutionConfig
from .proposal import Proposal, ProposalBook
from .whitelist import Path, ParamSpec, Risk, Whitelist, classify_risk, risk_rank

# 四象压力档位（与 core/taiji_policy 对齐的字符串）
PRESSURE_BLOCKED = ("high", "frozen")  # 高压/冻结：禁用变更


class ParameterStore:
    """参数存取门面：统一读取/写入白名单参数。

    设计
    ----
    - ``provider`` 为可选的回调 ``(Path) -> value``，用于从真实配置
      （taiji_policy.BAGUA_KEEP_STEPS / YangYinConfig 等）读取当前值；
      未提供时回退到 ``defaults``（由引擎初始化时注入）。
    - ``overlay`` 为进化覆盖层：apply() 写入覆盖值后，get() 优先返回
      覆盖值——真实宿主可在 apply 回调中同步到运行时配置。
    - 未命中白名单的路径一律拒绝读写（安全边界）。
    """

    def __init__(
        self,
        whitelist: Whitelist,
        provider: Optional[Callable[[Path], Any]] = None,
        defaults: Optional[Dict[Path, Any]] = None,
    ):
        self.whitelist = whitelist
        self._provider = provider
        self._defaults: Dict[Path, Any] = dict(defaults or {})
        self._overlay: Dict[Path, Any] = {}

    def get(self, path: Path) -> Optional[Any]:
        spec = self.whitelist.get(path)
        if spec is None:
            return None
        if path in self._overlay:
            return self._overlay[path]
        if self._provider is not None:
            val = self._provider(path)
            if val is not None:
                return val
        return self._defaults.get(path)

    def apply(self, path: Path, value: Any) -> Any:
        """写入覆盖层并按白名单 clamp；返回最终值。未命中白名单抛 KeyError。"""
        spec = self.whitelist.get(path)
        if spec is None:
            raise KeyError("path not in whitelist: {}".format(dotted(path)))
        final = spec.clamp(value)
        self._overlay[path] = final
        return final

    def overlay_items(self) -> List[Tuple[Path, Any]]:
        return list(self._overlay.items())

    def reset(self, path: Path) -> bool:
        if path in self._overlay:
            del self._overlay[path]
            return True
        return False


class EvolutionEngine:
    """软件总体进化引擎（零依赖）。

    用法
    ----
        engine = EvolutionEngine(
            config=EvolutionConfig(enabled=True, learn_window=200, min_samples=10),
            metrics=metrics, trace=trace, store=store, archive=archive,
        )
        book = engine.evolve()          # 返回本轮建议书（含自动应用结果）
        engine.apply_all(book)          # （可选）人工审阅后批量应用 medium/high
    """

    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        metrics: Optional[Metrics] = None,
        trace: Optional[DecisionTrace] = None,
        store: Optional[ParameterStore] = None,
        whitelist: Optional[Whitelist] = None,
        archive: Optional[EvolutionArchive] = None,
        proposal_dir: Optional[str] = None,
        defaults: Optional[Dict[Path, Any]] = None,
    ):
        self.config = config or EvolutionConfig()
        self.metrics = metrics
        self.trace = trace
        self.whitelist = whitelist or Whitelist()
        if defaults is None:
            defaults = _auto_defaults(self.whitelist)
        self.store = store or ParameterStore(self.whitelist, defaults=defaults)
        if archive is None:
            archive = EvolutionArchive(
                os_path_join(self.config.archive_dir, "software_archive.json"),
                line="software",
                backup_keep=self.config.backup_keep,
            )
        self.archive = archive
        self.proposal_dir = proposal_dir or self.config.proposal_dir
        self._last_signals: Dict[str, Any] = {}
        self.config.ensure_dirs()

    # ---- 学习信号 ----
    def _collect_signals(self) -> Dict[str, Any]:
        """从 trace/metrics 提炼运行规律信号。"""
        signals: Dict[str, Any] = {}
        if self.trace is not None:
            signals["trace_summary"] = self.trace.summary()
            recent = self.trace.entries(limit=self.config.learn_window)
            signals["window_size"] = len(recent)
            # 四象压力分布
            pressure_counts: Dict[str, int] = {}
            for e in recent:
                pressure_counts[e.pressure] = pressure_counts.get(e.pressure, 0) + 1
            signals["pressure_counts"] = pressure_counts
            # 域淘汰频次（action 含 evict 的按域统计）
            evict_by_domain: Dict[str, int] = {}
            for e in recent:
                if "evict" in e.action:
                    evict_by_domain[e.primary] = evict_by_domain.get(e.primary, 0) + 1
            signals["evict_by_domain"] = evict_by_domain
            # 预取成功/失败
            prefetch_ok = sum(1 for e in recent if e.action == "prefetch" and e.applied)
            prefetch_skip = sum(1 for e in recent if e.action == "skip")
            signals["prefetch_ok"] = prefetch_ok
            signals["prefetch_skip"] = prefetch_skip
        if self.metrics is not None:
            signals["hit_rate"] = self.metrics.hit_rate()
            signals["metrics_report"] = self.metrics.report()
        return signals

    # ---- 候选生成 ----
    def _candidates(self, signals: Dict[str, Any]) -> List[Proposal]:
        """基于信号生成候选变更（白名单内）。"""
        cands: List[Proposal] = []
        trace_summary = signals.get("trace_summary", {})
        pressure_counts = signals.get("pressure_counts", {})
        evict_by_domain = signals.get("evict_by_domain", {})
        hit_rate = signals.get("hit_rate")
        window = max(1, int(signals.get("window_size", 0)))

        def add(path: Path, old: Any, new: Any, reason: str, evidence: Dict[str, Any]) -> None:
            spec = self.whitelist.get(path)
            if spec is None or old is None or new is None:
                return
            risk = classify_risk(spec, float(old), float(new))
            cands.append(Proposal(
                param=dotted(path), old_value=old, new_value=new,
                risk=risk, reason=reason, evidence=evidence,
            ))

        # 信号1：某域淘汰频繁 + 命中率低于 0.95 → 增大该域 keep_steps
        high_evict_threshold = max(3, int(window * 0.2))
        for domain, cnt in evict_by_domain.items():
            if cnt >= high_evict_threshold and (hit_rate is None or hit_rate < 0.95):
                path = ("BAGUA_KEEP_STEPS", domain)
                old = self.store.get(path)
                if old is not None:
                    new = float(old) + 1.0
                    add(path, old, new,
                        "域 {} 淘汰 {} 次且命中率偏低，建议增大保活窗口".format(domain, cnt),
                        {"evict_count": cnt, "hit_rate": hit_rate})

        # 信号2：命中率 >0.99 且窗口样本充足 → 全局缩小保活窗口（逐域试探）
        if hit_rate is not None and hit_rate >= 0.99 and window >= self.config.min_samples:
            for path in self.whitelist.paths():
                if len(path) == 2 and path[0] == "BAGUA_KEEP_STEPS":
                    old = self.store.get(path)
                    if old is not None and float(old) > 1:
                        new = float(old) - 1.0
                        add(path, old, new,
                            "全局命中率 {:.3f} 长期高位，尝试收窄该域保活窗口".format(hit_rate),
                            {"hit_rate": hit_rate, "window": window})

        # 信号3：高压档位占比显著上升 → 建议调整压力阈值（medium）
        total = sum(pressure_counts.values())
        if total >= self.config.min_samples:
            high_ratio = pressure_counts.get("high", 0) / total
            frozen_ratio_obs = pressure_counts.get("frozen", 0) / total
            if high_ratio > 0.25:
                path = ("pressure_ratio",)
                old = self.store.get(path)
                if old is not None:
                    new = float(old) - 0.01
                    add(path, old, new,
                        "高压档占比 {:.0%} 偏高，建议下调压力触发阈值".format(high_ratio),
                        {"high_ratio": high_ratio})
            if frozen_ratio_obs > 0.05:
                path = ("frozen_ratio",)
                old = self.store.get(path)
                if old is not None:
                    new = float(old) - 0.01
                    add(path, old, new,
                        "冻结档出现（占比 {:.0%}），建议收紧冻结阈值".format(frozen_ratio_obs),
                        {"frozen_ratio_obs": frozen_ratio_obs})

        # 信号4：预取大量 skip（阳策略空转）→ 建议缩小预读提前步数
        prefetch_ok = int(signals.get("prefetch_ok", 0))
        prefetch_skip = int(signals.get("prefetch_skip", 0))
        if (prefetch_ok + prefetch_skip) >= self.config.min_samples and prefetch_skip > prefetch_ok * 2:
            path = ("yang_prefetch_advance_steps",)
            old = self.store.get(path)
            if old is not None and float(old) > 1:
                new = float(old) - 1.0
                add(path, old, new,
                    "预取 skip 次数远超成功（{} vs {}），建议缩小预读提前步数".format(
                        prefetch_skip, prefetch_ok),
                    {"prefetch_ok": prefetch_ok, "prefetch_skip": prefetch_skip})
        return cands

    # ---- 主入口 ----
    def evolve(self) -> ProposalBook:
        """执行一轮进化：采集 → 学习 → 应用/建议。默认关闭时 no-op。"""
        book = ProposalBook(self._proposal_file())
        if not self.config.enabled:
            return book
        # 四象压力守卫：高压/冻结禁止变更
        if self.config.pressure_guard and self._pressure_blocked():
            self.archive.append_changelog(
                op="skip", path="*", note="pressure_guard: high/frozen",
            )
            return book
        signals = self._collect_signals()
        self._last_signals = signals
        window = int(signals.get("window_size", 0))
        if window < self.config.min_samples:
            self.archive.append_changelog(
                op="skip", path="*", note="insufficient_samples",
            )
            return book
        for p in self._candidates(signals):
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
        """低风险自动应用：备份旧值 → 写覆盖层 → 档案变更 → 建议书标记。"""
        path = tuple(proposal.param.split("."))
        spec = self.whitelist.get(path)
        if spec is None:
            return
        final = self.store.apply(path, proposal.new_value)
        self.archive.backup_and_change(
            dotted(path), proposal.old_value, final,
            risk=proposal.risk.value, note=proposal.reason,
        )
        proposal.applied = True
        proposal.new_value = final

    def apply_all(self, book: ProposalBook) -> int:
        """人工审阅后批量应用建议书中的 medium/high 候选。返回应用条数。"""
        n = 0
        for p in book.pending_items():
            if p.risk == Risk.LOW:
                continue
            path = tuple(p.param.split("."))
            spec = self.whitelist.get(path)
            if spec is None:
                continue
            final = self.store.apply(path, p.new_value)
            self.archive.backup_and_change(
                dotted(path), p.old_value, final,
                risk=p.risk.value, note=p.reason + " (manual apply)",
            )
            p.applied = True
            p.new_value = final
            n += 1
        return n

    def revert(self, path: Path) -> bool:
        """按档案备份回退某参数。返回是否成功。"""
        rec = self.archive.latest_backup_for(dotted(path))
        if rec is None:
            return False
        self.store.apply(path, rec.old)
        self.archive.append_changelog(
            op="revert", path=dotted(path), old=rec.old, note="manual revert",
        )
        return True

    # ---- 辅助 ----
    def _pressure_blocked(self) -> bool:
        """从最近轨迹判断当前压力是否处于高压/冻结。"""
        if self.trace is None:
            return False
        last = self.trace.last()
        if last is None:
            return False
        return last.pressure in PRESSURE_BLOCKED

    def _proposal_file(self) -> Optional[str]:
        if not self.proposal_dir:
            return None
        return os_path_join(self.proposal_dir, "software_proposals.json")

    def signals(self) -> Dict[str, Any]:
        return dict(self._last_signals)


def os_path_join(*parts: str) -> str:
    """标准库 os.path.join 的薄封装（保持零依赖语义清晰）。"""
    import os
    return os.path.join(*parts)


def _auto_defaults(whitelist: Whitelist) -> Dict[Path, Any]:
    """自动装载白名单参数的当前默认值（延迟 import，保持核心零依赖）。

    来源（只读，不触碰宿主）：
    - tds.kv.taiji_policy：BAGUA_KEEP_STEPS / pressure_ratio / frozen_ratio
    - tds.core.yang_yin_strategy.YangYinConfig 默认字段
    任一来源缺失时跳过该组路径，调用方仍可显式注入 defaults。
    """
    defaults: Dict[Path, Any] = {}
    try:
        from ..kv.taiji_policy import BAGUA_KEEP_STEPS  # type: ignore
        for domain, steps in BAGUA_KEEP_STEPS.items():
            path = ("BAGUA_KEEP_STEPS", domain)
            if whitelist.contains(path):
                defaults[path] = steps
    except Exception:
        pass
    try:
        from ..core.yang_yin_strategy import YangYinConfig  # type: ignore
        cfg = YangYinConfig()
        for f in ("dialogue_yang_weight", "non_dialogue_yin_weight",
                  "yang_prefetch_advance_steps", "yin_fast_response_timeout_steps",
                  "emergency_pressure_ratio", "switching_decay_factor",
                  "max_prefetch_timeout_steps"):
            path = (f,)
            if whitelist.contains(path):
                defaults[path] = getattr(cfg, f)
    except Exception:
        pass
    try:
        from ..kv.taiji_policy import PressureState  # type: ignore
        # pressure_ratio / frozen_ratio 与 PressureState.from_ratio 默认一致
        for path, val in ((("pressure_ratio",), 0.92),
                          (("frozen_ratio",), 0.98)):
            if whitelist.contains(path):
                defaults[path] = val
    except Exception:
        pass
    return defaults
