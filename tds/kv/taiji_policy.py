"""太极·八卦域加权页调度（Taiji Bagua Page Policy, TBP）—— TDS 独有知识产权。

哲学内核
=========
将易经"太极生两仪，两仪生四象，四象生八卦"的生成逻辑合成为 KV 分页换入换出
的新调用逻辑，替代西方 LRU 无差别淘汰：

    太极（一）  →  每层 KV 整体（母张量），不可分割的整体认知
    两仪（阴阳）→  页热度二值：阳 = 近期将使用（对话/乾坤域热页），
                  阴 = 低频/历史（冷页），决定驻留意愿
    四象（压力）→  显存水位四档：少阴(低)/少阳(临界)/老阴(高)/老阳(冻结)，
                  决定动作强度：全驻留 → 换阴页 → 域序淘汰 → 整层卸载
    八卦（八域）→  页所属语义域（乾坤/对话/检索/代码/图像/视频/工具/其他），
                  按 BAGUA_KEEP_STEPS 加权出"每域保活步数"，取代固定阈值

合成后的调用逻辑（与传统 LRU 对比）
==================================
    传统：last_used_step 最小者先淘汰（无差别、无语义、一步到底）
    太极：候选排序 = f(两仪阴阳, 八卦域保活步数, 四象压力档位, 热度)
          - 阳页永驻（对话/乾坤域在保活窗口内，不被换出）
          - 阴页按八卦优先级逐域淘汰（视频>图像>代码>对话>检索>工具>其他）
          - 压力档位决定淘汰深度：临界只淘汰阴页；高压淘汰所有阴页并下调
            保活窗口；冻结整层卸载到 RAM/SSD（三才：天VRAM→地RAM→人SSD）

设计约束
=========
- 纯函数/数据驱动：不 import torch，不触碰宿主，仅输入页元数据输出决策
- 与现有 PageTable / PageCache 解耦：决策与搬运分离，可回退到 LRU
- 默认配置下与 LRU 等价可回退（保活步数 = 1 时阳页即"上一步刚用"）
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---- 四象：压力档位（与 core.scheduler PressureState 对齐） ----
class PressureState(enum.Enum):
    LOW = "low"            # 少阴：显存充裕，全驻留
    CRITICAL = "critical"  # 少阳：逼近阈值，淘汰阴页
    HIGH = "high"          # 老阴：超阈值，域序淘汰
    FROZEN = "frozen"      # 老阳：冻结，整层卸载

    @classmethod
    def from_ratio(cls, ratio: float, pressure_ratio: float = 0.92,
                   frozen_ratio: float = 0.98) -> "PressureState":
        if ratio >= frozen_ratio:
            return cls.FROZEN
        if ratio >= pressure_ratio:
            return cls.HIGH
        if ratio >= 0.85:
            return cls.CRITICAL
        return cls.LOW


# ---- 两仪：页阴阳标签 ----
YANG = "yang"  # 阳：近期将使用，倾向驻留
YIN = "yin"    # 阴：低频/历史，倾向换出

# ---- 八卦：八域保活步数（默认与 config.BAGUA_KEEP_STEPS 对齐） ----
BAGUA_KEEP_STEPS = {
    "qian":    1,   # 乾坤：系统指令，永久保护（pinned 语义）
    "duihua":  3,   # 对话：上下文连贯，多轮复用
    "jiansuo": 2,   # 检索：片段复用率中等
    "daima":   2,   # 代码：逻辑连续，块间引用
    "tuxiang": 2,   # 图像：单次为主，偶有迭代
    "shipin":  1,   # 视频：单次生成为主
    "gongju":  1,   # 工具：短暂调用
    "qita":    1,   # 其他：默认保守
}

# 八卦淘汰优先级（并列时谁先被淘汰；重资源域保护优先，冷域先淘汰）
BAGUA_EVICT_ORDER = ["qita", "gongju", "jiansuo", "shipin", "tuxiang",
                     "daima", "duihua", "qian"]


# ---- 页元数据（与 Page 解耦的决策输入） ----
@dataclass
class PageMeta:
    """太极策略所需的页元数据（可从 Page 提取，无 torch 依赖）。"""
    page_id: str
    domain: str = "qita"          # 八卦域标签
    last_used_step: int = 0       # 最后使用步数
    access_count: int = 0         # 访问计数
    pinned: bool = False          # 乾坤域永久保护
    size_mb: float = 0.0          # 页占用（MB，用于预算核算）
    tier: int = 0                 # 0=VRAM 1=RAM 2=SSD（三才层级）


@dataclass
class PageDecision:
    """单页决策结果。"""
    page_id: str
    action: str                    # keep / evict_ram / evict_ssd / prefetch
    tag: str = YIN                 # 两仪标签
    domain: str = "qita"           # 八卦域
    reason: str = ""               # 决策理由（含哲学依据）
    priority: float = 0.0          # 排序权重（越大越先执行）

    def to_dict(self) -> dict:
        return {
            "page_id": self.page_id,
            "action": self.action,
            "tag": self.tag,
            "domain": self.domain,
            "reason": self.reason,
            "priority": self.priority,
        }


@dataclass
class TaijiPagePolicyConfig:
    """太极页策略配置。"""
    keep_steps: Dict[str, int] = field(default_factory=lambda: dict(BAGUA_KEEP_STEPS))
    evict_order: List[str] = field(default_factory=lambda: list(BAGUA_EVICT_ORDER))
    pressure_ratio: float = 0.92
    frozen_ratio: float = 0.98
    # 高压下调保活窗口的衰减因子（老阴压境，阳页窗口收缩）
    high_keep_decay: float = 0.5
    # 是否启用太极策略（False 回退传统 LRU）
    enabled: bool = True


class TaijiPagePolicy:
    """太极页策略引擎：输入页元数据，输出页级换入换出决策。

    调用逻辑（合成新逻辑，区别于 LRU）：
        1. 两仪定阴阳：distance <= 域保活步数 → 阳（驻留），否则 → 阴（换出）
        2. 四象定强度：LOW 全驻留；CRITICAL 只动阴页；HIGH 域序淘汰阴页；
           FROZEN 整层卸载（天VRAM→地RAM→人SSD）
        3. 八卦定序：阴页候选按 evict_order 域序 + 热度（last_used_step,
           access_count）排序，重资源域最后淘汰
    """

    def __init__(self, config: Optional[TaijiPagePolicyConfig] = None):
        self.cfg = config or TaijiPagePolicyConfig()

    # ---- 两仪：页阴阳判定 ----
    def tag_of(self, domain: str, distance: int) -> str:
        """阳 = 近期将使用（距离 <= 域保活步数）；阴 = 冷页。"""
        ks = self.cfg.keep_steps.get(domain, self.cfg.keep_steps["qita"])
        return YANG if distance <= ks else YIN

    # ---- 八卦：域保活步数 ----
    def keep_steps_of(self, domain: str) -> int:
        return self.cfg.keep_steps.get(domain, self.cfg.keep_steps["qita"])

    # ---- 四象压力衰减 ----
    def _effective_keep_steps(self, domain: str, pressure: PressureState) -> int:
        ks = self.keep_steps_of(domain)
        if pressure == PressureState.HIGH:
            ks = max(1, int(ks * self.cfg.high_keep_decay))
        return ks

    # ---- 核心：决策 ----
    def decide(self, pressure: PressureState, pages: List[PageMeta],
               current_step: int, budget_mb: Optional[float] = None
               ) -> List[PageDecision]:
        """对一批页做出决策，返回按执行优先级排序的决策列表。

        参数：
            pressure   四象压力档位
            pages      页元数据列表（仅需驻留页，即 is_in_vram=True）
            current_step 当前推理步数
            budget_mb  显存预算（MB）；None 表示不限制（全驻留）

        返回：PageDecision 列表，按 priority 降序（先执行高优先级的动作）。
        """
        if not self.cfg.enabled or pressure == PressureState.LOW:
            # 低压 / 禁用：太极不动，全驻留（零开销，等价默认）
            return [PageDecision(p.page_id, "keep", tag=YANG,
                                 domain=p.domain,
                                 reason="少阴·低压或禁用：全驻留零开销")
                    for p in pages]

        # 冻结：整层卸载到 RAM（三才：天→地），所有页统一动作
        if pressure == PressureState.FROZEN:
            return [PageDecision(p.page_id, "evict_ram", tag=YIN,
                                 domain=p.domain,
                                 reason="老阳·冻结：整层卸载（天VRAM→地RAM）")
                    for p in pages]

        # 高压：强制所有阴页换出，且保活窗口收缩
        if pressure == PressureState.HIGH:
            decisions: List[PageDecision] = []
            for p in pages:
                ks = self._effective_keep_steps(p.domain, pressure)
                distance = current_step - p.last_used_step
                if distance <= ks and not p.pinned:
                    decisions.append(PageDecision(
                        p.page_id, "keep", tag=YANG, domain=p.domain,
                        reason=f"老阴·高压：域{ p.domain}保活窗口收缩至{ks}步仍为阳",
                        priority=0.5))
                else:
                    decisions.append(PageDecision(
                        p.page_id, "evict_ram", tag=YIN, domain=p.domain,
                        reason="老阴·高压：阴页域序淘汰",
                        priority=1.0))
            return self._order(decisions)

        # 临界：只淘汰阴页（阳页永驻），预算超额时再按域序补充淘汰
        if pressure == PressureState.CRITICAL:
            decisions = []
            for p in pages:
                distance = current_step - p.last_used_step
                if distance <= self.keep_steps_of(p.domain) or p.pinned:
                    decisions.append(PageDecision(
                        p.page_id, "keep", tag=YANG, domain=p.domain,
                        reason="少阳·临界：阳页保活（对话/乾坤域窗口内）",
                        priority=0.3))
                else:
                    decisions.append(PageDecision(
                        p.page_id, "evict_ram", tag=YIN, domain=p.domain,
                        reason="少阳·临界：阴页换出",
                        priority=0.8))
            return self._order(decisions)

        # 兜底：全驻留
        return [PageDecision(p.page_id, "keep", tag=YANG, domain=p.domain,
                             reason="太极不动：全驻留")
                for p in pages]

    # ---- 八卦：淘汰排序（独有：域序 + 热度，而非纯 LRU） ----
    def _order(self, decisions: List[PageDecision]) -> List[PageDecision]:
        """按执行优先级排序：evict 动作优先于 keep；evict 之间按
        八卦域淘汰序 + 热度（越久未用越先淘汰）。"""
        order_idx = {d: i for i, d in enumerate(self.cfg.evict_order)}

        def sort_key(d: PageDecision):
            # 动作权重：evict=2 > keep=0（淘汰动作先执行）
            action_w = 2 if d.action.startswith("evict") else 0
            # 域淘汰序：靠前（qita=0）先淘汰；keep 的域序不参与
            dom_w = order_idx.get(d.domain, 7) if d.action.startswith("evict") else 0
            return (-action_w, dom_w, -d.priority)
        return sorted(decisions, key=sort_key, reverse=False)

    # ---- 预算核算：若驻留总需求超预算，从阴页开始补充淘汰 ----
    def fit_budget(self, pages: List[PageMeta], budget_mb: Optional[float]
                   ) -> List[str]:
        """按八卦域序淘汰阴页，把驻留总量压回预算内；返回需换出的 page_id。

        这是太极的"量入为出"：预算即太极之"度"，域序即"序"，缺一不可。
        """
        if budget_mb is None:
            return []
        resident = [p for p in pages if p.tier == 0]
        total = sum(p.size_mb for p in resident)
        if total <= budget_mb:
            return []
        evict_ids: List[str] = []
        # 阴页候选：pinned 与阳页不在此列（由调用方保证 pages 均为阴页）
        ordered = sorted(
            resident,
            key=lambda p: (self.cfg.evict_order.index(p.domain)
                           if p.domain in self.cfg.evict_order else 7,
                           -p.last_used_step))
        for p in ordered:
            if p.pinned:
                continue
            evict_ids.append(p.page_id)
            total -= p.size_mb
            if total <= budget_mb:
                break
        return evict_ids

    # ---- 预读决策：阳页预取（即将变热即预取） ----
    def should_prefetch(self, domain: str, distance: int,
                        pressure: Optional["PressureState"] = None) -> bool:
        """阳域页是否应预取：距离在保活窗口前沿（即将变热）。

        四象压力门控预取窗口（避免紧张预算下预取反而挤占本层消费页）：
          LOW       全窗口预取（预算充裕，提前驻留）
          CRITICAL  仅最近邻（0 < distance <= 1），避免挤占预算
          HIGH / FROZEN  不预取（预算紧张，预取造成抖动反效果）
        """
        if pressure is None:
            pressure = PressureState.LOW
        if pressure in (PressureState.HIGH, PressureState.FROZEN):
            return False
        if pressure == PressureState.CRITICAL:
            return 0 < distance <= 1
        ks = self.keep_steps_of(domain)
        return 0 < distance <= ks + 1
