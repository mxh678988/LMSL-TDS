"""预取计划器：预扫描 + Belady 近似排序 + 计划生成。"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..config import TDSConfig
from .storage import Tier


@dataclass
class PlanItem:
    block_id: str
    action: str          # prefetch / pin / demote / evict
    target_tier: Tier
    priority: float = 0.0


@dataclass
class Plan:
    items: List[PlanItem] = field(default_factory=list)

    def summary(self) -> str:
        if not self.items:
            return "Plan(empty)"
        by_action: Dict[str, int] = {}
        for it in self.items:
            by_action[it.action] = by_action.get(it.action, 0) + 1
        return "Plan(" + ", ".join(f"{k}={v}" for k, v in by_action.items()) + ")"


class Planner:
    """产出预取/保护/淘汰计划。"""

    def __init__(self, config: TDSConfig):
        self.cfg = config

    def scan_model_usage(self, graph) -> List[str]:
        """预扫描执行图，返回模型/块使用序列。

        具体解析由适配层提供（例如 ComfyUI 的 prompt 执行图）。
        骨架返回空序列。
        """
        return []

    def rank_by_next_use(self, seq: List[str], history: Dict[str, int]) -> List[Tuple[str, int]]:
        """按"下次使用距离"排序（Belady 近似：距离越近越优先保护）。"""
        ranked: List[Tuple[str, int]] = []
        for i, block_id in enumerate(seq):
            distance = i - history.get(block_id, -1)
            ranked.append((block_id, distance))
        ranked.sort(key=lambda kv: kv[1])
        return ranked

    def plan(self, profile, blocks=None, pressure="low",
             current_step: int = 0, keep_steps_fn=None) -> Plan:
        """结合任务画像 + 阴阳标签 + 四象压力，产出调度计划。

        规则组合（非枚举）：
          - 乾坤域 → 永久 pin
          - 阳（distance <= keep_steps）→ pin（驻留快层）
          - 阴（distance >  keep_steps）→ demote（下沉一层）
          - 四象覆盖：frozen→全部 evict；high→不预取；critical→仅保活；low→可预取
        """
        plan = Plan()
        ks_fn = keep_steps_fn or (lambda _d: 1)

        # 1) 系统域（乾坤）永久保护
        if profile is not None and getattr(profile, "primary", None) == "qian":
            plan.items.append(PlanItem("sys", "pin", Tier.VRAM, priority=1.0))

        if not blocks:
            return plan

        for b in blocks:
            # 常驻块不受压力影响
            if b.pinned:
                plan.items.append(PlanItem(b.id, "pin", Tier.VRAM, priority=1.0))
                continue

            if pressure == "frozen":
                plan.items.append(PlanItem(b.id, "evict", Tier.SSD, priority=0.0))
                continue

            if pressure == "high":
                # 高压力：禁止上载（apply 侧裁剪 pin），非常驻块一律下沉让位，
                # 避免"VRAM 存量全为阳块 → 无释放 → 高压持续"的收缩空转。
                plan.items.append(PlanItem(b.id, "demote", Tier.RAM, priority=0.2))
                continue

            if pressure == "critical":
                # 临界：仅保活（不 pin 新块、不预取），只做释放类动作，避免加剧压力
                plan.items.append(PlanItem(b.id, "demote", Tier.RAM, priority=0.2))
                continue

            distance = current_step - b.last_used_step
            is_yang = distance <= ks_fn(b.domain)

            if is_yang:
                plan.items.append(PlanItem(b.id, "pin", Tier.VRAM, priority=1.0))
                if pressure == "low" and self.cfg.prefetch_enabled:
                    plan.items.append(PlanItem(b.id, "prefetch", Tier.VRAM, priority=0.5))
            else:
                plan.items.append(PlanItem(b.id, "demote", Tier.RAM, priority=0.2))

        return plan
