"""显存预算、水位监控与压力保护。"""
from dataclasses import dataclass

from ..config import TDSConfig


@dataclass
class VramState:
    """某时刻显存快照（由适配层填充真实值）。"""
    total_mb: int
    used_mb: int
    free_mb: int = 0
    fragmentation_mb: int = 0

    @property
    def usage_ratio(self) -> float:
        return self.used_mb / self.total_mb if self.total_mb else 0.0


class MemoryManager:
    """显存预算与压力保护。"""

    def __init__(self, config: TDSConfig):
        self.cfg = config

    def effective_budget(self) -> int:
        """可用预算 = 配置预算 - 预留。"""
        return max(0, self.cfg.vram_budget_mb - self.cfg.reserve_mb)

    def under_pressure(self, state: VramState) -> bool:
        """是否处于显存压力状态。"""
        return state.usage_ratio >= self.cfg.pressure_ratio

    def can_fit(self, size_mb: float, state: VramState) -> bool:
        """是否还有空间放下指定大小的块（含预留）。"""
        headroom = state.total_mb - state.used_mb - self.cfg.reserve_mb
        return size_mb <= headroom
