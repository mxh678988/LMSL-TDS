"""三级存储抽象：显存(VRAM) / 内存(RAM) / 磁盘(SSD)。

所有数据以 Block 为单位管理，支持提升/下沉/淘汰。
淘汰语义 = 降级而非删除（miss 时从下层取，优于全量重算）。
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class Tier(Enum):
    VRAM = 0   # GPU 显存，微秒级
    RAM = 1    # CPU 内存，毫秒级
    SSD = 2    # 本地 NVMe，十毫秒级

    def faster_than(self, other: "Tier") -> bool:
        return self.value < other.value


@dataclass
class Block:
    """一个可调度的数据块（权重层 / KV 分页 / 前缀段）。"""
    id: str
    size_mb: float
    domain: str = "qita"
    tier: Tier = Tier.SSD
    last_used_step: int = 0
    pinned: bool = False     # 常驻保护（如系统指令）
    refcount: int = 0        # 引用计数（前缀共享）


class TieredStore:
    """分层存储管理器。"""

    def __init__(self):
        self._blocks: Dict[str, Block] = {}

    def add(self, block: Block) -> None:
        self._blocks[block.id] = block

    def get(self, block_id: str) -> Optional[Block]:
        return self._blocks.get(block_id)

    def promote(self, block_id: str) -> Optional[Block]:
        """向更快层提升一级。"""
        b = self._blocks.get(block_id)
        if b and b.tier.value > 0:
            b.tier = Tier(b.tier.value - 1)
        return b

    def set_tier(self, block_id: str, tier: Tier) -> Optional[Block]:
        """直接置为指定层（物理动作完成后账本同步，防错位）。

        与 promote 的区别：promote 只升一级（保守、供通用链路上调），
        而 adapter.load/prefetch 是物理加载到目标层（VRAM），账本必须
        一步到位，否则 pin 后块仍在 VRAM 却记作 RAM，导致重复加载与
        命中率误判（问题2）。
        """
        b = self._blocks.get(block_id)
        if b is not None:
            b.tier = tier
        return b

    def demote(self, block_id: str) -> Optional[Block]:
        """向更慢层下沉一级。"""
        b = self._blocks.get(block_id)
        if b and b.tier.value < Tier.SSD.value and not b.pinned:
            b.tier = Tier(b.tier.value + 1)
        return b

    def evict(self, block_id: str) -> Optional[Block]:
        """淘汰 = 降级到最慢层（非删除）。"""
        b = self._blocks.get(block_id)
        if b and not b.pinned:
            b.tier = Tier.SSD
        return b

    def blocks_in(self, tier: Tier) -> List[Block]:
        return [b for b in self._blocks.values() if b.tier == tier]
