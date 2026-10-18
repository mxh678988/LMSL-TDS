"""平台适配器接口：核心调度逻辑与具体运行时之间的唯一契约。

core 只依赖本接口；新增平台（PC / 移动 / 新框架）只需实现本契约。
"""
from abc import ABC, abstractmethod
from typing import List

from ..core.memory import VramState


class BaseAdapter(ABC):
    """适配器统一契约。"""

    @abstractmethod
    def scan_usage(self, graph) -> List[str]:
        """预扫描执行图，返回资源/模型使用序列。"""
        raise NotImplementedError

    @abstractmethod
    def load(self, block_id: str) -> bool:
        """将块加载到更快层。"""
        raise NotImplementedError

    @abstractmethod
    def unload(self, block_id: str) -> bool:
        """将块从更快层释放/下沉。"""
        raise NotImplementedError

    @abstractmethod
    def prefetch(self, block_id: str) -> bool:
        """预取块（异步/提前唤醒）。"""
        raise NotImplementedError

    @abstractmethod
    def vram_state(self) -> VramState:
        """返回当前显存快照。"""
        raise NotImplementedError
