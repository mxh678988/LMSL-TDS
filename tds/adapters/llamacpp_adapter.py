"""llama.cpp / Ollama 适配器（PC）。

理念：通过公开接口控制"层 offload"与 mmap 预取，不做引擎内部改写。
  - 层数：动态调整 n_gpu_layers（公开参数）
  - 显存：读取引擎公开日志 / API
  - 预取：mmap advice（MADV_WILLNEED）触发按需读入
"""
from typing import List, Optional

from ..config import TDSConfig
from ..core.memory import VramState
from .base import BaseAdapter


class LlamaCppAdapter(BaseAdapter):

    def __init__(self, config: Optional[TDSConfig] = None):
        self.cfg = config or TDSConfig()

    def scan_usage(self, graph) -> List[str]:
        # 层使用序列即模型层索引（0..n_layer-1）
        return []

    def load(self, block_id: str) -> bool:
        # TODO: 提升层到 GPU（调整 n_gpu_layers / partial offload）
        return False

    def unload(self, block_id: str) -> bool:
        # TODO: 层回退到 CPU
        return False

    def prefetch(self, block_id: str) -> bool:
        # TODO: mmap MADV_WILLNEED 预读对应层权重
        return False

    def vram_state(self) -> VramState:
        try:
            import torch
            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info()
                total_mb = int(total_b // (1024 * 1024))
                free_mb = int(free_b // (1024 * 1024))
                return VramState(total_mb=total_mb, used_mb=total_mb - free_mb, free_mb=free_mb)
        except Exception:
            pass
        return VramState(total_mb=0, used_mb=0, free_mb=0)
