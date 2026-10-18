"""太极调度层 TDS —— 大模型权重-显存动态调度中间层。

SDK 入口
--------
from tds import TDSRuntime

runtime = TDSRuntime()
runtime.register_adapter(MyAdapter())
runtime.attach()  # 挂载
result = runtime.run_task("用户输入")
runtime.detach()  # 卸载
"""
__version__ = "0.1.0"
__author__ = "BY LAOMENG 网络工作室"

from .config import TDSConfig, BAGUA_DOMAINS
from .api import (
    TDSRuntime,
    RuntimeState,
    AttachResult,
    DetachResult,
    BenchmarkResult,
)

__all__ = [
    "TDSConfig",
    "BAGUA_DOMAINS",
    "__version__",
    "TDSRuntime",
    "RuntimeState",
    "AttachResult",
    "DetachResult",
    "BenchmarkResult",
]
