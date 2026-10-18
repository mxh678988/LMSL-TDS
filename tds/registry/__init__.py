"""tds.registry —— 模型自动获取与显存可行性预判。

子系统职责
----------
* ``hardware``  : 整机画像（显存 / 内存 / 算力），nvidia-smi 主通道，纯标准库；
* ``remote``    : 端点解析（国内镜像优先）与远端元数据查询；
* ``catalog``   : 声明式模型目录（不内嵌权重、不硬编码校验和，支持外部注入）；
* ``estimate``  : 显存估算（权重 / KV / 工作区）与档位判定；
* ``planner``   : 把硬件与配置合成保守预算，输出加载方案与候选排序；
* ``downloader``: 断点续传 + 体积校验 + 原子落盘的下载器。

对外主入口
----------
``probe_hardware`` / ``plan`` / ``rank`` / ``recommend`` / ``download_spec``

依赖约束：本子包只依赖标准库（可选 requests），不引入任何深度学习框架运行时，
不依赖任何宿主运行时，保持与 tds.core / tds.kv 同等的可售洁净度。
"""
from .catalog import (
    DOMAINS,
    ModelSpec,
    clear_remote_cache,
    get,
    list_entries,
    load_catalog,
    probe_entry,
    register,
)
from .downloader import download_file, download_spec
from .errors import (
    CatalogError,
    DownloadError,
    HardwareProbeError,
    IntegrityError,
    RegistryError,
)
from .estimate import (
    Footprint,
    Verdict,
    estimate_footprint,
    estimate_kv_mb,
    judge,
)
from .hardware import DeviceProfile, HardwareProfile, probe_hardware
from .planner import budget_mb, plan, rank, recommend
from .remote import DEFAULT_ENDPOINT, resolve_endpoint

__all__ = [
    # 硬件
    "probe_hardware",
    "HardwareProfile",
    "DeviceProfile",
    # 目录
    "ModelSpec",
    "DOMAINS",
    "load_catalog",
    "get",
    "list_entries",
    "register",
    "probe_entry",
    "clear_remote_cache",
    # 估算
    "estimate_footprint",
    "estimate_kv_mb",
    "judge",
    "Footprint",
    "Verdict",
    # 决策
    "budget_mb",
    "plan",
    "rank",
    "recommend",
    # 下载
    "download_file",
    "download_spec",
    # 端点
    "resolve_endpoint",
    "DEFAULT_ENDPOINT",
    # 异常
    "RegistryError",
    "CatalogError",
    "HardwareProbeError",
    "DownloadError",
    "IntegrityError",
]
