"""太极调度层 TDS · KV 引擎门面（对外唯一开闸入口）。

Phase 2 A 段：零侵入包装宿主 init_kv_cache 做只读观测。
Phase 2 B 段：在此基础上实现分页换入换出。

开闸三态（``TDS_KV`` 环境变量，或 ``activate(mode)`` 显式指定），互斥且幂等：

    off      完全回退，等价于宿主原生行为
    observe  仅观测（默认），产出 KVObservationReport
    page     分页缓存引擎，把宿主 KV 结构替换为 PageCache

**为什么要互斥**：observer 与 page_manager 都会包装同一个 ``init_kv_cache``，
同时启用会层叠成双重包装、语义错乱。统一经 ``activate()`` 开闸，可保证任一
时刻至多一个引擎生效。

引擎层零宿主依赖：宿主对象一律由 ``host.set_host()`` 注入，本包不含任何
宿主模块 import，可脱离宿主独立安装 / 测试 / 授权。
"""
from __future__ import annotations

import os

from . import observer, page_manager
from .observer import (
    get_report,
    reset_report,
    get_summary_text,
)

__all__ = [
    "enable",
    "disable",
    "is_enabled",
    "get_report",
    "reset_report",
    "get_summary_text",
    "resolve_mode",
    "activate",
    "active_mode",
    "observer",
    "page_manager",
]

_ACTIVE_MODE = "off"  # off / observe / page


def resolve_mode() -> str:
    """按 TDS_KV 环境变量解析开闸模式（三态，默认 observe）。"""
    value = os.environ.get("TDS_KV", "observe").strip().lower()
    if value in ("0", "off", "false", "no", "disable", "disabled"):
        return "off"
    if value in ("page", "paging", "pages"):
        return "page"
    return "observe"


def active_mode() -> str:
    """当前已生效的开闸模式（未开闸为 off）。"""
    return _ACTIVE_MODE


def activate(mode: str = None) -> str:
    """统一开闸：三态互斥、幂等。返回实际生效模式。

    ``mode`` 为空时取 ``TDS_KV``。执行策略是先全关（保证互斥）再按模式开启。
    重复以同一模式调用会直接返回，不重复装卸。
    """
    global _ACTIVE_MODE
    target = (mode or resolve_mode()).strip().lower()
    if target in ("0", "false", "no", "disable", "disabled"):
        target = "off"
    if target not in ("off", "observe", "page"):
        target = "observe"

    if target == _ACTIVE_MODE and target != "off":
        return _ACTIVE_MODE

    # 先卸载双方，确保任一时刻至多一个引擎包装 init_kv_cache
    page_manager.disable()
    observer.disable()
    _ACTIVE_MODE = "off"

    if target == "page":
        page_manager.enable()
        _ACTIVE_MODE = "page"
    elif target == "observe":
        observer.enable()
        _ACTIVE_MODE = "observe"
    return _ACTIVE_MODE


def enable() -> None:
    """兼容入口：等价于 ``activate("observe")``。"""
    activate("observe")


def disable() -> None:
    """关闭全部 KV 引擎，回到宿主原生行为。"""
    activate("off")


def is_enabled() -> bool:
    """任一 KV 引擎是否已开闸。"""
    return observer.is_enabled() or page_manager.is_enabled()
