"""太极调度层 TDS · KV 宿主供给层（引擎与宿主之间的唯一缝隙）。

IP 边界（硬约束）
----------------
引擎（observer / page_manager）**不得** import 任何宿主模块，也不得出现
宿主类名字面量。宿主对象一律由外部经 ``set_host()`` / ``set_host_provider()``
注入；ComfyUI 的对接实现位于 ``tds/adapters/comfyui_kv_adapter.py``，属可选
适配层，不计入可售引擎 IP。

这样引擎包可在无 ComfyUI 环境下独立安装、独立测试、独立授权。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

__all__ = [
    "KvHost",
    "set_host",
    "set_host_provider",
    "clear_host",
    "get_host",
    "has_host",
    "register_host_hook",
    "unregister_host_hook",
]


@dataclass(frozen=True)
class KvHost:
    """宿主 KV 描述 —— 引擎唯一需要知道的宿主信息。

    model_cls      : 拥有 ``init_kv_cache(batch, capacity, device, dtype)`` 的类
    fixed_kv_cls   : FixedKV 类（用于分支守卫），可为 None
    source         : 来源标识（仅日志用）
    """
    model_cls: Any
    fixed_kv_cls: Any = None
    source: str = ""


_host: Optional[KvHost] = None
_provider: Optional[Callable[[], Optional[KvHost]]] = None

# 宿主就绪回调：引擎（observer / page_manager）在 enable 时登记，
# 宿主后到时会自动补包装，避免「先开闸、后注册宿主」的顺序依赖。
_host_hooks: list = []


def register_host_hook(fn: Callable[[], None]) -> None:
    """登记宿主就绪回调（去重，幂等）。"""
    if fn not in _host_hooks:
        _host_hooks.append(fn)


def unregister_host_hook(fn: Callable[[], None]) -> None:
    """撤销宿主就绪回调（不存在时静默）。"""
    try:
        _host_hooks.remove(fn)
    except ValueError:
        pass


def _notify_host_hooks() -> None:
    """宿主绑定发生变化后通知各引擎补包装；单个回调异常不影响其它回调。"""
    for fn in list(_host_hooks):
        try:
            fn()
        except Exception:
            pass


def set_host(host: Optional[KvHost]) -> None:
    """直接注入宿主（测试 / 单宿主场景）。"""
    global _host
    _host = host
    _notify_host_hooks()


def set_host_provider(fn: Optional[Callable[[], Optional[KvHost]]]) -> None:
    """注册宿主探测函数（延迟解析，宿主可缺席）。"""
    global _provider, _host
    _provider = fn
    _host = None
    _notify_host_hooks()


def clear_host() -> None:
    """清空宿主绑定（回退到无宿主状态）。"""
    global _host, _provider
    _host = None
    _provider = None
    _notify_host_hooks()


def get_host() -> Optional[KvHost]:
    """解析当前宿主：显式注入优先，其次探测函数；均无则返回 None。"""
    global _host
    if _host is not None:
        return _host
    if _provider is not None:
        try:
            _host = _provider()
        except Exception:
            _host = None
    return _host


def has_host() -> bool:
    """当前是否存在可用宿主。"""
    return get_host() is not None
