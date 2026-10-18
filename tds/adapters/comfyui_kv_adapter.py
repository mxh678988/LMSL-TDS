"""ComfyUI × TDS 的 KV 宿主对接（**唯一** `import comfy` 收敛点）。

归属声明
--------
本文件属于**适配层**，是可选插件，不计入可售引擎 IP：
引擎（``tds/core``、``tds/kv``）保持零宿主依赖，由本文件把 ComfyUI 的
``Llama2_`` / ``FixedKV`` 交给引擎。

引擎包可脱离 ComfyUI 独立安装 / 测试 / 授权；本文件随 ComfyUI 部署。
"""
from __future__ import annotations

import logging
from typing import Optional

from ..kv.host import KvHost, set_host_provider

LOG = logging.getLogger("tds.adapter.comfyui.kv")

TARGET_MODULE = "comfy.text_encoders.llama"


def probe() -> Optional[KvHost]:
    """探测 ComfyUI 的 KV 宿主；不可用返回 None。"""
    try:
        from comfy.text_encoders import llama as _mod
    except Exception:
        return None
    model_cls = getattr(_mod, "Llama2_", None)
    if model_cls is None:
        return None
    return KvHost(
        model_cls=model_cls,
        fixed_kv_cls=getattr(_mod, "FixedKV", None),
        source=TARGET_MODULE,
    )


def install() -> bool:
    """把 ComfyUI 注册为 KV 宿主供给器；返回当前是否可用。"""
    set_host_provider(probe)
    host = probe()
    if host is None:
        LOG.info("[TDS] ComfyUI KV 宿主不可用，引擎待命")
        return False
    LOG.info("[TDS] ComfyUI KV 宿主已注册: %s", host.source)
    return True


def uninstall() -> None:
    """撤销宿主注册（回退幂等）。"""
    from ..kv.host import clear_host
    clear_host()
