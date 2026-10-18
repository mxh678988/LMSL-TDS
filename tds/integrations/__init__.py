"""集成层：TDS 核心策略与外部运行时/插件的桥接。

core/ 保持平台无关；integrations/ 负责把 TDS 策略注入到既有宿主
（当前为 ComfyUI + lmsl-smart-offload），全程零侵入、可一键回退。
"""
from .comfyui_bridge import (
    ComfyBridge,
    attach_default,
    current_bridge,
    domain_of_key,
    find_smart_offload,
)

__all__ = [
    "ComfyBridge",
    "attach_default",
    "current_bridge",
    "domain_of_key",
    "find_smart_offload",
]
