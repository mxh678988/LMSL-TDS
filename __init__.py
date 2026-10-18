"""太极调度层 TDS · Taiji Dispatch Scheduler（Phase 1）。

定位：大模型与显存之间的分类预判调度中间层。
  - 核心层 tds/core/     平台无关（八域分类 / 八卦加权 / 阴阳分层 / 四象压力）
  - 集成层 tds/integrations/  宿主桥接（ComfyUI：注入 lmsl-smart-offload）
  - 插件层 plugins/      可挂载入口（ComfyUI custom_node，移动端 Adapter 预留）

作为 ComfyUI custom_node 使用时，整个 TDS 目录放入 <ComfyUI>/custom_nodes/ 即可，
本文件把加载转发给 plugins.comfyui_node，无需改动核心层。
"""
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from .plugins.comfyui_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception:  # 非 ComfyUI 环境 / 宿主不匹配时保持可导入，不影响本地开发
    logging.getLogger("tds").exception("[TDS] custom_node 转发失败（已忽略）")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__version__ = "0.1.0-phase1"
