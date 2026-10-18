"""ComfyUI 适配器（PC）。

接入方式：作为 custom_node 注入 ComfyUI 进程内（进程外无法干预加载）。
关键接入点（只读观测 + 受控包装，不修改 ComfyUI 源码）：
  - comfy.model_management.free_memory / load_models_gpu / current_loaded_models
  - server.PromptServer.add_on_prompt_handler（执行图预扫描）

安全约束：默认保守（keep_steps=1、预取关闭、压力保护开启），失败即回退。
"""
import json
import logging
from typing import List, Optional

from ..config import TDSConfig
from ..core.memory import VramState
from .base import BaseAdapter

logger = logging.getLogger(__name__)


class ComfyUIAdapter(BaseAdapter):
    """ComfyUI 运行时适配器。"""

    def __init__(self, config: Optional[TDSConfig] = None):
        self.cfg = config or TDSConfig()
        self._mm = None  # 惰性导入 comfy.model_management
        # block_id -> ModelPatcher 映射（真机闭环关键：load/unload 需要真实对象）。
        # 宿主（bridge）在 ComfyUI 加载模型时调用 register_model 登记。
        self._model_map: dict = {}
        # 模型文件名 -> block_id（宿主按名称注册时反查）
        self._name_to_block: dict = {}

    # ------------------------------------------------------------------ 注册
    def register_model(self, block_id: str, model) -> None:
        """登记 block_id -> ModelPatcher 映射（由宿主/bridge 在加载模型时调用）。"""
        if model is None or not block_id:
            return
        self._model_map[block_id] = model
        name = getattr(model, "model_name", "") or ""
        if name:
            self._name_to_block[name] = block_id

    def register_model_by_name(self, model_name: str, model) -> None:
        """按模型文件名登记（scan_usage 输出 model_name 时反向使用）。"""
        if model is None or not model_name:
            return
        block_id = self._name_to_block.get(model_name)
        if block_id:
            self._model_map[block_id] = model

    def model_for(self, block_id: str):
        """返回 block_id 对应的 ModelPatcher（未登记时返回 None）。"""
        return self._model_map.get(block_id)

    # ------------------------------------------------------------------ 内部
    def _ensure_mm(self):
        if self._mm is None:
            try:
                import comfy.model_management as mm
                self._mm = mm
            except Exception as e:
                logger.warning(f"ComfyUI model_management unavailable: {e}")
                self._mm = None
        return self._mm

    # ------------------------------------------------------------------ VRAM
    def vram_state(self) -> VramState:
        mm = self._ensure_mm()
        if mm is not None:
            try:
                free_b = mm.get_free_memory()
                total_b = mm.get_total_memory()
                total_mb = int(total_b // (1024 * 1024))
                free_mb = int(free_b // (1024 * 1024))
                return VramState(
                    total_mb=total_mb,
                    used_mb=total_mb - free_mb,
                    free_mb=free_mb,
                )
            except Exception:
                pass
        # 回退
        try:
            import torch
            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info()
                total_mb = int(total_b // (1024 * 1024))
                free_mb = int(free_b // (1024 * 1024))
                return VramState(
                    total_mb=total_mb, used_mb=total_mb - free_mb, free_mb=free_mb
                )
        except Exception:
            pass
        return VramState(total_mb=0, used_mb=0, free_mb=0)

    # ------------------------------------------------------------------ scan
    def scan_usage(self, graph) -> List[str]:
        """解析 prompt 执行图，构建"模型使用序列 + 下次使用距离"。

        返回格式：
        [
            {"node_id": int, "block_type": "ckpt", "block_id": str},
            ...
        ]
        按执行顺序排序，block_id 即模型文件路径/节点标识。
        """
        mm = self._ensure_mm()
        if mm is None or graph is None:
            return []

        if isinstance(graph, str):
            try:
                graph = json.loads(graph)
            except (json.JSONDecodeError, TypeError):
                return []
        elif isinstance(graph, dict):
            pass
        else:
            return []

        result = []
        # 兼容两种图格式（问题5 模型名映射）：
        #   TDS graph：{"graph": {nid: {...}}, "edges": [...]}
        #   标准 ComfyUI prompt：顶层即节点 dict {nid: {class_type, inputs}}
        nodes = graph.get("graph", {})
        edges = graph.get("edges", [])
        if not nodes and isinstance(graph, dict):
            nodes = graph

        # 构建执行图：从输入节点开始拓扑排序
        node_types = {
            "CheckpointLoaderSimple": ("ckpt", "ckpt"),
            "CheckpointLoader": ("ckpt", "ckpt"),
            "CLIPLoader": ("clip", "text_encoder"),
            "CLIPLoaderGGUF": ("clip", "text_encoder"),
            "CLIPSetLastLayer": ("clip", "text_encoder"),
            "VAELoader": ("vae", "vae"),
            "VAEDecode": ("vae", "vae"),
            "VAEEncode": ("vae", "vae"),
            "UNETLoader": ("unet", "unet"),
            "UNETLoaderAdvanced": ("unet", "unet"),
            "UnetLoaderGGUF": ("unet", "unet"),
            "DiffControlNetLoader": ("controlnet", "controlnet"),
            "ControlNetApply": ("controlnet", "controlnet"),
        }

        # 各 loader 节点输入字段中携带真实模型文件名的 key
        NAME_KEYS = {
            "CheckpointLoaderSimple": ["ckpt_name"],
            "CheckpointLoader": ["ckpt_name"],
            "CLIPLoader": ["clip_name"],
            "CLIPLoaderGGUF": ["clip_name"],
            "CLIPSetLastLayer": ["clip_name"],
            "VAELoader": ["vae_name"],
            "VAEDecode": ["vae_name"],
            "VAEEncode": ["vae_name"],
            "UNETLoader": ["unet_name"],
            "UNETLoaderAdvanced": ["unet_name"],
            "UnetLoaderGGUF": ["unet_name"],
            "DiffControlNetLoader": ["control_net_name"],
            "ControlNetApply": ["control_net_name"],
        }

        for nid, node in nodes.items():
            cls = node.get("class_type", "")
            for pattern, (block_type, block_id_prefix) in node_types.items():
                if pattern in cls:
                    model_name = ""
                    inputs = node.get("inputs", {})
                    for k in NAME_KEYS.get(pattern, []):
                        v = inputs.get(k)
                        if isinstance(v, dict):
                            v = v.get("content", "") or ""
                        if v:
                            model_name = str(v)
                            break
                    # 问题5：block_id 优先用真实模型名（跨图稳定、可反查、可复用缓存），
                    # 无模型名时回退 "nid:cls" 节点标识
                    block_id = f"{block_type}:{model_name}" if model_name else f"{nid}:{cls}"
                    entry = {
                        "node_id": nid,
                        "block_type": block_type,
                        "block_id": block_id,
                    }
                    if model_name:
                        entry["model_name"] = model_name
                        self._name_to_block[model_name] = block_id
                    result.append(entry)
                    break

        if not result:
            # 退化为按节点 ID 排序
            for nid in sorted(nodes.keys(), key=lambda x: int(x)):
                node = nodes[nid]
                cls = node.get("class_type", "unknown")
                result.append({"node_id": nid, "block_type": "unknown", "block_id": f"{nid}:{cls}"})

        return result

    # ------------------------------------------------------------------ load
    def load(self, block_id: str) -> bool:
        """通过 load_model_gpu 唤醒目标模型（受控、带压力保护）。

        真机路径：优先走 block_id -> ModelPatcher 映射（load_model_gpu 需要真实
        模型对象，仅传字符串会失败）；未登记时回退字符串加载（仅 sim/骨架可用）。
        """
        mm = self._ensure_mm()
        if mm is None:
            return False

        patcher = self._model_map.get(block_id)
        if patcher is not None:
            try:
                mm.load_model_gpu(patcher)
                return True
            except Exception as e:
                logger.warning(f"ComfyUI load(map) {block_id}: {e}")
                return False

        try:
            # 回退：以 block_id 作为模型标识加载（真机上大概率失败，属预期）
            model = mm.load_model_gpu(block_id)
            return model is not None
        except Exception as e:
            logger.warning(f"ComfyUI load(str) {block_id}: {e}")
            return False

    # ------------------------------------------------------------------ unload
    def unload(self, block_id: str) -> bool:
        """通过 unload_model_and_clones 释放指定模型（保留 keep_loaded 白名单）。

        真机路径：优先走 block_id -> ModelPatcher 映射直接卸载对应对象；
        回退按模型名/路径匹配 current_loaded_models（LoadedModel 列表）。
        """
        mm = self._ensure_mm()
        if mm is None:
            return False

        patcher = self._model_map.get(block_id)
        if patcher is not None:
            try:
                mm.unload_model_and_clones(patcher)
                return True
            except Exception as e:
                logger.warning(f"ComfyUI unload(map) {block_id}: {e}")

        try:
            loaded = mm.current_loaded_models
            if isinstance(loaded, dict):
                for m in loaded.keys():
                    if block_id in str(m):
                        try:
                            mm.unload_model_and_clones(m)
                            return True
                        except Exception:
                            continue
                return False
            if isinstance(loaded, list):
                for m in loaded:
                    name = str(getattr(m, "name", "") or getattr(m.model, "model_name", "") or m)
                    if block_id in name or block_id in str(m):
                        try:
                            mm.unload_model_and_clones(m)
                            return True
                        except Exception:
                            continue
                return False
            return False
        except Exception as e:
            logger.warning(f"ComfyUI unload {block_id}: {e}")
            return False

    # ------------------------------------------------------------------ prefetch
    def prefetch(self, block_id: str) -> bool:
        """空闲窗口提前唤醒下一模型。"""
        mm = self._ensure_mm()
        if mm is None:
            return False

        try:
            return self.load(block_id)
        except Exception as e:
            logger.warning(f"ComfyUI prefetch {block_id}: {e}")
            return False
