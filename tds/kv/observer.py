"""太极调度层 TDS · KV 可观测层（Phase 2 A 段）。

目标
----
零侵入包装宿主模型类的 `init_kv_cache()`，仅做**只读观测**：记录每层 KV
cache 的 capacity / dtype / device / 实测占用，不改变任何分配行为。
导出观测报告供 Phase 2 B 段（分页换入换出）与 Phase 0 baseline 交叉校验。

设计约束
--------
1. 原始调用与观测逻辑严格解耦：观测失败**绝不**影响宿主返回值。
2. 不硬依赖 torch：dtype/device 一律按字符串解析，字节数优先实测张量。
3. 开关式回退：TDS_KV=off 时等价于宿主原生行为。

使用方式
--------
1. `TDS_KV=off`       完全回退
2. `TDS_KV=observe`   仅观测（默认）

        from tds.kv import observer
        observer.enable()
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("tds.kv")

# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------

_KV_ENABLED = True  # 全局开关（TDS_KV=off 时置 False）

# dtype 名 -> 字节数（不依赖 torch）
_DTYPE_BYTES = {
    "float32": 4, "float": 4, "float64": 8, "double": 8,
    "float16": 2, "half": 2, "bfloat16": 2,
    "int64": 8, "long": 8, "int32": 4, "int": 4,
    "int16": 2, "short": 2, "int8": 1, "uint8": 1, "bool": 1,
    "float8_e4m3fn": 1, "float8_e5m2": 1,
}


def _dtype_bytes(dtype: Any) -> int:
    """按 dtype 名称解析字节数，无法识别返回 0。"""
    name = str(dtype).split(".")[-1].strip().lower()
    return _DTYPE_BYTES.get(name, 0)


def _tensor_bytes(obj: Any) -> int:
    """对单个张量求实测字节数，非张量返回 0。"""
    try:
        numel = getattr(obj, "numel", None)
        if numel is None:
            return 0
        return int(numel()) * int(getattr(obj, "element_size")())
    except Exception:
        return 0


def _layer_bytes(layer: Any, batch: int, capacity: int,
                 head_dim: int, num_kv_heads: int, dtype: Any) -> int:
    """求单层 KV 实际占用字节：优先实测张量，退化到公式估算。"""
    # 形态 1：FixedKV（有 key/value 属性）
    key = getattr(layer, "key", None)
    val = getattr(layer, "value", None)
    if key is not None:
        measured = _tensor_bytes(key) + _tensor_bytes(val)
        if measured > 0:
            return measured
    # 形态 2：元组 (key, value, index)
    if isinstance(layer, (tuple, list)):
        measured = sum(_tensor_bytes(t) for t in layer)
        if measured > 0:
            return measured
    # 退化：公式估算
    per_token = 2 * head_dim * num_kv_heads * _dtype_bytes(dtype)
    return per_token * capacity


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------

@dataclass
class KVLayerRecord:
    """单层 KV cache 的观测记录。"""
    layer_id: int
    capacity_tokens: int
    dtype: str
    device: str
    batch: int
    num_kv_heads: int
    head_dim: int
    bytes_allocated: int
    last_used_step: int = 0
    use_count: int = 0
    is_fixed_kv: bool = False  # fixed_kv 模式（Comfy Kitchen flash_attention_decode）
    page_id: Optional[str] = None  # Phase 2 B 段：页表页 ID


@dataclass
class KVObservationReport:
    """KV 观测汇总报告。"""
    layers: List[KVLayerRecord] = field(default_factory=list)
    total_bytes_allocated: int = 0
    total_capacity_tokens: int = 0
    layers_by_device: Dict[str, int] = field(default_factory=dict)
    layers_by_dtype: Dict[str, int] = field(default_factory=dict)
    layers_by_fixed_kv: Dict[bool, int] = field(default_factory=dict)

    @property
    def summary_lines(self) -> List[str]:
        lines = [f"KV 观测报告（{len(self.layers)} 层）"]
        lines.append(f"总容量: {self.total_capacity_tokens:,} tokens")
        lines.append(
            f"总分配: {self.total_bytes_allocated:,} bytes "
            f"({self.total_bytes_allocated / 1024 ** 2:.2f} MB)"
        )
        lines.append(f"设备分布: {dict(self.layers_by_device)}")
        lines.append(f"类型分布: {dict(self.layers_by_dtype)}")
        lines.append(f"fixed_kv: {dict(self.layers_by_fixed_kv)}")
        return lines


# --------------------------------------------------------------------------
# 观测状态
# --------------------------------------------------------------------------

_KV_STATE: Dict[str, Any] = {
    "patched": False,
    "original_init_kv_cache": None,
    "original_class_dict": False,
    "report": None,
}

_current_model_id: str = ""


def _make_report() -> KVObservationReport:
    return KVObservationReport()


# --------------------------------------------------------------------------
# 开关
# --------------------------------------------------------------------------

def _on_host_ready() -> None:
    """宿主就绪 / 变更回调：补包装（幂等，仅在本引擎已开闸时生效）。"""
    if _KV_ENABLED:
        _ensure_wrapped()


def enable() -> None:
    """启用 KV 可观测（幂等）。

    互斥声明：本函数与 ``tds.kv.page_manager.enable()`` 互斥——二者都会包装
    同一个 ``init_kv_cache``，同时启用会产生双重包装。生产路径请统一经
    ``tds.kv.activate()`` 开闸，由该入口保证三态互斥。
    """
    global _KV_ENABLED
    _KV_ENABLED = True
    try:
        from .host import register_host_hook
        register_host_hook(_on_host_ready)
    except Exception:
        pass
    _ensure_wrapped()


def disable() -> None:
    """禁用 KV 可观测（恢复原生行为，幂等）。"""
    global _KV_ENABLED
    _KV_ENABLED = False
    _ensure_unwrapped()


def is_enabled() -> bool:
    return _KV_ENABLED


# --------------------------------------------------------------------------
# Patch 逻辑
# --------------------------------------------------------------------------

def _resolve_model_cls() -> Any:
    """解析宿主中拥有 init_kv_cache 的类；无宿主返回 None。"""
    try:
        from .host import get_host
        h = get_host()
    except Exception:
        h = None
    return getattr(h, "model_cls", None) if h is not None else None


def _ensure_wrapped() -> None:
    """将 init_kv_cache 包装到宿主模型类上（幂等）。"""
    if _KV_STATE["patched"]:
        return
    try:
        model_cls = _resolve_model_cls()
        if model_cls is None:
            LOG.info("[TDS-KV] 未绑定宿主，包装待命")
            return
        if "init_kv_cache" in model_cls.__dict__:
            original = model_cls.__dict__["init_kv_cache"]
            own_attr = True
        else:
            original = getattr(model_cls, "init_kv_cache")
            own_attr = False
        _KV_STATE["model_cls"] = model_cls
        _KV_STATE["original_init_kv_cache"] = original
        _KV_STATE["original_class_dict"] = own_attr
        model_cls.init_kv_cache = _make_kv_observer_wrapper(original)
        _KV_STATE["patched"] = True
        LOG.info("[TDS-KV] init_kv_cache 已包装 (own_attr=%s)", own_attr)
    except Exception:
        LOG.exception("[TDS-KV] 包装 init_kv_cache 失败（已忽略，不影响宿主行为）")


def _ensure_unwrapped() -> None:
    """恢复宿主模型类的原始 init_kv_cache（幂等）。"""
    if not _KV_STATE["patched"]:
        return
    try:
        model_cls = _KV_STATE.get("model_cls") or _resolve_model_cls()
        if model_cls is None:
            _KV_STATE["patched"] = False
            return
        orig = _KV_STATE.get("original_init_kv_cache")
        if orig is not None:
            if _KV_STATE.get("original_class_dict"):
                model_cls.init_kv_cache = orig
            else:
                # 原函数继承自父类，删除覆盖层即可还原
                try:
                    delattr(model_cls, "init_kv_cache")
                except AttributeError:
                    model_cls.init_kv_cache = orig
        _KV_STATE["patched"] = False
        _KV_STATE["model_cls"] = None
        _KV_STATE["original_init_kv_cache"] = None
        LOG.info("[TDS-KV] init_kv_cache 已恢复原生")
    except Exception:
        LOG.exception("[TDS-KV] 恢复 init_kv_cache 异常（已忽略）")


def _record(self, batch, capacity, device, dtype, result) -> None:
    """执行只读观测；任何异常均不向上抛出。"""
    global _current_model_id
    _current_model_id = f"{self.__class__.__name__}_{id(self)}"

    if _KV_STATE["report"] is None:
        _KV_STATE["report"] = _make_report()
    report = _KV_STATE["report"]

    config = getattr(self, "config", None)
    head_dim = int(getattr(config, "head_dim", 0) or 0)
    num_kv_heads = int(getattr(config, "num_key_value_heads", 0) or 0)
    fixed_kv_flags = getattr(self, "fixed_kv", None)
    dtype_str = str(dtype)
    device_str = str(device)

    layer_list = result if isinstance(result, (list, tuple)) else [result]
    n_layers = len(layer_list)
    total_bytes = 0

    for layer_id, layer in enumerate(layer_list):
        bytes_alloc = _layer_bytes(layer, batch, capacity, head_dim, num_kv_heads, dtype)
        is_fixed = bool(getattr(layer, "key", None) is not None) or bool(fixed_kv_flags)
        report.layers.append(KVLayerRecord(
            layer_id=layer_id,
            capacity_tokens=int(capacity),
            dtype=dtype_str,
            device=device_str,
            batch=int(batch),
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bytes_allocated=bytes_alloc,
            is_fixed_kv=is_fixed,
        ))
        total_bytes += bytes_alloc
        report.layers_by_device[device_str] = report.layers_by_device.get(device_str, 0) + 1
        report.layers_by_dtype[dtype_str] = report.layers_by_dtype.get(dtype_str, 0) + 1
        report.layers_by_fixed_kv[is_fixed] = report.layers_by_fixed_kv.get(is_fixed, 0) + 1

    report.total_bytes_allocated += total_bytes
    report.total_capacity_tokens += int(capacity) * n_layers

    LOG.info(
        "[TDS-KV] 记录 %d 层: batch=%s capacity=%s dtype=%s device=%s fixed_kv=%s bytes=%s",
        n_layers, batch, capacity, dtype_str, device_str, fixed_kv_flags, total_bytes,
    )


def _make_kv_observer_wrapper(original_fn):
    """为 init_kv_cache 创建只读观测包装器。

    关键：先执行并返回原始调用结果，观测逻辑完全隔离在其后。
    """

    def wrapper(self, batch, capacity, device, dtype):
        result = original_fn(self, batch, capacity, device, dtype)  # 宿主行为原样执行
        try:
            if _KV_ENABLED:
                _record(self, batch, capacity, device, dtype, result)
        except Exception:
            LOG.exception("[TDS-KV] 观测内部异常（已忽略，不影响宿主返回值）")
        return result

    return wrapper


# --------------------------------------------------------------------------
# 报告接口
# --------------------------------------------------------------------------

def get_report() -> Optional[KVObservationReport]:
    """获取当前观测报告（不触发包装）。"""
    return _KV_STATE.get("report")


def reset_report() -> None:
    """重置观测报告。"""
    _KV_STATE["report"] = _make_report()
    LOG.info("[TDS-KV] 观测报告已重置")


def get_summary_text() -> str:
    """获取观测摘要文本。"""
    report = get_report()
    if report is None:
        return "无观测数据"
    return "\n".join(report.summary_lines)


# --------------------------------------------------------------------------
# 环境变量初始化
# --------------------------------------------------------------------------

def _env_enabled() -> bool:
    """按 TDS_KV 环境变量解析观测开关（默认 observe=开启）。"""
    value = os.environ.get("TDS_KV", "observe").strip().lower()
    return value not in ("0", "off", "false", "no", "disable", "disabled")


_KV_ENABLED = _env_enabled()

LOG.info("[TDS-KV] 初始化: TDS_KV=%s, enabled=%s",
         os.environ.get("TDS_KV", "observe"), _KV_ENABLED)
