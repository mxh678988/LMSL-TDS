"""显存占用估算与可行性判定。

估算口径
--------
* **权重**：优先由文件体积直接推导（``bytes / 1024^2``）——这是最可靠的口径，
  不依赖参数规模反推；仅当体积未知时才用 ``params_b × quant_bits`` 兜底；
* **KV Cache**：``2 × layers × kv_heads × head_dim × ctx × dtype_bytes``；
* **工作区**：按经验系数折算（注意力中间张量、编解码缓冲等），默认 15%。

注意：工作区与 KV 的经验系数属于**待实测校准**的保守估计，Phase0 出图实测后
应以实测峰值回填；本模块不声称其为精确值。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

MB = 1024.0 * 1024.0

#: 工作区经验系数（占权重比）—— 待 Phase0 实测校准
DEFAULT_ACTIVATION_RATIO = 0.15
#: KV 元素默认字节数（bf16）
DEFAULT_KV_DTYPE_BYTES = 2
#: 判定档位阈值：占用 ≤ 预算×OK_FACTOR 记为 ok，≤ 预算 记为 tight
OK_FACTOR = 0.85


@dataclass
class Footprint:
    """一次加载的显存画像（单位 MB）。"""

    weights_mb: float = 0.0
    kv_mb: float = 0.0
    workspace_mb: float = 0.0
    total_mb: float = 0.0
    basis: str = ""          # 权重估算依据：file_size / params / unknown
    complete: bool = False   # 依据是否完整（体积全解析）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Verdict:
    """可行性判定结果。"""

    level: str               # ok / tight / oom / unknown
    fits: bool
    budget_mb: float
    required_mb: float
    headroom_mb: float
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def weights_mb_from_size(size_bytes: int) -> float:
    return float(size_bytes) / MB


def weights_mb_from_params(params_b: float, quant_bits: int) -> float:
    if params_b <= 0 or quant_bits <= 0:
        return 0.0
    return params_b * 1e9 * (quant_bits / 8.0) / MB


def estimate_kv_mb(
    layers: int,
    kv_heads: int,
    head_dim: int,
    ctx_tokens: int,
    dtype_bytes: int = DEFAULT_KV_DTYPE_BYTES,
) -> float:
    """KV Cache 显存；任一维度缺失即返回 0（不臆测）。"""
    if min(layers, kv_heads, head_dim, ctx_tokens) <= 0:
        return 0.0
    return 2.0 * layers * kv_heads * head_dim * ctx_tokens * dtype_bytes / MB


def estimate_footprint(
    spec: Any,
    ctx_tokens: Optional[int] = None,
    activation_ratio: float = DEFAULT_ACTIVATION_RATIO,
) -> Footprint:
    """给出条目的显存画像。

    ``spec`` 需具备 files / sizes / params_b / quant_bits / layers / kv_heads /
    head_dim 字段（即 ModelSpec 契约），此处按鸭子类型使用，避免循环导入。
    """
    size_bytes = int(getattr(spec, "total_size_bytes", 0) or 0)
    if size_bytes > 0:
        weights = weights_mb_from_size(size_bytes)
        basis = "file_size" if getattr(spec, "size_complete", False) else "file_size_partial"
    else:
        weights = weights_mb_from_params(
            float(getattr(spec, "params_b", 0.0) or 0.0),
            int(getattr(spec, "quant_bits", 0) or 0),
        )
        basis = "params" if weights > 0 else "unknown"

    ctx = int(ctx_tokens if ctx_tokens is not None else getattr(spec, "ctx_tokens", 0) or 0)
    kv = 0.0
    if getattr(spec, "needs_kv", False):
        kv = estimate_kv_mb(
            int(getattr(spec, "layers", 0) or 0),
            int(getattr(spec, "kv_heads", 0) or 0),
            int(getattr(spec, "head_dim", 0) or 0),
            ctx,
        )

    workspace = weights * max(0.0, float(activation_ratio))
    total = weights + kv + workspace
    return Footprint(
        weights_mb=round(weights, 2),
        kv_mb=round(kv, 2),
        workspace_mb=round(workspace, 2),
        total_mb=round(total, 2),
        basis=basis,
        complete=(basis == "file_size"),
    )


def judge(footprint: Footprint, budget_mb: float) -> Verdict:
    """按可用预算判定档位。预算 <= 0 或依据缺失时给 unknown，不硬判。"""
    if budget_mb <= 0:
        return Verdict("unknown", False, 0.0, footprint.total_mb, 0.0, "无可用显存预算")
    if footprint.basis == "unknown":
        return Verdict(
            "unknown",
            False,
            budget_mb,
            footprint.total_mb,
            0.0,
            "缺少体积与参数量信息，无法估算",
        )

    headroom = budget_mb - footprint.total_mb
    if footprint.total_mb <= budget_mb * OK_FACTOR:
        level, fits = "ok", True
        reason = "预算充足，余量 {:.0f} MB".format(headroom)
    elif footprint.total_mb <= budget_mb:
        level, fits = "tight", True
        reason = "可装但余量仅 {:.0f} MB，无压力保护余度".format(headroom)
    else:
        level, fits = "oom", False
        reason = "超出预算 {:.0f} MB".format(-headroom)
    if not footprint.complete:
        reason += "（体积未完全解析，结论偏乐观）"
    return Verdict(
        level, fits, round(budget_mb, 2), footprint.total_mb, round(headroom, 2), reason
    )
