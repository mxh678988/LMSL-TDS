"""竞品对比模块：TDS vs 同类方案的诚实对照。

设计原则（避免虚假宣传）
------------------------
1. **实测 / 引用分离**：任何"某竞品性能是 X"的数字，若我们没在本机实测，
   一律标注 ``measured=False`` 并附来源；绝不编造。
2. **可对比维度优先用布尔能力矩阵**：跨模型调度、语义预判、分层卸载、
   KV 分页、宿主无关、零依赖内核、多卡、OOM 防护。
3. TDS 一侧的数字来自本框架的真实测量（baseline / attached），可复现。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .baseline import BaselineResult
from .attached import AttachedResult


# --------------------------------------------------------------------------
# 竞品能力档案（定性 + 来源标注）
# --------------------------------------------------------------------------

@dataclass
class CompetitorSpec:
    """一个竞品的档案。"""

    name: str
    kind: str = ""                       # 代理 / 推理服务 / 节点扩展 / 内置
    capabilities: Dict[str, bool] = field(default_factory=dict)
    measured: bool = False               # 是否本机实测过
    source: str = ""                     # 定性信息的来源 URL
    note: str = ""
    # 公开引用数据（非本机实测，一律标注来源，绝不混入实测区）
    reference: List[Dict[str, Any]] = field(default_factory=list)

    def to_row(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "capabilities": dict(self.capabilities),
            "measured": self.measured,
            "source": self.source,
            "note": self.note,
            "reference": list(self.reference),
        }


# 能力矩阵统一维度（布尔）
CAPABILITY_DIMS = [
    "跨模型调度",
    "语义预判",
    "分层卸载",
    "KV分页",
    "宿主无关",
    "零依赖内核",
    "多卡",
    "OOM防护",
]


def default_competitors() -> List[CompetitorSpec]:
    """内置竞品档案（定性来自公开资料，未标 measured 者均未本机实测）。"""
    return [
        CompetitorSpec(
            name="TDS",
            kind="宿主无关调度中间层",
            capabilities={
                "跨模型调度": True, "语义预判": True, "分层卸载": True, "KV分页": True,
                "宿主无关": True, "零依赖内核": True, "多卡": False, "OOM防护": True,
            },
            measured=True,
            source="本机实测（tds/benchmarks）",
            note="多卡为架构预留，当前未实现",
        ),
        CompetitorSpec(
            name="llama-swap",
            kind="代理（llama.cpp）",
            capabilities={
                "跨模型调度": True, "语义预判": False, "分层卸载": False, "KV分页": False,
                "宿主无关": False, "零依赖内核": False, "多卡": False, "OOM防护": False,
            },
            measured=False,
            source="https://github.com/mostlygeek/llama-swap",
            note="按需整模型切换，无显存分页",
        ),
        CompetitorSpec(
            name="vLLM",
            kind="推理服务",
            capabilities={
                "跨模型调度": True, "语义预判": False, "分层卸载": False, "KV分页": True,
                "宿主无关": False, "零依赖内核": False, "多卡": True, "OOM防护": True,
            },
            measured=False,
            source="https://docs.vllm.ai/ (PagedAttention)",
            note="PagedAttention 显存分页，吞吐强，但强绑定自身服务",
            reference=[
                {
                    "title": "Scaling AI Infrastructure (PagedAttention 论文)",
                    "source": "https://eajournals.org/bjms/wp-content/uploads/sites/21/2025/05/Scaling-AI-Infrastructure.pdf",
                    "note": "引用、非本机实测",
                    "metrics": [
                        "PagedAttention 块级分页：显存利用率 70-89%，相对连续分配 3.9x 提升",
                        "LLaMA-7B 单 A100（batch32/seq512）：126 vs 54 tok/s，吞吐 2.33x",
                        "LLaMA-13B：吞吐 2.47x",
                    ],
                },
                {
                    "title": "Local AI Model Serving Frameworks 2026（转引 Prem AI benchmarking 2026-03）",
                    "source": "https://baeseokjae.github.io/posts/local-ai-model-serving-frameworks-2026",
                    "note": "引用、非本机实测",
                    "metrics": [
                        "H100 SXM5 / Llama-3.1-70B FP16：吞吐 12,553 tok/s（GPU 89%）",
                        "H100 SXM5 / Llama-3.2-8B FP16：吞吐 47,200 tok/s（GPU 86%）",
                        "8B 中等并发 TTFT <100ms",
                    ],
                },
            ],
        ),
        CompetitorSpec(
            name="VibeQwen",
            kind="实验推理引擎（Baseten）",
            capabilities={
                "跨模型调度": False, "语义预判": False, "分层卸载": False, "KV分页": False,
                "宿主无关": False, "零依赖内核": False, "多卡": False, "OOM防护": False,
            },
            measured=False,
            source="https://www.claw-world.app/blog/ai-agent-built-inference-engine-faster-than-vllm",
            note="Claude Code 生成的实验性引擎，非本机实测；无公开能力矩阵证据，八维均按无证据保守填 False",
            reference=[
                {
                    "title": "AI Agent 内置推理引擎（比 vLLM 更快）",
                    "source": "https://www.claw-world.app/blog/ai-agent-built-inference-engine-faster-than-vllm",
                    "note": "引用、非本机实测；实验性引擎、Claude Code 生成",
                    "metrics": [
                        "单 B200 / Qwen-3.6-35B-A3B（NVFP4）：decode 1,792 vs 调优 vLLM 943 tok/s（+90%）",
                        "TTFT 12ms vs 28ms（2.3x）",
                        "并发 32：10,307 vs 6,030 TPS（+71%）",
                    ],
                },
            ],
        ),
        CompetitorSpec(
            name="ComfyUI-MultiGPU",
            kind="节点扩展",
            capabilities={
                "跨模型调度": False, "语义预判": False, "分层卸载": True, "KV分页": False,
                "宿主无关": False, "零依赖内核": False, "多卡": True, "OOM防护": False,
            },
            measured=False,
            source="https://github.com/pollockjj/ComfyUI-MultiGPU",
            note="层级卸载到 RAM/VRAM，绑定 ComfyUI",
        ),
        CompetitorSpec(
            name="ComfyUI 原生",
            kind="内置调度",
            capabilities={
                "跨模型调度": False, "语义预判": False, "分层卸载": False, "KV分页": False,
                "宿主无关": False, "零依赖内核": False, "多卡": False, "OOM防护": True,
            },
            measured=False,
            source="comfy/model_management.py",
            note="VRAMState + ModelPatcher，硬规则调度，无语义预判",
        ),
    ]


# --------------------------------------------------------------------------
# 对比结果模型
# --------------------------------------------------------------------------

@dataclass
class CompetitiveResult:
    """竞品对比结果。"""

    timestamp: str
    task_description: str = ""
    model_name: str = ""

    # TDS 实测（来自 baseline / attached）
    tds_baseline: Optional[Dict[str, Any]] = None
    tds_attached: Optional[Dict[str, Any]] = None

    # 能力矩阵
    matrix: List[Dict[str, Any]] = field(default_factory=list)

    # 相对基线的提升（TDS 自证，全部实测）
    improvement: Dict[str, float] = field(default_factory=dict)

    def print_report(self) -> None:
        print("=" * 78)
        print("TDS 竞品对比报告")
        print("=" * 78)
        print(f"时间: {self.timestamp}")
        if self.task_description:
            print(f"任务: {self.task_description}")
        print()

        # 能力矩阵
        print("能力矩阵（True/False 为定性，非本机实测项见 note）")
        header = f"{'方案':<20}" + "".join(f"{d:<9}" for d in CAPABILITY_DIMS)
        print(header)
        print("-" * len(header))
        for row in self.matrix:
            caps = row.get("capabilities", {})
            cells = "".join(("Y" if caps.get(d) else "-").ljust(9) for d in CAPABILITY_DIMS)
            mark = "" if row.get("measured") else "  [未实测]"
            print(f"{row['name']:<20}{cells}{mark}")
        print()

        # TDS 实测提升
        if self.improvement:
            print("TDS 实测提升（相对无 SDK 基线，全部可复现）")
            for k, v in self.improvement.items():
                print(f"  {k:<24} {v:+.1f}%")
            print()

        # 竞品公开引用数据（非本机实测）
        print("竞品公开引用数据（引用、非本机实测，仅供参照）")
        for row in self.matrix:
            refs = row.get("reference") or []
            if not refs:
                continue
            print(f"  [{row['name']}]")
            for ref in refs:
                print(f"    - {ref.get('title', '')} (source: {ref.get('source', '')})")
                for m in ref.get("metrics", []):
                    print(f"      * {m}")
        print()

        print("说明: 竞品的量化性能未在本机实测，故仅列出公开引用数字，不混入 TDS 实测区。")
        print("=" * 78)


def compare_competitors(
    baseline: Optional[BaselineResult] = None,
    attached: Optional[AttachedResult] = None,
    competitors: Optional[List[CompetitorSpec]] = None,
    task_description: str = "",
    model_name: str = "",
) -> CompetitiveResult:
    """构建竞品对比结果。

    参数:
        baseline: 无 SDK 基线（可选）
        attached: SDK 挂载后测量（可选）
        competitors: 竞品档案列表（默认内置）
        task_description: 任务描述
        model_name: 模型名

    返回:
        CompetitiveResult
    """
    if competitors is None:
        competitors = default_competitors()

    matrix = [c.to_row() for c in competitors]

    tds_baseline = None
    tds_attached = None
    improvement: Dict[str, float] = {}

    if baseline is not None:
        tds_baseline = {
            "peak_vram_mb": baseline.peak_vram_mb,
            "avg_latency_ms": baseline.avg_latency_ms,
            "throughput_tps": baseline.throughput_tps,
        }

    if attached is not None:
        tds_attached = {
            "peak_vram_mb": attached.peak_vram_mb,
            "decision_latency_avg_ms": attached.decision_latency_avg_ms,
            "cache_hit_rate": attached.cache_hit_rate,
            "applied_rate": attached.applied_rate,
            "memory_savings_pct": attached.memory_savings_pct,
        }
        # 提升度（仅用真实可比的量）
        improvement["显存峰值节省"] = attached.memory_savings_pct
        improvement["缓存命中率"] = attached.cache_hit_rate * 100
        improvement["计划执行率"] = attached.applied_rate * 100

    result = CompetitiveResult(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        task_description=task_description,
        model_name=model_name,
        tds_baseline=tds_baseline,
        tds_attached=tds_attached,
        matrix=matrix,
        improvement=improvement,
    )

    result.print_report()
    return result
