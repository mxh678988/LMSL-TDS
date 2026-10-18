"""TDS 挂载后性能测量模块。

目标：测量 TDS SDK 挂载后、相同调度负载下的性能特征，量化 SDK 的价值。

接口语义（与 SDK 真实输出对齐，杜绝臆测字段）
----------------------------------------------
TDSRuntime.run_task 返回的是**调度决策元数据**：
  {step, primary, scores, pressure, action, plan, applied, metrics}。
SDK 的价值在于"调度决策本身的开销、命中率与分流质量"，而**不是**模型生成的
正文吞吐（那取决于宿主后端）。因此本模块只采集 SDK 真实暴露的接口：

  - 调度开销       <- Scheduler.trace_report()['latency_ms']（内核自报）
  - 缓存/块命中率  <- Metrics.report()['cache_hit_rate']
  - 计划执行率     <- DecisionTrace.summary()['applied_rate']
  - 四象/八卦/动作 <- DecisionTrace.summary()['by_pressure'|'by_domain'|'by_action']
  - 显存变化       <- nvidia-smi 采样（宿主侧观测）

全部字段均可由真实调用复现，不含任何推理吞吐类推断值。
"""

from __future__ import annotations

import time
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..api import TDSRuntime
from .baseline import BaselineResult


@dataclass
class AttachedResult:
    """TDS 挂载后测量结果（字段全部对应 SDK 真实接口）。"""

    timestamp: str
    task_description: str
    model_name: str = ""
    repeat: int = 0

    # ---- 调度开销（内核自报，来自 trace.summary）----
    decision_latency_avg_ms: float = 0.0
    decision_latency_max_ms: float = 0.0

    # ---- 墙钟延迟（外部观测，含适配器调用与显存采样）----
    wall_latency_avg_ms: float = 0.0
    wall_latency_min_ms: float = 0.0
    wall_latency_max_ms: float = 0.0

    # ---- 命中与执行质量（SDK 核心价值指标）----
    cache_hit_rate: float = 0.0        # 块/缓存复用命中率
    applied_rate: float = 0.0          # 计划被适配层真正执行的比例

    # ---- 分流质量分布 ----
    pressure_distribution: Dict[str, int] = field(default_factory=dict)  # 四象
    domain_distribution: Dict[str, int] = field(default_factory=dict)    # 八卦
    action_distribution: Dict[str, int] = field(default_factory=dict)    # 动作

    # ---- 显存（宿主侧观测）----
    vram_samples_mb: List[float] = field(default_factory=list)
    peak_vram_mb: float = 0.0
    memory_savings_pct: float = 0.0    # 相对 baseline 的峰值显存节省

    # ---- 原始决策快照（逐次 run_task 的结果）----
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    trace_lines: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"TDS 挂载后测量结果:\n"
            f"  任务: {self.task_description}\n"
            f"  模型: {self.model_name or '未指定'}\n"
            f"  重复: {self.repeat} 次\n"
            f"  调度开销(内核): avg {self.decision_latency_avg_ms:.2f} ms / "
            f"max {self.decision_latency_max_ms:.2f} ms\n"
            f"  墙钟延迟: avg {self.wall_latency_avg_ms:.2f} ms\n"
            f"  缓存命中率: {self.cache_hit_rate * 100:.1f}%\n"
            f"  计划执行率: {self.applied_rate * 100:.1f}%\n"
            f"  峰值显存: {self.peak_vram_mb:.0f} MB\n"
            f"  显存节省: {self.memory_savings_pct:.1f}%"
        )


def measure_attached(
    runtime: TDSRuntime,
    task_description: str,
    model_name: str = "",
    repeat: int = 3,
    baseline: Optional[BaselineResult] = None,
) -> AttachedResult:
    """执行 TDS 挂载后测量。

    参数:
        runtime: TDSRuntime 实例（需先 attach）
        task_description: 测试任务描述
        model_name: 模型名称
        repeat: 重复次数
        baseline: 基线测量结果（可选，用于显存节省计算）

    返回:
        AttachedResult: TDS 挂载后测量结果
    """
    if runtime.is_detached():
        raise RuntimeError("SDK 未挂载，请先调用 runtime.attach()")

    results: List[Dict[str, Any]] = []
    wall_latencies_ms: List[float] = []
    vram_samples_mb: List[float] = []
    decisions: List[Dict[str, Any]] = []
    trace_lines: List[str] = []
    cache_hit_rates: List[float] = []
    applied_rates: List[float] = []
    decision_latencies_ms: List[float] = []
    peak_vram_mb = 0.0

    print(f"[TDS测量] 任务: {task_description}")
    print(f"[TDS测量] 模型: {model_name or '未指定'}")
    print(f"[TDS测量] 重复: {repeat} 次")
    print()

    for i in range(repeat):
        print(f"  [测试 {i+1}/{repeat}] 执行中...")

        # 采样前记录显存
        vram_before = _get_vram_state()
        vram_samples_mb.append(vram_before.get("used_mb", 0))

        # 测量墙钟延迟（含适配器调用与调度）
        t_wall_start = time.perf_counter()

        # 执行 TDS run_task
        try:
            result = runtime.run_task(task_description)
            wall_latency = (time.perf_counter() - t_wall_start) * 1000.0
        except Exception as e:
            wall_latency = (time.perf_counter() - t_wall_start) * 1000.0
            print(f"    测试失败: {e}")
            continue

        wall_latencies_ms.append(wall_latency)

        # 提取 SDK 核心指标
        metrics = result.get("metrics", {})
        # run_task 返回值不含 trace_report；调度开销须从 SDK 真实暴露的
        # DecisionTrace.summary() 读取（scheduler.trace_report()）。
        trace_report = {}
        sched = getattr(runtime, "scheduler", None)
        if sched is not None:
            try:
                trace_report = sched.trace_report()
            except Exception:
                trace_report = {}
        scores = result.get("scores", {})

        # 缓存命中率
        cache_hit = metrics.get("cache_hit_rate", 0.0)
        cache_hit_rates.append(cache_hit)

        # 计划执行率
        applied = result.get("applied", False)
        applied_rates.append(1.0 if applied else 0.0)

        # 调度开销（从 trace_report 取平均延迟）
        if trace_report and "latency_ms" in trace_report:
            lat = trace_report["latency_ms"]
            decision_latencies_ms.append(lat.get("avg", 0))

        # 决策快照
        decisions.append({
            "repeat": i + 1,
            "wall_latency_ms": wall_latency,
            "primary": result.get("primary", ""),
            "pressure": result.get("pressure", ""),
            "action": result.get("action", ""),
            "applied": applied,
            "scores": scores,
        })

        # 显存采样
        vram_after = _get_vram_state()
        vram_samples_mb.append(vram_after.get("used_mb", 0))
        peak_vram_mb = max(peak_vram_mb, vram_after.get("used_mb", 0))

        # 打印摘要
        print(f"    墙钟延迟: {wall_latency:.1f} ms")
        print(f"    优先级域: {result.get('primary', 'N/A')} | "
              f"压力: {result.get('pressure', 'N/A')} | "
              f"动作: {result.get('action', 'N/A')} | "
              f"命中: {cache_hit * 100:.1f}%")
        print(f"    显存: {vram_after.get('used_mb', 0):.0f} MB")

        # 获取决策轨迹行
        if hasattr(runtime, "scheduler") and runtime.scheduler:
            lines = runtime.scheduler.trace_lines(limit=5)
            trace_lines.extend(lines)

    # ---- 计算统计指标 ----
    n = len(wall_latencies_ms)
    if n == 0:
        print("[警告] 无有效测试数据")
        return AttachedResult(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            task_description=task_description,
            model_name=model_name,
            repeat=repeat,
        )

    avg_wall = sum(wall_latencies_ms) / n
    min_wall = min(wall_latencies_ms)
    max_wall = max(wall_latencies_ms)

    avg_cache = sum(cache_hit_rates) / n if cache_hit_rates else 0
    avg_applied = sum(applied_rates) / n if applied_rates else 0

    avg_decision_lat = sum(decision_latencies_ms) / len(decision_latencies_ms) if decision_latencies_ms else 0
    max_decision_lat = max(decision_latencies_ms) if decision_latencies_ms else 0

    # 四象分布
    pressure_counts: Dict[str, int] = {}
    domain_counts: Dict[str, int] = {}
    action_counts: Dict[str, int] = {}
    for d in decisions:
        p = d.get("pressure", "unknown")
        pressure_counts[p] = pressure_counts.get(p, 0) + 1
        dom = d.get("primary", "unknown")
        domain_counts[dom] = domain_counts.get(dom, 0) + 1
        act = d.get("action", "unknown")
        action_counts[act] = action_counts.get(act, 0) + 1

    # 显存节省
    memory_savings_pct = 0.0
    if baseline and baseline.peak_vram_mb > 0:
        memory_savings_pct = (baseline.peak_vram_mb - peak_vram_mb) / baseline.peak_vram_mb * 100

    result = AttachedResult(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        task_description=task_description,
        model_name=model_name,
        repeat=repeat,
        decision_latency_avg_ms=avg_decision_lat,
        decision_latency_max_ms=max_decision_lat,
        wall_latency_avg_ms=avg_wall,
        wall_latency_min_ms=min_wall,
        wall_latency_max_ms=max_wall,
        cache_hit_rate=avg_cache,
        applied_rate=avg_applied,
        pressure_distribution=pressure_counts,
        domain_distribution=domain_counts,
        action_distribution=action_counts,
        vram_samples_mb=vram_samples_mb,
        peak_vram_mb=peak_vram_mb,
        memory_savings_pct=memory_savings_pct,
        decisions=decisions,
        trace_lines=trace_lines,
    )

    print()
    print(result.summary())
    print()

    return result


def _get_vram_state() -> Dict[str, Any]:
    """获取当前显存状态。"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            return {
                "used_mb": float(parts[0]),
                "total_mb": float(parts[1]),
                "free_mb": float(parts[2]),
            }
    except Exception:
        pass
    return {"used_mb": 0, "total_mb": 0, "free_mb": 0}
