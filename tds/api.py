"""SDK 挂载/卸载入口层 —— 宿主无关的调度中间层 API。

定位
----
本模块是 TDS SDK 面向宿主的唯一入口。宿主通过 TDSRuntime 完成三件事：

  1. 注册适配器（扫描/加载/卸载/显存快照）；
  2. 挂载（attach）—— 将宿主当前上下文（模型、显存状态等）接入调度链路；
  3. 卸载（detach）—— 断开调度链路，释放 TDS 占用的资源，宿主继续独立运行。

挂载/卸载的设计要点：
  - **唯一入口**：所有操作通过 TDSRuntime 一个对象完成，无全局 state。
  - **幂等安全**：attach/detach 可多次调用，内部做状态保护。
  - **宿主解耦**：SDK 不依赖任何宿主类名硬编码，宿主只需实现 BaseAdapter 接口。
  - **热插拔**：挂载/卸载不重启宿主，调度状态（分类器、计划器、存储层）保留，
    仅断开外部适配器的调度驱动。

设计哲学
--------
挂载（attach）= 让 TDS 接管调度决策，adapter 的 load/unload/prefetch 被调度器驱动；
卸载（detach）= 让宿主恢复自主调度，adapter 直接服务于宿主自身框架，不再受 TDS 驱动。

宿主对接收敛：宿主把自身的适配器实现注册到 TDSRuntime，SDK 内部不引用
宿主任何类名、模块名或版本信息，仅通过 BaseAdapter 抽象层交互。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .core.scheduler import Scheduler
from .core.memory import VramState
from .config import TDSConfig, BAGUA_DOMAINS
from .core.classifier import BaguaClassifier


# --------------------------------------------------------------------------
# 运行时状态
# --------------------------------------------------------------------------

class RuntimeState:
    """SDK 运行时状态机。"""
    INIT = "init"       # 初始化，未挂载
    ATTACHED = "attached"   # 已挂载，调度器接管
    DETACHED = "detached"   # 已卸载，适配器独立

    ALL = [INIT, ATTACHED, DETACHED]


@dataclass
class AttachResult:
    """attach() 返回结果。"""
    success: bool = True
    adapter_name: str = ""
    vram_before: Optional[Dict[str, Any]] = None
    vram_after: Optional[Dict[str, Any]] = None
    message: str = ""


@dataclass
class DetachResult:
    """detach() 返回结果。"""
    success: bool = True
    adapters_detached: int = 0
    message: str = ""


@dataclass
class BenchmarkResult:
    """基准测试返回结果。"""
    name: str = ""
    mode: str = ""        # single | concurrent
    duration_ms: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_latency_ms: float = 0.0
    avg_tps: float = 0.0
    peak_vram_mb: float = 0.0
    min_vram_mb: float = 0.0
    details: List[Dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------
# TDSRuntime — 唯一入口
# --------------------------------------------------------------------------

class TDSRuntime:
    """SDK 挂载/卸载运行时。

    使用方式
    --------
    1. 创建：runtime = TDSRuntime()
    2. 注册适配器：runtime.register_adapter(MyAdapter())
    3. 挂载：runtime.attach()
    4. 执行调度：result = runtime.run_task("用户输入文本")
    5. 卸载：runtime.detach()

    参数
    ----
    config: TDSConfig, 可选，SDK 配置，默认 TDSConfig 实例
    """

    def __init__(self, config: Optional[TDSConfig] = None):
        self.cfg = config or TDSConfig()
        self.scheduler: Optional[Scheduler] = None
        self._adapters: Dict[str, Any] = {}
        self._state = RuntimeState.INIT
        self._lock = threading.RLock()
        # 显存账本：最近一次调度后的 vram 快照（成功/异常路径均回填，问题6）
        self.last_vram: Optional[Dict[str, Any]] = None

    @property
    def state(self) -> str:
        return self._state

    def is_attached(self) -> bool:
        return self._state == RuntimeState.ATTACHED

    def is_detached(self) -> bool:
        return self._state == RuntimeState.DETACHED

    # ---- 适配器管理 ----

    def register_adapter(self, adapter: Any, name: str = "default") -> bool:
        """注册宿主适配器。

        参数
        ----
        adapter: 实现 BaseAdapter 接口的适配器实例
        name: 适配器名称（支持多适配器注册）

        返回
        ----
        bool: 注册成功为 True

        注意
        ----
        - 适配器是宿主的实现，SDK 内部不引用宿主类名
        - 同一 name 注册会被覆盖
        """
        if not hasattr(adapter, "scan_usage"):
            raise ValueError("适配器必须实现 scan_usage 方法")
        if not hasattr(adapter, "load"):
            raise ValueError("适配器必须实现 load 方法")
        if not hasattr(adapter, "unload"):
            raise ValueError("适配器必须实现 unload 方法")
        if not hasattr(adapter, "prefetch"):
            raise ValueError("适配器必须实现 prefetch 方法")
        if not hasattr(adapter, "vram_state"):
            raise ValueError("适配器必须实现 vram_state 方法")

        with self._lock:
            self._adapters[name] = adapter

        return True

    def get_adapter(self, name: str = "default") -> Optional[Any]:
        """获取已注册的适配器。"""
        return self._adapters.get(name)

    def list_adapters(self) -> List[str]:
        """列出所有已注册的适配器名称。"""
        return list(self._adapters.keys())

    # ---- 挂载 / 卸载 ----

    def attach(self, adapter_name: str = "default") -> AttachResult:
        """挂载 SDK —— 将适配器接入调度链路。

        挂载后，调度器会接管 adapter 的 load/unload/prefetch 调用，
        宿主无需直接调用适配器方法。

        参数
        ----
        adapter_name: 要挂载的适配器名称

        返回
        ----
        AttachResult: 挂载结果

        异常
        ----
        RuntimeError: 适配器未注册或 SDK 已挂载
        """
        with self._lock:
            if adapter_name not in self._adapters:
                raise RuntimeError(f"适配器未注册: {adapter_name}")

            if self._state == RuntimeState.ATTACHED:
                return AttachResult(
                    success=False,
                    message=f"SDK 已挂载，无需重复挂载"
                )

            adapter = self._adapters[adapter_name]
            vram_before = self._capture_vram(adapter)

            # 初始化调度器并绑定适配器
            self.scheduler = Scheduler(config=self.cfg, adapter=adapter)

            # 标记挂载完成
            self._state = RuntimeState.ATTACHED

            vram_after = self._capture_vram(adapter)

            return AttachResult(
                success=True,
                adapter_name=adapter_name,
                vram_before=vram_before,
                vram_after=vram_after,
                message=f"SDK 挂载成功，adapter={adapter_name}"
            )

    def detach(self) -> DetachResult:
        """卸载 SDK —— 断开调度链路，释放 TDS 资源。

        卸载后，适配器恢复到独立运行状态，宿主框架直接管理调度。
        TDS 内部状态（分类器、计划器）保留，可随时重新挂载。

        返回
        ----
        DetachResult: 卸载结果

        异常
        ----
        RuntimeError: SDK 未挂载
        """
        with self._lock:
            if self._state == RuntimeState.DETACHED:
                return DetachResult(
                    success=False,
                    message="SDK 已卸载，无需重复卸载"
                )

            if self._state != RuntimeState.ATTACHED or self.scheduler is None:
                raise RuntimeError("SDK 未挂载，无法卸载")

            # 获取调度器关联的适配器
            scheduler_adapter = self.scheduler.adapter
            scheduler_adapter_name = None
            for name, adapter in self._adapters.items():
                if adapter is scheduler_adapter:
                    scheduler_adapter_name = name
                    break

            # 断开调度器（adapter 仍保留在 _adapters 中供后续重新挂载）
            self.scheduler.adapter = None
            self.scheduler = None
            self._state = RuntimeState.DETACHED

            return DetachResult(
                success=True,
                adapters_detached=1,
                message=f"SDK 卸载成功"
            )

    # ---- 调度入口 ----

    def run_task(self, task_desc: str, graph=None, blocks=None) -> Dict[str, Any]:
        """执行调度任务。

        挂载后调用，由调度器自动驱动适配器完成 load/unload/prefetch。

        参数
        ----
        task_desc: 任务描述（文本，用于语义分类）
        graph: 执行图（可选）。传入时调度器先经 adapter.scan_usage(graph)
               扫描出块序列并登记，再据此生成调度计划 —— 这是 scan 结果
               真实驱动 adapter.load/unload 的闭环接线。
        blocks: 可选，已构造的 Block 列表（与 graph 二选一，优先级更高）

        返回
        ----
        dict: 调度结果，包含 step、primary、scores、pressure、plan、metrics 等

        异常
        ----
        RuntimeError: SDK 未挂载
        """
        with self._lock:
            if self._state != RuntimeState.ATTACHED or self.scheduler is None:
                raise RuntimeError("SDK 未挂载，请先调用 attach()")

            try:
                result = self.scheduler.run_task(task_desc, blocks=blocks, graph=graph)
            except Exception:
                # 问题6：异常路径同样回填显存账本快照，避免账本停留在
                # 上一次成功值（外部 catch 后据 last_vram 判断水位会失真）。
                self.last_vram = self._capture_vram(self.scheduler.adapter)
                raise
            self.last_vram = self._capture_vram(self.scheduler.adapter)
            return result

    # ---- 性能基准测试 ----

    def run_benchmark(self, test_name: str = "default", repeat: int = 3,
                      concurrent_tasks: int = 1) -> BenchmarkResult:
        """运行性能基准测试。

        测试类型：
        - single: 单流测试，逐个任务串行执行
        - concurrent: 并发测试，同时执行 N 个任务

        参数
        ----
        test_name: 测试名称
        repeat: 测试重复次数
        concurrent_tasks: 并发任务数（concurrent 模式下）

        返回
        ----
        BenchmarkResult: 测试结果

        异常
        ----
        RuntimeError: SDK 未挂载
        """
        with self._lock:
            if self._state != RuntimeState.ATTACHED or self.scheduler is None:
                raise RuntimeError("SDK 未挂载，请先调用 attach()")

            adapter = self.scheduler.adapter
            results: List[Dict[str, Any]] = []
            durations = []
            completed = 0
            failed = 0
            peak_vram = 0.0
            min_vram = float('inf')

            if concurrent_tasks > 1:
                # 并发模式
                for i in range(repeat):
                    start_time = time.perf_counter()
                    concurrent_results = []

                    for j in range(concurrent_tasks):
                        try:
                            # 每个并发任务使用不同的描述以模拟真实负载
                            task_desc = f"并发任务 {j+1}/{concurrent_tasks} 在测试 {i+1}/{repeat}"
                            result = self.scheduler.run_task(task_desc)
                            concurrent_results.append(result)
                        except Exception as e:
                            concurrent_results.append({"error": str(e)})
                            failed += 1

                        # 记录显存
                        try:
                            vram = adapter.vram_state()
                            if vram:
                                usage = vram.used_mb
                                peak_vram = max(peak_vram, usage)
                                min_vram = min(min_vram, usage)
                        except:
                            pass

                    end_time = time.perf_counter()
                    duration = (end_time - start_time) * 1000.0
                    durations.append(duration)
                    completed += len(concurrent_results)

                    results.append({
                        "repeat": i + 1,
                        "concurrent_tasks": concurrent_tasks,
                        "duration_ms": duration,
                        "completed": len(concurrent_results),
                        "failed": sum(1 for r in concurrent_results if "error" in r)
                    })
            else:
                # 单流模式
                for i in range(repeat):
                    start_time = time.perf_counter()
                    try:
                        result = self.scheduler.run_task(f"基准测试单流任务 {i+1}/{repeat}")
                        completed += 1
                    except Exception as e:
                        failed += 1
                        results.append({"error": str(e)})

                    end_time = time.perf_counter()
                    duration = (end_time - start_time) * 1000.0
                    durations.append(duration)

                    # 记录显存
                    try:
                        vram = adapter.vram_state()
                        if vram:
                            usage = vram.used_mb
                            peak_vram = max(peak_vram, usage)
                            min_vram = min(min_vram, usage)
                    except:
                        pass

                    results.append({
                        "repeat": i + 1,
                        "duration_ms": duration,
                        "completed": 1,
                        "failed": 0 if "error" not in str(result) else 1
                    })

            avg_duration = sum(durations) / len(durations) if durations else 0.0
            avg_tps = completed / (sum(durations) / 1000.0) if durations and sum(durations) > 0 else 0.0

            if min_vram == float('inf'):
                min_vram = 0.0

            return BenchmarkResult(
                name=test_name,
                mode="concurrent" if concurrent_tasks > 1 else "single",
                duration_ms=sum(durations),
                tasks_completed=completed,
                tasks_failed=failed,
                avg_latency_ms=avg_duration,
                avg_tps=avg_tps,
                peak_vram_mb=peak_vram,
                min_vram_mb=min_vram,
                details=results
            )

    def attach_detach_benchmark(self, test_name: str = "attach_detach",
                                iterations: int = 5) -> BenchmarkResult:
        """挂载/卸载性能基准测试。

        测试挂载/卸载操作的开销，验证热插拔性能。

        参数
        ----
        test_name: 测试名称
        iterations: 测试迭代次数

        返回
        ----
        BenchmarkResult: 测试结果
        """
        if self._state != RuntimeState.INIT:
            raise RuntimeError("SDK 必须处于 INIT 状态才能运行挂载/卸载测试")

        if not self._adapters:
            raise RuntimeError("没有注册适配器，无法运行挂载/卸载测试")

        adapter_name = list(self._adapters.keys())[0]
        results: List[Dict[str, Any]] = []
        attach_durations = []
        detach_durations = []

        for i in range(iterations):
            # 挂载
            start_attach = time.perf_counter()
            try:
                attach_result = self.attach(adapter_name)
                attach_duration = (time.perf_counter() - start_attach) * 1000.0
                attach_durations.append(attach_duration)
            except Exception as e:
                attach_duration = (time.perf_counter() - start_attach) * 1000.0
                attach_durations.append(attach_duration)
                results.append({
                    "iteration": i + 1,
                    "phase": "attach",
                    "duration_ms": attach_duration,
                    "success": False,
                    "error": str(e)
                })
                continue

            # 卸载
            start_detach = time.perf_counter()
            try:
                detach_result = self.detach()
                detach_duration = (time.perf_counter() - start_detach) * 1000.0
                detach_durations.append(detach_duration)
            except Exception as e:
                detach_duration = (time.perf_counter() - start_detach) * 1000.0
                detach_durations.append(detach_duration)
                results.append({
                    "iteration": i + 1,
                    "phase": "detach",
                    "duration_ms": detach_duration,
                    "success": False,
                    "error": str(e)
                })

            results.append({
                "iteration": i + 1,
                "phase": "complete",
                "attach_duration_ms": attach_durations[-1],
                "detach_duration_ms": detach_durations[-1] if detach_durations else 0,
                "success": True
            })

        avg_attach = sum(attach_durations) / len(attach_durations) if attach_durations else 0.0
        avg_detach = sum(detach_durations) / len(detach_durations) if detach_durations else 0.0
        total_duration = sum(attach_durations) + sum(detach_durations)

        return BenchmarkResult(
            name=test_name,
            mode="attach_detach",
            duration_ms=total_duration,
            tasks_completed=iterations,
            tasks_failed=0,
            avg_latency_ms=(avg_attach + avg_detach),
            avg_tps=iterations / (total_duration / 1000.0) if total_duration > 0 else 0.0,
            peak_vram_mb=0.0,
            min_vram_mb=0.0,
            details=results,
            # 挂载/卸载测试额外字段
            avg_attach_ms=avg_attach,
            avg_detach_ms=avg_detach
        )

    # ---- 内部方法 ----

    def _capture_vram(self, adapter: Any) -> Optional[Dict[str, Any]]:
        """捕获当前显存状态。"""
        try:
            state = adapter.vram_state()
            if state:
                return {
                    "total_mb": state.total_mb,
                    "used_mb": state.used_mb,
                    "free_mb": state.free_mb,
                    "usage_mb": state.used_mb,
                    "usage_ratio": state.usage_ratio
                }
        except Exception:
            pass
        return None

    def reset(self):
        """重置 SDK 运行时到初始状态。"""
        with self._lock:
            self.scheduler = None
            self._adapters.clear()
            self._state = RuntimeState.INIT

    def __repr__(self):
        return (
            f"TDSRuntime(state={self._state}, "
            f"adapters={len(self._adapters)}, "
            f"version={__version__})"
        )


# ---- 版本信息 ----
__version__ = "0.1.0"
__author__ = "BY LAOMENG 网络工作室"
__all__ = ["TDSRuntime", "RuntimeState", "AttachResult", "DetachResult",
           "BenchmarkResult"]
