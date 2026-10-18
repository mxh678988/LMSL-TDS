"""TDS SDK 性能基准测试模块。

定位
----
SDK 级别的性能测试工具，覆盖三类测试：

  1. **单流测试**：串行执行任务，测量平均延迟、吞吐量、显存占用
  2. **并发测试**：并行执行 N 个任务，测量并发吞吐、资源竞争、内存膨胀
  3. **挂载/卸载测试**：测试热插拔开销，验证 SDK 动态挂载能力

使用方式
--------
from tds.benchmark import SDKBenchmark

benchmark = SDKBenchmark(runtime)

# 单流测试
result = benchmark.single_stream(test_cases=["测试用例1", "测试用例2"], repeat=5)

# 并发测试
result = benchmark.concurrent_streams(num_tasks=10, repeat=3)

# 挂载/卸载测试
result = benchmark.attach_detach_test(iterations=10)

# 综合基准
result = benchmark.run_full_benchmark()
"""
from __future__ import annotations

import concurrent.futures
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Callable

from .api import TDSRuntime, BenchmarkResult


# --------------------------------------------------------------------------
# 测试用例
# --------------------------------------------------------------------------

@dataclass
class TestCase:
    """一个基准测试用例。"""
    name: str
    description: str
    category: str = "general"  # general | math | code | reasoning | other

    def to_description(self) -> str:
        return self.description


# 内置测试用例集
BASIC_TEST_CASES: List[TestCase] = [
    TestCase(
        name="short_response",
        description="请回答：中国的首都是哪座城市？请用一句话回答。",
        category="general",
    ),
    TestCase(
        name="math_reasoning",
        description="一个书架上三层共放84本书，第二层是第一层的2倍，第三层比第一层多4本。第一层有多少本？只回答数字。",
        category="math",
    ),
    TestCase(
        name="code_generation",
        description="请生成一个Python函数，计算两个数的最大公约数，并给出注释。",
        category="code",
    ),
    TestCase(
        name="logical_reasoning",
        description="如果所有的猫都喜欢吃鱼，而汤姆是一只猫，那么汤姆喜欢吃什么？请用一句话回答。",
        category="reasoning",
    ),
    TestCase(
        name="summarize",
        description="用不超过50个字总结下面这段话：TDS是一个宿主无关的调度中间层，在显存与模型之间做分类预判，把不同规模的请求分流到合适的执行单元，内核不依赖任何第三方库，宿主对接统一收敛到适配器。",
        category="general",
    ),
    TestCase(
        name="list_tasks",
        description="请列出三个提高工作效率的方法。",
        category="general",
    ),
]


# --------------------------------------------------------------------------
# 结果模型
# --------------------------------------------------------------------------

@dataclass
class SingleStreamResult:
    """单流测试结果。"""
    total_duration_ms: float = 0.0
    num_tasks: int = 0
    num_failed: int = 0
    avg_latency_ms: float = 0.0
    avg_tps: float = 0.0
    peak_vram_mb: float = 0.0
    min_vram_mb: float = 0.0
    details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ConcurrentStreamResult:
    """并发测试结果。"""
    num_tasks: int = 0
    num_concurrent: int = 0
    total_duration_ms: float = 0.0
    num_failed: int = 0
    avg_latency_ms: float = 0.0
    avg_tps: float = 0.0
    peak_vram_mb: float = 0.0
    min_vram_mb: float = 0.0
    concurrency_overhead: float = 0.0  # 并发开销百分比
    details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FullBenchmarkResult:
    """综合基准测试结果。"""
    single_stream: Optional[SingleStreamResult] = None
    concurrent_stream: Optional[ConcurrentStreamResult] = None
    attach_detach: Optional[BenchmarkResult] = None
    total_duration_ms: float = 0.0
    summary: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# SDKBenchmark — 性能基准测试器
# --------------------------------------------------------------------------

class SDKBenchmark:
    """TDS SDK 性能基准测试器。

    参数
    ----
    runtime: TDSRuntime 实例，已注册适配器并处于挂载状态
    """

    def __init__(self, runtime: TDSRuntime):
        self.runtime = runtime

    # ---- 单流测试 ----

    def single_stream(self, test_cases: Optional[Sequence[TestCase]] = None,
                      repeat: int = 3) -> SingleStreamResult:
        """运行单流基准测试。

        参数
        ----
        test_cases: 测试用例列表，默认使用 BASIC_TEST_CASES
        repeat: 每个测试用例重复执行次数

        返回
        ----
        SingleStreamResult: 单流测试结果
        """
        if not test_cases:
            test_cases = BASIC_TEST_CASES

        results: List[Dict[str, Any]] = []
        total_duration = 0.0
        num_failed = 0
        total_tasks = 0
        peak_vram = 0.0
        min_vram = float('inf')

        for case in test_cases:
            for i in range(repeat):
                start_time = time.perf_counter()
                try:
                    result = self.runtime.run_task(case.to_description())
                    duration = (time.perf_counter() - start_time) * 1000.0
                    total_duration += duration
                    total_tasks += 1

                    # 记录显存
                    adapter = self.runtime.get_adapter()
                    if adapter:
                        try:
                            vram = adapter.vram_state()
                            if vram:
                                usage = vram.usage_mb
                                peak_vram = max(peak_vram, usage)
                                min_vram = min(min_vram, usage)
                        except:
                            pass

                    results.append({
                        "case": case.name,
                        "repeat": i + 1,
                        "duration_ms": duration,
                        "success": True,
                    })
                except Exception as e:
                    duration = (time.perf_counter() - start_time) * 1000.0
                    total_duration += duration
                    num_failed += 1
                    results.append({
                        "case": case.name,
                        "repeat": i + 1,
                        "duration_ms": duration,
                        "success": False,
                        "error": str(e),
                    })

        if min_vram == float('inf'):
            min_vram = 0.0

        return SingleStreamResult(
            total_duration_ms=total_duration,
            num_tasks=total_tasks,
            num_failed=num_failed,
            avg_latency_ms=total_duration / total_tasks if total_tasks > 0 else 0.0,
            avg_tps=total_tasks / (total_duration / 1000.0) if total_duration > 0 else 0.0,
            peak_vram_mb=peak_vram,
            min_vram_mb=min_vram,
            details=results,
        )

    # ---- 并发测试 ----

    def concurrent_streams(self, num_tasks: int = 10,
                           repeat: int = 3,
                           test_cases: Optional[Sequence[TestCase]] = None) -> ConcurrentStreamResult:
        """运行并发基准测试。

        参数
        ----
        num_tasks: 并发任务数
        repeat: 测试重复次数
        test_cases: 测试用例列表，默认使用 BASIC_TEST_CASES

        返回
        ----
        ConcurrentStreamResult: 并发测试结果
        """
        if not test_cases:
            test_cases = BASIC_TEST_CASES

        results: List[Dict[str, Any]] = []
        total_duration = 0.0
        num_failed = 0
        total_tasks = 0
        peak_vram = 0.0
        min_vram = float('inf')

        for r in range(repeat):
            start_time = time.perf_counter()

            def run_single_task(task_idx: int) -> Dict[str, Any]:
                case = test_cases[task_idx % len(test_cases)]
                task_start = time.perf_counter()
                try:
                    result = self.runtime.run_task(case.to_description())
                    duration = (time.perf_counter() - task_start) * 1000.0
                    return {"success": True, "duration_ms": duration}
                except Exception as e:
                    duration = (time.perf_counter() - task_start) * 1000.0
                    return {"success": False, "duration_ms": duration, "error": str(e)}

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_tasks) as executor:
                futures = [executor.submit(run_single_task, i) for i in range(num_tasks)]
                task_results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        task_result = future.result()
                        task_results.append(task_result)
                        total_tasks += 1
                        if task_result["success"]:
                            duration = task_result["duration_ms"]
                            total_duration += duration

                            # 记录显存
                            adapter = self.runtime.get_adapter()
                            if adapter:
                                try:
                                    vram = adapter.vram_state()
                                    if vram:
                                        usage = vram.usage_mb
                                        peak_vram = max(peak_vram, usage)
                                        min_vram = min(min_vram, usage)
                                except:
                                    pass
                        else:
                            num_failed += 1
                    except Exception as e:
                        num_failed += 1

            end_time = time.perf_counter()
            batch_duration = (end_time - start_time) * 1000.0
            total_duration += batch_duration

            results.append({
                "repeat": r + 1,
                "num_tasks": num_tasks,
                "duration_ms": batch_duration,
                "completed": total_tasks,
                "failed": num_failed,
            })

        if min_vram == float('inf'):
            min_vram = 0.0

        # 计算并发开销：与单流对比
        single_stream_avg = 0.0
        if total_tasks > 0:
            single_stream_avg = total_duration / total_tasks

        return ConcurrentStreamResult(
            num_tasks=total_tasks,
            num_concurrent=num_tasks,
            total_duration_ms=total_duration,
            num_failed=num_failed,
            avg_latency_ms=total_duration / total_tasks if total_tasks > 0 else 0.0,
            avg_tps=total_tasks / (total_duration / 1000.0) if total_duration > 0 else 0.0,
            peak_vram_mb=peak_vram,
            min_vram_mb=min_vram,
            concurrency_overhead=0.0,  # 需要与单流对比计算
            details=results,
        )

    # ---- 挂载/卸载测试 ----

    def attach_detach_test(self, iterations: int = 10) -> BenchmarkResult:
        """运行挂载/卸载性能基准测试。

        参数
        ----
        iterations: 测试迭代次数

        返回
        ----
        BenchmarkResult: 测试结果
        """
        return self.runtime.attach_detach_benchmark("attach_detach", iterations=iterations)

    # ---- 综合基准测试 ----

    def run_full_benchmark(self) -> FullBenchmarkResult:
        """运行综合基准测试。

        顺序执行：
        1. 单流测试
        2. 并发测试
        3. 挂载/卸载测试

        返回
        ----
        FullBenchmarkResult: 综合测试结果
        """
        full_start = time.perf_counter()

        # 单流测试
        single_stream = self.single_stream()

        # 并发测试
        concurrent_stream = self.concurrent_streams(num_tasks=10, repeat=3)

        # 挂载/卸载测试
        attach_detach = self.attach_detach_test(iterations=10)

        full_duration = (time.perf_counter() - full_start) * 1000.0

        return FullBenchmarkResult(
            single_stream=single_stream,
            concurrent_stream=concurrent_stream,
            attach_detach=attach_detach,
            total_duration_ms=full_duration,
            summary={
                "single_stream_tasks": single_stream.num_tasks,
                "single_stream_avg_latency_ms": single_stream.avg_latency_ms,
                "single_stream_avg_tps": single_stream.avg_tps,
                "concurrent_stream_tasks": concurrent_stream.num_tasks,
                "concurrent_stream_avg_latency_ms": concurrent_stream.avg_latency_ms,
                "concurrent_stream_avg_tps": concurrent_stream.avg_tps,
                "attach_detach_iterations": 10,
                "attach_detach_avg_ms": attach_detach.avg_latency_ms,
                "total_duration_ms": full_duration,
            },
        )

    # ---- 报告输出 ----

    def print_report(self, result: FullBenchmarkResult):
        """打印基准测试报告。"""
        print("=" * 60)
        print("TDS SDK 性能基准测试报告")
        print("=" * 60)

        # 单流测试结果
        if result.single_stream:
            ss = result.single_stream
            print("\n单流测试结果:")
            print(f"  总任务数: {ss.num_tasks}")
            print(f"  失败任务数: {ss.num_failed}")
            print(f"  平均延迟: {ss.avg_latency_ms:.2f} ms")
            print(f"  平均吞吐: {ss.avg_tps:.2f} tasks/s")
            print(f"  峰值显存: {ss.peak_vram_mb:.0f} MB")
            print(f"  最小显存: {ss.min_vram_mb:.0f} MB")

        # 并发测试结果
        if result.concurrent_stream:
            cs = result.concurrent_stream
            print("\n并发测试结果:")
            print(f"  总任务数: {cs.num_tasks}")
            print(f"  并发数: {cs.num_concurrent}")
            print(f"  失败任务数: {cs.num_failed}")
            print(f"  平均延迟: {cs.avg_latency_ms:.2f} ms")
            print(f"  平均吞吐: {cs.avg_tps:.2f} tasks/s")
            print(f"  峰值显存: {cs.peak_vram_mb:.0f} MB")
            print(f"  最小显存: {cs.min_vram_mb:.0f} MB")

        # 挂载/卸载测试结果
        if result.attach_detach:
            ad = result.attach_detach
            print("\n挂载/卸载测试结果:")
            print(f"  迭代次数: {ad.tasks_completed}")
            print(f"  平均挂载+卸载时间: {ad.avg_latency_ms:.2f} ms")
            print(f"  平均挂载时间: {ad.avg_attach_ms:.2f} ms")
            print(f"  平均卸载时间: {ad.avg_detach_ms:.2f} ms")

        print(f"\n总耗时: {result.total_duration_ms:.2f} ms")
        print("=" * 60)


# --------------------------------------------------------------------------
# 命令行入口
# --------------------------------------------------------------------------

def main():
    """命令行入口：python -m tds.benchmark"""
    import sys

    print("TDS SDK 性能基准测试工具")
    print("用法: python -m tds.benchmark")
    print()

    # 导入 Runtime 来创建测试
    from .api import TDSRuntime

    # 创建运行时（这里需要适配器才能测试，仅展示接口）
    print("接口已就绪，需要注册适配器后方可运行实际测试")
    print("示例代码:")
    print("  runtime = TDSRuntime()")
    print("  benchmark = SDKBenchmark(runtime)")
    print("  result = benchmark.run_full_benchmark()")
    print("  benchmark.print_report(result)")


if __name__ == "__main__":
    main()