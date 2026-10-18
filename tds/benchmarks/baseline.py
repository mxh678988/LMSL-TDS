"""基线测量模块：无TDS时的性能基准数据收集。

目标：建立纯推理框架（ComfyUI/Ollama）的原始性能基线，用于后续对比。
"""

from __future__ import annotations

import time
import json
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class BaselineResult:
    """基线测量结果。"""
    timestamp: str
    task_description: str
    model_name: str = ""
    
    # 核心指标
    peak_vram_mb: float = 0.0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    throughput_tps: float = 0.0
    
    # 内存指标
    system_memory_mb: float = 0.0
    system_memory_percent: float = 0.0
    
    # 磁盘指标
    disk_read_mb_s: float = 0.0
    disk_write_mb_s: float = 0.0
    
    # 重复测试数据
    latencies_ms: List[float] = field(default_factory=list)
    responses: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        return (
            f"基线测量结果:\n"
            f"  任务: {self.task_description}\n"
            f"  模型: {self.model_name}\n"
            f"  峰值显存: {self.peak_vram_mb:.0f} MB\n"
            f"  平均延迟: {self.avg_latency_ms:.1f} ms\n"
            f"  吞吐量: {self.throughput_tps:.2f} tokens/s\n"
            f"  系统内存: {self.system_memory_mb:.1f} MB ({self.system_memory_percent:.1f}%)"
        )


def measure_baseline(
    task_description: str,
    model_name: str = "",
    repeat: int = 3,
    host: str = "http://127.0.0.1:8200",  # ComfyUI
    ollama_host: str = "http://127.0.0.1:11434",  # Ollama
) -> BaselineResult:
    """执行基线测量，采集无TDS时的性能数据。
    
    参数:
        task_description: 测试任务描述
        model_name: 模型名称
        repeat: 重复次数
        host: ComfyUI 端点
        ollama_host: Ollama 端点
        
    返回:
        BaselineResult: 基线测量结果
    """
    results: List[Dict[str, Any]] = []
    latencies_ms: List[float] = []
    responses: List[str] = []
    
    print(f"[基线测量] 任务: {task_description}")
    print(f"[基线测量] 模型: {model_name or '未指定'}")
    print(f"[基线测量] 重复: {repeat} 次")
    print()
    
    # 获取当前显存状态
    vram_state = _get_vram_state()
    peak_vram = vram_state.get("used_mb", 0)
    
    # 执行重复测试
    for i in range(repeat):
        print(f"  [测试 {i+1}/{repeat}] 执行中...")
        
        # 测试前记录显存
        vram_before = _get_vram_state()
        mem_before = vram_before.get("used_mb", 0)
        
        # 测量延迟
        start_time = time.perf_counter()
        try:
            response = _test_inference(task_description, model_name, ollama_host)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            print(f"    测试失败: {e}")
            continue
        
        # 记录响应
        responses.append(response)
        latencies_ms.append(latency_ms)
        
        # 测试后记录显存
        vram_after = _get_vram_state()
        mem_after = vram_after.get("used_mb", 0)
        peak_vram = max(peak_vram, mem_after)
        
        # 计算吞吐量（粗略估算）
        throughput = len(response) / (latency_ms / 1000.0) if latency_ms > 0 else 0
        
        print(f"    延迟: {latency_ms:.1f} ms, 响应长度: {len(response)} 字符")
        print(f"    显存: {mem_after:.0f} MB (峰值: {peak_vram:.0f} MB)")
        
        results.append({
            "repeat": i + 1,
            "latency_ms": latency_ms,
            "response_length": len(response),
            "throughput_tps": throughput,
            "peak_vram_mb": mem_after,
            "response": response,
        })
    
    # 计算统计指标
    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0
    min_latency = min(latencies_ms) if latencies_ms else float('inf')
    max_latency = max(latencies_ms) if latencies_ms else 0
    
    # 计算平均吞吐量
    total_throughput = sum(r["throughput_tps"] for r in results) / len(results) if results else 0
    
    # 获取系统内存
    sys_mem = _get_system_memory()
    
    # 获取磁盘带宽
    disk_bw = _measure_disk_bandwidth()
    
    result = BaselineResult(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        task_description=task_description,
        model_name=model_name,
        peak_vram_mb=peak_vram,
        avg_latency_ms=avg_latency,
        min_latency_ms=min_latency,
        max_latency_ms=max_latency,
        throughput_tps=total_throughput,
        system_memory_mb=sys_mem.get("used_mb", 0),
        system_memory_percent=sys_mem.get("percent", 0),
        disk_read_mb_s=disk_bw.get("read_mb_s", 0),
        disk_write_mb_s=disk_bw.get("write_mb_s", 0),
        latencies_ms=latencies_ms,
        responses=responses,
    )
    
    print()
    print(result.summary())
    print()
    
    return result


def _test_inference(task_description: str, model_name: str, host: str) -> str:
    """测试推理，返回响应文本。"""
    import urllib.request
    import json
    
    # 构建请求
    data = {
        "prompt": task_description,
        "seed": 42,
    }
    if model_name:
        data["model"] = model_name
    
    # 发送到 Ollama
    url = f"{host}/api/generate"
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    response = urllib.request.urlopen(req, timeout=60)
    result = json.loads(response.read().decode())
    return result.get("response", "")


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


def _get_system_memory() -> Dict[str, Any]:
    """获取系统内存状态。"""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "used_mb": vm.used / (1024 * 1024),
            "total_mb": vm.total / (1024 * 1024),
            "percent": vm.percent,
        }
    except Exception:
        return {"used_mb": 0, "total_mb": 0, "percent": 0}


def _measure_disk_bandwidth(workdir: str = None) -> Dict[str, Any]:
    """测量磁盘读写带宽。"""
    import os
    if workdir is None:
        import tempfile
        workdir = tempfile.mkdtemp()
    
    size_mb = 128
    path = os.path.join(workdir, "tds_bw_test.bin")
    chunk = b"\0" * (8 * 1024 * 1024)
    loops = max(1, size_mb // 8)
    
    try:
        # 写入测试
        t0 = time.time()
        with open(path, "wb") as f:
            for _ in range(loops):
                f.write(chunk)
                f.flush()
                os.fsync(f.fileno())
        write_s = max(1e-6, time.time() - t0)
        
        # 读取测试
        t0 = time.time()
        with open(path, "rb") as f:
            while f.read(8 * 1024 * 1024):
                pass
        read_s = max(1e-6, time.time() - t0)
        
        return {
            "size_mb": loops * 8,
            "write_mb_s": round(loops * 8 / write_s, 1),
            "read_mb_s": round(loops * 8 / read_s, 1),
        }
    finally:
        try:
            os.remove(path)
        except OSError:
            pass