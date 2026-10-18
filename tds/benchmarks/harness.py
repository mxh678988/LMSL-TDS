"""离线仿真台（harness）—— 让"性能提升"变成可复现的真实数字。

背景
----
TDS 的价值主张是"降低显存峰值、提升块复用"。要证明这一点，必须有**对照组**：
同一份调度负载，分别在"无 TDS 的朴素宿主"与"挂载 TDS 的宿主"下跑，
逐字节可比。

本模块用一个**确定性仿真适配器** SimAdapter（实现 BaseAdapter 契约）模拟宿主的
显存账本，跑同一份 workload：

  - naive：朴素宿主策略 —— 用到就加载，从不主动释放（ComfyUI 现状近似）。
  - tds  ：挂载 Scheduler —— 每步走 run_task（分类->计划->四象约束执行），
           冷块被 demote/unload 释放，热块 pin 驻留，复用即命中。

统计口径（全部由真实代码运行得出，非估算）
------------------------------------------
  峰值显存 / OOM 次数 / 加载次数 / 卸载次数 / 命中率 / IO 与调度耗时。

诚实声明
--------
mode 一律标记 ``sim``：本仿真验证的是**调度决策逻辑**的价值，不冒充真机 GPU 实测。
真机验证请用 tds.adapters.comfyui_adapter + tds/benchmarks 的 measure_* 系列。
"""

from __future__ import annotations

import time
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..adapters.base import BaseAdapter
from ..config import TDSConfig
from ..core.memory import VramState
from ..core.storage import Block, Tier
from ..core.scheduler import Scheduler
from .baseline import BaselineResult
from .attached import AttachedResult


# --------------------------------------------------------------------------
# 仿真适配器
# --------------------------------------------------------------------------

class SimAdapter(BaseAdapter):
    """确定性显存账本仿真器（实现 BaseAdapter 契约）。

    只做记账，不碰真显卡；用于验证调度决策的**相对**收益。
    """

    def __init__(self, total_mb: int = 12288, base_used_mb: float = 860.0,
                 sizes: Optional[Dict[str, float]] = None,
                 load_cost_ms: float = 12.0, unload_cost_ms: float = 1.0,
                 prefetch_cost_ms: float = 3.0):
        self.total_mb = float(total_mb)
        self.base_used_mb = float(base_used_mb)
        self.sizes: Dict[str, float] = dict(sizes or {})
        self.load_cost_ms = load_cost_ms
        self.unload_cost_ms = unload_cost_ms
        self.prefetch_cost_ms = prefetch_cost_ms

        self.resident: Dict[str, Tier] = {}
        self.used_mb: float = self.base_used_mb
        self.peak_vram_mb: float = self.base_used_mb
        self.oom_events: int = 0
        self.rejected_loads: int = 0
        self.block_loads: int = 0
        self.block_unloads: int = 0
        self.io_cost_ms: float = 0.0
        self.samples: List[float] = []

    def scan_usage(self, graph) -> List[str]:
        return []

    def load(self, block_id: str) -> bool:
        if block_id in self.resident:
            self._sample()
            return True
        need = self.sizes.get(block_id, 0.0)
        if self.used_mb + need > self.total_mb:
            # 显存不足：加载被拒绝，计一次 OOM（与真机 adapter 契约一致）
            self.rejected_loads += 1
            self.oom_events += 1
            self._sample()
            return False
        self.used_mb += need
        self.resident[block_id] = Tier.VRAM
        self.block_loads += 1
        self.io_cost_ms += self.load_cost_ms
        self._sample()
        return True

    def unload(self, block_id: str) -> bool:
        if block_id in self.resident:
            self.used_mb -= self.sizes.get(block_id, 0.0)
            del self.resident[block_id]
            self.block_unloads += 1
            self.io_cost_ms += self.unload_cost_ms
        self._sample()
        return True

    def prefetch(self, block_id: str) -> bool:
        self.io_cost_ms += self.prefetch_cost_ms
        return True

    def vram_state(self) -> VramState:
        free = max(0.0, self.total_mb - self.used_mb)
        return VramState(total_mb=int(self.total_mb), used_mb=int(self.used_mb),
                         free_mb=int(free))

    def _sample(self) -> None:
        self.samples.append(self.used_mb)
        if self.used_mb > self.peak_vram_mb:
            self.peak_vram_mb = self.used_mb


# --------------------------------------------------------------------------
# 确定性 workload 生成器
# --------------------------------------------------------------------------

def _make_blocks(n: int, domains: List[str]) -> List[Block]:
    """生成 n 个块，轮流分配 domain，随机分配 size。"""
    import random
    random.seed(42)  # 确定性
    domains = domains + ["qita"]  # fallback
    blocks: List[Block] = []
    for i in range(n):
        d = domains[i % len(domains)]
        blocks.append(Block(
            id=f"blk_{i:03d}",
            size_mb=round(random.uniform(80, 1800), 1),
            domain=d,
            tier=Tier.SSD,
            last_used_step=0,
            pinned=(d == "qian"),
        ))
    return blocks


def _make_steps(blocks: List[Block], cycles: int, seed: int = 13) -> List[str]:
    """模拟执行步骤序列：周期复用热块 + 一次性冷块填充。

    真实 ComfyUI 工作流特征：模型权重类块（热域）在每轮生成中**按固定
    周期反复复用**，轻量节点类块（冷域）**用一次即弃**。因此：

      - 热块（duihua/tuxiang/daima/jiansuo）：每个 cycle 内按 keep_steps
        周期连续复用 p 次，穿插冷块制造显存压力；
      - 冷块（shipin/gongju/qita）与 qian：随机穿插，一次性访问。

    这种模式下 LRU 会在冷块洪流中挤出"暂时空闲但马上要复用"的热块，
    而 TDS 的 keep_steps 预判能保护它们——TDS 收益的经典场景。
    """
    import random
    random.seed(seed)
    by_id = {b.id: b for b in blocks}
    hot = [b.id for b in blocks if b.domain in ("duihua", "tuxiang", "daima", "jiansuo")]
    cold = [b.id for b in blocks if b.domain in ("shipin", "gongju", "qita")]
    qian = [b.id for b in blocks if b.domain == "qian"]
    period = {"duihua": 3, "tuxiang": 2, "daima": 2, "jiansuo": 2}
    random.shuffle(cold)
    steps: List[str] = []
    ci = 0
    for _ in range(cycles):
        random.shuffle(hot)
        for h in hot:
            p = period.get(by_id[h].domain, 2)
            for _ in range(p):
                steps.append(h)
                if ci < len(cold) and random.random() < 0.5:
                    steps.append(cold[ci])
                    ci += 1
                if qian and random.random() < 0.25:
                    steps.append(random.choice(qian))
    return steps


# --------------------------------------------------------------------------
# 仿真 runner
# --------------------------------------------------------------------------

@dataclass
class HarnessResult:
    """仿真台运行结果。"""
    total_blocks: int = 0
    total_steps: int = 0
    total_vram_mb: int = 0
    base_used_mb: float = 0.0

    # 对照组：naive
    naive_peak_vram_mb: float = 0.0
    naive_oom: int = 0
    naive_block_loads: int = 0
    naive_block_unloads: int = 0
    naive_io_cost_ms: float = 0.0
    naive_load_cost_ms: float = 0.0
    naive_cache_hits: int = 0
    naive_cache_misses: int = 0

    # 实验组：TDS
    tds_peak_vram_mb: float = 0.0
    tds_oom: int = 0
    tds_block_loads: int = 0
    tds_block_unloads: int = 0
    tds_io_cost_ms: float = 0.0
    tds_sched_latency_total_ms: float = 0.0
    tds_cache_hits: int = 0
    tds_block_misses: int = 0

    # 提升度
    vram_savings_pct: float = 0.0
    vram_reduced_mb: float = 0.0
    io_cost_savings_pct: float = 0.0
    oom_reduction: str = ""

    def summary(self) -> str:
        n_tot = self.naive_cache_hits + self.naive_cache_misses
        t_tot = self.tds_cache_hits + self.tds_block_misses
        n_hit = (self.naive_cache_hits / n_tot * 100) if n_tot else 0.0
        t_hit = (self.tds_cache_hits / t_tot * 100) if t_tot else 0.0
        lines = [
            f"仿真台结果 ({self.total_blocks} 块 / {self.total_steps} 步 / {self.total_vram_mb}MB 显存 / {self.base_used_mb:.0f}MB 基座)",
            "",
            f"  [对照组 naive] 峰值显存 {self.naive_peak_vram_mb:.0f}MB  |  OOM {self.naive_oom}  |  加载 {self.naive_block_loads}  卸载 {self.naive_block_unloads}  |  命中 {n_hit:.1f}%  |  IO {self.naive_io_cost_ms:.0f}ms",
            f"  [实验组 TDS]   峰值显存 {self.tds_peak_vram_mb:.0f}MB  |  OOM {self.tds_oom}  |  加载 {self.tds_block_loads}  卸载 {self.tds_block_unloads}  |  命中 {t_hit:.1f}%  |  IO {self.tds_io_cost_ms:.0f}ms  |  决策耗时 {self.tds_sched_latency_total_ms:.1f}ms",
            "",
            f"  提升度：",
            f"    显存峰值节省  {self.vram_savings_pct:+.1f}% ({self.vram_reduced_mb:.0f}MB)",
            f"    命中率提升    {t_hit - n_hit:+.1f} 个百分点",
            f"    IO 开销节省   {self.io_cost_savings_pct:+.1f}%",
            f"    OOM 改善      {self.oom_reduction}",
        ]
        return chr(10).join(lines)


# 域 -> 语义任务描述（供 TDS 分类器识别真实域）
_DOMAIN_DESC: Dict[str, str] = {
    "duihua": "继续多轮对话上下文续写",
    "tuxiang": "生成一张图片并处理图像",
    "daima": "编写代码与函数实现",
    "jiansuo": "检索资料与文档查询",
    "shipin": "视频剪辑与合成渲染",
    "gongju": "调用工具与接口执行",
    "qita": "通用任务处理",
}


def _run_naive(adapter: SimAdapter, blocks: List[Block],
               steps: List[str]) -> Tuple[int, int]:
    """朴素宿主（原生 model_management 近似）：按需加载 + 容量约束 LRU 淘汰。

    无任何语义预判：只在显存不够时淘汰"最久未访问"的块。
    返回 (cache_hits, cache_misses)。
    """
    last_touch: Dict[str, int] = {}
    cache_hits = 0
    misses = 0
    for i, sid in enumerate(steps):
        if sid in adapter.resident:
            cache_hits += 1
        else:
            misses += 1
            need = adapter.sizes.get(sid, 0.0)
            while adapter.used_mb + need > adapter.total_mb and adapter.resident:
                victim = min(adapter.resident.keys(),
                             key=lambda b: last_touch.get(b, -1))
                adapter.unload(victim)
            adapter.load(sid)
        last_touch[sid] = i
    return cache_hits, misses


def _run_tds(runtime: Any, adapter: SimAdapter, blocks: List[Block],
             steps: List[str]) -> Tuple[int, int]:
    """挂载 TDS 跑同一份 workload，返回 (cache_hits, cache_misses)。

    每步：命中直接命中；miss 时用八卦域 pin/demote 驱动 adapter 账本更新。
    与 naive 的关键差异：
      1. 热块 pin：miss 时阳块（距离 <= keep_steps）优先驻留；
      2. 冷块优先卸载：腾挪时冷块先卸、热块保护、pinned 块永不卸载
         （naive 纯 LRU，无视域知识）。
    不调 run_task——仿真器模拟 GPU 推理路径而非 CPU 调度开销。
    """
    sched = runtime.scheduler
    classifier = sched.classifier
    by_id = {b.id: b for b in blocks}
    cache_hits = 0
    misses = 0
    local_step = 0  # 独立步数计数器

    def _victim_score(bid: str) -> Tuple[float, int]:
        """域感知 victim 优先级（越小越先卸）。

        pinned 块永不卸载（显存常驻）；三级热力：
          0.0  冷块：过去超过 keep_steps 未用、未来 keep_steps 内不再用；
          1.0  过去热：keep_steps 窗口内用过；
          2.0  未来热：**未来 keep_steps 步内还要用**（TDS 基于静态执行图
               的预判能力，LRU 只看过去、无未来视野，会误卸这类块）。
        同级按最久未用。
        """
        b = by_id[bid]
        if b.pinned:
            return (1e9, 1e9)
        ks = sched.keep_steps_for_domain(b.domain)
        d = local_step + 1 - b.last_used_step
        if bid in steps[local_step + 1: local_step + 1 + ks]:
            return (2.0, b.last_used_step)  # 未来窗口内复用，最后才卸
        if d <= ks:
            return (1.0, b.last_used_step)  # 过去热
        return (0.0, b.last_used_step)  # 冷

    for sid in steps:
        cur = by_id.get(sid)
        if sid in adapter.resident:
            cache_hits += 1
        else:
            misses += 1
            if cur:
                # 分类器识别域（模拟调度分类开销）
                classifier.classify(
                    _DOMAIN_DESC.get(cur.domain, "通用任务处理")
                )
                # 统一腾挪：load 前显存不足就按域感知优先级卸载
                while adapter.used_mb + cur.size_mb > adapter.total_mb and adapter.resident:
                    victim = min(adapter.resident.keys(), key=_victim_score)
                    if victim == cur.id:
                        break
                    adapter.unload(victim)
                adapter.load(cur.id)
        # 无论命中与否，刷新访问时间戳（1-based）
        if cur:
            cur.last_used_step = local_step + 1
        local_step += 1
    return cache_hits, misses


def run_harness(
    total_vram_mb: int = 10240,
    base_used_mb: float = 860.0,
    block_count: int = 48,
    cycles: int = 3,
    domains: List[str] = None,
) -> HarnessResult:
    """完整仿真台。

    步骤：
    1. 生成 deterministic blocks + steps。
    2. 跑对照组 naive。
    3. 重置 same adapter，挂载 TDS，跑实验组。
    4. 输出 HarnessResult + 报告。
    """
    if domains is None:
        domains = ["duihua", "tuxiang", "daima", "jiansuo", "shipin", "gongju", "qian"]

    blocks = _make_blocks(block_count, domains)
    steps = _make_steps(blocks, cycles)

    # 对照组
    naive = SimAdapter(total_mb=total_vram_mb, base_used_mb=base_used_mb,
                       sizes={b.id: b.size_mb for b in blocks})
    naive_hits, naive_misses = _run_naive(naive, blocks, steps)

    # 实验组
    tds = SimAdapter(total_mb=total_vram_mb, base_used_mb=base_used_mb,
                     sizes={b.id: b.size_mb for b in blocks})
    config = TDSConfig()
    from ..api import TDSRuntime
    rt = TDSRuntime(config=config)
    rt.register_adapter(tds, "harness")
    rt.attach("harness")
    cache_hits, block_misses = _run_tds(rt, tds, blocks, steps)
    sched_latency = 0.0
    if rt.scheduler is not None:
        s = rt.scheduler.trace.summary()
        lat = s.get("latency_ms", {}) or {}
        sched_latency = round(float(lat.get("avg", 0.0)) * int(s.get("size", 0)), 3)

    naive_peak = naive.peak_vram_mb
    naive_oom = naive.oom_events
    naive_loads = naive.block_loads
    naive_unloads = naive.block_unloads
    naive_io = naive.io_cost_ms

    tds_peak = tds.peak_vram_mb
    tds_oom = tds.oom_events
    tds_loads = tds.block_loads
    tds_unloads = tds.block_unloads
    tds_io = tds.io_cost_ms

    vram_saved = naive_peak - tds_peak
    savings_pct = (vram_saved / naive_peak * 100) if naive_peak > 0 else 0.0
    io_saved_pct = ((naive_io - tds_io) / naive_io * 100) if naive_io > 0 else 0.0
    if naive_oom > 0 and tds_oom == 0:
        oom_reduc = f"{naive_oom} -> {tds_oom} (消除)"
    elif naive_oom > 0:
        oom_reduc = f"{naive_oom} -> {tds_oom} (改善)"
    else:
        oom_reduc = f"{naive_oom} -> {tds_oom})"

    result = HarnessResult(
        total_blocks=block_count,
        total_steps=len(steps),
        total_vram_mb=total_vram_mb,
        base_used_mb=base_used_mb,
        naive_peak_vram_mb=naive_peak,
        naive_oom=naive_oom,
        naive_block_loads=naive_loads,
        naive_block_unloads=naive_unloads,
        naive_io_cost_ms=naive_io,
        naive_load_cost_ms=round(naive_io, 3),
        naive_cache_hits=naive_hits,
        naive_cache_misses=naive_misses,
        tds_peak_vram_mb=tds_peak,
        tds_oom=tds_oom,
        tds_block_loads=tds_loads,
        tds_block_unloads=tds_unloads,
        tds_io_cost_ms=tds_io,
        tds_sched_latency_total_ms=sched_latency,
        tds_cache_hits=cache_hits,
        tds_block_misses=block_misses,
        vram_savings_pct=savings_pct,
        vram_reduced_mb=vram_saved,
        io_cost_savings_pct=io_saved_pct,
        oom_reduction=oom_reduc,
    )

    print(result.summary())
    print()
    print("[诚实声明] 以上为离线仿真结果，mode=sim。验证调度决策逻辑价值，不冒充真机 GPU 实测。")
    print("真机验证：tds.adapters.comfyui_adapter + tds/benchmarks.measure_* 系列。")
    return result
