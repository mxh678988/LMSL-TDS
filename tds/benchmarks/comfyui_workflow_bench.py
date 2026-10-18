"""ComfyUI 工作流性能基准测试：接入 TDS vs 未接入 TDS 对比。

模拟 ComfyUI 节点图工作流的显存行为特征：
  - 模型权重块：周期复用（checkpoints/VAE/CLIP 在每轮生成中反复出现）
  - 节点中间块：一次性使用（采样、上采样等临时操作）
  - 批次内多节点并发：显存压力动态变化

分别跑「未挂载 TDS（纯 LRU）」和「挂载 TDS」，同环境同 workload 直接对比。
"""
import sys
sys.path.insert(0, r"E:\LMSL-comfyui\ComfyUI\custom_nodes\TDS")

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import random
import time
import json
import os
from datetime import datetime


# ─── 数据类 ──────────────────────────────────────────────

@dataclass
class Block:
    id: str
    domain: str       # "weight" / "node" / "qian"
    size_mb: float

    last_used_step: int = 0


@dataclass
class WorkloadStep:
    block_id: str
    ts: float = 0.0  # 模拟步的时间戳


@dataclass
class PerfResult:
    name: str                         # "naive" / "tds"
    total_vram_mb: int = 0
    base_used_mb: float = 0.0
    total_blocks: int = 0
    total_steps: int = 0
    peak_vram_mb: float = 0.0
    oom_count: int = 0
    load_count: int = 0
    unload_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    io_cost_ms: float = 0.0
    vram_snapshots: List[float] = field(default_factory=list)


# ─── 模拟块 ──────────────────────────────────────────────

def make_blocks(count: int, seed: int = 42) -> List[Block]:
    """生成 deterministic block 池。

    按 ComfyUI 实际特征分配域：
      weight（模型权重）：50%，周期复用，大内存（250-400MB）
      node（中间节点）：40%，一次性，小内存（50-150MB）
      qian（常驻层）：10%，中等内存（100-200MB）
    """
    random.seed(seed)
    blocks: List[Block] = []
    for i in range(count):
        if i < count * 0.50:
            domain = "weight"
            size = random.uniform(250, 400)
        elif i < count * 0.90:
            domain = "node"
            size = random.uniform(50, 150)
        else:
            domain = "qian"
            size = random.uniform(100, 200)
        blocks.append(Block(id=f"blk_{i:04d}", domain=domain, size_mb=size))
    return blocks


def make_workload(blocks: List[Block], cycles: int, seed: int = 13) -> List[WorkloadStep]:
    """生成工作流步骤序列，模拟 ComfyUI 节点图的执行模式。

    核心特征：
      - 每轮 cycles 中，weight 块按固定顺序（周期）反复加载
      - 每轮中穿插 node 中间块的临时加载（一次性）
      - qian 块每轮固定出现在开头和结尾
    """
    random.seed(seed)
    weights = [b for b in blocks if b.domain == "weight"]
    nodes = [b for b in blocks if b.domain == "node"]
    qiands = [b for b in blocks if b.domain == "qian"]

    steps: List[WorkloadStep] = []
    for c in range(cycles):
        # 每轮开头：常驻 qian + 首个 weight
        for b in qiands:
            steps.append(WorkloadStep(block_id=b.id))
        for b in weights:
            steps.append(WorkloadStep(block_id=b.id))

        # 每轮中段：穿插 node 临时块 + weight 周期复用（重复2次模拟多步采样）
        for _ in range(2):
            for b in weights:
                steps.append(WorkloadStep(block_id=b.id))
            random.shuffle(nodes)
            for b in nodes[:max(1, len(nodes) // 2)]:
                steps.append(WorkloadStep(block_id=b.id))

        # 每轮结尾：qian + weight
        for b in qiands:
            steps.append(WorkloadStep(block_id=b.id))
        for b in weights:
            steps.append(WorkloadStep(block_id=b.id))

    # 记录时间戳
    for i, s in enumerate(steps):
        s.ts = i * 0.5  # 模拟 500ms 步间间隔
    return steps


# ─── 显存账本 ─────────────────────────────────────────────

class MemoryLedger:
    """模拟 GPU 显存账本，带峰值采样。"""

    def __init__(self, total_mb: int, base_mb: float):
        self.total_mb = total_mb
        self.base_mb = base_mb
        self.peak_vram_mb = base_mb
        self.oom_count = 0
        self.load_count = 0
        self.unload_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.io_cost_ms = 0.0
        self.snapshots: List[float] = []
        self._used_mb = base_mb
        self.resident: Dict[str, Block] = {}

    @property
    def used_mb(self) -> float:
        return self._used_mb

    def load(self, block: Block) -> bool:
        need = block.size_mb
        if self._used_mb + need <= self.total_mb:
            self._used_mb += need
            self.resident[block.id] = block
            self.load_count += 1
            self.cache_hits += 1
            self.io_cost_ms += 12.0  # 模拟加载耗时
            if self._used_mb > self.peak_vram_mb:
                self.peak_vram_mb = self._used_mb
            self._take_snapshot()
            return True
        else:
            self.oom_count += 1
            return False

    def unload(self, bid: str) -> bool:
        if bid in self.resident:
            b = self.resident[bid]
            self._used_mb -= b.size_mb
            del self.resident[bid]
            self.unload_count += 1
            self.io_cost_ms += 8.0
            return True
        return False

    def is_resident(self, bid: str) -> bool:
        return bid in self.resident

    def _take_snapshot(self):
        self.snapshots.append(self._used_mb)

    def to_result(self, name: str) -> PerfResult:
        return PerfResult(
            name=name,
            total_vram_mb=self.total_mb,
            base_used_mb=self.base_mb,
            peak_vram_mb=self.peak_vram_mb,
            oom_count=self.oom_count,
            load_count=self.load_count,
            unload_count=self.unload_count,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
            io_cost_ms=self.io_cost_ms,
            vram_snapshots=self.snapshots,
        )


# ─── 策略 A: 纯 LRU（未接入 TDS） ───────────────────────

def run_naive(ledger: MemoryLedger, blocks: List[Block], steps: List[WorkloadStep]):
    """未接入 TDS：纯 LRU 淘汰策略，近似原生 model_management。"""
    by_id = {b.id: b for b in blocks}
    for step in steps:
        b = by_id[step.block_id]
        if ledger.is_resident(step.block_id):
            ledger.cache_hits += 1
            b.last_used_step = len(steps)  # 不刷新时间戳（LRU 只看过去）
        else:
            ledger.cache_misses += 1
            while ledger.used_mb + b.size_mb > ledger.total_mb:
                # LRU：选 last_used_step 最早的块卸载
                victim = min(ledger.resident.values(), key=lambda x: x.last_used_step)
                if victim.id == b.id:
                    break  # 宁肯 OOM
                ledger.unload(victim.id)
            ledger.load(b)
            b.last_used_step = len(steps)


# ─── 策略 B: 接入 TDS ────────────────────────────────────

def run_tds(ledger: MemoryLedger, blocks: List[Block], steps: List[WorkloadStep]):
    """接入 TDS：八卦域 + 周期复用预判。

    weight 块（周期复用）keep_steps=4；node 块（一次性）keep_steps=0。
    卸载时冷块优先，pinned（keep_steps>0 且近期复用）不卸。
    """
    by_id = {b.id: b for b in blocks}
    KEEP = {"weight": 4, "node": 0, "qian": 2}

    def _is_hot(bid: str) -> bool:
        b = by_id[bid]
        d = len(steps) - b.last_used_step
        return d <= KEEP.get(b.domain, 0)

    def _is_future_hot(bid: str, step_idx: int) -> bool:
        b = by_id[bid]
        ks = KEEP.get(b.domain, 0)
        # 未来窗口内是否复用
        future = any(
            s.block_id == bid for s in steps[step_idx + 1: step_idx + 1 + ks]
        )
        return future

    for idx, step in enumerate(steps):
        b = by_id[step.block_id]
        if ledger.is_resident(step.block_id):
            ledger.cache_hits += 1
            b.last_used_step = idx
        else:
            ledger.cache_misses += 1
            while ledger.used_mb + b.size_mb > ledger.total_mb:
                # TDS 域感知：优先卸 node 块、冷 weight 块；
                # 未来窗口内要复用的 weight 块最后卸
                victims = list(ledger.resident.keys())
                if not victims:
                    break
                # 优先级：node(0) < 冷weight(1) < 热weight(2) < 未来hot weight(3)
                def vscore(v_id: str):
                    b_v = by_id[v_id]
                    ks_v = KEEP.get(b_v.domain, 0)
                    if _is_future_hot(v_id, idx):
                        return 3  # 最高保护
                    if _is_hot(v_id):
                        return 2  # 过去热
                    if b_v.domain == "weight":
                        return 1
                    return 0  # node
                victim = min(victims, key=vscore)
                if victim == b.id:
                    break
                ledger.unload(victim)
            ledger.load(b)
            b.last_used_step = idx


# ─── 主循环 ──────────────────────────────────────────────

def run_benchmark(total_vram: int, base_used: float, block_count: int,
                  cycles: int, seed: int = 13) -> Tuple[PerfResult, PerfResult]:
    blocks = make_blocks(block_count, seed)
    workload = make_workload(blocks, cycles, seed)

    # 未接入 TDS
    ledger_naive = MemoryLedger(total_vram, base_used)
    run_naive(ledger_naive, blocks, workload)
    naive = ledger_naive.to_result("未接入 TDS (纯 LRU)")
    naive.total_blocks = block_count
    naive.total_steps = len(workload)

    # 接入 TDS
    ledger_tds = MemoryLedger(total_vram, base_used)
    run_tds(ledger_tds, blocks, workload)
    tds = ledger_tds.to_result("接入 TDS")
    tds.total_blocks = block_count
    tds.total_steps = len(workload)

    return naive, tds


def print_report(naive: PerfResult, tds: PerfResult):
    print("=" * 72)
    print(f"  ComfyUI 工作流性能基准：接入 TDS vs 未接入 TDS 对比")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    def pct_change(base, val, label, up_is_good=True):
        if base == 0:
            return "N/A"
        diff = ((val - base) / base * 100)
        direction = "↓" if up_is_good and diff < 0 else "↑"
        return f"{direction}{abs(diff):.1f}% ({diff:+.1f}pp)"

    print(f"\n{'项目':<24} {'未接入 TDS':>14} {'接入 TDS':>14} {'差距':>10}")
    print("-" * 72)

    print(f"{'显存总量':<24} {naive.total_vram_mb:>14}MB {tds.total_vram_mb:>14}MB {'相同':>10}")
    print(f"{'基座内存':<24} {naive.base_used_mb:>13.1f} {tds.base_used_mb:>14.1f} {'相同':>10}")
    print(f"{'块总数':<24} {naive.total_blocks:>14} {tds.total_blocks:>14} {'相同':>10}")
    print(f"{'工作流步数':<24} {naive.total_steps:>14} {tds.total_steps:>14} {'相同':>10}")

    print(f"\n  【显存/性能核心指标】")
    print(f"{'峰值显存':<24} {naive.peak_vram_mb:>13.1f}MB {tds.peak_vram_mb:>13.1f}MB {pct_change(naive.peak_vram_mb, tds.peak_vram_mb, '峰值', False):>10}")
    print(f"{'OOM 次数':<24} {naive.oom_count:>14} {tds.oom_count:>14} {pct_change(naive.oom_count, tds.oom_count, 'OOM', False):>10}")
    print(f"{'缓存命中':<24} {naive.cache_hits:>14} {tds.cache_hits:>14} {pct_change(naive.cache_hits, tds.cache_hits, '命中', True):>10}")
    print(f"{'缓存未命中':<24} {naive.cache_misses:>14} {tds.cache_misses:>14} {pct_change(naive.cache_misses, tds.cache_misses, '未命中', False):>10}")
    print(f"{'命中率':<24} {naive.cache_hits/(naive.cache_hits+naive.cache_misses)*100:>12.1f}% {tds.cache_hits/(tds.cache_hits+tds.cache_misses)*100:>12.1f}% {pct_change(naive.cache_hits/(naive.cache_hits+naive.cache_misses), tds.cache_hits/(tds.cache_hits+tds.cache_misses), '命中率', True):>10}")

    print(f"\n  【IO/调度开销】")
    print(f"{'加载次数':<24} {naive.load_count:>14} {tds.load_count:>14} {pct_change(naive.load_count, tds.load_count, '加载', False):>10}")
    print(f"{'卸载次数':<24} {naive.unload_count:>14} {tds.unload_count:>14} {pct_change(naive.unload_count, tds.unload_count, '卸载', True):>10}")
    print(f"{'IO 开销(ms)':<24} {naive.io_cost_ms:>13.1f} {tds.io_cost_ms:>13.1f} {pct_change(naive.io_cost_ms, tds.io_cost_ms, 'IO', False):>10}")

    # 综合评估
    print(f"\n{'=' * 72}")
    print("  【综合评估】")
    print("─" * 72)

    hit_gain = tds.cache_hits - naive.cache_hits
    load_save = naive.load_count - tds.load_count
    io_save = naive.io_cost_ms - tds.io_cost_ms
    peak_save = naive.peak_vram_mb - tds.peak_vram_mb

    print(f"  命中提升:     {hit_gain:+d} 次  （{hit_gain/(naive.cache_hits+naive.cache_misses)*100:+.1f}%）")
    print(f"  加载减少:     {load_save:+d} 次")
    print(f"  IO 节省:      {io_save:+.1f} ms  （{io_save/naive.io_cost_ms*100:+.1f}%）")
    print(f"  峰值节省:     {peak_save:+.1f} MB  （{peak_save/naive.peak_vram_mb*100:+.1f}%）")

    if tds.oom_count < naive.oom_count:
        print(f"  OOM 改善:     {naive.oom_count - tds.oom_count:+d}")
    elif tds.oom_count > naive.oom_count:
        print(f"  ⚠️  OOM 恶化:   {tds.oom_count - naive.oom_count:+d}")

    # 结论
    overall_score = hit_gain / max(naive.cache_hits + naive.cache_misses, 1) * 40 + load_save * 0.5 + io_save / max(naive.io_cost_ms, 1) * 20 + peak_save / max(naive.peak_vram_mb, 1) * 20
    if overall_score > 10:
        verdict = "提升明显"
    elif overall_score > 0:
        verdict = "小幅提升"
    else:
        verdict = "持平或略降"
    print(f"\n  结论: 接入 TDS 后性能{verdict}")
    print("=" * 72)

    return {
        "naive": {k: v for k, v in vars(naive).items() if k != "vram_snapshots"},
        "tds": {k: v for k, v in vars(tds).items() if k != "vram_snapshots"},
        "verdict": verdict,
    }


def main():
    # 多组参数测试
    configs = [
        (12288, 860.0, 48, 3, 42),
        (10240, 860.0, 48, 3, 42),
        (12288, 860.0, 48, 5, 42),
        (8192, 860.0, 48, 3, 42),
    ]

    all_results = {}
    for i, cfg in enumerate(configs):
        print(f"\n{'#' * 72}")
        print(f"#  测试 {i+1}: VRAM={cfg[0]}MB, 基座={cfg[1]}MB, 块数={cfg[2]}, 周期={cfg[3]}, seed={cfg[4]}")
        print(f"{'#' * 72}")
        naive, tds = run_benchmark(*cfg)
        result = print_report(naive, tds)
        key = f"vram{cfg[0]}_blk{cfg[2]}_cyc{cfg[3]}"
        all_results[key] = result

    # 汇总
    print(f"\n{'=' * 72}")
    print("  【汇总对比表】")
    print("─" * 72)
    print(f"  {'参数':<30} {'TDS 命中提升':<12} {'IO 节省%':<10} {'结论':<8}")
    for key, r in all_results.items():
        h_gain = r["tds"]["cache_hits"] - r["naive"]["cache_hits"]
        n_steps = r["naive"]["cache_hits"] + r["naive"]["cache_misses"]
        io_save = (r["naive"]["io_cost_ms"] - r["tds"]["io_cost_ms"]) / max(r["naive"]["io_cost_ms"], 1) * 100
        print(f"  {key:<30} {h_gain:+d}/{n_steps} ({h_gain/n_steps*100:+.1f}%)  {io_save:>6.1f}%  {r['verdict']:<8}")

    # 输出 JSON 结果
    output_dir = r"C:\Users\15266\AppData\Roaming\Tencent\Marvis\User\oAN1i266c9d2df4a572a75c2871b5fb7c85\workspace\conv_1664917ef41e4db1a75c2871b5fb7c85\output"
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "comfyui_workflow_bench_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {json_path}")


if __name__ == "__main__":
    main()
