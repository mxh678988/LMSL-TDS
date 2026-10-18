#!/usr/bin/env python
"""TDS Phase 2 页级收益量化：预算联动扫描（离线仿真）。

背景
----
P2 真机压测结论：TDS ON 端到端均值 +11.4%（冷启动拖累），仅抖动达标。
待办：调预算（TDS_KV_BUDGET_MB 触发 evict_pages_to_budget）凸显**页级收益**——
在显存预算收紧时，KV 分页引擎按 LRU（默认）或太极域策略淘汰冷页、保护热页，
同等显存预算下可承载更长的上下文（≥1.5× 是 Phase2B 验收线），
并把"淘汰后换入"的 IO 成本与"保留热页命中"的收益量化成端到端时延影响。

本脚本做确定性离线仿真（mode=sim）：
  - 构造 8 层 PageTable，页大小 1MiB/页（head_dim=128, num_kv_heads=32, 128 tok/页）；
  - 模拟 3 轮长上下文生成：每轮每层追加 4 页，并跨轮复用既有热页；
  - 每轮结束调用 evict_pages_to_budget(budget)（与推理循环同一函数）；
  - 扫描多档预算：无预算（全驻留基线）→ 256MiB → 192MiB → 128MiB → 64MiB；
  - 统计：驻留峰值 / 淘汰页数 / freed / 下一轮换入 IO 成本 / 上下文承载倍数。

诚实声明：结果为调度决策逻辑的**相对**收益（mode=sim），不冒充真机 GPU 实测；
真机验证请走 tools/bench_multi_model.py + TDS_KV_BUDGET_MB 实机复测。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
BENCH_OUT = os.path.join(ROOT, "bench_out")
sys.path.insert(0, ROOT)

from tds.kv.page_manager import Page, PageTable, evict_pages_to_budget  # noqa: E402

# 页规格（对齐真实 llama KV：head_dim=128, num_kv_heads=32, 128 tokens/页）
HEAD_DIM = 128
NUM_KV_HEADS = 32
PAGE_TOKENS = 128
BYTES_PER_PAGE = 2 * HEAD_DIM * NUM_KV_HEADS * PAGE_TOKENS  # 1,048,576 = 1 MiB
LAYERS = 8
PAGES_PER_LAYER = 32          # 每层 32 页 = 4096 tokens 容量
ROUNDS = 3
NEW_PAGES_PER_ROUND = 4       # 每轮每层追加页数


def make_tables() -> List[PageTable]:
    """构造 8 层页表，全部页预建并绑定（模拟已写入的 KV 页）。"""
    tables: List[PageTable] = []
    for layer in range(LAYERS):
        pt = PageTable(
            capacity_tokens=PAGES_PER_LAYER * PAGE_TOKENS,
            page_size_tokens=PAGE_TOKENS,
            page_size_bytes=BYTES_PER_PAGE,
            layer=layer,
            dtype="float16",
            device="cuda:0",
            batch=1,
            head_dim=HEAD_DIM,
            num_kv_heads=NUM_KV_HEADS,
        )
        for i in range(PAGES_PER_LAYER):
            pid = f"L{layer}_p{i}"
            page = Page(
                page_id=pid,
                capacity_tokens=PAGE_TOKENS,
                dtype_bytes=BYTES_PER_PAGE // PAGE_TOKENS,
                device="cuda:0",
                offset=i * PAGE_TOKENS,
                layer_id=layer,
                head_dim=HEAD_DIM,
                num_kv_heads=NUM_KV_HEADS,
                domain="duihua",
                is_in_vram=True,
            )
            pt.pages[pid] = page
        tables.append(pt)
    return tables


def resident_bytes(tables: List[PageTable]) -> int:
    """当前驻留显存的页合计字节（仅统计 is_in_vram 页）。"""
    total = 0
    for t in tables:
        for p in t.pages.values():
            if p.is_in_vram:
                total += p.memory_bytes
    return total


def _get_page(tables: List[PageTable], layer: int, idx: int):
    """统一取页：优先驻留表，其次 RAM 缓存表；返回 (page, in_vram)。"""
    t = tables[layer]
    pid = f"L{layer}_p{idx}"
    if pid in t.pages:
        return t.pages[pid], True
    if pid in t.ram_buffer:
        return t.ram_buffer[pid], False
    return None, False


def _swap_in(tables: List[PageTable], layer: int, page: Page) -> None:
    """模拟换入：RAM 缓存表 -> 驻留表（等价 prefetch_to_vram 语义，不计真实张量）。"""
    t = tables[layer]
    t.ram_buffer.pop(page.page_id, None)
    t.pages[page.page_id] = page
    page.is_in_vram = True


def simulate(budget_mb: Optional[int]) -> Dict[str, Any]:
    """模拟 3 轮生成：每轮追加 NEW_PAGES_PER_ROUND 页/层并刷新访问序，
    然后按预算执行 evict_pages_to_budget。返回量化统计。"""
    tables = make_tables()
    step = 0
    # 首轮：仅访问前 12 页/层（模拟首轮已写入的热页）
    for layer in range(LAYERS):
        for idx in range(12):
            p = tables[layer].pages[f"L{layer}_p{idx}"]
            p.last_used_step = step
            p.access_count += 1
    step += 1

    # 稳态驻留：仅统计每轮预算淘汰后的驻留（初始全驻留是仿真装载态，不计入峰值）
    steady_mb = 0
    total_evicted = 0
    total_freed_mb = 0
    round_reuse_hits = 0
    round_reuse_misses = 0
    round_io_ms = 0.0
    budgets_used = budget_mb is not None

    for r in range(1, ROUNDS + 1):
        # 每轮：追加 4 页/层（模拟上下文增长），并跨轮复用前 12 页热页
        for layer in range(LAYERS):
            for idx in range(12):
                p, in_vram = _get_page(tables, layer, idx)
                if p is None:
                    continue
                p.last_used_step = step
                p.access_count += 1
                if in_vram:
                    round_reuse_hits += 1
                else:
                    round_reuse_misses += 1
                    round_io_ms += 12.0  # 换入成本（与 SimAdapter.load_cost_ms 一致）
                    _swap_in(tables, layer, p)
            for n in range(NEW_PAGES_PER_ROUND):
                idx = 12 + n + (r - 1) * NEW_PAGES_PER_ROUND
                if idx < PAGES_PER_LAYER:
                    p, in_vram = _get_page(tables, layer, idx)
                    if p is None:
                        continue
                    p.last_used_step = step
                    p.access_count += 1
                    if not in_vram:
                        round_reuse_misses += 1
                        round_io_ms += 12.0
                        _swap_in(tables, layer, p)
        step += 1

        # 预算淘汰联动（与推理循环同一入口）
        if budget_mb:
            stats = evict_pages_to_budget(budget_mb * 1024 * 1024,
                                          page_tables=tables)
            total_evicted += stats["evicted_pages"]
            total_freed_mb += stats["freed_bytes"] // (1024 * 1024)

        cur = resident_bytes(tables) // (1024 * 1024)
        if cur:
            steady_mb = max(steady_mb, cur) if steady_mb else cur

    return {
        "budget_mb": budget_mb,
        "total_pages": LAYERS * PAGES_PER_LAYER,
        "total_bytes_mb": LAYERS * PAGES_PER_LAYER * BYTES_PER_PAGE // (1024 * 1024),
        "steady_resident_mb": steady_mb,
        "evicted_pages": total_evicted,
        "freed_mb": total_freed_mb,
        "reuse_hits": round_reuse_hits,
        "reuse_misses": round_reuse_misses,
        "reuse_io_ms": round(round_io_ms, 1),
        "budgets_used": budgets_used,
        "context_pages_kept": max(0, PAGES_PER_LAYER - (total_evicted // LAYERS)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="TDS Phase2 预算联动页级收益扫描")
    ap.add_argument("--budgets", default="0,256,192,128,64",
                    help="预算档位（MiB，0=无预算基线），逗号分隔")
    ap.add_argument("--tag", default="", help="报告标签")
    args = ap.parse_args()

    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    rows: List[Dict[str, Any]] = []
    for b in budgets:
        r = simulate(b or None)
        rows.append(r)
        print(f"[sweep] budget={b or 'none':>4}MiB  steady={r['steady_resident_mb']:>4}MiB  "
              f"evicted={r['evicted_pages']:>3}  freed={r['freed_mb']:>4}MiB  "
              f"reuse_hit={r['reuse_hits']:>3}/{r['reuse_hits'] + r['reuse_misses']:>3}  "
              f"reuse_io={r['reuse_io_ms']:>6.1f}ms")

    # 基线（无预算）用于计算相对收益
    base = rows[0]
    report: Dict[str, Any] = {
        "meta": {
            "tool": "tools/bench_budget_sweep.py",
            "ts": datetime.now().isoformat(timespec="seconds"),
            "tag": args.tag or "budget-sweep",
            "mode": "sim",
            "page_bytes": BYTES_PER_PAGE,
            "layers": LAYERS,
            "pages_per_layer": PAGES_PER_LAYER,
            "rounds": ROUNDS,
            "new_pages_per_round": NEW_PAGES_PER_ROUND,
            "budgets_mb": budgets,
        },
        "base_nobudget": base,
        "scenarios": rows,
    }

    os.makedirs(BENCH_OUT, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or "budget-sweep"
    json_path = os.path.join(BENCH_OUT, f"bench_budget_{ts}_{tag}.json")
    md_path = os.path.join(BENCH_OUT, f"bench_budget_{ts}_{tag}.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# TDS Phase2 预算联动页级收益扫描（离线仿真）",
        "",
        f"- 时间: {report['meta']['ts']}",
        f"- 模式: sim（调度决策逻辑相对收益，非真机 GPU 实测）",
        f"- 页规格: {BYTES_PER_PAGE // (1024 * 1024)} MiB/页 "
        f"(head_dim={HEAD_DIM}, num_kv_heads={NUM_KV_HEADS}, {PAGE_TOKENS} tok/页)",
        f"- 规模: {LAYERS} 层 × {PAGES_PER_LAYER} 页/层 = {base['total_pages']} 页 "
        f"({base['total_bytes_mb']} MiB 全驻留)",
        f"- 负载: {ROUNDS} 轮生成，每轮每层追加 {NEW_PAGES_PER_ROUND} 页 + 跨轮复用 12 热页",
        "",
        "## 预算档位对比",
        "",
        "| 预算(MiB) | 稳态驻留(MiB) | 驻留节省 | 淘汰页 | 释放(MiB) | 热页复用命中 | 复用换入IO(ms) |",
        "|---|---|---|---|---|---|---|",
    ]
    base_peak = base["steady_resident_mb"]
    for r in rows:
        save_pct = f"{(1 - r['steady_resident_mb'] / base_peak) * 100:.1f}%" if base_peak else "-"
        tot = r["reuse_hits"] + r["reuse_misses"]
        hit_str = f"{r['reuse_hits']}/{tot}" if tot else "-"
        budget_label = "无预算" if not r["budgets_used"] else f"{r['budget_mb']}"
        lines.append(
            f"| {budget_label} | {r['steady_resident_mb']} | {save_pct} | "
            f"{r['evicted_pages']} | {r['freed_mb']} | {hit_str} | {r['reuse_io_ms']} |")
    lines += [
        "",
        "## 结论口径",
        "",
        "- 峰值节省：预算收紧后驻留显存相对无预算基线的下降比例（越紧越省显存）；",
        "- 热页复用命中：跨轮复用页仍驻留的比例；淘汰越多，下一轮换入 IO 成本越高；",
        "- 预算与 IO 的权衡点即页级调优空间：预算越紧显存越省、但换入成本上升；",
        "  TDS 域感知淘汰（TDS_KV_POLICY=taiji）可在同等预算下保护预测域热页、"
        "降低误淘汰（本扫描默认 LRU 口径）。",
        "",
        "*本报告由 tools/bench_budget_sweep.py 自动生成（mode=sim）。*",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[sweep] 完成: {json_path}")
    print(f"[sweep] 报告: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
