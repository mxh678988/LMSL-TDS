"""TDS Phase 1 离线验收：四象（压力） × 八卦（任务域）决策矩阵。

零 GPU、零第三方依赖，用于在无 ComfyUI 环境下验证调度核心行为。
运行：python tools/tds_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tds.config import TDSConfig
from tds.core import Scheduler, VramState, Block, Tier
from tds.adapters.base import BaseAdapter


class OfflineAdapter(BaseAdapter):
    """离线适配器：只记录调用，不触碰真实运行环境。"""

    def __init__(self, state):
        self._state = state
        self.calls = []

    def scan_usage(self, graph):
        return []

    def load(self, block_id):
        self.calls.append(("load", block_id))
        return True

    def unload(self, block_id):
        self.calls.append(("unload", block_id))
        return True

    def prefetch(self, block_id):
        self.calls.append(("prefetch", block_id))
        return True

    def vram_state(self):
        return self._state


SCENES = [
    ("低压力", VramState(total_mb=12288, used_mb=4000, free_mb=8288)),
    ("临界", VramState(total_mb=12288, used_mb=11000, free_mb=1288)),
    ("高压力", VramState(total_mb=12288, used_mb=11700, free_mb=588)),
    ("冻结", VramState(total_mb=12288, used_mb=12100, free_mb=188)),
]

TASKS = ["帮我生成一张图片", "这段 python 代码报错了", "继续", "嗯"]


def make_scheduler(state):
    cfg = TDSConfig()
    sched = Scheduler(cfg, adapter=OfflineAdapter(state))
    sched.step = 5
    blocks = [
        Block(id="ctx", size_mb=1024, domain="duihua", tier=Tier.RAM, last_used_step=5),
        Block(id="old", size_mb=1024, domain="gongju", tier=Tier.VRAM, last_used_step=0),
        Block(id="sys", size_mb=64, domain="qian", tier=Tier.VRAM, pinned=True),
    ]
    for b in blocks:
        sched.store.add(b)
    return sched, blocks


def main():
    print("TDS Phase 1 离线决策矩阵（RTX 3060 12GB 阈值）")
    print("=" * 74)
    for name, state in SCENES:
        print(f"\n[四象] {name}  水位 {state.used_mb}/{state.total_mb} "
              f"= {state.usage_ratio:.3f}")
        print(f"{'任务':<22}{'主域':<10}{'压力':<10}{'策略':<11}驻留/预取/释放")
        print("-" * 74)
        for task in TASKS:
            sched, blocks = make_scheduler(state)
            res = sched.run_task(task, blocks=blocks)
            calls = sched.adapter.calls
            pin_n = sum(1 for a, _ in calls if a == "load")
            pf_n = sum(1 for a, _ in calls if a == "prefetch")
            rel_n = sum(1 for a, _ in calls if a == "unload")
            print(f"{task:<22}{res['primary']:<10}{res['pressure']:<10}"
                  f"{res['action']:<11}{pin_n}/{pf_n}/{rel_n}")
        sched, blocks = make_scheduler(state)
        res = sched.run_task(TASKS[0], blocks=blocks)
        print(f"  「{TASKS[0]}」计划摘要: {res['plan']}")
        print(f"  适配器调用: {sched.adapter.calls}")
    print("\n" + "=" * 74)
    print("说明 1：低压力/临界允许驻留；高压力起禁止驻留与预取，仅放行释放类动作")
    print("        （下沉/淘汰），确保显存能被降下来。")
    print("说明 2：预取默认关闭（保守起步，prefetch_enabled=False），")
    print("        置 True 后低压力档的阳块才会真正下发 prefetch。")


if __name__ == "__main__":
    main()
