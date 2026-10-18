"""EvolutionEngine 配置：双线独立进化引擎的总开关与档案位置。

设计约束
--------
1. 零依赖（仅标准库），core 平台无关；
2. 默认关闭（enabled=False）：不接入任何 hook，不改变 scheduler /
   classifier 等既有模块的既有行为；
3. 四象阶段控制（pressure guard）在此声明：高压/冻结状态下禁用任何变更；
4. 两条进化线路（软件总体进化 / 知识产权进化）共享同一套开关与目录，
   但各自持有独立档案文件与独立策略，互不混用。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def default_archive_dir() -> str:
    """默认档案目录：TDS 仓库根 / evolution_archives。

    仓库根由本文件路径上推两级（tds/evolution/config.py -> 仓库根）。
    调用方可显式覆盖，便于测试与宿主集成。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))
    return os.path.join(repo_root, "evolution_archives")


@dataclass
class EvolutionConfig:
    """EvolutionEngine 总配置。

    - enabled=False（默认关闭）：两线引擎 evolve() 均为 no-op；
    - 高压/冻结（high/frozen）下即使 enabled=True 也禁止任何变更；
    - auto_apply_low=True：risk=low 且参数在白名单内的候选自动应用
      （应用前备份旧值 + 写 changelog）；medium/high 永远只出建议书。
    """
    enabled: bool = False
    archive_dir: str = field(default_factory=default_archive_dir)
    # 四象阶段控制：高压/冻结禁用变更
    pressure_guard: bool = True
    # 档案回退备份保留条数（环形）
    backup_keep: int = 8
    # risk=low 是否自动应用（关闭后所有候选只进建议书）
    auto_apply_low: bool = True
    # 学习窗口：每次 evolve 采集的决策轨迹条数上限
    learn_window: int = 200
    # 学习的最小样本量：不足则视为"无信号"，不产出候选变更
    min_samples: int = 10
    # 建议书持久化目录（None 表示不落盘，仅内存）
    proposal_dir: Optional[str] = None

    def ensure_dirs(self) -> None:
        if self.archive_dir:
            os.makedirs(self.archive_dir, exist_ok=True)
        if self.proposal_dir:
            os.makedirs(self.proposal_dir, exist_ok=True)
