"""太极紫梅系列 · 进化引擎包。

双线独立进化：
- engine.py   ：软件总体进化（运行规则参数：BAGUA_KEEP_STEPS / 压力阈值 / 阴阳权重）
- ip_engine.py：知识产权进化（分类词典 / 域知识权重 / 域别名）

默认关闭（EvolutionConfig.enabled=False），不改变既有调度行为。
"""
from .config import EvolutionConfig, default_archive_dir
from .whitelist import Whitelist, ParamSpec, Risk, classify_risk, DEFAULT_WHITELIST
from .proposal import Proposal, ProposalBook
from .archive import EvolutionArchive, ChangeRecord, BackupEntry
from .engine import EvolutionEngine, ParameterStore
from .ip_engine import IPEvolutionEngine, KnowledgeStore

__all__ = [
    "EvolutionConfig", "default_archive_dir",
    "Whitelist", "ParamSpec", "Risk", "classify_risk", "DEFAULT_WHITELIST",
    "Proposal", "ProposalBook",
    "EvolutionArchive", "ChangeRecord", "BackupEntry",
    "EvolutionEngine", "ParameterStore",
    "IPEvolutionEngine", "KnowledgeStore",
]
