"""TDS 性能证据收集框架

为 TDS SDK 上架提供可信的性能数据、竞品横向对比、可复现的基准测试。
"""

from .baseline import measure_baseline, BaselineResult
from .attached import measure_attached, AttachedResult
from .competitive import compare_competitors, CompetitiveResult
from .report import generate_report, ReportResult

__all__ = [
    "measure_baseline",
    "measure_attached",
    "compare_competitors",
    "generate_report",
    "BaselineResult",
    "AttachedResult",
    "CompetitiveResult",
    "ReportResult",
]