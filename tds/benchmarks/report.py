"""报告生成模块。

将基线、SDK 挂载后、竞品能力矩阵汇总为可上架的性能证据报告（Markdown + JSON）。
JSON 附带 SHA256，便于跨版本 diff 与防篡改。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .baseline import BaselineResult
from .attached import AttachedResult
from .competitive import CompetitiveResult, CAPABILITY_DIMS


@dataclass
class ReportResult:
    """报告生成结果。"""

    timestamp: str
    title: str
    md_path: str = ""
    json_path: str = ""
    sha256: str = ""
    summary: str = ""


def generate_report(
    baseline: Optional[BaselineResult] = None,
    attached: Optional[AttachedResult] = None,
    competitive: Optional[CompetitiveResult] = None,
    output_dir: str = "",
    title: str = "TDS 性能证据报告",
) -> ReportResult:
    """生成性能证据报告（Markdown + JSON with SHA256）。"""
    if not output_dir:
        output_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "bench_out")
        )
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    ts_short = time.strftime("%Y%m%d_%H%M%S")

    md: List[str] = []
    md.append(f"# {title}")
    md.append("")
    md.append(f"> 生成时间: {timestamp}")
    md.append("")
    md.append("---")
    md.append("")

    # 一、基线
    if baseline:
        md.append("## 一、基线（无 SDK）")
        md.append("")
        md.append("| 指标 | 数值 |")
        md.append("|------|------|")
        md.append(f"| 任务 | {baseline.task_description} |")
        md.append(f"| 模型 | {baseline.model_name or '未指定'} |")
        md.append(f"| 峰值显存 | {baseline.peak_vram_mb:.0f} MB |")
        md.append(f"| 平均延迟 | {baseline.avg_latency_ms:.1f} ms |")
        md.append(f"| 延迟范围 | {baseline.min_latency_ms:.1f} ~ {baseline.max_latency_ms:.1f} ms |")
        md.append(f"| 吞吐量 | {baseline.throughput_tps:.2f} tokens/s |")
        md.append(f"| 系统内存 | {baseline.system_memory_mb:.1f} MB ({baseline.system_memory_percent:.1f}%) |")
        md.append("")

    # 二、SDK 挂载后
    if attached:
        md.append("## 二、SDK 挂载后实测")
        md.append("")
        md.append("| 指标 | 数值 |")
        md.append("|------|------|")
        md.append(f"| 任务 | {attached.task_description} |")
        md.append(f"| 重复次数 | {attached.repeat} |")
        md.append(f"| 调度开销(内核) | avg {attached.decision_latency_avg_ms:.2f} ms / max {attached.decision_latency_max_ms:.2f} ms |")
        md.append(f"| 墙钟延迟 | avg {attached.wall_latency_avg_ms:.2f} ms |")
        md.append(f"| 缓存命中率 | {attached.cache_hit_rate * 100:.1f}% |")
        md.append(f"| 计划执行率 | {attached.applied_rate * 100:.1f}% |")
        md.append(f"| 峰值显存 | {attached.peak_vram_mb:.0f} MB |")
        md.append(f"| 显存节省 | {attached.memory_savings_pct:.1f}% |")
        md.append("")
        if attached.pressure_distribution:
            md.append("四象压力分布: " + ", ".join(
                f"{k}={v}" for k, v in attached.pressure_distribution.items()) )
            md.append("")
        if attached.domain_distribution:
            md.append("八卦域分布: " + ", ".join(
                f"{k}={v}" for k, v in attached.domain_distribution.items()))
            md.append("")

    # 三、提升度
    if baseline and attached:
        md.append("## 三、提升度（核心指标）")
        md.append("")
        md.append("| 维度 | 数值 |")
        md.append("|------|------|")
        md.append(f"| 显存峰值节省 | {attached.memory_savings_pct:+.1f}% |")
        md.append(f"| 缓存命中率 | {attached.cache_hit_rate * 100:.1f}% |")
        md.append(f"| 计划执行率 | {attached.applied_rate * 100:.1f}% |")
        md.append("")

    # 四、能力矩阵
    if competitive and competitive.matrix:
        md.append("## 四、竞品能力矩阵")
        md.append("")
        header = "| 方案 | " + " | ".join(CAPABILITY_DIMS) + " | 实测 |"
        sep = "|" + "---|" * (len(CAPABILITY_DIMS) + 2)
        md.append(header)
        md.append(sep)
        for row in competitive.matrix:
            caps = row.get("capabilities", {})
            cells = " | ".join("Y" if caps.get(d) else "-" for d in CAPABILITY_DIMS)
            md.append(f"| {row['name']} | {cells} | {'是' if row.get('measured') else '否'} |")
        md.append("")
        md.append("> 能力为定性判断；竞品量化性能未在本机实测，故不列数字。")
        md.append("")

    md.append("---")
    md.append("")
    md.append("*本报告由 TDS 性能证据收集框架自动生成。*")
    md.append("")

    md_content = "\n".join(md)
    md_path = os.path.join(output_dir, f"performance_evidence_{ts_short}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    json_data = {
        "timestamp": timestamp,
        "title": title,
        "baseline": _to_dict(baseline),
        "attached": _to_dict(attached),
        "competitive": _to_dict(competitive),
    }
    payload = json.dumps(json_data, ensure_ascii=False, indent=2, default=str)
    sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    json_data["sha256"] = sha
    payload_final = json.dumps(json_data, ensure_ascii=False, indent=2, default=str)

    json_path = os.path.join(output_dir, f"performance_evidence_{ts_short}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(payload_final)

    parts = [f"报告: {md_path}", f"sha256={sha[:16]}"]
    if attached:
        parts.append(f"命中率 {attached.cache_hit_rate * 100:.1f}%")
        parts.append(f"显存节省 {attached.memory_savings_pct:.1f}%")
    summary = " | ".join(parts)
    print(summary)

    return ReportResult(
        timestamp=timestamp, title=title,
        md_path=md_path, json_path=json_path,
        sha256=sha, summary=summary,
    )


def _to_dict(obj) -> Optional[Dict[str, Any]]:
    if obj is None:
        return None
    try:
        return asdict(obj)
    except Exception:
        return None
