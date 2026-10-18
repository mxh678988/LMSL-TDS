"""加载决策：可用预算 → 显存画像 → 可行性判定 / 候选排序。

这是 registry 与调度层（tds.core）之间的接缝：
调度层关心的是"给定预算，这个模型装不装得下、装哪一档"，
本模块把硬件实测值与配置预算合并为保守可用预算，再对条目逐一出结论。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from . import catalog as _catalog
from .estimate import Footprint, Verdict, estimate_footprint, judge
from .hardware import HardwareProfile, probe_hardware

#: 无配置时的兜底预算（MB）——沿用引擎默认（12G 卡 - 预留）
FALLBACK_BUDGET_MB = 10788
FALLBACK_RESERVE_MB = 1500


def budget_mb(
    config: Optional[Any] = None,
    hw: Optional[HardwareProfile] = None,
    use_live: bool = True,
) -> float:
    """保守可用预算 = min(配置预算, 实测空闲) - 预留。

    * 配置预算来自 ``TDSConfig.vram_budget_mb``；
    * 实测空闲来自 ``nvidia-smi``，取不到时忽略该项（只用配置预算）；
    * 预留来自 ``TDSConfig.reserve_mb``，用于覆盖桌面/驱动常驻与抖动。
    """
    if config is not None:
        cfg_budget = float(getattr(config, "vram_budget_mb", FALLBACK_BUDGET_MB))
        reserve = float(getattr(config, "reserve_mb", FALLBACK_RESERVE_MB))
    else:
        cfg_budget, reserve = float(FALLBACK_BUDGET_MB), float(FALLBACK_RESERVE_MB)

    candidates = [cfg_budget]
    if use_live:
        profile = hw or probe_hardware(refresh=True)
        if profile.free_vram_mb > 0:
            candidates.append(float(profile.free_vram_mb))
    return max(0.0, min(candidates) - reserve)


def _as_spec(item: Union[str, Any], probe: bool = False, endpoint: Optional[str] = None) -> Any:
    spec = _catalog.get(item) if isinstance(item, str) else item
    if probe:
        spec = _catalog.probe_entry(spec, endpoint=endpoint)
    return spec


def plan(
    item: Union[str, Any],
    config: Optional[Any] = None,
    hw: Optional[HardwareProfile] = None,
    ctx_tokens: Optional[int] = None,
    probe: bool = False,
    endpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """对单个条目给出加载方案。

    Returns
    -------
    dict: ``{key, domain, footprint, verdict, budget_mb, hardware, spec}``
    """
    spec = _as_spec(item, probe=probe, endpoint=endpoint)
    profile = hw or probe_hardware(refresh=True)
    budget = budget_mb(config=config, hw=profile, use_live=True)
    footprint: Footprint = estimate_footprint(spec, ctx_tokens=ctx_tokens)
    verdict: Verdict = judge(footprint, budget)
    return {
        "key": getattr(spec, "key", "?"),
        "domain": getattr(spec, "domain", "other"),
        "weight_basis": footprint.basis,
        "footprint": footprint.to_dict(),
        "verdict": verdict.to_dict(),
        "budget_mb": round(budget, 2),
        "hardware": profile.summary(),
        "spec": spec,
    }


def rank(
    domain: Optional[str] = None,
    config: Optional[Any] = None,
    hw: Optional[HardwareProfile] = None,
    ctx_tokens: Optional[int] = None,
    probe: bool = False,
    endpoint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """对目录（可按 domain 过滤）逐条判定并排序。

    排序键：可行优先 → 占用从小到大（同档内优先选省显存的那一份）。
    """
    profile = hw or probe_hardware(refresh=True)
    budget = budget_mb(config=config, hw=profile, use_live=True)
    rows: List[Dict[str, Any]] = []
    for spec in _catalog.list_entries(domain=domain):
        if probe:
            try:
                spec = _catalog.probe_entry(spec, endpoint=endpoint)
            except Exception as exc:  # 单条失败不阻断整体排名，如实标注
                rows.append(
                    {
                        "key": spec.key,
                        "domain": spec.domain,
                        "verdict": {
                            "level": "unknown", "fits": False, "reason": "远端校验失败：{}".format(exc),
                            "budget_mb": round(budget, 2), "required_mb": 0.0, "headroom_mb": 0.0,
                        },
                        "footprint": {},
                        "spec": spec,
                    }
                )
                continue
        footprint = estimate_footprint(spec, ctx_tokens=ctx_tokens)
        verdict = judge(footprint, budget)
        rows.append(
            {
                "key": spec.key,
                "domain": spec.domain,
                "verdict": verdict.to_dict(),
                "footprint": footprint.to_dict(),
                "spec": spec,
            }
        )
    rows.sort(key=lambda r: (not r["verdict"]["fits"], r["footprint"].get("total_mb", 0.0)))
    return rows


def recommend(
    domain: str,
    config: Optional[Any] = None,
    hw: Optional[HardwareProfile] = None,
    ctx_tokens: Optional[int] = None,
    probe: bool = False,
    endpoint: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """在指定域内挑一条最优可行项；全部不可行时返回 None。"""
    for row in rank(
        domain=domain, config=config, hw=hw, ctx_tokens=ctx_tokens,
        probe=probe, endpoint=endpoint,
    ):
        if row["verdict"]["fits"]:
            return row
    return None
