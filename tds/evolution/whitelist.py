"""软件总体进化的参数白名单与风险分级。

职责
----
1. 定义可进化参数的准入清单（ParamSpec）与风险等级；
2. 提供范围校验（min/max）与变更幅度风险判定（classify_risk）；
3. 提供"参数读取路径"约定：path 为元组，如 ("BAGUA_KEEP_STEPS", "duihua")，
   由 ParameterStore 解析读写——引擎不关心值存在哪里，只通过白名单+store 操作。

风险分级语义
------------
- LOW   ：微调类参数（各域 keep_steps ±1、预读窗口 ±1、阴阳权重 ±0.05 内）——
          可自动应用（应用前备份+changelog）。
- MEDIUM：全局压力阈值 / 衰减因子等影响面较大的参数——只输出建议书，绝不自动应用。
- HIGH  ：冻结阈值等安全底线参数——只输出建议书，且需人工审阅。

变更幅度防线：即使参数本身 risk=LOW，若本次变更幅度超出白名单设定，
classify_risk 会将其提升到 MEDIUM/HIGH，禁止自动应用。
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# 八卦八域：与 tds/config.py 保持一致（零依赖，不 import 上游配置，
# 便于引擎独立演进；领域清单由 IP 进化线的档案维护）
BAGUA_DOMAINS = ["qian", "duihua", "jiansuo", "daima",
                 "tuxiang", "shipin", "gongju", "qita"]

Path = Tuple[str, ...]


class Risk(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    def __str__(self) -> str:  # pragma: no cover - 纯展示
        return self.value


@dataclass(frozen=True)
class ParamSpec:
    """一条白名单参数定义。"""
    path: Path               # 参数读取路径，如 ("BAGUA_KEEP_STEPS", "duihua")
    kind: str                # "int" | "float"
    risk: Risk = Risk.LOW
    min_value: float = 0.0
    max_value: float = 10.0
    step: float = 1.0        # 单次建议的默认步长
    desc: str = ""           # 人类可读说明

    def clamp(self, value: float) -> float:
        v = min(max(value, self.min_value), self.max_value)
        if self.kind == "int":
            return int(round(v))
        return round(float(v), 4)


def _keep_steps(domain: str, risk: Risk = Risk.LOW) -> ParamSpec:
    return ParamSpec(
        path=("BAGUA_KEEP_STEPS", domain),
        kind="int",
        risk=risk,
        min_value=1,
        max_value=64,
        step=1,
        desc=f"八卦域 {domain} 的 keep_steps（下次使用距离阈值）",
    )


# ---- 默认白名单（软件总体进化线专用；IP 线不在此列） ----
DEFAULT_WHITELIST: List[ParamSpec] = [
    # 八域 keep_steps：域内微调，low risk
    *[_keep_steps(d) for d in BAGUA_DOMAINS],
    # 阴阳策略参数
    ParamSpec(("yang_prefetch_advance_steps",), "int", Risk.LOW, 1, 20, 1,
              "阳策略预读提前步数"),
    ParamSpec(("yin_fast_response_timeout_steps",), "int", Risk.LOW, 1, 20, 1,
              "阴策略快速响应超时步数"),
    ParamSpec(("max_prefetch_timeout_steps",), "int", Risk.LOW, 1, 30, 1,
              "阳策略预读最大超时步数"),
    ParamSpec(("dialogue_yang_weight",), "float", Risk.LOW, 0.5, 0.99, 0.05,
              "对话域阳策略权重"),
    ParamSpec(("non_dialogue_yin_weight",), "float", Risk.LOW, 0.5, 0.99, 0.05,
              "非对话域阴策略权重"),
    # 全局压力阈值：影响面大，medium
    ParamSpec(("pressure_ratio",), "float", Risk.MEDIUM, 0.5, 0.99, 0.01,
              "压力触发阈值（>= 即进入四象·高压力）"),
    ParamSpec(("emergency_pressure_ratio",), "float", Risk.MEDIUM, 0.9, 0.99, 0.01,
              "极端压力下切换阴策略的压力阈值"),
    ParamSpec(("switching_decay_factor",), "float", Risk.MEDIUM, 0.05, 0.9, 0.05,
              "阴阳切换衰减因子"),
    # 冻结阈值：安全底线，high
    ParamSpec(("frozen_ratio",), "float", Risk.HIGH, 0.9, 0.999, 0.01,
              "冻结阈值（>= 即进入四象·冻结）"),
]


class Whitelist:
    """白名单容器：按 path 索引，支持查询与风险判定。"""

    def __init__(self, specs: Optional[List[ParamSpec]] = None):
        self._by_path: Dict[Path, ParamSpec] = {}
        for spec in specs or DEFAULT_WHITELIST:
            self._by_path[spec.path] = spec

    def get(self, path: Path) -> Optional[ParamSpec]:
        return self._by_path.get(path)

    def contains(self, path: Path) -> bool:
        return path in self._by_path

    def specs(self) -> List[ParamSpec]:
        return list(self._by_path.values())

    def paths(self) -> List[Path]:
        return list(self._by_path.keys())


def classify_risk(spec: ParamSpec, old_value: float, new_value: float) -> Risk:
    """按变更幅度动态判定风险（叠加在 spec.risk 之上）。

    - int：单步（±1）保持原风险；3 步内升至 MEDIUM；更大升至 HIGH；
    - float：相对变化 <=5% 保持原风险；<=15% 至少 MEDIUM；更大至少 HIGH。
    """
    base = spec.risk
    delta = abs(float(new_value) - float(old_value))
    if spec.kind == "int":
        if delta <= 1:
            return base
        if delta <= 3:
            return Risk.MEDIUM if base.value == "low" else base
        return Risk.HIGH
    rel = delta / (abs(float(old_value)) + 1e-9)
    if rel <= 0.05:
        return base
    if rel <= 0.15:
        return Risk.MEDIUM if base.value == "low" else base
    return Risk.HIGH


def risk_rank(risk: Risk) -> int:
    return {"low": 0, "medium": 1, "high": 2}[risk.value]
