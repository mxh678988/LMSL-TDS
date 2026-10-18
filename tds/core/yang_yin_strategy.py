"""阴阳策略层：两仪（阳=主动预判预读，阴=被动响应快速加载）驱动调度。

与八卦域分类和四象压力系统完全兼容：
  - 八卦域分类决定任务类型（chat/duihua -> 阳，其他 -> 阴）
  - 四象压力决定执行窗口（低压/临界/高压/冻结）
  - 阴阳策略决定调度行为（预读/快速加载/阻塞）

两仪哲学：阳主动预判，阴被动响应；阴阳调和，调度如太极流转。
"""
import enum
from dataclasses import dataclass, field
from typing import Optional, List

from .memory import VramState
from .storage import Block, Tier
from .classifier import DomainProfile


# ---- 阴阳策略枚举 ----
class StrategyMode(enum.Enum):
    """调度策略模式。"""
    YANG = "yang"       # 阳：主动预判，预读优先
    YIN = "yin"         # 阴：被动响应，快速加载


class PressureGuardMode(enum.Enum):
    """压力保护模式。"""
    NORMAL = "normal"   # 正常：按压力状态执行
    EMERGENCY = "emergency"  # 紧急：强制阴策略保活


@dataclass
class YangYinConfig:
    """阴阳策略配置。"""
    # 对话域权重：chat/duihua 领域阳策略比例（0~1，越高越偏阳）
    dialogue_yang_weight: float = 0.85
    # 非对话域权重：其他领域阴策略比例（0~1，越高越偏阴）
    non_dialogue_yin_weight: float = 0.80
    # 预读提前步数（阳策略下预读提前步数）
    yang_prefetch_advance_steps: int = 5
    # 快速响应超时步数（阴策略下响应超时步数）
    yin_fast_response_timeout_steps: int = 3
    # 极端压力下（高压/冻结）切换阴策略的压力阈值
    emergency_pressure_ratio: float = 0.95
    # 阴阳切换衰减因子（极端压力下平滑切换）
    switching_decay_factor: float = 0.3
    # 预读最大超时步数（阳策略预读超时阈值）
    max_prefetch_timeout_steps: int = 10
    # 是否启用阴阳策略层
    enabled: bool = True
    # 是否显式启用紧急模式（高压/冻结下强制阴策略保活；关闭时紧急压力按普通策略处理）
    emergency_mode_enabled: bool = True


# ---- 阴阳策略结果 ----
@dataclass
class StrategyResult:
    """阴阳策略决策结果。"""
    mode: StrategyMode = StrategyMode.YANG       # 当前策略模式
    action: str = "prefetch"                     # 推荐动作
    priority: int = 100                          # 优先级（100 = 最高）
    reason: str = ""                             # 决策理由
    pressure_guard: bool = False                 # 是否需要压力保护
    emergency_mode: bool = False                 # 是否紧急模式

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "action": self.action,
            "priority": self.priority,
            "reason": self.reason,
            "pressure_guard": self.pressure_guard,
            "emergency_mode": self.emergency_mode,
        }


class YangYinStrategy:
    """阴阳策略引擎：根据任务类型、压力状态、八卦域分类决定调度行为。"""

    def __init__(self, config: Optional[YangYinConfig] = None):
        self.cfg = config or YangYinConfig()
        self._step_history: List[int] = []  # 记录最近步数用于平滑切换

    # ---- 核心：阴阳策略决策 ----
    def decide(self, profile: DomainProfile, pressure: str, vram_state: VramState,
               blocks: List[Block] = None, current_step: int = None) -> StrategyResult:
        """根据任务画像、压力状态和块状态，决定阴阳策略。

        决策逻辑：
        1. 对话域（chat/duihua）-> 阳策略（主动预读）
        2. 非对话域（tuxiang/shipin/代码等）-> 阴策略（被动响应）
        3. 极端压力（高压/冻结）-> 强制阴策略保活
        4. 平滑切换：极端压力下阴阳切换衰减
        """
        if not self.cfg.enabled:
            return StrategyResult(
                mode=StrategyMode.YANG,
                action="prefetch",
                priority=100,
                reason="阴阳策略层已禁用",
            )

        # 1. 检查压力状态（紧急模式显式启用时才强制保活）
        is_emergency = pressure in ("high", "frozen") and self.cfg.emergency_mode_enabled
        if is_emergency:
            # 极端压力：强制阴策略保活
            if pressure == "frozen":
                return StrategyResult(
                    mode=StrategyMode.YIN,
                    action="evict",
                    priority=10,
                    reason="冻结状态：强制阴策略保活",
                    pressure_guard=True,
                    emergency_mode=True,
                )
            else:
                return StrategyResult(
                    mode=StrategyMode.YIN,
                    action="skip",
                    priority=50,
                    reason="高压状态：强制阴策略保活",
                    pressure_guard=True,
                    emergency_mode=True,
                )

        # 2. 判断任务类型：对话域 vs 其他
        primary = profile.primary
        is_dialogue = primary in ("duihua", "qian")  # 对话域

        # 3. 计算阴阳权重
        if is_dialogue:
            yang_weight = self.cfg.dialogue_yang_weight
            yin_weight = 1.0 - yang_weight
            mode = StrategyMode.YANG
            reason = f"对话域（{primary}）：阳策略预读，权重={yang_weight:.2f}"
        else:
            yang_weight = 1.0 - self.cfg.non_dialogue_yin_weight
            yin_weight = self.cfg.non_dialogue_yin_weight
            mode = StrategyMode.YIN
            reason = f"非对话域（{primary}）：阴策略快速响应，权重={yin_weight:.2f}"

        # 4. 检查预读窗口（阳策略下）
        if mode == StrategyMode.YANG and blocks:
            # 检查是否有块在预读窗口内
            for block in blocks:
                if block.domain == primary or block.domain in ("qian", "duihua"):
                    distance = current_step - block.last_used_step if current_step else 0
                    if distance <= self.cfg.yang_prefetch_advance_steps:
                        return StrategyResult(
                            mode=StrategyMode.YANG,
                            action="prefetch",
                            priority=100,
                            reason=f"块在预读窗口内（距离={distance}步）",
                        )

        # 5. 决策最终动作
        if mode == StrategyMode.YANG:
            action = "prefetch"
            priority = 100
        else:
            # 阴策略：快速响应
            if pressure == "low":
                action = "pin"  # 快速加载
                priority = 80
            else:  # critical
                action = "pin"  # 快速加载
                priority = 60

        # 6. 记录步数用于平滑切换
        if current_step is not None:
            self._step_history.append(current_step)
            if len(self._step_history) > 10:
                self._step_history.pop(0)

        return StrategyResult(
            mode=mode,
            action=action,
            priority=priority,
            reason=reason,
        )

    # ---- 计划增强：阴阳策略驱动 ----
    def enhance_plan(self, plan_items, profile: DomainProfile, pressure: str,
                     vram_state: VramState = None, current_step: int = None,
                     blocks: List[Block] = None) -> List[dict]:
        """基于阴阳策略增强计划项。

        增强逻辑：
        - 阳策略：增加预读项，提升优先级
        - 阴策略：保留快速加载，降低预读
        """
        strategy = self.decide(profile, pressure, vram_state, blocks, current_step)
        enhanced = []

        for item in plan_items:
            action = item.action
            block = item.block  # 假设 item 是 dataclass 或 dict 有 block 属性

            if strategy.mode == StrategyMode.YANG:
                # 阳策略：对对话域块提升预读优先级
                if block and block.domain in ("qian", "duihua"):
                    if action == "prefetch":
                        # 增加预读提前量
                        pass
                    elif action == "pin":
                        action = "prefetch"  # 提升为预读
                elif block and block.domain not in ("qian", "duihua"):
                    # 非对话域：降为快速加载
                    if action == "prefetch":
                        action = "pin"
                enhanced.append({
                    "block_id": item.block_id,
                    "action": action,
                    "priority": strategy.priority,
                    "reason": strategy.reason,
                })
            else:
                # 阴策略：保持快速加载，减少预读
                if action == "prefetch":
                    action = "pin"  # 改为快速加载
                enhanced.append({
                    "block_id": item.block_id,
                    "action": action,
                    "priority": strategy.priority,
                    "reason": strategy.reason,
                })

        return enhanced

    # ---- 执行适配：阴阳策略驱动 ----
    def adapt_execute(self, strategy: StrategyResult, plan_item: dict) -> dict:
        """基于阴阳策略调整执行动作。"""
        if strategy.emergency_mode:
            # 紧急模式：强制阴策略
            if plan_item.get("action") in ("prefetch", "pin"):
                return {
                    "block_id": plan_item["block_id"],
                    "action": "skip",  # 跳过所有上载
                    "reason": "紧急模式：强制阴策略保活",
                }

        if strategy.mode == StrategyMode.YANG:
            # 阳策略：保持预读
            return plan_item
        else:
            # 阴策略：快速加载
            if plan_item.get("action") == "prefetch":
                return {
                    "block_id": plan_item["block_id"],
                    "action": "pin",
                    "reason": "阴策略：快速加载替代预读",
                }
            return plan_item

    # ---- 极端压力下平滑切换 ----
    def smooth_transition(self, current_pressure: str, target_pressure: str,
                          current_strategy: StrategyResult) -> StrategyResult:
        """极端压力下阴阳策略平滑切换。

        当压力从正常（low/critical）跃升到紧急（high/frozen）时，
        使用衰减因子平滑切换，避免调度震荡。
        """
        if not self.cfg.enabled or not self.cfg.emergency_mode_enabled:
            return current_strategy

        is_normal_to_emergency = (current_pressure in ("low", "critical") and
                                  target_pressure in ("high", "frozen"))

        if not is_normal_to_emergency:
            return current_strategy

        # 平滑切换：衰减因子控制切换速度
        decay = self.cfg.switching_decay_factor
        if current_strategy.mode == StrategyMode.YANG:
            # 阳->阴：逐步切换
            new_mode = StrategyMode.YIN
            new_action = "skip"
            new_priority = int(current_strategy.priority * decay)
            new_reason = f"压力跃升：阳->阴平滑切换（衰减={decay:.2f}）"
        else:
            new_mode = StrategyMode.YIN
            new_action = current_strategy.action
            new_priority = current_strategy.priority
            new_reason = "压力跃升：阴策略保活"

        return StrategyResult(
            mode=new_mode,
            action=new_action,
            priority=new_priority,
            reason=new_reason,
            pressure_guard=True,
            emergency_mode=True,
        )

    # ---- 预读优化：对话域预读窗口计算 ----
    def calculate_prefetch_window(self, block: Block, profile: DomainProfile,
                                  current_step: int) -> Optional[int]:
        """计算对话域块的预读窗口。

        返回预读提前步数，0 表示无需预读。
        """
        if block.domain not in ("qian", "duihua"):
            return None

        distance = current_step - block.last_used_step
        advance_steps = self.cfg.yang_prefetch_advance_steps

        if distance <= advance_steps:
            return advance_steps - distance  # 剩余预读窗口
        return 0  # 无需预读

    # ---- 快速响应优化：非对话域快速加载计算 ----
    def calculate_fast_response_window(self, block: Block,
                                       current_step: int) -> Optional[int]:
        """计算非对话域块的快速响应窗口。"""
        if block.domain in ("qian", "duihua"):
            return None

        timeout_steps = self.cfg.yin_fast_response_timeout_steps
        distance = current_step - block.last_used_step

        if distance <= timeout_steps:
            return timeout_steps - distance
        return 0


# ---- 策略注册与配置 ----
def register_strategy(config: Optional[YangYinConfig] = None):
    """注册阴阳策略到调度器配置。"""
    from ..config import TDSConfig

    cfg = config or YangYinConfig()
    # 可以在这里配置全局策略
    return cfg
