"""调度器：太极层主控。

串联 分类 -> 计划 -> 执行，并维护 阴阳（冷热）标记。
所有平台相关动作通过 adapter 下发，core 保持平台无关。
八卦生成规则：道生一（Scheduler）→ 一生二（阴阳分层）→ 二生三（三才存储）→
  四象（四档压力策略）→ 八卦（八域语义分类）。

规则本身很简单（几行代码），但组合起来可以覆盖 current_loaded_models 中每个模型实例
在每种状态下的调度决策——正是「易有太极，是生两仪，两仪生四象，四象生八卦」的调度哲学。
"""
import time
from typing import Optional, List

from ..config import TDSConfig, YANG, YIN, BAGUA_KEEP_STEPS
from .classifier import BaguaClassifier
from .planner import Planner
from .storage import TieredStore, Block, Tier
from .memory import MemoryManager
from .metrics import Metrics
from .trace import DecisionTrace, TraceEntry
from .yang_yin_strategy import YangYinStrategy, YangYinConfig, StrategyMode, StrategyResult


class PressureState:
    """四象：压力状态。"""
    LOW = "low"        # 低压力：可预取
    CRITICAL = "critical"  # 临界：保活不预取
    HIGH = "high"      # 高压力：跳过调度，交由框架自身
    FROZEN = "frozen"  # 冻结：全部降级到最慢层

    ALL = [LOW, CRITICAL, HIGH, FROZEN]


class Scheduler:
    """太极调度主控。"""

    def __init__(self, config: Optional[TDSConfig] = None, adapter=None):
        self.cfg = config or TDSConfig()
        self.adapter = adapter
        self.classifier = BaguaClassifier()
        self.planner = Planner(self.cfg)
        self.store = TieredStore()
        self.memory = MemoryManager(self.cfg)
        self.metrics = Metrics()
        self.trace = DecisionTrace(getattr(self.cfg, "trace_maxlen", 200))
        self.step = 0
        # 两仪：阴阳策略层（默认配置，可经 YangYinConfig 定制）
        self.yin_yang_strategy = YangYinStrategy()

    # ---- 八卦加权动态 keep_steps ----
    def keep_steps_for_domain(self, domain: str) -> int:
        """根据域八卦权重获取动态 keep_steps。"""
        return BAGUA_KEEP_STEPS.get(domain, BAGUA_KEEP_STEPS["qita"])

    # ---- 阴阳标记 ----
    def yin_yang_tag(self, block: Block) -> str:
        """阳：近期将使用；阴：低频/历史。

        使用八卦加权：不同域有不同的 keep_steps 基准。
        """
        distance = self.step - block.last_used_step
        ks = self.keep_steps_for_domain(block.domain)
        return YANG if distance <= ks else YIN

    # ---- 四象压力策略 ----
    def pressure_state(self, state) -> str:
        """根据显存水位返回四象状态。"""
        if not state or state.total_mb == 0:
            return PressureState.LOW  # 无数据时默认宽松
        ratio = state.usage_ratio
        if ratio >= self.cfg.frozen_ratio:
            return PressureState.FROZEN
        if ratio >= self.cfg.pressure_ratio:
            return PressureState.HIGH
        # 临界：接近但未达压力阈值
        if ratio >= 0.85:
            return PressureState.CRITICAL
        return PressureState.LOW

    # ---- 四象行为模板 ----
    def pressure_action(self, pressure_state: str, plan_items: list, block: Optional[Block] = None) -> str:
        """根据压力状态决定调度行为。"""
        if pressure_state == PressureState.LOW:
            return "prefetch"       # 低压力：正常预取
        elif pressure_state == PressureState.CRITICAL:
            return "hold"           # 临界：保活不预取
        elif pressure_state == PressureState.HIGH:
            return "skip"           # 高压力：跳过调度
        elif pressure_state == PressureState.FROZEN:
            return "evict_all"      # 冻结：全部降级
        return "prefetch"

    # ---- 任务入口 ----
    def on_task(self, task_desc: str, graph=None):
        """语义感知：分类任务 -> 产出画像（并生成初步计划）。

        八卦生成：道生一 → 分类器产出八域画像 → 动态阈值 → 四象策略 → 生成计划
        """
        self.step += 1
        profile = self.classifier.classify(task_desc)
        self.metrics.record(f"task.{profile.primary}", 1)

        # 记录各域权重
        for dom, score in profile.scores.items():
            self.metrics.record(f"domain.{dom}.score", score)

        return profile

    # ---- 计划生成 ----
    def plan(self, profile, blocks=None, vram_state=None):
        """八卦（域阈值）× 四象（压力）→ 计划 → 阴阳增强。"""
        pressure = self.pressure_state(vram_state) if vram_state is not None else PressureState.LOW
        plan = self.planner.plan(
            profile,
            blocks=blocks,
            pressure=pressure,
            current_step=self.step,
            keep_steps_fn=self.keep_steps_for_domain,
        )
        # 两仪增强：按任务画像与压力决定阴阳模式，调整动作与优先级。
        # 增强保留 Plan 对象结构（仅改写 PlanItem.action/priority）。
        # blocks 为空时仍按画像/压力产出模式（对话域→阳、非对话→阴、高压→强制阴）。
        sr = self.yin_yang_strategy.decide(
            profile=profile,
            pressure=pressure,
            vram_state=vram_state,
            blocks=blocks,
            current_step=self.step,
        )
        self._apply_strategy_to_plan(plan, sr, pressure)
        return plan

    def _apply_strategy_to_plan(self, plan, sr, pressure) -> None:
        """将阴阳策略决策应用到计划项（原地改写，不破坏 Plan 结构）。

        - 阳（YANG）：对话域预读优先，pin 提升优先级
        - 阴（YIN）：快速响应优先，prefetch 降为 pin（避免无谓预读）
        - 紧急（emergency_mode）：极端压力下保活，计划项降级为快速加载
        """
        for item in plan.items:
            if sr.mode == StrategyMode.YANG:
                if item.action == "pin":
                    item.priority = max(item.priority, 0.9)
                elif item.action == "prefetch":
                    item.priority = max(item.priority, 0.6)
            elif sr.mode == StrategyMode.YIN:
                if item.action == "prefetch":
                    # 阴策略：被动响应，预读降为按需加载
                    item.action = "pin"
            if sr.emergency_mode and item.action in ("prefetch", "pin"):
                # 极端压力：保活优先，上载类动作降级为快速加载
                item.priority = min(item.priority, 0.5)

    # ---- 预算腾挪判定（问题3）：计划上载是否超预算 ----
    def _need_lru_room(self, items: List, state) -> bool:
        """非高压时，若计划上载（pin/prefetch 且未驻留 VRAM）的估算总需求
        会使显存水位越过压力阈值，则返回 True 触发 LRU 腾挪。

        state 缺失或无显存数据时保守返回 False（交给适配层/ComfyUI 自身保护）。
        """
        if self.adapter is None or state is None or state.total_mb <= 0:
            return False
        wanted = [
            it for it in items
            if it.action in ("pin", "prefetch")
            and (self.store.get(it.block_id) is None
                 or self.store.get(it.block_id).tier != Tier.VRAM)
        ]
        if not wanted:
            return False
        need = 0.0
        for it in wanted:
            b = self.store.get(it.block_id)
            if b is not None:
                need += max(b.size_mb, 1.0)
        if need <= 0:
            return False
        # 上载后水位预期：used + need；超过压力阈值(total*pressure_ratio)即需腾挪
        return state.used_mb + need > state.total_mb * self.cfg.pressure_ratio

    # ---- 高压腾挪：LRU/驻留窗口存量清理（问题3） ----
    def _evict_lru_for_room(self, wanted: List[str]) -> int:
        """高压/冻结下为需要上载的新块腾挪显存。

        按"最久未用"（last_used_step 升序）从 VRAM 中淘汰非 pinned 块，
        直到累计释放空间 >= 目标（以各块 size_mb 估算；size 未知按 1MB 计）
        或无可淘汰块。返回实际淘汰块数（0 = 无空间可腾）。
        """
        if self.adapter is None or not wanted:
            return 0
        need = 0.0
        for block_id in wanted:
            b = self.store.get(block_id)
            if b is not None:
                need += max(b.size_mb, 1.0)
        if need <= 0:
            return 0

        candidates = [b for b in self.store.blocks_in(Tier.VRAM) if not b.pinned]
        candidates.sort(key=lambda b: (b.last_used_step, b.id))  # 最久未用在前
        freed = 0.0
        evicted = 0
        for b in candidates:
            if freed >= need:
                break
            try:
                ok = self.adapter.unload(b.id)
            except Exception:
                ok = False
            if ok:
                self.store.evict(b.id)
                freed += max(b.size_mb, 1.0)
                evicted += 1
                self.metrics.record("action.lru_evict", 1)
        if evicted:
            self.metrics.record("lru_evict_count", evicted)
            self.metrics.record("lru_evict_mb", freed)
        return evicted

    # ---- 高压水位 LRU 兜底（问题4）：无 demote 候选时不因收缩空转 OOM ----
    def _lru_backstop(self, state) -> int:
        """高压/冻结下按"最久未用"（last_used_step 升序）淘汰 VRAM 中非 pinned 块，
        直到释放量 >= 当前水位 - 压力阈值水位。返回实际淘汰块数（0 = 无需或无可淘汰）。

        与 _evict_lru_for_room 的区别：后者为非高压超预算时的计划上载腾挪，
        本方法为高压下计划释放类动作（demote/evict）执行完毕后，若真实水位仍
        超阈值，对"plan 未覆盖的存量 VRAM 块"做最终水位兜底——保证高压下
        不因"无匹配 demote 候选"而持续 OOM。
        """
        if self.adapter is None or state is None or state.total_mb <= 0:
            return 0
        target_mb = state.total_mb * self.cfg.pressure_ratio
        if state.used_mb <= target_mb:
            return 0
        need = state.used_mb - target_mb
        candidates = [b for b in self.store.blocks_in(Tier.VRAM) if not b.pinned]
        candidates.sort(key=lambda b: (b.last_used_step, b.id))  # 最久未用在前
        freed = 0.0
        evicted = 0
        for b in candidates:
            if freed >= need:
                break
            try:
                ok = self.adapter.unload(b.id)
            except Exception:
                ok = False
            if ok:
                self.store.evict(b.id)
                freed += max(b.size_mb, 1.0)
                evicted += 1
                self.metrics.record("action.lru_backstop", 1)
        if evicted:
            self.metrics.record("lru_backstop_count", evicted)
            self.metrics.record("lru_backstop_mb", freed)
        return evicted

    # ---- 计划执行 ----
    def apply(self, plan) -> bool:
        """按计划驱动适配层执行；受显存压力保护约束 + 块级缓存命中直通。

        四象生成：四象策略决定动作，八卦加权决定阈值。
        命中直通：计划要求 pin 的块若已驻留 VRAM（块级缓存命中），
        直接计数 hit 而不重复调用 adapter.load；demote/evict 的块若
        不在 VRAM，直接计数直通而不调用 adapter.unload。
        """
        if self.adapter is None:
            return False

        state = self.adapter.vram_state() if self.cfg.pressure_guard else None
        pressure = self.pressure_state(state) if state is not None else PressureState.LOW

        items = list(plan.items)

        def _in_vram(block_id: str) -> bool:
            b = self.store.get(block_id)
            return b is not None and b.tier == Tier.VRAM

        need_backstop = False  # 高压/冻结执行后按水位做 LRU 兜底（问题4）
        if pressure in (PressureState.HIGH, PressureState.FROZEN):
            # 高压力/冻结：禁止一切上载（pin/prefetch 全部跳过，避免加剧 OOM），
            # 但必须放行"释放类"动作（下沉/淘汰）——否则显存永远降不下来。
            # 新块上载改由下方"预算腾挪"路径服务（非高压但空间不足时 LRU 清场）。
            self.metrics.record("pressure_skip", 1)
            items = [it for it in items if it.action in ("demote", "evict")]
            # 问题4：高压/冻结分支允许"释放类"动作，但 plan 可能没有 demote/evict
            # 候选（如所有块均已下沉或全部 pinned），此时若水位仍超阈值将空转。
            # 标记执行后按真实水位做 LRU 兜底（见循环后的 _lru_backstop）。
            need_backstop = True
        elif self._need_lru_room(items, state):
            # 非高压但计划上载超预算：先按 LRU 腾挪存量再执行，
            # 问题3 的 LRU/驻留窗口能力在此生效（高压由上面分支整体禁止上载）。
            need_backstop = False
            if pressure != PressureState.LOW:
                self.metrics.record("critical_guard", 1)
            wanted = [it for it in items
                      if it.action in ("pin", "prefetch") and not _in_vram(it.block_id)]
            if wanted:
                self._evict_lru_for_room([it.block_id for it in wanted])

        for item in items:
            block = self.store.get(item.block_id)
            in_vram = block is not None and block.tier == Tier.VRAM
            if item.action == "prefetch":
                if not self.cfg.prefetch_enabled:
                    self.metrics.record("prefetch_disabled", 1)
                    continue
                if pressure != PressureState.LOW:
                    # 预取仅在低压力窗口放行（临界/高压/冻结一律不预取，问题6 评估结论）
                    self.metrics.record("prefetch_skip", 1)
                    continue
                if in_vram:
                    self.metrics.hit()
                    self.metrics.record("action.prefetch_hit", 1)
                else:
                    self.adapter.prefetch(item.block_id)
                    self.metrics.record("action.prefetch", 1)
                    if block is not None:
                        # 物理已预取到目标层（VRAM）：账本直达，防错位（问题2）
                        self.store.set_tier(item.block_id, Tier.VRAM)
            elif item.action == "pin":
                if in_vram:
                    # 块级缓存命中直通：不重复加载
                    self.metrics.hit()
                    self.metrics.record("action.pin_hit", 1)
                else:
                    self.metrics.miss()
                    loaded = self.adapter.load(item.block_id)
                    self.metrics.record("action.pin_load", 1)
                    if loaded and block is not None:
                        # 物理已加载到 VRAM：账本直达，防错位（问题2）
                        self.store.set_tier(item.block_id, Tier.VRAM)
                self.metrics.record("action.pin", 1)
            elif item.action == "demote":
                if in_vram:
                    self.adapter.unload(item.block_id)
                    self.store.demote(item.block_id)
                    self.metrics.record("action.demote", 1)
                else:
                    # 非快层直通：无需卸载
                    self.metrics.record("action.demote_skip", 1)
            elif item.action == "evict":
                if in_vram:
                    self.adapter.unload(item.block_id)
                    self.store.evict(item.block_id)
                    self.metrics.record("action.evict", 1)
                else:
                    self.metrics.record("action.evict_skip", 1)

        # 问题4：高压/冻结分支执行完释放类动作后，若真实水位仍超压力阈值，
        # 对 plan 未覆盖的存量 VRAM 块按 LRU（last_used_step 升序）兜底淘汰，
        # 防止"无 demote/evict 候选"时高压下持续 OOM。
        if need_backstop:
            self._lru_backstop(state)
        return True

    # ---- 块注册 / 触达（扫描 → 调度闭环的关键接线） ----
    def ensure_block(self, block_id: str, domain: str = "qita",
                     size_mb: float = 0.0, pinned: bool = False) -> Block:
        """登记一个块到分层存储（幂等：已存在则保留原 tier 状态）。"""
        b = self.store.get(block_id)
        if b is None:
            b = Block(id=block_id, size_mb=size_mb, domain=domain,
                      tier=Tier.SSD, last_used_step=self.step, pinned=pinned)
            self.store.add(b)
        return b

    def register_blocks_from_scan(self, scan_result) -> List[Block]:
        """把 adapter.scan_usage(graph) 的块描述列表登记进 store，返回 Block 列表。

        scan_result 元素支持两种形态：
          - dict: {"block_id": str, "block_type": str, "node_id": ...}
          - str:  块标识字符串（域取 qita）
        块注册后即 touch（last_used_step=当前 step），供阴阳判定使用。
        """
        if not scan_result:
            return []
        blocks: List[Block] = []
        for item in scan_result:
            if isinstance(item, dict):
                block_id = str(item.get("block_id") or item.get("node_id") or "")
                domain = str(item.get("domain") or item.get("block_type") or "qita")
                size = float(item.get("size_mb") or 0.0)
            else:
                block_id = str(item)
                domain = "qita"
                size = 0.0
            if not block_id:
                continue
            b = self.ensure_block(block_id, domain=domain, size_mb=size)
            self.touch_block(b.id)
            blocks.append(b)
        return blocks

    def touch_block(self, block_id: str) -> Optional[Block]:
        """标记块在本步被触达（刷新 last_used_step，阴阳判定的依据）。"""
        b = self.store.get(block_id)
        if b is not None:
            b.last_used_step = self.step
        return b

    # ---- 全链路入口 ----
    def run_task(self, task_desc: str, blocks=None, graph=None, vram_state=None) -> dict:
        """端到端：语义分类 → 八卦加权计划 → 四象约束执行 → 阴阳增强 → 指标 + 决策轨迹。

        参数
        ----
        task_desc: 任务描述文本（语义分类输入）
        blocks: 可选的 Block 列表（调用方已构造）；与 graph 二选一
        graph: 可选的执行图；优先经 adapter.scan_usage(graph) 扫描为块并登记
        vram_state: 可选显存快照；缺省时取 adapter 实时快照
        """
        t0 = time.perf_counter()
        profile = self.on_task(task_desc)

        # 接线：graph 优先走 scan_usage → 块注册（缺失调用链的补齐点）
        if blocks is None and graph is not None and self.adapter is not None:
            scan_result = self.adapter.scan_usage(graph)
            blocks = self.register_blocks_from_scan(scan_result)
            self.metrics.record("scan.blocks", len(blocks))

        # 压力水位唯一来源：显式入参优先，否则取适配器的实时快照（与 apply 保持一致）
        if vram_state is None and self.adapter is not None:
            vram_state = self.adapter.vram_state()
        pressure = self.pressure_state(vram_state) if vram_state is not None else PressureState.LOW
        plan = self.plan(profile, blocks=blocks, vram_state=vram_state)
        applied = self.apply(plan)

        # 两仪决策（记录用；计划增强已在 plan() 内完成）
        yang_yin_result = self.yin_yang_strategy.decide(
            profile=profile,
            pressure=pressure,
            vram_state=vram_state,
            blocks=blocks,
            current_step=self.step,
        )

        result = {
            "step": self.step,
            "primary": profile.primary,
            "scores": dict(profile.scores),
            "pressure": pressure,
            "action": self.pressure_action(pressure, plan.items),
            "plan": plan.summary(),
            "applied": applied,
            "metrics": self.report(),
            "yang_yin_mode": yang_yin_result.mode.value if yang_yin_result else None,
            "yang_yin_reason": yang_yin_result.reason if yang_yin_result else None,
        }

        # 记录决策轨迹（含阴阳信息）
        self.record_decision(
            step=result["step"],
            primary=result["primary"],
            pressure=result["pressure"],
            action=result["action"],
            plan=result["plan"],
            applied=applied,
            scores=result["scores"],
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            yang_yin_mode=result["yang_yin_mode"],
            yang_yin_reason=result["yang_yin_reason"],
        )
        return result

    # ---- 决策轨迹 ----
    def record_decision(self, step=None, primary="", pressure="", action="",
                        plan="", applied=False, scores=None,
                        latency_ms=0.0, note="", yang_yin_mode=None,
                        yang_yin_reason=None) -> TraceEntry:
        """记录一条调度决策（含阴阳信息，供 run_task 及外部自定义链路复用）。"""
        note_parts = [note] if note else []
        if yang_yin_mode:
            note_parts.append(f"mode={yang_yin_mode}")
        if yang_yin_reason:
            note_parts.append(f"reason={yang_yin_reason}")
        note_str = " | ".join(note_parts) if note_parts else ""
        return self.trace.record(
            step=self.step if step is None else step,
            primary=primary,
            pressure=pressure,
            action=action,
            plan=plan,
            applied=applied,
            scores=scores,
            latency_ms=latency_ms,
            note=note_str,
        )

    def trace_entries(self, limit: Optional[int] = None) -> List[TraceEntry]:
        return self.trace.entries(limit)

    def trace_report(self) -> dict:
        """轨迹聚合摘要（含四象 / 动作 / 域分布与平均耗时）。"""
        return self.trace.summary()

    def trace_lines(self, limit: Optional[int] = None) -> List[str]:
        return self.trace.lines(limit)

    def report(self) -> dict:
        return self.metrics.report()
