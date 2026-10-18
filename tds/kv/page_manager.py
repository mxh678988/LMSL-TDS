"""太极调度层 TDS · KV 分页缓存引擎（Phase 2 B 段）。

目标
----
零侵入包装宿主模型类的 init_kv_cache()，将每层 KV cache 从连续大张量替换为
PageCache：内存中只保留活跃页，冷页换出到 RAM。基于八卦加权距离
(keep/gain/prefetch) 驱动换入/换出决策。

架构
----
    init_kv_cache()
        └─ PageCache(key=mem_tensor, value=mem_tensor, index=mem_tensor)
            └─ PageTable {page_id: Page}
                └─ RAMBuffer (LRU 淘汰)

零侵入约束
----------
- PageCache 必须有 .key / .value / .index 属性（与 FixedKV 同接口）
- .key / .value / .index 必须返回 Tensor 或支持 .numel() 的代理
- 所有操作均在 wrapper 内完成，不修改宿主代码
"""
from __future__ import annotations

import gc
import logging
import os
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from ..config import TDSConfig, TDS_PAGE_SIZE_TOKENS, TDS_PAGE_SIZE_BYTES

LOG = logging.getLogger("tds.kv.page")
print("[PROBE-PAGE] module loaded (page_manager)", flush=True)

# --------------------------------------------------------------------------
# 太极页策略（TBP）：独有知识产权，替换西方 LRU 的页级换入换出调用逻辑
# 由 TDS_KV_POLICY 环境变量控制：lru=传统 LRU（默认，可回退），
# taiji=太极·八卦域加权页调度（两仪阴阳 × 四象压力 × 八卦八域）
# --------------------------------------------------------------------------

_KV_POLICY_TAIJI = os.environ.get("TDS_KV_POLICY", "lru").strip().lower() == "taiji"

try:
    from .taiji_policy import (
        TaijiPagePolicy, TaijiPagePolicyConfig, PressureState as TaijiPressure,
        YANG, YIN,
    )
    _HAS_TAIJI = True
except Exception:  # pragma: no cover - 导入失败不影响 LRU 主路径
    _HAS_TAIJI = False

_taiji_policy = TaijiPagePolicy() if (_HAS_TAIJI and _KV_POLICY_TAIJI) else None

# 紫梅层（TBP 二期 IP）：紫微斗数命宫/主辅星 × 梅花易数动态卦象
try:
    from .taiji_zimei import (
        zimei_evict_order, classify_star, wuxing_of,
    )
    _HAS_ZIMEI = True
except Exception:  # pragma: no cover - 导入失败回退纯太极
    _HAS_ZIMEI = False

# --------------------------------------------------------------------------
# 预算联动（Phase2）：CUDA 错误降级保护 + 页空即释放母张量
# --------------------------------------------------------------------------

_BUDGET_FAILED = False          # 预算淘汰路径触发过 CUDA 错误 → 降级无预算模式
_BUDGET_MAX_EVICT = 100000      # 单次预算淘汰最大页数（防 memory_bytes=0 死循环）


def _cuda_safe_sync() -> None:
    """流序同步，把 cudaMallocAsync 的异步错误转成可捕获异常。

    ComfyUI --cuda-malloc 下（PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync），
    显存块的归还/复用是流序的；若此前有未决的非法访问或未完成搬运，
    torch.cuda.synchronize() 会抛 RuntimeError。此处捕获并置 _BUDGET_FAILED，
    让上层降级为无预算模式，避免进程在 C++/驱动层无 traceback 硬崩
    （10/17 实机 256MiB 第 2 轮模型加载 6/12 步消失现场）。
    """
    global _BUDGET_FAILED
    if not (_HAS_TORCH and torch.cuda.is_available()):
        return
    try:
        torch.cuda.synchronize()
    except Exception as exc:  # pragma: no cover - 驱动级错误路径
        LOG.error("[TDS-KV-PHASE2] CUDA 同步失败，预算联动降级为无预算模式: %r", exc)
        _BUDGET_FAILED = True

# --------------------------------------------------------------------------
# 固定 KV 检测（延迟导入，避免在测试阶段硬依赖 comfy）
# --------------------------------------------------------------------------

_FixedKV_class: Optional[type] = None


def _get_FixedKV_class() -> Optional[type]:
    global _FixedKV_class
    if _FixedKV_class is not None:
        return _FixedKV_class
    if _injected_host is not None:                 # 测试注入优先（按 KvHost 字段名约定）
        _FixedKV_class = getattr(_injected_host, "fixed_kv_cls", None)
        return _FixedKV_class
    try:
        from .host import get_host
        h = get_host()
    except Exception:
        h = None
    _FixedKV_class = getattr(h, "fixed_kv_cls", None) if h is not None else None
    return _FixedKV_class


def _is_fixed_kv(layer: Any) -> bool:
    cls = _get_FixedKV_class()
    return cls is not None and isinstance(layer, cls)


# --------------------------------------------------------------------------
# 页表结构
# --------------------------------------------------------------------------

@dataclass
class Page:
    """单个 KV 页，包含 token 偏移和页内状态。"""
    page_id: str                              # 唯一标识
    capacity_tokens: int                      # 页内 token 容量
    dtype_bytes: int                          # 每个 token 的字节数 (2*dim)
    device: str                               # GPU device 标签 (如 "cuda:0")
    offset: int                               # 页在 batch 中的 token 偏移
    layer_id: int                             # 所属层
    head_dim: int = 0                         # 注意力头维度
    num_kv_heads: int = 0                     # KV 头数
    domain: str = "qita"                      # 八卦域标签（太极页策略使用）
    is_in_vram: bool = True                   # 是否驻留显存
    last_used_step: int = 0                   # 最后使用步数
    access_count: int = 0                     # 访问计数 (用于 LRU)
    key_tensor: Optional[torch.Tensor] = None # 页内 key 张量 (batch, cap, heads, dim)
    value_tensor: Optional[torch.Tensor] = None # 页内 value 张量

    @property
    def memory_bytes(self) -> int:
        """计算页内存字节数：2 * head_dim * num_kv_heads * token_count。"""
        if self.head_dim and self.num_kv_heads:
            return 2 * self.head_dim * self.num_kv_heads * self.capacity_tokens
        return self.capacity_tokens * self.dtype_bytes

    def bind(self, key_master: "torch.Tensor", value_master: "torch.Tensor") -> None:
        """把本页绑定为母张量的切片视图（共享内存，非拷贝）。

        attention 内对 KV 的原位 scatter_/copy_ 会直接落到母张量对应区间，
        因此母张量必须与页视图共享存储。
        """
        start = self.offset
        end = self.offset + self.capacity_tokens
        self.key_tensor = key_master[:, start:end]
        self.value_tensor = value_master[:, start:end]
        self.is_in_vram = True

    def unbind_to_cpu(self) -> None:
        """脱离母张量视图，落成独立的 CPU 副本（真实搬运，母张量方可被释放）。

        必须**同步**完成 D2H 拷贝：调用方紧接着释放母张量，其显存块可能被后续
        分配（例如另一层的 allocate_master）立即复用；若拷贝仍在异步队列中，
        副本会读到已被覆写的显存，造成该层 KV 静默丢失（实测仅首层复现）。
        卸载属慢路径，正确性优先于微小延迟。
        """
        if self.key_tensor is not None:
            self.key_tensor = self.key_tensor.detach().to("cpu").clone()
        if self.value_tensor is not None:
            self.value_tensor = self.value_tensor.detach().to("cpu").clone()
        self.is_in_vram = False


@dataclass
class PageTable:
    """页面索引表 + 换入换出缓存。"""
    capacity_tokens: int                      # 总 token 容量
    page_size_tokens: int                     # 页大小 (tokens)
    page_size_bytes: int                      # 页大小 (bytes)
    pages: OrderedDict[str, Page] = field(default_factory=OrderedDict)
    ram_buffer: OrderedDict[str, Page] = field(default_factory=OrderedDict)
    layer: int = -1                           # 所属层
    dtype: str = ""                           # 数据类型
    device: str = ""                          # 设备标签
    batch: int = 0                            # 批量大小
    head_dim: int = 0                         # 注意力头维度
    num_kv_heads: int = 0                     # KV 头数

    # --- 与 FixedKV 对齐的全局状态 ---
    _token_count: int = 0                     # 该层已处理的 token 总数
    position_tensor: Optional[torch.Tensor] = None  # (batch,) int64
    seqlen_tensor: Optional[torch.Tensor] = None    # (batch,) int32

    # --- 母张量：对外暴露给 attention 的完整连续 KV ---
    key_tensor: Optional[torch.Tensor] = None    # (batch, capacity, heads, dim)
    value_tensor: Optional[torch.Tensor] = None  # (batch, capacity, heads, dim)

    @property
    def total_pages(self) -> int:
        return (self.capacity_tokens + self.page_size_tokens - 1) // self.page_size_tokens

    def _page_id(self, idx: int) -> str:
        return f"layer{self.layer}_p{idx}"

    def get_page(self, token_idx: int) -> Optional[Page]:
        """根据 token 索引找到对应页。"""
        page_idx = token_idx // self.page_size_tokens
        page_id = self._page_id(page_idx)
        return self.pages.get(page_id)

    def get_or_create_page(self, token_idx: int, domain_hint: Optional[str] = None) -> Page:
        """获取或创建页（懒分配）。

        domain_hint: 从 prompt 推断的域，优先使用；回退到 _ACTIVE_DOMAIN。
        """
        page_idx = token_idx // self.page_size_tokens
        page_id = self._page_id(page_idx)
        domain = domain_hint or (_ACTIVE_DOMAIN or _TAIJI_LAYER_DOMAIN)
        if page_id not in self.pages:
            offset = page_idx * self.page_size_tokens
            page = Page(
                page_id=page_id,
                capacity_tokens=min(
                    self.page_size_tokens,
                    self.capacity_tokens - offset),
                dtype_bytes=self.page_size_bytes,
                device=self.device,
                offset=offset,
                layer_id=self.layer,
                head_dim=self.head_dim,
                num_kv_heads=self.num_kv_heads,
                domain=domain,
                is_in_vram=True,
            )
            self.pages[page_id] = page
            LOG.debug("[TDS-KV-PAGE] 创建页 %s (偏移=%d, 域=%s)",
                      page_id, offset, domain)
        else:
            self.pages[page_id].domain = domain
        return self.pages[page_id]

    def evict_to_ram(self, page: Page) -> None:
        """将页换出到 RAM 缓冲区（真实张量搬运）。

        页从母张量视图脱离为独立 CPU 副本，母张量对应区间不再被该页引用，
        因此当整层全部换出后母张量可被安全释放，实现真实显存回收。
        """
        if page.page_id in self.pages:
            del self.pages[page.page_id]
        if page.page_id not in self.ram_buffer:
            self.ram_buffer[page.page_id] = page
        page.unbind_to_cpu()

        LOG.info("[TDS-KV-PAGE] 换出页 %s -> RAM (设备=%s)",
                 page.page_id, page.device)

    def prefetch_to_vram(self, page: Page) -> bool:
        """将页换入显存（真实张量搬运）。返回是否成功。

        优先复用 RAM 缓冲区中的同一 Page 对象，以保留其
        last_used_step / access_count 等热度元数据。
        """
        if page.page_id in self.pages:
            page.is_in_vram = True
            return True

        # 命中 RAM 缓冲区：复用原对象，保留热度元数据
        src = self.ram_buffer.pop(page.page_id, None)
        if src is not None:
            # 真实搬运：把 CPU 页数据写回母张量对应切片，并重新绑定为视图
            if self.key_tensor is not None and src.key_tensor is not None:
                start = src.offset
                end = src.offset + src.capacity_tokens
                self.key_tensor[:, start:end].copy_(
                    src.key_tensor.to(self.device, non_blocking=True))
                self.value_tensor[:, start:end].copy_(
                    src.value_tensor.to(self.device, non_blocking=True))
                src.bind(self.key_tensor, self.value_tensor)
            else:
                src.is_in_vram = True
            self.pages[src.page_id] = src
            LOG.debug("[TDS-KV-PAGE] 换入页 %s <- RAM", src.page_id)
            return True

        # 未命中：以调用方传入的页对象重建
        self.pages[page.page_id] = page
        if self.key_tensor is not None and page.key_tensor is not None:
            page.bind(self.key_tensor, self.value_tensor)
        else:
            page.is_in_vram = True
        LOG.debug("[TDS-KV-PAGE] 换入页 %s (RAM 未命中)", page.page_id)
        return True

    def evict_lru_page(self, _ctx: Optional[Dict[str, Any]] = None) -> Optional[Page]:
        """从页面表中淘汰一个页（统一页级淘汰入口）。

        - TDS_KV_POLICY=taiji：走太极页决策（两仪×八卦域序，见 _evict_taiji_page）
        - 否则：传统 LRU（last_used_step 最小，并列 access_count 小者先淘汰）

        _ctx: 批量淘汰上下文缓存（含 order_idx/median），由 evict_pages_to_budget
        预计算后传入，避免同批淘汰内反复重建序字典与重排求中位数；None 时按
        单页淘汰路径自行计算（行为完全一致）。
        """
        if not self.pages:
            return None

        if taiji_enabled() and _taiji_policy is not None:
            victim = self._evict_taiji_page(_ctx)
        else:
            victim = min(
                self.pages.values(),
                key=lambda p: (p.last_used_step, p.access_count))

        self.evict_to_ram(victim)
        return victim

    def _evict_taiji_page(self, _ctx: Optional[Dict[str, Any]] = None) -> Optional[Page]:
        """紫梅页级淘汰：命宫保护 × 主辅星 × 两仪阴阳 × 动态卦象域序 × 热度。

        排序键（值越小越先淘汰）：
            1) 命宫软保护：非命宫域(0) 先于 命宫域(1) —— 当前活跃域页最后动。
               （必须置于首位：decode 后期当前域页也会转阴，若阴阳优先会
                把刚新建的命宫页当作冷页优先淘汰，造成"新建即淘汰"抖振。）
            2) 主辅星：辅星(0) 先于 主星(1) —— 低频页可杀，高频页保活
            3) 两仪：阴页(0) 先于 阳页(1) —— 距离 > 域保活步数者为阴
            4) 动态卦象域序：命宫五行生克驱动的紫梅序（梅花易数动态卦象）
            5) 热度：last_used_step 越久未用先淘汰；并列 access_count 小者先
        软保护保证极端预算下仍有 victim 可选（防卡脖子），
        命宫"永不整层卸载"的硬保护在层间调度 taiji_select_evict_layers 执行。

        _ctx: 批量淘汰上下文（含 order_idx/median），由 evict_pages_to_budget
        预计算传入；None 时按单页淘汰路径自行计算（行为完全一致）。
        """
        step = _current_step
        ming = active_domain()
        if _ctx is not None:
            order_idx, median = _ctx["order_idx"], _ctx["median"]
        else:
            if _HAS_ZIMEI:
                order_idx = {d: i for i, d in enumerate(zimei_evict_order(ming))}
            else:
                order_idx = {d: i for i, d in enumerate(_taiji_policy.cfg.evict_order)}
            # 域内访问计数中位数：主辅星分界（紫微斗数：主星掌吉凶，驻留；辅星佐助，可杀）
            counts = sorted(p.access_count for p in self.pages.values())
            median = float(counts[len(counts) // 2]) if counts else None
        dom_default = len(order_idx)

        def key(p: Page):
            ming_w = 0 if p.domain != ming else 1
            star_w = 0
            if _HAS_ZIMEI and classify_star(p.access_count, median) == "zhu":
                star_w = 1
            distance = step - p.last_used_step if step else 0
            tag = 0 if distance > _taiji_policy.keep_steps_of(p.domain) else 1
            dom_w = order_idx.get(p.domain, dom_default)
            return (ming_w, star_w, tag, dom_w, p.last_used_step, p.access_count)

        return min(self.pages.values(), key=key)

    def _prepare_tensors(self, dtype_name: str, device_str: str) -> None:
        """初始化与 FixedKV 对齐的 position/seqlen 张量。

        真实 FixedKV 中 position 为 int64、seqlen 为 int32，形状均为 (batch,)。
        """
        if self.position_tensor is None:
            self.position_tensor = torch.zeros(self.batch, dtype=torch.int64, device=device_str)
        if self.seqlen_tensor is None:
            self.seqlen_tensor = torch.zeros(self.batch, dtype=torch.int32, device=device_str)

    def _kv_shape(self) -> tuple:
        return (self.batch, self.capacity_tokens, self.num_kv_heads, self.head_dim)

    def allocate_master(self, dtype_name: str) -> None:
        """分配该层完整 KV 母张量（FixedKV 契约形状 (batch, capacity, heads, dim)）。"""
        dtype = _dtype_from_name(dtype_name)
        if self.key_tensor is None:
            self.key_tensor = torch.zeros(self._kv_shape(), dtype=dtype, device=self.device)
        if self.value_tensor is None:
            self.value_tensor = torch.zeros(self._kv_shape(), dtype=dtype, device=self.device)

    def ensure_pages_bound(self) -> None:
        """确保所有页存在并绑定为母张量切片视图。"""
        for i in range(self.total_pages):
            pid = self._page_id(i)
            if pid not in self.pages and pid not in self.ram_buffer:
                self.get_or_create_page(i * self.page_size_tokens)
        for page in self.pages.values():
            page.bind(self.key_tensor, self.value_tensor)

    def restore(self, dtype_name: str) -> None:
        """装载整层：分配母张量 -> 把 RAM 中的页搬回 -> 补齐并绑定剩余页。"""
        self.allocate_master(dtype_name)
        for pid in list(self.ram_buffer.keys()):
            self.prefetch_to_vram(self.ram_buffer[pid])
        self.ensure_pages_bound()

    def _prepare_pages(self, dtype_name: str) -> None:
        """兼容入口：装载整层母张量与页视图。"""
        self.restore(dtype_name)


# --------------------------------------------------------------------------
# PageCache: 包装层（替换 FixedKV）
# --------------------------------------------------------------------------

class PageCache:
    """页缓存包装器，替换 FixedKV。

    必须有 .key / .value / .index 属性以兼容 FixedKV 接口。
    实际数据由 PageTable 管理。
    """
    __slots__ = ("_page_table", "_device", "_dtype", "_layer_id", "_capacity",
                 "_key_tensor", "_value_tensor", "_index_int",
                 "_position_dirty", "_value_dirty")

    def __init__(self, page_table: PageTable):
        self._page_table = page_table
        self._device = page_table.device
        self._dtype = page_table.dtype
        self._layer_id = page_table.layer
        self._capacity = page_table.capacity_tokens
        self._index_int = 0
        self._key_tensor = None
        self._value_tensor = None
        self._position_dirty = True
        self._value_dirty = True

    @property
    def key(self) -> torch.Tensor:
        """返回该层完整连续 KV 母张量（FixedKV 契约）。

        必须是稳定对象且完整：attention 内部会对 key 做原位 scatter_ 并整块
        送入 flash_attention_decode，任何“临时拼接”都会导致写入丢失。
        若母张量已被整层卸载，此处作为安全网自动重新装载。
        """
        pt = self._page_table
        if pt.key_tensor is None:
            pt.restore(self._dtype)
        return pt.key_tensor

    @property
    def value(self) -> torch.Tensor:
        """返回该层完整连续 V 母张量，语义同 key。"""
        pt = self._page_table
        if pt.value_tensor is None:
            pt.restore(self._dtype)
        return pt.value_tensor

    @property
    def index(self) -> int:
        """索引：该层已处理的 token 总数（int，与 FixedKV.index 同类型）。"""
        return self._index_int

    @property
    def position(self) -> torch.Tensor:
        """下一个 token 的位置（与 FixedKV.position 同形状：(batch,) int64）。

        必须返回**持久张量**：真实 attention 会取其 view/expand 作为
        scatter_ 的目标索引，若每次访问返回新建临时张量，位置推进即失效。
        """
        pt = self._page_table
        if pt.position_tensor is None:
            pt._prepare_tensors(self._dtype, self._device)
        return pt.position_tensor

    @property
    def seqlen(self) -> torch.Tensor:
        """总序列长度（与 FixedKV.seqlen 同形状：(batch,) int32），同样必须持久。"""
        pt = self._page_table
        if pt.seqlen_tensor is None:
            pt._prepare_tensors(self._dtype, self._device)
        return pt.seqlen_tensor

    @property
    def device(self) -> str:
        return self._device

    @property
    def capacity_tokens(self) -> int:
        return self._capacity

    @property
    def is_fixed_kv(self) -> bool:
        return False  # PageCache 模式

    def prepare(self, num_tokens: int) -> None:
        """prepare 钩子，严格对齐 FixedKV 语义：

            position.copy_(seqlen); seqlen.add_(num_tokens)

        注意：prepare 不推进 index（index 只由 advance 推进），二者语义不同，
        若在 prepare 里同时加 index 会造成 index 翻倍、KV 写入位置错乱。
        同时作为层进入钩子，预换入本层所需的页（在 CUDA capture 区外完成）。
        """
        print(f"[PROBE-PAGE] prepare called layer={self._layer_id} num_tokens={num_tokens}", flush=True)
        pt = self._page_table
        # position.copy_(seqlen)
        if pt.position_tensor is not None and pt.seqlen_tensor is not None:
            pt.position_tensor.copy_(pt.seqlen_tensor)
        # seqlen.add_(num_tokens)
        if pt.seqlen_tensor is not None:
            pt.seqlen_tensor.add_(num_tokens)

        # 层进入钩子：确保本层母张量与所需页驻留 VRAM
        if pt.key_tensor is None:
            pt.restore(self._dtype)
        self._prefetch_needed_pages(num_tokens)

        # 每层进入即记一次访问（供 LRU 与 Phase3 域序列学习）
        self.record_access()

        # 预算淘汰联动（Phase3 预判联动层调度）：当前层豁免，其余按
        # 策略淘汰至预算内。取代"页级预取"成为预测域页的保护手段——
        # 宿主跨轮丢弃缓存导致无页可预取，预测域页的保护只能靠"不淘汰"。
        self._evict_to_budget()

        self._position_dirty = True
        self._value_dirty = True

    def _evict_to_budget(self) -> None:
        """预算淘汰联动：占用超预算时按策略淘汰，保护当前层与预测域层。

        仅在设置了 TDS_KV_BUDGET_MB 或 TDS_KV_BUDGET_AUTO 时生效；未设置
        预算时保持原有（全驻留 + 页级 LRU 自然淘汰）行为，零回归。

        档位解析（Phase2 必修：固定档改自适应）：
          - TDS_KV_BUDGET_AUTO=<比例>（如 0.35 = 总 KV 全驻留规模的 35%）：
            按当前模型层数/头数/维度估算总 KV 字节后等比缩放，避免
            192/256 固定 MiB 在 7B 文本编码器上压力比失控（10/17 实测）。
          - TDS_KV_BUDGET_MB=<MiB>：固定字节预算（压测 / 兼容旧行为）。

        CUDA 错误降级：淘汰释放路径若触发 cudaMallocAsync 异步错误，
        置 _BUDGET_FAILED 并立即降级为无预算模式，绝不重试。
        """
        if _BUDGET_FAILED:
            return
        budget_mb = os.environ.get("TDS_KV_BUDGET_MB")
        budget_auto = os.environ.get("TDS_KV_BUDGET_AUTO")
        if not budget_mb and not budget_auto:
            return
        try:
            if budget_auto:
                try:
                    frac = float(budget_auto)
                except (TypeError, ValueError):
                    return
                if not (0.0 < frac <= 1.0):
                    return
                budget = int(_kv_total_bytes() * frac)
            else:
                budget = int(float(budget_mb) * 1024 * 1024)
        except (TypeError, ValueError):
            return
        if budget <= 0:
            return
        try:
            stats = evict_pages_to_budget(budget, protect_layer=self._layer_id)
            if stats["evicted_pages"]:
                LOG.info("[TDS-KV-PHASE2] 预算淘汰联动 layer=%d: evicted=%d freed=%d bytes",
                         self._layer_id, stats["evicted_pages"], stats["freed_bytes"])
        except Exception as exc:  # pragma: no cover - CUDA 错误路径
            LOG.error("[TDS-KV-PHASE2] 预算淘汰异常，降级为无预算模式: %r", exc)
            _BUDGET_FAILED = True

    def advance(self, num_tokens: int) -> None:
        """advance 钩子：index += num_tokens（唯一推进 index 的位置）。"""
        self._index_int += num_tokens
        self._page_table._token_count += num_tokens

    @property
    def vram_bytes(self) -> int:
        """该层当前实际驻留的显存字节数（母张量已卸载时为 0）。"""
        pt = self._page_table
        if pt.key_tensor is None:
            return 0
        return pt.key_tensor.numel() * pt.key_tensor.element_size() * 2

    def release_vram(self) -> int:
        """整层卸载：把所有页落成 CPU 副本并释放母张量，返回回收的字节数。

        搬运以层为原子单位——层内任一页被换出后，母张量对该区间已无引用，
        故必须整层释放才能真实回收显存；这是唯一能真正省下显存的粒度。
        必须在 CUDA graph capture 区外调用。
        """
        pt = self._page_table
        freed = self.vram_bytes
        for page in list(pt.pages.values()):
            pt.evict_to_ram(page)
        # 兜底同步：确认所有 D2H 搬运已落地，再归还母张量显存。
        # cudaMallocAsync 下必须流序同步，否则归还的块可能被复用，
        # 与未完成的搬运/计算产生 use-after-free（256MiB 硬崩溃根因）。
        if freed and torch.cuda.is_available():
            _cuda_safe_sync()
        pt.key_tensor = None
        pt.value_tensor = None
        LOG.info("[TDS-KV-PAGE] 整层卸载 layer=%d, 回收 %d bytes", self._layer_id, freed)
        return freed

    def restore_vram(self) -> None:
        """整层装载：重新分配母张量并把 RAM 中的页数据搬回。"""
        self._page_table.restore(self._dtype)

    def _prefetch_needed_pages(self, num_tokens: int) -> None:
        """层进入时预换入本层所需的页（安全网，正常路径在 capture 区外完成）。

        引入 TBP 预读决策：先无条件换入本层即将消费的页（保正确性），
        再对 RAM 中所有距离在保活窗口前沿的阳页提前预取（TBP 工作流感知预调度），
        最后基于访问序列预判，将预测域页提前驻留（Phase 3 访问序列学习）。
        """
        if num_tokens <= 0:
            return
        pt = self._page_table

        start_token = pt._token_count
        end_token = pt._token_count + num_tokens
        start_page = start_token // pt.page_size_tokens
        end_page = (end_token - 1) // pt.page_size_tokens

        for i in range(start_page, end_page + 1):
            pid = pt._page_id(i)
            if pid in pt.ram_buffer:
                pt.prefetch_to_vram(pt.ram_buffer[pid])

        # TBP 预读决策：对距离在保活窗口前沿的页提前驻留
        # 四象压力门控：紧张预算下收窄/禁用预取，避免挤占本层消费页
        if taiji_enabled() and _taiji_policy is not None:
            pressure = _pressure_state_obj()
            for pid, page in list(pt.ram_buffer.items()):
                distance = _current_step - page.last_used_step
                if _taiji_policy.should_prefetch(page.domain, distance, pressure):
                    pt.prefetch_to_vram(page)

        # Phase 3：访问序列预判预取（TDS_KV_PREDICT=off 可关闭，用于对照实验）
        if os.environ.get("TDS_KV_PREDICT", "on").strip().lower() not in ("0", "off", "false", "no", "disabled"):
            from tds.kv.domain_predictor import get_predictor
            predictor = get_predictor()
            last_domain = _ACTIVE_DOMAIN or _TAIJI_LAYER_DOMAIN or "qita"
            predicted, conf = predictor.predict(last_domain)
            if predicted is not None and conf >= 0.60:
                n_prefetch = 0
                for pid, page in list(pt.ram_buffer.items()):
                    if page.domain == predicted:
                        pt.prefetch_to_vram(page)
                        n_prefetch += 1
                LOG.info("[TDS-KV-PHASE3] 预判 %s->%s (conf=%.2f) 预取 %d 页",
                         last_domain, predicted, conf, n_prefetch)
            else:
                LOG.info("[TDS-KV-PHASE3] 预判 %s -> %s (conf=%.2f) 低于阈值或未命中",
                         last_domain, predicted if predicted else "None", conf)

    def record_access(self) -> None:
        """记录访问，用于 LRU 淘汰。"""
        print(f"[PROBE-PAGE] record_access layer={self._layer_id}", flush=True)
        for page in self._page_table.pages.values():
            if page.is_in_vram:
                page.access_count += 1
                page.last_used_step = _current_step
        # Phase 3：记录全局域访问序列（供预判预取使用）
        from tds.kv.domain_predictor import get_predictor
        last_domain = _ACTIVE_DOMAIN or _TAIJI_LAYER_DOMAIN or "qita"
        get_predictor().observe(last_domain)

    def get_memory_bytes(self) -> int:
        """计算当前驻留显存的内存字节数。"""
        total = 0
        for page in self._page_table.pages.values():
            if page.is_in_vram:
                total += page.memory_bytes
        return total

    def __repr__(self) -> str:
        vram_pages = sum(1 for p in self._page_table.pages.values() if p.is_in_vram)
        ram_pages = sum(1 for p in self._page_table.pages.values() if not p.is_in_vram)
        return (f"PageCache(layer={self._layer_id}, vram={vram_pages}+ram={ram_pages},"
                f" capacity={self._capacity} tokens, memory={self.get_memory_bytes()} bytes)")


# --------------------------------------------------------------------------
# 宿主兼容派生：PageCache 必须在运行时继承宿主 FixedKV
# --------------------------------------------------------------------------

_PAGE_CACHE_VARIANTS: Dict[type, type] = {}


def _get_page_cache_cls() -> type:
    """返回与宿主 FixedKV 兼容的 PageCache 类（运行时派生，引擎不 import 宿主）。

    宿主以 ``isinstance(past_key_value, FixedKV)`` 判定分页缓存——attention
    分支与 ``get_past_len()`` 都依赖该判定（见 comfy.text_encoders.llama）。
    若 PageCache 与宿主 FixedKV 无继承关系，真实推理会把分页缓存当作"非分页"
    处理并按下标解包，直接抛 TypeError。故此处按当前宿主动态派生一个同时继承
    PageCache 与宿主 FixedKV 的变体并缓存；无宿主时回退为独立实现。
    """
    base = _get_FixedKV_class()
    if base is None:
        return PageCache
    try:
        if issubclass(PageCache, base):
            return PageCache
        cached = _PAGE_CACHE_VARIANTS.get(base)
        if cached is not None:
            return cached
        variant = type("PageCache", (PageCache, base), {"__slots__": ()})
    except TypeError:
        LOG.warning("[TDS-KV-PAGE] 宿主 %s 无法派生 PageCache 子类，退回独立实现",
                    getattr(base, "__name__", base))
        return PageCache
    _PAGE_CACHE_VARIANTS[base] = variant
    LOG.info("[TDS-KV-PAGE] PageCache 已派生宿主子类: %s",
             getattr(base, "__name__", base))
    return variant


# --------------------------------------------------------------------------
# 全局状态
# --------------------------------------------------------------------------

_KV_PAGE_ENABLED = True
_KV_PAGE_STATE: Dict[str, Any] = {
    "patched": False,
    "model_cls": None,
    "original_init_kv_cache": None,
    "original_class_dict": False,
    "page_tables": [],
    "cache_refs": [],  # 与 page_tables 一一对应的 PageCache 弱引用，用于回收判定
    # 跨轮 RAM 归档（Phase3 推理能力闭环）：
    #   ram_archive: OrderedDict[int, List[Page]]  轮号 -> 该轮被回收页表的 RAM 副本页
    #   archive_round_seq: int                      自增轮号计数器
    #   受 TDS_KV_RAM_KEEP_ROUNDS 控制（默认 0 = 关闭，行为与历史完全一致）
    "ram_archive": OrderedDict(),
    "archive_round_seq": 0,
}

_current_step = 0  # 当前推理步数，用于 LRU 和距离计算

# 懒初始化的域分类器（避免非 tds.kv 路径 import 开销）
_classifier = None


def _get_classifier() -> Any:
    """懒加载域分类器。"""
    global _classifier
    if _classifier is None:
        try:
            from .domain_classifier import DomainClassifier
            _classifier = DomainClassifier()
        except ImportError:
            _classifier = False  # 分类器不可用
    return _classifier


def _classify_domain(prompt: str) -> str:
    """根据 prompt 文本推断域，回退到 qita。"""
    cls = _get_classifier()
    if cls is False:
        return "qita"
    if cls is None:
        return "qita"
    try:
        return cls.classify(prompt)
    except Exception:
        return "qita"


def tick(step: Optional[int] = None) -> int:
    """推进当前推理步数（每 decode 一步调用一次）。

    LRU 的 last_used_step 与太极的距离计算均以 _current_step 为基准；
    若不推进，所有页恒为"刚使用过"（distance=0 → 全阳），淘汰与卸载决策
    会退化为纯 access_count / 全驻留，压测与实机都无区分度。
    显式传 step 用于校准 / 对齐外部步进。
    """
    global _current_step
    _current_step = step if step is not None else _current_step + 1
    return _current_step


def _env_enabled() -> bool:
    """按 TDS_KV 环境变量解析分页开关。"""
    value = os.environ.get("TDS_KV", "observe").strip().lower()
    return value not in ("0", "off", "false", "no", "disable", "disabled")


_KV_PAGE_ENABLED = _env_enabled()


def _dtype_from_name(name: str) -> Any:
    """将 dtype 名称映射到 torch dtype（延迟导入 torch）。"""
    if not _HAS_TORCH:
        LOG.warning("[TDS-KV-PAGE] 无法导入 torch，回退到 float32")
        return torch.float32

    mapping = {
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
        "torch.int64": torch.int64,
        "torch.int32": torch.int32,
        "torch.int16": torch.int16,
        "torch.int8": torch.int8,
        "torch.uint8": torch.uint8,
        "torch.bool": torch.bool,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
        "half": torch.float16,
    }
    return mapping.get(name, torch.float32)


def _page_table_vram_bytes(page_table: PageTable) -> int:
    """页表当前驻留显存字节数（母张量已卸载则为 0）。"""
    if page_table.key_tensor is None:
        return 0
    return page_table.key_tensor.numel() * page_table.key_tensor.element_size() * 2


def _evict_pages(page_table: PageTable, max_vram_bytes: int) -> int:
    """按显存预算卸载层，返回释放的字节数。

    安全约束：attention 每步都要消费整块 KV，单页换出会让母张量出现空洞、
    直接破坏结果。因此换出粒度只能是“整层”——母张量整体释放、页落 RAM。
    必须在 CUDA graph capture 区外调用（通常由 bridge 在层间调度点触发）。
    """
    current = _page_table_vram_bytes(page_table)
    if current <= max_vram_bytes:
        return 0
    # 整层卸载：把所有页落 CPU 并释放母张量
    freed = current
    for page in list(page_table.pages.values()):
        page_table.evict_to_ram(page)
    page_table.key_tensor = None
    page_table.value_tensor = None
    LOG.info("[TDS-KV-PAGE] 预算淘汰：整层卸载 layer=%d, 回收 %d bytes",
             page_table.layer, freed)
    return freed


def _prefetch_pages(page_table: PageTable, dtype_name: str = "") -> int:
    """把 RAM 中的页换回 VRAM 并重建母张量，返回搬回的页数。"""
    if page_table.key_tensor is None and not page_table.ram_buffer:
        return 0
    page_table.restore(dtype_name or page_table.dtype or "float16")
    return len(page_table.pages)


# --------------------------------------------------------------------------
# Patch 逻辑
# --------------------------------------------------------------------------

# 可注入的宿主描述命名空间（测试用；生产环境为 None，改由 host 供给）
# 字段名与 KvHost 契约一致（model_cls / fixed_kv_cls），引擎不识别任何宿主专有类名。
_injected_host: Any = None


def _resolve_model_cls() -> Any:
    """解析宿主中拥有 init_kv_cache 的类；无宿主返回 None。"""
    if _injected_host is not None:
        return getattr(_injected_host, "model_cls", None)
    try:
        from .host import get_host
        h = get_host()
    except Exception:
        h = None
    return getattr(h, "model_cls", None) if h is not None else None


def _on_host_ready() -> None:
    """宿主就绪 / 变更回调：补包装（幂等，仅在本引擎已开闸时生效）。"""
    if _KV_PAGE_ENABLED:
        _ensure_wrapped()


# --------------------------------------------------------------------------
# 页表生命周期回收
# --------------------------------------------------------------------------
# 每次 host.init_kv_cache() 都会为一批层新建 PageTable，并把它们登记进
# _KV_PAGE_STATE["page_tables"] 以便观测 / 报告。若只登记不回收，上一轮推理
# 的母张量会一直被该登记表强引用而无法释放——真实 ComfyUI 每次文生图都会
# 重新 init_kv_cache，显存将随出图次数线性增长直至 OOM。
#
# 因此以弱引用跟踪每层 PageCache：仅当宿主已丢弃该缓存对象（本轮推理确已
# 结束）时，才释放其母张量与 RAM 副本。仍在推理中的缓存对象不会被回收。


def _free_page_table(page_table: PageTable, keep_ram: bool = False,
                     top_keep: int = 0) -> int:
    """释放单层页表的母张量 / 页视图 / RAM 副本，返回回收的显存字节数。

    keep_ram=True：RAM 副本页不销毁、移入跨轮归档（供下一轮同参数页表
    导入复用）；仍释放 VRAM 母张量与页视图，显存回收语义不变。

    top_keep>0（配合 keep_ram）：除被淘汰到 RAM 的页外，再把本层仍驻留
    VRAM 的高频页（按 access_count 降序、同分按 last_used_step 降序）拷为
    CPU 副本并入归档，实现核心上下文跨轮无损；显存仍按原语义回收。
    """
    freed = 0
    if page_table.key_tensor is not None:
        freed += page_table.key_tensor.numel() * page_table.key_tensor.element_size() * 2
    page_table.key_tensor = None
    page_table.value_tensor = None
    if keep_ram:
        if top_keep > 0:
            keep_ids = set(page_table.ram_buffer.keys())
            ranked = sorted(
                (p for p in page_table.pages.values()
                 if p.page_id not in keep_ids),
                key=lambda p: (p.access_count, p.last_used_step),
                reverse=True)[:top_keep]
            for page in ranked:
                page_table.evict_to_ram(page)
        # 归档本轮 RAM 副本：只释放 VRAM 视图，CPU 页数据整体保留
        archived = list(page_table.ram_buffer.values())
        for page in list(page_table.pages.values()):
            page.key_tensor = None
            page.value_tensor = None
        page_table.pages.clear()
        page_table.ram_buffer.clear()
        if archived:
            seq = _KV_PAGE_STATE.get("archive_round_seq", 0)
            _KV_PAGE_STATE.setdefault("ram_archive", OrderedDict())
            bucket = _KV_PAGE_STATE["ram_archive"].setdefault(seq, {})
            key = (page_table.layer, page_table.batch,
                   page_table.capacity_tokens, page_table.device, page_table.dtype)
            bucket.setdefault(key, []).extend(archived)
    else:
        for page in list(page_table.pages.values()) + list(page_table.ram_buffer.values()):
            page.key_tensor = None
            page.value_tensor = None
        page_table.pages.clear()
        page_table.ram_buffer.clear()
    page_table.position_tensor = None
    page_table.seqlen_tensor = None
    page_table._token_count = 0
    return freed


def _reap_stale_tables() -> int:
    """回收宿主已丢弃的页表（弱引用失效者），返回回收的显存字节数。

    保守策略：weakref 不可用或引用仍存活时一律保留，绝不动可能在用中的缓存。

    跨轮 RAM 归档：TDS_KV_RAM_KEEP_ROUNDS>0 时，被回收页表的 RAM 副本
    移入归档并保留最近 N 轮（供下一轮同参数页表导入复用，实现预测域保护
    与上下文复用的跨轮闭环）；默认 0 保持历史行为（RAM 副本随页表销毁）。
    """
    tables = _KV_PAGE_STATE.get("page_tables") or []
    refs = _KV_PAGE_STATE.get("cache_refs")
    if not tables or refs is None:
        return 0

    keep_rounds = 0
    try:
        keep_rounds = int(os.environ.get("TDS_KV_RAM_KEEP_ROUNDS", "0"))
    except (TypeError, ValueError):
        keep_rounds = 0
    keep_ram = keep_rounds > 0
    top_keep = 0
    try:
        top_keep = int(os.environ.get("TDS_KV_RAM_ARCHIVE_TOP_PAGES", "0"))
    except (TypeError, ValueError):
        top_keep = 0
    max_pages = 0
    try:
        max_pages = int(os.environ.get("TDS_KV_RAM_ARCHIVE_MAX_PAGES", "0"))
    except (TypeError, ValueError):
        max_pages = 0

    freed = 0
    kept_tables, kept_refs = [], []
    archived_any = False
    for ref, page_table in zip(refs, tables):
        alive = ref is None or ref() is not None
        if alive:
            kept_tables.append(page_table)
            kept_refs.append(ref)
        else:
            freed += _free_page_table(page_table, keep_ram=keep_ram,
                                      top_keep=top_keep if keep_ram else 0)
            archived_any = True

    if keep_ram and archived_any:
        # 新一轮清理完成：归档轮号 +1，并裁剪超出保留轮数的最老桶
        _KV_PAGE_STATE["archive_round_seq"] = _KV_PAGE_STATE.get("archive_round_seq", 0) + 1
        archive = _KV_PAGE_STATE.setdefault("ram_archive", OrderedDict())
        while len(archive) > keep_rounds:
            archive.popitem(last=False)
        # 稳定性保护：归档页总数超上限时从最老桶逐页裁剪
        if max_pages > 0:
            total = sum(len(pgs) for b in archive.values()
                        for pgs in b.values())
            while total > max_pages and archive:
                _, bucket = next(iter(archive.items()))
                # 先清理空 key，避免空 key 阻塞裁剪进度造成死循环
                for k in [k for k, pgs in bucket.items() if not pgs]:
                    del bucket[k]
                if not bucket:
                    archive.popitem(last=False)
                    continue
                _, pages = next(iter(bucket.items()))
                need = total - max_pages
                for _ in range(min(len(pages), need)):
                    p = pages.pop()
                    if p is not None:
                        p.key_tensor = None
                        p.value_tensor = None
                    total -= 1
                if not pages:
                    del bucket[next(iter(bucket.keys()))]
        # 归档页是 CPU 副本，轮末显存回收不受影响；仅记录归档规模便于观测
        archived_pages = sum(len(pgs) for b in archive.values() for pgs in b.values())
        if archived_pages:
            LOG.info("[TDS-KV-PHASE3] 跨轮 RAM 归档: 保留 %d 轮 %d 页, 裁剪至 %d 轮",
                     _KV_PAGE_STATE["archive_round_seq"], archived_pages, keep_rounds)

    if len(kept_tables) != len(tables):
        _KV_PAGE_STATE["page_tables"] = kept_tables
        _KV_PAGE_STATE["cache_refs"] = kept_refs
        if freed and _HAS_TORCH and torch.cuda.is_available():
            _cuda_safe_sync()
        LOG.info("[TDS-KV-PAGE] 回收废弃页表 %d 个, 释放 %d bytes",
                 len(tables) - len(kept_tables), freed)
    return freed


def _import_archive_pages(page_table: PageTable) -> int:
    """从跨轮归档导入与当前页表同参数的 RAM 副本页（Phase3 跨轮闭环）。

    匹配条件：layer / batch / capacity_tokens / device / dtype 完全一致，
    避免把不同模型或不同容量的轮次数据错接到新页表。导入的页进入
    ram_buffer，随后由 restore() 统一搬回 VRAM；未匹配的归档页滞留
    直至轮次裁剪。返回导入页数。
    """
    archive = _KV_PAGE_STATE.get("ram_archive") or OrderedDict()
    if not archive:
        return 0

    want = (page_table.layer, page_table.batch,
            page_table.device, page_table.dtype)
    imported = 0
    # 从新到旧遍历桶，同一层优先取最近一轮的数据
    for bucket in reversed(list(archive.values())):
        for (layer, batch, cap_tokens, device, dtype), pages in list(bucket.items()):
            if (layer, batch, device, dtype) != want:
                continue
            # 容量漂移兼容：真实模型轮间 capacity 会随输入长度变化，
            # 旧页容量不超过新页表容量即可安全承接（bind/copy 用旧页区间）
            if cap_tokens > page_table.capacity_tokens:
                continue
            for page in pages:
                if page.page_id in page_table.pages or page.page_id in page_table.ram_buffer:
                    continue
                if page.offset + page.capacity_tokens > page_table.capacity_tokens:
                    continue
                page.is_in_vram = False
                page_table.ram_buffer[page.page_id] = page
                imported += 1
        if imported:
            break  # 每层只从最近一轮导入，避免同页多桶重复接错
    if imported:
        LOG.info("[TDS-KV-PHASE3] 跨轮导入 layer=%d: %d 页 (RAM 副本复用)",
                 page_table.layer, imported)
    return imported


def enable() -> None:
    """启用 KV 分页（幂等）。

    互斥声明：本函数与 ``tds.kv.observer.enable()`` 互斥——二者都会包装
    同一个 ``init_kv_cache``，同时启用会产生双重包装。生产路径请统一经
    ``tds.kv.activate()`` 开闸，由该入口保证三态互斥。
    """
    global _KV_PAGE_ENABLED
    _KV_PAGE_ENABLED = True
    try:
        from .host import register_host_hook
        register_host_hook(_on_host_ready)
    except Exception:
        pass
    _ensure_wrapped()


def disable() -> None:
    """禁用 KV 分页（恢复原生行为，幂等）。"""
    global _KV_PAGE_ENABLED
    _KV_PAGE_ENABLED = False
    _ensure_unwrapped()


def is_enabled() -> bool:
    return _KV_PAGE_ENABLED


def _ensure_wrapped() -> None:
    """将 init_kv_cache 包装到宿主模型类上（幂等）。"""
    if _KV_PAGE_STATE["patched"]:
        return
    try:
        model_cls = _resolve_model_cls()
        if model_cls is None:
            LOG.info("[TDS-KV-PAGE] 未绑定宿主，包装待命")
            return
        if "init_kv_cache" in model_cls.__dict__:
            original = model_cls.__dict__["init_kv_cache"]
            own_attr = True
        else:
            original = getattr(model_cls, "init_kv_cache")
            own_attr = False
        _KV_PAGE_STATE["model_cls"] = model_cls
        _KV_PAGE_STATE["original_init_kv_cache"] = original
        _KV_PAGE_STATE["original_class_dict"] = own_attr
        model_cls.init_kv_cache = _make_kv_page_wrapper(original)
        _KV_PAGE_STATE["patched"] = True
        LOG.info("[TDS-KV-PAGE] init_kv_cache 已包装 (own_attr=%s)", own_attr)
    except Exception:
        LOG.exception("[TDS-KV-PAGE] 包装 init_kv_cache 失败（已忽略，不影响宿主行为）")


def _ensure_unwrapped() -> None:
    """恢复宿主模型类的原始 init_kv_cache（幂等）。"""
    if not _KV_PAGE_STATE["patched"]:
        return
    try:
        # 优先使用包装时记录的真实类，避免测试注入模块后类不一致
        model_cls = _KV_PAGE_STATE.get("model_cls") or _resolve_model_cls()
        if model_cls is None:
            _KV_PAGE_STATE["patched"] = False
            return
        orig = _KV_PAGE_STATE.get("original_init_kv_cache")
        if orig is not None:
            if _KV_PAGE_STATE.get("original_class_dict"):
                model_cls.init_kv_cache = orig
            else:
                # 原函数继承自父类，删除覆盖层即可还原
                try:
                    delattr(model_cls, "init_kv_cache")
                except AttributeError:
                    model_cls.init_kv_cache = orig
        _KV_PAGE_STATE["patched"] = False
        _KV_PAGE_STATE["model_cls"] = None
        _KV_PAGE_STATE["original_init_kv_cache"] = None
        LOG.info("[TDS-KV-PAGE] init_kv_cache 已恢复原生")
    except Exception:
        LOG.exception("[TDS-KV-PAGE] 恢复 init_kv_cache 异常（已忽略）")


def _make_kv_page_wrapper(original_fn):
    """为 init_kv_cache 创建分页包装器。

    关键：先执行原始调用，然后用 PageCache 替换返回结果。
    """

    def wrapper(self, batch, capacity, device, dtype):
        # 入口先回收上一轮推理遗留（宿主已丢弃）的页表，避免显存随出图次数累积
        print(f"[PROBE-PAGE] wrapper called batch={batch} capacity={capacity} dtype={dtype}", flush=True)
        _reap_stale_tables()

        # 先执行原始调用
        result = original_fn(self, batch, capacity, device, dtype)

        # 分页禁用时原样透传（不新建任何结构）
        if not _KV_PAGE_ENABLED:
            return result

        # 分支守卫：仅对 FixedKV 分支替换，其他分支必须原样透传
        cls = _get_FixedKV_class()
        if cls is None:
            return result

        layer_list = result if isinstance(result, (list, tuple)) else [result]
        if not layer_list or not isinstance(layer_list[0], cls):
            # 裸三元组（非 fixed_kv）分支：原样返回
            print(f"[PROBE-PAGE] PASS-THROUGH non-fixed_kv: first={type(layer_list[0]).__name__ if layer_list else None} cls={cls.__name__}", flush=True)
            LOG.debug("[TDS-KV-PAGE] 非 fixed_kv 分支，透传原生结果")
            return result

        config = getattr(self, "config", None)
        head_dim = int(getattr(config, "head_dim", 0) or 0)
        num_kv_heads = int(getattr(config, "num_key_value_heads", 0) or 0)
        dtype_str = str(dtype)
        device_str = str(device)
        batch_size = int(batch)

        page_caches = []
        for layer_idx, layer in enumerate(layer_list):
            page_table = PageTable(
                capacity_tokens=capacity,
                page_size_tokens=TDS_PAGE_SIZE_TOKENS,
                page_size_bytes=TDS_PAGE_SIZE_BYTES,
                layer=layer_idx,
                dtype=dtype_str,
                device=device_str,
                batch=batch_size,
                head_dim=head_dim,
                num_kv_heads=num_kv_heads,
            )
            # 分配母张量并绑定页视图（保持与 FixedKV 相同的形状契约）
            page_table._prepare_tensors(dtype_str, device_str)
            # Phase3 跨轮闭环：先导入上一轮同参数页表的 RAM 副本，
            # restore 时统一搬回 VRAM（预测域页跨轮存活的前提）
            _import_archive_pages(page_table)
            page_table.restore(dtype_str)

            page_cache = _get_page_cache_cls()(page_table)
            page_caches.append(page_cache)
            _KV_PAGE_STATE["page_tables"].append(page_table)
            try:
                cache_ref = weakref.ref(page_cache)
            except TypeError:
                # 对象无 __weakref__ 槽时不支持弱引用：退化为 None，按"存活"处理
                cache_ref = None
            _KV_PAGE_STATE.setdefault("cache_refs", []).append(cache_ref)

            LOG.info(
                "[TDS-KV-PAGE] 层 %d: 创建 PageCache (capacity=%d tokens, pages=%d, dtype=%s, device=%s)",
                layer_idx, capacity, page_table.total_pages, dtype_str, device_str)

        LOG.info(
            "[TDS-KV-PAGE] init_kv_cache: %d 层, batch=%s, capacity=%s, dtype=%s, device=%s",
            len(page_caches), batch, capacity, dtype_str, device_str)

        return page_caches if isinstance(result, (list, tuple)) else page_caches[0]

    return wrapper


# --------------------------------------------------------------------------
# 报告接口
# --------------------------------------------------------------------------

def get_page_tables() -> List[PageTable]:
    """获取所有页面表。"""
    return _KV_PAGE_STATE.get("page_tables", [])


def get_summary_text() -> str:
    """获取分页摘要文本。"""
    if not _KV_PAGE_STATE.get("page_tables"):
        return "无分页数据"

    lines = ["KV 分页缓存状态", "=" * 50]
    for pt in _KV_PAGE_STATE["page_tables"]:
        vram_pages = sum(1 for p in pt.pages.values() if p.is_in_vram)
        ram_pages = sum(1 for p in pt.pages.values() if not p.is_in_vram)
        total_bytes = sum(p.memory_bytes for p in pt.pages.values() if p.is_in_vram)
        lines.append(
            f"层 {pt.layer}: vram={vram_pages}+ram={ram_pages}, "
            f"total={total_bytes} bytes, capacity={pt.capacity_tokens} tokens")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# 太极层间调度（TBP）：两仪阴阳 × 四象压力 → 整层换入换出顺序
# --------------------------------------------------------------------------
# KV 天然属"对话·多轮上下文"（duihua 域）：它是上下文之气的载体，先天地
# 归于坎水（duihua）。太极层调度因此以 duihua 域保活窗口为"两仪"基准，
# 以四象压力定淘汰深度，替代顺序遍历的无差别整层卸载。

_TAIJI_LAYER_DOMAIN = "duihua"  # KV 先天八卦域：对话·多轮上下文

_ACTIVE_DOMAIN: Optional[str] = None  # 当前活跃域（多域负载由外部按轮注入）


def set_active_domain(domain: Optional[str]) -> None:
    """设置当前活跃八卦域（供多域负载 / 调度器按任务域注入）。

    置 None / 空串 时回退到 _TAIJI_LAYER_DOMAIN（KV 先天域）。
    仅影响此后新建页的 domain 标签；已有页标签保持不变。
    """
    global _ACTIVE_DOMAIN
    _ACTIVE_DOMAIN = (domain or "").strip() or None


def active_domain() -> str:
    """返回当前活跃域（缺省回退先天域，保证单域场景行为不变）。"""
    return _ACTIVE_DOMAIN or _TAIJI_LAYER_DOMAIN


def taiji_enabled() -> bool:
    """太极页策略是否已启用。"""
    return _KV_POLICY_TAIJI and _taiji_policy is not None


def _layer_tag(layer_id: int, distance: int) -> str:
    """两仪判定：层距离 <= duihua 保活窗口 → 阳（保活），否则 → 阴（可卸）。"""
    if not taiji_enabled():
        return YANG  # 未启用：一律视作阳，等价全驻留
    ks = _taiji_policy.keep_steps_of(_TAIJI_LAYER_DOMAIN)
    return YANG if distance <= ks else YIN


def _layer_domain(t: "PageTable") -> Optional[str]:
    """返回层的八卦域标签：取层内任意页的 domain（层建页时按活跃域注入）。"""
    for p in (t.pages.values() if hasattr(t, "pages") else []):
        return p.domain
    return None


def _is_ming_layer(t: "PageTable", ming: str) -> bool:
    """命宫层判定：层内页域 == 当前活跃域（命宫）→ 永不整层卸载。"""
    d = _layer_domain(t)
    return d is not None and d == ming


def taiji_select_evict_layers(pressure: str, current_step: Optional[int] = None,
                              budget_bytes: Optional[int] = None) -> List[Tuple["PageTable", str]]:
    """紫梅层间调度：返回应按顺序整层卸载的 (page_table, reason) 列表。

    - 低压/未启用：空列表（全驻留，零开销）
    - 临界：仅阴层可卸载（阳层保活）
    - 高压：所有阴层卸载，阳层窗口收缩（由策略配置衰减）
    - 冻结：全部卸载（天VRAM→地RAM）
    排序：先按四象动作，再按两仪（阴先于阳），层号升序稳定。
    紫梅硬保护：命宫层（当前活跃域层）永不进入卸载候选（防错杀当前域）。
    候选直接携带 PageTable 对象：多轮共享页池下 layer 号每轮重复，
    若只带 layer_id 会让执行端按号匹配到"最新一轮"的层，造成整层误卸。
    """
    tables = list(_KV_PAGE_STATE.get("page_tables") or [])
    if not tables or not taiji_enabled():
        return []
    if pressure == "low":
        return []

    ming = active_domain()
    step = _current_step if current_step is None else current_step
    from .taiji_policy import PressureState as _PS

    if pressure == "frozen":
        # 冻结：仍保活窗口内阳层，只卸窗口外阴层（防止 decode 每步
        # 卸载→重建全层抖振；真正的"全层清场"交由显式 unload_all 场景）。
        # 紫梅：命宫层（当前活跃域）永不清场。
        ks = _taiji_policy.keep_steps_of(_TAIJI_LAYER_DOMAIN)
        eff_ks = max(1, int(ks * _taiji_policy.cfg.high_keep_decay))
        frozen_candidates = []
        if step:
            frozen_candidates = [
                (t, "老阳·冻结：窗口外阴层整层卸载（天VRAM→地RAM）")
                for t in tables
                if step - t._token_count > eff_ks and not _is_ming_layer(t, ming)
            ]
        return frozen_candidates

    candidates: List[Tuple["PageTable", str, str]] = []  # (table, tag, reason)
    for t in tables:
        # 紫梅命宫硬保护：当前活跃域层永不整层卸载（防错杀当前域页）
        if _is_ming_layer(t, ming):
            continue
        distance = step - t._token_count if step else 0
        tag = _layer_tag(t.layer, distance)
        if tag == YIN:
            if pressure == "critical":
                candidates.append((t, YIN, "少阳·临界：阴层换出（duihua窗口外）"))
            else:  # high
                candidates.append((t, YIN, "老阴·高压：阴层域序卸载"))
        elif pressure == "high":
            # 高压：阳层窗口收缩（衰减因子），窗口外也进入候选
            ks = _taiji_policy.keep_steps_of(_TAIJI_LAYER_DOMAIN)
            eff_ks = max(1, int(ks * _taiji_policy.cfg.high_keep_decay))
            if distance > eff_ks:
                candidates.append((t, YIN, f"老阴·高压：阳层窗口收缩至{eff_ks}步后转阴"))

    # 阴先于阳（实际上述高压只有阴候选），层号稳定升序
    candidates.sort(key=lambda x: (0 if x[1] == YIN else 1, x[0].layer))
    return [(table, reason) for table, _tag, reason in candidates]


def taiji_plan_report() -> str:
    """太极层间调度决策报告（独有 IP 可观测性：每层阴阳标签与理由）。"""
    if not taiji_enabled():
        return "太极页策略未启用（TDS_KV_POLICY=lru）"

    lines = ["紫梅·太极八卦域加权页调度（TBP-II）", "=" * 56]
    lines.append(f"先天域：{_TAIJI_LAYER_DOMAIN}（对话·多轮上下文） "
                 f"保活窗口={_taiji_policy.keep_steps_of(_TAIJI_LAYER_DOMAIN)}步")
    if _HAS_ZIMEI:
        from .taiji_zimei import zimei_evict_order, relation_to_ming, wuxing_of
        ming = active_domain()
        lines.append(f"命宫（当前活跃域）：{ming}（五行={wuxing_of(ming)}）")
        order = zimei_evict_order(ming)
        rel_desc = {d: relation_to_ming(d, ming) for d in order if d != ming}
        lines.append("动态卦象淘汰序（梅花易数生克）：")
        lines.append("  " + " > ".join(
            f"{d}({rel_desc[d]})" for d in order if d != ming))
    for t in _KV_PAGE_STATE.get("page_tables") or []:
        distance = _current_step - t._token_count if _current_step else 0
        tag = _layer_tag(t.layer, distance)
        lines.append(f"层 {t.layer}: {tag}（距离={distance}步）")
    return "\n".join(lines)


def _kv_total_bytes() -> int:
    """估算当前所有层 KV 全驻留字节数（自适应预算基准，Phase2 必修）。

    按 PageTable 母张量实际字节（numel*element_size*2）优先；母张量已释放的
    层回退到 2*num_kv_heads*head_dim*capacity*dtype_bytes 字段估算。
    结果随模型规模等比缩放——固定 192/256 MiB 一刀切在 7B 文本编码器上
    压力比失控（10/17 实机：192MiB 抖动×14、256MiB 硬崩），自适应档位
    TDS_KV_BUDGET_AUTO 以"总 KV 规模的百分比"替代固定字节数。
    """
    tables = _KV_PAGE_STATE.get("page_tables") or []
    total = 0
    for t in tables:
        if t.key_tensor is not None:
            total += t.key_tensor.numel() * t.key_tensor.element_size() * 2
        elif t.num_kv_heads and t.head_dim and t.capacity_tokens:
            try:
                dtype = _dtype_from_name(t.dtype)
                total += (2 * t.num_kv_heads * t.head_dim
                          * t.capacity_tokens * dtype.itemsize)
            except Exception:
                pass
    return total


def evict_pages_to_budget(budget_bytes: int,
                          page_tables: Optional[List["PageTable"]] = None,
                          protect_layer: Optional[int] = None) -> Dict[str, int]:
    """把 KV 总占用压到显存预算内（运行时预算控制点，供推理循环调用）。

    按策略选出的次序逐页淘汰（taiji 走八卦域序，lru 走最后使用序），
    直至所有表合计占用 ≤ budget_bytes 或无可再淘汰页。
    返回统计：{evicted_pages, freed_bytes, tables}。

    protect_layer: 当前正在进入的层号，该层页整体豁免淘汰（避免"换入即换出"
    抖振）；同时 Phase3 预测目标域层也在排序中排最后。
    """
    tables = page_tables if page_tables is not None else (
        _KV_PAGE_STATE.get("page_tables") or [])
    if not tables:
        return {"evicted_pages": 0, "freed_bytes": 0, "tables": 0}
    # 预算联动已因 CUDA 错误降级：直接空转，避免反复触碰驱动级错误路径
    if _BUDGET_FAILED:
        return {"evicted_pages": 0, "freed_bytes": 0, "tables": len(tables)}

    evicted = 0
    freed = 0
    total = sum(p.memory_bytes for t in tables for p in t.pages.values())

    # 预算缺口：当前驻留 KV 页（memory_bytes 合计）超过预算才算溢出
    if total <= budget_bytes:
        return {"evicted_pages": 0, "freed_bytes": 0, "tables": len(tables)}

    # 批量淘汰上下文：命宫与访问计数分布在同批淘汰内不变，
    # 预计算一次 order_idx / median，供每层 _evict_taiji_page 复用，
    # 避免每淘汰一页都重建序字典并 O(P log P) 排序求中位数。
    ctx = None
    if taiji_enabled() and _taiji_policy is not None:
        ming = active_domain()
        if _HAS_ZIMEI:
            order_idx = {d: i for i, d in enumerate(zimei_evict_order(ming))}
        else:
            order_idx = {d: i for i, d in enumerate(_taiji_policy.cfg.evict_order)}
        counts = sorted(p.access_count for t in tables for p in t.pages.values())
        median = float(counts[len(counts) // 2]) if counts else None
        ctx = {"order_idx": order_idx, "median": median}

    # Phase3 预判联动层调度：预测域页面不参与淘汰
    predict_domain = None
    try:
        from .domain_predictor import get_predictor
        p = get_predictor()
        pred, _conf = p.predict(active_domain())
        if pred is not None:
            predict_domain = pred
            LOG.info("[TDS-KV-PHASE3] 淘汰保护预测域 %s (conf=%.2f)",
                     pred, _conf)
    except Exception:
        pass  # Phase3 未就绪时降级回纯 LRU

    # 层占用增量表：避免每轮对全部层重算 sum(memory_bytes)
    usage = [(t.layer, sum(p.memory_bytes for p in t.pages.values()), t)
             for t in tables]
    while total > budget_bytes and evicted < _BUDGET_MAX_EVICT:
        victim = None
        owner = None
        # 层间顺序：先挑驻留占用最多的层，其次按层序；预测目标域层排最后
        if predict_domain or protect_layer is not None:
            def sort_key(x):
                _, t = x[0], x[2]
                has_pred = any(getattr(pg, "domain", None) == predict_domain for pg in t.pages.values())
                protected = int(t.layer == protect_layer or has_pred)
                # 保护层（当前层/预测域层）整体豁免，排在最后；其余按占用降序
                return (protected, -x[1], x[0])
            usage.sort(key=sort_key)
        else:
            usage.sort(key=lambda x: (-x[1], x[0]))
        for _, _, t in usage:
            if t.pages:
                victim = t.evict_lru_page(ctx)
                owner = t
                break
        if victim is None:
            break
        total -= victim.memory_bytes
        # 仅更新 victim 所属层的占用值，避免全表重算
        for i, (layer_id, _, t) in enumerate(usage):
            if t is owner:
                usage[i] = (layer_id,
                            sum(p.memory_bytes for p in t.pages.values()), t)
                break
        evicted += 1
        freed += victim.memory_bytes
        # Phase2 必修：页空即释放母张量。页级淘汰只把页视图脱离母张量、
        # 并不归还显存（母张量仍整块驻留），低预算下造成"逻辑淘汰、
        # 物理空占"的假象，且母张量与 ComfyUI lowvram 动态加载竞争显存
        # （256MiB 第 2 轮模型加载 6/12 步硬崩现场）。页全空即整层卸载，
        # 显存真实归还；流序同步防止 cudaMallocAsync 下 use-after-free。
        if not owner.pages and owner.key_tensor is not None:
            owner_freed = (owner.key_tensor.numel() * owner.key_tensor.element_size() * 2)
            owner.key_tensor = None
            owner.value_tensor = None
            freed += owner_freed
            LOG.info("[TDS-KV-PHASE2] 预算淘汰页空释放 layer=%d freed=%d bytes",
                     owner.layer, owner_freed)
            _cuda_safe_sync()

    return {"evicted_pages": evicted, "freed_bytes": freed, "tables": len(tables)}


def _kv_layer_count() -> int:
    """当前持有页的层数（用于调度报告 / 可观测性）。"""
    return sum(1 for t in _KV_PAGE_STATE.get("page_tables") or [] if t.pages)


def _pressure_ratio() -> float:
    """计算当前压力比：KV 实际分配容量 / 目标预算。

    目标预算优先取环境变量 TDS_KV_BUDGET_MB（压测 / 实机统一可控）；
    未设置预算时按 1.0（视为高压力），避免把"无预算约束"误判成 LOW。

    注意：不截断到 1.0。超预算时返回真实超支比例（>1.0），
    由 from_ratio 阈值区分 CRITICAL / HIGH / FROZEN，
    避免"预算略紧就被 min(1.0,...) 一律判成 FROZEN 全层卸载"的抖振。
    """
    tables = _KV_PAGE_STATE.get("page_tables") or []
    total = float(sum(p.memory_bytes for t in tables for p in t.pages.values()))
    if total <= 0:
        return 1.0
    budget_mb = os.environ.get("TDS_KV_BUDGET_MB")
    budget_auto = os.environ.get("TDS_KV_BUDGET_AUTO")
    if budget_auto:
        try:
            frac = float(budget_auto)
            if 0.0 < frac <= 1.0:
                kv_total = _kv_total_bytes()
                if kv_total > 0:
                    return total / (kv_total * frac)
        except (TypeError, ValueError):
            pass
    if budget_mb:
        budget = float(budget_mb) * 1024 * 1024
        if budget > 0:
            return total / budget
    return 1.0


def _pressure_status() -> str:
    """把压力比映射为四象压力状态名（LOW / HIGH / CRITICAL / FROZEN）。"""
    ratio = _pressure_ratio()
    try:
        from .taiji_policy import PressureState
        state = PressureState.from_ratio(ratio)
        return state.name
    except Exception:  # pragma: no cover - 兜底
        return "CRITICAL" if ratio >= 0.92 else "LOW"


def _pressure_state_obj():
    """返回四象压力状态枚举对象（供预取/调度决策直接消费）。"""
    try:
        from .taiji_policy import PressureState
        return PressureState.from_ratio(_pressure_ratio())
    except Exception:  # pragma: no cover - 兜底 LOW（不拦截预取）
        return None


def run_taiji_schedule(pressure_ratio: Optional[float] = None,
                       pressure: Optional[str] = None) -> str:
    """太极层间运行时调度入口（实机/压测统一调用点）。

    消费 taiji_select_evict_layers 的层卸载决策，并立即执行 release_vram，
    把"决策"真正接到"搬运"上。仅在 TDS_KV_POLICY=taiji 时生效；
    否则原样返回 lru 回退说明，不执行任何卸载（由页级 LRU 自然淘汰）。

    返回人可读的调度执行报告（同显存压测对比时的关键区分依据）。
    """
    if not taiji_enabled() or _taiji_policy is None:
        return "太极调度未启用：TDS_KV_POLICY!=taiji，层间卸载保持原 LRU 行为（不回退执行）。"

    layers = _KV_PAGE_STATE.get("page_tables") or []
    if not layers:
        return "太极调度未启用：当前无已注册 PageTable。"

    pressure = (pressure or _pressure_status()).strip().lower()
    ratio = pressure_ratio if pressure_ratio is not None else _pressure_ratio()

    decision = taiji_select_evict_layers(pressure=pressure,
                                         current_step=_current_step,
                                         budget_bytes=int(
                                             os.environ.get("TDS_KV_BUDGET_MB", "0")
                                         ) * 1024 * 1024 or None)
    if not decision:
        return f"太极调度：压力={pressure} ratio={ratio:.3f}，无需要卸载的层（全驻留）。"

    offloaded = 0
    # 决策已携带 PageTable 对象：逐项精确匹配其对应的 PageCache。
    # 不可再按 layer 号建 dict——多轮共享页池下每轮都新建 layer 0..N-1
    # 的 PageTable，键会被最新一轮覆盖，导致"决策选旧层、执行卸新层"。
    tables = _KV_PAGE_STATE.get("page_tables") or []
    refs = _KV_PAGE_STATE.get("cache_refs") or []
    for table, reason in decision:
        cache = None
        for ref, pt in zip(refs, tables):
            if pt is table:
                obj = ref() if ref is not None else None
                if obj is not None:
                    cache = obj
                break
        if cache is None or not getattr(cache, "_page_table", None).pages:
            continue
        freed = cache.release_vram()  # 整层卸载：页表清空 → 0 占用
        if freed:
            LOG.info("TBP层间卸载 layer=%d freed_bytes=%d reason=%s",
                     table.layer, freed, reason)
            offloaded += 1

    return (f"太极调度：压力={pressure} ratio={ratio:.3f} "
            f"决策卸载 {len(decision)} 层，实际卸载 {offloaded} 层，"
            f"当前驻留层数={_kv_layer_count()}")
