"""TDS × lmsl-smart-offload 桥接层（Phase 1 收口）。

目标
----
让太极调度层 TDS 的「八卦加权 keep_steps + 四象压力策略」驱动已实装于
ComfyUI 的插件 lmsl-smart-offload（Belady 近似驻留调度器），在
**不修改其源码、不修改 ComfyUI 源码**的前提下完成策略接管，并可一键回退。

原理（零侵入策略注入）
----------------------
lmsl-smart-offload 的驻留判据是**全局单一阈值**：

    dist(key) <= smart_offload.config.keep_steps   ->  把 LoadedModel 并入 keep_loaded

其中 ``dist`` 由该模块的 ``_next_use_distance(key)`` 给出（拓扑 rank 距离）。
TDS 需要的是**按域加权的阈值**：``dist <= keep_steps_for_domain(domain(key))``。

桥接层替换 ``_next_use_distance``，返回「域加权有效距离」：

    dist_eff = dist - (ks_domain(key) - base)      base = smart_offload.config.keep_steps

于是 ``dist_eff <= base  <=>  dist <= ks_domain``，与 TDS 策略**精确等价**；
当 ``ks_domain == base`` 时返回值与原始实现逐位一致（安全退化，零行为漂移）。

四象压力策略在**同一注入点**生效（显存水位唯一判据）：

    LOW      正常保护 + 允许预取
    CRITICAL 正常保护 + 禁止预取
    HIGH     放弃保护（交由框架自身，避免加剧 OOM）+ 禁止预取
    FROZEN   放弃保护 + 禁止预取

释放类动作（卸载/下沉）始终放行——否则显存永远降不下来。

安全约束
--------
- 默认保守：以 smart_offload 自身 config 为基准，TDSConfig.prefetch_enabled=False。
- fail-soft：任何异常都被吞掉并回退到原始行为。
- attach/detach 幂等，支持热切换与灰度回退（detach 后与未安装桥接完全一致）。
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Callable, Dict, List, Optional

from ..config import BAGUA_KEEP_STEPS, TDSConfig
from ..core.memory import VramState
from ..core.scheduler import PressureState, Scheduler

LOG = logging.getLogger("tds.bridge")

# --------------------------------------------------------------------------
# 域判定：把 lmsl-smart-offload 的模型 key（"class_type|file_name"）映射到八域
# --------------------------------------------------------------------------

# 顺序敏感：先命中者胜。未命中一律回落 "qita"（keep_steps=1 = 保守，等价不干预）。
_DOMAIN_SIGNALS = (
    ("shipin", ("wan", "ltx", "svd", "video", "animate", "cogvideo",
                "hunyuanvideo", "cosmos", "framepack", "interpolation")),
    ("tuxiang", ("flux", "sdxl", "sd15", "stable-diffusion", "z-image", "zimage",
                 "kolors", "qwen-image", "qwen_image", "pony", "realistic",
                 "unet", "diffusion", "lora", "controlnet", "ipadapter",
                 "faceid", "instantid", "esrgan", "upscal", "vae", "checkpoint",
                 "segment", "sam2", "sam3", "depth", "pose")),
    ("duihua", ("clip", "text_encoder", "t5", "llm", "ollama", "prompt", "tts")),
    ("gongju", ("get_meta", "housekeeper", "utils", "tool")),
)


def domain_of_key(key: str) -> str:
    """模型 key -> TDS 八域。无法判定时返回 "qita"（保守）。"""
    k = (key or "").lower()
    for dom, signals in _DOMAIN_SIGNALS:
        for sig in signals:
            if sig in k:
                return dom
    return "qita"


# --------------------------------------------------------------------------
# 显存水位探针（默认走 TDS 的 ComfyUIAdapter → torch.cuda.mem_get_info）
# --------------------------------------------------------------------------

_DEFAULT_PROBE: Optional[Callable[[], Optional[VramState]]] = None


def _default_vram_probe() -> Optional[VramState]:
    """读取真实显存水位；不可用时返回 None（Scheduler 视作 LOW，即宽松）。"""
    global _DEFAULT_PROBE
    if _DEFAULT_PROBE is None:
        try:
            from ..adapters.comfyui_adapter import ComfyUIAdapter
            adapter = ComfyUIAdapter()
            _DEFAULT_PROBE = adapter.vram_state
        except Exception:
            _DEFAULT_PROBE = lambda: None  # type: ignore[assignment]
    try:
        state = _DEFAULT_PROBE()  # type: ignore[misc]
        if state is not None and getattr(state, "total_mb", 0) > 0:
            return state
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------
# 桥接主体
# --------------------------------------------------------------------------

class ComfyBridge:
    """把 TDS 策略注入 lmsl-smart-offload（纯函数级替换，零源码改动）。"""

    def __init__(self,
                 config: Optional[TDSConfig] = None,
                 scheduler: Optional[Scheduler] = None,
                 vram_probe: Optional[Callable[[], Optional[VramState]]] = None,
                 gain: float = 1.0,
                 respect_tds_prefetch_gate: bool = True):
        self.cfg = config or TDSConfig()
        self.sched = scheduler or Scheduler(self.cfg)
        self.gain = float(gain)                    # 加权强度（0=完全等价原插件，1=标准 TDS）
        self.respect_tds_prefetch_gate = bool(respect_tds_prefetch_gate)
        self._vram_probe = vram_probe or _default_vram_probe
        self.module: Any = None                    # 被注入的 smart_offload 模块
        self._orig: Dict[str, Any] = {}
        self._domain_cache: Dict[str, str] = {}
        self.stats: Dict[str, int] = {}

    # ---------------- 统计 ----------------
    def _bump(self, key: str, n: int = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + n

    @property
    def orig_next_use_distance(self) -> Optional[Callable[[str], Optional[int]]]:
        """宿主插件原始的 _next_use_distance（未注入时返回 None）。"""
        if not self._orig:
            return None
        return self._orig.get("_next_use_distance")

    def snapshot(self) -> dict:
        return {
            "attached": self.module is not None,
            "module": getattr(self.module, "__name__", None),
            "gain": self.gain,
            "base_keep_steps": self.base_keep_steps(),
            "prefetch_gate": self.cfg.prefetch_enabled,
            "stats": dict(self.stats),
        }

    # ---------------- 策略：域 / 阈值 / 压力 ----------------
    def domain_of(self, key: str) -> str:
        dom = self._domain_cache.get(key)
        if dom is None:
            dom = domain_of_key(key)
            self._domain_cache[key] = dom
        return dom

    def keep_steps_for_key(self, key: str) -> int:
        """该模型所属域的八卦加权 keep_steps。"""
        return self.sched.keep_steps_for_domain(self.domain_of(key))

    def base_keep_steps(self) -> int:
        """宿主插件当前的全局基准阈值（尊重工作流内 SmartResidentScheduler 节点设置）。"""
        try:
            return int(getattr(self.module.config, "keep_steps", 1))
        except Exception:
            return 1

    def pressure_state(self) -> str:
        """四象压力状态（显存水位唯一判据）。"""
        try:
            state = self._vram_probe()
        except Exception:
            state = None
        return self.sched.pressure_state(state)

    def effective_distance(self, dist: int, key: str, base: Optional[int] = None) -> int:
        """域加权有效距离：与 ``dist <= keep_steps_for_key(key)`` 精确等价。"""
        if base is None:
            base = self.base_keep_steps()
        ks = self.keep_steps_for_key(key)
        bonus = int(round((ks - base) * self.gain))
        if bonus > 0:
            self._bump("keep.tightened")
        elif bonus < 0:
            self._bump("keep.relaxed")
        else:
            self._bump("keep.neutral")
        return dist - bonus

    # ---------------- 注入 ----------------
    def attach(self, module: Any) -> bool:
        """把策略注入给定的 smart_offload 模块（幂等）。"""
        if self.module is module:
            return True
        if self.module is not None:
            self.detach()
        try:
            if not hasattr(module, "_next_use_distance"):
                LOG.warning("[TDS] 目标模块缺少 _next_use_distance，放弃注入")
                return False
            mod_dict = getattr(module, "__dict__", {})
            self._orig = {
                "_next_use_distance": module._next_use_distance,
                "_maybe_prefetch": getattr(module, "_maybe_prefetch", None),
                # 记录属性原本是否驻留在模块 __dict__（决定回退用 setattr 还是 delattr）
                "_next_in_dict": "_next_use_distance" in mod_dict,
                "_pf_in_dict": "_maybe_prefetch" in mod_dict,
            }
            module._next_use_distance = self._make_next_use_distance(
                module, self._orig["_next_use_distance"])
            if self._orig["_maybe_prefetch"] is not None:
                module._maybe_prefetch = self._make_maybe_prefetch(
                    self._orig["_maybe_prefetch"])
            self.module = module
            self._bump("attach")
            LOG.info("[TDS] 桥接已注入 %s (gain=%.2f, base=%s, prefetch_gate=%s)",
                     getattr(module, "__name__", "smart_offload"), self.gain,
                     self.base_keep_steps(), self.cfg.prefetch_enabled)
            return True
        except Exception:
            LOG.exception("[TDS] 桥接注入失败，保持插件原始行为")
            self._orig = {}
            self.module = None
            return False

    def detach(self) -> bool:
        """回退到原始行为（与未安装桥接完全一致）。"""
        if self.module is None:
            return False
        try:
            self._restore("_next_use_distance", self._orig.get("_next_use_distance"),
                          self._orig.get("_next_in_dict", False))
            self._restore("_maybe_prefetch", self._orig.get("_maybe_prefetch"),
                          self._orig.get("_pf_in_dict", False))
            self._bump("detach")
            LOG.info("[TDS] 桥接已回退，插件恢复原始行为")
        except Exception:
            LOG.exception("[TDS] 桥接回退异常")
        finally:
            self.module = None
            self._orig = {}
        return True

    def _restore(self, attr: str, value: Any, was_in_dict: bool) -> None:
        """回退单个属性：原本驻留模块 __dict__ 则还原，否则移除临时注入。"""
        if value is None:
            return
        if was_in_dict:
            setattr(self.module, attr, value)
        else:
            try:
                delattr(self.module, attr)   # 类属性/方法：删除实例覆盖即自动回落
            except Exception:
                pass

    # ---------------- 注入实现 ----------------
    def _make_next_use_distance(self, module: Any, orig: Callable[[str], Optional[int]]):
        """替换 _next_use_distance：叠加八卦加权 + 四象压力门控。"""
        def _next_use_distance(key: str) -> Optional[int]:
            try:
                dist = orig(key)
            except Exception:
                return None
            if dist is None:
                self._bump("dist.no_future")
                return None
            pressure = self.pressure_state()
            if pressure in (PressureState.HIGH, PressureState.FROZEN):
                # 高压力/冻结：放弃驻留保护，交由框架自身卸载，避免加剧 OOM。
                self._bump("pressure_skip")
                return None
            if pressure == PressureState.CRITICAL:
                self._bump("critical.hold")
            return self.effective_distance(dist, key, base=self.base_keep_steps())
        return _next_use_distance

    def _make_maybe_prefetch(self, orig: Callable[[], None]):
        """包装 _maybe_prefetch：仅 LOW 压力且 TDS 预取总闸开启时放行。"""
        def _maybe_prefetch() -> None:
            # 安全维度优先：先看显存水位，再看 TDS 预取总闸
            if self.pressure_state() != PressureState.LOW:
                self._bump("prefetch.blocked_pressure")
                return
            if self.respect_tds_prefetch_gate and not self.cfg.prefetch_enabled:
                self._bump("prefetch.blocked_gate")
                return
            self._bump("prefetch.allowed")
            return orig()
        return _maybe_prefetch


# --------------------------------------------------------------------------
# 模块发现与默认挂载（供 custom_node / 离线验证复用）
# --------------------------------------------------------------------------

_PACKAGE_NAMES = ("lmsl-smart-offload", "lmsl_smart_offload")
_PLUGIN_DIR_TOKEN = "lmsl-smart-offload"
_PLUGIN_ALT_TOKEN = "lmsl_smart_offload"


def _pick_scheduler(mod: Any) -> Optional[Any]:
    """从候选模块中取出真正带调度入口的模块（具备 _next_use_distance）。"""
    if mod is None:
        return None
    if hasattr(mod, "_next_use_distance"):
        return mod
    sub = getattr(mod, "smart_offload", None)
    if sub is not None and hasattr(sub, "_next_use_distance"):
        return sub
    return None


def _haystack(name: Any, mod: Any) -> str:
    try:
        f = getattr(mod, "__file__", None) or ""
    except Exception:
        f = ""
    return f"{name} {f}".lower().replace("\\", "/")


def find_smart_offload() -> Optional[Any]:
    """在当前进程内定位 lmsl-smart-offload 的调度入口模块。

    ComfyUI 对目录型 custom_node 的加载方式决定了 sys.modules 的键
    **不是**目录名：nodes.py 中 ``sys_module_name = module_path.replace(".", "_x_")``，
    即键为**插件目录的绝对路径字符串**（真机实测的断点，包名探测必然 miss）。

    因此这里不再依赖任何固定键名，改为三级探测：
      1) 常规包名快路径（软链/手工导入等场景）
      2) sys.modules 全量扫描，按「模块名 + __file__ 路径」双匹配定位插件包
      3) 子模块兜底：键名含 "smart_offload" 的模块
    """
    for name in _PACKAGE_NAMES:                      # 1) 包名快路径
        hit = _pick_scheduler(sys.modules.get(name))
        if hit is not None:
            return hit

    for name, mod in list(sys.modules.items()):      # 2) 路径 / 名字全量扫描
        if mod is None:
            continue
        hay = _haystack(name, mod)
        if _PLUGIN_DIR_TOKEN in hay or _PLUGIN_ALT_TOKEN in hay:
            hit = _pick_scheduler(mod)
            if hit is not None:
                return hit

    for name, mod in list(sys.modules.items()):      # 3) 子模块兜底
        if "smart_offload" in _haystack(name, mod):
            hit = _pick_scheduler(mod)
            if hit is not None:
                return hit
    return None


_BRIDGE: Optional[ComfyBridge] = None
_KV_ENGINE_ACTIVE = False  # Phase 2：KV 引擎开闸标记（三态互斥，见 tds.kv.activate）


def _try_activate_kv() -> None:
    """桥接挂载后按 TDS_KV 统一开闸 KV 引擎（三态互斥，幂等）。

    模式由 ``tds.kv.activate()`` 判定：off / observe / page。开闸与宿主注册
    解耦——宿主缺席时引擎挂起，待 ``host.set_host()`` 后由宿主回调自动补包装。
    """
    global _KV_ENGINE_ACTIVE
    if _KV_ENGINE_ACTIVE:
        return
    try:
        from ..kv import activate
        mode = activate()
        _KV_ENGINE_ACTIVE = True
        LOG.info("[TDS] KV 引擎开闸模式: %s", mode)
    except Exception:
        LOG.exception("[TDS] KV 引擎开闸异常（已忽略）")



def attach_default(**kwargs) -> Optional[ComfyBridge]:
    """定位 smart_offload 并注入 TDS 策略；找不到宿主返回 None。"""
    global _BRIDGE
    module = find_smart_offload()
    if module is None:
        LOG.info("[TDS] 未发现 lmsl-smart-offload，桥接待命")
        return None
    if _BRIDGE is None or _BRIDGE.module is not module:
        bridge = ComfyBridge(**kwargs)
        bridge.attach(module)
        _BRIDGE = bridge
    # Phase 2：挂载桥接后按 TDS_KV 统一开闸 KV 引擎
    _try_activate_kv()
    return _BRIDGE


def current_bridge() -> Optional[ComfyBridge]:
    return _BRIDGE

