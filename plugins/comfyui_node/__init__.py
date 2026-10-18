"""太极调度层 TDS · ComfyUI Custom Node 入口（Phase 1：桥接接入点）。

职责边界（重要）
----------------
本节点**不再自行实现 Belady 驻留调度**（早期版本曾包装 free_memory，已废弃）。
真实驻留调度由已实装的第三方插件 lmsl-smart-offload 承担；TDS 只做一件事：

    把「八卦加权 keep_steps + 四象压力策略」注入该插件的策略判据（零源码改动）。

因此本文件极薄：发现宿主插件 → 注入桥接 → 打印状态。二者不重复包装，
避免双重 Belady 与 keep_loaded 语义冲突。

安装
----
方式 A（推荐，整目录）：把整个 TDS 项目目录放到 <ComfyUI>/custom_nodes/TDS/，
  由 TDS/__init__.py 转发到本模块；目录名可改，路径探测自动完成。
方式 B（junction/软链）：custom_nodes\tds -> <TDS>/plugins/comfyui_node，
  此时建议设置环境变量 TDS_ROOT=<TDS 根目录>。

重启 ComfyUI，终端应出现：
       [TDS] 桥接已注入 lmsl-smart-offload ...
       [TDS] 太极调度层已挂载 ...
未安装 lmsl-smart-offload 时，本节点静默待命（不产生任何副作用）。

环境变量开关
------------
  TDS_BRIDGE=off      完全禁用桥接（等价于未安装 TDS 节点）
  TDS_PREFETCH=on     打开 TDS 预取总闸（默认关，沿用保守策略）
  TDS_GAIN=0.0~1.0    加权强度（默认 1.0；0 表示与宿主插件原行为等价）
"""
import logging
import os
import sys


def _find_tds_root():
    """定位 TDS 项目根（含 tds/config.py 的目录）；找不到返回 None。"""
    def has_pkg(base):
        return bool(base) and os.path.isfile(os.path.join(base, "tds", "config.py"))

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("TDS_ROOT"),
        os.path.join(here, "..", ".."),     # <TDS>/plugins/comfyui_node -> <TDS>
        os.path.join(here, ".."),           # 兼容本文件被平铺到 <TDS> 根
        here,
        os.path.join(here, "..", "..", ".."),
    ]
    for base in candidates:
        if has_pkg(base):
            return os.path.abspath(base)
    for p in list(sys.path):                # 已由外部注入
        if has_pkg(p):
            return os.path.abspath(p)
    return None


_TDS_ROOT = _find_tds_root()
if _TDS_ROOT and _TDS_ROOT not in sys.path:
    sys.path.insert(0, _TDS_ROOT)

LOG = logging.getLogger("tds.node")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "on", "yes", "y")


BRIDGE_ENABLED = _env_flag("TDS_BRIDGE", True)
PREFETCH_ENABLED = _env_flag("TDS_PREFETCH", False)
try:
    GAIN = float(os.environ.get("TDS_GAIN", "1.0"))
except ValueError:
    GAIN = 1.0


def _ensure_attached() -> None:
    """幂等挂载：发现宿主插件则注入桥接，找不到则静默待命。

    导入期与每次 prompt 到来时都会尝试，用于兼容 custom_nodes 的任意加载顺序。
    """
    if not BRIDGE_ENABLED:
        return
    try:
        from tds.config import TDSConfig
        from tds.integrations import attach_default, current_bridge

        bridge = current_bridge()
        if bridge is not None and bridge.module is not None:
            return

        cfg = TDSConfig()
        cfg.prefetch_enabled = PREFETCH_ENABLED
        bridge = attach_default(config=cfg, gain=GAIN)
        if bridge is None:
            return
        snap = bridge.snapshot()
        LOG.info("[TDS] custom_node 已接入宿主 %s (gain=%s, base_keep_steps=%s, prefetch=%s)",
                 snap["module"], snap["gain"], snap["base_keep_steps"], snap["prefetch_gate"])
    except Exception:
        LOG.exception("[TDS] 桥接挂载异常（已忽略，宿主插件行为不受影响）")


def _on_prompt(json_data):
    """prompt 钩子：兜底挂载 + 记录本次执行图的八域构成。

    注意（真机实测修复）：ComfyUI 的 ``trigger_on_prompt`` 会用每个 handler 的
    返回值覆盖 json_data 并传给下一个 handler（server.py 1462 行）。
    因此本钩子**必须原样返回 json_data**——任何提前 return/None 都会把
    json_data 置空，导致 post_prompt 后续 ``"number" in json_data`` 崩溃（HTTP 500）。
    """
    try:
        _ensure_attached()
        bridge = None
        try:
            from tds.integrations import current_bridge
            bridge = current_bridge()
        except Exception:
            pass
        if bridge is None or bridge.module is None:
            return json_data

        prompt = (json_data or {}).get("prompt") or {}
        if not prompt:
            return json_data

        # Phase 3 域注入：外部压测 / 多域负载经 extra_data.tds_domain 显式注入当前
        # 活跃八卦域（duihua/tuxiang/daima/jiansuo/gongju 等），驱动访问序列预判。
        # 未携带时保持既有域不变（回退先天域 duihua），单域场景行为不变。
        try:
            extra = (json_data or {}).get("extra_data") or {}
            dom = extra.get("tds_domain")
            if dom:
                from tds.kv.page_manager import set_active_domain
                set_active_domain(str(dom))
        except Exception:
            LOG.exception("[TDS] 域注入异常（已忽略，保持既有域）")

        from tds.integrations.comfyui_bridge import domain_of_key
        hits = {}
        for node in prompt.values():
            if not isinstance(node, dict):
                continue
            for val in (node.get("inputs") or {}).values():
                if isinstance(val, str) and "." in val:
                    dom = domain_of_key(f"{node.get('class_type','')}|{val}")
                    hits[dom] = hits.get(dom, 0) + 1
        if hits:
            LOG.info("[TDS] prompt 八域构成: %s | 压力=%s", hits, bridge.pressure_state())
    except Exception:
        LOG.exception("[TDS] _on_prompt 异常（已忽略）")
    return json_data


def _ensure_kv_host() -> None:
    """把当前宿主注册给 KV 引擎（幂等）。

    引擎（tds/kv）零宿主依赖，这里是非侵入式接线点：未装适配层或宿主缺失时
    静默待命，不影响宿主行为。
    """
    try:
        from tds.adapters.comfyui_kv_adapter import install as _install_kv_host
        if _install_kv_host():
            LOG.info("[TDS] KV 引擎宿主已注册")
    except Exception:
        LOG.exception("[TDS] KV 宿主注册失败（已忽略）")


# ---------------------------------------------------------------- 挂载
_ensure_attached()
_ensure_kv_host()

try:
    from server import PromptServer
    PromptServer.instance.add_on_prompt_handler(_on_prompt)
    LOG.info("[TDS] prompt 钩子已注册")
except Exception:
    LOG.exception("[TDS] prompt 钩子注册失败（已忽略）")

try:
    from tds.integrations import current_bridge
    _bridge = current_bridge()
    if _bridge is not None:
        snap = _bridge.snapshot()
        LOG.info("[TDS] 太极调度层已挂载 (attached=%s, gain=%s, prefetch=%s, stats=%s)",
                 snap["attached"], snap["gain"], snap["prefetch_gate"], snap["stats"])
    else:
        LOG.info("[TDS] 太极调度层已加载，等待宿主插件 lmsl-smart-offload 出现")
except Exception:
    LOG.exception("[TDS] 状态输出异常")
