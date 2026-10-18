#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TDS × ComfyUI custom_node 真机挂载仿真验证（Phase 1 收口）。

目的
----
在**不启动 ComfyUI** 的前提下，尽可能等价地复现 ComfyUI 的 custom_nodes 加载流程：

  1. 以包名 "lmsl-smart-offload" 从磁盘加载真实宿主插件（importlib，子模块 smart_offload）
  2. 以包名 "TDS" 加载 TDS 的 custom_node 入口（TDS/__init__.py -> plugins/comfyui_node）
  3. 用 stub 顶替 comfy.model_management / nodes / server

从而验证四件事：
  A. 正常加载顺序（宿主先 / TDS 后）：节点自动发现宿主并注入桥接，策略真实接管驻留决策
  B. 逆序（TDS 先 / 宿主后）：节点静默待命，prompt 钩子兜底完成注入
  C. TDS_BRIDGE=off：节点完全不注入（一键关闭）
  D. 宿主缺失：节点加载不抛异常，零副作用

用法: python tools/verify_node_mount.py
"""
import importlib.util
import logging
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
TDS_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (TDS_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from verify_bridge_mount import (  # noqa: E402
    DEFAULT_SMART_OFFLOAD_DIR,
    build_prompt,
    install_stub_env,
    run_case,
)
from tds.integrations import current_bridge  # noqa: E402
import tds.integrations.comfyui_bridge as igc  # noqa: E402

HOST_PKG = "lmsl-smart-offload"
NODE_PKG = "TDS"


# ==========================================================================
# 日志捕获
# ==========================================================================

class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.messages = []

    def emit(self, record):
        try:
            self.messages.append(record.getMessage())
        except Exception:
            pass

    def has(self, needle: str) -> bool:
        return any(needle in m for m in self.messages)


# ==========================================================================
# 加载器（模拟 ComfyUI custom_nodes 加载）
# ==========================================================================

def install_server_stub():
    mod = types.ModuleType("server")

    class PromptServer:
        instance = None

        def __init__(self):
            self.handlers = []

        def add_on_prompt_handler(self, fn):
            self.handlers.append(fn)

    ins = PromptServer()
    PromptServer.instance = ins
    mod.PromptServer = PromptServer
    sys.modules["server"] = mod
    return ins


def load_host_as_comfyui(directory: str):
    """按 ComfyUI 方式加载宿主插件：包名 = 目录名，子模块 smart_offload。"""
    if not os.path.isdir(directory):
        raise FileNotFoundError(directory)
    pkg = types.ModuleType(HOST_PKG)
    pkg.__path__ = [directory]
    pkg.__package__ = HOST_PKG
    sys.modules[HOST_PKG] = pkg

    sub_path = os.path.join(directory, "smart_offload.py")
    spec = importlib.util.spec_from_file_location(f"{HOST_PKG}.smart_offload", sub_path)
    sub = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sub
    spec.loader.exec_module(sub)          # 真实执行：内部 import nodes / comfy.model_management
    pkg.smart_offload = sub
    return pkg, sub


def load_node_pkg(name: str, tds_root: str):
    """按 ComfyUI 方式加载 TDS custom_node（包名可任意）。"""
    init_py = os.path.join(tds_root, "__init__.py")
    spec = importlib.util.spec_from_file_location(name, init_py,
                                                  submodule_search_locations=[tds_root])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_host_as_comfyui_pathstyle(directory: str):
    """完全复刻 ComfyUI nodes.py 对目录型 custom_node 的加载方式：
    ``sys_module_name = module_path.replace(".", "_x_")``
    → sys.modules 的键是**插件目录的绝对路径字符串**，而不是目录名。
    另：真机执行的是插件 __init__.py（会自行调用 install()）。
    """
    sys_module_name = directory.replace(".", "_x_")
    spec = importlib.util.spec_from_file_location(
        sys_module_name,
        os.path.join(directory, "__init__.py"),
        submodule_search_locations=[directory],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[sys_module_name] = mod
    spec.loader.exec_module(mod)
    return sys_module_name, mod


def unload(*_ignored: str) -> None:
    """清场：节点包 TDS.* + 宿主插件（包名式键与 ComfyUI『路径字符串』键都要覆盖）。

    注意：**不能**清 tds.*（TDS 库本体）。否则脚本顶部持有的 igc 引用会与
    节点重新 import 出的新模块对象分叉，桥接单例读到的是旧对象 → 误报未注入。
    """
    for n in list(sys.modules):
        s = str(n)
        low = s.lower().replace("\\", "/")
        if (s == "TDS" or s.startswith("TDS.")
                or "lmsl-smart-offload" in low
                or "lmsl_smart_offload" in low
                or "smart_offload" in low):
            sys.modules.pop(n, None)


def fresh(bridge_reset: bool = True):
    """清空上一场景残留：模块 + 桥接单例 + 日志。"""
    unload(NODE_PKG, HOST_PKG)
    if bridge_reset:
        igc._BRIDGE = None
    cap = LogCapture()
    lg = logging.getLogger("tds")
    lg.setLevel(logging.INFO)
    for h in list(lg.handlers):
        lg.removeHandler(h)
    lg.addHandler(cap)
    return cap


# ==========================================================================
# 场景
# ==========================================================================

def scenario_a(failures) -> None:
    print("\n[A] 正常顺序：宿主先加载，TDS custom_node 后加载 → 应自动注入")
    cap = fresh()
    install_server_stub()
    mm = install_stub_env()
    _, so = load_host_as_comfyui(DEFAULT_SMART_OFFLOAD_DIR)
    so.install()
    node = load_node_pkg(NODE_PKG, TDS_ROOT)

    bridge = current_bridge()
    ok = bridge is not None and bridge.module is so
    print(f"    节点加载完成 -> current_bridge={'有' if bridge else '无'} "
          f"| 指向真实宿主={bridge.module is so if bridge else False}")
    print(f"    日志: {[m for m in cap.messages if '桥接' in m or '挂载' in m]}")
    if not ok:
        failures.append("场景A: 节点未自动注入桥接")
        return

    handlers = sys.modules["server"].PromptServer.instance.handlers
    print(f"    prompt 钩子注册数: {len(handlers)}")
    if not handlers:
        failures.append("场景A: prompt 钩子未注册")

    for h in handlers:                                   # 触发钩子（兜底路径）
        h({"prompt": build_prompt()})

    rows = run_case(so, mm, bridge, base_keep_steps=1, load_events=3)
    probe = next((r for r in rows if r["domain"] == "duihua"), None)
    if probe is None:
        failures.append("场景A: 未找到 duihua 域模型")
        return
    print(f"    duihua: 原生dist={probe['orig_dist']} 有效dist={probe['eff_dist']} "
          f"实际保留={'是' if probe['kept'] else '否'}（原生基准1下本应不保）")
    if not probe["kept"]:
        failures.append("场景A: 节点注入后 duihua 域未被 TDS 策略保护")
    if probe["orig_dist"] != 2:
        failures.append(f"场景A: 期望原生 dist=2，实际 {probe['orig_dist']}（用例前提变化）")
    if not cap.has("已接入宿主"):
        failures.append("场景A: 节点未输出『custom_node 已接入宿主』日志")
    if not cap.has("prompt 八域构成") and not cap.has("八域"):
        print("    （提示：prompt 八域构成日志未出现）")


def scenario_e(failures) -> None:
    print("\n[E] 真机形态复刻：宿主以『路径字符串』为 sys.modules 键加载（Phase1 真机断点回归）")
    cap = fresh()
    install_server_stub()
    mm = install_stub_env()
    key, host = load_host_as_comfyui_pathstyle(DEFAULT_SMART_OFFLOAD_DIR)
    print(f"    宿主模块键: {key}")
    sub = getattr(host, "smart_offload", None)
    print(f"    宿主包内 smart_offload 可达: {sub is not None}")
    if sub is None:
        failures.append("场景E: 复刻环境下宿主包未暴露 smart_offload（复刻本身失败）")
        return

    load_node_pkg(NODE_PKG, TDS_ROOT)                 # TDS 后加载
    bridge = current_bridge()
    if bridge is None:
        failures.append("场景E: 路径式模块键下 TDS 仍未探测到宿主（真机断点未修复）")
        return
    print(f"    桥接指向真实调度模块: {bridge.module is sub}")
    if bridge.module is not sub:
        failures.append("场景E: 桥接未指向真实宿主调度模块")
    if cap.has("未发现"):
        failures.append("场景E: 仍输出『未发现宿主』，新探测逻辑未生效")

    for h in sys.modules["server"].PromptServer.instance.handlers:
        h({"prompt": build_prompt()})
    rows = run_case(sub, mm, bridge, base_keep_steps=1, load_events=3)
    probe = next((r for r in rows if r["domain"] == "duihua"), None)
    if probe is None:
        failures.append("场景E: 未找到 duihua 域模型")
        return
    print(f"    duihua: 原生dist={probe['orig_dist']} 有效dist={probe['eff_dist']} "
          f"实际保留={'是' if probe['kept'] else '否'}")
    if not probe["kept"]:
        failures.append("场景E: 路径式加载下 TDS 八卦加权未生效")


def scenario_b(failures) -> None:
    print("\n[B] 逆序：TDS 先加载（宿主缺席）→ 待命；宿主出现后 prompt 钩子兜底注入")
    cap = fresh()
    handlers_holder = install_server_stub()
    install_stub_env()
    node = load_node_pkg(NODE_PKG, TDS_ROOT)
    if current_bridge() is not None:
        failures.append("场景B: 宿主缺席时不应注入")
        return
    print(f"    待命日志: {[m for m in cap.messages if '等待宿主' in m or '待命' in m]}")

    _, so = load_host_as_comfyui(DEFAULT_SMART_OFFLOAD_DIR)
    so.install()
    for h in handlers_holder.handlers:
        h({"prompt": build_prompt()})

    bridge = current_bridge()
    print(f"    兜底后 current_bridge={'有' if bridge else '无'}")
    if bridge is None or bridge.module is not so:
        failures.append("场景B: prompt 钩子未能兜底注入")


def scenario_c(failures) -> None:
    print("\n[C] TDS_BRIDGE=off：节点应完全不注入")
    fresh()
    install_server_stub()
    install_stub_env()
    _, so = load_host_as_comfyui(DEFAULT_SMART_OFFLOAD_DIR)
    so.install()

    os.environ["TDS_BRIDGE"] = "off"
    try:
        load_node_pkg(NODE_PKG, TDS_ROOT)
    finally:
        os.environ.pop("TDS_BRIDGE", None)
    bridge = current_bridge()
    print(f"    current_bridge={'有' if bridge else '无'}")
    if bridge is not None:
        failures.append("场景C: TDS_BRIDGE=off 时仍注入了桥接")


def scenario_d(failures) -> None:
    print("\n[D] 宿主缺失：节点加载应静默、无异常、零副作用")
    cap = fresh()
    install_server_stub()
    mm = install_stub_env()
    try:
        load_node_pkg(NODE_PKG, TDS_ROOT)
    except Exception as e:
        failures.append(f"场景D: 节点加载抛异常 {e!r}")
        return
    if current_bridge() is not None:
        failures.append("场景D: 无宿主却建立了桥接")
    if "free_memory" in mm.__dict__:                 # 只有被 setattr 打过补丁才会进实例字典
        failures.append("场景D: 无宿主却改动了宿主 free_memory")
    print(f"    待命日志: {[m for m in cap.messages if '等待宿主' in m or '待命' in m]}")


def main() -> int:
    print("=" * 96)
    print("TDS × ComfyUI custom_node 真机挂载仿真验证（真实插件包 + stub 运行时）")
    print("=" * 96)
    failures = []
    scenario_a(failures)
    scenario_e(failures)
    scenario_b(failures)
    scenario_c(failures)
    scenario_d(failures)

    print("\n" + "=" * 96)
    if failures:
        print(f"[FAIL] {len(failures)} 项未通过：")
        for f in failures:
            print("  - " + f)
        return 1
    print("[OK] custom_node 挂载验证通过：加载顺序无关、自动/兜底注入、可一键关闭、无宿主零副作用。")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())
