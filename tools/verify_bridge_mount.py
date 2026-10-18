#!/usr/bin/env python
"""TDS × lmsl-smart-offload 离线挂载验证。

验证对象是**磁盘上真实的第三方插件源码**
（E:\\LMSL-comfyui\\ComfyUI\\custom_nodes\\lmsl-smart-offload\\smart_offload.py），
而非替身：脚本用 stub 顶替 ComfyUI 运行时（nodes / comfy.model_management），
真实执行该插件的「prompt 预扫描 → plan/events → free_memory 驻留决策」全链路，
再对比 桥接开启 / 关闭 两种状态下被保护的模型集合。

用法:
    python tools/verify_bridge_mount.py
    python tools/verify_bridge_mount.py --smart-offload-dir D:\\path\\to\\lmsl-smart-offload
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from tds.config import BAGUA_KEEP_STEPS
from tds.core.memory import VramState
from tds.integrations.comfyui_bridge import ComfyBridge, domain_of_key

DEFAULT_SMART_OFFLOAD_DIR = r"E:\LMSL-comfyui\ComfyUI\custom_nodes\lmsl-smart-offload"
DEVICE = "cuda:0"


# ==========================================================================
# 1. 最小运行时 stub
# ==========================================================================

class _NodeDef:
    """ComfyUI 节点类替身：只需要 RETURN_TYPES + INPUT_TYPES()。"""

    def __init__(self, return_types, file_field: Optional[str] = None, choices=()):
        self.RETURN_TYPES = tuple(return_types)
        self._file_field = file_field
        self._choices = list(choices) or [f"{file_field}.safetensors"]

    def INPUT_TYPES(self):
        req: Dict[str, Any] = {}
        if self._file_field:
            req[self._file_field] = [list(self._choices)]
        return {"required": req}


NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    "CheckpointLoaderSimple": _NodeDef(("MODEL", "CLIP", "VAE"),
                                       "ckpt_name", ["flux1-dev.safetensors"]),
    "LoraLoader": _NodeDef(("MODEL", "CLIP"), "lora_name", ["detail.safetensors"]),
    "CLIPLoader": _NodeDef(("CLIP",), "clip_name", ["t5xxl.safetensors"]),
    "WanVideoModelLoader": _NodeDef(("MODEL",), "model", ["wan2.1_t2v.safetensors"]),
    "VAELoader": _NodeDef(("VAE",), "vae_name", ["wan_2.1_vae.safetensors"]),
    "KSampler": _NodeDef(("LATENT",)),
}


class _MM(types.ModuleType):
    """comfy.model_management 替身：只保留被包装到的那几个入口。"""

    def __init__(self):
        super().__init__("comfy.model_management")
        self.current_loaded_models: List[Any] = []
        self.last_keep: List[Any] = []
        self.free_calls = 0

    # --- 原版（会被插件 install() 包装） ---
    def free_memory(self, memory_required=0, device=None, keep_loaded=None,
                    for_dynamic=False, pins_required=0, ram_required=0):
        self.free_calls += 1
        self.last_keep = list(keep_loaded or [])
        return []

    def load_models_gpu(self, models, memory_required=0, force_patch_weights=False,
                        minimum_memory_required=None, force_full_load=False):
        for lm in models:
            self.current_loaded_models.append(lm)
        return None

    # --- 探针 ---
    def get_free_memory(self, device=None):
        return 8 * 1024 ** 3

    def get_total_memory(self, device=None):
        return 12 * 1024 ** 3

    def minimum_inference_memory(self):
        return 1 * 1024 ** 3


class FakeLoadedModel:
    """comfy.model_management.LoadedModel 替身。"""

    _seq = 0

    def __init__(self):
        FakeLoadedModel._seq += 1
        self.uid = FakeLoadedModel._seq
        self.device = DEVICE
        self.currently_used = True
        self.last_used_step = 0

    def is_dead(self) -> bool:
        return False

    def real_model(self):
        return types.SimpleNamespace(model=types.SimpleNamespace())

    def model_memory(self) -> int:
        return 1024 * 1024 * 1024

    def model_offloaded_memory(self) -> int:
        return 512 * 1024 * 1024

    def model_load(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"<LoadedModel #{self.uid}>"


def install_stub_env() -> _MM:
    nodes_mod = types.ModuleType("nodes")
    nodes_mod.NODE_CLASS_MAPPINGS = NODE_CLASS_MAPPINGS
    sys.modules["nodes"] = nodes_mod

    comfy_pkg = types.ModuleType("comfy")
    comfy_pkg.__path__ = []          # 声明为包
    mm = _MM()
    comfy_pkg.model_management = mm
    sys.modules["comfy"] = comfy_pkg
    sys.modules["comfy.model_management"] = mm
    return mm


# ==========================================================================
# 2. 加载磁盘上真实的插件源码
# ==========================================================================

def load_real_smart_offload(directory: str):
    path = os.path.join(directory, "smart_offload.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到真实插件源码: {path}")
    spec = importlib.util.spec_from_file_location("lmsl_smart_offload_real", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)     # 真实执行（内部 import nodes / comfy.model_management）
    return mod, path


# ==========================================================================
# 3. 测试用执行图（7 个节点；CLIP / Wan 各复用一次，制造"未来使用"）
# ==========================================================================

def build_prompt() -> Dict[str, Any]:
    def node(ct: str, **inputs):
        return {"class_type": ct, "inputs": inputs}

    return {
        "1": node("CheckpointLoaderSimple", ckpt_name="flux1-dev.safetensors"),
        "2": node("LoraLoader", lora_name="detail.safetensors"),
        "3": node("CLIPLoader", clip_name="t5xxl.safetensors"),
        "4": node("WanVideoModelLoader", model="wan2.1_t2v.safetensors"),
        "5": node("CLIPLoader", clip_name="t5xxl.safetensors"),   # 复用 -> 有未来使用
        "6": node("WanVideoModelLoader", model="wan2.1_t2v.safetensors"),  # 复用
        "7": node("VAELoader", vae_name="wan_2.1_vae.safetensors"),
    }


# ==========================================================================
# 4. 单轮仿真
# ==========================================================================

def run_case(smart_offload, mm: _MM, bridge: Optional[ComfyBridge],
             base_keep_steps: int, load_events: int) -> List[Dict[str, Any]]:
    """跑一轮：预扫描 -> 逐事件加载 -> 触发 free_memory -> 收集驻留决策。"""
    # 复位插件状态（等价于新 prompt 到来）
    smart_offload.config.keep_steps = base_keep_steps
    smart_offload.config.enabled = True
    smart_offload._plan = None
    smart_offload._events = []
    smart_offload._event_index = 0
    smart_offload._current_rank = -1
    smart_offload._used_count = {}
    smart_offload._key_of_loadedmodel.clear()
    mm.current_loaded_models.clear()
    mm.last_keep = []

    # 真实预扫描（插件自身的 on_prompt_handler）
    smart_offload.on_prompt_handler({"prompt": build_prompt()})

    # 真实加载路径（mm.load_models_gpu 已被 install() 包装成 patched 版本）
    for _ in range(load_events):
        mm.load_models_gpu([FakeLoadedModel()])

    # 真实卸载决策路径
    mm.free_memory(0, DEVICE, keep_loaded=[])

    # 区分「插件原始判据」与「TDS 加权后的判据」
    orig_fn = None
    if bridge is not None and bridge.module is smart_offload:
        orig_fn = bridge.orig_next_use_distance
    if orig_fn is None:
        orig_fn = smart_offload._next_use_distance      # 桥接关闭：两者同为原始实现
    eff_fn = smart_offload._next_use_distance

    rows = []
    for lm in mm.current_loaded_models:
        key = smart_offload._key_of_loadedmodel.get(id(lm), "?")
        rows.append({
            "lm": lm,
            "key": key,
            "domain": domain_of_key(key),
            "orig_dist": orig_fn(key),      # 插件原生距离
            "eff_dist": eff_fn(key),        # TDS 域加权有效距离
            "kept": lm in mm.last_keep,
        })
    return rows


def explain(rows: List[Dict[str, Any]], base: int) -> None:
    for r in rows:
        ks = BAGUA_KEEP_STEPS.get(r["domain"], 1)
        od, ed = r["orig_dist"], r["eff_dist"]
        od_s = "无未来使用" if od is None else f"dist={od}"
        ed_s = "-" if ed is None else str(ed)
        print(f"    {r['key']:<58} 域={r['domain']:<8} {od_s:<10} "
              f"有效距离={ed_s:<4} 插件(base={base})={'保' if (od is not None and od <= base) else '不保'}  "
              f"TDS(ks={ks})={'保' if (od is not None and od <= ks) else '不保'}  "
              f"实际保留={'是' if r['kept'] else '否'}")


# ==========================================================================
# 5. 主流程
# ==========================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smart-offload-dir", default=DEFAULT_SMART_OFFLOAD_DIR)
    args = ap.parse_args()

    print("=" * 96)
    print("TDS × lmsl-smart-offload 离线挂载验证（真实插件源码 + stub 运行时）")
    print("=" * 96)

    mm = install_stub_env()
    smart_offload, path = load_real_smart_offload(args.smart_offload_dir)
    print(f"[1] 真实插件源码已加载: {path}")
    print(f"    free_memory 原始实现: {smart_offload._orig_free_memory}")
    print(f"    load_models_gpu 原始实现: {smart_offload._orig_load_models_gpu}")

    smart_offload.install()
    print(f"[2] 插件 install() 完成 → mm.free_memory = {mm.free_memory.__name__}")

    bridge = ComfyBridge(vram_probe=lambda: VramState(total_mb=12288, used_mb=3686, free_mb=8602))
    ok = bridge.attach(smart_offload)
    print(f"[3] 桥接注入: {'成功' if ok else '失败'} ; "
          f"_next_use_distance -> {smart_offload._next_use_distance.__name__}")
    if not ok:
        return 1

    failures: List[str] = []

    # ---- 场景 A/B：base=1（插件默认），分别在 cursor=rank2 / rank3 ----
    for label, load_events in (("A · 已加载前 3 个模型（cursor=rank2）", 3),
                               ("B · 已加载前 4 个模型（cursor=rank3）", 4)):
        print(f"\n[4] 场景 {label}")
        rows = run_case(smart_offload, mm, bridge, base_keep_steps=1, load_events=load_events)
        explain(rows, base=1)
        for r in rows:
            od, ed = r["orig_dist"], r["eff_dist"]
            ks = BAGUA_KEEP_STEPS.get(r["domain"], 1)
            expected = od is not None and od <= ks          # TDS 期望：按域加权阈值
            if r["kept"] != expected:
                failures.append(f"场景{label}: {r['key']} 期望{'保' if expected else '不保'}，"
                                f"实际{'保' if r['kept'] else '不保'}（dist={od}, ks={ks}）")
            # 同时校验注入后的有效距离与 base 的判据仍然等价
            if (ed is not None) and (ed <= 1) != (od is not None and od <= ks):
                failures.append(f"场景{label}: 有效距离判据不等价 {r['key']} eff={ed} orig={od} ks={ks}")
        probe = next((r for r in rows if r["domain"] == "duihua"), None)
        if probe is not None and probe["orig_dist"] == 2:
            print(f"    → 关键差异: duihua 域 原生 dist=2 > 插件基准 1，插件本会放弃驻留；"
                  f"TDS 八卦加权 ks=3 使其被保护 = {'是' if probe['kept'] else '否'}"
                  f"（有效距离 {probe['eff_dist']} <= 1）")

    # ---- 场景 C：对比"桥接关闭"的原插件行为 ----
    print(f"\n[5] 场景 C · 桥接关闭后（应完全退回原插件行为）")
    bridge.detach()
    rows_off = run_case(smart_offload, mm, bridge, base_keep_steps=1, load_events=3)
    explain(rows_off, base=1)
    for r in rows_off:
        od = r["orig_dist"]
        expected = od is not None and od <= 1
        if r["kept"] != expected:
            failures.append(f"场景C(回退): {r['key']} 期望{'保' if expected else '不保'}，"
                            f"实际{'保' if r['kept'] else '不保'}")
    if rows_off and any(r["kept"] for r in rows_off if r["domain"] == "duihua"):
        failures.append("场景C(回退): 桥接关闭后 duihua 域仍被保护，说明回退不彻底")
    if smart_offload._next_use_distance.__name__ != "_next_use_distance":
        failures.append("回退不彻底：_next_use_distance 仍为桥接版本")
    bridge.attach(smart_offload)     # 复原，供后续场景使用

    # ---- 场景 D：等价性全量扫描（base=1/2/3，dist=-2..10） ----
    print(f"\n[6] 场景 D · 等价性全量扫描：dist_eff <= base  ⟺  dist <= ks_domain")
    keys = ["CheckpointLoaderSimple|flux1-dev.safetensors",
            "CLIPLoader|t5xxl.safetensors",
            "WanVideoModelLoader|wan2.1_t2v.safetensors",
            "Unknown|mystery.bin"]
    checked = 0
    for base in (1, 2, 3):
        for key in keys:
            ks = BAGUA_KEEP_STEPS.get(domain_of_key(key), 1)
            for dist in range(-2, 11):
                dist_eff = bridge.effective_distance(dist, key, base=base)
                if (dist_eff <= base) != (dist <= ks):
                    failures.append(f"等价性破坏: base={base} key={key} dist={dist} ks={ks} eff={dist_eff}")
                checked += 1
    print(f"    扫描 {checked} 组 (base × key × dist)，不一致 {sum(1 for f in failures if '等价性' in f)} 组")
    print(f"    桥接统计: {bridge.snapshot()['stats']}")

    print("\n" + "=" * 96)
    if failures:
        print(f"[FAIL] 共 {len(failures)} 项不一致：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[OK] 真实插件源码挂载验证通过：TDS 八卦加权 + 四象压力策略已正确接管驻留决策，"
          "且回退后与原插件行为一致。")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
