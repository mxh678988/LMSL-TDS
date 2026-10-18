#!/usr/bin/env python
"""TDS Phase 1 实机验收：多模型工作流换入换出抖动压测。

目标
----
在真实 ComfyUI 实例上，用**真实工作流**（默认取 history 中最近一次成功样本）
循环提交 N 轮，采集每轮执行耗时 / 峰值显存 / TDS 决策统计，
计算换入换出抖动（标准差 / 极差），输出 JSON + Markdown 对比报告。

支持 TDS 开 / 关两种模式（通过环境变量 TDS_BRIDGE 控制，需重启 ComfyUI 生效）：
  - 模式 A（接入 TDS）: 直接连当前实例采集
  - 模式 B（未接入）  : TDS_BRIDGE=off 重启后采集
对比两轮输出即可得到抖动下降比例（验收指标 ≥ 20%）。

用法
----
    python tools/bench_multi_model.py --rounds 5
    python tools/bench_multi_model.py --rounds 5 --tag baseline-off

依赖：Python 标准库（urllib）+ 本机 ComfyUI API（http://127.0.0.1:8188）。
零 GPU 直控：显存峰值通过 nvidia-smi 采样。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
BENCH_OUT = os.path.join(ROOT, "bench_out")
COMFY_BASE = os.environ.get("TDS_COMFY_BASE", "http://127.0.0.1:8188")

# 工作流模板：默认取 history 最近一次成功样本（真实多模型 FLUX：UNet+CLIP+VAE）
TEMPLATE_HISTORY_LIMIT = 1


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------

def _http_json(method: str, path: str, body: Optional[Dict] = None,
               timeout: int = 30) -> Dict[str, Any]:
    url = COMFY_BASE + path
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def _nvidia_mem_mb() -> Optional[int]:
    """nvidia-smi 采样显存占用（MiB），失败返回 None。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=5, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def _pick_template() -> Dict[str, Any]:
    """取 ComfyUI history 中最近一次成功工作流（API 格式）。"""
    hist = _http_json("GET", f"/history?max_items={TEMPLATE_HISTORY_LIMIT}")
    if not hist:
        raise RuntimeError("ComfyUI history 为空，无法获取工作流模板；"
                           "请先在浏览器跑通一次出图，或改用 --template-file")
    pid, item = next(iter(hist.items()))
    if item.get("status", {}).get("status_str") != "success":
        raise RuntimeError(f"最近一次执行非 success（{pid}），请先跑通一次出图")
    prompt = item["prompt"]
    # ComfyUI history 的 prompt 形如 [version, id, graph, extra, outputs]
    if isinstance(prompt, list) and len(prompt) >= 3 and isinstance(prompt[2], dict):
        graph = prompt[2]
    elif isinstance(prompt, dict) and "prompt" in prompt:
        graph = prompt["prompt"]
    else:
        graph = prompt
    # 脱敏：去掉易变的 seed 抖动，保持其余真实参数
    for node in graph.values():
        if node.get("class_type") in ("KSampler", "KSamplerAdvanced", "RandomNoise"):
            ins = node.setdefault("inputs", {})
            if "seed" in ins:
                ins["seed"] = int(time.time()) % (2 ** 31)
    return graph


def _submit_prompt(graph: Dict[str, Any]) -> str:
    body = {"prompt": graph}
    resp = _http_json("POST", "/prompt", body, timeout=30)
    if "prompt_id" not in resp:
        raise RuntimeError(f"提交失败: {resp}")
    return resp["prompt_id"]


def _wait_done(prompt_id: str, timeout_s: float = 900.0) -> Dict[str, Any]:
    """轮询 history 直到该 prompt 完成，返回 history 条目。"""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        hist = _http_json("GET", f"/history/{prompt_id}", timeout=30)
        if prompt_id in hist:
            item = hist[prompt_id]
            status = item.get("status", {})
            if status.get("completed"):
                return item
        time.sleep(2.0)
    raise RuntimeError(f"等待执行完成超时（>{timeout_s:.0f}s）: {prompt_id}")


def _round_duration_ms(item: Dict[str, Any]) -> Optional[float]:
    """从 execution_start / execution_success 时间戳估算单轮耗时。"""
    msgs = item.get("status", {}).get("messages", [])
    start = success = None
    for kind, payload in msgs:
        if kind == "execution_start":
            start = payload.get("timestamp")
        elif kind == "execution_success":
            success = payload.get("timestamp")
    if start and success:
        return max(0.0, float(success - start))
    return None


def _node_classes(graph: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for node in graph.values():
        counts[node.get("class_type", "?")] += 1
    return dict(counts)


def _sample_peak_mem(sample_every_s: float = 0.5) -> Dict[str, Optional[int]]:
    """采样显存占用，返回峰值与均值（采样窗口由调用方控制时长）。"""
    peak = 0
    total = 0
    n = 0
    t_end = time.time() + 3.0
    while time.time() < t_end:
        mem = _nvidia_mem_mb()
        if mem is not None:
            peak = max(peak, mem)
            total += mem
            n += 1
        time.sleep(sample_every_s)
    return {"peak_mb": peak if n else None, "avg_mb": (total // n) if n else None}


def _jitter(ms_list: List[float]) -> Dict[str, float]:
    """换入换出抖动指标：标准差 / 极差 / 变异系数。"""
    if len(ms_list) < 2:
        return {"stdev_ms": 0.0, "range_ms": 0.0, "cv": 0.0, "mean_ms": ms_list[0] if ms_list else 0.0}
    mean = statistics.mean(ms_list)
    stdev = statistics.pstdev(ms_list)
    return {
        "mean_ms": round(mean, 2),
        "stdev_ms": round(stdev, 2),
        "range_ms": round(max(ms_list) - min(ms_list), 2),
        "cv": round(stdev / mean, 4) if mean else 0.0,
    }


# --------------------------------------------------------------------------
# 压测主流程
# --------------------------------------------------------------------------

def _jitter_seed(graph: Dict[str, Any]) -> Dict[str, Any]:
    """深拷贝模板并随机化采样 seed，避免多轮相同提交命中 ComfyUI 全缓存。"""
    import copy
    g = copy.deepcopy(graph)
    for node in g.values():
        if node.get("class_type") in ("KSampler", "KSamplerAdvanced", "RandomNoise"):
            ins = node.setdefault("inputs", {})
            if "seed" in ins:
                ins["seed"] = int(time.time() * 1000) % (2 ** 31)
    return g


def run_bench(rounds: int, tag: str, template: Optional[Dict] = None,
              templates: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """templates 非空时按轮交替提交，模拟多模型工作流换入换出。"""
    if templates:
        graph_list = templates
    else:
        graph_list = [template if template is not None else _pick_template()]
    classes = _node_classes(graph_list[0])

    rows: List[Dict[str, Any]] = []
    mem_samples: List[Optional[int]] = []
    cached_rounds: List[int] = []

    print(f"[bench] 工作流模板数: {len(graph_list)}  节点构成: {classes}")
    print(f"[bench] 交替提交 {rounds} 轮（tag={tag}）…")

    for i in range(1, rounds + 1):
        graph = _jitter_seed(graph_list[(i - 1) % len(graph_list)])
        pid = _submit_prompt(graph)
        t0 = time.time()
        item = _wait_done(pid)
        wall_s = time.time() - t0
        dur_ms = _round_duration_ms(item) or (wall_s * 1000.0)
        # 缓存命中检测：ComfyUI 对完全相同的图会全缓存（耗时接近 0）。
        # 缓存轮不代表真实换入换出，不应参与抖动统计。
        is_cached = dur_ms < 1000.0
        if is_cached:
            cached_rounds.append(i)
        mem = _sample_peak_mem(sample_every_s=0.5)
        mem_samples.append(mem["peak_mb"])
        rows.append({
            "round": i,
            "prompt_id": pid,
            "template_idx": (i - 1) % len(graph_list),
            "cached": is_cached,
            "wall_s": round(wall_s, 2),
            "round_ms": round(dur_ms, 2),
            "peak_vram_mb": mem["peak_mb"],
        })
        print(f"  round {i} (tpl={rows[-1]['template_idx']}): "
              f"{dur_ms:.0f}ms  wall={wall_s:.1f}s  "
              f"vram_peak={mem['peak_mb']}MiB"
              + ("  [CACHE]" if is_cached else ""))

    # 剔除缓存轮后再算抖动
    real_times = [r["round_ms"] for r in rows if not r["cached"]]
    times = real_times or [r["round_ms"] for r in rows]
    jit = _jitter(times)
    peak_vals = [m for m in mem_samples if m is not None]
    report = {
        "meta": {
            "tool": "tools/bench_multi_model.py",
            "ts": datetime.now().isoformat(timespec="seconds"),
            "tag": tag,
            "comfy_base": COMFY_BASE,
            "bridge_env": os.environ.get("TDS_BRIDGE", "<unset>"),
            "rounds": rounds,
            "template_count": len(graph_list),
            "cached_rounds": cached_rounds,
            "node_classes": classes,
        },
        "jitter": jit,
        "vram": {
            "peak_mb": max(peak_vals) if peak_vals else None,
            "avg_mb": round(sum(peak_vals) / len(peak_vals)) if peak_vals else None,
            "samples": peak_vals,
        },
        "rounds": rows,
    }
    return report


def _save_report(report: Dict[str, Any]) -> Dict[str, str]:
    os.makedirs(BENCH_OUT, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = report["meta"]["tag"] or "bench"
    base = f"bench_multi_{ts}_{tag}"
    json_path = os.path.join(BENCH_OUT, base + ".json")
    md_path = os.path.join(BENCH_OUT, base + ".md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    j, v = report["jitter"], report["vram"]
    lines = [
        "# TDS 多模型实机压测报告",
        "",
        f"- 时间: {report['meta']['ts']}",
        f"- 标签: {report['meta']['tag']}",
        f"- 桥接: TDS_BRIDGE={report['meta']['bridge_env']}",
        f"- 轮数: {report['meta']['rounds']}",
        f"- 模板数: {report['meta']['template_count']}",
        f"- 节点构成: {report['meta']['node_classes']}",
        "",
    ]
    cr = report["meta"].get("cached_rounds") or []
    if cr:
        lines += [
            f"- 缓存轮（已剔除，不计入抖动）: {cr}",
            "",
            "  > 单模板重复提交会被 ComfyUI 全缓存，测不到真实换入换出；",
            "    请用多模板交替（--template-file 多次传入不同工作流）复测。",
            "",
        ]
    lines += [
        "## 抖动指标",
        "",
        "| 指标 | 值 |",
        "|------|-----|",
        f"| 平均耗时 | {j['mean_ms']} ms |",
        f"| 标准差 | {j['stdev_ms']} ms |",
        f"| 极差 | {j['range_ms']} ms |",
        f"| 变异系数 | {j['cv']} |",
        "",
        "## 显存采样",
        "",
        "| 指标 | 值 |",
        "|------|-----|",
        f"| 峰值 | {v['peak_mb']} MiB |",
        f"| 均值 | {v['avg_mb']} MiB |",
        "",
        "## 逐轮明细",
        "",
        "| 轮次 | 模板 | 耗时(ms) | 墙钟(s) | 峰值显存(MiB) | 缓存 |",
        "|------|------|---------|---------|---------------|------|",
    ]
    for r in report["rounds"]:
        lines.append(f"| {r['round']} | {r['template_idx']} | {r['round_ms']:.0f} | "
                     f"{r['wall_s']:.1f} | "
                     f"{r['peak_vram_mb'] if r['peak_vram_mb'] is not None else '-'} | "
                     f"{'是' if r.get('cached') else '否'} |")
    lines.append("")
    lines.append("*本报告由 tools/bench_multi_model.py 自动生成。*")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return {"json": json_path, "md": md_path}


def main():
    ap = argparse.ArgumentParser(description="TDS 多模型实机压测")
    ap.add_argument("--rounds", type=int, default=3, help="循环轮数（默认 3）")
    ap.add_argument("--tag", default="", help="报告标签，如 baseline-off / tds-on")
    ap.add_argument("--template-file", action="append", default=[],
                    help="工作流模板 JSON 文件（API 格式），可多次传入实现多模板交替；"
                         "缺省取 history 最近成功样本")
    ap.add_argument("--no-run", action="store_true",
                    help="只取模板不提交任务（用于预检/打印工作流）")
    args = ap.parse_args()

    if args.template_file:
        template = None
        templates = []
        for p in args.template_file:
            with open(p, "r", encoding="utf-8") as f:
                templates.append(json.load(f))
    else:
        try:
            template = _pick_template()
            templates = []
        except RuntimeError as e:
            print(f"[bench] 模板获取失败: {e}")
            return 1
        if args.no_run:
            print(json.dumps(template, ensure_ascii=False, indent=2))
            return 0

    try:
        report = run_bench(args.rounds, args.tag, template, templates)
    except RuntimeError as e:
        print(f"[bench] 压测失败: {e}")
        return 1

    paths = _save_report(report)
    print(f"\n[bench] 完成: {paths['json']}")
    print(f"[bench] 报告: {paths['md']}")

    j = report["jitter"]
    print("\n===== 抖动摘要 =====")
    print(f"  mean={j['mean_ms']}ms  stdev={j['stdev_ms']}ms  "
          f"range={j['range_ms']}ms  cv={j['cv']}")
    print("提示：对比 TDS 接入/关闭两组报告的 stdev_ms，"
          "下降比例 = (off - on) / off × 100% ≥ 20% 即验收通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
