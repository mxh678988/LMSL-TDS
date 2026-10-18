"""tds.registry 命令行工具。

用法（在含 tds 包的路径下执行）
------------------------------
    python -m tds.registry.cli hw                        # 整机画像
    python -m tds.registry.cli endpoint                  # 镜像端点连通性
    python -m tds.registry.cli list [domain]             # 目录清单
    python -m tds.registry.cli plan <key> [--probe] [--ctx N] [--json]
    python -m tds.registry.cli rank [domain] [--probe] [--ctx N]
    python -m tds.registry.cli pull <key> --to <dir> [--probe] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import catalog as _catalog
from . import planner as _planner
from .downloader import download_spec
from .hardware import probe_hardware
from .remote import probe_endpoint, resolve_endpoint

_MARK = {"ok": "[OK]  ", "tight": "[TIGHT]", "oom": "[OOM]", "unknown": "[?]  "}


def _print_hw() -> int:
    profile = probe_hardware(refresh=True)
    print("OS        : {}".format(profile.os_name))
    print("Python    : {}".format(profile.python_version))
    print("CPU cores : {}".format(profile.cpu_count))
    print("RAM       : {} MB".format(profile.ram_mb or "?"))
    print("Device    : {}".format(profile.summary()))
    print("Source    : {}".format(profile.source))
    print("Budget    : {:.0f} MB（保守可用，已扣预留）".format(_planner.budget_mb(hw=profile)))
    return 0


def _print_endpoint() -> int:
    print("Default endpoint : {}".format(resolve_endpoint()))
    print("Env override     : TDS_HF_ENDPOINT")
    print("Probe            : {}".format(json.dumps(probe_endpoint(), ensure_ascii=False)))
    return 0


def _print_list(domain: Optional[str]) -> int:
    rows = _catalog.list_entries(domain=domain)
    if not rows:
        print("（无条目）")
        return 1
    print("{:<34} {:<14} {:<10} {}".format("KEY", "DOMAIN", "VERIFIED", "REPO"))
    for spec in rows:
        print(
            "{:<34} {:<14} {:<10} {}".format(
                spec.key, spec.domain, "yes" if spec.verified else "no", spec.repo
            )
        )
    return 0


def _print_plan(key: str, probe: bool, ctx: Optional[int], as_json: bool) -> int:
    result = _planner.plan(key, ctx_tokens=ctx, probe=probe)
    if as_json:
        payload = {k: v for k, v in result.items() if k != "spec"}
        payload["spec"] = result["spec"].to_dict()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    fp, vd = result["footprint"], result["verdict"]
    print("条目     : {}".format(result["key"]))
    print("域       : {}".format(result["domain"]))
    print("硬件     : {}".format(result["hardware"]))
    print("可用预算 : {:.0f} MB".format(result["budget_mb"]))
    print(
        "显存画像 : 权重 {weights_mb} + KV {kv_mb} + 工作区 {workspace_mb} "
        "= {total_mb} MB（依据 {basis}{partial}）".format(
            partial="" if fp.get("complete") else "·部分",
            **fp,
        )
    )
    print("结论     : {} {}（余量 {:.0f} MB）".format(
        _MARK.get(vd["level"], vd["level"]), vd["reason"], vd["headroom_mb"]
    ))
    return 0 if vd["fits"] else 2


def _print_rank(domain: Optional[str], probe: bool, ctx: Optional[int]) -> int:
    rows = _planner.rank(domain=domain, ctx_tokens=ctx, probe=probe)
    if not rows:
        print("（无条目）")
        return 1
    print("{:<34} {:<14} {:>12} {}".format("KEY", "DOMAIN", "TOTAL(MB)", "VERDICT"))
    for row in rows:
        fp, vd = row["footprint"], row["verdict"]
        total = fp.get("total_mb", 0.0)
        print(
            "{:<34} {:<14} {:>12} {} {}".format(
                row["key"], row["domain"], "{:.0f}".format(total) if total else "-",
                _MARK.get(vd["level"], vd["level"]), vd["reason"],
            )
        )
    return 0


def _do_pull(key: str, dest: str, probe: bool, force: bool) -> int:
    spec = _catalog.get(key)
    if probe:
        spec = _catalog.probe_entry(spec)
        _catalog.register(spec, override=True)

    def _progress(name: str, done: int, total: int) -> None:
        pct = "{:.1f}%".format(done * 100.0 / total) if total else "?"
        sys.stderr.write("\r  {} {} / {} ({})".format(name, done, total or "?", pct))
        sys.stderr.flush()

    print("下载 {} → {}".format(spec.key, dest))
    print("  端点: {}".format(resolve_endpoint()))
    paths = download_spec(spec, dest, progress=_progress, force=force)
    sys.stderr.write("\n")
    for name, path in paths.items():
        print("  {} → {}".format(name, path))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="tds.registry", description="TDS 模型获取与显存预判")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("hw", help="打印整机画像与保守可用预算")
    sub.add_parser("endpoint", help="打印镜像端点并做连通性自检")

    p_list = sub.add_parser("list", help="列出目录条目")
    p_list.add_argument("domain", nargs="?", choices=list(_catalog.DOMAINS))

    p_plan = sub.add_parser("plan", help="对单个条目做加载可行性判定")
    p_plan.add_argument("key")
    p_plan.add_argument("--probe", action="store_true", help="先向远端校验并回填体积")
    p_plan.add_argument("--ctx", type=int, default=None, help="上下文长度（覆盖条目默认）")
    p_plan.add_argument("--json", action="store_true", help="以 JSON 输出")

    p_rank = sub.add_parser("rank", help="按域对候选排序")
    p_rank.add_argument("domain", nargs="?", choices=list(_catalog.DOMAINS))
    p_rank.add_argument("--probe", action="store_true")
    p_rank.add_argument("--ctx", type=int, default=None)

    p_pull = sub.add_parser("pull", help="下载条目声明的全部文件")
    p_pull.add_argument("key")
    p_pull.add_argument("--to", required=True, help="目标目录")
    p_pull.add_argument("--probe", action="store_true", help="下载前校验并回填体积")
    p_pull.add_argument("--force", action="store_true", help="忽略已存在文件强制重下")

    args = parser.parse_args(argv)
    try:
        if args.cmd == "hw":
            return _print_hw()
        if args.cmd == "endpoint":
            return _print_endpoint()
        if args.cmd == "list":
            return _print_list(args.domain)
        if args.cmd == "plan":
            return _print_plan(args.key, args.probe, args.ctx, args.json)
        if args.cmd == "rank":
            return _print_rank(args.domain, args.probe, args.ctx)
        if args.cmd == "pull":
            return _do_pull(args.key, args.to, args.probe, args.force)
    except Exception as exc:  # CLI 兜底：异常转为可读错误
        print("错误：{}".format(exc), file=sys.stderr)
        return 1
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
