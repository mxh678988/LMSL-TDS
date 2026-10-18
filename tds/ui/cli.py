"""tds.ui 命令行入口。

用法（在含 tds 包的路径下执行）
------------------------------
    python -m tds.ui.cli serve [--port 8770] [--backend URL] [--model NAME] [--open]
    python -m tds.ui.cli probe [--backend URL]
    python -m tds.ui.cli bench [--suite full] [--models A,B] [--out DIR] [--list]
                                [--repeat N] [--temperature 0]

``serve`` 启动对话测试台（纯标准库 HTTP 服务 + 单文件面板）；
``probe`` 只做后端连通性与模型清单自检，不启动服务；
``bench`` 跑内置用例集的批量回归并产出 Markdown/JSON 报告。
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .backend import BackendClient, BackendError, ModelInfo
from .bench import load_suite, render_markdown, run_bench, write_report
from .server import DEFAULT_BACKEND, UiServer


def _cmd_probe(backend: str, api_key: str) -> int:
    client = BackendClient(backend, api_key)
    info = client.ping()
    print("后端     : {}".format(client.base_url))
    print("状态     : {}".format(
        "在线（{} {}）".format(info["kind"], info["version"]) if info["online"] else "离线"
    ))
    if not info["online"]:
        print("错误     : {}".format(info["error"]))
        return 1
    models = client.list_models()
    print("模型数   : {}".format(len(models)))
    for model in models:
        print("  - {}".format(model.label()))
    return 0


def _select_models(
    client: BackendClient, spec: str, limit: int, include_remote: bool = False
) -> List[str]:
    """确定参与回归的模型：显式指定优先，否则取在线清单（本地优先）。"""
    explicit = [item.strip() for item in (spec or "").split(",") if item.strip()]
    if explicit:
        return explicit
    models: List[ModelInfo] = client.list_models()
    pool = models if include_remote else ([m for m in models if not m.remote] or models)
    names: List[str] = []
    for model in pool:  # 去重保序：同一模型可能在后端清单里重复登记
        if model.name not in names:
            names.append(model.name)
    return names[:limit] if limit > 0 else names


def _cmd_bench(args: argparse.Namespace) -> int:
    suite_name, cases = load_suite(args.suite)
    if args.list:
        print("套件 {}：{} 条用例".format(suite_name, len(cases)))
        for case in cases:
            head = (case.prompt or "").splitlines()[0] if case.prompt else ""
            print("  - {:<14} {:<6} {}{}".format(
                case.id, case.kind, head[:46], "  [图] {}".format(case.fixture) if case.fixture else ""
            ))
        return 0

    client = BackendClient(args.backend, args.api_key)
    info = client.ping()
    if not info["online"]:
        print("后端离线：{}".format(info["error"]), file=sys.stderr)
        return 1
    try:
        models = _select_models(client, args.models, args.limit_models, args.include_remote)
    except BackendError as exc:
        print("获取模型清单失败：{}".format(exc), file=sys.stderr)
        return 1
    if not models:
        print("没有可用模型，请用 --models 显式指定", file=sys.stderr)
        return 1

    print("后端   : {}".format(client.base_url))
    print("套件   : {}（{} 条用例）".format(suite_name, len(cases)))
    print("模型   : {}".format("、".join(models)))
    print("重复   : {} 轮".format(max(1, args.repeat)))
    if args.temperature is not None:
        print("温度   : 统一覆盖为 {}".format(args.temperature))
    print("-" * 78)

    try:
        report = run_bench(
            client,
            args.suite,
            models,
            repeat=max(1, args.repeat),
            skip_vision_for_text_models=not args.vision_on_text,
            verbose=not args.quiet,
            temperature=args.temperature,
        )
    except BackendError as exc:
        print("回归中断：{}".format(exc), file=sys.stderr)
        return 1

    summary = report.get("summary", {})
    print("-" * 78)
    print("{:<26}{:>6}{:>8}{:>6}{:>6}{:>14}{:>16}".format(
        "模型", "通过", "未命中", "报错", "跳过", "平均首包ms", "平均总耗时ms"
    ))
    for model in models:
        row = summary.get(model, {})
        print("{:<26}{:>6}{:>8}{:>6}{:>6}{:>14}{:>16}".format(
            model[:26],
            row.get("pass", 0),
            row.get("fail", 0),
            row.get("error", 0),
            row.get("skip", 0),
            row.get("avg_first_ms", 0.0),
            row.get("avg_total_ms", 0.0),
        ))

    if not args.no_write:
        paths = write_report(report, args.out)
        print("报告   : {}".format(paths["markdown"]))
        print("数据   : {}".format(paths["json"]))
    if args.show:
        print()
        print(render_markdown(report))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    server = UiServer(
        host=args.host,
        port=args.port,
        backend_url=args.backend,
        api_key=args.api_key,
        default_model=args.model,
        verbose=True,
    )
    if args.scheduler:
        try:
            from ..core.scheduler import Scheduler

            server.scheduler = Scheduler()
            print('调度器已挂载，对话完成后将展示决策轨迹')
        except Exception as exc:
            print('调度器挂载失败（不影响测试台）：{}'.format(exc), file=sys.stderr)
    if args.open:
        import threading
        import webbrowser

        httpd = server.build()
        url = "http://{}:{}/".format(args.host, httpd.server_address[1])
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        server.verbose = True
        print("TDS 对话测试台已启动")
        print("  面板地址 : {}".format(url))
        print("  模型后端 : {}".format(server.backend_url))
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()
        return 0
    server.serve_forever()
    return 0

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="tds.ui", description="TDS 对话交互测试台（知识分支）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="启动对话测试台")
    p_serve.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机）")
    p_serve.add_argument("--port", type=int, default=8770, help="监听端口，0 表示随机空闲端口")
    p_serve.add_argument("--backend", default=DEFAULT_BACKEND, help="模型后端根地址")
    p_serve.add_argument("--api-key", default="", help="后端鉴权口令（本地服务通常留空）")
    p_serve.add_argument("--model", default="", help="新会话默认模型")
    p_serve.add_argument("--open", action="store_true", help="启动后自动打开系统浏览器")
    p_serve.add_argument("--scheduler", action="store_true", help="启用调度器（决策轨迹视图）")

    p_probe = sub.add_parser("probe", help="后端连通性与模型清单自检")
    p_probe.add_argument("--backend", default=DEFAULT_BACKEND)
    p_probe.add_argument("--api-key", default="")

    p_bench = sub.add_parser("bench", help="固定用例批量回归并产出报告")
    p_bench.add_argument("--suite", default="full", help="内置套件 basic/vision/full，或 .json 套件文件路径")
    p_bench.add_argument("--models", default="", help="逗号分隔的模型名；留空自动取在线清单（本地优先）")
    p_bench.add_argument("--limit-models", type=int, default=2, help="自动选型时的模型上限，0 表示不限制")
    p_bench.add_argument("--include-remote", action="store_true", help="自动选型时纳入云端模型")
    p_bench.add_argument("--repeat", type=int, default=1, help="每个用例重复轮次（取最优结果）")
    p_bench.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="统一覆盖用例自带温度；回归建议用 0 压到确定状态",
    )
    p_bench.add_argument("--vision-on-text", action="store_true", help="视觉用例也对非多模态模型执行")
    p_bench.add_argument("--backend", default=DEFAULT_BACKEND)
    p_bench.add_argument("--api-key", default="")
    p_bench.add_argument("--out", default="bench_out", help="报告输出目录")
    p_bench.add_argument("--no-write", action="store_true", help="只在终端输出，不落盘报告")
    p_bench.add_argument("--show", action="store_true", help="执行后把 Markdown 报告打到终端")
    p_bench.add_argument("--quiet", action="store_true", help="不打印逐条进度")
    p_bench.add_argument("--list", action="store_true", help="只列出套件用例，不执行回归")

    args = parser.parse_args(argv)
    try:
        if args.cmd == "serve":
            return _cmd_serve(args)
        if args.cmd == "probe":
            return _cmd_probe(args.backend, args.api_key)
        if args.cmd == "bench":
            return _cmd_bench(args)
    except Exception as exc:
        print("错误：{}".format(exc), file=sys.stderr)
        return 1
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())