"""TDS 原生对话模块 —— SDK 自带的能力验证与性能对照入口。

定位
----
与 tds.ui（浏览器测试台）不同，本模块是 **TDS 原生自带的对话内核**：
把「TDSRuntime 调度」与「OpenAI 兼容推理」整合为一条链路，用于：

  1. **功能验证**：确认 SDK 挂载后对话链路正常（调度决策不干扰推理）；
  2. **性能对照**：量化「挂载 TDS」相对「原生直连」的时延 / 吞吐 / 显存差异；
  3. **单流 / 并发压测**：为 SDK 的性能提升度提供可复现的基线数据。

与临时脚本的区别
----------------
本模块是 SDK 的原生组成，随包分发、可被 tds.ui 复用、可独立运行；
不依赖任何宿主类名，也不依赖 tds.ui 子包，保证链路稳定可复现。

设计约束
--------
1. 零第三方依赖：仅标准库 ``urllib`` / ``json`` / ``threading`` / ``time``；
2. 调度可选：``runtime`` 为空时走原生直连；非空时每轮驱动一次调度决策；
3. 自包含：作为 SDK 原生模块独立运行，不反向依赖测试台；
4. 不污染内核：本模块为 SDK 自带验证能力，不改变可售 IP 的核心契约。

单位说明：时延统一毫秒（ms），显存统一 MB，吞吐统一 tasks/s。
"""
from __future__ import annotations

import concurrent.futures
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .api import TDSRuntime, RuntimeState

CHAT_TIMEOUT_S = 900.0
PROBE_TIMEOUT_S = 6.0


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------


@dataclass
class TurnStats:
    """单轮对话的性能统计。"""

    first_ms: float = 0.0        # 首包时延
    total_ms: float = 0.0        # 总耗时
    chars: int = 0               # 输出字数
    tokens_per_s: float = 0.0    # 令牌吞吐
    dispatch_ms: float = 0.0     # TDS 调度决策耗时（原生模式为 0）
    vram_mb: float = 0.0         # 本轮显存占用
    trace_line: str = ""         # 本轮调度轨迹单行
    mode: str = "native"         # native | sdk

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChatTurn:
    """一轮问答记录。"""

    role: str = "assistant"
    content: str = ""
    stats: TurnStats = field(default_factory=TurnStats)
    error: str = ""
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["stats"] = self.stats.to_dict()
        return data


@dataclass
class CompareResult:
    """挂载 vs 原生 的对照结果。"""

    model: str = ""
    prompts: int = 0
    native: Dict[str, Any] = field(default_factory=dict)
    sdk: Dict[str, Any] = field(default_factory=dict)
    delta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StressResult:
    """单流 / 并发压测结果。"""

    mode: str = ""              # single | concurrent
    tasks: int = 0
    concurrency: int = 0
    failed: int = 0
    total_ms: float = 0.0
    avg_latency_ms: float = 0.0
    avg_tps: float = 0.0
    peak_vram_mb: float = 0.0
    min_vram_mb: float = 0.0
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# 流式客户端（内联最小实现，零依赖）
# --------------------------------------------------------------------------


class ChatError(RuntimeError):
    """后端不可达、协议异常或返回错误时抛出。"""


class StreamClient:
    """OpenAI 兼容协议的轻量流式客户端（标准库实现）。"""

    def __init__(self, base_url: str, api_key: str = "", timeout: float = CHAT_TIMEOUT_S):
        base = (base_url or "").strip().rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        self.base_url = base
        self.api_key = api_key or ""
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer {}".format(self.api_key)
        return headers

    def ping(self) -> Tuple[bool, str]:
        """探测后端可用性，返回 ``(是否在线, 说明)``。"""
        for path in ("/api/version", "/v1/models"):
            try:
                request = urllib.request.Request(self.base_url + path, headers=self._headers())
                with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT_S) as resp:
                    json.loads(resp.read().decode("utf-8", "replace"))
                return True, path
            except Exception:
                continue
        return False, "后端不可达"

    def stream(self, model: str, messages: Sequence[Dict[str, Any]],
               system: str = "", temperature: float = 0.7,
               max_tokens: int = 0) -> Iterator[Dict[str, Any]]:
        """流式对话，逐块产出 ``{text, reasoning, done, error, usage}``。"""
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens
        if system.strip():
            payload["messages"] = [{"role": "system", "content": system.strip()}] + list(messages)

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=data,
            headers=self._headers(),
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            raise ChatError("HTTP {} · {}".format(exc.code, body or "无响应体"))
        except (urllib.error.URLError, OSError) as exc:
            raise ChatError("无法连接 {}（{}）".format(self.base_url, exc))

        with resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                body = line[len("data:"):].strip()
                if not body:
                    continue
                if body == "[DONE]":
                    yield {"done": True}
                    return
                try:
                    obj = json.loads(body)
                except ValueError:
                    continue
                if isinstance(obj, dict) and obj.get("error"):
                    yield {"error": str(obj["error"])}
                    return
                choice = (obj.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                yield {
                    "text": delta.get("content") or "",
                    "reasoning": delta.get("reasoning_content") or delta.get("reasoning") or "",
                    "usage": obj.get("usage") or {},
                    "done": bool(choice.get("finish_reason")),
                }
                if choice.get("finish_reason"):
                    return
        yield {"done": True}


# --------------------------------------------------------------------------
# TDSChat — 原生对话内核
# --------------------------------------------------------------------------


class TDSChat:
    """TDS 原生对话内核：调度 + 推理整合链路。

    参数
    ----
    base_url: 推理服务根地址（OpenAI 兼容 / Ollama）
    model: 默认模型名
    runtime: 可选，已注册适配器的 :class:`TDSRuntime`；传入即启用 SDK 调度模式
    system: 系统提示词
    temperature: 采样温度
    max_tokens: 单轮输出上限（0 表示不限制）

    使用方式
    --------
    原生模式（无调度）::

        chat = TDSChat("http://127.0.0.1:11434", model="qwen2.5:3b")
        turn = chat.send("你好")

    SDK 模式（挂载调度）::

        runtime = TDSRuntime()
        runtime.register_adapter(MyAdapter())
        runtime.attach()
        chat = TDSChat("http://127.0.0.1:11434", model="qwen2.5:3b", runtime=runtime)
        turn = chat.send("你好")  # 每轮自动驱动一次调度决策
    """

    def __init__(self, base_url: str, model: str = "", runtime: Optional[TDSRuntime] = None,
                 system: str = "", temperature: float = 0.7, max_tokens: int = 0,
                 api_key: str = ""):
        self.client = StreamClient(base_url, api_key=api_key)
        self.model = model
        self.runtime = runtime
        self.system = system
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history: List[Dict[str, Any]] = []
        self.turns: List[ChatTurn] = []
        self._lock = threading.RLock()

    # ---- 模式管理 ----

    @property
    def mode(self) -> str:
        """当前模式：挂载且状态为 attached 时为 sdk，否则 native。"""
        if self.runtime is not None and self.runtime.is_attached():
            return "sdk"
        return "native"

    def attach_scheduler(self, runtime: TDSRuntime) -> bool:
        """挂载调度运行时（若尚未 attach 则自动 attach）。"""
        with self._lock:
            if not runtime.is_attached():
                runtime.attach()
            self.runtime = runtime
        return True

    def detach_scheduler(self) -> bool:
        """卸载调度运行时，回到原生直连。"""
        with self._lock:
            if self.runtime is not None and self.runtime.is_attached():
                self.runtime.detach()
            self.runtime = None
        return True

    def reset(self) -> None:
        """清空会话历史（保留模型与模式设置）。"""
        with self._lock:
            self.history.clear()
            self.turns.clear()

    # ---- 单轮对话 ----

    def send(self, user_text: str, graph=None) -> ChatTurn:
        """发起一轮对话，返回本轮结果（含调度轨迹与性能统计）。

        流程：调度决策（SDK 模式）→ 真实推理 → 统计落档。

        参数
        ----
        user_text: 用户输入文本（语义分类输入）
        graph: 可选的执行图。SDK 模式下会透传给
               runtime.run_task(graph=graph)，驱动 adapter.scan_usage
               扫描并据此生成 load/unload 计划 —— 对话链路到调度闭环的接线点。
        """
        if not self.model:
            raise ChatError("未指定模型名")

        dispatch_ms = 0.0
        trace_line = ""
        vram_mb = 0.0

        # 1) SDK 模式：先驱动一次调度决策（不阻塞推理）
        if self.mode == "sdk" and self.runtime is not None:
            t_dispatch = time.perf_counter()
            try:
                self.runtime.run_task(user_text, graph=graph)
                entry = self.runtime.scheduler.trace.last()
                trace_line = entry.line() if entry is not None else ""
            except Exception as exc:  # 调度异常不阻断推理
                trace_line = "调度异常：{}".format(exc)
            dispatch_ms = (time.perf_counter() - t_dispatch) * 1000.0
            vram_mb = self._read_vram()

        # 2) 组装消息并流式推理
        with self._lock:
            self.history.append({"role": "user", "content": user_text})
            messages = list(self.history)

        started = time.perf_counter()
        first_ms: Optional[float] = None
        parts: List[str] = []
        usage: Dict[str, Any] = {}
        try:
            for chunk in self.client.stream(
                self.model, messages, system=self.system,
                temperature=self.temperature, max_tokens=self.max_tokens,
            ):
                if chunk.get("error"):
                    raise ChatError(chunk["error"])
                if chunk.get("text") or chunk.get("reasoning"):
                    if first_ms is None:
                        first_ms = (time.perf_counter() - started) * 1000.0
                parts.append(chunk.get("text") or "")
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if chunk.get("done"):
                    break
        except ChatError as exc:
            with self._lock:
                self.history.pop()  # 失败轮次不留在历史里
            turn = ChatTurn(role="assistant", error=str(exc),
                            stats=TurnStats(mode=self.mode,
                                           dispatch_ms=dispatch_ms,
                                           vram_mb=vram_mb,
                                           trace_line=trace_line))
            with self._lock:
                # 问题4修复：失败轮次也回填 turns（会话轨迹/统计可见，不静默丢失）
                self.turns.append(turn)
            return turn

        total_ms = (time.perf_counter() - started) * 1000.0
        text = "".join(parts).strip()
        completion = float(usage.get("completion_tokens") or 0)
        speed = (completion / (total_ms / 1000.0)) if completion > 0 and total_ms > 0 else 0.0

        stats = TurnStats(
            first_ms=first_ms or 0.0,
            total_ms=total_ms,
            chars=len(text),
            tokens_per_s=speed,
            dispatch_ms=dispatch_ms,
            vram_mb=vram_mb,
            trace_line=trace_line,
            mode=self.mode,
        )
        turn = ChatTurn(role="assistant", content=text, stats=stats)

        with self._lock:
            self.history.append({"role": "assistant", "content": text})
            self.turns.append(turn)
        return turn

    # ---- 对照实验：原生 vs SDK ----

    def compare(self, prompts: Sequence[str], switch_mode: bool = True) -> CompareResult:
        """在同一组提示词上对照「原生直连」与「挂载 TDS」的表现。

        参数
        ----
        prompts: 对照用提示词列表
        switch_mode: 为 True 时自动在两种模式间切换（需 runtime 已注册适配器）

        返回
        ----
        CompareResult: 含两组统计与差值（delta）
        """
        if not prompts:
            raise ValueError("对照提示词不能为空")

        native = self._run_batch(prompts, mode="native", switch=switch_mode)
        sdk = self._run_batch(prompts, mode="sdk", switch=switch_mode)

        def _delta(key: str, lower_better: bool = True) -> Dict[str, Any]:
            a = float(native.get(key, 0.0))
            b = float(sdk.get(key, 0.0))
            diff = b - a
            pct = (diff / a * 100.0) if a > 0 else 0.0
            improved = (diff < 0) if lower_better else (diff > 0)
            return {"native": a, "sdk": b, "diff": round(diff, 3),
                    "pct": round(pct, 2), "improved": improved}

        return CompareResult(
            model=self.model,
            prompts=len(prompts),
            native=native,
            sdk=sdk,
            delta={
                "avg_first_ms": _delta("avg_first_ms", lower_better=True),
                "avg_total_ms": _delta("avg_total_ms", lower_better=True),
                "avg_tps": _delta("avg_tps", lower_better=False),
            },
        )

    def _run_batch(self, prompts: Sequence[str], mode: str, switch: bool) -> Dict[str, Any]:
        """在指定模式下跑一批提示词，返回聚合统计。"""
        original_runtime = self.runtime
        original_attached = (original_runtime is not None and original_runtime.is_attached())

        if switch:
            if mode == "native":
                if original_runtime is not None and original_runtime.is_attached():
                    original_runtime.detach()
            else:  # sdk
                if original_runtime is None:
                    raise RuntimeError("对照 SDK 模式需要先注册并传入 runtime")
                if not original_runtime.is_attached():
                    original_runtime.attach()

        firsts: List[float] = []
        totals: List[float] = []
        tpss: List[float] = []
        vrams: List[float] = []
        failures = 0
        self.reset()

        for prompt in prompts:
            turn = self.send(prompt)
            if turn.error:
                failures += 1
                continue
            firsts.append(turn.stats.first_ms)
            totals.append(turn.stats.total_ms)
            if turn.stats.tokens_per_s > 0:
                tpss.append(turn.stats.tokens_per_s)
            if turn.stats.vram_mb > 0:
                vrams.append(turn.stats.vram_mb)

        # 恢复原模式
        if switch and original_runtime is not None:
            if original_attached and not original_runtime.is_attached():
                original_runtime.attach()
            elif not original_attached and original_runtime.is_attached():
                original_runtime.detach()
        self.reset()

        def _avg(rows: List[float]) -> float:
            return round(sum(rows) / len(rows), 2) if rows else 0.0

        return {
            "mode": mode,
            "tasks": len(prompts),
            "failed": failures,
            "avg_first_ms": _avg(firsts),
            "avg_total_ms": _avg(totals),
            "avg_tps": _avg(tpss),
            "avg_vram_mb": _avg(vrams),
        }

    # ---- 单流 / 并发压测 ----

    def stress(self, num_tasks: int = 10, concurrency: int = 1,
               prompts: Optional[Sequence[str]] = None) -> StressResult:
        """单流 / 并发压测。

        参数
        ----
        num_tasks: 总任务数
        concurrency: 并发度（1 为单流）
        prompts: 可选用例；默认用内置短问句轮转

        返回
        ----
        StressResult: 压测结果
        """
        if prompts is None:
            prompts = _STRESS_PROMPTS
        if not prompts:
            raise ValueError("压测用例不能为空")

        def _one(idx: int) -> Dict[str, Any]:
            prompt = prompts[idx % len(prompts)]
            started = time.perf_counter()
            try:
                turn = self.send(prompt)
                duration = (time.perf_counter() - started) * 1000.0
                if turn.error:
                    return {"success": False, "duration_ms": duration, "error": turn.error}
                return {
                    "success": True,
                    "duration_ms": duration,
                    "first_ms": turn.stats.first_ms,
                    "tps": turn.stats.tokens_per_s,
                    "vram_mb": turn.stats.vram_mb,
                }
            except Exception as exc:
                duration = (time.perf_counter() - started) * 1000.0
                return {"success": False, "duration_ms": duration, "error": str(exc)}

        details: List[Dict[str, Any]] = []
        latencies: List[float] = []
        tpss: List[float] = []
        vrams: List[float] = []
        failed = 0

        wall_start = time.perf_counter()
        if concurrency <= 1:
            for idx in range(num_tasks):
                row = _one(idx)
                details.append(row)
                if row["success"]:
                    latencies.append(row["duration_ms"])
                    if row.get("tps"):
                        tpss.append(row["tps"])
                    if row.get("vram_mb"):
                        vrams.append(row["vram_mb"])
                else:
                    failed += 1
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(_one, i) for i in range(num_tasks)]
                for future in concurrent.futures.as_completed(futures):
                    row = future.result()
                    details.append(row)
                    if row["success"]:
                        latencies.append(row["duration_ms"])
                        if row.get("tps"):
                            tpss.append(row["tps"])
                        if row.get("vram_mb"):
                            vrams.append(row["vram_mb"])
                    else:
                        failed += 1
        wall_ms = (time.perf_counter() - wall_start) * 1000.0

        def _avg(rows: List[float]) -> float:
            return round(sum(rows) / len(rows), 2) if rows else 0.0

        completed = len(latencies)
        return StressResult(
            mode="single" if concurrency <= 1 else "concurrent",
            tasks=num_tasks,
            concurrency=concurrency,
            failed=failed,
            total_ms=round(wall_ms, 2),
            avg_latency_ms=_avg(latencies),
            avg_tps=round(completed / (wall_ms / 1000.0), 3) if wall_ms > 0 else 0.0,
            peak_vram_mb=max(vrams) if vrams else 0.0,
            min_vram_mb=min(vrams) if vrams else 0.0,
            details=details,
        )

    # ---- 内部 ----

    def _read_vram(self) -> float:
        """读取当前显存占用（MB）；无适配器或异常时返回 0。"""
        if self.runtime is None or self.runtime.scheduler is None:
            return 0.0
        adapter = self.runtime.scheduler.adapter
        if adapter is None:
            return 0.0
        try:
            state = adapter.vram_state()
            return float(state.usage_mb) if state else 0.0
        except Exception:
            return 0.0


# --------------------------------------------------------------------------
# 内置压测用例
# --------------------------------------------------------------------------

_STRESS_PROMPTS: List[str] = [
    "用一个词回答：天空是什么颜色？",
    "用一个词回答：一年有几个季节？",
    "用一个词回答：水的化学式是什么？",
    "用一个词回答：太阳从哪边升起？",
    "用一个词回答：一周有几天？",
]


def render_compare(result: CompareResult) -> str:
    """把对照结果渲染为 Markdown 表格。"""
    lines: List[str] = []
    lines.append("## TDS SDK 性能对照（原生直连 vs 挂载调度）")
    lines.append("")
    lines.append("- 模型：{}".format(result.model or "-"))
    lines.append("- 提示词数：{}".format(result.prompts))
    lines.append("")
    lines.append("| 指标 | 原生直连 | 挂载 TDS | 差值 | 变化 | 结论 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    labels = {
        "avg_first_ms": "平均首包(ms)",
        "avg_total_ms": "平均总耗时(ms)",
        "avg_tps": "平均吞吐(tasks/s)",
    }
    for key, label in labels.items():
        row = result.delta.get(key, {})
        lines.append(
            "| {} | {} | {} | {} | {}% | {} |".format(
                label, row.get("native", 0), row.get("sdk", 0),
                row.get("diff", 0), row.get("pct", 0),
                "提升" if row.get("improved") else "未提升",
            )
        )
    lines.append("")
    lines.append("- 原生模式失败数：{}".format(result.native.get("failed", 0)))
    lines.append("- SDK 模式失败数：{}".format(result.sdk.get("failed", 0)))
    return "\n".join(lines)


def render_stress(result: StressResult) -> str:
    """把压测结果渲染为 Markdown。"""
    lines: List[str] = []
    lines.append("## TDS 压测结果（{}）".format(result.mode))
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    lines.append("| 总任务数 | {} |".format(result.tasks))
    lines.append("| 并发度 | {} |".format(result.concurrency))
    lines.append("| 失败数 | {} |".format(result.failed))
    lines.append("| 墙钟耗时(ms) | {} |".format(result.total_ms))
    lines.append("| 平均延迟(ms) | {} |".format(result.avg_latency_ms))
    lines.append("| 平均吞吐(tasks/s) | {} |".format(result.avg_tps))
    lines.append("| 峰值显存(MB) | {:.0f} |".format(result.peak_vram_mb))
    lines.append("| 最小显存(MB) | {:.0f} |".format(result.min_vram_mb))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 命令行入口（python -m tds.chat）
# --------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="tds.chat",
        description="TDS 原生对话内核 —— SDK 能力验证与性能对照",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:11434", help="推理服务地址")
    parser.add_argument("--model", default="", help="模型名")
    parser.add_argument("--system", default="", help="系统提示词")
    parser.add_argument("--once", default="", help="单轮对话（传入用户输入）")
    parser.add_argument("--compare", default="", help="对照实验：逗号分隔的提示词")
    parser.add_argument("--stress", type=int, default=0, help="压测任务数")
    parser.add_argument("--concurrency", type=int, default=1, help="压测并发度")
    parser.add_argument("--api-key", default="", help="鉴权口令（可选）")
    args = parser.parse_args(argv)

    chat = TDSChat(args.base_url, model=args.model, system=args.system, api_key=args.api_key)

    online, note = chat.client.ping()
    print("后端：{}（{}）".format("在线" if online else "离线", note))
    if not online:
        return 1
    if not chat.model:
        print("请用 --model 指定模型")
        return 1

    if args.once:
        turn = chat.send(args.once)
        if turn.error:
            print("错误：{}".format(turn.error))
            return 2
        print(turn.content)
        print("--- 首包 {:.0f} ms / 总 {:.0f} ms / {} 字".format(
            turn.stats.first_ms, turn.stats.total_ms, turn.stats.chars))
        return 0

    if args.compare:
        prompts = [p.strip() for p in args.compare.split(",") if p.strip()]
        result = chat.compare(prompts)
        print(render_compare(result))
        return 0

    if args.stress:
        result = chat.stress(num_tasks=args.stress, concurrency=args.concurrency)
        print(render_stress(result))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
