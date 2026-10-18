"""固定用例批量回归：用例集定义 + 执行器 + 报告渲染（零第三方依赖）。

定位
----
测试台面板解决"单轮手点测试"，本模块解决"可复现的批量回归"：

1. **用例集**：文本 / 视觉两类固定用例，提示词、期望关键词、生成参数全部内置，
   支持外部 JSON 套件覆盖，保证不同时间、不同模型跑的是同一批题；
2. **执行器**：对每个 ``(模型 × 用例)`` 组合走真实流式链路，记录首包时延、总耗时、
   输出字数、令牌吞吐与错误，可选重复多次取稳定值；
3. **报告**：输出 Markdown（人看）与 JSON（机读）双份，含"模型 × 用例"结果矩阵；
4. **判定**：期望关键词命中为轻量判定，只用于快速筛查，不替代人工复核。

与引擎的边界：本模块同样属于测试台（知识分支），不随可售包分发，不承载调度逻辑。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import fixtures
from .backend import BackendClient, BackendError, ChatRequest, ModelInfo
from .session import Message, make_attachment

SUITE_BASIC = "basic"
SUITE_VISION = "vision"
SUITE_FULL = "full"

VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_ERROR = "error"
VERDICT_SKIP = "skip"

_VERDICT_MARK = {
    VERDICT_PASS: "通过",
    VERDICT_FAIL: "未命中",
    VERDICT_ERROR: "报错",
    VERDICT_SKIP: "跳过",
}


# --------------------------------------------------------------------------
# 用例模型
# --------------------------------------------------------------------------


@dataclass
class Case:
    """一条固定用例。"""

    id: str
    kind: str = "text"  # text | vision
    prompt: str = ""
    fixture: str = ""  # vision 用例的测试图场景名
    system: str = ""
    expect: List[str] = field(default_factory=list)
    expect_mode: str = "any"  # any=命中任一 | all=全部命中
    max_tokens: int = 256
    # 回归默认贪心解码：小模型在 0.2 下仍会抖动，期望值判定需要可复现
    temperature: float = 0.0
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "Case":
        if not isinstance(raw, dict) or not raw.get("id"):
            raise ValueError("用例缺少 id 字段")
        kind = str(raw.get("kind") or "text").strip().lower()
        if kind not in ("text", "vision"):
            raise ValueError("用例 {} 的 kind 非法：{}".format(raw["id"], kind))
        expect = raw.get("expect") or []
        if isinstance(expect, str):
            expect = [expect]
        mode = str(raw.get("expect_mode") or "any").strip().lower()
        if mode not in ("any", "all"):
            raise ValueError("用例 {} 的 expect_mode 非法：{}".format(raw["id"], mode))
        return Case(
            id=str(raw["id"]),
            kind=kind,
            prompt=str(raw.get("prompt") or ""),
            fixture=str(raw.get("fixture") or ""),
            system=str(raw.get("system") or ""),
            expect=[str(k) for k in expect],
            expect_mode=mode,
            max_tokens=int(raw.get("max_tokens") or 256),
            temperature=float(raw.get("temperature", 0.0)),
            note=str(raw.get("note") or ""),
        )


# --------------------------------------------------------------------------
# 内置用例集
# --------------------------------------------------------------------------

_BASIC_CASES: List[Case] = [
    Case(
        id="zh_fact",
        kind="text",
        prompt="用一句话回答：中国的首都是哪座城市？",
        expect=["北京"],
        note="中文常识与指令遵循",
    ),
    Case(
        id="math_reason",
        kind="text",
        prompt="一个书架上三层共放 84 本书，第二层是第一层的 2 倍，第三层比第一层多 4 本。第一层有多少本？只回答数字。",
        expect=["20"],
        note="多步推理（方程求解：x+2x+(x+4)=84 ⇒ x=20）",
    ),
    Case(
        id="code_fix",
        kind="text",
        prompt="下面这段 Python 会报错，请指出原因并给出修正后的完整代码：\n\ndef add(a, b):\n    return a + b\nprint(add(1))",
        expect=["参数", "argument"],
        expect_mode="any",
        note="代码缺陷定位与修复",
    ),
    Case(
        id="summarize",
        kind="text",
        prompt=(
            "用不超过 40 个字总结下面这段话：TDS 是一个宿主无关的调度中间层，"
            "在显存与模型之间做分类预判，把不同规模的请求分流到合适的执行单元，"
            "内核不依赖任何第三方库，宿主对接统一收敛到适配器。"
        ),
        # 摘要考察"要点是否保留"而非字面复述，"分流"与"调度"同义并列
        expect=["调度", "分流"],
        note="长文压缩与要点保留",
    ),
    Case(
        id="json_format",
        kind="text",
        prompt='只输出 JSON，不要任何解释：把"型号=TDS，版本=0.2.0，宿主=ComfyUI"整理为 {"model":..,"version":..,"host":..} 形式。',
        expect=['"model"', '"version"', '"host"'],
        expect_mode="all",
        note="结构化输出约束",
    ),
]

_VISION_CASES: List[Case] = [
    Case(
        id="vis_colors",
        kind="vision",
        fixture="bars_rgb",
        prompt="这张图从左到右有哪几种颜色？按顺序用一句话回答。",
        expect=["红", "绿", "蓝"],
        expect_mode="all",
        max_tokens=128,
        note="颜色识别与左右顺序",
    ),
    Case(
        id="vis_count",
        kind="vision",
        fixture="count_circles",
        prompt="图中有几个圆？只回答数字。",
        expect=["3", "三"],
        max_tokens=64,
        note="计数能力",
    ),
    Case(
        id="vis_position",
        kind="vision",
        fixture="circle_topleft",
        prompt="图中的圆位于画面的什么位置？用方位词回答。",
        expect=["左上", "左上方", "upper left", "top left"],
        max_tokens=128,
        note="空间定位（历史已知偏差点，用于跟踪）",
    ),
    Case(
        id="vis_texture",
        kind="vision",
        fixture="checker",
        prompt="用一句话描述这张图的图案。",
        expect=["棋", "格", "黑白", "checker"],
        max_tokens=128,
        note="纹理结构描述",
    ),
]

_BUILTIN_SUITES: Dict[str, List[Case]] = {
    SUITE_BASIC: _BASIC_CASES,
    SUITE_VISION: _VISION_CASES,
    SUITE_FULL: _BASIC_CASES + _VISION_CASES,
}


def suite_names() -> List[str]:
    return [SUITE_BASIC, SUITE_VISION, SUITE_FULL]


def load_suite(spec: str) -> Tuple[str, List[Case]]:
    """按名称或 JSON 文件路径载入用例集，返回 ``(套件名, 用例列表)``。"""
    key = (spec or "").strip()
    if not key:
        raise ValueError("套件名不能为空")
    if key.lower() in _BUILTIN_SUITES:
        name = key.lower()
        return name, [Case.from_dict(c.to_dict()) for c in _BUILTIN_SUITES[name]]

    import os

    if os.path.isfile(key):
        with open(key, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload.get("cases") if isinstance(payload, dict) else payload
        if not isinstance(rows, list) or not rows:
            raise ValueError("套件文件不含有效 cases 列表：{}".format(key))
        cases = [Case.from_dict(row) for row in rows]
        name = str(payload.get("name") or os.path.splitext(os.path.basename(key))[0]) if isinstance(payload, dict) else "custom"
        _validate_cases(cases)
        return name, cases

    raise ValueError(
        "未知套件：{}（内置 {}，或传入 .json 套件文件路径）".format(spec, "、".join(suite_names()))
    )


def _validate_cases(cases: Sequence[Case]) -> None:
    seen = set()
    for case in cases:
        if case.id in seen:
            raise ValueError("用例 id 重复：{}".format(case.id))
        seen.add(case.id)
        if not case.prompt.strip():
            raise ValueError("用例 {} 缺少 prompt".format(case.id))
        if case.kind == "vision" and not case.fixture:
            raise ValueError("视觉用例 {} 缺少 fixture".format(case.id))


# --------------------------------------------------------------------------
# 判定
# --------------------------------------------------------------------------


def judge(text: str, expect: Sequence[str], mode: str = "any") -> Tuple[bool, List[str], List[str]]:
    """关键词轻量判定，返回 ``(是否通过, 命中, 未命中)``。"""
    body = (text or "").lower()
    hits = [kw for kw in expect if kw.lower() in body]
    missing = [kw for kw in expect if kw not in hits]
    if not expect:
        return True, hits, missing
    ok = len(hits) == len(expect) if mode == "all" else bool(hits)
    return ok, hits, missing


# --------------------------------------------------------------------------
# 结果模型
# --------------------------------------------------------------------------


@dataclass
class CaseResult:
    """单个 (模型 × 用例) 的执行结果。"""

    case_id: str
    model: str
    kind: str
    verdict: str
    first_ms: float = 0.0
    total_ms: float = 0.0
    chars: int = 0
    tokens_per_s: float = 0.0
    text: str = ""
    error: str = ""
    hits: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    expected: List[str] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    # 重复轮次与其中通过的次数：attempts>1 且 passes<attempts 即抖动用例
    attempts: int = 1
    passes: int = 0

    @property
    def mark(self) -> str:
        return {
            VERDICT_PASS: "✓",
            VERDICT_FAIL: "✗",
            VERDICT_ERROR: "!",
            VERDICT_SKIP: "-",
        }.get(self.verdict, "?")

    @property
    def stable(self) -> bool:
        return self.attempts <= 1 or self.passes == self.attempts

    @property
    def verdict_text(self) -> str:
        return _VERDICT_MARK.get(self.verdict, self.verdict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# 执行器
# --------------------------------------------------------------------------


class BenchRunner:
    """批量回归执行器。

    Parameters
    ----------
    client:
        已构造的 :class:`BackendClient`。
    skip_vision_for_text_models:
        视觉用例对非多模态模型默认跳过（省时且避免必然报错）。
    verbose:
        是否把进度打到标准输出。
    """

    def __init__(
        self,
        client: BackendClient,
        skip_vision_for_text_models: bool = True,
        verbose: bool = False,
        temperature: Optional[float] = None,
    ) -> None:
        self.client = client
        self.skip_vision_for_text_models = skip_vision_for_text_models
        self.verbose = verbose
        # 非 None 时统一覆盖用例自带温度，用于把回归压到确定状态
        self.temperature = temperature

    # ---- 单例执行 ----

    def run_case(self, model: str, case: Case, info: Optional[ModelInfo] = None) -> CaseResult:
        if case.kind == "vision" and self.skip_vision_for_text_models:
            if info is not None and not info.multimodal:
                return CaseResult(
                    case_id=case.id,
                    model=model,
                    kind=case.kind,
                    verdict=VERDICT_SKIP,
                    error="非多模态模型，已跳过视觉用例",
                )

        images = []
        if case.fixture:
            try:
                raw = fixtures.image_bytes(case.fixture)
            except KeyError as exc:
                return CaseResult(
                    case_id=case.id,
                    model=model,
                    kind=case.kind,
                    verdict=VERDICT_ERROR,
                    error=str(exc),
                )
            images.append(make_attachment("{}.png".format(case.fixture), "image/png", raw))

        message = Message(role="user", content=case.prompt, images=images)
        request = ChatRequest(
            model=model,
            messages=[message.to_openai()],
            system=case.system,
            temperature=case.temperature if self.temperature is None else self.temperature,
            max_tokens=case.max_tokens,
            stream=True,
        )

        # 时延统一用 perf_counter（单调、亚毫秒分辨率）；time.time 在部分平台
        # 分辨率可达 15.6 ms，会把首包时延抹成 0，导致报告失真。
        started = time.perf_counter()
        first_ms: Optional[float] = None
        parts: List[str] = []
        usage: Dict[str, Any] = {}
        try:
            for chunk in self.client.chat_stream(request):
                if chunk.error:
                    raise BackendError(chunk.error)
                if chunk.text or chunk.reasoning:
                    if first_ms is None:
                        first_ms = (time.perf_counter() - started) * 1000.0
                parts.append(chunk.text or "")
                if chunk.usage:
                    usage = chunk.usage
                if chunk.done:
                    break
        except BackendError as exc:
            return CaseResult(
                case_id=case.id,
                model=model,
                kind=case.kind,
                verdict=VERDICT_ERROR,
                total_ms=(time.perf_counter() - started) * 1000.0,
                error=str(exc),
                expected=list(case.expect),
            )

        total_ms = (time.perf_counter() - started) * 1000.0
        text = "".join(parts).strip()
        ok, hits, missing = judge(text, case.expect, case.expect_mode)
        completion = float(usage.get("completion_tokens") or 0) or 0.0
        speed = (completion / (total_ms / 1000.0)) if completion > 0 and total_ms > 0 else 0.0
        return CaseResult(
            case_id=case.id,
            model=model,
            kind=case.kind,
            verdict=VERDICT_PASS if ok else VERDICT_FAIL,
            first_ms=first_ms or 0.0,
            total_ms=total_ms,
            chars=len(text),
            tokens_per_s=speed,
            text=text,
            hits=hits,
            missing=missing,
            expected=list(case.expect),
            usage=usage,
        )

    # ---- 批量执行 ----

    def run(
        self,
        models: Sequence[str],
        cases: Sequence[Case],
        model_map: Optional[Dict[str, ModelInfo]] = None,
        repeat: int = 1,
    ) -> List[CaseResult]:
        """按 ``模型 × 用例 × 重复次数`` 执行，返回全部结果（重复结果取最优）。"""
        info_map = model_map or {}
        results: List[CaseResult] = []
        total = len(models) * len(cases)
        done = 0
        for model in models:
            for case in cases:
                done += 1
                attempts = [self.run_case(model, case, info_map.get(model)) for _ in range(max(1, repeat))]
                best = _pick_best(attempts)
                best.attempts = max(1, repeat)
                best.passes = sum(1 for a in attempts if a.verdict == VERDICT_PASS)
                results.append(best)
                if self.verbose:
                    print(
                        "[{:>3}/{:<3}] {:<18} {:<14} {} {:>8.0f} ms  {} 字{}".format(
                            done,
                            total,
                            model,
                            case.id,
                            best.mark,
                            best.total_ms,
                            best.chars,
                            "  ({})".format(best.error) if best.error else "",
                        )
                    )
        return results


def _pick_best(attempts: Sequence[CaseResult]) -> CaseResult:
    """多轮重复的择优规则：通过 > 未命中 > 报错，同级取总耗时更短者。"""
    rank = {VERDICT_PASS: 0, VERDICT_FAIL: 1, VERDICT_SKIP: 2, VERDICT_ERROR: 3}
    return sorted(attempts, key=lambda r: (rank.get(r.verdict, 9), r.total_ms))[0]


# --------------------------------------------------------------------------
# 汇总与报告
# --------------------------------------------------------------------------


def summarize(results: Sequence[CaseResult], models: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """按模型汇总通过 / 未命中 / 报错 / 跳过与平均时延。"""
    summary: Dict[str, Dict[str, Any]] = {}
    for model in models:
        rows = [r for r in results if r.model == model]
        usable = [r for r in rows if r.verdict in (VERDICT_PASS, VERDICT_FAIL)]
        firsts = [r.first_ms for r in usable if r.first_ms > 0]
        summary[model] = {
            "total": len(rows),
            "pass": sum(1 for r in rows if r.verdict == VERDICT_PASS),
            "fail": sum(1 for r in rows if r.verdict == VERDICT_FAIL),
            "error": sum(1 for r in rows if r.verdict == VERDICT_ERROR),
            "skip": sum(1 for r in rows if r.verdict == VERDICT_SKIP),
            "avg_first_ms": round(sum(firsts) / len(firsts), 1) if firsts else 0.0,
            "avg_total_ms": round(sum(r.total_ms for r in usable) / len(usable), 1) if usable else 0.0,
        }
    return summary


def build_report(
    suite_name: str,
    cases: Sequence[Case],
    models: Sequence[str],
    results: Sequence[CaseResult],
    model_map: Optional[Dict[str, ModelInfo]] = None,
    backend: str = "",
    repeat: int = 1,
) -> Dict[str, Any]:
    """组装机读报告（可 JSON 序列化）。"""
    return {
        "kind": "tds.ui.bench",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "backend": backend,
        "suite": suite_name,
        "repeat": repeat,
        "models": [
            (model_map[model].to_dict() if model_map and model in model_map else {"name": model})
            for model in models
        ],
        "cases": [c.to_dict() for c in cases],
        "summary": summarize(results, models),
        "results": [r.to_dict() for r in results],
    }


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _excerpt(text: str, limit: int = 160) -> str:
    body = " ".join((text or "").split())
    return body if len(body) <= limit else body[:limit] + "…"


def render_markdown(report: Dict[str, Any]) -> str:
    """把报告渲染为 Markdown 文本。"""
    cases = report.get("cases", [])
    models = [m.get("name", "") for m in report.get("models", [])]
    results = report.get("results", [])
    summary = report.get("summary", {})
    index = {(r["model"], r["case_id"]): r for r in results}

    lines: List[str] = []
    lines.append("# TDS 对话测试台 · 用例回归报告")
    lines.append("")
    lines.append("- 生成时间：{}".format(report.get("generated_at", "")))
    lines.append("- 模型后端：{}".format(report.get("backend", "")))
    lines.append("- 用例套件：{}（{} 条用例）".format(report.get("suite", ""), len(cases)))
    lines.append("- 重复轮次：{}".format(report.get("repeat", 1)))
    lines.append("- 判定方式：期望关键词命中（轻量筛查，不替代人工复核）")
    lines.append("")

    lines.append("## 一、模型概览")
    lines.append("")
    lines.append("| 模型 | 参数量 | 量化 | 多模态 | 云端 | 体积(MB) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for model in report.get("models", []):
        lines.append(
            "| {} | {} | {} | {} | {} | {:.0f} |".format(
                _cell(model.get("name", "")),
                _cell(model.get("parameter_size") or "-"),
                _cell(model.get("quantization") or "-"),
                "是" if model.get("multimodal") else "否",
                "是" if model.get("remote") else "否",
                (model.get("size_bytes") or 0) / (1024 * 1024),
            )
        )
    lines.append("")

    lines.append("## 二、结果汇总")
    lines.append("")
    lines.append("| 模型 | 用例 | 通过 | 未命中 | 报错 | 跳过 | 平均首包(ms) | 平均总耗时(ms) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for model in models:
        row = summary.get(model, {})
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _cell(model),
                row.get("total", 0),
                row.get("pass", 0),
                row.get("fail", 0),
                row.get("error", 0),
                row.get("skip", 0),
                row.get("avg_first_ms", 0.0),
                row.get("avg_total_ms", 0.0),
            )
        )
    lines.append("")

    lines.append("## 三、结果矩阵")
    lines.append("")
    lines.append("> 单元格为 `标记 首包/总耗时(ms)`；✓ 通过、✗ 未命中、! 报错、- 跳过。")
    lines.append("")
    lines.append("| 用例 | 类型 | " + " | ".join(_cell(m) for m in models) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in models) + " |")
    for case in cases:
        cells = []
        for model in models:
            row = index.get((model, case["id"]))
            if not row:
                cells.append("·")
            elif row["verdict"] == VERDICT_SKIP:
                cells.append("-")
            elif row["verdict"] == VERDICT_ERROR:
                cells.append("!")
            else:
                cell = "{} {:.0f}/{:.0f}".format(
                    {"pass": "✓", "fail": "✗"}.get(row["verdict"], "?"),
                    row.get("first_ms", 0.0),
                    row.get("total_ms", 0.0),
                )
                attempts = int(row.get("attempts") or 1)
                if attempts > 1:
                    cell += " {}/{}".format(int(row.get("passes") or 0), attempts)
                cells.append(cell)
        lines.append(
            "| {} | {} | ".format(_cell(case["id"]), "视觉" if case["kind"] == "vision" else "文本")
            + " | ".join(cells)
            + " |"
        )
    lines.append("")

    lines.append("## 四、用例明细")
    lines.append("")
    for case in cases:
        lines.append("### {} · {}".format(case["id"], "视觉" if case["kind"] == "vision" else "文本"))
        lines.append("")
        if case.get("note"):
            lines.append("- 考察点：{}".format(case["note"]))
        if case.get("fixture"):
            try:
                lines.append("- 测试图：{}（{}）".format(case["fixture"], fixtures.describe(case["fixture"])))
            except KeyError:
                lines.append("- 测试图：{}（未注册）".format(case["fixture"]))
        if case.get("expect"):
            lines.append(
                "- 期望关键词（{}）：{}".format(
                    "全部命中" if case.get("expect_mode") == "all" else "任一命中",
                    "、".join(case["expect"]),
                )
            )
        lines.append("- 提示词：{}".format(_excerpt(case.get("prompt", ""), 200)))
        lines.append("")
        for model in models:
            row = index.get((model, case["id"]))
            if not row:
                continue
            head = "**{}** — {}".format(model, _VERDICT_MARK.get(row["verdict"], row["verdict"]))
            if row["verdict"] in (VERDICT_PASS, VERDICT_FAIL):
                attempts = int(row.get("attempts") or 1)
                passes = int(row.get("passes") or 0)
                flaky_note = (
                    " / {} 轮中仅通过 {}".format(attempts, passes) if 0 < passes < attempts else ""
                )
                head += "（首包 {:.0f} ms / 总 {:.0f} ms / {} 字{}{}）".format(
                    row.get("first_ms", 0.0),
                    row.get("total_ms", 0.0),
                    row.get("chars", 0),
                    " / {:.1f} tok·s⁻¹".format(row["tokens_per_s"]) if row.get("tokens_per_s") else "",
                    flaky_note,
                )
            lines.append(head)
            if row.get("error"):
                lines.append("")
                lines.append("> 错误：{}".format(_cell(row["error"])))
            if row.get("text"):
                lines.append("")
                lines.append("> {}".format(_excerpt(row["text"], 300)))
            lines.append("")
        lines.append("")

    flaky = [
        r
        for r in results
        if r["verdict"] in (VERDICT_PASS, VERDICT_FAIL)
        and int(r.get("attempts") or 1) > 1
        and 0 < int(r.get("passes") or 0) < int(r["attempts"])
    ]
    errors = [r for r in results if r["verdict"] == VERDICT_ERROR]
    skips = [r for r in results if r["verdict"] == VERDICT_SKIP]
    if errors or skips or flaky:
        lines.append("## 五、备注")
        lines.append("")
        if flaky:
            lines.append("### 抖动用例（重复轮次内结论不一致）")
            lines.append("")
            for row in flaky:
                lines.append(
                    "- {} × {}：{} 轮中通过 {} 轮".format(
                        row["model"], row["case_id"], row["attempts"], row["passes"]
                    )
                )
            lines.append("")
        if errors:
            lines.append("### 报错")
            lines.append("")
            for row in errors:
                lines.append("- {} × {}：{}".format(row["model"], row["case_id"], _cell(row["error"])))
            lines.append("")
        if skips:
            lines.append("### 跳过")
            lines.append("")
            seen = sorted({"{:<20}".format(r["model"]) for r in skips})
            lines.append("- 非多模态模型跳过视觉用例：{}".format("、".join(s.strip() for s in seen)))
            lines.append("")
    return "\n".join(lines)


def write_report(report: Dict[str, Any], out_dir: str) -> Dict[str, str]:
    """把 Markdown 与 JSON 报告写入目录，返回两者路径。"""
    import os

    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = "bench_{}_{}".format(report.get("suite", "suite"), stamp)
    md_path = os.path.join(out_dir, base + ".md")
    json_path = os.path.join(out_dir, base + ".json")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(report))
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return {"markdown": md_path, "json": json_path}


def run_bench(
    client: BackendClient,
    suite_spec: str,
    models: Sequence[str],
    repeat: int = 1,
    skip_vision_for_text_models: bool = True,
    verbose: bool = False,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """一站式入口：载入套件 → 执行 → 组装报告。"""
    suite_name, cases = load_suite(suite_spec)
    _validate_cases(cases)
    info_map: Dict[str, ModelInfo] = {}
    try:
        info_map = {m.name: m for m in client.list_models()}
    except BackendError:
        info_map = {}
    runner = BenchRunner(
        client,
        skip_vision_for_text_models=skip_vision_for_text_models,
        verbose=verbose,
        temperature=temperature,
    )
    results = runner.run(list(models), cases, info_map, repeat=repeat)
    return build_report(
        suite_name=suite_name,
        cases=cases,
        models=list(models),
        results=results,
        model_map=info_map,
        backend=client.base_url,
        repeat=repeat,
    )


if __name__ == "__main__":
    import sys

    for name in suite_names():
        _, rows = load_suite(name)
        print("套件 {}：{} 条".format(name, len(rows)))
        for row in rows:
            print("  - {:<14} {:<6} {}".format(row.id, row.kind, row.prompt.splitlines()[0][:44]))
    print("测试图：{}".format("、".join(fixtures.names())))
    sys.exit(0)
