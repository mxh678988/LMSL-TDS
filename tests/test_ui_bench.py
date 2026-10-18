"""tds.ui 回归引擎单测：测试图生成、套件载入、判定、执行器与报告（不连真实后端）。

后端用 ``FakeClient`` 替身：既能构造正常流式回答，也能模拟报错与用量字段，
从而在无模型环境下覆盖"通过 / 未命中 / 报错 / 跳过"四条判定分支。
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tds.ui import fixtures  # noqa: E402
from tds.ui.backend import BackendError, Chunk, ModelInfo  # noqa: E402
from tds.ui.bench import (  # noqa: E402
    BenchRunner,
    Case,
    CaseResult,
    build_report,
    judge,
    load_suite,
    render_markdown,
    suite_names,
    summarize,
    write_report,
)
from tds.ui.cli import main  # noqa: E402

OFFLINE_BACKEND = "http://127.0.0.1:9"


class FakeClient:
    """替身后端：按用例提示词返回确定性回答。"""

    def __init__(self, answer: str = "北京", error: str = "", usage: bool = False) -> None:
        self.answer = answer
        self.error = error
        self.usage = usage
        self.base_url = "http://fake"
        self.requests = []

    def list_models(self):
        return [
            ModelInfo(name="text-model", multimodal=False),
            ModelInfo(name="vision-model:7b", multimodal=True, parameter_size="7B"),
        ]

    def chat_stream(self, req):
        self.requests.append(req)
        if self.error:
            yield Chunk(error=self.error)
            return
        yield Chunk(text=self.answer)
        if self.usage:
            yield Chunk(usage={"completion_tokens": 20})
        yield Chunk(done=True)


class TestFixtures(unittest.TestCase):
    def test_all_scenes_are_valid_png(self) -> None:
        for name in fixtures.names():
            raw = fixtures.image_bytes(name)
            width, height = fixtures.read_png_size(raw)
            self.assertGreater(width, 0)
            self.assertGreater(height, 0)
            self.assertIn(name, fixtures.names())
            self.assertTrue(fixtures.describe(name))

    def test_scene_sizes(self) -> None:
        self.assertEqual(fixtures.read_png_size(fixtures.image_bytes("bars_rgb")), (240, 120))
        self.assertEqual(fixtures.read_png_size(fixtures.image_bytes("checker")), (240, 240))

    def test_deterministic(self) -> None:
        self.assertEqual(fixtures.image_bytes("checker"), fixtures.scene_checker())
        self.assertEqual(fixtures.image_bytes("bars_rgb"), fixtures.scene_bars())

    def test_unknown_scene_raises(self) -> None:
        with self.assertRaises(KeyError):
            fixtures.image_bytes("no_such_scene")
        with self.assertRaises(ValueError):
            fixtures.read_png_size(b"not a png at all")

    def test_canvas_bounds_are_clamped(self) -> None:
        canvas = fixtures.Canvas(4, 4)
        canvas.fill_circle(-5, -5, 3, fixtures.BLACK)
        canvas.set_pixel(99, 99, fixtures.RED)
        self.assertEqual(fixtures.read_png_size(canvas.to_png()), (4, 4))


class TestSuiteLoading(unittest.TestCase):
    def test_builtin_suites(self) -> None:
        self.assertEqual(set(suite_names()), {"basic", "vision", "full"})
        _, basic = load_suite("basic")
        _, vision = load_suite("vision")
        _, full = load_suite("full")
        self.assertEqual(len(full), len(basic) + len(vision))
        self.assertTrue(all(c.kind == "vision" for c in vision))
        self.assertTrue(all(c.fixture for c in vision))
        self.assertTrue(all(c.prompt for c in full))

    def test_builtin_is_isolated_copy(self) -> None:
        _, first = load_suite("basic")
        first[0].prompt = "篡改"
        _, second = load_suite("basic")
        self.assertNotEqual(second[0].prompt, "篡改")

    def test_load_from_json_file(self) -> None:
        payload = {
            "name": "custom",
            "cases": [
                {"id": "c1", "kind": "text", "prompt": "1+1=?", "expect": "2"},
                {
                    "id": "c2",
                    "kind": "vision",
                    "prompt": "看图",
                    "fixture": "checker",
                    "expect": ["黑白"],
                    "expect_mode": "all",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "suite.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            name, cases = load_suite(path)
        self.assertEqual(name, "custom")
        self.assertEqual([c.id for c in cases], ["c1", "c2"])
        self.assertEqual(cases[0].expect, ["2"])  # 字符串自动包成列表
        self.assertEqual(cases[1].expect_mode, "all")

    def test_builtin_math_expectation_matches_reference_answer(self) -> None:
        # 参考解：x + 2x + (x + 4) = 84 ⇒ x = 20，期望值写错会把正确回答判成未命中
        _, cases = load_suite("basic")
        math_case = next(c for c in cases if c.id == "math_reason")
        self.assertEqual(math_case.expect, ["20"])

    def test_rejects_bad_specs(self) -> None:
        with self.assertRaises(ValueError):
            load_suite("")
        with self.assertRaises(ValueError):
            load_suite("no_such_suite")
        with self.assertRaises(ValueError):
            Case.from_dict({"kind": "text"})
        with self.assertRaises(ValueError):
            Case.from_dict({"id": "x", "kind": "audio"})
        with self.assertRaises(ValueError):
            Case.from_dict({"id": "x", "kind": "text", "prompt": "p", "expect_mode": "all_of"})

    def test_rejects_dup_and_incomplete_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for rows in (
                [{"id": "d", "prompt": "a"}, {"id": "d", "prompt": "b"}],
                [{"id": "v", "kind": "vision", "prompt": "看图"}],
                [{"id": "e", "prompt": ""}],
            ):
                path = os.path.join(tmp, "bad.json")
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(rows, handle, ensure_ascii=False)
                with self.assertRaises(ValueError):
                    load_suite(path)


class TestJudge(unittest.TestCase):
    def test_any_mode(self) -> None:
        ok, hits, missing = judge("图片里是三个圆形", ["3", "三"], "any")
        self.assertTrue(ok)
        self.assertEqual(hits, ["三"])
        self.assertEqual(missing, ["3"])

    def test_all_mode(self) -> None:
        ok, _, missing = judge("红、绿、蓝", ["红", "绿", "蓝"], "all")
        self.assertTrue(ok)
        ok, _, missing = judge("只有红色和绿色", ["红", "绿", "蓝"], "all")
        self.assertFalse(ok)
        self.assertEqual(missing, ["蓝"])

    def test_case_insensitive_and_empty(self) -> None:
        self.assertTrue(judge("Upper Left corner", ["upper left"], "any")[0])
        ok, hits, missing = judge("随便什么", [], "any")
        self.assertTrue(ok)
        self.assertEqual((hits, missing), ([], []))


class TestRunner(unittest.TestCase):
    def test_pass_and_payload_contains_image(self) -> None:
        client = FakeClient(answer="从左到右是红色、绿色、蓝色")
        runner = BenchRunner(client, verbose=False)
        case = next(c for c in load_suite("vision")[1] if c.id == "vis_colors")
        result = runner.run_case("vision-model:7b", case, ModelInfo(name="vision-model:7b", multimodal=True))
        self.assertEqual(result.verdict, "pass")
        self.assertGreater(result.first_ms, 0.0)
        self.assertGreater(result.total_ms, 0.0)
        self.assertGreater(result.chars, 0)
        content = client.requests[0].messages[0]["content"]
        self.assertIsInstance(content, list)
        self.assertTrue(any(part.get("type") == "image_url" for part in content))

    def test_fail_when_keyword_absent(self) -> None:
        runner = BenchRunner(FakeClient(answer="我不知道"), verbose=False)
        result = runner.run_case("text-model", Case(id="x", prompt="p", expect=["北京"]))
        self.assertEqual(result.verdict, "fail")
        self.assertEqual(result.missing, ["北京"])

    def test_error_path(self) -> None:
        runner = BenchRunner(FakeClient(error="模型未加载"), verbose=False)
        result = runner.run_case("text-model", Case(id="x", prompt="p", expect=["北京"]))
        self.assertEqual(result.verdict, "error")
        self.assertIn("模型未加载", result.error)
        self.assertEqual(result.text, "")

    def test_error_when_backend_raises(self) -> None:
        class Boom(FakeClient):
            def chat_stream(self, req):
                raise BackendError("连接被拒绝")
                yield  # pragma: no cover

        runner = BenchRunner(Boom(), verbose=False)
        result = runner.run_case("text-model", Case(id="x", prompt="p"))
        self.assertEqual(result.verdict, "error")
        self.assertIn("连接被拒绝", result.error)

    def test_bad_fixture_reports_error(self) -> None:
        runner = BenchRunner(FakeClient(), verbose=False)
        case = Case(id="v", kind="vision", prompt="看图", fixture="missing_scene")
        self.assertEqual(runner.run_case("vision-model:7b", case).verdict, "error")

    def test_vision_skip_for_text_models(self) -> None:
        case = Case(id="v", kind="vision", prompt="看图", fixture="checker")
        skipped = BenchRunner(FakeClient(), verbose=False).run_case(
            "text-model", case, ModelInfo(name="text-model", multimodal=False)
        )
        self.assertEqual(skipped.verdict, "skip")
        enforced = BenchRunner(FakeClient(answer="黑白"), False).run_case(
            "text-model", case, ModelInfo(name="text-model", multimodal=False)
        )
        self.assertEqual(enforced.verdict, "pass")

    def test_repeat_prefers_pass(self) -> None:
        class Flaky(FakeClient):
            def __init__(self) -> None:
                super().__init__(answer="北京")
                self.calls = 0

            def chat_stream(self, req):
                self.calls += 1
                if self.calls == 1:
                    yield Chunk(error="冷启动失败")
                    return
                yield Chunk(text=self.answer)
                yield Chunk(done=True)

        runner = BenchRunner(Flaky(), verbose=False)
        results = runner.run(["text-model"], [Case(id="x", prompt="p", expect=["北京"])], repeat=2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].verdict, "pass")

    def test_verbose_progress_prints(self) -> None:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            BenchRunner(FakeClient(), verbose=True).run(
                ["text-model"], [Case(id="x", prompt="p", expect=["北京"])]
            )
        self.assertIn("x", buffer.getvalue())


    def test_temperature_override(self) -> None:
        client = FakeClient(answer="北京")
        BenchRunner(client, verbose=False, temperature=0.0).run_case(
            "text-model", Case(id="x", prompt="p", expect=["北京"], temperature=0.9)
        )
        self.assertEqual(client.requests[0].temperature, 0.0)

        plain = FakeClient(answer="北京")
        BenchRunner(plain, verbose=False).run_case("text-model", Case(id="x", prompt="p", temperature=0.3))
        self.assertEqual(plain.requests[0].temperature, 0.3)

        default = FakeClient(answer="北京")
        BenchRunner(default, verbose=False).run_case("text-model", Case(id="x", prompt="p"))
        self.assertEqual(default.requests[0].temperature, 0.0)


class TestReport(unittest.TestCase):
    def _report(self):
        client = FakeClient(answer="北京", usage=True)
        runner = BenchRunner(client, verbose=False)
        cases = [
            Case(id="ok", prompt="首都是哪", expect=["北京"]),
            Case(id="bad", prompt="看图", kind="vision", fixture="checker", expect=["黑白"]),
        ]
        results = [
            runner.run_case("text-model", cases[0], ModelInfo(name="text-model")),
            runner.run_case("text-model", cases[1], ModelInfo(name="text-model", multimodal=False)),
            runner.run_case("vision-model:7b", cases[1], ModelInfo(name="vision-model:7b", multimodal=True)),
        ]
        model_map = {m.name: m for m in client.list_models()}
        return build_report("mixed", cases, ["text-model", "vision-model:7b"], results, model_map, "http://fake")

    def test_summary_counts(self) -> None:
        report = self._report()
        text_row = report["summary"]["text-model"]
        self.assertEqual(text_row["pass"], 1)
        self.assertEqual(text_row["skip"], 1)
        self.assertEqual(report["summary"]["vision-model:7b"]["fail"], 1)

    def test_summarize_is_pure(self) -> None:
        results = [
            CaseResult(case_id="a", model="m", kind="text", verdict="pass", first_ms=10.0, total_ms=100.0),
            CaseResult(case_id="b", model="m", kind="text", verdict="error"),
        ]
        row = summarize(results, ["m"])["m"]
        self.assertEqual((row["pass"], row["error"], row["avg_first_ms"]), (1, 1, 10.0))

    def test_repeat_tracks_stability(self) -> None:
        class FlakyClient(FakeClient):
            def __init__(self) -> None:
                super().__init__(answer="北京")
                self.calls = 0

            def chat_stream(self, req):
                self.calls += 1
                self.requests.append(req)
                yield Chunk(text="北京" if self.calls == 1 else "上海")
                yield Chunk(done=True)

        runner = BenchRunner(FlakyClient(), verbose=False)
        case = Case(id="flaky", prompt="首都是哪", expect=["北京"])
        results = runner.run(["text-model"], [case], repeat=2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].verdict, "pass")
        self.assertEqual(results[0].attempts, 2)
        self.assertEqual(results[0].passes, 1)
        self.assertFalse(results[0].stable)

        steady = BenchRunner(FakeClient(answer="北京"), verbose=False).run(["text-model"], [case], repeat=2)
        self.assertTrue(steady[0].stable)
        self.assertEqual(steady[0].passes, 2)

    def test_markdown_flags_flaky_case(self) -> None:
        report = self._report()
        report["results"][0]["attempts"] = 3
        report["results"][0]["passes"] = 1
        markdown = render_markdown(report)
        self.assertIn("1/3", markdown)
        self.assertIn("抖动用例", markdown)
        self.assertIn("3 轮中仅通过 1", markdown)

    def test_markdown_contains_key_sections(self) -> None:
        markdown = render_markdown(self._report())
        for token in ("TDS 对话测试台 · 用例回归报告", "结果矩阵", "用例明细", "text-model", "vis", "ok", "bad"):
            if token == "vis":
                continue
            self.assertIn(token, markdown)
        self.assertIn("✓", markdown)
        self.assertIn("-", markdown)

    def test_write_report_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_report(self._report(), tmp)
            self.assertTrue(os.path.isfile(paths["markdown"]))
            self.assertTrue(os.path.isfile(paths["json"]))
            with open(paths["json"], "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["kind"], "tds.ui.bench")
            self.assertEqual(len(payload["results"]), 3)


class TestCli(unittest.TestCase):
    def test_select_models_dedupes_and_limits(self) -> None:
        from tds.ui.cli import _select_models

        class Dup(FakeClient):
            def list_models(self):
                return [
                    ModelInfo(name="a"),
                    ModelInfo(name="a"),
                    ModelInfo(name="b"),
                    ModelInfo(name="cloud", remote=True),
                ]

        client = Dup()
        self.assertEqual(_select_models(client, "", 0), ["a", "b"])
        self.assertEqual(_select_models(client, "", 1), ["a"])
        self.assertEqual(_select_models(client, "", 0, include_remote=True), ["a", "b", "cloud"])
        self.assertEqual(_select_models(client, " x , y ", 5), ["x", "y"])

    def test_list_suite_exits_zero(self) -> None:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(["bench", "--suite", "full", "--list"])
        self.assertEqual(code, 0)
        self.assertIn("vis_colors", buffer.getvalue())

    def test_bench_offline_returns_error(self) -> None:
        buffer = io.StringIO()
        errors = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(errors):
            code = main(["bench", "--suite", "basic", "--backend", OFFLINE_BACKEND, "--no-write"])
        self.assertEqual(code, 1)
        self.assertIn("离线", errors.getvalue())

    def test_unknown_suite_surfaces_error(self) -> None:
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            code = main(["bench", "--suite", "nope", "--list"])
        self.assertEqual(code, 1)
        self.assertIn("未知套件", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
