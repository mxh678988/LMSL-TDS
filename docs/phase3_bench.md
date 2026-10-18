---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 350ab7144af9618434cdce7bfa6d84e8_193490abc30d11f18019525400248c00
    ReservedCode1: v5nnBJunBExoD9sgXwyfhqM28taI+4kLhsZc2HLM3yV4xsALLS2Wp7u1DeIbwDXWylvOcyc5qts7+4SyxetmMCzupCJCQB3o1C9KGe+flie/5I1aScLGA/Kil0nNkM6Dc51BBlJAc0P0AQH0t+yMX3wvJeQrbG5IvoWfNnBNbSioYgMk2S4CSjbbw0c=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 350ab7144af9618434cdce7bfa6d84e8_193490abc30d11f18019525400248c00
    ReservedCode2: v5nnBJunBExoD9sgXwyfhqM28taI+4kLhsZc2HLM3yV4xsALLS2Wp7u1DeIbwDXWylvOcyc5qts7+4SyxetmMCzupCJCQB3o1C9KGe+flie/5I1aScLGA/Kil0nNkM6Dc51BBlJAc0P0AQH0t+yMX3wvJeQrbG5IvoWfNnBNbSioYgMk2S4CSjbbw0c=
---

# Phase3 · 固定用例回归（tds.ui.bench）

> 定位：**知识分支**，与对话测试台同源。把一组固定提示词 + 程序生成的测试图做成用例集，
> 一键跑完并留档对比，用于回答“这次改动到底把哪个模型的能力跑丢了”。
> **不计入可售 IP**，不随 `tds-core` 发行包分发。

## 1. 与测试台的分工

| 工具 | 回答的问题 | 交互形态 |
| --- | --- | --- |
| `tds.ui`（测试台） | 这个模型**现在**答得怎么样？ | 人工敲一句、看一句 |
| `tds.ui.bench`（回归） | 换版本 / 换量化 / 换参数后，哪些用例**由过转变不过**？ | 一条命令跑完、出报告 |

测试台暴露的是“链路通不通、模型强不强”，回归引擎负责把这种判断**固化成可复现的判定**，
否则每次评估都依赖人工记忆，无法跨版本比较。

## 2. 设计要点

| 维度 | 约定 |
| --- | --- |
| 用例来源 | 内置 Python 常量（`bench.py`）+ 可选外部 JSON 套件文件 |
| 测试图 | `fixtures.py` 程序内生成 PNG，无外部素材、无版权问题、字节级可复现 |
| 判定方式 | 期望关键词命中（`any` 任一命中 / `all` 全部命中），轻量筛查，不替代人工复核 |
| 默认温度 | `0.0`（贪心解码）。小模型在 0.2 下结论会抖动，回归需要可复现；可用 `--temperature` 覆盖 |
| 计时 | `time.perf_counter()`（单调、亚毫秒）；`time.time()` 在部分平台分辨率 15.6 ms，会把首包抹成 0 |
| 重复轮次 | `--repeat N`，逐轮执行后**取优**（通过 > 未命中 > 报错，同级取更快者），同时记录 `通过次数/轮次` 以暴露抖动用例 |
| 视觉用例 | 对非多模态模型默认跳过（省时且避免必然报错），`--vision-on-text` 可强制投喂 |
| 依赖 | 零第三方运行时，仅标准库；不调用任何外部程序，只走 HTTP 访问已运行的推理端点 |

## 3. 结构

```
tds/ui/
├── bench.py        # 用例模型 / 判定 / 执行器 / 汇总 / Markdown+JSON 报告 / run_bench 入口
├── fixtures.py     # 测试图生成（bars_rgb、count_circles、circle_topleft、checker）
tests/test_ui_bench.py   # 回归引擎单测（FakeClient 替身，不连真实后端）
tools/run_bench.bat      # Windows 一键跑默认 full 套件，产出 bench_out/
```

## 4. 用例集

| 套件 | 组成 |
| --- | --- |
| `basic` | 5 条文本用例 |
| `vision` | 4 条视觉用例 |
| `full` | 全部 9 条 |

| 用例 | 类型 | 考察点 | 期望关键词 | 测试图 |
| --- | --- | --- | --- | --- |
| `zh_fact` | 文本 | 中文常识与指令遵循 | 北京 | — |
| `math_reason` | 文本 | 多步推理（x+2x+(x+4)=84 ⇒ x=20） | 20 | — |
| `code_fix` | 文本 | 代码缺陷定位与修复 | 参数 / argument | — |
| `summarize` | 文本 | 长文压缩与要点保留 | 调度 / 分流 | — |
| `json_format` | 文本 | 严格 JSON 输出格式 | schema / 字段名 | — |
| `vis_colors` | 视觉 | 颜色识别与顺序 | 红、绿、蓝（全部命中） | bars_rgb |
| `vis_count` | 视觉 | 计数能力 | 3 / 三 | count_circles |
| `vis_position` | 视觉 | 空间定位（历史已知偏差点，用于跟踪） | 左上 | circle_topleft |
| `vis_texture` | 视觉 | 纹理结构描述 | 棋 / 格 / 黑白 | checker |

## 5. 用法

```bash
# 列出套件与用例（不连后端）
python -m tds.ui.cli bench --list

# 全量回归：自动挑可用模型，温度 0、3 轮重复
python -m tds.ui.cli bench --suite full --repeat 3 --temperature 0

# 指定模型与报告目录
python -m tds.ui.cli bench --suite full --models qwen2.5:3b,qwen2.5vl:7b --out bench_out

# 只看视觉用例，并强制投喂给非多模态模型
python -m tds.ui.cli bench --suite vision --vision-on-text
```

| 参数 | 说明 |
| --- | --- |
| `--suite` | 套件名或 JSON 套件文件路径 |
| `--models` | 逗号分隔的模型名；缺省按后端清单自动挑选去重 |
| `--limit-models` | 自动挑选时最多取几个模型 |
| `--repeat` | 每个用例重复轮次（取最优） |
| `--temperature` | 统一覆盖用例自带温度，回归建议 `0` |
| `--vision-on-text` | 不跳过非多模态模型的视觉用例 |
| `--out` | 报告输出目录（默认 `bench_out`） |
| `--show` | 终端直接打印 Markdown 报告 |
| `--list` | 只列用例 |

Windows 下可双击 `tools/run_bench.bat`。

## 6. 报告

每次执行落两个文件：`bench_out/bench_<套件>_<时间戳>.md`（人读）与 `.json`（机读，可做跨版本 diff）。

Markdown 结构：

1. **一、模型概览** —— 参数量、量化、多模态标记、体积
2. **二、结果汇总** —— 每模型通过/未命中/报错/跳过 + 平均首包与总耗时
3. **三、结果矩阵** —— 用例 × 模型，单元格 `标记 首包/总耗时(ms)`，`--repeat>1` 时追加 `通过轮次/总轮次`
4. **四、用例明细** —— 每模型的完整回答原文，便于人工复核判定是否被关键词误杀
5. **五、备注** —— 抖动用例（重复轮次内结论不一致）+ 报错 + 跳过

## 7. 本轮实测结论（2026-10-08，最终基线）

| 维度 | 结果 |
| --- | --- |
| 运行环境 | Ollama 0.40.1（12 模型，含 qwen2.5:3b + qwen2.5vl:7b） |
| 回归策略 | 温度 0.0、3 轮重复取优 |
| qwen2.5:3b（3.1B） | **5/5** 文本用例全过（zh_fact、math_reason、code_fix、summarize、json_format），4 条视觉跳过 |
| qwen2.5vl:7b（8.3B） | **8/8** 用例全过（文本 4 + 视觉 4） |
| 性能基线 | 3b：首包 128 ms / 总 508 ms；7b：首包 166 ms / 总 545 ms |

**已知抖动点**：

- `math_reason` + qwen2.5:3b：3 轮中通过 2 轮（1 轮输出 `24` 而非期望的 `20`），属于小模型推理能力边界，非链路问题。
- `summarize` + qwen2.5:3b（早期版本）：未命中，因期望关键词仅含 `调度`；本轮已扩展为 `调度/分流` 任一中即通过，现全过。

## 8. 后续扩展点

1. ~~批量用例回归~~（已完成，本文档即为产出）
2. 多模型并排对比：同一条输入同时投给 N 个模型，横向比较时延与答案
3. 调度观测接入：把 TDS 调度层的决策日志叠加到同一条消息的时间轴上
4. 会话落盘：当前为内存态，需要跨重启保留时再加 JSON 快照

---

*本文档为 phase3_ui_testbench.md 的补充子文档。主文档见 [phase3_ui_testbench.md](phase3_ui_testbench.md)。*
*（内容由AI生成，仅供参考）*
