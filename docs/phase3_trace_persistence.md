---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 350ab7144af9618434cdce7bfa6d84e8_d6de59e9c31911f18019525400248c00
    ReservedCode1: w0O+NNKx+/BQziuc+vii81QLpuCOEJYS2QHd/WqAPeuvWLYmEilDs3vXuxuGZY8P741EoRT5Fww1p8HROTS3gpQh0TxmqP/Pd1B8nVedY3/y1qEFXTy7gcwWoLm3TBvX8vvAYCJK9g5E8kqJxxTl8C1NY/Gbsv7dCRo1Ts5DZJGcTNXGuzNqBI+EWC0=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 350ab7144af9618434cdce7bfa6d84e8_d6de59e9c31911f18019525400248c00
    ReservedCode2: w0O+NNKx+/BQziuc+vii81QLpuCOEJYS2QHd/WqAPeuvWLYmEilDs3vXuxuGZY8P741EoRT5Fww1p8HROTS3gpQh0TxmqP/Pd1B8nVedY3/y1qEFXTy7gcwWoLm3TBvX8vvAYCJK9g5E8kqJxxTl8C1NY/Gbsv7dCRo1Ts5DZJGcTNXGuzNqBI+EWC0=
---

# Phase3 · 决策轨迹与会话落盘

> 本篇记录两项可观测性 / 持久化能力：调度层的**决策轨迹**（属可售引擎，
> `tds/core/trace.py`）与测试台的**会话落盘**（属知识分支，`tds/ui/session.py`）。

## 1. 决策轨迹（可售 IP）

### 1.1 为什么需要

调度层此前的可观测手段只有 `Metrics` 计数器（命中/淘汰/压力跳过的**累计值**），
无法回答"第几步做了什么决策、为什么"。归因抖动、复盘回退、标定阈值都需要一条
**按步有序的时间线**。

### 1.2 设计

| 维度 | 约定 |
| --- | --- |
| 依赖 | 仅标准库（`collections` / `time` / `dataclasses`），随 `tds-core` 分发 |
| 容量 | 固定容量环形缓冲，由 `TDSConfig.trace_maxlen` 控制（默认 200） |
| 内容 | 只记决策元数据：步号 / 主域 / 四象压力 / 动作 / 计划摘要 / 是否落地 / 耗时 |
| 正文 | **不落任何宿主态正文**，跨宿主复用无泄漏风险 |

环形缓冲的意义：长跑进程内存占用恒定（`maxlen` 封顶），且 `total` 与 `dropped`
保留"被淘汰条数"，不会因为裁剪而丢失统计口径。

### 1.3 接口

```python
sch = Scheduler(config=TDSConfig(trace_maxlen=200))

# 1) 端到端调用自动埋点
sch.run_task("生成一段视频", blocks=blocks, vram_state=state)

# 2) 自定义链路手动埋点
sch.record_decision(primary="jiansuo", pressure="low", action="prefetch",
                    plan="Plan(prefetch=3)", applied=True, latency_ms=1.2)

# 3) 读取
sch.trace_entries(limit=10)   # List[TraceEntry]
sch.trace_lines(limit=10)     # 单行文本视图
sch.trace_report()            # 聚合摘要
```

`trace_report()` 返回：`size / maxlen / total / dropped / applied_rate /
latency_ms{avg,max} / by_action / by_pressure / by_domain`。

单行视图示例：

```
#12   [shipin/high]     skip             Plan(demote=4)      18.31ms ok
```

### 1.4 与 Metrics 的分工

- `Metrics`：**累计口径**，回答"一共发生多少次"，用于阈值标定与看板；
- `DecisionTrace`：**序列口径**，回答"第几步、按什么状态做了什么"，用于归因。

两者互不覆盖，`Scheduler.report()` 的返回结构保持不变（无回归）。

## 2. 会话落盘（知识分支）

### 2.1 设计

| 维度 | 约定 |
| --- | --- |
| 依赖 | 仅标准库 `json` / `os` / `pathlib`，不引入数据库 |
| 写入 | **原子替换**：先写同目录 `.tmp`，再 `os.replace`，避免半截文件 |
| 版本 | 快照含 `version`（当前 `SNAPSHOT_VERSION = 1`）与 `saved_at` |
| 裁剪 | `with_content=False` 归档时清空图片 base64，仅保留元信息 |

### 2.2 接口

```python
store = SessionStore()
store.create(model="qwen2.5vl:7b")
store.save("D:/snap/chat.json")            # 无后缀自动补 .json，目录自动创建
SessionStore().load("D:/snap/chat.json")   # 默认替换现有内存态
store.load(path, merge=True)               # 合并：同 id 覆盖，其余保留
store.from_dict(raw, merge=False)          # 内存态导入（非法顶层类型抛 ValueError）
```

消息、附件、统计（`stats`）与采样参数全部往返一致，`history_for_model()` 的
OpenAI 协议形态（带图时 `content` 为数组）在恢复后保持不变。

## 3. 依赖洁净度审计（本轮复核）

对 `tds/` 全包做静态扫描（`ast` 解析 import，比对 `sys.stdlib_module_names`）：

| 范围 | 结论 |
| --- | --- |
| `tds/core`（含新增 `trace.py`） | **零第三方依赖**，仅标准库 |
| `tds/ui`（含 `session.py` / `bench.py`） | **零第三方依赖**，仅标准库 |
| `tds/adapters`、`tds/kv` | 依赖宿主 `torch` / `comfy`（预期，平台侧） |
| `tds/registry` | 依赖 `psutil` / `requests`（预期，硬件与远端模型库） |

遗留小项：`tds/integrations/comfyui_bridge.py` 首字节为 U+FEFF（BOM），
Python 解释器可容忍，但 `ast.parse` 等静态解析会失败，后续清理为无 BOM 的 UTF-8。

## 4. 测试

| 文件 | 覆盖点 |
| --- | --- |
| `tests/test_core_trace.py` | 环形缓冲淘汰、`total`/`dropped` 口径、scores 拷贝语义、聚合与空态、`clear` 语义、maxlen 下限、单行视图、`run_task` 自动埋点、配置容量生效、冻结态动作、手动埋点、调度器间轨迹隔离、`report()` 无回归 |
| `tests/test_ui_session_store.py` | 快照版本号、字段与图片往返、OpenAI 协议形态保持、`with_content=False` 裁剪、后缀自动补全与无 `.tmp` 残留、嵌套目录创建、替换 / 合并语义、非对象报错、缺文件报错、残缺记录容错、空库往返 |

全量结果：**189 passed, 1 skipped, 16 subtests passed**（skip 为依赖宿主
`torch` 的 KV 分页引擎测试，属预期跳过）。

## 5. 后续可扩展点

1. 测试台时间轴叠加轨迹：把 `Scheduler.trace_lines()` 与对话消息按时间轴对齐，
   形成"调度—模型"联合视图（需先完成测试台与调度层的运行时接线）；
2. 快照自动落盘策略：进程退出 / 定时归档，避免依赖手动 `save()`；
3. 轨迹导出：把 `to_list()` 落为 JSONL，供离线分析与阈值标定。
*（内容由AI生成，仅供参考）*
