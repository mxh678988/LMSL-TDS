---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 350ab7144af9618434cdce7bfa6d84e8_0ee33137c2f611f1887c525400de85a5
    ReservedCode1: vrsOpZo9D7Gfo+hs3D1f6EB9q8HIwV9oTOCInqGOHCaUuUAe3nB3VgqQkNpsqJ/XINNsf60BXKH9AQPgmXdfivfybNPwTaEptpFBmwhkOnBLN934of/qkZrIBo7clZErmR5gP2gI1+BZFJDbuCHSR5Q4eQh9nm7NpNjqxv7kQTOTMbjY+cE4ldCg09s=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 350ab7144af9618434cdce7bfa6d84e8_0ee33137c2f611f1887c525400de85a5
    ReservedCode2: vrsOpZo9D7Gfo+hs3D1f6EB9q8HIwV9oTOCInqGOHCaUuUAe3nB3VgqQkNpsqJ/XINNsf60BXKH9AQPgmXdfivfybNPwTaEptpFBmwhkOnBLN934of/qkZrIBo7clZErmR5gP2gI1+BZFJDbuCHSR5Q4eQh9nm7NpNjqxv7kQTOTMbjY+cE4ldCg09s=
---

# Phase3 · 对话交互测试台（tds.ui）

> 定位：**知识分支**。用于全方位测试 TDS 调度层所对接的文本 / 多模态模型能力，
> **不计入可售 IP**，不随 `tds-core` 发行包分发。

## 1. 为什么需要它

哑巴后端（无 UI、宿主无关、可售 IP）是主攻方向，但引擎"跑得对不对、模型"用起来
怎么样"需要一块可直接上手的观测面板。此前的观测手段只有日志与环境变量，无法区分
"调度错了"还是"模型本身弱"。测试台补齐的正是这一段：把模型的输入输出、时延、
吞吐、图片理解结果摊开在同一个页面里。

## 2. 设计边界

| 维度 | 约定 |
| --- | --- |
| 依赖 | 零第三方运行时；仅标准库 `http.server` / `urllib` / `json` |
| 外部程序 | 不调用任何外部程序；仅通过 HTTP 访问已运行的推理服务端点 |
| 前端 | 随包分发的**单文件 HTML**，无 CDN、无构建步骤、无框架 |
| 网络暴露 | 默认只监听 `127.0.0.1`，不对外 |
| IP 归属 | 属知识分支，不进入可售引擎包；`pyproject.toml` 的 `packages` 不含 `tds.ui` |

前端与引擎解耦的理由与 adapters 一致：面板会不断换形态（对话流、图片、审计面板），
一旦并入引擎层，宿主无关性与可售 IP 的洁净度都会被拖累。

## 3. 结构

```
tds/ui/
├── __init__.py      # 子包出口与定位声明
├── backend.py       # 模型后端访问层：OpenAI 兼容协议 + SSE 流式解析 + Ollama 画像
├── session.py       # 会话模型：消息、图片附件、多会话容器 + JSON 快照原子落盘
├── server.py        # HTTP 服务：静态面板 + JSON API + SSE 代理
├── cli.py           # 命令行入口：serve / probe / bench
├── bench.py         # 固定用例回归引擎（用例集、判定、执行器、报告）
├── fixtures.py      # 回归测试图（程序内生成，无需外部素材）
└── web/index.html   # 单文件前端（对话流 + 侧边栏 + 状态条）
```

回归引擎的用法与基线见 [phase3_bench.md](phase3_bench.md)；
决策轨迹与会话落盘的设计见 [phase3_trace_persistence.md](phase3_trace_persistence.md)。

## 4. 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 面板页 |
| GET | `/api/status` | 后端连通性 + 硬件画像（显存/内存）+ 会话数 |
| GET | `/api/models` | 模型清单（含体积、量化、多模态/云端标记） |
| GET | `/api/sessions` | 会话列表 |
| POST | `/api/sessions` | 新建会话 |
| GET | `/api/sessions/<id>` | 会话详情（含历史与统计） |
| DELETE | `/api/sessions/<id>` | 删除会话 |
| POST | `/api/sessions/<id>/clear` | 清空会话消息 |
| POST | `/api/sessions/<id>/params` | 更新模型 / system / temperature / top_p / max_tokens |
| POST | `/api/chat` | 对话主通道，`text/event-stream` 逐块回传 |

SSE 事件：`start`（模型与入参回执）、`delta`（增量文本）、`done`（统计与消息回执）、
`error`（失败原因，不抛出异常中断连接）。

图片输入：前端读文件为 data URI，服务端解码为内联附件，转成 OpenAI `content` 数组
的 `image_url` 形式发给后端；单条消息上限 4 张、单张上限 12 MB。

## 5. 用法

```bash
# 启动面板（默认后端 http://127.0.0.1:11434）
python -m tds.ui.cli serve --open

# 换后端 / 指定新会话默认模型
python -m tds.ui.cli serve --backend http://127.0.0.1:11434 --model qwen2.5vl:7b

# 只做后端自检，不启动服务
python -m tds.ui.cli probe
```

Windows 下可直接双击 `tools/start_ui.bat`。

## 6. 首轮实测结论（2026-10-08）

| 项目 | 结果 |
| --- | --- |
| 后端探测 | Ollama 0.40.1 在线，识别 12 个模型，多模态标记正确（llava:latest、qwen2.5vl:7b） |
| 硬件画像 | RTX 3060 12288 MB（空闲 9682 MB），cc8.6 |
| 文本对话 | qwen2.5:3b 正常流式返回，无错误 |
| 多模态对话 | qwen2.5vl:7b 准确识别程序生成图片中的蓝底、绿色方块与红色圆形 |
| 会话回读 | 标题自动生成、消息与附件计数正确 |
| 面板渲染 | 无头浏览器截图确认左栏 / 对话区 / 状态条渲染完整，无报错 |

> 备注：冷启动首包时延（首次加载模型）显著高于热态，评估模型能力应取热态数据。
> 视觉模型对形状位置的表述存在偏差（如把居中圆形描述为"右下角"），属模型自身
> 能力问题，不是链路问题——这正是测试台要暴露的信息。

## 7. 后续可扩展点

1. ~~批量用例回归：把一组固定提示词 + 图片做成用例集，一键跑完并留档对比；~~
   已完成，见 [phase3_bench.md](phase3_bench.md)；
2. 多模型并排对比：同一条输入同时投给 N 个模型，横向比较时延与答案；
3. 调度观测接入：把 TDS 调度层的决策日志（命中分类器、显存预判、KV 分页动作）
   叠加到同一条消息的时间轴上，形成"调度—模型"联合视图；
   引擎侧序列化轨迹已就绪（`Scheduler.trace_lines()` / `trace_report()`），
   待补齐测试台与调度层的运行时接线，见 [phase3_trace_persistence.md](phase3_trace_persistence.md)；
4. ~~会话落盘：当前为内存态，需要跨重启保留时再加 JSON 快照。~~
   已完成：`SessionStore.save/load` 原子落盘与恢复，见 [phase3_trace_persistence.md](phase3_trace_persistence.md)。
*（内容由AI生成，仅供参考）*
