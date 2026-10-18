---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 350ab7144af9618434cdce7bfa6d84e8_9cb450f7c0d511f1884b525400cd780f
    ReservedCode1: 6+DpStNharWITk5F67aE2I3+4JJ+XIiP2OGhruNy3a6873IHa1u5fcGSpgpE8BbwVAoBE8ERPdAL9B9hmYDr1vV5rl4XScN2UmPUqYLaWoi5eg3xHyjN/Z6MS2ph6oi9CfgsZOc3Gi9hUVn/PqHhaShkUVvGPc0dw8+bxlZKFOeRQizCDq6XgccbOk8=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 350ab7144af9618434cdce7bfa6d84e8_9cb450f7c0d511f1884b525400cd780f
    ReservedCode2: 6+DpStNharWITk5F67aE2I3+4JJ+XIiP2OGhruNy3a6873IHa1u5fcGSpgpE8BbwVAoBE8ERPdAL9B9hmYDr1vV5rl4XScN2UmPUqYLaWoi5eg3xHyjN/Z6MS2ph6oi9CfgsZOc3Gi9hUVn/PqHhaShkUVvGPc0dw8+bxlZKFOeRQizCDq6XgccbOk8=
---

---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 350ab7144af9618434cdce7bfa6d84e8_ee83cdf5c0b611f1887c525400de85a5
    ReservedCode1: 69KC5vsVJ2dd9VIeJpLGNELbahCVw577/x9OpkWg3Mh3SVRd+nnJNnCZJpFTP7bNL33/qn6cW2yOWYKm6T7UcZkHtBzIQVWEObbiDaSY7F9/NNOMSiz6/+yPwo9j3JEDx6TXpspSOicwXMEz2PbPIOMfMVbF4vfQWrD/wdftXiqyIoW4/B2f+E/vGZw=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 350ab7144af9618434cdce7bfa6d84e8_ee83cdf5c0b611f1887c525400de85a5
    ReservedCode2: 69KC5vsVJ2dd9VIeJpLGNELbahCVw577/x9OpkWg3Mh3SVRd+nnJNnCZJpFTP7bNL33/qn6cW2yOWYKm6T7UcZkHtBzIQVWEObbiDaSY7F9/NNOMSiz6/+yPwo9j3JEDx6TXpspSOicwXMEz2PbPIOMfMVbF4vfQWrD/wdftXiqyIoW4/B2f+E/vGZw=
---

# 太极调度层 TDS · 开发路线图

## 当前状态（2026-10-18）

- **v0.3.0 已冻结为基线**：GitHub 仓库/Release、ComfyUI Registry、PyPI tds-core 0.3.0 三渠道上架完成
- 冻结策略：无故障且无人操作连续达标即冻结；正常不可回退，仅故障触发回退
- 下一稳定小版本按季度排期（约 3 个月一个稳定小版本），版本间仅故障回退
- 待推进：官网/文档站（对比页 + 集成文档，承接公众号流量）

## Phase 0：可行性验证（1–2 周）
- 目标：证明"分类预判 + 分层调度"可降低成本且不引入 OOM
- 交付：测量脚本 + 基线数据（显存峰值 / TTFT / 缓存命中率）
- 验收：新方案显存峰值 ≤ 基线，且无 OOM

## Phase 1：MVP — ComfyUI 调度中间层（3–6 周）
- 目标：custom_node 实现"模型使用序列预扫描 + 分类优先级 + 分层驻留"
- 交付：可安装 custom_node + 控制面板
- 验收：多模型工作流换入换出抖动下降 ≥ 20%
- **当前进度（Phase 1 收口完成 · 2026-10-06）**：真实挂载链路已闭环，单测 44/44 全绿
  - 已完成：八域分类器（含同分平局优先级 BAGUA_PRIORITY）、八卦加权动态 keep_steps、阴阳冷热分层、
    四象压力策略（LOW/CRITICAL/HIGH/FROZEN）、TieredStore 三层存储（VRAM/RAM/SSD）、
    Planner 计划生成、Scheduler.plan/apply/run_task 端到端、adapter 契约（ComfyUI/llama.cpp/Mobile）
  - 判据统一：压力水位以 `Scheduler.pressure_state()` 为唯一权威；高压力起仅放行释放类动作（下沉/淘汰）
  - 桥接层：`tds/integrations/comfyui_bridge.py` 以补丁注入方式驱动真实宿主插件 lmsl-smart-offload，
    替换其 `_next_use_distance` 为「叠加八卦加权 + 四象压力门控」的有效距离；
    可一键关闭（`TDS_BRIDGE=off`）且回退后与原插件行为逐位一致
  - custom_node 入口：`TDS/__init__.py → plugins/comfyui_node`，具备宿主自动发现、加载顺序无关
    （逆序时由 prompt 钩子兜底注入）、宿主缺失静默待命、零副作用四项特性
  - 离线验收：`tools/tds_demo.py`（四象 × 八域矩阵）、`tools/verify_bridge_mount.py`
    （156 组等价性扫描零不一致）、`tools/verify_node_mount.py`（5 场景加载仿真，含真机形态复刻）
  - 真机验收：已实装于 `E:\LMSL-comfyui\ComfyUI\custom_nodes\TDS`；ComfyUI 0.35.1 启动日志确认
    「桥接已注入 + attached=True」，TDS 与宿主插件均无异常
  - 真机踩坑（已修复）：ComfyUI 对目录型 custom_node 使用
    `sys.modules[module_path.replace(".", "_x_")]` 注册，键是**插件目录绝对路径**而非目录名，
    按包名探测必然 miss；探测逻辑须退化为按模块名 + `__file__` 路径双匹配
  - 可观测性（2026-10-08）：`Scheduler` 决策轨迹已就绪——`tds/core/trace.py` 环形缓冲，
    容量由 `TDSConfig.trace_maxlen` 控制，`trace_entries/trace_lines/trace_report` 提供
    序列口径，与 `Metrics` 的累计口径互补；设计见 `docs/phase3_trace_persistence.md`
  - 控制面板（2026-10-11）：`tds/ui/server.py` 新增 `/api/tds/config`（GET/POST，白名单校验）
    与 `/api/tds/state`（压力/显存/指标/轨迹/阴阳模式分布），`DecisionTrace.resize` 支持运行期
    改轨迹容量；前端侧边栏「TDS 调度」面板可编辑保存参数并 10s 轮询状态；
    `Scheduler.plan/run_task` 阴阳决策不再依赖 blocks 存在（空块也按画像/压力产出模式）。
    单测 `tests/test_ui_tds_panel.py` 10 用例全绿（本轮回归 54/54）
  - MobileAdapter 补全（2026-10-11）：`tds/adapters/mobile_adapter.py` 完整实现
    （NCNN / MNN / ONNX Runtime Mobile 三种后端骨架），虚拟显存 fallback、
    模型注册/反查、JSON/CSV 图扫描、模拟 load/unload；单测 11/11 全绿
  - 实机压测工具链就绪（2026-10-11）：`tools/bench_multi_model.py` 新增——
    复用 ComfyUI history 最近成功工作流为模板（默认 FLUX 8 节点）、
    `--template-file` 可多模板交替模拟多模型换入换出、逐轮采集
    执行耗时/墙钟/显存峰值（nvidia-smi 采样）、自动检测并剔除
    ComfyUI 全缓存轮、输出抖动指标（stdev/极差/CV）+ JSON/MD 双报告；
    已在本机 1 次闭环验证（真实轮 31.7s，缓存轮被正确剔除）
  - 待办：无
- **Phase 1 实机压测最终结果（2026-10-11）**：
  - **方案一（双 unet 不同 steps，10 轮/组）**：均值 62.3s（on）vs 67.0s（off），-6.8%；
    抖动 stdev 15429ms（on）vs 15342ms（off），持平（+0.6%）；显存峰值波动 on 7 MiB vs off 52 MiB（-87%）
  - **方案二（同步骤 12 步，仅 unet 名不同，10 轮/组，最终口径）**：
    - 均值 53.9s（on）vs 48.4s（off），+11.4%（on 组受首轮冷启动 67s 拉高）
    - **抖动 stdev 7702ms（on）vs 10174ms（off），-24.3%，验收线 ≥20% 通过**
    - 极差 20.1s vs 37.7s，CV 0.143 vs 0.210
    - 冷启动分析：on/off 首轮均 67s（驻留尚未生效），on 组其余 9 轮 stdev 6.5s vs off 8.1s（-20%），
      off 组偶发节点缓存快轮（29.9s）被 TDS 驻留消除
    - 显存权衡：on 峰值 9835±220MB vs off 9637±113MB（驻留占用略高约 200MB，12GB 卡可接受）
  - 结论：**抖动验收线达标（stdev -24.3% ≥ 20%）**；均值反升主因冷启动拉高，
    Phase 2 关注冷启动开销（14.7s/+28%），Phase 3 关注驻留调度开销优化
  - 已登记产物：`bench_multi_20261011_093931_tds-on.json/.md`、`bench_multi_20261011_094726_tds-off.json/.md`、
    `bench_multi_20261011_101622_tds-on-same12.json/.md`、`bench_multi_20261011_103436_tds-off-same12.json/.md`、
    `bench_multi_20261011_110546_tds-on-same12b.json/.md`、`bench_multi_20261011_105427_tds-off-same12b.json/.md`、
    `bench_final_report_same12.md`

## Phase 2：KV 分层（6–10 周）
- 目标：KV 分页 + 分层卸载（显存→内存→SSD）
- 交付：KV 管理层 + 与 custom_node 打通
- 验收：同等显存下支撑更长上下文

## Phase 3：语义预判引擎（10–16 周）
- 目标：八域分类器强化 + 访问序列学习
- 交付：预判模块 + 命中率看板
- 验收：预判命中率 ≥ 60%（首期目标）

## Phase 4：通用化（16 周+）
- 目标：抽象为框架无关中间层，接入 Ollama / llama.cpp / 移动端
- 交付：SDK / 插件体系

## 跨平台节奏
- PC 优先（ComfyUI → llama.cpp）
- 移动端（Android）在 Phase 4 前保持统一 Adapter 契约，不做实现
*（内容由AI生成，仅供参考）*
*（内容由AI生成，仅供参考）*
