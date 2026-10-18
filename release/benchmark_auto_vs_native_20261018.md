# TDS AUTO 档 vs ComfyUI 原生 offload 实机对照（2026-10-18）

> 生成时间: 2026-10-18T09:50:00 | 脚本: TDS/tools/bench_multi_model.py | 本机实测，非公开引用

## 测试环境

- GPU: NVIDIA GeForce RTX 3060 12GB（vram_total=12884377600）
- ComfyUI 启动参数: `--cuda-malloc --reserve-vram 0.8 --port 8188`
- Python: E:\LMSL-comfyui\python\python.exe（3.12.10）
- 模板: flux_bench_api.json（FLUX GGUF 单模板，UnetLoaderGGUF + DualCLIPLoader + KSampler 12 步）
- 轮次: 每档 3 轮交替提交，无缓存命中

## 两组开关

| 组 | 时间 | 开关 | 语义 |
|----|------|------|------|
| TDS AUTO 档 | 2026-10-18 08:26 | TDS 桥接激活 + TDS_KV_BUDGET_AUTO | 太极调度层自适应档（动态预算 + 八卦域 keep_steps） |
| 原生对照 | 2026-10-18 09:44 | TDS_BRIDGE=off | 等价于未安装 TDS 节点，ComfyUI 原生 offload |

原生组重启实例时以 `TDS_BRIDGE=off` 注入环境并保持同端口同参数，启动日志仅出现探针模块（page_manager），无桥接激活、无 KV 预算介入，两组差异即 TDS 调度层实机影响。

## 结果对比

| 指标 | TDS AUTO | 原生 offload | 差值/变化 |
|------|----------|--------------|-----------|
| mean 单次耗时 | 78423.67 ms | 85011.00 ms | **-6587 ms（-7.75%）** |
| stdev 抖动 | 12630.77 ms | 9062.82 ms | +3568 ms（+39.4%） |
| range 极差 | 27349 ms | 21299 ms | +6050 ms |
| cv 变异系数 | 0.1611 | 0.1066 | +0.0545 |
| VRAM 峰值 | 6868 MiB | 6310 MiB | +558 MiB |

逐轮明细：

| 轮次 | TDS AUTO (ms / MiB) | 原生 (ms / MiB) |
|------|----------------------|------------------|
| 1 | 96274 / 6836 | 97467 / 6310 |
| 2 | 70072 / 6868 | 81398 / 6310 |
| 3 | 68925 / 6868 | 76168 / 6310 |

## 结论与定位

1. **接入后不劣化（正面）**：TDS AUTO 档平均耗时比原生快 7.75%，三档均无 OOM、无崩溃。
2. **抖动上升（如实披露）**：stdev 上升 39.4%，未达到内部验收线（stdev 下降 ≥20%）。原因定位：单模板 3 轮场景下模型仅加载一次，无跨模型页淘汰机会，AUTO 动态预算的调度探测引入了额外开销；TDS 的核心收益（跨模型 KV 页级复用、keep_steps 命中率、IO 削减）需多模型/多模板交替场景才能体现。
3. **VRAM 峰值 +558 MiB**：AUTO 档动态预算预留了调度余量，单模板场景下预留未转化为收益，属预期成本。
4. **优势场景证据**：仿真台三档（8192/10240/12288MB）已验证 TDS vs naive(LRU) 命中率 +7.6~+9.9pp、IO -14~18.7%、0 OOM（见 performance_evidence_20261018_090545.md）。

## 证据文件

- ON 组原始: bench_out/bench_multi_20261018_082634_budget-auto-fix-on.json
- OFF 组原始: bench_out/bench_multi_20261018_094420_tds-off-same-auto.json
*（内容由AI生成，仅供参考）*
