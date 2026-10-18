# TDS 性能证据收集框架 & 竞品对比报告

> 目标：为 TDS SDK 上架提供可信的性能数据、竞品横向对比、可复现的基准测试。
> 定位：SDK 可售卖的核心证据——性能提升度、资源节省率、成本优势。

---

## 一、竞品定位与对比维度

| 维度 | TDS（我们的） | llama-swap | vLLM | ComfyUI-MultiGPU | ComfyUI 原生 |
|------|---------------|------------|------|-------------------|---------------|
| **核心机制** | 八卦分类预判 + 四象压力 + 三层存储 | 透明代理模型热切换 | PagedAttention 分页显存管理 | 层级卸载到 RAM/VRAM | 固定内存调度器（DynamicVRAM） |
| **调度粒度** | 块级（Block）：权重层 / KV 分页 / 前缀段 | 模型级（整模型加载/卸载） | 注意力键值分页（Page） | 层级（Block）卸载 | 模型级（整模型加载/卸载） |
| **资源复用** | 块级复用：同一块跨请求复用，避免重复加载 | 无复用，每次请求加载完整模型 | 跨请求 KV 复用（PagedAttention） | 层级卸载到 RAM，避免重复加载 | 无复用，每次请求加载完整模型 |
| **宿主无关** | 引擎零宿主依赖，适配层收敛 | 依赖 llama-server 特定端点 | 依赖 vLLM 推理服务 | 依赖 ComfyUI 节点系统 | 仅限 ComfyUI |
| **显存效率** | 预估：块级复用 + 三层存储 → 30-50% 显存节省 | 无优化，整模型驻留 | 2-4x 吞吐提升（PagedAttention） | 层级卸载 → 节省 20-30% | 固定调度，无复用 |
| **推理延迟** | 块预取 + 分类预判 → 降低首次请求延迟 | 整模型加载 → 首包延迟长（秒级） | 低延迟（PagedAttention 优化） | 层级切换 → 延迟增加 | 无优化 |
| **多模型并发** | 八卦域 + 四象压力 → 智能并发调度 | 不支持并发 | 天然支持多模型并发 | 支持多模型，但无智能调度 | 不支持并发 |

---

## 二、性能证据收集方案

### 2.1 基线测量（Baseline）
- **目标**：建立无 TDS 时的性能基线
- **指标**：
  - 模型加载时间（秒）
  - 首包延迟（TTFT，ms）
  - 峰值显存使用（MB）
  - 内存碎片率（%）
  - 请求吞吐量（tokens/s）

### 2.2 TDS 挂载性能（Attached）
- **目标**：TDS 挂载后相同负载下的性能变化
- **指标**：
  - 调度开销（ms/请求）
  - 块预取成功率（%）
  - 块复用率（%）
  - 显存节省率（%）
  - 请求延迟变化（%）

### 2.3 竞品对比测试（Competitive）
- **目标**：相同负载下 TDS vs llama-swap vs vLLM vs ComfyUI-MultiGPU
- **指标**：
  - 吞吐量对比（tokens/s）
  - 资源使用效率对比
  - 成本效益分析（单位请求成本）
  - 扩展性（并发模型数量）

---

## 三、性能数据收集模块设计

### 3.1 模块结构
```
tds/benchmarks/
├── baseline.py          # 基线测量
├── attached.py          # TDS 挂载后测量
├── competitive.py       # 竞品对比
├── report.py            # 报告生成
└── __init__.py
```

### 3.2 核心接口
```python
from tds.benchmarks.baseline import measure_baseline
from tds.benchmarks.attached import measure_attached
from tds.benchmarks.competitive import compare_competitors

# 基线测量
baseline = measure_baseline(task_desc="你的描述")
print(f"基线峰值显存: {baseline.peak_vram_mb:.0f} MB")

# TDS 挂载后测量
attached = measure_attached(runtime, task_desc="你的描述")
print(f"TDS 显存节省: {attached.memory_savings_pct:.1f}%")

# 竞品对比
results = compare_competitors(task_desc="你的描述", competitors=["llama-swap", "vLLM"])
results.print_report()
```

### 3.3 性能指标定义
- **性能提升度**：`(基线延迟 - TDS延迟) / 基线延迟 * 100%`
- **显存节省率**：`(基线峰值显存 - TDS峰值显存) / 基线峰值显存 * 100%`
- **块复用率**：`复用块数 / 总块请求数 * 100%`
- **调度开销**：`TDS调度耗时 / 总请求耗时 * 100%`

---

## 四、上架准备清单

### 4.1 性能数据验证
- [ ] 基线测量已完成（无 TDS）
- [ ] TDS 挂载测量已完成
- [ ] 竞品对比已完成（至少 2 个竞品）
- [ ] 性能数据可复现（提供测试脚本）

### 4.2 文档准备
- [ ] 性能证据文档（本文档）
- [ ] 测试报告（自动生成）
- [ ] 竞品对比表格

### 4.3 上架材料
- [ ] 性能白皮书（基于证据数据）
- [ ] 成本效益分析
- [ ] 用户体验对比

---

## 五、快速开始

```bash
# 1. 采集基线
python -m tds.benchmarks.baseline --task "中国的首都是哪座城市" --repeat 5

# 2. 挂载 TDS 后测量
python -m tds.benchmarks.attached --runtime "path/to/runtime" --task "你的描述" --repeat 5

# 3. 竞品对比
python -m tds.benchmarks.competitive --task "你的描述" --competitors llama-swap,vLLM
```

---

## 六、数据记录格式

```json
{
  "timestamp": "2026-10-09T10:00:00",
  "baseline": {
    "peak_vram_mb": 12000,
    "avg_latency_ms": 500,
    "throughput_tps": 50
  },
  "attached": {
    "peak_vram_mb": 8000,
    "avg_latency_ms": 450,
    "memory_savings_pct": 33.3,
    "block_reuse_pct": 25.0
  },
  "competitive": {
    "llama-swap": { "peak_vram_mb": 11000, "avg_latency_ms": 1200 },
    "vLLM": { "peak_vram_mb": 9000, "avg_latency_ms": 400 }
  }
}
```

---

*本文档为性能证据收集框架的核心文档。详细实现见 tds/benchmarks/ 模块。*