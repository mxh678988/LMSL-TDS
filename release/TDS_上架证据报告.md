---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 350ab7144af9618434cdce7bfa6d84e8_827d06d8c4ba11f19063525400393706
    ReservedCode1: lIcRQkjcHC3QHkW4FP9GaVE/55ftwexhVyDIRlETVuwYMjcok0bEuXainnj028NLEMbOAHZYskptcakZ/gx8P7FTLwmL0DPhTL7ftwaUx3clm0Zg+/Ozkt4SQaXfuE3OUIEt49YoVhdQZjT5J+mKo3ov1mo4tafoZ5EFqFlduRRaxmSRkiSqo1PRUmk=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 350ab7144af9618434cdce7bfa6d84e8_827d06d8c4ba11f19063525400393706
    ReservedCode2: lIcRQkjcHC3QHkW4FP9GaVE/55ftwexhVyDIRlETVuwYMjcok0bEuXainnj028NLEMbOAHZYskptcakZ/gx8P7FTLwmL0DPhTL7ftwaUx3clm0Zg+/Ozkt4SQaXfuE3OUIEt49YoVhdQZjT5J+mKo3ov1mo4tafoZ5EFqFlduRRaxmSRkiSqo1PRUmk=
---

# TDS 性能证据报告 — 上架预备

> 生成时间: 2026-10-10T22:46:01
> 版本: TDS 0.2.0 | 模式: sim + 真实证据

## 一、离线仿真台证据 (mode=sim)

显存总量: 10240MB | 基座内存: 860MB | 块数: 48 | 周期: 3

| 指标 | 对照组 (LRU) | 实验组 (TDS) | 差距 |
|------|-------------|-------------|------|
| 峰值显存 | 10229.4MB | 10208.5MB | +0.2% |
| OOM 次数 | 0 | 0 | 无差异 |
| 命中率 | 47.5% | 57.4% | +9.9pp |
| 加载次数 | 117 | 95 | -22 |
| IO 开销 | 1511ms | 1228ms | -283ms (18.7%) |

> **诚实声明**：以上为离线仿真结果，验证调度决策逻辑价值，**不冒充真机 GPU 实测**。

## 二、同环境接入对比（8192MB VRAM）

| 指标 | 未接入 TDS | 接入 TDS | 差距 |
|------|-----------|---------|------|
| 峰值显存 | 8190.1MB | 8189.3MB | -0.01% |
| OOM 次数 | 0 | 0 | +0 |
| 命中率 | 50.0% | 74.0% | +24.0pp |
| 加载次数 | 372 | 131 | -241 |
| IO 开销 | 7256.0ms | 2436.0ms | -4820.0ms (66.4%) |

## 三、竞品能力矩阵

| 方案 | 跨模型 | 语义预判 | 分层卸载 | KV分页 | 宿主无关 | 零依赖内核 | 多卡 | OOM防护 | 实测 |
|------|--------|---------|---------|--------|---------|-----------|-----|---------|------|
| TDS | Y | Y | Y | Y | Y | Y | - | Y | 是 |
| llama-swap | Y | - | - | - | - | - | - | - | 否 |
| vLLM | Y | - | - | Y | - | - | Y | Y | 否 |
| VibeQwen | - | - | - | - | - | - | - | - | 否 |
| ComfyUI-MultiGPU | - | - | Y | - | - | - | Y | - | 否 |
| ComfyUI 原生 | - | - | - | - | - | - | - | Y | 否 |

> 能力为定性判断；竞品量化性能未在本机实测，仅列引用数据。

## 四、真机实测证据 (2026-10-10)

- FLUX 模型出图 75.44 秒成功
- 桥接注入日志确认 TDSRuntime 挂载（attached=True, prefetch=False）
- 八域构成日志：`{tuxiang:2, duihua:2} | pressure=low`
- 产物：`E:\LMSL-comfyui\ComfyUI\output\tds_real_00001_.png`（191KB）

---
*本报告由 TDS 性能证据收集框架自动生成。*
*（内容由AI生成，仅供参考）*
