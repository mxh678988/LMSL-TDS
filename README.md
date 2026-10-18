---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 350ab7144af9618434cdce7bfa6d84e8_ed2c6f76c0b611f1887c525400de85a5
    ReservedCode1: BWZosjG/pVF++lI/xYS0x6OCnpMSUzLHkHIqkUBEhssXmf9SUO7E913PepbflKuV9Kxp7cXURkvjLe1DMbe4cYH/QN9xtCQe1kh6Tx/S7CIwizqf4uB+YcSJMGTAG2KAvmi1nIU3Dv/cpj4vEvC2snuDZkFflfgaYk+F0ecXRWgCgWt2mxadGPsS7No=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 350ab7144af9618434cdce7bfa6d84e8_ed2c6f76c0b611f1887c525400de85a5
    ReservedCode2: BWZosjG/pVF++lI/xYS0x6OCnpMSUzLHkHIqkUBEhssXmf9SUO7E913PepbflKuV9Kxp7cXURkvjLe1DMbe4cYH/QN9xtCQe1kh6Tx/S7CIwizqf4uB+YcSJMGTAG2KAvmi1nIU3Dv/cpj4vEvC2snuDZkFflfgaYk+F0ecXRWgCgWt2mxadGPsS7No=
---

# 太极调度层 TDS（Taiji Dispatch Scheduler）

> 大模型「权重—显存」动态调度中间层
> 在模型与显存之间建立统一的**分类 / 分层 / 预判**调度机制，实现按需加载与预取，
> 在有限显存下最大化可用上下文与推理性能。

- 版本：v0.3.0（含自有签名校验安全模块）
- 版权：BY LAOMENG 网络工作室（孟祥辉）· 独立实现

## 一、核心理念

| 用户直觉 | 工程落地 |
| --- | --- |
| 二进制控制 | 位级压缩与分页（KV 量化 / 块级管理） |
| 八卦分类 | 八域语义分类索引（乾坤/对话/检索/代码/图像/视频/工具/其他） |
| 阴阳 | 冷热分层（阳=热数据驻留显存；阴=冷数据下沉内存/SSD） |
| 预判 | 统计热度 + 启发式预取（Belady 近似） |

## 二、三层架构

```
任务/对话输入
   ↓
[语义感知层] 八域分类 -> 任务画像          (core/classifier.py)
   ↓
[调度决策层] 冷热权衡 / 预取 / 淘汰 / 压力保护 (core/planner.py, scheduler.py, memory.py)
   ↓
[执行接入层] 框架 Hook（ComfyUI / llama.cpp / Mobile）  (adapters/*)
   ↓
存储层级：GPU 显存 -> CPU 内存 -> 本地 SSD   (core/storage.py)
```

## 三、跨平台设计

- `tds/core/`：**平台无关**的核心调度逻辑（自有知识产权）
- `tds/adapters/`：平台适配层，实现统一契约 `BaseAdapter`
  - PC：`ComfyUIAdapter`（custom_node 注入）、`LlamaCppAdapter`（llama.cpp / Ollama）
  - 移动端：`MobileAdapter`（Android，预留）

## 四、目录结构

```
TDS/
├── README.md
├── LICENSE
├── requirements.txt
├── tds/
│   ├── __init__.py
│   ├── config.py
│   ├── core/
│   │   ├── classifier.py     # 八卦八域分类
│   │   ├── memory.py         # 显存预算 / 压力保护
│   │   ├── storage.py        # 三级存储抽象
│   │   ├── planner.py        # 预扫描 + 预取计划
│   │   ├── scheduler.py      # 太极调度主控
│   │   └── metrics.py        # 指标埋点
│   ├── security/
│   │   └── verify.py         # 发布包自有签名校验（Ed25519 + SHA256）
│   └── adapters/
│       ├── base.py           # 统一契约
│       ├── comfyui_adapter.py
│       ├── llamacpp_adapter.py
│       └── mobile_adapter.py
├── plugins/
│   └── comfyui_node/         # ComfyUI custom_node 入口
├── tests/
│   └── test_core.py
└── docs/
    └── roadmap.md
```

## 五、快速开始

```bash
# 运行基础自测（零第三方依赖）
python -m unittest discover -s tests -v

# 校验发布包自有签名（签名即授权：验签通过 = 官方签名发布且未被篡改）
python -m tds.security.verify <wheel文件> <wheel.sig文件>

# 作为模块调用
from tds.security import verify_wheel
result = verify_wheel("tds_core-0.3.0-py3-none-any.whl", "tds_core-0.3.0-py3-none-any.whl.sig")
assert result["ok"]
```

## 六、发布包安全（v0.3.0）

- **授权形态**：随包分发公钥、离线校验；Ed25519 签名通过即视为官方授权发布（签名即授权）
- **三重校验**：SHA256 完整性 + Ed25519 验签 + 公钥指纹绑定（防公钥整体替换）
- **公钥指纹**：`C33CA76AEBA8B861C9CBB419472C213D504E2DF92B761824908AB34C56FFAC0A`
- 私钥由 BY LAOMENG 网络工作室本地保管，绝不随包分发
- 该模块零第三方运行时依赖；仅验签场景需要 `cryptography`（发布端与校验端均可用）

## 七、开发阶段

Phase 0 可行性验证 → Phase 1 ComfyUI MVP → Phase 2 KV 分层 → Phase 3 语义预判引擎 → Phase 4 通用化

## 八、版权与独立实现声明

本项目核心调度逻辑（`tds/core/` 三平面）为**独立设计实现，拥有完整自主知识产权**。
对底层推理引擎的对接，仅通过其**公开、开放的接口**完成，不复制、不内联任何第三方受版权保护的实现代码：

- llama.cpp（MIT License）— 仅通过其公开 API / GGUF 格式交互
- ComfyUI — 仅通过其官方 custom_node 扩展接口交互
- 移动端（NCNN / MNN / llama.cpp-mobile）— 仅通过统一 Adapter 契约对接
*（内容由AI生成，仅供参考）*
