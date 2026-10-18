<<<<<<< HEAD
﻿# TDS 太极调度层 · 上架主文档（README）
=======
# TDS 太极调度层 · 上架主文档（README）
>>>>>>> origin/main

> 版本: 0.9.0-rc | 日期: 2026-10-18 | 状态: 上架预备完成（真机对照 + wheel 签名已齐），待实际上架
> 校验: 本目录 CHECKSUMS.txt 含全部发布物 SHA256；wheel 另附 Ed25519 自有签名（公钥 tds_core_public.pem）

## 1. 一句话定位

TDS（太极调度层）是**宿主无关的显存调度 SDK**：以太极理论逻辑（阴阳分层、三才存储、四象压力、八卦域分类）做全域淘汰序推演与动态显存预算，让 12GB 级消费卡跑 14GB 级模型，且无宿主硬依赖。

## 2. 核心卖点（全部本机实测，口径见证据包）

| 卖点 | 数字 | 证据 |
|------|------|------|
| 预算自适应无崩溃 | AUTO=0.5 档 3/3 轮完成，256MiB 修复档 3/3 轮完成 | 实机汇总 20261018 |
| 抖动收敛 | AUTO CV 0.161（修复前 192MiB CV 0.614） | 同上 |
| 峰值显存受控 | AUTO 峰值 6868MiB（较无预算省 36%） | 同上 |
| 命中率提升 | 仿真三档 +7.6 ~ +9.9pp | evidence 20261018 |
| IO 开销节省 | 仿真三档 -14.0% ~ -18.7% | 同上 |
| OOM 防护 | 仿真三档 0 OOM；`_BUDGET_FAILED` 降级兜底 | 同上 |

## 3. 快速集成（宿主无关契约）

```python
from tds.api import TDSRuntime
from tds.adapters import YourAdapter  # 实现 BaseAdapter 契约即可

rt = TDSRuntime(config=cfg)
rt.register_adapter(YourAdapter(...), "your-host")
rt.attach("your-host")
# 每次推理决策前: rt.run_task(...) -> {step, primary, scores, pressure, action, plan, applied}
```

适配层只依赖 5 个契约方法（load/unload/prefetch/vram_state/io_cost），接 ComfyUI、A1111、Fooocus 或自研后端同等成本。

> 注：`tds.adapters` 为随宿主部署的可选配件，不随 `tds-core` SDK 包分发（见分发约束）；上例仅示意宿主侧契约，实际由集成方按 BaseAdapter 契约自实现。

## 5. 安装与完整性校验

```bash
# 安装 SDK（宿主无关内核，零第三方运行时依赖）
pip install tds-core

# 校验发布包完整性（三重校验：SHA256 + 自有签名）
# 1) SHA256 对照 CHECKSUMS.txt
# 2) 用公钥验证 wheel 的 detached 签名
# 3) 核对公钥指纹（首次信任后固化）
```

公钥指纹（Ed25519 公钥 SHA256）: `C33CA76AEBA8B861C9CBB419472C213D504E2DF92B761824908AB34C56FFAC0A`

## 6. 对标范围与竞品矩阵

对标: LM Studio / Ollama / vLLM / llama-swap / VibeQwen / ComfyUI-MultiGPU / ComfyUI 原生。

结论（能力矩阵，定性判断）: TDS 在跨模型调度、语义预判、分层卸载、KV 分页、宿主无关、零依赖内核、OOM 防护 7 维全 Y；竞品均为部分能力。竞品量化数字仅列公开引用，非本机实测，不混入 TDS 实测区。详见 evidence 报告。

## 7. 分发渠道（建议，上架前可再调）

| 渠道 | 载体 | 说明 |
|------|------|------|
| PyPI | tds-core（SDK） | 宿主无关包，`pip install tds-core` |
| GitHub Release | 源码 + wheel + 证据包 | 附 CHECKSUMS.txt 与 wheel Ed25519 自有签名 |
| ComfyUI 生态 | custom_nodes 插件 | 经 ComfyUI-Manager 分发，即现有 lmsl-smart-offload 路线 |
| 官网/文档站 | 对比页 + 集成文档 | 承接公众号流量（中华传统文化 + 调度 IP 双叙事） |

约束: ui/bench/chat/harness 不随可售包分发；可售 IP 无宿主硬依赖；不借第三方知识产权。

## 8. 证据包清单（本目录）

- `performance_evidence_20261018_090545.md / .json` — 仿真台多维压力 + 竞品矩阵（mode=sim + 引用）
- `benchmark_auto_vs_native_20261018.md` — AUTO 档 vs ComfyUI 原生 offload 真机对照（本机实测）
- `tds_core-0.2.0-py3-none-any.whl` — SDK 构建产物（pip 可装，23 模块导入验证通过）
- `tds_core-0.2.0-py3-none-any.whl.sig` — wheel Ed25519 detached 签名（自有密钥，私钥本机保管）
- `tds_core_public.pem` — 校验公钥（指纹见第 5 节）
- `TDS_上架证据报告.md` / `tds_evidence.json` — 前期证据
- `CHECKSUMS.txt` — 全部发布物 SHA256

真机证据另见工作区 `output\TDS_实机性能证据汇总_20261018.md`、`output\TDS_竞品对比调研_20261018.md`。

## 9. 后续里程碑

- [x] 1. AUTO 档真机复测 + 与 ComfyUI 原生 offload 真机对照（mean -7.75%，stdev 上升如实披露）
- [x] 2. 打包 wheel + Ed25519 自有签名发布流程固化
- [ ] 3. 渠道实际上架（PyPI / GitHub Release / ComfyUI 生态 / 官网文档站）
*（内容由AI生成，仅供参考）*
