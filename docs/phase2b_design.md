---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 350ab7144af9618434cdce7bfa6d84e8_8e6cbba3c15c11f1887c525400de85a5
    ReservedCode1: UzwN/0Tou85nPMaaXCFZLzzF5D3xgyRSzsJIfQfgp5T2dhKFPvW2TuZhJWVOguQOELzItHaF9pjl6uZ9BqrpL4zzgOhdnTOwXmqvU2WNRIQdLEC7N3JbdOfSAVf8TjuINxhThjfo80Kg5CxMp+VNdqERZ0lByycVovATiHVZg0e5OowAdzLDHC22gwo=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 350ab7144af9618434cdce7bfa6d84e8_8e6cbba3c15c11f1887c525400de85a5
    ReservedCode2: UzwN/0Tou85nPMaaXCFZLzzF5D3xgyRSzsJIfQfgp5T2dhKFPvW2TuZhJWVOguQOELzItHaF9pjl6uZ9BqrpL4zzgOhdnTOwXmqvU2WNRIQdLEC7N3JbdOfSAVf8TjuINxhThjfo80Kg5CxMp+VNdqERZ0lByycVovATiHVZg0e5OowAdzLDHC22gwo=
---

# Phase 2B 设计：KV 分页换入换出（VRAM ⇄ RAM）

> 版本：v1（设计定稿）
> 环境：RTX 3060 12GB / ComfyUI（E:\LMSL-comfyui\ComfyUI）/ Python E:\LMSL-comfyui\python\python.exe
> 前置：Phase 2A（KV 只读可观测层）已闭环，`observer.py` 单测 20/20、全量回归通过。

---

## 1. 目标

在 KV Cache 维度上实现"分页换入换出"，使同等显存预算下可承载的上下文更长：

- 目标：**同等显存下上下文长度 ≥ 1.5×**，且**逐 token 输出与原生一致**（bitwise 对齐）。
- 分级：一期仅做 **VRAM ⇄ RAM 两级**（本机无 PagedAttention，不引入 SSD 级）。
- 开关：环境变量 `TDS_KV`，取值 `off` / `observe` / `page`，缺省即原生。
- 红线：不承接他人知识产权，仅经公开接口对接 ComfyUI；任一环节异常必须可无痕回退。

---

## 2. 源码侦察结论（实测）

### 2.1 KV 唯一分配入口

`comfy/text_encoders/llama.py` 中 `Llama2_.init_kv_cache(self, batch, capacity, device, dtype)`（约 800 行）
是本机 Llama 系文本编码器 KV 的**唯一分配入口**，返回 `num_hidden_layers` 个 cache 对象。

### 2.2 双返回形态（关键约束）

`init_kv_cache` 依据 `self.fixed_kv and comfy_kitchen.flash_attention_decode_is_available(device)` 分两路：

| 分支 | 返回元素 | key 形状 |
|---|---|---|
| `fixed_kv=True` | `FixedKV(key, value, 0, position, seqlen)` | `(batch, capacity, num_kv_heads, head_dim)` |
| `fixed_kv=False` | 三元组 `(key, value, 0)` | `(batch, num_kv_heads, capacity, head_dim)` |

两条路径的**张量维度顺序不同**，且下游 `get_past_len` 以
`first.index if isinstance(first, FixedKV) else first[2]` 区分形态。

**结论：分页替换只能作用于 `fixed_kv=True` 分支**；非 fixed_kv 分支为裸三元组，
下游按 tuple 解包，替换会破坏宿主语义。

### 2.3 `FixedKV` 真实接口（`comfy/text_encoders/llama.py` 约 21 行）

```python
@dataclass
class FixedKV:
    key: torch.Tensor
    value: torch.Tensor
    index: int                 # 注意：是 int，不是 tensor
    position: torch.Tensor
    seqlen: torch.Tensor

    def prepare(self, num_tokens):  # position.copy_(seqlen); seqlen.add_(num_tokens)
    def advance(self, num_tokens):  # index += num_tokens
```

**这是分页包装层必须完整实现的接口契约。**

---

## 3. container_function 钩子冲突探测（实测证据）

三个候选钩子：

| 钩子 | 落点 | 本机实测 | 结论 |
|---|---|---|---|
| A. `init_kv_cache` | `Llama2_.init_kv_cache` | 未被占用 | **可用（选定）** |
| B. `AttentionTensorContainer` + `container_function` | `attention.py` | 已被官方占用 | **不可用** |
| C. `set_model_optimized_attention` | attention 注册层 | — | 备选，暂不启用 |

探测输出（`tests/_probe_int8.txt`）：

```
IMPORT_OK
INT8_AVAILABLE=True
basic.container_function=None
int8fn=present
int8.container_function=<function _attention_comfy_kitchen_int8_containers at 0x...>
AttentionTensorContainer=True
comfy_kitchen=importable
```

即：本机 `comfy_kitchen` 可用，`attention_comfy_kitchen_int8.container_function`
已被官方 int8 attention 后端占用（`attention.py:646`）。同时
`AttentionTensorContainer` 为 **single-owner** 语义（`take()` 后即置空），
链式包装会导致容器被提前消费。

**决策：放弃钩子 B，采用钩子 A（`init_kv_cache`）。**

---

## 4. 架构

```
init_kv_cache（Llama2_）
        │  ← TDS 包装器（钩子A，链式：先调 original，再包裹返回）
        ▼
   原生返回 FixedKV 列表  ──►  PageCache 列表（每层一个）
                                   │
                        PageTable（层内页表）
                        ├── pages      : 驻留显存的页（dict）
                        ├── ram_buffer : 换出至内存的页（dict）
                        ├── get_or_create_page(offset)
                        ├── evict_to_ram(page)
                        ├── evict_lru_page()          # 按 last_used_step 最小
                        └── prefetch_to_vram(page)
                                   │
                        PageCache 对外暴露 FixedKV 兼容接口
                        .key / .value / .index / .position / .seqlen
                        .prepare() / .advance()
```

### 4.1 分页粒度

- `TDS_PAGE_SIZE_TOKENS = 128`（`config.py`）
- `TDS_PAGE_SIZE_BYTES = TDS_PAGE_SIZE_TOKENS * 2 * 64`（占位，真实字节数由
  `2 * head_dim * num_kv_heads * tokens` 计算）

### 4.2 淘汰与换入策略

- 淘汰：LRU，按 `last_used_step` 升序（并列时 `access_count` 小者先出）。
- 换入：优先复用 `ram_buffer` 中的同一 `Page` 对象，保留热度元数据。
- 距离：复用 TDS 既有"八卦加权距离"模型估算下次使用距离，用于
  `_evict_pages` / `_prefetch_pages` 的优先级排序（当前为 step 简化实现）。

---

## 5. 当前实现状态与差距

| 项 | 状态 |
|---|---|
| `page_manager.py` 页表/页缓存骨架 | ✅ 已实现 |
| 包装器（enable/disable/幂等/异常吞没） | ✅ 已实现 |
| 单测 `test_kv_page_manager.py` | ✅ 34/34 通过 |
| 全量回归 | ✅ 98/98 通过 |
| container_function 冲突探测 | ✅ 已完成（结论见 §3） |
| **PageCache 对齐 FixedKV 真实接口** | ⚠️ **未完成**：当前 `index` 返回 tensor，与 `FixedKV.index: int` 不符；缺 `position` / `seqlen` / `prepare()` / `advance()` |
| **真实张量搬运** | ⚠️ **未完成**：`evict_to_ram` / `prefetch_to_vram` 仅切换 `is_in_vram` 状态位，未真正在显存/内存间搬移 KV 数据 |
| `fixed_kv` 分支判定 | ⚠️ 未在包装器中判定，需仅在 `FixedKV` 分支替换 |
| bridge 集成（`TDS_KV=page`） | ⚠️ 未接入 |
| 真机验证（≥1.5× 上下文、逐 token 一致） | ⚠️ 未做 |

---

## 6. 下一步（按优先级）

1. **对齐接口（阻断项）**：将 `PageCache` 改造为完整实现 `FixedKV` 契约——
   `index: int`、`position`、`seqlen`、`prepare()`、`advance()`，并按需在
   `flash_attention_decode` 消费前物化连续 KV。
2. **分支守卫**：包装器中判定 `isinstance(result[0], FixedKV)`，非 fixed_kv 分支
   直接透传原生结果。
3. **真实搬运**：`evict_to_ram` 将页切片 `.to("cpu", non_blocking=True)` 存入
   `ram_buffer`；`prefetch_to_vram` 反向 `.to(device)` 并写回页槽位。
4. **bridge 集成**：`comfyui_bridge.py` 的 `attach_default` 末尾按
   `TDS_KV=page` 调用 `page_manager.enable()`，与 A 段 observer 并列。
5. **真机验证**：固定提示词与随机种子，比对 `TDS_KV=off` 与 `page` 的
   逐 token 输出一致性，并测量可承载上下文上限。

---

## 7. 验收标准

- [ ] `TDS_KV=off` 与原生行为 bitwise 一致（回归基线不变）
- [ ] `TDS_KV=page` 下同等显存上下文长度 ≥ 1.5×
- [ ] 任一异常可无痕回退（禁用开关即恢复原生）
- [ ] 全量单测 + 真机一致性验证通过
*（内容由AI生成，仅供参考）*
