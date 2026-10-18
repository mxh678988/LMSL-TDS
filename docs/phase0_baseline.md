---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 350ab7144af9618434cdce7bfa6d84e8_f2bb5b5ac0bb11f1884b525400cd780f
    ReservedCode1: 2cW5ufJZycd55eg9esPCe93oN6VRqkJrhXqbrrwio04aD7oDubGPp1pwGeM9JFnTEw4xxv36eXNAnTP7QgVF4gmVBIZB39kNjezeFo2WI36U/n6SiOC7Swtg9FjAPixYYn8Ea9U+G/WFIY6ynGAfC9v8PFEloxWLmEFKYZj2uJdna/5zU6fUn+I/hlw=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 350ab7144af9618434cdce7bfa6d84e8_f2bb5b5ac0bb11f1884b525400cd780f
    ReservedCode2: 2cW5ufJZycd55eg9esPCe93oN6VRqkJrhXqbrrwio04aD7oDubGPp1pwGeM9JFnTEw4xxv36eXNAnTP7QgVF4gmVBIZB39kNjezeFo2WI36U/n6SiOC7Swtg9FjAPixYYn8Ea9U+G/WFIY6ynGAfC9v8PFEloxWLmEFKYZj2uJdna/5zU6fUn+I/hlw=
---

# TDS Phase 0 · 基线报告

> 采集时间：2026-10-05 20:40 · 主机「闲云工作站」· 工具：`tools/measure_baseline.py`
> 原始数据：`benchmarks/baseline.json`

## 一、采集环境

| 项 | 值 |
| --- | --- |
| 操作系统 | Windows 10（Build 级） |
| Python | 3.11.8 |
| GPU | NVIDIA GeForce RTX 3060 |
| 驱动 | 616.92 |
| 显存总量 | 12288 MiB |

## 二、实测数据

### 2.1 显存（8s 采样 14 次）

| 指标 | 值 |
| --- | --- |
| 空载占用（均值） | 861.6 MiB |
| 峰值 | 863 MiB |
| 谷值 | 860 MiB |
| 空闲显存 | 11254 MiB |
| GPU 利用率 | ~26%（外部进程） |

结论：空载底噪 ~860 MiB，8 秒内波动 < 3 MiB，**底噪稳定**。

### 2.2 系统内存

| 指标 | 值 |
| --- | --- |
| 总量 | 63.89 GB |
| 可用 | 46.2 GB |

### 2.3 磁盘

| 盘符 | 文件系统 | 总量 | 剩余 | 占用率 |
| --- | --- | --- | --- | --- |
| C: | NTFS | 299.1 GB | 47.7 GB | 84.0% |
| D: | NTFS | 653.9 GB | 380.9 GB | 41.8% |
| E: | NTFS | 931.5 GB | 519.4 GB | 44.2% |

### 2.4 顺序读写带宽（C 盘 NTFS，128 MB）

| 方向 | 带宽 |
| --- | --- |
| 写 | 344.5 MB/s |
| 读 | 1116.3 MB/s |

## 三、分析与结论

1. **可用显存 ≈ 11.4 GB（空载）**。扣除底噪后，调度可用额度定为 10788 MB（= 12288 − 1500），reserve 1500 MB。
2. **reserve 余量充分**：底噪 860 MB，预留 1500 MB，剩 ~640 MB 抗抖动，满足"压力保护阈值 0.92"的设计。
3. **分层卸载可行**：SSD 顺序读 1.1 GB/s，一个 512 MB 的层/块从磁盘回提约 0.46 s。需**异步预取**掩盖该延迟，否则会反噬 TTFT。
4. **落盘区选择**：C 盘仅剩 47.7 GB（84% 占用），不适合作为卸载落盘区；分层 SSD 层应落 **D 盘**（剩 380.9 GB）。已写入 `config.ssd_tier_root`。
5. **读写不对称**：写 344 MB/s 仅为读的 ~31%，下沉（demote）路径需批量 + 异步，避免拖慢主推理。

## 四、阈值校准（已写回 `tds/config.py`）

| 参数 | 原值 | 校准值 | 依据 |
| --- | --- | --- | --- |
| `vram_budget_mb` | 11000 | **10788** | 12288 − 1500 |
| `reserve_mb` | 1500 | 1500（保留） | 底噪 860 + 抖动垫 |
| `ssd_tier_root` | — | **D:\** | C 盘剩余紧张 |
| `pressure_ratio` | 0.92 | 0.92（保留） | 压力保护阈值 |

## 五、遗留与下一步

- **待补**：当前解释器无 torch，`gpu_torch_view` 为空。需在 ComfyUI 虚拟环境内复跑，补全 torch 视图与真实工作流下的 TTFT / 显存峰值。
- **Phase 0 验收**：基线数据已就绪（显存水位、带宽、落盘区）。
- **Phase 1**：进入 ComfyUI custom_node MVP 编码——执行图预扫描 + 分类优先级 + 分层驻留。
*（内容由AI生成，仅供参考）*
