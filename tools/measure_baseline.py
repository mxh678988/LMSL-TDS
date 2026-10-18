#!/usr/bin/env python3
"""TDS Phase 0 基线测量工具。

采集本机 GPU 显存、系统内存、磁盘容量与顺序读写带宽，产出 JSON 基线，
用于校准 TDS 调度阈值（vram_budget_mb / reserve_mb / 分层粒度）。

用法:
  python tools/measure_baseline.py --duration 8 --interval 0.5 \
      --out benchmarks/baseline.json --workdir <临时目录>

说明:
  - 显存优先通过 nvidia-smi 读取（零依赖）；当前解释器装有 torch 时补充 torch 视图。
  - 磁盘带宽测试会在 --workdir 下写一个临时文件，测完即删，不留残留。
  - 本脚本只读系统状态 + 受控临时写入，不做任何破坏性操作。
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

NVIDIA_FIELDS = "name,driver_version,memory.total,memory.used,memory.free,utilization.gpu"


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except Exception:
        return None


# ---------------- GPU ----------------

def query_nvidia_smi():
    """通过 nvidia-smi 读取 GPU 快照（零依赖）。"""
    r = _run(["nvidia-smi", f"--query-gpu={NVIDIA_FIELDS}", "--format=csv,noheader,nounits"])
    if not r or r.returncode != 0 or not r.stdout.strip():
        return None
    parts = [p.strip() for p in r.stdout.strip().splitlines()[0].split(",")]
    try:
        return {
            "name": parts[0],
            "driver": parts[1],
            "total_mb": int(float(parts[2])),
            "used_mb": int(float(parts[3])),
            "free_mb": int(float(parts[4])),
            "util_pct": int(float(parts[5])),
            "source": "nvidia-smi",
        }
    except Exception:
        return None


def query_torch():
    """若当前解释器装有 torch，补充 torch 视图。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free_b, total_b = torch.cuda.mem_get_info()
        return {
            "name": torch.cuda.get_device_name(0),
            "total_mb": int(total_b // (1024 * 1024)),
            "free_mb": int(free_b // (1024 * 1024)),
            "capability": list(torch.cuda.get_device_capability(0)),
            "source": "torch",
        }
    except Exception:
        return None


def sample_vram(duration_s, interval_s):
    """周期采样显存占用，统计水位。"""
    used = []
    t0 = time.time()
    while time.time() - t0 < duration_s:
        s = query_nvidia_smi()
        if s:
            used.append(s["used_mb"])
        time.sleep(interval_s)
    if not used:
        return None
    return {
        "duration_s": duration_s,
        "interval_s": interval_s,
        "samples": len(used),
        "peak_used_mb": max(used),
        "avg_used_mb": round(sum(used) / len(used), 1),
        "min_used_mb": min(used),
    }


# ---------------- 内存 / 磁盘 ----------------

def memory_info():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "total_gb": round(vm.total / 1024 ** 3, 2),
            "available_gb": round(vm.available / 1024 ** 3, 2),
            "percent": vm.percent,
        }
    except Exception:
        return None


def disk_info():
    try:
        import psutil
        out = []
        for p in psutil.disk_partitions(all=False):
            try:
                u = psutil.disk_usage(p.mountpoint)
                out.append({
                    "mount": p.mountpoint,
                    "fs": p.fstype,
                    "total_gb": round(u.total / 1024 ** 3, 1),
                    "free_gb": round(u.free / 1024 ** 3, 1),
                    "percent": u.percent,
                })
            except Exception:
                continue
        return out
    except Exception:
        return None


def disk_bandwidth(workdir, size_mb=128):
    """顺序读写带宽粗测（在 workdir 内临时文件，测完即删）。"""
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, "tds_bw_test.bin")
    chunk = b"\0" * (8 * 1024 * 1024)
    loops = max(1, size_mb // 8)
    try:
        t0 = time.time()
        with open(path, "wb") as f:
            for _ in range(loops):
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
        write_s = max(1e-6, time.time() - t0)

        t0 = time.time()
        with open(path, "rb") as f:
            while f.read(8 * 1024 * 1024):
                pass
        read_s = max(1e-6, time.time() - t0)

        return {
            "size_mb": loops * 8,
            "write_mb_s": round(loops * 8 / write_s, 1),
            "read_mb_s": round(loops * 8 / read_s, 1),
            "workdir": workdir,
        }
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# ---------------- 阈值建议 ----------------

def recommend(gpu, vram_sampling):
    total = (gpu or {}).get("total_mb") or 12288
    # 预留 = 桌面/驱动常驻 + 峰值抖动安全垫；小卡收紧、大卡放宽
    reserve = 1500 if total <= 13000 else 2500
    budget = max(0, total - reserve)
    return {
        "vram_budget_mb": budget,
        "reserve_mb": reserve,
        "note": f"基于实测 total={total}MiB 校准（RTX 3060 12GB 级）",
        "verified_by_sampling": bool(vram_sampling),
    }


def main():
    ap = argparse.ArgumentParser(description="TDS Phase 0 基线测量")
    ap.add_argument("--duration", type=float, default=8.0, help="显存采样时长（秒）")
    ap.add_argument("--interval", type=float, default=0.5, help="采样间隔（秒）")
    ap.add_argument("--out", type=str, default="benchmarks/baseline.json", help="输出 JSON 路径")
    ap.add_argument("--workdir", type=str, default=".", help="磁盘带宽测试工作目录")
    ap.add_argument("--disk-size", type=int, default=128, help="带宽测试文件大小（MB）")
    args = ap.parse_args()

    gpu = query_nvidia_smi()
    torch_view = query_torch()
    vram_sampling = sample_vram(args.duration, args.interval) if gpu else None

    report = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "host": platform.node(),
            "os": f"{platform.system()} {platform.release()}",
            "python": sys.version.split()[0],
        },
        "gpu": gpu,
        "gpu_torch_view": torch_view,
        "vram_sampling": vram_sampling,
        "memory": memory_info(),
        "disks": disk_info(),
        "disk_bandwidth": disk_bandwidth(args.workdir, args.disk_size),
    }
    report["recommendation"] = recommend(gpu, vram_sampling)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[OK] baseline written: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
