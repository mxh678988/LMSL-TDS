"""硬件画像探测：显存 / 内存 / 算力档位。

设计约束
--------
1. 只依赖标准库，**不引入任何深度学习框架运行时**，保证 registry 可在纯净
   解释器中运行；
2. 主通道为 ``nvidia-smi``（Windows / Linux 通用，随驱动分发，无需额外依赖）；
3. 全部通道失败时不抛错，回退为"未知画像"，由上层按保守策略处理。

单位统一使用 MB（1024 进制），与 TDSConfig.vram_budget_mb 对齐。
"""
from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

_SMI_QUERY = "index,name,memory.total,memory.free,compute_cap"
_SMI_TIMEOUT_S = 8.0


@dataclass
class DeviceProfile:
    """单块加速设备画像。"""

    index: int
    name: str
    total_mb: int
    free_mb: int = 0
    compute_cap: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HardwareProfile:
    """整机画像。

    source 记录显存数据来源（``nvidia-smi`` / ``unknown``），便于上层判断
    数据可信度；devices 为空表示未探测到加速设备。
    """

    os_name: str = ""
    python_version: str = ""
    cpu_count: int = 0
    ram_mb: int = 0
    devices: List[DeviceProfile] = field(default_factory=list)
    source: str = "unknown"

    # ---- 便捷视图 ----

    @property
    def has_gpu(self) -> bool:
        return bool(self.devices)

    @property
    def primary_device(self) -> Optional[DeviceProfile]:
        """默认主设备 = 显存最大的一块（多卡异构时取上限做可行性判断）。"""
        if not self.devices:
            return None
        return max(self.devices, key=lambda d: d.total_mb)

    @property
    def total_vram_mb(self) -> int:
        dev = self.primary_device
        return dev.total_mb if dev else 0

    @property
    def free_vram_mb(self) -> int:
        dev = self.primary_device
        return dev.free_mb if dev else 0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["has_gpu"] = self.has_gpu
        return data

    def summary(self) -> str:
        """单行人类可读摘要。"""
        if not self.devices:
            return "无加速设备（RAM {} MB）".format(self.ram_mb or "?")
        parts = []
        for d in self.devices:
            parts.append(
                "GPU{} {} {}MB(空闲 {}MB) cc{}".format(
                    d.index, d.name, d.total_mb, d.free_mb, d.compute_cap or "?"
                )
            )
        return " | ".join(parts)


# --------------------------------------------------------------------------
# 探测通道
# --------------------------------------------------------------------------


def _probe_gpu_smi() -> List[DeviceProfile]:
    """通过 nvidia-smi 读取 GPU 画像；不可用时返回空列表。"""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        proc = subprocess.run(
            [
                exe,
                "--query-gpu=" + _SMI_QUERY,
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=_SMI_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0 or not proc.stdout.strip():
        return []

    devices: List[DeviceProfile] = []
    for line in proc.stdout.strip().splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 4:
            continue
        try:
            index = int(cols[0])
            total_mb = int(float(cols[2]))
            free_mb = int(float(cols[3]))
        except ValueError:
            continue
        devices.append(
            DeviceProfile(
                index=index,
                name=cols[1],
                total_mb=total_mb,
                free_mb=free_mb,
                compute_cap=cols[4] if len(cols) > 4 else "",
            )
        )
    return devices


def _probe_ram_mb() -> int:
    """读取物理内存总量（MB）；失败返回 0。"""
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().total / (1024 * 1024))
    except Exception:
        pass

    if platform.system() == "Windows":
        try:

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemStatus()
            stat.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullTotalPhys / (1024 * 1024))
        except Exception:
            pass

    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return int(kb / 1024)
        except Exception:
            pass
    return 0


# --------------------------------------------------------------------------
# 对外入口
# --------------------------------------------------------------------------

_CACHE: Optional[HardwareProfile] = None


def probe_hardware(refresh: bool = False) -> HardwareProfile:
    """探测并返回整机画像。

    Parameters
    ----------
    refresh:
        True 时绕过进程内缓存重新探测（显存空闲量是动态值，
        做多轮加载决策时应传 True）。
    """
    global _CACHE
    if _CACHE is not None and not refresh:
        return _CACHE

    devices = _probe_gpu_smi()
    profile = HardwareProfile(
        os_name="{} {}".format(platform.system(), platform.release()),
        python_version=platform.python_version(),
        cpu_count=os.cpu_count() or 0,
        ram_mb=_probe_ram_mb(),
        devices=devices,
        source="nvidia-smi" if devices else "unknown",
    )
    _CACHE = profile
    return profile
