"""确定性测试图生成（零第三方依赖）。

用途
----
为多模态用例提供**可复现**的输入图：不引用外部素材、不依赖 PIL、不写随机数，
同一场景每次生成的 PNG 字节完全一致，从而保证回归结果可比、失败可复现。

图像编码仅使用标准库 ``zlib``（压缩）与 ``struct``（块头/尾），生成 8 位真彩
PNG（无 gAMA/无 ICC，扫描行过滤器恒为 0）。

场景一览
--------
- ``bars_rgb``      ：左中右三色竖条（红 / 绿 / 蓝），测颜色识别与顺序
- ``count_circles`` ：水平等距三个黑圆，测计数
- ``circle_topleft``：仅左上象限一个蓝圆，测空间位置（历史已知偏差用例）
- ``checker``       ：4×4 黑白棋盘，测纹理/结构描述
"""
from __future__ import annotations

import struct
import zlib
from typing import Callable, Dict, List, Sequence, Tuple

RGB = Tuple[int, int, int]

WHITE: RGB = (255, 255, 255)
BLACK: RGB = (0, 0, 0)
RED: RGB = (220, 30, 30)
GREEN: RGB = (30, 170, 60)
BLUE: RGB = (30, 80, 220)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------
# PNG 编码
# --------------------------------------------------------------------------


def _chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def encode_png(width: int, height: int, rows: Sequence[Sequence[int]]) -> bytes:
    """把 RGB 扫描行编码为 PNG 字节。

    ``rows`` 为 ``height`` 行，每行长度必须是 ``width * 3``（R,G,B 依次排列）。
    """
    if len(rows) != height:
        raise ValueError("扫描行数 {} 与高度 {} 不符".format(len(rows), height))
    payload = bytearray()
    for row in rows:
        if len(row) != width * 3:
            raise ValueError("扫描行长度 {} 与宽度 {} 不符".format(len(row), width))
        payload.append(0)  # 过滤器类型 0（None）
        payload.extend(bytes(row))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8bit truecolor
    return (
        PNG_SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(payload), 9))
        + _chunk(b"IEND", b"")
    )


def read_png_size(raw: bytes) -> Tuple[int, int]:
    """从 PNG 字节读回 ``(宽, 高)``，供自检与单测校验。"""
    if not raw.startswith(PNG_SIGNATURE):
        raise ValueError("不是 PNG 数据")
    if raw[12:16] != b"IHDR":
        raise ValueError("缺少 IHDR 块")
    width, height = struct.unpack(">II", raw[16:24])
    return int(width), int(height)


# --------------------------------------------------------------------------
# 画布
# --------------------------------------------------------------------------


class Canvas:
    """极简 RGB 画布（行优先，纯 Python 列表）。"""

    def __init__(self, width: int, height: int, color: RGB = WHITE) -> None:
        self.width = width
        self.height = height
        self.rows: List[bytearray] = [bytearray(color * width) for _ in range(height)]

    def set_pixel(self, x: int, y: int, color: RGB) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = x * 3
            self.rows[y][offset : offset + 3] = bytes(color)

    def fill_rect(self, x0: int, y0: int, x1: int, y1: int, color: RGB) -> None:
        for y in range(max(0, y0), min(self.height, y1)):
            offset = max(0, x0) * 3
            end = min(self.width, x1) * 3
            if end > offset:
                self.rows[y][offset:end] = bytes(color) * ((end - offset) // 3)

    def fill_circle(self, cx: int, cy: int, radius: int, color: RGB) -> None:
        r2 = radius * radius
        for y in range(cy - radius, cy + radius + 1):
            dy = y - cy
            span = r2 - dy * dy
            if span < 0:
                continue
            half = int(span ** 0.5)
            self.fill_rect(cx - half, y, cx + half + 1, y + 1, color)

    def to_png(self) -> bytes:
        return encode_png(self.width, self.height, self.rows)


# --------------------------------------------------------------------------
# 场景
# --------------------------------------------------------------------------


def scene_bars() -> bytes:
    """240×120：横向三等分，左红、中绿、右蓝。"""
    canvas = Canvas(240, 120, WHITE)
    canvas.fill_rect(0, 0, 80, 120, RED)
    canvas.fill_rect(80, 0, 160, 120, GREEN)
    canvas.fill_rect(160, 0, 240, 120, BLUE)
    return canvas.to_png()


def scene_count_circles() -> bytes:
    """240×120：白底上三个等距黑圆（水平居中对齐）。"""
    canvas = Canvas(240, 120, WHITE)
    for cx in (50, 120, 190):
        canvas.fill_circle(cx, 60, 22, BLACK)
    return canvas.to_png()


def scene_circle_topleft() -> bytes:
    """240×240：白底，仅左上象限一个蓝圆。"""
    canvas = Canvas(240, 240, WHITE)
    canvas.fill_circle(60, 60, 45, BLUE)
    return canvas.to_png()


def scene_checker() -> bytes:
    """240×240：4×4 黑白棋盘，每格 60 像素。"""
    canvas = Canvas(240, 240, WHITE)
    for row in range(4):
        for col in range(4):
            if (row + col) % 2 == 0:
                canvas.fill_rect(col * 60, row * 60, (col + 1) * 60, (row + 1) * 60, BLACK)
    return canvas.to_png()


_SCENES: Dict[str, Tuple[str, Callable[[], bytes]]] = {
    "bars_rgb": ("三色竖条（红绿蓝）", scene_bars),
    "count_circles": ("三个黑圆", scene_count_circles),
    "circle_topleft": ("左上单圆", scene_circle_topleft),
    "checker": ("黑白棋盘", scene_checker),
}

_cache: Dict[str, bytes] = {}


def names() -> List[str]:
    """已注册的场景名列表（字典序）。"""
    return sorted(_SCENES)


def describe(name: str) -> str:
    """场景的中文说明；未知场景抛 ``KeyError``。"""
    return _SCENES[name][0]


def image_bytes(name: str) -> bytes:
    """取场景 PNG 字节（进程内缓存，保证同进程多次取用完全一致）。"""
    if name not in _SCENES:
        raise KeyError("未知测试图：{}（可选：{}）".format(name, "、".join(names())))
    if name not in _cache:
        _cache[name] = _SCENES[name][1]()
    return _cache[name]


if __name__ == "__main__":
    for scene in names():
        raw = image_bytes(scene)
        width, height = read_png_size(raw)
        print("{:<16} {:>4}×{:<4} {:>7} 字节  {}".format(scene, width, height, len(raw), describe(scene)))
