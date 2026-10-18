"""Registry 层统一异常定义。

本模块只依赖标准库，保证 registry 子包在无外部运行时依赖的环境下也可导入。
"""


class RegistryError(Exception):
    """Registry 层异常基类。"""


class CatalogError(RegistryError):
    """目录条目非法、未命中或重复注册。"""


class HardwareProbeError(RegistryError):
    """硬件画像探测失败（全部探测通道不可用）。"""


class DownloadError(RegistryError):
    """下载失败：网络错误、HTTP 非 2xx、体积校验不符、写入中断。"""


class IntegrityError(DownloadError):
    """产物完整性校验失败（体积 / 校验和不匹配）。"""
