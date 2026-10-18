"""模型目录：声明式元数据，可外部注入。

设计原则
--------
1. **不内嵌任何权重文件，也不硬编码校验和**：条目只描述"去哪儿取、取哪几个
   文件"，真实体积与哈希一律由远端 API 动态解析（见 ``probe_entry``）；
2. 内置条目仅为**骨架**（``verified=False``），未经 ``probe_entry`` 校验前不得
   视为可用；校验通过后回填体积并置 ``verified=True``；
3. 支持通过环境变量 ``TDS_REGISTRY_CATALOG`` 指向的 JSON 追加/覆盖条目，
   便于按项目随包分发自有目录，而不污染引擎内置默认值。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from .errors import CatalogError
from .remote import list_remote_files, resolve_endpoint

DOMAINS: Tuple[str, ...] = ("text_encoder", "diffusion", "vae", "llm", "other")
_ENV_CATALOG = "TDS_REGISTRY_CATALOG"


@dataclass
class ModelSpec:
    """一个可获取的模型条目。

    显存估算相关字段（params_b / quant_bits / layers / ...）**允许留空**：
    留空时由文件体积直接推导权重占用，不做参数规模反推，避免臆测。
    """

    key: str
    repo: str
    files: Tuple[str, ...]
    domain: str = "other"
    params_b: float = 0.0
    quant_bits: int = 0
    ctx_tokens: int = 0
    layers: int = 0
    kv_heads: int = 0
    head_dim: int = 0
    sizes: Dict[str, int] = field(default_factory=dict)
    checksums: Dict[str, str] = field(default_factory=dict)
    tags: Tuple[str, ...] = ()
    note: str = ""
    verified: bool = False

    def __post_init__(self) -> None:
        if not self.key or not str(self.key).strip():
            raise CatalogError("ModelSpec.key 不能为空")
        if not self.repo or not str(self.repo).strip():
            raise CatalogError("ModelSpec[{}].repo 不能为空".format(self.key))
        if not self.files:
            raise CatalogError("ModelSpec[{}].files 不能为空".format(self.key))
        if self.domain not in DOMAINS:
            raise CatalogError(
                "ModelSpec[{}].domain 非法：{}（可选 {}）".format(
                    self.key, self.domain, "/".join(DOMAINS)
                )
            )
        self.files = tuple(self.files)
        self.tags = tuple(self.tags)

    # ---- 体积 ----

    @property
    def total_size_bytes(self) -> int:
        """已知体积之和；存在未解析文件时只返回已知部分。"""
        return sum(int(self.sizes.get(f, 0) or 0) for f in self.files)

    @property
    def size_complete(self) -> bool:
        """是否所有文件体积均已解析。"""
        return bool(self.files) and all(
            int(self.sizes.get(f, 0) or 0) > 0 for f in self.files
        )

    @property
    def needs_kv(self) -> bool:
        """是否需要 KV Cache 估算（文本编码器 / 语言模型按上下文占用显存）。"""
        return self.domain in ("text_encoder", "llm")

    # ---- 序列化 ----

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["files"] = list(self.files)
        data["tags"] = list(self.tags)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelSpec":
        if not isinstance(data, dict):
            raise CatalogError("目录条目必须是 JSON 对象")
        payload = dict(data)
        for key in ("files", "tags"):
            if key in payload and payload[key] is not None:
                payload[key] = tuple(payload[key])
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(payload) - allowed
        if unknown:
            raise CatalogError("条目含未知字段：{}".format(", ".join(sorted(unknown))))
        return cls(**payload)

    def with_sizes(
        self,
        sizes: Dict[str, int],
        checksums: Optional[Dict[str, str]] = None,
        verified: bool = True,
    ) -> "ModelSpec":
        """回填远端体积与摘要，返回新实例（保留原实例不变）。"""
        merged = dict(self.sizes)
        merged.update({k: int(v or 0) for k, v in sizes.items()})
        sums = dict(self.checksums)
        if checksums:
            sums.update({k: str(v) for k, v in checksums.items() if v})
        return replace(self, sizes=merged, checksums=sums, verified=verified)


# --------------------------------------------------------------------------
# 内置目录（外置数据文件）
# --------------------------------------------------------------------------
# 目录数据存放在 data/builtin_catalog.json：仓库名与文件名为**公开分发源标识**，
# 属数据字段，不构成任何宿主模块引用；与引擎代码解耦后可整体替换
# （直接编辑该文件，或用环境变量 TDS_REGISTRY_CATALOG 指向外部 JSON 覆盖同名 key）。
# 内置条目一律未校验（verified=False），可用性以 probe_entry() 的远端清单为准。

_BUILTIN_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "builtin_catalog.json"
)


def _builtin() -> Dict[str, ModelSpec]:
    if not os.path.isfile(_BUILTIN_PATH):
        raise CatalogError("内置目录数据缺失：{}".format(_BUILTIN_PATH))
    return {spec.key: spec for spec in _load_external(_BUILTIN_PATH)}


_CATALOG: Optional[Dict[str, ModelSpec]] = None


def _load_external(path: str) -> List[ModelSpec]:
    """从 JSON 文件读取条目列表（内置目录与外部注入共用此通道）。"""
    if not os.path.isfile(path):
        raise CatalogError("目录文件不存在：{}".format(path))
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError("目录文件解析失败：{} ({})".format(path, exc)) from exc
    items = raw.get("models") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise CatalogError("目录格式应为 {\"models\": [...]} 或直接为数组")
    return [ModelSpec.from_dict(item) for item in items]


def load_catalog(path: Optional[str] = None, force: bool = False) -> Dict[str, ModelSpec]:
    """返回目录副本；首次调用时按 内置 → 外部覆盖 的顺序合并。"""
    global _CATALOG
    if _CATALOG is not None and not force and path is None:
        return dict(_CATALOG)

    catalog = _builtin()
    external = path or os.environ.get(_ENV_CATALOG)
    if external:
        for spec in _load_external(external):
            catalog[spec.key] = spec
    if path is None:
        _CATALOG = catalog
    return dict(catalog)


def get(key: str) -> ModelSpec:
    catalog = load_catalog()
    if key not in catalog:
        raise CatalogError(
            "未命中条目「{}」；可用：{}".format(key, ", ".join(sorted(catalog)))
        )
    return catalog[key]


def list_entries(domain: Optional[str] = None) -> List[ModelSpec]:
    items = sorted(load_catalog().values(), key=lambda s: (s.domain, s.key))
    if domain:
        items = [s for s in items if s.domain == domain]
    return items


def register(spec: ModelSpec, override: bool = False) -> ModelSpec:
    """运行时注册条目；已存在同名条目时需显式 override=True。"""
    global _CATALOG
    catalog = load_catalog()
    if spec.key in catalog and not override:
        raise CatalogError("条目「{}」已存在，如需替换请传 override=True".format(spec.key))
    catalog[spec.key] = spec
    _CATALOG = catalog
    return spec


_REMOTE_CACHE: Dict[str, Dict[str, int]] = {}


def clear_remote_cache() -> None:
    """清空进程内远端清单缓存（诊断 / 强制刷新时使用）。"""
    _REMOTE_CACHE.clear()


def probe_entry(spec: ModelSpec, endpoint: Optional[str] = None) -> ModelSpec:
    """按远端实际清单回填体积；文件缺失即抛 CatalogError。

    * 仅解析条目声明的文件，不触发任何权重下载；
    * 同一 (端点, 仓库) 在一次进程内只查询一次远端，避免批量排名时重复请求。
    """
    cache_key = "{}|{}".format(resolve_endpoint(endpoint), spec.repo)
    remote_files = _REMOTE_CACHE.get(cache_key)
    if remote_files is None:
        remote_files = list_remote_files(spec.repo, endpoint)
        _REMOTE_CACHE[cache_key] = remote_files

    missing = [f for f in spec.files if f not in remote_files]
    if missing:
        raise CatalogError(
            "仓库 {} 缺少声明的文件：{}".format(spec.repo, ", ".join(missing))
        )
    sizes = {f: int(remote_files[f].get("size", 0) or 0) for f in spec.files}
    sums = {
        f: remote_files[f]["sha256"]
        for f in spec.files
        if remote_files[f].get("sha256")
    }
    return spec.with_sizes(sizes, checksums=sums, verified=True)
