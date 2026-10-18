"""远端仓库的低层访问：端点解析与元数据查询。

约定
----
* 端点优先取环境变量 ``TDS_HF_ENDPOINT``，其次函数参数，最后内置默认
  ``https://hf-mirror.com``（国内镜像，直连官方源不稳定时的首选）；
* 只使用标准库 ``urllib``；若环境存在 ``requests`` 则自动启用（连接复用更好）；
* 本模块不做任何写盘，下载写盘由 ``downloader`` 负责。
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .errors import DownloadError

DEFAULT_ENDPOINT = "https://hf-mirror.com"
OFFICIAL_ENDPOINT = "https://huggingface.co"
_ENV_KEY = "TDS_HF_ENDPOINT"
_USER_AGENT = "tds-registry/0.1"
_TIMEOUT_S = 30.0

try:  # 可选加速，不构成硬依赖
    import requests as _requests  # type: ignore
except Exception:  # pragma: no cover - 环境相关
    _requests = None


def resolve_endpoint(endpoint: Optional[str] = None) -> str:
    """按 环境变量 > 显式参数 > 内置默认 的优先级解析端点。"""
    for candidate in (os.environ.get(_ENV_KEY), endpoint, DEFAULT_ENDPOINT):
        if candidate and str(candidate).strip():
            return str(candidate).strip().rstrip("/")
    return DEFAULT_ENDPOINT


def model_api_url(
    repo: str, endpoint: Optional[str] = None, with_blobs: bool = True
) -> str:
    """仓库元数据接口地址。

    ``with_blobs=true`` 时响应才携带每个文件的 ``size`` 与 ``lfs.sha256``
    （默认接口只给文件名，体积需另一次请求才能拿到）。
    """
    url = "{}/api/models/{}".format(resolve_endpoint(endpoint), repo.strip("/"))
    return url + "?blobs=true" if with_blobs else url


def file_url(
    repo: str,
    filename: str,
    revision: str = "main",
    endpoint: Optional[str] = None,
) -> str:
    return "{}/{}/resolve/{}/{}".format(
        resolve_endpoint(endpoint), repo.strip("/"), revision, filename
    )


def http_get_json(url: str, timeout: float = _TIMEOUT_S) -> Dict[str, Any]:
    """GET 并解析 JSON；失败统一抛 DownloadError。"""
    raw = http_get_bytes(url, timeout=timeout)
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DownloadError("响应非合法 JSON：{} ({})".format(url, exc)) from exc


def http_get_bytes(url: str, timeout: float = _TIMEOUT_S) -> bytes:
    if _requests is not None:
        try:
            resp = _requests.get(
                url, timeout=timeout, headers={"User-Agent": _USER_AGENT}
            )
            if resp.status_code != 200:
                raise DownloadError(
                    "HTTP {}：{}".format(resp.status_code, url)
                )
            return resp.content
        except DownloadError:
            raise
        except Exception as exc:  # 网络异常
            raise DownloadError("请求失败：{} ({})".format(url, exc)) from exc

    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise DownloadError("HTTP {}：{}".format(exc.code, url)) from exc
    except Exception as exc:
        raise DownloadError("请求失败：{} ({})".format(url, exc)) from exc


def _parse_siblings(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """把元数据响应归一化为 ``{路径: {size, sha256, blob_id}}``（纯函数，便于测试）。"""
    out: Dict[str, Dict[str, Any]] = {}
    for item in payload.get("siblings") or []:
        name = item.get("rfilename")
        if not name:
            continue
        lfs = item.get("lfs") or {}
        try:
            size = int(item.get("size") or lfs.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        out[name] = {
            "size": size,
            "sha256": str(lfs.get("sha256") or ""),
            "blob_id": str(item.get("blobId") or ""),
        }
    return out


def list_remote_files(
    repo: str, endpoint: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """查询仓库文件清单 → ``{相对路径: {"size", "sha256", "blob_id"}}``。

    使用 ``?blobs=true`` 元数据接口以取得体积与 LFS 摘要；远端 404 / 不可达
    时抛 DownloadError，由调用方决定是否降级。
    """
    payload = http_get_json(model_api_url(repo, endpoint))
    files = _parse_siblings(payload)
    if not files:
        raise DownloadError("仓库 {} 未返回任何文件".format(repo))
    return files


def probe_endpoint(endpoint: Optional[str] = None) -> Dict[str, Any]:
    """端点连通性自检，供 CLI / 诊断使用。"""
    url = resolve_endpoint(endpoint)
    try:
        raw = http_get_bytes(url, timeout=10.0)
        return {"endpoint": url, "reachable": True, "bytes": len(raw)}
    except DownloadError as exc:
        return {"endpoint": url, "reachable": False, "error": str(exc)}
