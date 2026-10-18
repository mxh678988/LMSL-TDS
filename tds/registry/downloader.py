"""权重与配套文件的下载器。

特性
----
* 国内镜像优先：端点解析见 ``remote.resolve_endpoint``（默认 hf-mirror.com）；
* **断点续传**：残留 ``<file>.part`` 存在时以 ``Range`` 续传，服务端不支持
  Range（返回 200）时自动从头重写，不产生半包文件；
* **体积校验 + 原子落盘**：先写 ``.part``，校验通过后 ``os.replace`` 改名，
  过程中任何异常都不会污染正式目标路径；
* **幂等**：目标文件已存在且体积相符、摘要一致时直接跳过，重复调用无副作用；
* **摘要校验**：条目提供 SHA-256 时于改名落盘前逐字节校验，不符即丢弃临时
  文件并报错，杜绝静默落盘的坏权重；体积相符而摘要不符的既有文件会被重下。
"""
from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Optional

from .errors import DownloadError, IntegrityError
from .remote import _USER_AGENT, _requests, file_url, resolve_endpoint

_CHUNK = 256 * 1024
_PART_SUFFIX = ".part"
_TIMEOUT_S = 60.0

ProgressFn = Callable[[str, int, int], None]


class _Stream:
    """统一的流式读取封装（requests 优先，urllib 兜底）。"""

    def __init__(self, url: str, offset: int = 0) -> None:
        self.url = url
        self.offset = offset
        self.total = 0      # 远端总字节数（0 = 未知）
        self.start = 0      # 实际起始偏移（服务端可能忽略 Range）
        self._resp = None
        self._mode = ""

    def __enter__(self) -> "_Stream":
        headers = {"User-Agent": _USER_AGENT}
        if self.offset > 0:
            headers["Range"] = "bytes={}-".format(self.offset)

        if _requests is not None:
            try:
                resp = _requests.get(
                    self.url, headers=headers, stream=True,
                    timeout=_TIMEOUT_S, allow_redirects=True,
                )
            except Exception as exc:
                raise DownloadError("连接失败：{} ({})".format(self.url, exc)) from exc
            if resp.status_code not in (200, 206):
                resp.close()
                raise DownloadError("HTTP {}：{}".format(resp.status_code, self.url))
            self._resp, self._mode = resp, "requests"
            if resp.status_code == 206 and self.offset > 0:
                self.start = self.offset
                cr = resp.headers.get("Content-Range", "")
                self.total = _parse_total(cr) or int(resp.headers.get("Content-Length") or 0)
            else:
                self.start = 0
                self.total = int(resp.headers.get("Content-Length") or 0)
            return self

        req = urllib.request.Request(self.url, headers=headers)
        try:
            resp = urllib.request.urlopen(req, timeout=_TIMEOUT_S)
        except urllib.error.HTTPError as exc:
            raise DownloadError("HTTP {}：{}".format(exc.code, self.url)) from exc
        except Exception as exc:
            raise DownloadError("连接失败：{} ({})".format(self.url, exc)) from exc
        self._resp, self._mode = resp, "urllib"
        status = getattr(resp, "status", resp.getcode())
        length = int(resp.headers.get("Content-Length") or 0)
        if status == 206 and self.offset > 0:
            self.start = self.offset
            self.total = _parse_total(resp.headers.get("Content-Range", "")) or length
        else:
            self.start = 0
            self.total = length
        return self

    def chunks(self):
        if self._mode == "requests":
            yield from self._resp.iter_content(_CHUNK)
        else:
            while True:
                data = self._resp.read(_CHUNK)
                if not data:
                    break
                yield data

    def __exit__(self, *exc_info) -> None:
        try:
            if self._resp is not None:
                if self._mode == "requests":
                    self._resp.close()
                else:
                    self._resp.close()
        except Exception:
            pass


def _parse_total(content_range: str) -> int:
    """从 ``bytes 100-999/1000`` 解析总长。"""
    if "/" not in content_range:
        return 0
    tail = content_range.rsplit("/", 1)[-1].strip()
    try:
        return int(tail)
    except ValueError:
        return 0


def verify_file(path: str, sha256: str) -> bool:
    """校验文件 SHA-256；摘要为空或文件不存在时返回 False。"""
    if not sha256 or not os.path.isfile(path):
        return False
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest().lower() == sha256.strip().lower()


def download_file(
    repo: str,
    filename: str,
    dest_dir: str,
    endpoint: Optional[str] = None,
    revision: str = "main",
    expected_size: int = 0,
    expected_sha256: str = "",
    progress: Optional[ProgressFn] = None,
    force: bool = False,
) -> str:
    """下载单个文件到 ``dest_dir/<filename>``，返回最终绝对路径。

    Parameters
    ----------
    expected_size:
        远端已知体积（字节）；>0 时用于跳过判定与落盘后校验。
    expected_sha256:
        远端已知摘要；非空时参与跳过判定，并在改名落盘前做完整性校验。
    force:
        True 时忽略已存在的同名文件，重新下载。
    """
    dest_dir = os.path.abspath(dest_dir)
    target = os.path.join(dest_dir, filename.replace("/", os.sep))
    os.makedirs(os.path.dirname(target) or dest_dir, exist_ok=True)

    restart = False
    if not force and os.path.isfile(target):
        existing = os.path.getsize(target)
        size_ok = not expected_size or existing == expected_size
        if size_ok and (not expected_sha256 or verify_file(target, expected_sha256)):
            if progress:
                progress(filename, existing, existing)
            return target
        if size_ok and expected_sha256:
            # 体积相符但摘要不符：判定为损坏文件，丢弃重下而非续传
            restart = True

    url = file_url(repo, filename, revision=revision, endpoint=endpoint)
    part = target + _PART_SUFFIX
    if (restart or force) and os.path.isfile(part):
        try:
            os.remove(part)
        except OSError:
            pass
    offset = os.path.getsize(part) if os.path.isfile(part) else 0

    with _Stream(url, offset=offset) as stream:
        if stream.start != offset:
            offset = 0  # 服务端未接受 Range，重写
        mode = "ab" if offset > 0 else "wb"
        written = offset
        with open(part, mode) as fh:
            for chunk in stream.chunks():
                if not chunk:
                    continue
                fh.write(chunk)
                written += len(chunk)
                if progress:
                    progress(filename, written, stream.total or expected_size or written)

    done = os.path.getsize(part)
    if expected_size and done != expected_size:
        raise IntegrityError(
            "{} 体积不符：期望 {} 字节，实际 {} 字节（保留 {} 以便续传）".format(
                filename, expected_size, done, part
            )
        )
    if expected_sha256 and not verify_file(part, expected_sha256):
        try:
            os.remove(part)
        except OSError:
            pass
        raise IntegrityError(
            "{} 摘要校验失败（期望 {}…，已丢弃临时文件，请重试）".format(
                filename, expected_sha256[:12]
            )
        )
    os.replace(part, target)
    return target

    with _Stream(url, offset=offset) as stream:
        if stream.start != offset:
            offset = 0  # 服务端未接受 Range，重写
        mode = "ab" if offset > 0 else "wb"
        written = offset
        with open(part, mode) as fh:
            for chunk in stream.chunks():
                if not chunk:
                    continue
                fh.write(chunk)
                written += len(chunk)
                if progress:
                    progress(filename, written, stream.total or expected_size or written)

    done = os.path.getsize(part)
    if expected_size and done != expected_size:
        raise IntegrityError(
            "{} 体积不符：期望 {} 字节，实际 {} 字节（保留 {} 以便续传）".format(
                filename, expected_size, done, part
            )
        )
    os.replace(part, target)
    return target


def download_spec(
    spec: Any,
    dest_dir: str,
    endpoint: Optional[str] = None,
    revision: str = "main",
    progress: Optional[ProgressFn] = None,
    force: bool = False,
) -> Dict[str, str]:
    """按条目下载全部声明文件 → ``{filename: 本地路径}``。

    用 ``spec.sizes`` 校验体积、``spec.checksums`` 校验摘要；两者缺失时仅做
    落盘（此时不保证内容正确性，调用方可先经 ``probe_entry`` 补齐）。
    """
    if not getattr(spec, "repo", None) or not getattr(spec, "files", None):
        raise DownloadError("传入对象缺少 repo/files，无法下载")
    sizes = getattr(spec, "sizes", None) or {}
    sums = getattr(spec, "checksums", None) or {}
    out: Dict[str, str] = {}
    for name in spec.files:
        out[name] = download_file(
            repo=spec.repo,
            filename=name,
            dest_dir=dest_dir,
            endpoint=endpoint,
            revision=revision,
            expected_size=int(sizes.get(name, 0) or 0),
            expected_sha256=str(sums.get(name, "") or ""),
            progress=progress,
            force=force,
        )
    return out


def endpoint_info(endpoint: Optional[str] = None) -> Dict[str, Any]:
    return {"endpoint": resolve_endpoint(endpoint)}
