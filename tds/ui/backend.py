"""模型后端访问层：零依赖 HTTP 客户端 + SSE 流式解析。

设计约束
--------
1. 只依赖标准库 ``urllib`` / ``json``，与 tds.core、tds.kv 一致的零第三方运行时；
2. 面向 OpenAI 兼容协议（``/v1/chat/completions``），并额外识别 Ollama 的
   原生端点（``/api/tags`` / ``/api/version``）以获取更完整的模型画像；
3. 本模块**只做协议对接**，不承载任何 TDS 调度逻辑：测试台通过它观测
   "模型能力本身"，TDS 调度层的行为另行观测。

单位说明：模型体积统一使用字节（bytes），时延统一使用毫秒（ms）。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Optional

PROBE_TIMEOUT_S = 6.0
CHAT_TIMEOUT_S = 900.0

# 判定"是否具备多模态（视觉）能力"的名称线索：命中任一即视为支持图片输入。
_MULTIMODAL_HINTS = (
    "vl",
    "vision",
    "llava",
    "bakllava",
    "moondream",
    "minicpm-v",
    "internvl",
    "pixtral",
    "gemma3",
    "granite3.2-vision",
    "phi-4-multimodal",
    "llama3.2-vision",
    "qwen2.5vl",
    "qwen2.5-vl",
)


class BackendError(RuntimeError):
    """后端不可达、协议异常或返回错误时抛出。"""


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------


@dataclass
class ModelInfo:
    """单个可用模型的画像。"""

    name: str
    size_bytes: int = 0
    family: str = ""
    parameter_size: str = ""
    quantization: str = ""
    multimodal: bool = False
    source: str = "openai"  # openai | ollama
    remote: bool = False  # Ollama cloud 类模型（本地无权重）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def label(self) -> str:
        """下拉框展示用单行标签。"""
        marks = []
        if self.multimodal:
            marks.append("多模态")
        if self.remote:
            marks.append("云端")
        if self.parameter_size:
            marks.append(self.parameter_size)
        if self.quantization:
            marks.append(self.quantization)
        suffix = " · ".join(marks)
        return "{}  [{}]".format(self.name, suffix) if suffix else self.name


@dataclass
class ChatRequest:
    """一次对话请求（已含完整历史）。"""

    model: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    system: str = ""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 0
    stream: bool = True

    def to_payload(self) -> Dict[str, Any]:
        msgs: List[Dict[str, Any]] = []
        if self.system.strip():
            msgs.append({"role": "system", "content": self.system.strip()})
        msgs.extend(self.messages)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.stream,
        }
        if self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens
        return payload


@dataclass
class Chunk:
    """流式增量。"""

    text: str = ""
    reasoning: str = ""
    done: bool = False
    error: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# 客户端
# --------------------------------------------------------------------------


def _is_multimodal(name: str, family: str = "") -> bool:
    probe = "{} {}".format(name or "", family or "").lower()
    return any(hint in probe for hint in _MULTIMODAL_HINTS)


class BackendClient:
    """OpenAI 兼容推理服务的轻量客户端。

    Parameters
    ----------
    base_url:
        服务根地址，如 ``http://127.0.0.1:11434``；末尾 ``/v1`` 可省略。
    api_key:
        可选鉴权口令（本地服务一般为空）。
    """

    def __init__(self, base_url: str, api_key: str = "", timeout: float = CHAT_TIMEOUT_S):
        base = (base_url or "").strip().rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        self.base_url = base
        self.api_key = api_key or ""
        self.timeout = timeout
        self.last_error = ""

    # ---- 内部工具 ----

    def _url(self, path: str) -> str:
        return "{}{}".format(self.base_url, path)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer {}".format(self.api_key)
        return headers

    def _get_json(self, path: str, timeout: Optional[float] = None) -> Any:
        request = urllib.request.Request(self._url(path), headers=self._headers())
        try:
            with urllib.request.urlopen(request, timeout=timeout or PROBE_TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            raise BackendError("HTTP {} @ {}".format(exc.code, path))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise BackendError("{} @ {}".format(exc, path))

    # ---- 探测 ----

    def ping(self) -> Dict[str, Any]:
        """后端可用性探测，返回 ``{online, kind, version, error}``。"""
        result: Dict[str, Any] = {
            "online": False,
            "kind": "unknown",
            "version": "",
            "base_url": self.base_url,
            "error": "",
        }
        try:
            info = self._get_json("/api/version")
            result["online"] = True
            result["kind"] = "ollama"
            result["version"] = str(info.get("version", ""))
            return result
        except BackendError as exc:
            first_error = str(exc)
        try:
            self._get_json("/v1/models")
            result["online"] = True
            result["kind"] = "openai"
            return result
        except BackendError as exc:
            result["error"] = "{} | {}".format(first_error, exc)
            self.last_error = result["error"]
            return result

    def list_models(self) -> List[ModelInfo]:
        """优先读取 Ollama 原生清单（画像更全），失败回退 OpenAI ``/v1/models``。"""
        try:
            data = self._get_json("/api/tags")
            rows: List[ModelInfo] = []
            for item in data.get("models", []) or []:
                details = item.get("details", {}) or {}
                name = str(item.get("name") or item.get("model") or "")
                family = str(details.get("family") or "")
                size = int(item.get("size") or 0)
                rows.append(
                    ModelInfo(
                        name=name,
                        size_bytes=size,
                        family=family,
                        parameter_size=str(details.get("parameter_size") or ""),
                        quantization=str(details.get("quantization_level") or ""),
                        multimodal=_is_multimodal(name, family),
                        source="ollama",
                        remote=size == 0,
                    )
                )
            if rows:
                return sorted(rows, key=lambda m: (not m.multimodal, m.name))
        except BackendError:
            pass

        data = self._get_json("/v1/models")
        rows = []
        for item in data.get("data", []) or []:
            name = str(item.get("id") or "")
            rows.append(
                ModelInfo(name=name, multimodal=_is_multimodal(name), source="openai")
            )
        return sorted(rows, key=lambda m: (not m.multimodal, m.name))

    # ---- 对话 ----

    def chat_stream(self, req: ChatRequest) -> Iterator[Chunk]:
        """以流式方式发起对话，逐块产出增量。"""
        req.stream = True
        data = json.dumps(req.to_payload(), ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self._url("/v1/chat/completions"),
            data=data,
            headers=self._headers(),
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "replace")[:400]
            except Exception:
                pass
            raise BackendError("HTTP {} · {}".format(exc.code, body or "无响应体"))
        except (urllib.error.URLError, OSError) as exc:
            raise BackendError("无法连接 {}（{}）".format(self.base_url, exc))

        with resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                body = line[len("data:") :].strip()
                if not body:
                    continue
                if body == "[DONE]":
                    yield Chunk(done=True)
                    return
                try:
                    obj = json.loads(body)
                except ValueError:
                    continue
                if isinstance(obj, dict) and obj.get("error"):
                    yield Chunk(error=str(obj["error"]))
                    return
                choice = (obj.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                usage = obj.get("usage") or {}
                finish = choice.get("finish_reason")
                if text or reasoning or usage:
                    yield Chunk(
                        text=text,
                        reasoning=reasoning,
                        done=False,
                        usage=usage if isinstance(usage, dict) else {},
                    )
                if finish:
                    yield Chunk(done=True)
                    return
        yield Chunk(done=True)


# --------------------------------------------------------------------------
# 独立自检（python -m tds.ui.backend <base_url>）
# --------------------------------------------------------------------------


def _selfcheck(base_url: str) -> int:
    client = BackendClient(base_url)
    info = client.ping()
    print("后端     : {}".format(client.base_url))
    print("状态     : {}（{} {}）".format(
        "在线" if info["online"] else "离线", info["kind"], info["version"]
    ))
    if info["error"]:
        print("错误     : {}".format(info["error"]))
        return 1
    models = client.list_models()
    print("模型数   : {}".format(len(models)))
    for model in models:
        print("  - {}".format(model.label()))
    if models:
        target = next((m for m in models if m.multimodal), models[-1])
        print("冒烟对话 : {} …".format(target.name))
        started = time.time()
        buffer = []
        for chunk in client.chat_stream(
            ChatRequest(model=target.name, messages=[{"role": "user", "content": "用一句话说明你是谁"}])
        ):
            if chunk.error:
                print("  [错误] {}".format(chunk.error))
                return 2
            buffer.append(chunk.text)
            if chunk.done:
                break
        print("  耗时 {:.0f} ms · 输出 {} 字".format((time.time() - started) * 1000, len("".join(buffer))))
        print("  回复: {}".format("".join(buffer)[:120]))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_selfcheck(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:11434"))
