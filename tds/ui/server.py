"""测试台 HTTP 服务：静态面板 + JSON API + SSE 流式代理。

设计约束
--------
1. 基于标准库 ``http.server``，**不引入 Web 框架**；前端为随包分发的单文件
   HTML，无 CDN、无构建步骤，双击浏览器即可用；
2. 服务端只做三件事：托管面板、维护会话、把对话请求代理到模型后端并
   以 SSE 逐块回传；不承载 TDS 调度决策；
3. 默认仅监听 ``127.0.0.1``，不对外暴露。
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

from .backend import BackendClient, BackendError, ChatRequest, ModelInfo
from .session import ALLOWED_IMAGE_MIME, Attachment, SessionStore, make_attachment

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
DEFAULT_BACKEND = "http://127.0.0.1:11434"


def _hardware_snapshot() -> Dict[str, Any]:
    """读取整机画像用于状态条；registry 不可用时降级为空画像。"""
    try:
        from ..registry.hardware import probe_hardware

        profile = probe_hardware(refresh=True)
        return {
            "ok": True,
            "summary": profile.summary(),
            "cpu_count": profile.cpu_count,
            "ram_mb": profile.ram_mb,
            "source": profile.source,
            "devices": [d.to_dict() for d in profile.devices],
            "free_vram_mb": profile.free_vram_mb,
            "total_vram_mb": profile.total_vram_mb,
        }
    except Exception as exc:  # 画像非关键路径，失败不阻断面板
        return {"ok": False, "summary": "硬件画像不可用：{}".format(exc)}


def _parse_data_uri(raw: str, name: str = "") -> Attachment:
    """把前端传来的 data URI 还原为附件对象。"""
    text = (raw or "").strip()
    mime = ""
    payload = text
    if text.startswith("data:"):
        head, _, payload = text.partition(",")
        mime = head[len("data:") :].split(";")[0]
    try:
        blob = base64.b64decode(payload, validate=True)
    except Exception:
        raise ValueError("图片数据不是合法的 base64")
    return make_attachment(name or "pasted.png", mime, blob)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "TDS-UI"

    # ---- 基础工具 ----

    @property
    def app(self) -> "UiServer":
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        if self.app.verbose:
            sys.stderr.write("[tds.ui] {} {}\n".format(self.address_string(), fmt % args))
            sys.stderr.flush()

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except ValueError:
            raise ValueError("请求体不是合法 JSON")
        return parsed if isinstance(parsed, dict) else {}

    def _begin_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True

    def _sse(self, payload: Dict[str, Any]) -> bool:
        """写一帧 SSE；连接已断开时返回 False。"""
        try:
            self.wfile.write(("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode("utf-8"))
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False

    # ---- 路由 ----

    def do_GET(self) -> None:  # noqa: N802
        self._route("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._route("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._route("DELETE")

    def _route(self, method: str) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path or "/"
        try:
            if method == "GET" and path in ("/", "/index.html"):
                return self._serve_panel()
            if method == "GET" and path.startswith("/assets/"):
                return self._serve_asset(path[len("/assets/") :])
            if path == "/api/status" and method == "GET":
                return self._send_json(self.app.status())
            if path == "/api/tds/config" and method == "GET":
                return self._send_json(self.app.tds_config())
            if path == "/api/tds/config" and method == "POST":
                return self._send_json(self.app.tds_update(self._read_json()))
            if path == "/api/tds/state" and method == "GET":
                return self._send_json(self.app.tds_state())
            if path == "/api/models" and method == "GET":
                return self._send_json(self.app.models(force=True))
            if path == "/api/sessions" and method == "GET":
                return self._send_json({"sessions": self.app.store.list()})
            if path == "/api/sessions" and method == "POST":
                return self._create_session()
            if path == "/api/chat" and method == "POST":
                return self._chat()
            if path.startswith("/api/sessions/"):
                return self._session_op(method, path[len("/api/sessions/") :])
            if path == "/favicon.ico":
                return self._send_bytes(b"", "image/x-icon", 204)
            self._send_json({"error": "未找到接口：{}".format(path)}, 404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, 400)
        except KeyError as exc:
            self._send_json({"error": str(exc)}, 404)
        except BackendError as exc:
            self._send_json({"error": str(exc)}, 502)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as exc:  # 兜底，避免单请求异常打死线程
            if self.app.verbose:
                import traceback

                traceback.print_exc()
            try:
                self._send_json({"error": "服务端异常：{}".format(exc)}, 500)
            except Exception:
                self.close_connection = True

    # ---- 静态资源 ----

    def _serve_panel(self) -> None:
        index = os.path.join(WEB_DIR, "index.html")
        if not os.path.isfile(index):
            return self._send_json({"error": "面板文件缺失：{}".format(index)}, 500)
        with open(index, "rb") as fh:
            return self._send_bytes(fh.read(), "text/html; charset=utf-8")

    def _serve_asset(self, rel: str) -> None:
        rel = urllib.parse.unquote(rel)
        target = os.path.normpath(os.path.join(WEB_DIR, rel))
        if not target.startswith(WEB_DIR) or not os.path.isfile(target):
            return self._send_json({"error": "资源不存在"}, 404)
        ctype = mimetypes.guess_type(target)[0] or "application/octet-stream"
        with open(target, "rb") as fh:
            return self._send_bytes(fh.read(), ctype)

    # ---- 会话接口 ----

    def _create_session(self) -> None:
        body = self._read_json()
        model = str(body.get("model") or self.app.default_model or "")
        session = self.app.store.create(model=model, system=str(body.get("system") or ""))
        self._send_json({"session": session.summary()})

    def _session_op(self, method: str, rest: str) -> None:
        parts = [p for p in rest.split("/") if p]
        if not parts:
            return self._send_json({"error": "缺少会话 ID"}, 400)
        sid = parts[0]
        action = parts[1] if len(parts) > 1 else ""

        if method == "GET" and not action:
            return self._send_json({"session": self.app.store.require(sid).to_dict()})
        if method == "DELETE" and not action:
            ok = self.app.store.delete(sid)
            return self._send_json({"deleted": ok}, 200 if ok else 404)
        if method == "POST" and action == "clear":
            session = self.app.store.require(sid)
            session.clear()
            return self._send_json({"session": session.to_dict()})
        if method == "POST" and action == "params":
            return self._update_params(sid)
        self._send_json({"error": "不支持的操作"}, 404)

    def _update_params(self, sid: str) -> None:
        session = self.app.store.require(sid)
        body = self._read_json()
        if "model" in body:
            session.model = str(body["model"])
        if "system" in body:
            session.system = str(body["system"])
        if "temperature" in body:
            session.temperature = max(0.0, min(2.0, float(body["temperature"])))
        if "top_p" in body:
            session.top_p = max(0.01, min(1.0, float(body["top_p"])))
        if "max_tokens" in body:
            session.max_tokens = max(0, int(body["max_tokens"]))
        if "title" in body:
            session.title = str(body["title"])[:40] or session.title
        session.updated_at = time.time()
        self._send_json({"session": session.to_dict()})

    # ---- 对话主通道 ----

    # ---- 调度层预判（可选） ----

    def _record_dispatch_trace(self, task_desc: str) -> List[str]:
        """调度层对本轮对话做一次预判，返回本轮新增的决策轨迹行。

        仅走「语义分类 → 生成计划 → 记录决策」，不驱动适配层显存动作
        （adapter 为空时 apply 直接返回 False），因此不会影响测试台稳定性。
        返回空列表表示未挂载调度器或预判失败，前端据此不渲染轨迹区。
        """
        scheduler = self.app.scheduler
        if scheduler is None:
            return []
        try:
            scheduler.run_task(task_desc)
            entry = scheduler.trace.last()
            return [entry.line()] if entry is not None else []
        except Exception:
            return []

    def _chat(self) -> None:
        body = self._read_json()
        session = self.app.store.require(str(body.get("session_id") or ""))
        content = str(body.get("content") or "")
        raw_images: List[Dict[str, Any]] = list(body.get("images") or [])
        model = str(body.get("model") or session.model or self.app.default_model or "")

        attachments: List[Attachment] = []
        for item in raw_images[:4]:
            if not isinstance(item, dict):
                continue
            attachments.append(_parse_data_uri(str(item.get("data") or ""), str(item.get("name") or "")))

        params = body.get("params") or {}
        if isinstance(params, dict):
            if "temperature" in params:
                session.temperature = max(0.0, min(2.0, float(params["temperature"])))
            if "top_p" in params:
                session.top_p = max(0.01, min(1.0, float(params["top_p"])))
            if "max_tokens" in params:
                session.max_tokens = max(0, int(params["max_tokens"]))
            if "system" in params:
                session.system = str(params["system"])
        if model:
            session.model = model
        if not content.strip() and not attachments:
            raise ValueError("消息内容为空")
        if not session.model:
            raise ValueError("未选择模型")

        session.add_user(content, attachments)
        history = session.history_for_model()[:-1]  # 末条为刚入队消息，由 content 单独承载
        payload_messages = history + [
            m.to_openai() for m in session.messages[-1:]
        ]

        request = ChatRequest(
            model=session.model,
            messages=payload_messages,
            system=session.system,
            temperature=session.temperature,
            top_p=session.top_p,
            max_tokens=session.max_tokens,
            stream=True,
        )

        self._begin_sse()
        if not self._sse(
            {
                "type": "start",
                "session_id": session.id,
                "model": session.model,
                "images": len(attachments),
                "history_len": len(history),
            }
        ):
            return

        started = time.time()
        ttft_ms = 0.0
        pieces: List[str] = []
        usage: Dict[str, Any] = {}
        error = ""
        try:
            for chunk in self.app.client.chat_stream(request):
                if chunk.error:
                    error = chunk.error
                    break
                if chunk.text or chunk.reasoning:
                    if not pieces and chunk.text:
                        ttft_ms = (time.time() - started) * 1000
                    if chunk.text:
                        pieces.append(chunk.text)
                    if not self._sse({"type": "delta", "text": chunk.text, "reasoning": chunk.reasoning}):
                        break
                if chunk.usage:
                    usage = chunk.usage
                if chunk.done:
                    break
        except BackendError as exc:
            error = str(exc)

        text = "".join(pieces)
        total_ms = (time.time() - started) * 1000
        stats = {
            "model": session.model,
            "images": len(attachments),
            "ttft_ms": round(ttft_ms, 1),
            "total_ms": round(total_ms, 1),
            "chars": len(text),
            "chars_per_s": round(len(text) * 1000.0 / total_ms, 1) if total_ms > 0 else 0.0,
            "usage": usage,
        }

        # 调度层对本轮对话做一次预判，采集本轮新增的决策轨迹
        trace_lines = self._record_dispatch_trace(
            "model={} images={} history={} :: {}".format(
                session.model, len(attachments), len(history), content[:200]
            )
        )

        message = session.add_assistant(text, stats)
        if error:
            message.error = error

        done_payload = {
            "type": "error" if error else "done",
            "message": error,
            "content": text,
            "stats": stats,
            "session": session.summary(),
        }
        if trace_lines:
            done_payload["trace"] = trace_lines


class UiServer:
    """测试台服务主体：绑定后端客户端与会话容器。

    设计约束
    --------
    面板本身（HTML）不参与调度；但测试台作为「观测台」，在对话完成后的
    ``done`` 事件里附带调度层的决策轨迹（如果有），供前端展示。
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8770,
        backend_url: str = DEFAULT_BACKEND,
        api_key: str = "",
        default_model: str = "",
        verbose: bool = True,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.backend_url = backend_url or DEFAULT_BACKEND
        self.client = BackendClient(self.backend_url, api_key)
        self.store = SessionStore()
        self.default_model = default_model
        self.verbose = verbose
        self._models: List[ModelInfo] = []
        self._httpd: Optional[ThreadingHTTPServer] = None
        self.scheduler: Optional[Any] = None

    # ---- 探测 ----

    def models(self, force: bool = False) -> Dict[str, Any]:
        if force or not self._models:
            try:
                self._models = self.client.list_models()
            except BackendError:
                self._models = []
        if self.default_model and not any(m.name == self.default_model for m in self._models):
            self._models.insert(0, ModelInfo(name=self.default_model))
        return {
            "models": [m.to_dict() for m in self._models],
            "default": self.default_model or (self._models[0].name if self._models else ""),
        }

    def status(self) -> Dict[str, Any]:
        info = self.client.ping()
        return {
            "panel": "TDS 对话测试台",
            "backend": info,
            "hardware": _hardware_snapshot(),
            "sessions": len(self.store),
            "server_time": time.time(),
        }

    # ---- TDS 调度控制面板（参数暴露 + 实时状态） ----

    # 面板可调参数白名单：key -> (类型, 最小值, 最大值)
    _TDS_EDITABLE = {
        "vram_budget_mb": (int, 256, 64 * 1024),
        "reserve_mb": (int, 0, 32 * 1024),
        "keep_steps": (int, 0, 1024),
        "prefetch_enabled": (bool, None, None),
        "pressure_guard": (bool, None, None),
        "pressure_ratio": (float, 0.1, 1.0),
        "frozen_ratio": (float, 0.1, 1.0),
        "trace_maxlen": (int, 10, 100000),
    }

    def tds_config(self) -> Dict[str, Any]:
        """当前 TDS 调度器可调参数（scheduler 未挂载时返回空配置）。"""
        sched = self.scheduler
        if sched is None:
            return {"ok": False, "mounted": False, "config": {}}
        cfg = sched.cfg
        return {
            "ok": True,
            "mounted": True,
            "config": {
                key: getattr(cfg, key)
                for key in self._TDS_EDITABLE
                if hasattr(cfg, key)
            },
        }

    def tds_update(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """按白名单校验并更新 TDSConfig 可调参数。

        只接受已列入 _TDS_EDITABLE 的键；越界、类型不符或未挂载均报错。
        成功后同步刷新调度器轨迹容量（trace_maxlen 生效）。
        """
        sched = self.scheduler
        if sched is None:
            return {"ok": False, "mounted": False, "error": "调度器未挂载"}
        cfg = sched.cfg
        updated: Dict[str, Any] = {}
        errors: List[str] = []
        for key, raw in body.items():
            spec = self._TDS_EDITABLE.get(key)
            if spec is None:
                errors.append("不支持参数：{}".format(key))
                continue
            cast, lo, hi = spec
            try:
                if cast is bool:
                    if isinstance(raw, str):
                        val = raw.strip().lower() in ("1", "true", "yes", "on")
                    else:
                        val = bool(raw)
                else:
                    val = cast(raw)
                    if lo is not None and val < lo:
                        raise ValueError("{} 不能小于 {}".format(key, lo))
                    if hi is not None and val > hi:
                        raise ValueError("{} 不能大于 {}".format(key, hi))
            except (TypeError, ValueError) as exc:
                errors.append("{}={!r} 非法：{}".format(key, raw, exc))
                continue
            setattr(cfg, key, val)
            updated[key] = val
        if "trace_maxlen" in updated:
            try:
                sched.trace.resize(int(updated["trace_maxlen"]))
            except Exception:
                pass  # 轨迹容量调整失败不阻断面板保存
        if errors:
            return {"ok": False, "updated": updated, "errors": errors}
        return {"ok": True, "updated": updated}

    def tds_state(self) -> Dict[str, Any]:
        """实时调度状态：压力水位、指标、决策轨迹摘要（含阴阳模式分布）。"""
        sched = self.scheduler
        if sched is None:
            return {"ok": False, "mounted": False}
        pressure = "low"
        vram_state = None
        try:
            if sched.adapter is not None and hasattr(sched.adapter, "vram_state"):
                vram_state = sched.adapter.vram_state()
        except Exception:
            vram_state = None
        if vram_state is not None:
            pressure = sched.pressure_state(vram_state)
        try:
            trace_report = sched.trace_report()
        except Exception:
            trace_report = {}
        # 阴阳模式分布：从轨迹 note 中统计 mode=
        mode_count: Dict[str, int] = {}
        try:
            for entry in sched.trace.entries():
                note = entry.note or ""
                if "mode=" in note:
                    mode = note.split("mode=", 1)[1].split("|", 1)[0].strip()
                    mode_count[mode] = mode_count.get(mode, 0) + 1
        except Exception:
            pass
        return {
            "ok": True,
            "mounted": True,
            "pressure": pressure,
            "vram": {
                "total_mb": vram_state.total_mb,
                "used_mb": vram_state.used_mb,
                "free_mb": vram_state.free_mb,
                "fragmentation_mb": vram_state.fragmentation_mb,
                "usage_ratio": round(vram_state.usage_ratio, 4),
            } if vram_state is not None else None,
            "metrics": sched.report(),
            "trace": trace_report,
            "yang_yin": mode_count,
            "step": sched.step,
        }

    # ---- 生命周期 ----

    def build(self) -> ThreadingHTTPServer:
        httpd = ThreadingHTTPServer((self.host, self.port), _Handler)
        httpd.daemon_threads = True
        httpd.app = self  # type: ignore[attr-defined]
        self._httpd = httpd
        return httpd

    def serve_forever(self) -> None:
        httpd = self._httpd or self.build()
        self.port = httpd.server_address[1]
        if self.verbose:
            print("TDS 对话测试台已启动")
            print("  面板地址 : http://{}:{}/".format(self.host, self.port))
            print("  模型后端 : {}".format(self.backend_url))
            probe = self.client.ping()
            print("  后端状态 : {}".format(
                "在线（{} {}）".format(probe["kind"], probe["version"])
                if probe["online"]
                else "离线 · {}".format(probe["error"])
            ))
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()


def serve(
    host: str = "127.0.0.1",
    port: int = 8770,
    backend_url: str = DEFAULT_BACKEND,
    api_key: str = "",
    default_model: str = "",
    verbose: bool = True,
    scheduler: Optional[Any] = None,
) -> None:
    """便捷入口：构建并阻塞运行测试台。

    scheduler 参数接收 TDS 调度器实例，用于在对话完成后附带决策轨迹。
    """
    srv = UiServer(
        host=host,
        port=port,
        backend_url=backend_url,
        api_key=api_key,
        default_model=default_model,
        verbose=verbose,
    )
    srv.scheduler = scheduler
    srv.serve_forever()
