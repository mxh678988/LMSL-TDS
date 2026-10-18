# -*- coding: utf-8 -*-
"""太易 · 紫梅桌面 —— 独立软件后端（零第三方依赖，仅标准库）。

太极紫梅系列作为终端桌面产品：宿主应用承载五大分支
    - 对话（太极chat）      /api/chat    转发本地 Ollama（可探测）
    - 创作台（多模态交互）   /api/comfy   探测本地 ComfyUI 状态
    - 模型管理（太极LLM）   /api/models  列出本地 Ollama 模型
    - 知识库（数据模型填喂） /api/kb      只读索引候选知识目录
    - 引擎舱（TDS 调度内核） /api/zimei /api/evolution /api/testing
引擎舱页面 = web/engine.html（原调度控制台，默认折叠为状态卡）。
不 import torch、不依赖 ComfyUI 运行时，可独立启动。

用法：
    python zimei_app/server.py [--port 8771] [--host 127.0.0.1] [--open]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional
from urllib.parse import urlparse, parse_qs
from urllib.request import urlopen, Request

APP_NAME = "太易 · 紫梅桌面"
APP_NAME_EN = "TAIYI ZiMei Desktop"
APP_VERSION = "1.0.0"
APP_SLOGAN = "命宫为尊 · 卦象定序 · 生克制衡 · 自进化不辍"

# ---- 工程根定位（zimei_app 位于 TDS 仓库根下） ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

WEB_DIR = os.path.join(_HERE, "web")
STATIC_FILES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/engine": "engine.html",
    "/engine.html": "engine.html",
    "/assets/logo.png": "assets/logo.png",
}
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}

# 八卦域中文展示
DOMAIN_CN = {
    "daima": "乾·代码", "jiansuo": "坤·检索", "tuxiang": "震·图像",
    "gongju": "巽·工具", "shipin": "坎·视频", "duihua": "离·对话",
    "qian": "艮·前端", "qita": "兑·其他",
}
WUXING_CN = {"金": "金", "木": "木", "水": "水", "火": "火", "土": "土"}
RELATION_CN = {
    "ming": "命宫（绝对保护）", "cai": "财·我克者（最先舍）",
    "shishang": "食伤·我生者", "bijie": "比劫·同我者",
    "yin": "印·生我者（保留）", "guan": "官·克我者（最后动）",
}
WUXING_HEX = {"金": "#e6c66b", "木": "#4cc38a", "水": "#4a9df5",
              "火": "#f2555a", "土": "#c9a86a"}


def _load_zimei(ming: str) -> Dict:
    """读取紫梅内核：八卦五行、命宫关系、动态淘汰序（真实内核函数）。"""
    from tds.kv import taiji_zimei as tz
    if ming not in tz.BAGUA_WUXING:
        ming = "duihua"
    order = tz.zimei_evict_order(ming)
    domains = []
    for d in tz.BAGUA_WUXING:
        wux = tz.wuxing_of(d)
        rel = tz.relation_to_ming(d, ming)
        domains.append({
            "key": d, "cn": DOMAIN_CN.get(d, d),
            "wuxing": wux, "wuxing_hex": WUXING_HEX.get(wux, "#888"),
            "relation": rel, "relation_cn": RELATION_CN.get(rel, rel),
            "priority": tz.zimei_evict_priority(d, ming),
            "is_ming": d == ming,
        })
    return {
        "ming": ming, "ming_cn": DOMAIN_CN.get(ming, ming),
        "ming_wuxing": tz.wuxing_of(ming),
        "order": [{"key": d, "cn": DOMAIN_CN.get(d, d),
                   "wuxing": tz.wuxing_of(d),
                   "wuxing_hex": WUXING_HEX.get(tz.wuxing_of(d), "#888"),
                   "relation": tz.relation_to_ming(d, ming),
                   "priority": tz.zimei_evict_priority(d, ming)} for d in order],
        "domains": domains,
    }


def _load_evolution() -> Dict:
    """读取双线进化引擎档案与白名单（只读，不触发进化）。"""
    from tds.evolution.config import EvolutionConfig, default_archive_dir
    from tds.evolution.engine import EvolutionEngine, _auto_defaults, os_path_join
    from tds.evolution.ip_engine import IPEvolutionEngine
    from tds.evolution.whitelist import Whitelist

    cfg = EvolutionConfig(enabled=False)
    cfg.ensure_dirs()
    whitelist = Whitelist()
    defaults = _auto_defaults(whitelist)

    software = EvolutionEngine(
        config=cfg, whitelist=whitelist, defaults=defaults,
        archive=None, proposal_dir=None,
    )
    ip = IPEvolutionEngine(config=cfg)

    params = []
    for path in whitelist.paths():
        spec = whitelist.get(path)
        if spec is None:
            continue
        cur = software.store.get(path)
        params.append({
            "path": ".".join(path),
            "kind": spec.kind, "risk": spec.risk.value,
            "current": cur if cur is not None else "-",
            "min": spec.min_value, "max": spec.max_value,
            "desc": spec.desc,
        })

    sw_data = software.archive.get_data()
    ip_data = ip.archive.get_data()
    return {
        "enabled": cfg.enabled,
        "auto_apply_low": cfg.auto_apply_low,
        "pressure_guard": cfg.pressure_guard,
        "archive_dir": cfg.archive_dir,
        "software": {
            "line": "software",
            "snapshot": software.archive.snapshot(),
            "changelog": software.archive.recent_changelog(12),
            "raw": {k: sw_data.get(k) for k in ("changes", "backups")},
        },
        "ip": {
            "line": "ip",
            "snapshot": ip.archive.snapshot(),
            "changelog": ip.archive.recent_changelog(12),
            "terms": ip.store.all_terms(),
            "weights": ip.store.snapshot().get("weights", {}),
        },
        "params": params,
    }


def _load_testing() -> Dict:
    """读取测试技术接口分支组件（只读状态/能力清单）。"""
    from tds import testing as T
    components = [
        {"key": "PressureInjector", "cn": "压力注入器",
         "desc": "正弦/尖峰压力场景注入，验证四象压力守卫",
         "ready": True, "apis": ["next_ratio()", "run(fn, expect_pressure)", "save()"]},
        {"key": "ABTrial", "cn": "A/B 试验",
         "desc": "基线参数 vs 候选参数对比，产出胜出报告",
         "ready": True, "apis": ["run(simulate)", "report(results)"]},
        {"key": "RegressionBenchmark", "cn": "回归基准",
         "desc": "注册指标阈值，防自进化引入性能回退",
         "ready": True, "apis": ["register(name, metric, min_value)", "run(fn)", "all_passed()"]},
        {"key": "Probe", "cn": "运行探针",
         "desc": "打点观测调度耗时/压力，支持触发链路定位",
         "ready": True, "apis": ["watch(point)", "fire(elapsed_ms, ratio)", "summary()"]},
    ]
    scenario = T.PressureScenario.sine_wave(steps=24)
    return {
        "components": components,
        "sample_scenario": {
            "name": scenario.name,
            "steps": len(scenario.steps),
            "ratios": [p["ratio"] for p in scenario.steps],
        },
    }


# ---------------------------------------------------------------------------
# 宿主分支探测（零依赖：仅标准库 urllib，全部只读）
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://127.0.0.1:11434"
COMFY_URL = "http://127.0.0.1:8188"
_HTTP_TIMEOUT = 1.5


def _http_get(url: str, timeout: float = _HTTP_TIMEOUT):
    req = Request(url, headers={"User-Agent": APP_NAME_EN}, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", "replace")


def _probe_ollama() -> Dict:
    """探测本地 Ollama：可用性 + 模型清单。"""
    try:
        status, body = _http_get(OLLAMA_URL + "/api/tags")
        if status != 200:
            return {"available": False, "url": OLLAMA_URL, "error": "http %d" % status}
        data = json.loads(body or "{}")
        models = []
        for m in data.get("models", []):
            name = m.get("name", "?")
            size = m.get("size", 0)
            models.append({
                "name": name,
                "size_gb": round(size / (1024 ** 3), 2),
            })
        return {"available": True, "url": OLLAMA_URL,
                "count": len(models), "models": models}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "url": OLLAMA_URL, "error": str(e)}


def _probe_comfy() -> Dict:
    """探测本地 ComfyUI（127.0.0.1:8188）。"""
    try:
        status, body = _http_get(COMFY_URL + "/system_stats")
        if status != 200:
            return {"available": False, "url": COMFY_URL, "error": "http %d" % status}
        data = json.loads(body or "{}")
        devs = []
        sysinfo = data.get("system", {})
        for d in data.get("devices", []):
            devs.append({
                "name": d.get("name", "?"),
                "type": d.get("type", "?"),
                "vram_total_gb": round((d.get("vram_total") or 0) / (1024 ** 3), 1),
                "vram_free_gb": round((d.get("vram_free") or 0) / (1024 ** 3), 1),
            })
        return {"available": True, "url": COMFY_URL,
                "devices": devs, "comfy_version": sysinfo.get("comfyui_version", "?")}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "url": COMFY_URL, "error": str(e)}


def _kb_index() -> Dict:
    """知识库只读索引：扫描 TDS 工程内候选知识目录（docs 等）。"""
    candidates = [
        os.path.join(_REPO_ROOT, "docs"),
        os.path.join(_HERE, "docs"),
    ]
    found = []
    for base in candidates:
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if not d.startswith((".", "__", "node_modules"))]
            for fn in files:
                if fn.lower().endswith((".md", ".txt", ".json", ".yaml", ".yml")):
                    found.append(os.path.relpath(os.path.join(root, fn), _REPO_ROOT))
    found.sort()
    return {"roots": [os.path.relpath(b, _REPO_ROOT) for b in candidates
                      if os.path.isdir(b)],
            "count": len(found), "files": found[:200]}


def _chat_with_ollama(message: str, model: Optional[str] = None) -> Dict:
    """对话代理：把本地消息转发给 Ollama /api/chat（只读无状态，不落盘）。"""
    probe = _probe_ollama()
    if not probe["available"]:
        return {"ok": False, "error": "Ollama 未在 %s 运行" % OLLAMA_URL,
                "hint": "请先启动 Ollama（ollama serve）"}
    model = model or (probe["models"][0]["name"] if probe["models"] else "")
    if not model:
        return {"ok": False, "error": "本机无可用模型"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "stream": False,
        "options": {"temperature": 0.7},
    }
    try:
        req = Request(
            OLLAMA_URL + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "User-Agent": APP_NAME_EN},
            method="POST",
        )
        with urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8", "replace")
        data = json.loads(body or "{}")
        content = (data.get("message") or {}).get("content", "")
        return {"ok": True, "model": model, "content": content,
                "done": data.get("done", False),
                "eval_count": data.get("eval_count", 0),
                "eval_duration_ms": round((data.get("eval_duration") or 0) / 1e6, 1)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": "Ollama 转发失败: %s" % e}


class Handler(BaseHTTPRequestHandler):
    server_version = "TAIYI-ZiMei/1.0"

    def log_message(self, fmt, *args):  # 精简日志
        pass

    def _send_json(self, obj: Dict, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, rel: str) -> None:
        path = os.path.normpath(os.path.join(WEB_DIR, rel))
        if not path.startswith(WEB_DIR) or not os.path.isfile(path):
            self._send_json({"error": "not found"}, 404)
            return
        ext = os.path.splitext(path)[1].lower()
        ctype = CONTENT_TYPES.get(ext, "application/octet-stream")
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            self._send_json({"error": "read failed"}, 500)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path in STATIC_FILES:
                self._send_file(STATIC_FILES[path])
                return
            if path == "/api/status":
                self._send_json({
                    "app": APP_NAME, "app_en": APP_NAME_EN,
                    "version": APP_VERSION, "slogan": APP_SLOGAN,
                    "kernel": "tds.kv.taiji_zimei + tds.evolution + tds.testing",
                    "ok": True,
                })
                return
            if path == "/api/zimei":
                q = parse_qs(parsed.query)
                ming = (q.get("ming") or ["duihua"])[0]
                self._send_json(_load_zimei(ming))
                return
            if path == "/api/evolution":
                self._send_json(_load_evolution())
                return
            if path == "/api/testing":
                self._send_json(_load_testing())
                return
            if path == "/api/models":
                self._send_json(_probe_ollama())
                return
            if path == "/api/comfy":
                self._send_json(_probe_comfy())
                return
            if path == "/api/kb":
                self._send_json(_kb_index())
                return
            if path == "/api/chat":
                probe = _probe_ollama()
                self._send_json({
                    "ok": probe["available"],
                    "service": "Ollama",
                    "url": OLLAMA_URL,
                    "models": probe.get("models", []),
                    "error": probe.get("error"),
                })
                return
            self._send_json({"error": "not found", "path": path}, 404)
        except Exception as e:  # noqa: BLE001 - 后端容错，前端可见错误
            self._send_json({"error": "internal", "detail": str(e)}, 500)

    def do_POST(self) -> None:  # noqa: N802
        """POST 路由：目前仅 /api/chat 对话代理（转发本地 Ollama）。"""
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/chat":
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    body = json.loads(raw.decode("utf-8", "replace") or "{}")
                except ValueError:
                    body = {}
                message = (body.get("message") or "").strip()
                model = body.get("model")
                if not message:
                    self._send_json({"ok": False, "error": "message 为空"}, 400)
                    return
                self._send_json(_chat_with_ollama(message, model))
                return
            self._send_json({"error": "not found", "path": path}, 404)
        except Exception as e:  # noqa: BLE001 - 后端容错，前端可见错误
            self._send_json({"error": "internal", "detail": str(e)}, 500)


def main() -> None:
    ap = argparse.ArgumentParser(description=APP_NAME)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    args = ap.parse_args()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = "http://{}:{}/".format(args.host, args.port)
    print("=" * 64)
    print("  {} v{}".format(APP_NAME, APP_VERSION))
    print("  {} · {}".format(APP_NAME_EN, APP_SLOGAN))
    print("  服务地址: {}".format(url))
    print("  内核接入: tds.kv.taiji_zimei / tds.evolution / tds.testing")
    print("  按 Ctrl+C 停止")
    print("=" * 64)
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
