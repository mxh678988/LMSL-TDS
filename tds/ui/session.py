"""测试台会话模型：消息、附件与多会话存储。

设计约束
--------
1. 会话数据为**纯内存**，落盘仅使用标准库 JSON 快照（原子替换写入），
   不引入数据库依赖；
2. 面向 OpenAI 消息协议组织历史，多模态图片以 data URI 形式内联（零依赖，
   无需本地文件服务）；
3. 每次调用产生的时延 / 吞吐统计挂在 assistant 消息上，供测试台做能力对比；
4. 快照可裁剪图片正文（with_content=False）用于轻量归档。
"""
from __future__ import annotations

import base64
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

SNAPSHOT_VERSION = 1

MAX_IMAGES_PER_MESSAGE = 4
MAX_IMAGE_BYTES = 12 * 1024 * 1024
ALLOWED_IMAGE_MIME = ("image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp")


@dataclass
class Attachment:
    """一条图片附件（内联 base64）。"""

    name: str
    mime: str
    data: str  # 纯 base64（不含 data: 前缀）
    size: int = 0

    @property
    def data_uri(self) -> str:
        return "data:{};base64,{}".format(self.mime, self.data)

    def to_dict(self, with_data: bool = True) -> Dict[str, Any]:
        data = asdict(self)
        if not with_data:
            data["data"] = ""
        return data

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "Attachment":
        return Attachment(
            name=str(raw.get("name") or "image"),
            mime=str(raw.get("mime") or ""),
            data=str(raw.get("data") or ""),
            size=int(raw.get("size") or 0),
        )


def make_attachment(name: str, mime: str, raw: bytes) -> Attachment:
    """由原始字节构造附件，超限或类型不支持时抛 ``ValueError``。"""
    mime = (mime or "").split(";")[0].strip().lower()
    if mime not in ALLOWED_IMAGE_MIME:
        raise ValueError("不支持的图片类型：{}".format(mime or "未知"))
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("图片超过 {} MB 上限".format(MAX_IMAGE_BYTES // (1024 * 1024)))
    return Attachment(
        name=name or "image",
        mime=mime,
        data=base64.b64encode(raw).decode("ascii"),
        size=len(raw),
    )


@dataclass
class Message:
    """一条会话消息。"""

    role: str  # user | assistant | system
    content: str = ""
    images: List[Attachment] = field(default_factory=list)
    ts: float = field(default_factory=time.time)
    stats: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_openai(self) -> Dict[str, Any]:
        """转为 OpenAI 消息体；带图时使用 content 数组形式。"""
        if not self.images:
            return {"role": self.role, "content": self.content}
        parts: List[Dict[str, Any]] = []
        if self.content:
            parts.append({"type": "text", "text": self.content})
        for img in self.images:
            parts.append({"type": "image_url", "image_url": {"url": img.data_uri}})
        return {"role": self.role, "content": parts}

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["images"] = [img.to_dict() for img in self.images]
        return data

    def summary(self) -> Dict[str, Any]:
        """前端列表用轻量视图（不含图片正文）。"""
        return {
            "role": self.role,
            "content": self.content,
            "ts": self.ts,
            "stats": self.stats,
            "error": self.error,
            "images": [
                {"name": img.name, "mime": img.mime, "size": img.size} for img in self.images
            ],
        }

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "Message":
        return Message(
            role=str(raw.get("role") or "user"),
            content=str(raw.get("content") or ""),
            images=[Attachment.from_dict(i) for i in (raw.get("images") or [])],
            ts=float(raw.get("ts") or time.time()),
            stats=dict(raw.get("stats") or {}),
            error=str(raw.get("error") or ""),
        )


@dataclass
class ChatSession:
    """一条独立测试会话。"""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = "新测试会话"
    model: str = ""
    system: str = ""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 0
    messages: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ---- 变更 ----

    def add_user(self, content: str, images: Optional[List[Attachment]] = None) -> Message:
        msg = Message(role="user", content=content, images=list(images or []))
        self.messages.append(msg)
        if self.title in ("", "新测试会话"):
            seed = content.strip().replace("\n", " ")
            self.title = (seed[:18] + "…") if len(seed) > 18 else (seed or "图片输入")
        self.updated_at = time.time()
        return msg

    def add_assistant(self, content: str, stats: Optional[Dict[str, Any]] = None) -> Message:
        msg = Message(role="assistant", content=content, stats=dict(stats or {}))
        self.messages.append(msg)
        self.updated_at = time.time()
        return msg

    def clear(self) -> None:
        self.messages.clear()
        self.updated_at = time.time()

    # ---- 视图 ----

    def history_for_model(self) -> List[Dict[str, Any]]:
        """排除报错消息，按协议组织历史。"""
        return [m.to_openai() for m in self.messages if not m.error]

    def to_dict(self, with_content: bool = True) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "model": self.model,
            "system": self.system,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [m.to_dict() if with_content else m.summary() for m in self.messages],
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "model": self.model,
            "count": len(self.messages),
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "ChatSession":
        session = ChatSession(
            id=str(raw.get("id") or uuid.uuid4().hex[:12]),
            title=str(raw.get("title") or "新测试会话"),
            model=str(raw.get("model") or ""),
            system=str(raw.get("system") or ""),
            temperature=float(raw.get("temperature", 0.7)),
            top_p=float(raw.get("top_p", 0.9)),
            max_tokens=int(raw.get("max_tokens") or 0),
            created_at=float(raw.get("created_at") or time.time()),
            updated_at=float(raw.get("updated_at") or time.time()),
        )
        session.messages = [Message.from_dict(m) for m in (raw.get("messages") or [])]
        return session


class SessionStore:
    """进程内多会话容器。"""

    def __init__(self) -> None:
        self._sessions: Dict[str, ChatSession] = {}

    def create(self, model: str = "", system: str = "") -> ChatSession:
        session = ChatSession(model=model, system=system)
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Optional[ChatSession]:
        return self._sessions.get(session_id)

    def require(self, session_id: str) -> ChatSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("会话不存在：{}".format(session_id))
        return session

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def list(self) -> List[Dict[str, Any]]:
        rows = [s.summary() for s in self._sessions.values()]
        return sorted(rows, key=lambda r: r["updated_at"], reverse=True)

    # ---- 落盘 / 恢复 ----
    def _ordered(self) -> List[ChatSession]:
        return sorted(self._sessions.values(), key=lambda s: s.updated_at, reverse=True)

    def to_dict(self, with_content: bool = True) -> Dict[str, Any]:
        """整体快照（含版本号与保存时间，便于后续迁移）。"""
        return {
            "version": SNAPSHOT_VERSION,
            "saved_at": time.time(),
            "sessions": [s.to_dict(with_content=with_content) for s in self._ordered()],
        }

    def from_dict(self, raw: Dict[str, Any], merge: bool = False) -> int:
        """从快照恢复会话；``merge=False`` 时先清空现有内存态。返回恢复条数。"""
        if not isinstance(raw, dict):
            raise ValueError("会话快照格式非法：顶层应为对象")
        if not merge:
            self._sessions.clear()
        count = 0
        for item in raw.get("sessions") or []:
            session = ChatSession.from_dict(item)
            self._sessions[session.id] = session
            count += 1
        return count

    def save(self, path: str, with_content: bool = True) -> str:
        """原子写快照：先写同目录 ``.tmp`` 再替换，避免半截文件。返回绝对路径。"""
        target = Path(path)
        if target.suffix.lower() != ".json":
            target = target.with_suffix(".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        payload = json.dumps(self.to_dict(with_content=with_content), ensure_ascii=False, indent=2)
        tmp.write_text(payload, encoding="utf-8")
        os.replace(str(tmp), str(target))
        return str(target)

    def load(self, path: str, merge: bool = False) -> int:
        """读取快照文件并恢复会话，返回恢复条数。"""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return self.from_dict(raw, merge=merge)

    def __len__(self) -> int:
        return len(self._sessions)
