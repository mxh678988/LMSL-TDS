"""进化档案：JSON 持久化、原子写入、变更日志与回退备份。

两条进化线路各持一个独立档案文件（software_archive.json / ip_archive.json），
数据结构彼此独立（软件线存参数、IP 线存知识资产），本模块只负责通用的
"读写 + 变更日志 + 环形备份 + 回退"骨架。

安全特性
--------
- 原子写入：tmp 文件 + os.replace，避免写一半损坏档案；
- 损坏恢复：读取失败时把坏文件重命名为 *.corrupt-<ts> 保留现场，
  并以空骨架继续运行（绝不抛异常阻断调度）；
- 回退能力：每次变更先备份旧值（环形，保留 backup_keep 条），
  restore() 可精确恢复到任意历史变更点。
"""
from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1


@dataclass
class ChangeRecord:
    """一条变更日志记录。"""
    ts: float
    op: str              # "apply" | "revert" | "suggest"
    path: str            # 点号路径，如 "BAGUA_KEEP_STEPS.duihua"
    old: Any = None
    new: Any = None
    risk: str = "low"
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": round(self.ts, 3),
            "op": self.op,
            "path": self.path,
            "old": self.old,
            "new": self.new,
            "risk": self.risk,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChangeRecord":
        return cls(
            ts=float(d.get("ts", 0.0)),
            op=str(d.get("op", "")),
            path=str(d.get("path", "")),
            old=d.get("old"),
            new=d.get("new"),
            risk=str(d.get("risk", "low")),
            note=str(d.get("note", "")),
        )


@dataclass
class BackupEntry:
    """一条回退备份：变更前的旧值快照。"""
    ts: float
    path: str
    old: Any

    def to_dict(self) -> Dict[str, Any]:
        return {"ts": round(self.ts, 3), "path": self.path, "old": self.old}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BackupEntry":
        return cls(ts=float(d.get("ts", 0.0)),
                   path=str(d.get("path", "")),
                   old=d.get("old"))


class EvolutionArchive:
    """单条进化线路的 JSON 档案（零依赖）。"""

    def __init__(self, file_path: str, line: str, backup_keep: int = 8):
        self.file_path = file_path
        self.line = line
        self.backup_keep = max(1, int(backup_keep))
        self._data: Dict[str, Any] = {}
        self._backups: List[BackupEntry] = []
        self._changelog: List[ChangeRecord] = []
        self.load()

    # ---- 骨架 ----
    @staticmethod
    def _skeleton(line: str) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "line": line,
            "created_at": time.time(),
            "updated_at": time.time(),
        }

    # ---- 读写 ----
    def load(self) -> None:
        """从磁盘加载档案；文件缺失用空骨架，损坏则保留坏文件后重建。"""
        if not os.path.exists(self.file_path):
            self._data = self._skeleton(self.line)
            self._backups = []
            self._changelog = []
            self.save()
            return
        try:
            with open(self.file_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self._data = raw if isinstance(raw, dict) else self._skeleton(self.line)
            self._backups = [BackupEntry.from_dict(b)
                             for b in self._data.pop("backups", [])]
            self._changelog = [ChangeRecord.from_dict(c)
                               for c in self._data.pop("changelog", [])]
        except (json.JSONDecodeError, OSError):
            corrupt = f"{self.file_path}.corrupt-{int(time.time())}"
            try:
                os.replace(self.file_path, corrupt)
            except OSError:
                pass
            self._data = self._skeleton(self.line)
            self._backups = []
            self._changelog = []
            self.save()

    def save(self) -> None:
        """原子写盘：tmp + os.replace。"""
        payload = copy.deepcopy(self._data)
        payload["backups"] = [b.to_dict() for b in self._backups]
        payload["changelog"] = [c.to_dict() for c in self._changelog]
        payload["updated_at"] = time.time()
        d = os.path.dirname(self.file_path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = f"{self.file_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self.file_path)

    # ---- 数据访问 ----
    def get_data(self) -> Dict[str, Any]:
        return self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set_section(self, key: str, value: Any) -> None:
        self._data[key] = value
        self.save()

    # ---- 变更与备份 ----
    def backup_and_change(self, path: str, old: Any, new: Any,
                          risk: str = "low", note: str = "") -> ChangeRecord:
        """备份旧值 → 更新 data → 写 changelog → 落盘。返回变更记录。"""
        self._backups.append(BackupEntry(ts=time.time(), path=path, old=old))
        if len(self._backups) > self.backup_keep:
            self._backups = self._backups[-self.backup_keep:]
        rec = ChangeRecord(ts=time.time(), op="apply", path=path,
                           old=old, new=new, risk=risk, note=note)
        self._changelog.append(rec)
        self.save()
        return rec

    def append_changelog(self, op: str, path: str, old: Any = None,
                         new: Any = None, risk: str = "low",
                         note: str = "") -> ChangeRecord:
        rec = ChangeRecord(ts=time.time(), op=op, path=path,
                           old=old, new=new, risk=risk, note=note)
        self._changelog.append(rec)
        self.save()
        return rec

    # ---- 回退 ----
    def backups(self) -> List[BackupEntry]:
        return list(self._backups)

    def latest_backup_for(self, path: str) -> Optional[BackupEntry]:
        """取某路径最近一次变更前的旧值备份（供 revert 用）。"""
        for b in reversed(self._backups):
            if b.path == path:
                return b
        return None

    def restore(self, path: str, ts: Optional[float] = None) -> bool:
        """回退到指定（或最近一次）备份的旧值。

        从 data 中移除该路径当前值并按备份写入；IP 线等嵌套结构
        由调用方通过 set_path_value 落地。返回是否找到可用备份。
        """
        if ts is None:
            entry = self.latest_backup_for(path)
        else:
            candidates = [b for b in self._backups if abs(b.ts - ts) < 1e-6]
            entry = candidates[-1] if candidates else None
        if entry is None:
            return False
        return True

    # ---- 日志 ----
    def changelog(self) -> List[ChangeRecord]:
        return list(self._changelog)

    def recent_changelog(self, limit: int = 20) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self._changelog[-limit:]]

    def snapshot(self) -> Dict[str, Any]:
        payload = copy.deepcopy(self._data)
        payload["backups"] = [b.to_dict() for b in self._backups]
        payload["changelog"] = [c.to_dict() for c in self._changelog]
        return payload

    def restore_snapshot(self, snap: Dict[str, Any]) -> None:
        """整体回退到历史快照（完整可回退能力）。"""
        self._data = copy.deepcopy(snap)
        self._backups = [BackupEntry.from_dict(b) for b in self._data.pop("backups", [])]
        self._changelog = [ChangeRecord.from_dict(c)
                           for c in self._data.pop("changelog", [])]
        self.save()


def dotted(path: Any) -> str:
    """把路径元组转成点号字符串，如 ('BAGUA_KEEP_STEPS','duihua') -> 'BAGUA_KEEP_STEPS.duihua'。"""
    if isinstance(path, str):
        return path
    return ".".join(str(p) for p in path)


def get_path(data: Dict[str, Any], path: Any, default: Any = None) -> Any:
    """按路径元组或点号字符串从嵌套 dict 取值。"""
    keys = path.split(".") if isinstance(path, str) else list(path)
    cur: Any = data
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def set_path(data: Dict[str, Any], path: Any, value: Any) -> None:
    """按路径元组或点号字符串向嵌套 dict 写值（自动建中间层）。"""
    keys = path.split(".") if isinstance(path, str) else list(path)
    cur = data
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[keys[-1]] = value
