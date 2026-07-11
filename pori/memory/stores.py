"""Pluggable persistence backends for agent memory.

`MemoryStore` is the protocol (namespace -> JSON snapshot), with the built-in
`InMemoryMemoryStore` and `SQLiteMemoryStore` implementations, the
`create_memory_store` factory, and the `pori.memory_stores` entry-point
factories used by third-party backends.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional, Protocol


class MemoryStore(Protocol):
    def load(self, namespace: str) -> Optional[Dict[str, Any]]: ...

    def save(self, namespace: str, snapshot: Dict[str, Any]) -> None: ...


class InMemoryMemoryStore:
    def __init__(self) -> None:
        self._db: Dict[str, Dict[str, Any]] = {}

    def load(self, namespace: str) -> Optional[Dict[str, Any]]:
        data = self._db.get(namespace)
        if data is None:
            return None
        return json.loads(json.dumps(data))

    def save(self, namespace: str, snapshot: Dict[str, Any]) -> None:
        self._db[namespace] = json.loads(json.dumps(snapshot))


class SQLiteMemoryStore:
    def __init__(self, db_path: str):
        self.db_path = str(Path(db_path).expanduser().resolve())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_snapshots (
                    namespace TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def load(self, namespace: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM memory_snapshots WHERE namespace = ?",
                (namespace,),
            ).fetchone()
            if row is None:
                return None
            return json.loads(str(row["payload"]))

    def save(self, namespace: str, snapshot: Dict[str, Any]) -> None:
        payload = json.dumps(snapshot, ensure_ascii=False)
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_snapshots (namespace, payload, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(namespace) DO UPDATE SET
                  payload = excluded.payload,
                  updated_at = excluded.updated_at
                """,
                (namespace, payload, now),
            )
            conn.commit()


def create_memory_store(
    backend: str = "memory",
    sqlite_path: Optional[str] = None,
) -> MemoryStore:
    normalized = (backend or "memory").strip().lower()
    if normalized in {"memory", "in_memory", "in-memory"}:
        return InMemoryMemoryStore()
    if normalized == "sqlite":
        path = sqlite_path or os.getenv("PORI_MEMORY_SQLITE_PATH") or ".pori/memory.db"
        return SQLiteMemoryStore(path)

    selected: Any = []
    try:
        eps = metadata.entry_points()
        if hasattr(eps, "select"):
            selected = eps.select(group="pori.memory_stores")
        else:  # pragma: no cover - legacy importlib_metadata API
            # Access .get dynamically: on modern Python EntryPoints has no .get,
            # and mypy's error code for this differs across versions.
            selected = getattr(eps, "get")("pori.memory_stores", [])
    except Exception:
        selected = []

    for ep in selected:
        if ep.name.strip().lower() != normalized:
            continue
        factory = ep.load()
        store = factory(sqlite_path=sqlite_path)
        if not hasattr(store, "load") or not hasattr(store, "save"):
            raise ValueError(
                f"Memory store plugin '{normalized}' does not implement load/save"
            )
        return store
    raise ValueError(f"Unsupported memory backend: {backend}")


def create_in_memory_store(**_: Any) -> MemoryStore:
    return InMemoryMemoryStore()


def create_sqlite_memory_store(
    *,
    sqlite_path: Optional[str] = None,
    **_: Any,
) -> MemoryStore:
    path = sqlite_path or os.getenv("PORI_MEMORY_SQLITE_PATH") or ".pori/memory.db"
    return SQLiteMemoryStore(path)
