"""Portable session lifecycle, lineage, search, export, and delete contracts."""

from __future__ import annotations

import json
import sqlite3
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_session_key(
    organization_id: str, user_id: str, agent_id: Optional[str] = None
) -> str:
    """Deterministic **session lane** key for a (org, user, agent) tuple (GW-2).

    Pori splits two ideas that were previously conflated:

    - a *session_key* — the stable lane that survives ``/new`` and ``/resume``
      (this function), and which the duplicate-run guard (GW-3) keys on;
    - a *session_id* — one instance within that lane (``SessionRecord.id``);
      ``/resume`` / ``/branch`` swap the session_id under a fixed session_key.
    """
    return f"{organization_id}:{user_id}:{agent_id or 'default'}"


class SessionMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str
    role: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)


class SessionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    organization_id: str
    user_id: str
    agent_id: Optional[str] = None
    title: Optional[str] = None
    parent_session_id: Optional[str] = None
    branched_from_message_id: Optional[str] = None
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)


class SessionSearchHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    message_id: str
    role: str
    content: str
    score: float = Field(ge=0.0, le=1.0)
    created_at: datetime


class SessionExport(BaseModel):
    model_config = ConfigDict(frozen=True)

    session: SessionRecord
    messages: tuple[SessionMessage, ...]
    exported_at: datetime = Field(default_factory=_utc_now)


class SessionRepository(ABC):
    @abstractmethod
    def create(self, session: SessionRecord) -> SessionRecord: ...

    @abstractmethod
    def get(
        self, organization_id: str, user_id: str, session_id: str
    ) -> Optional[SessionRecord]: ...

    @abstractmethod
    def add_message(self, message: SessionMessage) -> SessionMessage: ...

    @abstractmethod
    def search(
        self, organization_id: str, user_id: str, query: str, limit: int = 10
    ) -> List[SessionSearchHit]: ...

    @abstractmethod
    def export(
        self, organization_id: str, user_id: str, session_id: str
    ) -> SessionExport: ...

    @abstractmethod
    def delete(self, organization_id: str, user_id: str, session_id: str) -> bool: ...

    @abstractmethod
    def branch(
        self,
        organization_id: str,
        user_id: str,
        session_id: str,
        *,
        through_message_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> SessionRecord: ...


class SQLiteSessionRepository(SessionRepository):
    """Small local reference implementation with scope-first queries."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    organization_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    agent_id TEXT,
                    title TEXT,
                    parent_session_id TEXT,
                    branched_from_message_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_sessions_scope
                    ON sessions (organization_id, user_id, updated_at);
                CREATE TABLE IF NOT EXISTS session_messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_session_messages_session
                    ON session_messages (session_id, created_at);
                """
            )

    @staticmethod
    def _session(row: sqlite3.Row) -> SessionRecord:
        return SessionRecord.model_validate(dict(row))

    @staticmethod
    def _message(row: sqlite3.Row) -> SessionMessage:
        data = dict(row)
        data["metadata"] = json.loads(data["metadata"] or "{}")
        return SessionMessage.model_validate(data)

    def create(self, session: SessionRecord) -> SessionRecord:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session.id,
                    session.organization_id,
                    session.user_id,
                    session.agent_id,
                    session.title,
                    session.parent_session_id,
                    session.branched_from_message_id,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                ),
            )
        return session

    def get(
        self, organization_id: str, user_id: str, session_id: str
    ) -> Optional[SessionRecord]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE id=? AND organization_id=? AND user_id=?",
                (session_id, organization_id, user_id),
            ).fetchone()
        return self._session(row) if row else None

    def add_message(self, message: SessionMessage) -> SessionMessage:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO session_messages VALUES (?, ?, ?, ?, ?, ?)",
                (
                    message.id,
                    message.session_id,
                    message.role,
                    message.content,
                    json.dumps(message.metadata, sort_keys=True, default=str),
                    message.created_at.isoformat(),
                ),
            )
            connection.execute(
                "UPDATE sessions SET updated_at=? WHERE id=?",
                (_utc_now().isoformat(), message.session_id),
            )
        return message

    def _messages(self, session_id: str) -> List[SessionMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM session_messages WHERE session_id=? ORDER BY created_at, id",
                (session_id,),
            ).fetchall()
        return [self._message(row) for row in rows]

    def search(
        self, organization_id: str, user_id: str, query: str, limit: int = 10
    ) -> List[SessionSearchHit]:
        terms = {term for term in query.lower().split() if term}
        if not terms:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT m.* FROM session_messages m
                JOIN sessions s ON s.id=m.session_id
                WHERE s.organization_id=? AND s.user_id=? AND lower(m.content) LIKE ?
                ORDER BY m.created_at DESC
                LIMIT ?
                """,
                (organization_id, user_id, f"%{query.lower()}%", max(limit * 3, limit)),
            ).fetchall()
        hits = []
        for row in rows:
            words = set(str(row["content"]).lower().split())
            score = len(terms.intersection(words)) / len(terms)
            hits.append(
                SessionSearchHit(
                    session_id=row["session_id"],
                    message_id=row["id"],
                    role=row["role"],
                    content=row["content"],
                    score=score,
                    created_at=row["created_at"],
                )
            )
        return sorted(hits, key=lambda hit: (hit.score, hit.created_at), reverse=True)[
            :limit
        ]

    def export(
        self, organization_id: str, user_id: str, session_id: str
    ) -> SessionExport:
        session = self.get(organization_id, user_id, session_id)
        if session is None:
            raise KeyError("Session not found")
        return SessionExport(
            session=session, messages=tuple(self._messages(session_id))
        )

    def delete(self, organization_id: str, user_id: str, session_id: str) -> bool:
        if self.get(organization_id, user_id, session_id) is None:
            return False
        with self._connect() as connection:
            connection.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        return True

    def branch(
        self,
        organization_id: str,
        user_id: str,
        session_id: str,
        *,
        through_message_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> SessionRecord:
        parent = self.get(organization_id, user_id, session_id)
        if parent is None:
            raise KeyError("Session not found")
        messages = self._messages(session_id)
        if through_message_id:
            ids = [message.id for message in messages]
            if through_message_id not in ids:
                raise KeyError("Message not found")
            messages = messages[: ids.index(through_message_id) + 1]
        child = self.create(
            SessionRecord(
                organization_id=organization_id,
                user_id=user_id,
                agent_id=parent.agent_id,
                title=title or parent.title,
                parent_session_id=parent.id,
                branched_from_message_id=through_message_id,
            )
        )
        for message in messages:
            self.add_message(
                SessionMessage(
                    session_id=child.id,
                    role=message.role,
                    content=message.content,
                    metadata={**message.metadata, "copied_from_message_id": message.id},
                    created_at=message.created_at,
                )
            )
        return child


__all__ = [
    "SQLiteSessionRepository",
    "SessionExport",
    "SessionMessage",
    "SessionRecord",
    "SessionRepository",
    "SessionSearchHit",
    "build_session_key",
]
