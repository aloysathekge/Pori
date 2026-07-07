"""PostgreSQL-backed MemoryStore for persistent agent memory.

Implements pori's MemoryStore protocol using the same async Postgres
database that aloy_backend already uses. Each user gets persistent memory
scoped by namespace (user_id:agent_id:session_id).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import Column, DateTime, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import Field, SQLModel

logger = logging.getLogger("aloy_backend")


class MemorySnapshot(SQLModel, table=True):
    __tablename__ = "memory_snapshots"

    namespace: str = Field(primary_key=True)
    payload: dict = Field(sa_column=Column(JSONB, nullable=False))
    updated_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False)
    )


class PostgresMemoryStore:
    """MemoryStore backed by Postgres JSONB.

    Because pori's MemoryStore protocol is synchronous (load/save),
    but our DB is async, we use a sync-compatible wrapper that runs
    the async operations. However, for the aloy_backend use case we
    also expose async helpers that the API layer calls directly.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory

    # --- Async API (used by aloy_backend directly) ---

    async def aload(self, namespace: str) -> Optional[Dict[str, Any]]:
        async with self._session_factory() as session:
            snapshot = await session.get(MemorySnapshot, namespace)
            if snapshot is None:
                return None
            return snapshot.payload

    async def asave(self, namespace: str, snapshot: Dict[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        async with self._session_factory() as session:
            existing = await session.get(MemorySnapshot, namespace)
            if existing:
                existing.payload = snapshot
                existing.updated_at = now
                session.add(existing)
            else:
                record = MemorySnapshot(
                    namespace=namespace,
                    payload=snapshot,
                    updated_at=now,
                    created_at=now,
                )
                session.add(record)
            await session.commit()

    # --- Sync API (satisfies pori's MemoryStore protocol) ---
    # These are called by AgentMemory._persist() inside the agent loop.
    # Since the agent runs in an async context, we use a nested approach.

    def load(self, namespace: str) -> Optional[Dict[str, Any]]:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an async context — store the data for lazy save.
            # Return cached snapshot if we preloaded it.
            return getattr(self, "_preloaded", {}).get(namespace)
        else:
            return asyncio.run(self.aload(namespace))

    def save(self, namespace: str, snapshot: Dict[str, Any]) -> None:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an async context — stash for later async save.
            if not hasattr(self, "_pending_saves"):
                self._pending_saves: Dict[str, Dict[str, Any]] = {}
            self._pending_saves[namespace] = snapshot
        else:
            asyncio.run(self.asave(namespace, snapshot))

    # --- Helpers for the async API layer ---

    async def preload(self, namespace: str) -> Optional[Dict[str, Any]]:
        """Preload a snapshot so sync load() can return it."""
        data = await self.aload(namespace)
        if not hasattr(self, "_preloaded"):
            self._preloaded: Dict[str, Dict[str, Any]] = {}
        if data is not None:
            self._preloaded[namespace] = data
        return data

    async def flush_pending(self) -> None:
        """Persist any snapshots queued by sync save() calls."""
        pending = getattr(self, "_pending_saves", {})
        if not pending:
            return
        for ns, snapshot in pending.items():
            await self.asave(ns, snapshot)
            logger.debug("Flushed memory snapshot for %s", ns)
        pending.clear()
