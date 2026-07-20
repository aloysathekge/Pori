"""On-demand, tenant-safe page faults into older Event conversations."""

from __future__ import annotations

import asyncio
from typing import Any

from pori import RunContext

from ..config import settings
from ..database import async_session
from ..session_repository import CloudSessionRepository

EVENT_HISTORY_SEARCH_CONTEXT_KEY = "event_history_search"


class EventHistorySearchHandler:
    """Resolve historical evidence without eagerly hydrating Event transcripts."""

    def __init__(
        self,
        *,
        run_context: RunContext,
        session_factory: Any = async_session,
        owner_loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._run_context = run_context
        self._session_factory = session_factory
        self._owner_loop = owner_loop

    async def _search(
        self, *, query: str, limit: int, roles: list[str] | None
    ) -> list[dict[str, Any]]:
        event_id = self._run_context.event_id
        if not event_id:
            raise ValueError("Event history requires an Event-scoped Run")
        async with self._session_factory() as session:
            repository = CloudSessionRepository(
                session,
                organization_id=self._run_context.organization_id,
                user_id=self._run_context.user_id,
            )
            hits = await repository.search_event(
                event_id,
                query,
                limit=limit,
                roles=roles,
                candidate_limit=settings.event_history_search_max_candidates,
            )
        return [
            {
                "id": hit.message_id,
                "conversation_id": hit.session_id,
                "role": hit.role,
                "content": hit.content,
                "created_at": hit.created_at.isoformat(),
                "score": round(hit.score, 4),
            }
            for hit in hits
        ]

    async def search(
        self, *, query: str, limit: int, roles: list[str] | None = None
    ) -> list[dict[str, Any]]:
        current = asyncio.get_running_loop()
        if self._owner_loop is None or self._owner_loop is current:
            return await self._search(query=query, limit=limit, roles=roles)
        future = asyncio.run_coroutine_threadsafe(
            self._search(query=query, limit=limit, roles=roles), self._owner_loop
        )
        return await asyncio.wrap_future(future)


__all__ = ["EVENT_HISTORY_SEARCH_CONTEXT_KEY", "EventHistorySearchHandler"]
