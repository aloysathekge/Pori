"""Managed async session repository over Cloud's SQLModel database."""

from __future__ import annotations

import re
from typing import Optional

from sqlalchemy import func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from pori import SessionExport, SessionMessage, SessionRecord, SessionSearchHit

from .models import Conversation, Message


class CloudSessionRepository:
    """Scope-first Postgres/SQLite repository used by Cloud APIs."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        organization_id: str,
        user_id: str,
        allow_shared_search: bool = False,
    ):
        self.session = session
        self.organization_id = organization_id
        self.user_id = user_id
        self.allow_shared_search = allow_shared_search

    @staticmethod
    def _record(conversation: Conversation) -> SessionRecord:
        return SessionRecord(
            id=conversation.id,
            organization_id=conversation.organization_id,
            user_id=conversation.user_id,
            agent_id=conversation.agent_config_id,
            title=conversation.title,
            parent_session_id=conversation.parent_conversation_id,
            branched_from_message_id=conversation.branched_from_message_id,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )

    @staticmethod
    def _message(message: Message) -> SessionMessage:
        return SessionMessage(
            id=message.id,
            session_id=message.conversation_id,
            role=message.role,
            content=message.content,
            metadata=message.metadata_ or {},
            created_at=message.created_at,
        )

    async def get(self, session_id: str) -> Optional[SessionRecord]:
        result = await self.session.execute(
            select(Conversation).where(
                Conversation.id == session_id,
                Conversation.organization_id == self.organization_id,
            )
        )
        conversation = result.scalars().first()
        return self._record(conversation) if conversation else None

    async def messages(self, session_id: str) -> list[SessionMessage]:
        if await self.get(session_id) is None:
            return []
        result = await self.session.execute(
            select(Message)
            .where(Message.conversation_id == session_id)
            .order_by(col(Message.created_at), col(Message.id))
        )
        return [self._message(message) for message in result.scalars().all()]

    async def search(self, query: str, limit: int = 10) -> list[SessionSearchHit]:
        statement = (
            select(Message, Conversation)
            .join(Conversation, col(Conversation.id) == col(Message.conversation_id))
            .where(
                Conversation.organization_id == self.organization_id,
                func.lower(Message.content).contains(query.lower()),
            )
            .order_by(col(Message.created_at).desc())
            .limit(limit)
        )
        if not self.allow_shared_search:
            statement = statement.where(Conversation.user_id == self.user_id)
        result = await self.session.execute(statement)
        terms = set(query.lower().split())
        hits = []
        for message, conversation in result.all():
            words = set(message.content.lower().split())
            hits.append(
                SessionSearchHit(
                    session_id=conversation.id,
                    message_id=message.id,
                    role=message.role,
                    content=message.content,
                    score=len(terms.intersection(words)) / max(1, len(terms)),
                    created_at=message.created_at,
                )
            )
        return sorted(hits, key=lambda hit: (hit.score, hit.created_at), reverse=True)

    async def search_event(
        self,
        event_id: str,
        query: str,
        *,
        limit: int = 10,
        roles: list[str] | None = None,
        candidate_limit: int = 500,
    ) -> list[SessionSearchHit]:
        """Page-fault relevant messages from exactly one owned Event.

        Tenant and Event predicates are applied before candidate ranking. The
        bounded SQL candidate set prevents an old Event from being loaded into
        every Run merely to make occasional historical recall possible.
        """
        phrase = query.strip().lower()
        terms = list(dict.fromkeys(re.findall(r"[\w'-]{2,}", phrase)))[:12]
        if not terms:
            return []
        predicates = [func.lower(Message.content).contains(term) for term in terms]
        statement = (
            select(Message, Conversation)
            .join(Conversation, col(Conversation.id) == col(Message.conversation_id))
            .where(
                Conversation.organization_id == self.organization_id,
                Conversation.user_id == self.user_id,
                Conversation.event_id == event_id,
                or_(*predicates),
            )
            .order_by(col(Message.created_at).desc(), col(Message.id).desc())
            .limit(max(limit, min(int(candidate_limit), 2_000)))
        )
        if roles:
            statement = statement.where(col(Message.role).in_(roles))
        rows = (await self.session.execute(statement)).all()
        hits: list[SessionSearchHit] = []
        for message, conversation in rows:
            content = message.content.lower()
            matched = sum(1 for term in terms if term in content)
            coverage = matched / len(terms)
            phrase_bonus = 0.25 if phrase in content else 0.0
            hits.append(
                SessionSearchHit(
                    session_id=conversation.id,
                    message_id=message.id,
                    role=message.role,
                    content=message.content,
                    score=min(1.0, coverage + phrase_bonus),
                    created_at=message.created_at,
                )
            )
        return sorted(
            hits,
            key=lambda hit: (hit.score, hit.created_at, hit.message_id),
            reverse=True,
        )[:limit]

    async def export(self, session_id: str) -> SessionExport:
        record = await self.get(session_id)
        if record is None:
            raise KeyError("Session not found")
        return SessionExport(
            session=record,
            messages=tuple(await self.messages(session_id)),
        )

    async def branch(
        self,
        session_id: str,
        *,
        through_message_id: str | None = None,
        title: str | None = None,
    ) -> SessionRecord:
        parent = await self.get(session_id)
        if parent is None:
            raise KeyError("Session not found")
        parent_row = await self.session.get(Conversation, session_id)
        if parent_row is None:
            raise KeyError("Session not found")
        messages = await self.messages(session_id)
        if through_message_id:
            ids = [message.id for message in messages]
            if through_message_id not in ids:
                raise KeyError("Message not found")
            messages = messages[: ids.index(through_message_id) + 1]
        child = Conversation(
            organization_id=self.organization_id,
            user_id=self.user_id,
            event_id=parent_row.event_id,
            title=title or parent.title,
            agent_config_id=parent.agent_id,
            parent_conversation_id=parent.id,
            branched_from_message_id=through_message_id,
        )
        self.session.add(child)
        for message in messages:
            self.session.add(
                Message(
                    conversation_id=child.id,
                    role=message.role,
                    content=message.content,
                    metadata_={
                        **message.metadata,
                        "copied_from_message_id": message.id,
                    },
                    created_at=message.created_at,
                )
            )
        await self.session.commit()
        return self._record(child)

    async def delete(self, session_id: str) -> bool:
        conversation = await self.session.get(Conversation, session_id)
        if conversation is None or conversation.organization_id != self.organization_id:
            return False
        result = await self.session.execute(
            select(Message).where(Message.conversation_id == session_id)
        )
        for row in result.scalars().all():
            await self.session.delete(row)
        await self.session.delete(conversation)
        await self.session.commit()
        return True


__all__ = ["CloudSessionRepository"]
