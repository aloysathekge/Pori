"""Telegram gateway slice: pairing, inbound→run, delivery, chunking."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import select

from pori_cloud.gateway.base import BasePlatformAdapter
from pori_cloud.gateway.delivery import DeliveryRouter
from pori_cloud.gateway.service import (
    PAIRED_REPLY,
    PAIRING_HELP,
    collect_finished,
    handle_message,
)
from pori_cloud.models import Conversation, GatewayLink, GatewayPairingCode, Run

pytestmark = pytest.mark.asyncio


class FakeAdapter(BasePlatformAdapter):
    name = "telegram"
    max_message_chars = 40

    def __init__(self):
        self.sent: list[tuple[str, str]] = []

    async def send(self, chat_id: str, text: str) -> None:
        # Record whole messages; chunk() behavior is covered directly in
        # TestChunking so assertions here stay readable.
        self.sent.append((chat_id, text))


class TestChunking:
    def test_short_text_single_chunk(self):
        adapter = FakeAdapter()
        assert adapter.chunk("hello") == ["hello"]

    def test_long_text_splits_on_newlines(self):
        adapter = FakeAdapter()
        text = "line one is here\nline two is here\nline three is quite long"
        chunks = adapter.chunk(text)
        assert all(len(c) <= 40 for c in chunks)
        assert "".join(chunks).replace("\n", "") == text.replace("\n", "")

    def test_unbreakable_text_hard_splits(self):
        adapter = FakeAdapter()
        chunks = adapter.chunk("x" * 100)
        assert [len(c) for c in chunks] == [40, 40, 20]


async def _seed_pairing(db_session_maker, code="AB12CD34", expired=False):
    async with db_session_maker() as session:
        session.add(
            GatewayPairingCode(
                code=code,
                organization_id="org-1",
                user_id="alice",
                platform="telegram",
                expires_at=datetime.now(timezone.utc)
                + timedelta(minutes=-5 if expired else 5),
            )
        )
        await session.commit()


class TestPairing:
    async def test_unpaired_chat_gets_help(self, db_session_maker, monkeypatch):
        monkeypatch.setattr(
            "pori_cloud.gateway.service.async_session", db_session_maker
        )
        adapter = FakeAdapter()
        run_id = await handle_message(adapter, "chat-1", "Alice", "hello there")
        assert run_id is None
        assert adapter.sent[0][1].startswith(PAIRING_HELP[:20])

    async def test_valid_code_pairs_and_creates_conversation(
        self, db_session_maker, monkeypatch
    ):
        monkeypatch.setattr(
            "pori_cloud.gateway.service.async_session", db_session_maker
        )
        await _seed_pairing(db_session_maker)
        adapter = FakeAdapter()

        run_id = await handle_message(adapter, "chat-1", "Alice", "ab12cd34")

        assert run_id is None
        assert adapter.sent[-1][1] == PAIRED_REPLY
        async with db_session_maker() as session:
            link = (await session.execute(select(GatewayLink))).scalars().first()
            assert link is not None
            assert link.user_id == "alice"
            assert link.chat_id == "chat-1"
            conversation = await session.get(Conversation, link.conversation_id)
            assert conversation is not None
            assert conversation.user_id == "alice"
            # One-time use: the code is gone
            assert await session.get(GatewayPairingCode, "AB12CD34") is None

    async def test_expired_code_rejected(self, db_session_maker, monkeypatch):
        monkeypatch.setattr(
            "pori_cloud.gateway.service.async_session", db_session_maker
        )
        await _seed_pairing(db_session_maker, expired=True)
        adapter = FakeAdapter()
        await handle_message(adapter, "chat-1", "Alice", "AB12CD34")
        assert adapter.sent[0][1].startswith(PAIRING_HELP[:20])

    async def test_start_command_carries_code(self, db_session_maker, monkeypatch):
        monkeypatch.setattr(
            "pori_cloud.gateway.service.async_session", db_session_maker
        )
        await _seed_pairing(db_session_maker, code="ZZ99YY88")
        adapter = FakeAdapter()
        await handle_message(adapter, "chat-2", "Alice", "/start ZZ99YY88")
        assert adapter.sent[-1][1] == PAIRED_REPLY


class TestInboundRuns:
    async def _pair(self, db_session_maker):
        async with db_session_maker() as session:
            conversation = Conversation(
                organization_id="org-1", user_id="alice", title="Telegram"
            )
            session.add(conversation)
            await session.flush()
            link = GatewayLink(
                organization_id="org-1",
                user_id="alice",
                platform="telegram",
                chat_id="chat-1",
                conversation_id=conversation.id,
            )
            session.add(link)
            await session.commit()
            return link.conversation_id

    async def test_paired_message_enqueues_durable_run(
        self, db_session_maker, monkeypatch
    ):
        monkeypatch.setattr(
            "pori_cloud.gateway.service.async_session", db_session_maker
        )
        conversation_id = await self._pair(db_session_maker)
        adapter = FakeAdapter()

        run_id = await handle_message(adapter, "chat-1", "Alice", "summarize my week")

        assert run_id is not None
        async with db_session_maker() as session:
            run = await session.get(Run, run_id)
            assert run.status == "pending"
            assert run.task == "summarize my week"
            assert run.user_id == "alice"
            assert run.conversation_id == conversation_id

    async def test_finished_runs_collected_once(self, db_session_maker, monkeypatch):
        monkeypatch.setattr(
            "pori_cloud.gateway.service.async_session", db_session_maker
        )
        await self._pair(db_session_maker)
        adapter = FakeAdapter()
        run_id = await handle_message(adapter, "chat-1", "Alice", "do a thing")

        pending = {run_id: "chat-1"}
        # Still running: nothing collected
        assert await collect_finished(pending) == {}
        async with db_session_maker() as session:
            run = await session.get(Run, run_id)
            run.status = "completed"
            run.success = True
            run.final_answer = "All done: the thing."
            session.add(run)
            await session.commit()

        finished = await collect_finished(pending)
        assert finished == {run_id: ("chat-1", "All done: the thing.", True)}
        assert pending == {}  # drained
        assert await collect_finished(pending) == {}


class TestDeliveryRouter:
    async def test_delivers_to_all_user_links(self, db_session_maker):
        adapter = FakeAdapter()
        router = DeliveryRouter({"telegram": adapter})
        async with db_session_maker() as session:
            session.add_all(
                [
                    GatewayLink(
                        organization_id="org-1",
                        user_id="alice",
                        platform="telegram",
                        chat_id="chat-1",
                    ),
                    GatewayLink(
                        organization_id="org-1",
                        user_id="alice",
                        platform="telegram",
                        chat_id="chat-2",
                    ),
                    GatewayLink(
                        organization_id="org-1",
                        user_id="bob",
                        platform="telegram",
                        chat_id="chat-3",
                    ),
                ]
            )
            await session.commit()
            count = await router.deliver_to_user(
                session, "org-1", "alice", "cron result"
            )
        assert count == 2
        assert {c for c, _ in adapter.sent} == {"chat-1", "chat-2"}

    async def test_unknown_platform_is_not_fatal(self, db_session_maker):
        router = DeliveryRouter({})
        link = GatewayLink(
            organization_id="org-1",
            user_id="alice",
            platform="telegram",
            chat_id="chat-1",
        )
        assert await router.deliver_to_link(link, "hello") is False


class TestPairingRoute:
    async def test_pair_endpoint_mints_code(self, client):
        response = await client.post("/v1/gateway/pair")
        assert response.status_code == 201
        body = response.json()
        assert len(body["code"]) == 8
        assert body["platform"] == "telegram"

        links = await client.get("/v1/gateway/links")
        assert links.status_code == 200
        assert links.json() == []
