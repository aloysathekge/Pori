"""Read-only run replay: the event-log collector coalescing + the endpoint."""

from datetime import datetime, timezone

import pytest
from pori.observability import PoriEvent

from pori_cloud.event_log import MAX_EVENTS, EventLogCollector
from pori_cloud.models import RunEventLog

pytestmark = pytest.mark.asyncio

# The test client authenticates as TEST_USER_ID and auto-resolves to that
# user's personal org (tenancy.ensure_personal_organization).
CLIENT_ORG = "user:test-user"
CLIENT_USER = "test-user"


def _ev(t, step=0, **payload):
    return PoriEvent(type=t, payload=payload, step=step)


class TestCollector:
    def test_coalesces_consecutive_text_deltas(self):
        c = EventLogCollector()
        for piece in ["Hel", "lo ", "world"]:
            c.record(_ev("text_delta", text=piece))
        events = c.finalize()
        assert len(events) == 1
        assert events[0]["type"] == "text"
        assert events[0]["payload"]["text"] == "Hello world"

    def test_preserves_interleaving_with_structural_events(self):
        c = EventLogCollector()
        c.record(_ev("thinking_delta", text="let me "))
        c.record(_ev("thinking_delta", text="search"))
        c.record(_ev("tool_call_start", name="web_search"))
        c.record(_ev("tool_call_end", name="web_search", success=True))
        c.record(_ev("text_delta", text="The answer is 42"))
        events = c.finalize()
        types = [e["type"] for e in events]
        assert types == [
            "thinking",
            "tool_call_start",
            "tool_call_end",
            "text",
        ]
        assert events[0]["payload"]["text"] == "let me search"
        assert events[1]["payload"]["name"] == "web_search"
        assert events[3]["payload"]["text"] == "The answer is 42"

    def test_switching_delta_type_flushes(self):
        c = EventLogCollector()
        c.record(_ev("text_delta", text="a"))
        c.record(_ev("thinking_delta", text="b"))
        c.record(_ev("text_delta", text="c"))
        events = c.finalize()
        assert [e["type"] for e in events] == ["text", "thinking", "text"]

    def test_empty_deltas_dropped(self):
        c = EventLogCollector()
        c.record(_ev("text_delta", text=""))
        assert c.finalize() == []

    def test_caps_runaway_logs(self):
        c = EventLogCollector()
        for i in range(MAX_EVENTS + 50):
            c.record(_ev("tool_call_start", name=f"t{i}"))
        events = c.finalize()
        assert len(events) <= MAX_EVENTS + 1  # + the truncation marker
        assert events[-1]["type"] == "truncated"


class TestEndpoint:
    async def test_get_events_returns_log(self, client, db_session_maker):
        async with db_session_maker() as session:
            session.add(
                RunEventLog(
                    run_id="run-xyz",
                    organization_id=CLIENT_ORG,
                    user_id=CLIENT_USER,
                    conversation_id="conv-1",
                    events=[{"type": "text", "payload": {"text": "hi"}, "step": 1}],
                    event_count=1,
                )
            )
            await session.commit()

        resp = await client.get("/v1/runs/run-xyz/events")
        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == "run-xyz"
        assert body["event_count"] == 1
        assert body["events"][0]["payload"]["text"] == "hi"

    async def test_missing_log_is_404(self, client):
        resp = await client.get("/v1/runs/nope/events")
        assert resp.status_code == 404

    async def test_cross_org_log_is_404(self, client, db_session_maker):
        async with db_session_maker() as session:
            session.add(
                RunEventLog(
                    run_id="run-other",
                    organization_id="some-other-org",
                    user_id="mallory",
                    conversation_id=None,
                    events=[],
                    event_count=0,
                )
            )
            await session.commit()
        resp = await client.get("/v1/runs/run-other/events")
        assert resp.status_code == 404
