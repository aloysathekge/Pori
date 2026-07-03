"""Gateway clarify bridge — emit a request, resolve on the user's answer."""

import asyncio
import threading

import pytest

from pori.clarify import ClarificationRequest, ClarifyBridge

pytestmark = [pytest.mark.unit]


def test_request_to_event_shape():
    event = ClarificationRequest(id="i", question="q?", options=("A", "B")).to_event()
    assert event == {
        "type": "clarification_request",
        "id": "i",
        "question": "q?",
        "options": ["A", "B"],
    }


async def test_ask_emits_then_resolves_on_submit():
    emitted = []
    bridge = ClarifyBridge(emit=emitted.append, id_factory=lambda: "cid-1")

    task = asyncio.ensure_future(bridge.ask("Which engine?", ["Postgres", "MySQL"]))
    await asyncio.sleep(0)  # let ask emit and start awaiting

    assert emitted[0].id == "cid-1"
    assert emitted[0].question == "Which engine?"
    assert emitted[0].options == ("Postgres", "MySQL")

    assert bridge.submit_answer("cid-1", "Postgres") is True
    assert await task == "Postgres"
    # resolved + cleared: a repeat answer (double-tap / retry) is a no-op
    assert bridge.submit_answer("cid-1", "MySQL") is False


def test_submit_unknown_id_is_false():
    assert ClarifyBridge(emit=lambda r: None).submit_answer("nope", "x") is False


def test_as_sync_handler_bridges_from_a_worker_thread():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    seen = {}

    def emit(request):
        seen["id"] = request.id
        # the client "taps a button": resolve the pending request on the loop
        loop.call_soon_threadsafe(bridge.submit_answer, request.id, "SQLite")

    bridge = ClarifyBridge(emit=emit, id_factory=lambda: "cid")
    try:
        handler = bridge.as_sync_handler(loop)
        # called from this (worker) thread; blocks until the answer arrives
        assert handler("engine?", ["Postgres", "SQLite"]) == "SQLite"
        assert seen["id"] == "cid"
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()
