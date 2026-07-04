"""Clarify buttons resume path: POST /v1/clarify unblocks a waiting run.

The SSE emit side is covered by test_api_sse; the bridge's emit+block+resolve
mechanism by test_clarify_bridge. This ties them at the API seam — a bridge
registered by a stream, a worker blocked in ask_user (ask_sync), and the resume
endpoint delivering the tapped answer — without the httpx ASGITransport
streaming-concurrency limitation that an in-process full stream would hit.
"""

import asyncio
import threading

import pytest

pytest.importorskip("fastapi")

import httpx  # noqa: E402

from pori.api import create_app  # noqa: E402
from pori.api.security import get_api_key  # noqa: E402
from pori.clarify import ClarifyBridge  # noqa: E402

pytestmark = [pytest.mark.integration]


async def test_submit_endpoint_resolves_a_waiting_bridge():
    app = create_app()
    app.state.clarify_bridges = set()
    app.dependency_overrides[get_api_key] = lambda: "test"

    emitted: dict = {}
    bridge = ClarifyBridge(emit=lambda req: emitted.setdefault("id", req.id))
    app.state.clarify_bridges.add(bridge)  # a stream would do this

    # A worker thread blocks in ask_user (ask_sync), exactly as the agent would
    # when it calls ask_user with options over the stream.
    result: dict = {}
    worker = threading.Thread(
        target=lambda: result.setdefault(
            "value", bridge.ask_sync("Which engine?", ["Postgres", "SQLite"])
        ),
        daemon=True,
    )
    worker.start()

    for _ in range(100):  # wait for the emit to register the pending clarification
        if emitted.get("id"):
            break
        await asyncio.sleep(0.02)
    cid = emitted["id"]

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post(f"/v1/clarify/{cid}", json={"value": "Postgres"})
        assert resp.status_code == 200 and resp.json() == {"ok": True}

    worker.join(timeout=3)
    assert result["value"] == "Postgres"  # the blocked run resumed with the answer


async def test_submit_to_unknown_clarification_is_404():
    app = create_app()
    app.state.clarify_bridges = set()
    app.dependency_overrides[get_api_key] = lambda: "test"
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/v1/clarify/nope", json={"value": "x"})
    assert resp.status_code == 404
