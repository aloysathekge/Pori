"""SSE streaming endpoint (GW-4) — /v1/tasks/stream over the PoriEvent contract."""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from pori.api import create_app  # noqa: E402
from pori.api.deps import get_orchestrator, get_request_memory  # noqa: E402
from pori.api.security import get_api_key  # noqa: E402
from pori.memory import AgentMemory  # noqa: E402
from pori.observability import RUN_END, TEXT_DELTA, PoriEvent  # noqa: E402

pytestmark = [pytest.mark.integration]


class _StubOrchestrator:
    """Emits a couple of deltas + RUN_END without touching a real LLM."""

    async def execute_task(self, task, *, on_event=None, **kwargs):
        if on_event is not None:
            on_event(PoriEvent(TEXT_DELTA, {"text": "Hello"}, step=1))
            on_event(PoriEvent(TEXT_DELTA, {"text": " world"}, step=1))
            on_event(PoriEvent(RUN_END, {"completed": True, "steps": 1}, step=1))
        return {"task_id": "t", "success": True}


def test_stream_emits_events_and_closes_on_run_end(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")  # lifespan needs a key
    app = create_app()
    app.dependency_overrides[get_orchestrator] = lambda: _StubOrchestrator()
    app.dependency_overrides[get_api_key] = lambda: "test"
    app.dependency_overrides[get_request_memory] = lambda: AgentMemory()

    with TestClient(app) as client:
        with client.stream("POST", "/v1/tasks/stream", json={"task": "hi"}) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            body = "".join(response.iter_text())

    assert "event: text_delta" in body
    assert "Hello" in body and "world" in body
    assert "event: run_end" in body  # terminal frame closed the stream
