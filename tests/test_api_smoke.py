"""API smoke test — the app imports, starts up, and serves (API repair)."""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from pori.api import create_app  # noqa: E402

pytestmark = [pytest.mark.integration]


@pytest.fixture
def client(monkeypatch):
    # A dummy key lets the lifespan build the orchestrator without a real one;
    # the health path never calls the LLM. Memory stays in-process.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setenv("PORI_MEMORY_BACKEND", "memory")
    app = create_app()
    with TestClient(app) as test_client:  # runs lifespan startup
        yield test_client, app


def test_health_endpoint_serves(client):
    test_client, _ = client
    response = test_client.get("/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    # the request-id middleware stamps every response
    assert "X-Request-ID" in response.headers


def test_lifespan_builds_shared_state(client):
    _, app = client
    assert app.state.orchestrator is not None
    assert app.state.memory_store is not None
