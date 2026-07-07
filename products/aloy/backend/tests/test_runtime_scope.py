import pytest

from aloy_backend.runtime import authenticated_run_context


def test_authenticated_context_uses_server_owned_personal_tenant():
    context = authenticated_run_context(
        user_id="alice",
        run_id="run-1",
        session_id="session-1",
        agent_id="agent-1",
        max_steps=12,
    )

    assert context.organization_id == "user:alice"
    assert context.user_id == "alice"
    assert context.budget.max_steps == 12
    assert context.isolation_profile == "worker-process"


@pytest.mark.asyncio
async def test_run_api_does_not_leak_between_authenticated_users(client):
    created = await client.post(
        "/v1/runs",
        headers={"X-Test-User": "alice"},
        json={"task": "private task", "max_steps": 2},
    )
    assert created.status_code == 202
    run = created.json()
    assert run["organization_id"] == "user:alice"

    bob_list = await client.get("/v1/runs", headers={"X-Test-User": "bob"})
    assert bob_list.status_code == 200
    assert bob_list.json() == []

    bob_get = await client.get(f"/v1/runs/{run['id']}", headers={"X-Test-User": "bob"})
    assert bob_get.status_code == 404
