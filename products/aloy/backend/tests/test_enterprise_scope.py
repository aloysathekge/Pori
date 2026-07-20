import pytest

pytestmark = pytest.mark.asyncio


async def _create_org(client, user_id="alice", slug="enterprise-scope"):
    response = await client.post(
        "/v1/organizations",
        headers={"X-Test-User": user_id},
        json={"name": "Enterprise", "slug": slug},
    )
    assert response.status_code == 201
    return response.json()["id"]


async def test_resources_are_scoped_to_selected_organization(client):
    organization_id = await _create_org(client)
    org_headers = {
        "X-Test-User": "alice",
        "X-Pori-Organization": organization_id,
    }

    config = await client.post(
        "/v1/agent-configs",
        headers=org_headers,
        json={"name": "Org Agent"},
    )
    assert config.status_code == 201
    conversation = await client.post(
        "/v1/conversations",
        headers=org_headers,
        json={"title": "Org Conversation"},
    )
    assert conversation.status_code == 201
    memory = await client.post(
        "/v1/me/memory/knowledge",
        headers=org_headers,
        json={"content": "Organization-scoped private memory"},
    )
    assert memory.status_code == 201

    personal_headers = {"X-Test-User": "alice"}
    assert (
        await client.get("/v1/agent-configs", headers=personal_headers)
    ).json() == []
    assert (
        await client.get("/v1/conversations", headers=personal_headers)
    ).json() == []
    assert (
        await client.get("/v1/me/memory/knowledge", headers=personal_headers)
    ).json() == []


async def test_member_cannot_inspect_operator_config_or_another_users_memory(client):
    organization_id = await _create_org(client, slug="shared-enterprise")
    owner_headers = {
        "X-Test-User": "alice",
        "X-Pori-Organization": organization_id,
    }
    await client.post(
        f"/v1/organizations/{organization_id}/members",
        headers=owner_headers,
        json={"user_id": "bob", "role": "member"},
    )
    await client.post(
        "/v1/agent-configs",
        headers=owner_headers,
        json={"name": "Shared Agent"},
    )
    await client.post(
        "/v1/conversations",
        headers=owner_headers,
        json={"title": "Shared Conversation"},
    )
    await client.post(
        "/v1/me/memory/knowledge",
        headers=owner_headers,
        json={"content": "Alice memory"},
    )

    member_headers = {
        "X-Test-User": "bob",
        "X-Pori-Organization": organization_id,
    }
    configs = await client.get("/v1/agent-configs", headers=member_headers)
    conversations = await client.get("/v1/conversations", headers=member_headers)
    memory = await client.get("/v1/me/memory/knowledge", headers=member_headers)

    assert configs.status_code == 403
    for path in (
        "/v1/agent-configs/info/models",
        "/v1/agent-configs/info/tools",
        "/v1/agent-configs/info/setup",
    ):
        assert (await client.get(path, headers=member_headers)).status_code == 403
    # Life is personal even inside a shared organization. Shared resources do
    # not turn Alice's personal Conversation into Bob's chat history.
    assert conversations.json() == []
    assert memory.json() == []


async def test_member_can_start_conversation_but_cannot_select_agent_config(client):
    organization_id = await _create_org(client, slug="operator-agent-selection")
    owner_headers = {
        "X-Test-User": "alice",
        "X-Pori-Organization": organization_id,
    }
    await client.post(
        f"/v1/organizations/{organization_id}/members",
        headers=owner_headers,
        json={"user_id": "bob", "role": "member"},
    )
    config = await client.post(
        "/v1/agent-configs",
        headers=owner_headers,
        json={"name": "Operator managed"},
    )
    assert config.status_code == 201

    member_headers = {
        "X-Test-User": "bob",
        "X-Pori-Organization": organization_id,
    }
    ordinary = await client.post(
        "/v1/conversations",
        headers=member_headers,
        json={"title": "Ordinary Aloy conversation"},
    )
    assert ordinary.status_code == 201
    assert ordinary.json()["agent_config_id"] is None

    selected = await client.post(
        "/v1/conversations",
        headers=member_headers,
        json={
            "title": "Attempted custom runtime",
            "agent_config_id": config.json()["id"],
        },
    )
    assert selected.status_code == 403
    assert "operator" in selected.json()["detail"]


async def test_policy_limits_runs_and_viewer_cannot_create_them(client):
    organization_id = await _create_org(client, slug="policy-enterprise")
    owner_headers = {
        "X-Test-User": "alice",
        "X-Pori-Organization": organization_id,
    }
    await client.patch(
        f"/v1/organizations/{organization_id}/policy",
        headers=owner_headers,
        json={
            "policy": {
                "max_steps_per_run": 4,
                "max_tool_calls_per_run": 6,
                "max_tokens_per_run": 2_000,
                "max_cost_usd_per_run": 0.5,
                "max_attempts": 2,
            }
        },
    )
    queued = await client.post(
        "/v1/runs",
        headers=owner_headers,
        json={"task": "bounded", "max_steps": 99},
    )
    assert queued.status_code == 202
    assert queued.json()["max_steps"] == 4
    assert queued.json()["max_tool_calls"] == 6
    assert queued.json()["max_tokens"] == 2_000
    assert queued.json()["max_cost_usd"] == 0.5
    assert queued.json()["max_attempts"] == 2

    await client.post(
        f"/v1/organizations/{organization_id}/members",
        headers=owner_headers,
        json={"user_id": "viewer", "role": "viewer"},
    )
    denied = await client.post(
        "/v1/runs",
        headers={
            "X-Test-User": "viewer",
            "X-Pori-Organization": organization_id,
        },
        json={"task": "not allowed", "max_steps": 1},
    )
    assert denied.status_code == 403


async def test_conversation_messages_use_durable_worker_by_default(client):
    organization_id = await _create_org(client, slug="queued-conversation")
    headers = {
        "X-Test-User": "alice",
        "X-Pori-Organization": organization_id,
    }
    conversation = await client.post(
        "/v1/conversations",
        headers=headers,
        json={"title": "Queued"},
    )
    conversation_id = conversation.json()["id"]

    queued = await client.post(
        f"/v1/conversations/{conversation_id}/messages",
        headers=headers,
        json={"content": "Run outside the API process", "max_steps": 3},
    )

    assert queued.status_code == 202
    assert queued.json()["execution"] == "durable-worker"
    runs = await client.get("/v1/runs", headers=headers)
    assert [run["id"] for run in runs.json()] == [queued.json()["run_id"]]
    assert runs.json()[0]["status"] == "pending"
    assert runs.json()[0]["max_tool_calls"] == 100
