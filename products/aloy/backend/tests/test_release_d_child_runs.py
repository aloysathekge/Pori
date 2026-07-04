import pytest

pytestmark = pytest.mark.asyncio


async def test_child_run_queueing_is_bounded_and_idempotent(client):
    parent = await client.post("/v1/runs", json={"task": "parent", "max_steps": 3})
    assert parent.status_code == 202
    parent_body = parent.json()
    assert parent_body["root_run_id"] == parent_body["id"]

    first = await client.post(
        f"/v1/runs/{parent_body['id']}/children",
        json={
            "task": "child task",
            "agent_id": "researcher",
            "max_steps": 2,
            "idempotency_key": "child-1",
        },
    )
    assert first.status_code == 202
    child = first.json()
    assert child["parent_run_id"] == parent_body["id"]
    assert child["root_run_id"] == parent_body["id"]
    assert child["child_depth"] == 1
    assert child["idempotency_key"] == "child-1"

    repeated = await client.post(
        f"/v1/runs/{parent_body['id']}/children",
        json={
            "task": "child task changed but idempotent",
            "agent_id": "researcher",
            "idempotency_key": "child-1",
        },
    )
    assert repeated.status_code == 202
    assert repeated.json()["id"] == child["id"]


async def test_child_run_depth_policy_is_enforced(client):
    org = await client.post(
        "/v1/organizations",
        headers={"X-Test-User": "alice"},
        json={"name": "Depth", "slug": "depth-release-d"},
    )
    organization_id = org.json()["id"]
    headers = {"X-Test-User": "alice", "X-Pori-Organization": organization_id}
    await client.patch(
        f"/v1/organizations/{organization_id}/policy",
        headers=headers,
        json={"policy": {"max_child_depth": 0}},
    )
    parent = await client.post("/v1/runs", headers=headers, json={"task": "parent"})
    denied = await client.post(
        f"/v1/runs/{parent.json()['id']}/children",
        headers=headers,
        json={"task": "child"},
    )
    assert denied.status_code == 409
