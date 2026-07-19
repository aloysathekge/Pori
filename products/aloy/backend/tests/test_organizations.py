import pytest

pytestmark = pytest.mark.asyncio


async def test_organization_membership_rbac_and_policy(client):
    alice = {"X-Test-User": "alice"}
    created = await client.post(
        "/v1/organizations",
        headers=alice,
        json={"name": "Acme", "slug": "acme-enterprise"},
    )
    assert created.status_code == 201
    organization = created.json()
    organization_id = organization["id"]
    assert organization["role"] == "owner"

    org_headers = {
        "X-Test-User": "alice",
        "X-Pori-Organization": organization_id,
    }
    added = await client.post(
        f"/v1/organizations/{organization_id}/members",
        headers=org_headers,
        json={"user_id": "bob", "role": "viewer"},
    )
    assert added.status_code == 201
    bob_membership_id = added.json()["id"]

    bob_headers = {
        "X-Test-User": "bob",
        "X-Pori-Organization": organization_id,
    }
    visible = await client.get(
        f"/v1/organizations/{organization_id}", headers=bob_headers
    )
    assert visible.status_code == 200
    assert visible.json()["role"] == "viewer"

    agent_write = await client.post(
        "/v1/agent-configs",
        headers=bob_headers,
        json={"name": "Viewer cannot create this"},
    )
    assert agent_write.status_code == 403

    forbidden = await client.patch(
        f"/v1/organizations/{organization_id}/policy",
        headers=bob_headers,
        json={"policy": {"max_steps_per_run": 12}},
    )
    assert forbidden.status_code == 403

    updated = await client.patch(
        f"/v1/organizations/{organization_id}/policy",
        headers=org_headers,
        json={"policy": {"max_steps_per_run": 12, "max_attempts": 2}},
    )
    assert updated.status_code == 200
    assert updated.json()["policy"]["max_steps_per_run"] == 12

    promoted = await client.patch(
        f"/v1/organizations/{organization_id}/members/{bob_membership_id}",
        headers=org_headers,
        json={"role": "member"},
    )
    assert promoted.status_code == 200

    agent_write = await client.post(
        "/v1/agent-configs",
        headers=bob_headers,
        json={"name": "Member Agent"},
    )
    assert agent_write.status_code == 403

    promoted = await client.patch(
        f"/v1/organizations/{organization_id}/members/{bob_membership_id}",
        headers=org_headers,
        json={"role": "admin"},
    )
    assert promoted.status_code == 200

    agent_write = await client.post(
        "/v1/agent-configs",
        headers=bob_headers,
        json={"name": "Operator Agent"},
    )
    assert agent_write.status_code == 201


async def test_non_member_cannot_select_organization(client):
    created = await client.post(
        "/v1/organizations",
        headers={"X-Test-User": "alice"},
        json={"name": "Private", "slug": "private-enterprise"},
    )
    organization_id = created.json()["id"]

    denied = await client.get(
        f"/v1/organizations/{organization_id}",
        headers={
            "X-Test-User": "mallory",
            "X-Pori-Organization": organization_id,
        },
    )
    assert denied.status_code == 404
