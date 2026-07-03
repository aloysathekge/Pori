import pytest

pytestmark = pytest.mark.asyncio


async def _organization(client, user: str, slug: str) -> str:
    response = await client.post(
        "/v1/organizations",
        headers={"X-Test-User": user},
        json={"name": slug, "slug": slug},
    )
    assert response.status_code == 201
    return response.json()["id"]


def _proposal_payload(version: str = "1") -> dict:
    return {
        "artifact_kind": "skill",
        "target": "skills/brainstorming",
        "title": "Improve brainstorming",
        "summary": "Ask clarifying questions before implementation.",
        "rationale": "Repeated build tasks need design-before-build behavior.",
        "current_version": "0",
        "proposed_version": version,
        "proposed_content": "Ask one clarifying question before implementation.",
        "eval_cases": [
            {
                "name": "asks-before-coding",
                "input": "Build a sync workflow",
                "expected": "A clarifying question first",
                "criteria": "The answer asks a clarifying question first.",
            }
        ],
    }


async def test_cloud_evolution_lifecycle_requires_eval_and_policy_review(client):
    organization_id = await _organization(client, "alice", "release-e-evolution")
    headers = {"X-Test-User": "alice", "X-Pori-Organization": organization_id}

    created = await client.post(
        "/v1/evolution",
        headers=headers,
        json=_proposal_payload(),
    )
    assert created.status_code == 201
    proposal = created.json()
    assert proposal["status"] == "proposed"

    blocked = await client.post(
        f"/v1/evolution/{proposal['id']}/approve",
        headers=headers,
    )
    assert blocked.status_code == 409

    evaluated = await client.post(
        f"/v1/evolution/{proposal['id']}/evals",
        headers=headers,
        json={
            "results": [
                {
                    "case_name": "asks-before-coding",
                    "passed": True,
                    "score": 1.0,
                    "reason": "Passed",
                }
            ]
        },
    )
    assert evaluated.status_code == 200
    assert evaluated.json()["status"] == "evaluated"

    approved = await client.post(
        f"/v1/evolution/{proposal['id']}/approve",
        headers=headers,
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"
    assert approved.json()["approved_by"] == "alice"

    activated = await client.post(
        f"/v1/evolution/{proposal['id']}/activate",
        headers=headers,
    )
    assert activated.status_code == 201
    assert activated.json()["target"] == "skills/brainstorming"
    assert activated.json()["version"] == "1"

    active = await client.get(
        "/v1/evolution/active/skills/brainstorming",
        headers=headers,
    )
    assert active.status_code == 200
    assert active.json()["proposal_id"] == proposal["id"]


async def test_cloud_evolution_rollback_restores_previous_activation(client):
    organization_id = await _organization(client, "alice", "release-e-rollback")
    headers = {"X-Test-User": "alice", "X-Pori-Organization": organization_id}
    proposal_ids = []

    for version in ("1", "2"):
        created = await client.post(
            "/v1/evolution",
            headers=headers,
            json=_proposal_payload(version),
        )
        proposal_id = created.json()["id"]
        proposal_ids.append(proposal_id)
        await client.post(
            f"/v1/evolution/{proposal_id}/evals",
            headers=headers,
            json={
                "results": [
                    {
                        "case_name": "asks-before-coding",
                        "passed": True,
                        "reason": "Passed",
                    }
                ]
            },
        )
        await client.post(f"/v1/evolution/{proposal_id}/approve", headers=headers)
        await client.post(f"/v1/evolution/{proposal_id}/activate", headers=headers)

    rolled_back = await client.post(
        "/v1/evolution/active/skills/brainstorming/rollback",
        headers=headers,
    )

    assert rolled_back.status_code == 200
    assert rolled_back.json()["proposal_id"] == proposal_ids[0]
    second = await client.get(f"/v1/evolution/{proposal_ids[1]}", headers=headers)
    assert second.json()["status"] == "rolled_back"


async def test_cloud_evolution_is_organization_scoped(client):
    org_a = await _organization(client, "alice", "release-e-org-a")
    org_b = await _organization(client, "bob", "release-e-org-b")
    headers_a = {"X-Test-User": "alice", "X-Pori-Organization": org_a}
    headers_b = {"X-Test-User": "bob", "X-Pori-Organization": org_b}

    created = await client.post(
        "/v1/evolution",
        headers=headers_a,
        json=_proposal_payload(),
    )
    proposal_id = created.json()["id"]

    missing = await client.get(f"/v1/evolution/{proposal_id}", headers=headers_b)

    assert missing.status_code == 404
