from __future__ import annotations

import hashlib
from io import BytesIO

from sqlmodel import select

from aloy_backend.config import settings
from aloy_backend.models import (
    Event,
    EventTemplate,
    EventTemplateOperatorReceipt,
    EventTemplateRelease,
    EventTemplateSeed,
)


def _import_body(
    *,
    version: int = 1,
    request_suffix: str = "v1",
    expected_draft_checksum: str | None = None,
    title: str = "Career OS",
) -> dict:
    return {
        "intent_id": f"intent:template-import:{request_suffix}",
        "idempotency_key": f"template-import:{request_suffix}",
        "reason": "Stage a reviewed Career OS release",
        "template": {
            "slug": "career-os",
            "title": title,
            "summary": "A durable home for an intentional job search.",
            "discovery_group": "professional",
        },
        "version": version,
        "release_notes": f"Career OS release {version}",
        "expected_draft_checksum": expected_draft_checksum,
        "assets": [],
        "compatibility": [
            {
                "requirement_key": "event_template_schema",
                "requirement": {"operator": "eq", "value": 1},
                "required": True,
            },
            {
                "requirement_key": "surface_sdk",
                "requirement": {"operator": "eq", "value": "1"},
                "required": True,
            },
        ],
        "seeds": [
            {
                "seed_key": "event",
                "kind": "event",
                "ordinal": 0,
                "payload": {
                    "title": title,
                    "summary": "Manage a focused job search with evidence.",
                    "phase": "setup",
                    "metadata": {},
                },
            },
            {
                "seed_key": "context",
                "kind": "context",
                "ordinal": 10,
                "payload": {
                    "entries": [
                        {
                            "key": "purpose",
                            "content": "Organize a job search and preserve sourced opportunities.",
                            "tags": ["career"],
                        }
                    ],
                    "setup_gaps": ["Choose target roles"],
                },
            },
            {
                "seed_key": "surface",
                "kind": "surface",
                "ordinal": 20,
                "payload": {
                    "sdk_version": "1",
                    "files": {
                        "/surface.json": (
                            '{"capabilities":["data:career","data:setup"],'
                            '"widgets":["table","form"]}'
                        ),
                        "/src/App.tsx": (
                            'import { useSurfaceResourceState } from "@aloy/surface";'
                            "export default function App(){"
                            "const resource=useSurfaceResourceState('data:career');"
                            "return <main {...resource.feedbackProps}><h1>Career OS</h1>"
                            "</main>}"
                        ),
                    },
                },
            },
            {
                "seed_key": "role-types",
                "kind": "surface_data",
                "ordinal": 30,
                "payload": {
                    "namespace": "career",
                    "key": "role-types",
                    "posture": "sample",
                    "data": {"items": ["AI Engineer", "ML Engineer"]},
                },
            },
        ],
        "guided_jobs": [
            {
                "job_key": "complete-setup",
                "title": "Complete your Career OS setup",
                "instructions": "Add target roles and preferred locations.",
                "definition_of_done": "Targets are recorded.",
                "priority": "normal",
                "execution_profile": "general",
                "ordinal": 10,
                "materialize_task": True,
            }
        ],
    }


def _publish_body(
    *, checksum: str, suffix: str, current_release_id: str | None = None
) -> dict:
    return {
        "intent_id": f"intent:template-publish:{suffix}",
        "idempotency_key": f"template-publish:{suffix}",
        "reason": "Publish the reviewed release to the global catalog",
        "expected_checksum": checksum,
        "expected_current_release_id": current_release_id,
    }


class _AssetStore:
    def __init__(self, blobs: dict[str, bytes]):
        self.blobs = blobs

    def open(self, key: str) -> BytesIO:
        try:
            return BytesIO(self.blobs[key])
        except KeyError as exc:
            raise FileNotFoundError(key) from exc


async def test_catalog_authority_is_fail_closed_and_subject_scoped(client, monkeypatch):
    monkeypatch.setattr(settings, "event_template_catalog_operator_subjects", "")
    disabled = await client.get("/v1/event-templates/operator/releases")
    assert disabled.status_code == 503

    monkeypatch.setattr(
        settings, "event_template_catalog_operator_subjects", "test-user"
    )
    denied = await client.get(
        "/v1/event-templates/operator/releases",
        headers={"X-Test-User": "other-user"},
    )
    assert denied.status_code == 404
    allowed = await client.get("/v1/event-templates/operator/releases")
    assert allowed.status_code == 200
    assert allowed.json() == {"releases": []}

    organization = await client.post(
        "/v1/organizations",
        json={"name": "Catalog Review", "slug": "catalog-review-rbac"},
    )
    organization_id = organization.json()["id"]
    added = await client.post(
        f"/v1/organizations/{organization_id}/members",
        headers={"X-Pori-Organization": organization_id},
        json={"user_id": "catalog-member", "role": "member"},
    )
    assert added.status_code == 201
    monkeypatch.setattr(
        settings, "event_template_catalog_operator_subjects", "catalog-member"
    )
    missing_rbac = await client.get(
        "/v1/event-templates/operator/releases",
        headers={
            "X-Test-User": "catalog-member",
            "X-Pori-Organization": organization_id,
        },
    )
    assert missing_rbac.status_code == 403


async def test_import_stages_validated_draft_with_idempotent_audit_receipt(
    client, db_session_maker, monkeypatch
):
    monkeypatch.setattr(
        settings, "event_template_catalog_operator_subjects", "test-user"
    )
    body = _import_body()

    imported = await client.put(
        "/v1/event-templates/operator/releases/imports", json=body
    )
    replay = await client.put(
        "/v1/event-templates/operator/releases/imports", json=body
    )

    assert imported.status_code == replay.status_code == 201
    assert imported.json()["release"]["status"] == "draft"
    assert imported.json()["release"]["validation"] == "passed"
    assert imported.json()["replayed"] is False
    assert replay.json()["replayed"] is True
    assert replay.json()["audit_receipt"] == imported.json()["audit_receipt"]
    assert (await client.get("/v1/event-templates")).json() == {"templates": []}
    operator_view = await client.get("/v1/event-templates/operator/releases")
    assert operator_view.status_code == 200
    assert operator_view.json()["releases"][0]["validation"] == "passed"
    async with db_session_maker() as session:
        receipts = list(
            (await session.execute(select(EventTemplateOperatorReceipt)))
            .scalars()
            .all()
        )
        assert len(receipts) == 1
        assert receipts[0].action == "event_template.release.imported"


async def test_publish_advances_catalog_and_is_idempotent(
    client, db_session_maker, monkeypatch
):
    monkeypatch.setattr(
        settings, "event_template_catalog_operator_subjects", "test-user"
    )
    imported = await client.put(
        "/v1/event-templates/operator/releases/imports",
        json=_import_body(request_suffix="publish-v1"),
    )
    release = imported.json()["release"]
    body = _publish_body(checksum=release["checksum"], suffix="v1")

    published = await client.post(
        f"/v1/event-templates/operator/releases/{release['id']}/publish",
        json=body,
    )
    replay = await client.post(
        f"/v1/event-templates/operator/releases/{release['id']}/publish",
        json=body,
    )

    assert published.status_code == replay.status_code == 200
    assert published.json()["release"]["status"] == "published"
    assert published.json()["replayed"] is False
    assert replay.json()["replayed"] is True
    catalog = await client.get("/v1/event-templates")
    assert catalog.status_code == 200
    assert catalog.json()["templates"][0]["current_release"]["id"] == release["id"]
    installed = await client.post(
        f"/v1/event-templates/{published.json()['template']['id']}/install",
        json={"idempotency_key": "install-published-career-os"},
    )
    assert installed.status_code == 201
    async with db_session_maker() as session:
        receipts = list(
            (await session.execute(select(EventTemplateOperatorReceipt)))
            .scalars()
            .all()
        )
        assert {receipt.action for receipt in receipts} == {
            "event_template.release.imported",
            "event_template.release.published",
        }


async def test_publish_verifies_real_asset_bytes(client, monkeypatch):
    monkeypatch.setattr(
        settings, "event_template_catalog_operator_subjects", "test-user"
    )
    blob = b"reviewed template cover"
    storage_key = "event-templates/assets/reviewed-cover"
    body = _import_body(request_suffix="asset-v1")
    body["assets"] = [
        {
            "asset_key": "cover",
            "kind": "cover",
            "storage_key": storage_key,
            "content_type": "image/webp",
            "sha256": hashlib.sha256(blob).hexdigest(),
            "size_bytes": len(blob),
            "metadata": {},
        }
    ]
    imported = await client.put(
        "/v1/event-templates/operator/releases/imports", json=body
    )
    assert imported.status_code == 201
    release = imported.json()["release"]

    monkeypatch.setattr(
        "aloy_backend.event_template_authoring.get_object_store",
        lambda: _AssetStore({}),
    )
    missing = await client.post(
        f"/v1/event-templates/operator/releases/{release['id']}/publish",
        json=_publish_body(checksum=release["checksum"], suffix="asset-missing"),
    )
    assert missing.status_code == 409
    assert "unavailable" in missing.json()["detail"]

    monkeypatch.setattr(
        "aloy_backend.event_template_authoring.get_object_store",
        lambda: _AssetStore({storage_key: blob}),
    )
    published = await client.post(
        f"/v1/event-templates/operator/releases/{release['id']}/publish",
        json=_publish_body(checksum=release["checksum"], suffix="asset-valid"),
    )
    assert published.status_code == 200
    assert published.json()["release"]["status"] == "published"


async def test_draft_replacement_requires_exact_checksum(
    client, db_session_maker, monkeypatch
):
    monkeypatch.setattr(
        settings, "event_template_catalog_operator_subjects", "test-user"
    )
    first = await client.put(
        "/v1/event-templates/operator/releases/imports",
        json=_import_body(request_suffix="replace-first"),
    )
    release = first.json()["release"]
    missing_guard = await client.put(
        "/v1/event-templates/operator/releases/imports",
        json=_import_body(request_suffix="replace-missing", title="Career Workspace"),
    )
    stale_guard = await client.put(
        "/v1/event-templates/operator/releases/imports",
        json=_import_body(
            request_suffix="replace-stale",
            expected_draft_checksum="0" * 64,
            title="Career Workspace",
        ),
    )
    replaced = await client.put(
        "/v1/event-templates/operator/releases/imports",
        json=_import_body(
            request_suffix="replace-valid",
            expected_draft_checksum=release["checksum"],
            title="Career Workspace",
        ),
    )

    assert missing_guard.status_code == stale_guard.status_code == 409
    assert replaced.status_code == 201
    assert replaced.json()["release"]["id"] == release["id"]
    assert replaced.json()["release"]["checksum"] != release["checksum"]
    async with db_session_maker() as session:
        rows = list(
            (
                await session.execute(
                    select(EventTemplateSeed).where(
                        EventTemplateSeed.release_id == release["id"]
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 4
        assert len({row.seed_key for row in rows}) == 4


async def test_published_release_is_immutable(client, monkeypatch):
    monkeypatch.setattr(
        settings, "event_template_catalog_operator_subjects", "test-user"
    )
    imported = await client.put(
        "/v1/event-templates/operator/releases/imports",
        json=_import_body(request_suffix="immutable-import"),
    )
    release = imported.json()["release"]
    published = await client.post(
        f"/v1/event-templates/operator/releases/{release['id']}/publish",
        json=_publish_body(checksum=release["checksum"], suffix="immutable"),
    )
    assert published.status_code == 200

    mutation = await client.put(
        "/v1/event-templates/operator/releases/imports",
        json=_import_body(
            request_suffix="immutable-mutation",
            expected_draft_checksum=release["checksum"],
            title="Mutated title",
        ),
    )
    assert mutation.status_code == 409
    assert mutation.json()["detail"] == "Published template releases are immutable"


async def test_publication_rejects_content_modified_after_review(
    client, db_session_maker, monkeypatch
):
    monkeypatch.setattr(
        settings, "event_template_catalog_operator_subjects", "test-user"
    )
    imported = await client.put(
        "/v1/event-templates/operator/releases/imports",
        json=_import_body(request_suffix="tamper-import"),
    )
    release = imported.json()["release"]
    async with db_session_maker() as session:
        seed = (
            (
                await session.execute(
                    select(EventTemplateSeed).where(
                        EventTemplateSeed.release_id == release["id"],
                        EventTemplateSeed.seed_key == "event",
                    )
                )
            )
            .scalars()
            .one()
        )
        seed.payload = {**seed.payload, "title": "Tampered"}
        session.add(seed)
        await session.commit()

    rejected = await client.post(
        f"/v1/event-templates/operator/releases/{release['id']}/publish",
        json=_publish_body(checksum=release["checksum"], suffix="tamper"),
    )
    assert rejected.status_code == 409
    assert rejected.json()["detail"] == "Template release failed integrity validation"
    async with db_session_maker() as session:
        stored = await session.get(EventTemplateRelease, release["id"])
        template = (
            (
                await session.execute(
                    select(EventTemplate).where(EventTemplate.slug == "career-os")
                )
            )
            .scalars()
            .one()
        )
        assert stored is not None and stored.status == "draft"
        assert template.status == "draft"
        assert template.current_release_id is None


async def test_new_release_does_not_change_an_installed_event(
    client, db_session_maker, monkeypatch
):
    monkeypatch.setattr(
        settings, "event_template_catalog_operator_subjects", "test-user"
    )
    first_import = await client.put(
        "/v1/event-templates/operator/releases/imports",
        json=_import_body(request_suffix="pin-v1", version=1),
    )
    first_release = first_import.json()["release"]
    first_publish = await client.post(
        f"/v1/event-templates/operator/releases/{first_release['id']}/publish",
        json=_publish_body(checksum=first_release["checksum"], suffix="pin-v1"),
    )
    template_id = first_publish.json()["template"]["id"]
    installed = await client.post(
        f"/v1/event-templates/{template_id}/install",
        json={"idempotency_key": "install-before-template-v2"},
    )
    event_id = installed.json()["event"]["id"]

    second_import = await client.put(
        "/v1/event-templates/operator/releases/imports",
        json=_import_body(
            request_suffix="pin-v2",
            version=2,
            title="Career OS 2",
        ),
    )
    second_release = second_import.json()["release"]
    stale_publish = await client.post(
        f"/v1/event-templates/operator/releases/{second_release['id']}/publish",
        json=_publish_body(checksum=second_release["checksum"], suffix="pin-v2-stale"),
    )
    assert stale_publish.status_code == 409
    assert "catalog changed" in stale_publish.json()["detail"].lower()
    second_publish = await client.post(
        f"/v1/event-templates/operator/releases/{second_release['id']}/publish",
        json=_publish_body(
            checksum=second_release["checksum"],
            suffix="pin-v2",
            current_release_id=first_release["id"],
        ),
    )
    assert second_publish.status_code == 200
    assert second_publish.json()["template"]["title"] == "Career OS 2"

    async with db_session_maker() as session:
        event = await session.get(Event, event_id)
        assert event is not None
        assert event.title == "Career OS"
        assert event.metadata_["template"]["release_id"] == first_release["id"]
        assert event.metadata_["template"]["version"] == 1
