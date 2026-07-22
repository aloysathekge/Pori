from __future__ import annotations

import io
from datetime import datetime, timezone

import pytest
from sqlalchemy import func
from sqlmodel import select

from aloy_backend.event_templates import compute_template_release_checksum
from aloy_backend.models import (
    Conversation,
    Event,
    EventContextSnapshot,
    EventTemplate,
    EventTemplateCompatibility,
    EventTemplateGuidedJob,
    EventTemplateInstallation,
    EventTemplateRelease,
    EventTemplateSeed,
    EventTrailEntry,
    KnowledgeEntry,
    Run,
    SurfaceBuild,
    SurfaceDataRecord,
    SurfaceProject,
    SurfacePublication,
    SurfaceRevision,
    Task,
)
from aloy_backend.runtime import authenticated_run_context
from aloy_backend.surface_build_runner import (
    LocalDevelopmentSurfaceBuildRunner,
    SurfaceBuildRunnerResult,
)
from aloy_backend.surface_builds import SurfaceBuildHandler, SurfaceBuildParams
from aloy_backend.surface_materialization import (
    SURFACE_MATERIALIZATION_RUN_KIND,
    execute_claimed_surface_materialization,
)


class FakeTemplateBuildRunner:
    toolchain_version = "template-test-toolchain@1"

    async def build(self, *, build_id, files, manifest):
        del build_id, files, manifest
        return SurfaceBuildRunnerResult(
            status="succeeded",
            bundle=b"template-test-bundle",
        )


class MemoryObjectStore:
    def __init__(self):
        self.values: dict[str, bytes] = {}

    def put(self, key, data, *, content_type):
        del content_type
        value = data.read()
        self.values[key] = value
        return len(value)

    def open(self, key):
        return io.BytesIO(self.values[key])

    def delete(self, key):
        self.values.pop(key, None)

    def url(self, key, *, expires_s=300):
        del key, expires_s
        return None


def _run_context(event_id: str):
    return authenticated_run_context(
        user_id="test-user",
        organization_id="user:test-user",
        run_id="run-template-build",
        session_id="session-template-build",
        event_id=event_id,
        workspace_id=event_id,
        agent_id="surface-builder",
    )


async def _seed_career_os(db_session_maker, *, version: int = 1) -> tuple[str, str]:
    async with db_session_maker() as session:
        template = (
            (
                await session.execute(
                    select(EventTemplate).where(EventTemplate.slug == "career-os")
                )
            )
            .scalars()
            .first()
        )
        if template is None:
            template = EventTemplate(
                slug="career-os",
                title="Career OS",
                summary="A durable home for an intentional job search.",
                discovery_group="professional",
                status="published",
            )
            session.add(template)
            await session.flush()
        release = EventTemplateRelease(
            template_id=template.id,
            version=version,
            status="draft",
            release_notes=f"Career OS starter release {version}",
        )
        session.add(release)
        await session.flush()
        session.add_all(
            [
                EventTemplateCompatibility(
                    release_id=release.id,
                    requirement_key="event_template_schema",
                    requirement={"operator": "eq", "value": 1},
                ),
                EventTemplateCompatibility(
                    release_id=release.id,
                    requirement_key="surface_sdk",
                    requirement={"operator": "eq", "value": "1"},
                ),
                EventTemplateSeed(
                    release_id=release.id,
                    seed_key="event",
                    kind="event",
                    ordinal=0,
                    payload={
                        "title": "Career OS",
                        "summary": "Manage a focused job search with evidence and follow-up.",
                        "phase": "setup",
                        "metadata": {"cover": {"status": "none", "source": "none"}},
                    },
                ),
                EventTemplateSeed(
                    release_id=release.id,
                    seed_key="context",
                    kind="context",
                    ordinal=10,
                    payload={
                        "entries": [
                            {
                                "key": "purpose",
                                "content": (
                                    "Use this Event to organize a job search, preserve sourced "
                                    "opportunities, and track decisions over time."
                                ),
                                "tags": ["career", "job-search"],
                            }
                        ],
                        "setup_gaps": [
                            "Choose target roles",
                            "Choose preferred locations",
                            "Add a resume or portfolio",
                            "Add any existing applications",
                        ],
                    },
                ),
                EventTemplateSeed(
                    release_id=release.id,
                    seed_key="surface",
                    kind="surface",
                    ordinal=20,
                    payload={
                        "sdk_version": "1",
                        "files": {
                            "/surface.json": (
                                '{"capabilities":["data:career","data:setup"],'
                                '"widgets":["table","form"]}'
                            ),
                            "/src/App.tsx": (
                                'import { useSurfaceResourceState } from "@aloy/surface";'
                                "export default function App() {"
                                "const resource=useSurfaceResourceState('data:career');"
                                "return <main {...resource.feedbackProps}><h1>Career OS</h1>"
                                "<p>Complete setup to begin.</p></main>; }"
                            ),
                        },
                    },
                ),
                EventTemplateSeed(
                    release_id=release.id,
                    seed_key="role-types",
                    kind="surface_data",
                    ordinal=30,
                    payload={
                        "namespace": "career",
                        "key": "role-types",
                        "posture": "sample",
                        "data": {
                            "items": [
                                "AI Engineer",
                                "ML Engineer",
                                "Applied AI Engineer",
                            ]
                        },
                    },
                ),
                EventTemplateGuidedJob(
                    release_id=release.id,
                    job_key="complete-setup",
                    title="Complete your Career OS setup",
                    instructions=(
                        "Add your target roles, preferred locations, and the materials "
                        "Aloy may use for this Event."
                    ),
                    definition_of_done=(
                        "Target roles and locations are recorded and at least one useful "
                        "source or explicit no-source choice is present."
                    ),
                    priority="normal",
                    execution_profile="general",
                    ordinal=10,
                ),
            ]
        )
        await session.flush()
        release.checksum = await compute_template_release_checksum(session, release.id)
        release.status = "published"
        release.published_at = datetime.now(timezone.utc)
        template.current_release_id = release.id
        session.add(release)
        session.add(template)
        await session.commit()
        return template.id, release.id


async def _count(session, model, *filters) -> int:
    return int(
        (
            await session.execute(
                select(func.count()).select_from(model).where(*filters)
            )
        ).scalar_one()
    )


async def test_catalog_lists_valid_release_without_subscription_entitlement(
    client, db_session_maker
):
    template_id, release_id = await _seed_career_os(db_session_maker)

    response = await client.get("/v1/event-templates")

    assert response.status_code == 200
    assert response.json() == {
        "templates": [
            {
                "id": template_id,
                "slug": "career-os",
                "title": "Career OS",
                "summary": "A durable home for an intentional job search.",
                "discovery_group": "professional",
                "current_release": {
                    "id": release_id,
                    "version": 1,
                    "schema_version": 1,
                },
                "updated_at": response.json()["templates"][0]["updated_at"],
            }
        ]
    }
    assert "subscription" not in response.text.lower()

    detail = await client.get(f"/v1/event-templates/{template_id}")
    assert detail.status_code == 200
    assert detail.json()["release"]["guided_jobs"] == [
        {
            "key": "complete-setup",
            "title": "Complete your Career OS setup",
            "priority": "normal",
            "materializes_task": True,
        }
    ]


async def test_install_materializes_ordinary_event_context_surface_and_provenance(
    client, db_session_maker
):
    template_id, release_id = await _seed_career_os(db_session_maker)

    response = await client.post(
        f"/v1/event-templates/{template_id}/install",
        json={"idempotency_key": "install-career-os-001"},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    event_id = body["event"]["id"]
    assert body["event"]["title"] == "Career OS"
    assert body["event"]["phase"] == "setup"
    assert body["installation"]["release_id"] == release_id
    assert body["surface"]["status"] == "preparing"
    assert body["surface"]["run_id"]
    assert body["surface"]["run_status"] == "pending"
    assert body["replayed"] is False

    async with db_session_maker() as session:
        event = await session.get(Event, event_id)
        assert event is not None
        assert event.metadata_["template"]["release_id"] == release_id
        installation = (
            (
                await session.execute(
                    select(EventTemplateInstallation).where(
                        EventTemplateInstallation.event_id == event_id
                    )
                )
            )
            .scalars()
            .one()
        )
        assert installation.release_snapshot["release_checksum"]
        assert installation.release_snapshot["template"]["slug"] == "career-os"
        owner = (
            Event.organization_id == "user:test-user",
            Event.user_id == "test-user",
        )
        assert (
            await _count(session, Conversation, Conversation.event_id == event_id) == 1
        )
        assert (
            await _count(session, KnowledgeEntry, KnowledgeEntry.event_id == event_id)
            == 1
        )
        template_context = (
            (
                await session.execute(
                    select(KnowledgeEntry).where(KnowledgeEntry.event_id == event_id)
                )
            )
            .scalars()
            .one()
        )
        assert template_context.kind == "semantic"
        assert "event-template" in (template_context.tags or [])
        assert template_context.provenance["kind"] == "event_template"
        assert (
            await _count(
                session, EventContextSnapshot, EventContextSnapshot.event_id == event_id
            )
            == 1
        )
        assert await _count(session, Task, Task.event_id == event_id) == 1
        assert (
            await _count(session, SurfaceProject, SurfaceProject.event_id == event_id)
            == 1
        )
        assert (
            await _count(session, SurfaceRevision, SurfaceRevision.event_id == event_id)
            == 1
        )
        assert (
            await _count(session, SurfaceBuild, SurfaceBuild.event_id == event_id) == 0
        )
        assert (
            await _count(
                session,
                Run,
                Run.event_id == event_id,
                Run.run_kind == SURFACE_MATERIALIZATION_RUN_KIND,
            )
            == 1
        )
        assert (
            await _count(
                session, SurfacePublication, SurfacePublication.event_id == event_id
            )
            == 0
        )
        assert (
            await _count(
                session, SurfaceDataRecord, SurfaceDataRecord.event_id == event_id
            )
            == 2
        )
        records = list(
            (
                await session.execute(
                    select(SurfaceDataRecord).where(
                        SurfaceDataRecord.event_id == event_id
                    )
                )
            )
            .scalars()
            .all()
        )
        assert {record.posture for record in records} == {
            "template_sample",
            "template_setup_gap",
        }
        assert (
            await _count(
                session,
                EventTrailEntry,
                EventTrailEntry.event_id == event_id,
                EventTrailEntry.kind == "event_template_installed",
            )
            == 1
        )
        assert await _count(session, Event, *owner) == 1


async def test_install_replay_is_idempotent_but_new_key_creates_independent_event(
    client, db_session_maker
):
    template_id, _ = await _seed_career_os(db_session_maker)
    request = {"idempotency_key": "install-career-os-replay"}

    first = await client.post(
        f"/v1/event-templates/{template_id}/install", json=request
    )
    replay = await client.post(
        f"/v1/event-templates/{template_id}/install", json=request
    )
    another = await client.post(
        f"/v1/event-templates/{template_id}/install",
        json={"idempotency_key": "install-career-os-second", "title": "My Search"},
    )

    assert first.status_code == replay.status_code == another.status_code == 201
    assert replay.json()["replayed"] is True
    assert replay.json()["event"]["id"] == first.json()["event"]["id"]
    assert another.json()["event"]["id"] != first.json()["event"]["id"]
    assert another.json()["event"]["title"] == "My Search"
    async with db_session_maker() as session:
        assert await _count(session, EventTemplateInstallation) == 2
        assert await _count(session, Event) == 2
        assert await _count(session, SurfaceProject) == 2
        assert await _count(session, Task) == 2


async def test_seeded_source_enters_the_normal_surface_build_pipeline(
    client, db_session_maker
):
    template_id, _ = await _seed_career_os(db_session_maker)
    installed = await client.post(
        f"/v1/event-templates/{template_id}/install",
        json={"idempotency_key": "install-before-normal-build"},
    )
    event_id = installed.json()["event"]["id"]
    async with db_session_maker() as session:
        project = (
            (
                await session.execute(
                    select(SurfaceProject).where(SurfaceProject.event_id == event_id)
                )
            )
            .scalars()
            .one()
        )
        revision_id = project.draft_revision_id
        assert revision_id is not None
        assert await _count(session, SurfaceBuild) == 0

    result = await SurfaceBuildHandler(
        run_context=_run_context(event_id),
        runner=FakeTemplateBuildRunner(),
        object_store=MemoryObjectStore(),
        session_factory=db_session_maker,
    ).build(
        SurfaceBuildParams(
            revision_id=revision_id,
            idempotency_key="template-normal-host-build",
        )
    )

    assert result["status"] == "succeeded", result["diagnostics"]
    assert result["replayed"] is False
    async with db_session_maker() as session:
        assert await _count(session, SurfaceBuild) == 1


async def test_source_seeded_run_builds_inspects_and_publishes_without_a_model(
    client, db_session_maker
):
    runner = LocalDevelopmentSurfaceBuildRunner()
    if not runner.available:
        pytest.skip("The pinned local Surface toolchain is not installed")
    template_id, _ = await _seed_career_os(db_session_maker)
    installed = await client.post(
        f"/v1/event-templates/{template_id}/install",
        json={"idempotency_key": "install-auto-materialization"},
    )
    event_id = installed.json()["event"]["id"]
    run_id = installed.json()["surface"]["run_id"]
    assert installed.json()["surface"]["status"] == "preparing"
    assert run_id

    async with db_session_maker() as session:
        run = await session.get(Run, run_id)
        assert run is not None
        run.status = "running"
        run.attempt_count = 1
        run.lease_owner = "worker-template-materialization"
        run.lease_expires_at = datetime.now(timezone.utc)
        session.add(run)
        await session.commit()

    store = MemoryObjectStore()

    def handler_factory(**kwargs):
        return SurfaceBuildHandler(
            run_context=kwargs["run_context"],
            runner=runner,
            object_store=store,
            session_factory=kwargs["session_factory"],
        )

    completed = await execute_claimed_surface_materialization(
        run_id,
        "worker-template-materialization",
        session_factory=db_session_maker,
        handler_factory=handler_factory,
    )

    assert completed is True
    async with db_session_maker() as session:
        run = await session.get(Run, run_id)
        project = (
            (
                await session.execute(
                    select(SurfaceProject).where(SurfaceProject.event_id == event_id)
                )
            )
            .scalars()
            .one()
        )
        assert run is not None
        assert run.status == "completed"
        assert run.success is True
        assert run.model_assignment is None
        assert project.lifecycle == "published"
        assert project.published_revision_id == project.draft_revision_id
        assert project.published_build_id is not None
        assert (
            await _count(session, SurfaceBuild, SurfaceBuild.event_id == event_id) == 1
        )
        assert (
            await _count(
                session,
                SurfacePublication,
                SurfacePublication.event_id == event_id,
            )
            == 1
        )


async def test_catalog_changes_and_withdrawal_do_not_mutate_installed_event(
    client, db_session_maker
):
    template_id, release_id = await _seed_career_os(db_session_maker)
    installed = await client.post(
        f"/v1/event-templates/{template_id}/install",
        json={"idempotency_key": "install-before-catalog-change"},
    )
    event_id = installed.json()["event"]["id"]

    async with db_session_maker() as session:
        template = await session.get(EventTemplate, template_id)
        release = await session.get(EventTemplateRelease, release_id)
        assert template is not None and release is not None
        template.title = "Changed catalog title"
        template.status = "withdrawn"
        release.status = "withdrawn"
        session.add(template)
        session.add(release)
        await session.commit()

    hidden = await client.get("/v1/event-templates")
    event = await client.get(f"/v1/events/{event_id}")

    assert hidden.json() == {"templates": []}
    assert event.status_code == 200
    assert event.json()["event"]["title"] == "Career OS"
    async with db_session_maker() as session:
        installation = (
            (
                await session.execute(
                    select(EventTemplateInstallation).where(
                        EventTemplateInstallation.event_id == event_id
                    )
                )
            )
            .scalars()
            .one()
        )
        assert installation.release_snapshot["template"]["title"] == "Career OS"
        assert installation.release_id == release_id


async def test_corrupt_release_is_not_discoverable_or_installable(
    client, db_session_maker
):
    template_id, release_id = await _seed_career_os(db_session_maker)
    async with db_session_maker() as session:
        release = await session.get(EventTemplateRelease, release_id)
        assert release is not None
        release.checksum = "corrupt"
        session.add(release)
        await session.commit()

    catalog = await client.get("/v1/event-templates")
    install = await client.post(
        f"/v1/event-templates/{template_id}/install",
        json={"idempotency_key": "install-corrupt-release"},
    )

    assert catalog.json() == {"templates": []}
    assert install.status_code == 409
    assert install.json()["detail"] == "Template release failed integrity validation"
    async with db_session_maker() as session:
        assert await _count(session, Event) == 0
        assert await _count(session, EventTemplateInstallation) == 0


async def test_installations_are_tenant_owned_even_with_same_request_key(
    client, db_session_maker
):
    template_id, _ = await _seed_career_os(db_session_maker)
    payload = {"idempotency_key": "same-key-separate-users"}

    first = await client.post(
        f"/v1/event-templates/{template_id}/install", json=payload
    )
    second = await client.post(
        f"/v1/event-templates/{template_id}/install",
        json=payload,
        headers={"X-Test-User": "other-user"},
    )

    assert first.status_code == second.status_code == 201
    assert first.json()["event"]["id"] != second.json()["event"]["id"]
    hidden = await client.get(
        f"/v1/events/{first.json()['event']['id']}",
        headers={"X-Test-User": "other-user"},
    )
    assert hidden.status_code == 404
