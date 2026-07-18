from __future__ import annotations

from types import SimpleNamespace

from aloy_backend.models import Event, SurfaceDataRecord, SurfaceProject
from aloy_backend.tools.surface_state import (
    SURFACE_STATE_CONTEXT_KEY,
    SurfaceStateReader,
    SurfaceStateReadParams,
    surface_state_read_tool,
)


async def test_surface_state_tool_reads_only_the_bound_event(
    client,
    db_session_maker,
):
    first = (
        await client.post(
            "/v1/events",
            json={"title": "Career OS", "summary": "Applications", "phase": "active"},
        )
    ).json()
    second = (
        await client.post(
            "/v1/events",
            json={"title": "Private trip", "summary": "Travel", "phase": "active"},
        )
    ).json()
    async with db_session_maker() as session:
        first_event = await session.get(Event, first["id"])
        second_event = await session.get(Event, second["id"])
        assert first_event is not None and second_event is not None
        first_project = SurfaceProject(
            organization_id=first_event.organization_id,
            user_id=first_event.user_id,
            event_id=first_event.id,
            data_revision=1,
        )
        second_project = SurfaceProject(
            organization_id=second_event.organization_id,
            user_id=second_event.user_id,
            event_id=second_event.id,
            data_revision=1,
        )
        session.add(first_project)
        session.add(second_project)
        await session.flush()
        session.add(
            SurfaceDataRecord(
                organization_id=first_event.organization_id,
                user_id=first_event.user_id,
                event_id=first_event.id,
                project_id=first_project.id,
                namespace="career",
                record_key="app-1",
                data={"company": "Acme"},
                revision=1,
                actor_id=first_event.user_id,
            )
        )
        session.add(
            SurfaceDataRecord(
                organization_id=second_event.organization_id,
                user_id=second_event.user_id,
                event_id=second_event.id,
                project_id=second_project.id,
                namespace="travel",
                record_key="booking-1",
                data={"secret": "not visible"},
                revision=1,
                actor_id=second_event.user_id,
            )
        )
        await session.commit()

    reader = SurfaceStateReader(
        run_context=SimpleNamespace(
            organization_id=first_event.organization_id,
            user_id=first_event.user_id,
            event_id=first_event.id,
        ),
        session_factory=db_session_maker,
    )
    result = await surface_state_read_tool(
        SurfaceStateReadParams(namespace="career"),
        {SURFACE_STATE_CONTEXT_KEY: reader},
    )

    assert result["event_id"] == first["id"]
    assert result["data_revision"] == 1
    assert [record["key"] for record in result["records"]] == ["app-1"]
    assert "not visible" not in str(result)
