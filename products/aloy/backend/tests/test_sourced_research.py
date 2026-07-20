from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlmodel import select

from aloy_backend.models import EventTrailEntry, KnowledgeEntry, Run
from aloy_backend.research_outcomes import gate_and_index_research_run
from aloy_backend.run_profiles import (
    SOURCED_RESEARCH_RUN_PROFILE,
    resolve_persisted_run_profile,
)
from aloy_backend.runtime import authenticated_run_context
from aloy_backend.tools.research import (
    EventEvidenceRecorder,
    EventRecordHandler,
    EventRecordUpsertParams,
)
from pori import get_workspace_data

pytestmark = pytest.mark.asyncio


async def test_schedule_authority_metadata_is_not_misread_as_a_purpose_profile():
    assert (
        resolve_persisted_run_profile(
            {"authority": "report_only", "notification_mode": "in_app"}
        )
        is None
    )


async def _event(client, title: str = "Career OS") -> dict:
    response = await client.post(
        "/v1/events",
        json={"title": title, "summary": "Find sourced startup opportunities"},
    )
    assert response.status_code == 201
    return response.json()


def _run_context(event_id: str, run_id: str = "research-run"):
    return authenticated_run_context(
        user_id="test-user",
        organization_id="user:test-user",
        run_id=run_id,
        session_id="research-session",
        event_id=event_id,
        workspace_id=event_id,
        agent_id="research-agent",
    )


async def test_web_evidence_commits_before_same_event_record_and_is_inspectable(
    client, db_session_maker
):
    event = await _event(client)
    context = _run_context(event["id"])
    recorder = EventEvidenceRecorder(
        run_context=context,
        task_id="task-research",
        session_factory=db_session_maker,
    )
    observation = {
        "kind": "web_page",
        "url": "https://example.com/jobs/ai-engineer",
        "title": "AI Engineer",
        "retrieved_at": datetime.now(timezone.utc),
        "provider": "direct",
        "content_sha256": "a" * 64,
        "excerpt": "A current startup opening.",
    }
    receipts = await recorder.record_many([observation, observation])
    evidence_id = receipts[0]["evidence_id"]
    assert {receipt["evidence_id"] for receipt in receipts} == {evidence_id}
    assert receipts[0]["persisted"] is True

    handler = EventRecordHandler(
        run_context=context,
        task_id="task-research",
        session_factory=db_session_maker,
    )
    record = await handler.upsert(
        EventRecordUpsertParams(
            namespace="career.opportunities",
            record_key="example-ai-engineer",
            title="Example — AI Engineer",
            data={"company": "Example", "role": "AI Engineer"},
            evidence_ids=[evidence_id],
        )
    )
    replay = await handler.upsert(
        EventRecordUpsertParams(
            namespace="career.opportunities",
            record_key="example-ai-engineer",
            title="Example — AI Engineer",
            data={"company": "Example", "role": "AI Engineer"},
            evidence_ids=[evidence_id],
        )
    )
    assert replay["id"] == record["id"]
    assert record["posture"] == "observed"
    assert record["evidence_refs"][0]["url"] == ("https://example.com/jobs/ai-engineer")

    evidence_response = await client.get(f"/v1/events/{event['id']}/evidence")
    records_response = await client.get(
        f"/v1/events/{event['id']}/records",
        params={"namespace": "career.opportunities"},
    )
    assert evidence_response.status_code == 200
    assert evidence_response.json()[0]["id"] == evidence_id
    assert records_response.status_code == 200
    assert len(records_response.json()) == 1
    assert records_response.json()[0]["key"] == "example-ai-engineer"
    memory_response = await client.get(f"/v1/events/{event['id']}/memory")
    assert memory_response.status_code == 200
    memory_ids = {
        item["id"] for item in memory_response.json().get("event_records", [])
    }
    assert evidence_id not in memory_ids
    assert record["id"] not in memory_ids
    generic_memory = await client.get("/v1/me/memory/knowledge")
    assert generic_memory.status_code == 200
    generic_ids = {item["id"] for item in generic_memory.json()}
    assert evidence_id not in generic_ids
    assert record["id"] not in generic_ids
    protected_delete = await client.delete(f"/v1/me/memory/knowledge/{evidence_id}")
    assert protected_delete.status_code == 409
    reset = await client.delete("/v1/me/memory")
    assert reset.status_code == 204
    evidence_after_reset = await client.get(f"/v1/events/{event['id']}/evidence")
    assert evidence_after_reset.json()[0]["id"] == evidence_id

    async with db_session_maker() as session:
        evidence = await session.get(KnowledgeEntry, evidence_id)
        assert evidence is not None
        assert evidence.event_id == event["id"]
        trail = list(
            (
                await session.execute(
                    select(EventTrailEntry).where(
                        EventTrailEntry.event_id == event["id"]
                    )
                )
            )
            .scalars()
            .all()
        )
        assert {entry.kind for entry in trail}.issuperset(
            {"research_evidence_added", "event_record_changed"}
        )
        assert sum(entry.kind == "event_record_changed" for entry in trail) == 1


async def test_event_record_rejects_cross_event_evidence(client, db_session_maker):
    first = await _event(client, "First")
    second = await _event(client, "Second")
    recorder = EventEvidenceRecorder(
        run_context=_run_context(first["id"], "first-run"),
        session_factory=db_session_maker,
    )
    evidence_id = (
        await recorder.record_many(
            [
                {
                    "kind": "web_search_result",
                    "url": "https://example.com/source",
                    "title": "Source",
                    "retrieved_at": datetime.now(timezone.utc),
                    "provider": "test",
                    "query": "query",
                    "content_sha256": "b" * 64,
                    "excerpt": "Observed",
                }
            ]
        )
    )[0]["evidence_id"]
    handler = EventRecordHandler(
        run_context=_run_context(second["id"], "second-run"),
        session_factory=db_session_maker,
    )
    with pytest.raises(ValueError, match="Evidence is unavailable in this Event"):
        await handler.upsert(
            EventRecordUpsertParams(
                namespace="research",
                record_key="forbidden",
                title="Must not leak",
                evidence_ids=[evidence_id],
            )
        )


async def test_research_task_freezes_profile_on_queued_run(client, db_session_maker):
    event = await _event(client)
    created = await client.post(
        f"/v1/events/{event['id']}/tasks",
        json={
            "title": "Research US startups",
            "execution_profile": "sourced_research",
        },
    )
    assert created.status_code == 201
    task = created.json()
    assert task["execution_profile"] == "sourced_research"
    queued = await client.post(f"/v1/events/{event['id']}/tasks/{task['id']}/work")
    assert queued.status_code == 202
    async with db_session_maker() as session:
        run = await session.get(Run, queued.json()["run"]["id"])
        assert run is not None
        assert run.run_profile == SOURCED_RESEARCH_RUN_PROFILE.descriptor()


async def test_research_gate_indexes_only_cited_durable_report(
    client, db_session_maker, tmp_path, monkeypatch
):
    from aloy_backend import research_outcomes

    event = await _event(client)
    run_id = "gate-run"
    context = _run_context(event["id"], run_id)
    recorder = EventEvidenceRecorder(
        run_context=context,
        task_id="task-gate",
        session_factory=db_session_maker,
    )
    evidence_id = (
        await recorder.record_many(
            [
                {
                    "kind": "web_page",
                    "url": "https://example.com/careers",
                    "title": "Careers",
                    "retrieved_at": datetime.now(timezone.utc),
                    "provider": "direct",
                    "content_sha256": "c" * 64,
                    "excerpt": "AI Engineer opening",
                }
            ]
        )
    )[0]["evidence_id"]
    handler = EventRecordHandler(
        run_context=context,
        task_id="task-gate",
        session_factory=db_session_maker,
    )
    await handler.upsert(
        EventRecordUpsertParams(
            namespace="career.opportunities",
            record_key="example",
            title="Example role",
            evidence_ids=[evidence_id],
        )
    )

    sandbox = tmp_path / "sandbox"
    monkeypatch.setattr(research_outcomes.settings, "sandbox_base_dir", str(sandbox))
    thread = get_workspace_data(event["id"], run_id, str(sandbox))
    report = Path(thread.workspace_path) / "startup-opportunities.md"
    report.write_text(
        "# Opportunities\n\n[Example careers](https://example.com/careers)\n",
        encoding="utf-8",
    )
    run = Run(
        id=run_id,
        organization_id="user:test-user",
        user_id="test-user",
        event_id=event["id"],
        task_id="task-gate",
        agent_id="research-agent",
        session_id="research-session",
        task="Research",
        status="running",
        run_profile=SOURCED_RESEARCH_RUN_PROFILE.descriptor(),
    )
    async with db_session_maker() as session:
        session.add(run)
        await session.commit()
        gate = await gate_and_index_research_run(
            session,
            run=run,
            artifacts=[{"kind": "file", "path": str(report), "file_id": "report-file"}],
            # Simulate a worker restart: the durable rows survive, while the
            # new process has empty in-memory evidence/record collectors.
            evidence_ids=(),
            record_ids=(),
            session_factory=db_session_maker,
        )
        assert gate.accepted is True
        assert gate.report_file_ids == ("report-file",)
        await session.commit()

    async with db_session_maker() as session:
        reports = list(
            (
                await session.execute(
                    select(KnowledgeEntry).where(
                        KnowledgeEntry.event_id == event["id"],
                        KnowledgeEntry.source == "agent",
                    )
                )
            )
            .scalars()
            .all()
        )
        indexed = [
            row
            for row in reports
            if (row.metadata_ or {}).get("record_type") == "research_report"
        ]
        assert indexed[0].metadata_["file_id"] == "report-file"


async def test_research_gate_rejects_unverified_only_records(
    client, db_session_maker, tmp_path, monkeypatch
):
    from aloy_backend import research_outcomes

    event = await _event(client)
    run_id = "unverified-gate-run"
    context = _run_context(event["id"], run_id)
    recorder = EventEvidenceRecorder(
        run_context=context,
        task_id="task-unverified",
        session_factory=db_session_maker,
    )
    await recorder.record_many(
        [
            {
                "kind": "web_page",
                "url": "https://example.com/inaccessible-role",
                "title": "Inaccessible role",
                "retrieved_at": datetime.now(timezone.utc),
                "provider": "direct",
                "content_sha256": "d" * 64,
                "excerpt": "No material claim could be verified.",
            }
        ]
    )
    handler = EventRecordHandler(
        run_context=context,
        task_id="task-unverified",
        session_factory=db_session_maker,
    )
    await handler.upsert(
        EventRecordUpsertParams(
            namespace="career.opportunities",
            record_key="unverified-role",
            title="Unverified role",
            posture="unverified",
        )
    )

    sandbox = tmp_path / "sandbox"
    monkeypatch.setattr(research_outcomes.settings, "sandbox_base_dir", str(sandbox))
    thread = get_workspace_data(event["id"], run_id, str(sandbox))
    report = Path(thread.workspace_path) / "unverified-opportunity.md"
    report.write_text(
        "# Unverified\n\n[Source](https://example.com/inaccessible-role)\n",
        encoding="utf-8",
    )
    run = Run(
        id=run_id,
        organization_id="user:test-user",
        user_id="test-user",
        event_id=event["id"],
        task_id="task-unverified",
        agent_id="research-agent",
        session_id="research-session",
        task="Research",
        status="running",
        run_profile=SOURCED_RESEARCH_RUN_PROFILE.descriptor(),
    )
    async with db_session_maker() as session:
        session.add(run)
        await session.commit()
        gate = await gate_and_index_research_run(
            session,
            run=run,
            artifacts=[
                {"kind": "file", "path": str(report), "file_id": "unverified-file"}
            ],
            evidence_ids=(),
            record_ids=(),
            session_factory=db_session_maker,
        )

    assert gate.accepted is False
    assert "No canonical evidence-backed Event records were produced" in gate.errors
