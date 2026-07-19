"""Cron engine tests (marathon Phase 3): schedule parsing, at-most-once
firing, run enqueueing, and the CRUD routes."""

import importlib
from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect
from sqlmodel import select

from aloy_backend.cron import (
    compute_next_run,
    tick_cron_jobs,
    validate_schedule,
    validate_timezone,
)
from aloy_backend.models import Conversation, CronJob, Event, EventTrailEntry, Run
from aloy_backend.schedule_runtime import (
    record_schedule_terminal_trail,
    scheduled_denied_tools,
)

pytestmark = pytest.mark.asyncio


class TestSchedules:
    async def test_every_schedule(self):
        now = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)
        assert compute_next_run("@every:3600", now) == now + timedelta(hours=1)

    async def test_cron_expression(self):
        now = datetime(2026, 7, 6, 12, 30, tzinfo=timezone.utc)  # a Monday
        nxt = compute_next_run("0 7 * * *", now)
        assert (nxt.hour, nxt.minute) == (7, 0)
        assert nxt > now

    async def test_cron_expression_uses_event_timezone(self):
        now = datetime(2026, 7, 6, 12, 30, tzinfo=timezone.utc)
        nxt = compute_next_run("0 7 * * *", now, "America/New_York")
        assert nxt == datetime(2026, 7, 7, 11, 0, tzinfo=timezone.utc)

    async def test_invalid_timezone_rejected(self):
        with pytest.raises(ValueError, match="Unknown IANA timezone"):
            validate_timezone("Mars/Olympus_Mons")

    async def test_invalid_schedules_rejected(self):
        for bad in ("", "not a cron", "@every:abc", "@every:5", "* * *"):
            with pytest.raises(ValueError):
                validate_schedule(bad)


async def _seed_job(db_session_maker, **overrides):
    fields = {
        "organization_id": "org-1",
        "user_id": "alice",
        "name": "daily digest",
        "task": "summarize the day",
        "schedule": "@every:3600",
        "next_run_at": datetime.now(timezone.utc) - timedelta(seconds=5),
        **overrides,
    }
    job = CronJob(**fields)
    async with db_session_maker() as session:
        event = Event(
            id="evt-scheduled",
            organization_id=fields["organization_id"],
            user_id=fields["user_id"],
            title="Scheduled Event",
        )
        conversation = Conversation(
            id="conv-scheduled",
            organization_id=fields["organization_id"],
            user_id=fields["user_id"],
            event_id=event.id,
            title=event.title,
        )
        event.primary_conversation_id = conversation.id
        job.event_id = event.id
        job.conversation_id = conversation.id
        session.add(event)
        session.add(conversation)
        session.add(job)
        await session.commit()
        return job.id


class TestTick:
    async def test_due_job_enqueues_run_and_advances(
        self, db_session_maker, monkeypatch
    ):
        monkeypatch.setattr("aloy_backend.cron.async_session", db_session_maker)
        job_id = await _seed_job(db_session_maker)

        enqueued = await tick_cron_jobs()
        assert enqueued == 1

        async with db_session_maker() as session:
            runs = (await session.execute(select(Run))).scalars().all()
            assert len(runs) == 1
            run = runs[0]
            assert run.task.endswith("Scheduled instruction:\nsummarize the day")
            assert "Read and report only" in run.task
            assert run.status == "pending"
            assert run.organization_id == "org-1"
            assert run.event_id == "evt-scheduled"
            assert run.conversation_id == "conv-scheduled"
            assert run.cron_job_id == job_id
            assert run.run_kind == "scheduled"
            assert run.run_profile["authority"] == "report_only"
            assert run.session_id == run.id
            job = await session.get(CronJob, job_id)
            assert job.last_run_id == run.id
            nxt = job.next_run_at
            if nxt.tzinfo is None:
                nxt = nxt.replace(tzinfo=timezone.utc)
            assert nxt > datetime.now(timezone.utc)
            trail = (
                (
                    await session.execute(
                        select(EventTrailEntry).where(
                            EventTrailEntry.run_id == run.id,
                            EventTrailEntry.kind == "schedule_triggered",
                        )
                    )
                )
                .scalars()
                .one()
            )
            assert trail.payload["wake_reason"] == "time_trigger"

    async def test_at_most_once_per_due_time(self, db_session_maker, monkeypatch):
        monkeypatch.setattr("aloy_backend.cron.async_session", db_session_maker)
        await _seed_job(db_session_maker)

        first = await tick_cron_jobs()
        second = await tick_cron_jobs()  # immediately re-tick: clock advanced

        assert first == 1
        assert second == 0
        async with db_session_maker() as session:
            runs = (await session.execute(select(Run))).scalars().all()
            assert len(runs) == 1

    async def test_disabled_job_skipped(self, db_session_maker, monkeypatch):
        monkeypatch.setattr("aloy_backend.cron.async_session", db_session_maker)
        await _seed_job(db_session_maker, enabled=False)

        assert await tick_cron_jobs() == 0

    async def test_invalid_schedule_disables_instead_of_crashing(
        self, db_session_maker, monkeypatch
    ):
        monkeypatch.setattr("aloy_backend.cron.async_session", db_session_maker)
        job_id = await _seed_job(db_session_maker, schedule="corrupted !!")

        assert await tick_cron_jobs() == 0
        async with db_session_maker() as session:
            job = await session.get(CronJob, job_id)
            assert job.enabled is False

    async def test_dormant_event_stays_quiet(self, db_session_maker, monkeypatch):
        monkeypatch.setattr("aloy_backend.cron.async_session", db_session_maker)
        job_id = await _seed_job(db_session_maker)
        async with db_session_maker() as session:
            event = await session.get(Event, "evt-scheduled")
            event.lifecycle = "dormant"
            session.add(event)
            await session.commit()

        assert await tick_cron_jobs() == 0
        async with db_session_maker() as session:
            assert (await session.execute(select(Run))).scalars().all() == []
            job = await session.get(CronJob, job_id)
            assert job.enabled is True


class TestAuthority:
    async def test_report_only_denies_mutation_tools(self):
        report_run = Run(
            user_id="alice",
            organization_id="org-1",
            event_id="evt-1",
            agent_id="default_agent",
            session_id="run-1",
            cron_job_id="cron-1",
            run_profile={"authority": "report_only"},
            task="report",
        )
        organize_run = report_run.model_copy(
            update={"id": "run-2", "run_profile": {"authority": "organize"}}
        )
        assert "task_create" in scheduled_denied_tools(report_run)
        assert "gmail_send" in scheduled_denied_tools(report_run)
        assert "task_create" not in scheduled_denied_tools(organize_run)
        assert "request_event_surface" in scheduled_denied_tools(organize_run)


class TestRoutes:
    async def test_crud_lifecycle(self, client):
        event_response = await client.post(
            "/v1/events",
            json={"title": "University 2026", "cover_mode": "none"},
        )
        assert event_response.status_code == 201
        event_id = event_response.json()["id"]
        created = await client.post(
            "/v1/cron",
            json={
                "event_id": event_id,
                "name": "morning briefing",
                "task": "prepare my morning briefing",
                "schedule": "0 7 * * 1-5",
                "timezone": "Africa/Johannesburg",
                "authority": "organize",
                "notification_mode": "always",
                "max_steps": 10,
            },
        )
        assert created.status_code == 201
        body = created.json()
        assert body["enabled"] is True
        assert body["event_id"] == event_id
        assert body["timezone"] == "Africa/Johannesburg"
        assert body["authority"] == "organize"
        assert body["next_run_at"] is not None
        job_id = body["id"]

        listed = await client.get("/v1/cron")
        assert listed.status_code == 200
        assert any(j["id"] == job_id for j in listed.json())

        patched = await client.patch(f"/v1/cron/{job_id}", json={"enabled": False})
        assert patched.status_code == 200
        assert patched.json()["enabled"] is False

        deleted = await client.delete(f"/v1/cron/{job_id}")
        assert deleted.status_code == 204
        listed_after_delete = await client.get("/v1/cron")
        assert all(job["id"] != job_id for job in listed_after_delete.json())

    async def test_invalid_schedule_rejected(self, client):
        event_response = await client.post(
            "/v1/events",
            json={"title": "Career OS", "cover_mode": "none"},
        )
        created = await client.post(
            "/v1/cron",
            json={
                "event_id": event_response.json()["id"],
                "name": "bad job",
                "task": "does not matter",
                "schedule": "not a schedule",
            },
        )
        assert created.status_code == 400

    async def test_due_schedule_surfaces_in_today_and_failure_notifies(
        self, client, db_session_maker, monkeypatch
    ):
        event_response = await client.post(
            "/v1/events",
            json={"title": "University 2026", "cover_mode": "none"},
        )
        event_id = event_response.json()["id"]
        created = await client.post(
            "/v1/cron",
            json={
                "event_id": event_id,
                "name": "Deadline watch",
                "task": "Review upcoming tests and deadlines",
                "schedule": "0 7 * * *",
                "timezone": "Africa/Johannesburg",
                "notification_mode": "attention",
            },
        )
        job_id = created.json()["id"]
        async with db_session_maker() as session:
            job = await session.get(CronJob, job_id)
            job.next_run_at = datetime.now(timezone.utc) - timedelta(seconds=1)
            session.add(job)
            await session.commit()

        monkeypatch.setattr("aloy_backend.cron.async_session", db_session_maker)
        assert await tick_cron_jobs() == 1

        today = (await client.get("/v1/today")).json()
        event_group = next(
            group for group in today["events"] if group["event"]["id"] == event_id
        )
        assert event_group["scheduled_work"][0]["schedule_name"] == "Deadline watch"
        assert not any(
            notification["kind"] == "schedule_triggered"
            for notification in today["notifications"]
        )

        async with db_session_maker() as session:
            run = (
                (await session.execute(select(Run).where(Run.cron_job_id == job_id)))
                .scalars()
                .one()
            )
            run.status = "failed"
            run.completed_at = datetime.now(timezone.utc)
            session.add(run)
            await record_schedule_terminal_trail(session, run=run)
            await session.commit()

        history = await client.get(f"/v1/cron/{job_id}/runs")
        assert history.status_code == 200
        assert history.json()[0]["status"] == "failed"
        today = (await client.get("/v1/today")).json()
        assert any(
            notification["kind"] == "schedule_failed"
            and notification["event_id"] == event_id
            for notification in today["notifications"]
        )


def test_event_schedule_migration_round_trip(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'event-schedules.db'}")
    metadata = sa.MetaData()
    events = sa.Table(
        "events", metadata, sa.Column("id", sa.String(), primary_key=True)
    )
    sa.Table(
        "cron_jobs",
        metadata,
        sa.Column("id", sa.String(), primary_key=True),
    )
    sa.Table(
        "runs",
        metadata,
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("event_id", sa.String(), sa.ForeignKey(events.c.id)),
    )
    metadata.create_all(engine)

    with engine.begin() as connection:
        migration = importlib.import_module(
            "aloy_backend.alembic.versions.h0e1f2a3b4c5_event_owned_schedules"
        )
        original_op = migration.op
        migration.op = Operations(MigrationContext.configure(connection))
        try:
            migration.upgrade()
            cron_columns = {
                column["name"]
                for column in inspect(connection).get_columns("cron_jobs")
            }
            run_columns = {
                column["name"] for column in inspect(connection).get_columns("runs")
            }
            assert {
                "event_id",
                "timezone",
                "authority",
                "notification_mode",
                "deleted_at",
            } <= cron_columns
            assert "cron_job_id" in run_columns

            migration.downgrade()
            cron_columns = {
                column["name"]
                for column in inspect(connection).get_columns("cron_jobs")
            }
            run_columns = {
                column["name"] for column in inspect(connection).get_columns("runs")
            }
            assert "event_id" not in cron_columns
            assert "cron_job_id" not in run_columns
        finally:
            migration.op = original_op
    engine.dispose()
