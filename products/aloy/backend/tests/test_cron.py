"""Cron engine tests (marathon Phase 3): schedule parsing, at-most-once
firing, run enqueueing, and the CRUD routes."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import select

from aloy_backend.cron import compute_next_run, tick_cron_jobs, validate_schedule
from aloy_backend.models import CronJob, Run

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
            assert run.task == "summarize the day"
            assert run.status == "pending"
            assert run.organization_id == "org-1"
            assert run.session_id == run.id
            job = await session.get(CronJob, job_id)
            assert job.last_run_id == run.id
            nxt = job.next_run_at
            if nxt.tzinfo is None:
                nxt = nxt.replace(tzinfo=timezone.utc)
            assert nxt > datetime.now(timezone.utc)

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


class TestRoutes:
    async def test_crud_lifecycle(self, client):
        created = await client.post(
            "/v1/cron",
            json={
                "name": "morning briefing",
                "task": "prepare my morning briefing",
                "schedule": "0 7 * * 1-5",
                "max_steps": 10,
            },
        )
        assert created.status_code == 201
        body = created.json()
        assert body["enabled"] is True
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
        assert await client.get("/v1/cron") is not None

    async def test_invalid_schedule_rejected(self, client):
        created = await client.post(
            "/v1/cron",
            json={
                "name": "bad job",
                "task": "does not matter",
                "schedule": "not a schedule",
            },
        )
        assert created.status_code == 400
