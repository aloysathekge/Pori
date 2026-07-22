"""Archive, restore, and permanently erase one user-owned Event.

Archiving is reversible and immediately removes an Event from ordinary product
views. Permanent deletion is deliberately a second, explicit operation. It
erases the Event aggregate while preserving global catalog releases and other
Events that may have originated from the same template.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select
from starlette.concurrency import run_in_threadpool

from .models import (
    ActionProposal,
    ContextArtifact,
    Conversation,
    CronJob,
    Event,
    EventBrief,
    EventConnectionGrant,
    EventContextSnapshot,
    EventSetupContextItem,
    EventSetupDraft,
    EventTemplateInstallation,
    EventTrailEntry,
    KnowledgeEntry,
    Message,
    Run,
    RunEventLog,
    RunTimelineCursor,
    RunTimelineEvent,
    StoredFile,
    SurfaceBuild,
    SurfaceCommandAttempt,
    SurfaceDataRecord,
    SurfaceEvidenceArtifact,
    SurfaceEvolutionProposal,
    SurfaceInspection,
    SurfaceInteraction,
    SurfaceProject,
    SurfacePublication,
    SurfaceRevision,
    Task,
    TraceRecord,
    UsageRecord,
)
from .storage import get_object_store

logger = logging.getLogger(__name__)


class EventLifecycleError(RuntimeError):
    def __init__(self, detail: str, *, status_code: int = 409):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


async def set_event_lifecycle(
    session: AsyncSession,
    *,
    event: Event,
    lifecycle: str,
    actor_id: str,
) -> Event:
    if event.is_life:
        raise EventLifecycleError("Life cannot be archived")
    if lifecycle not in {"active", "archived"}:
        raise EventLifecycleError(
            "Event lifecycle must be active or archived", status_code=422
        )
    if event.lifecycle == lifecycle:
        return event

    event.lifecycle = lifecycle
    event.updated_at = datetime.now(timezone.utc)
    if lifecycle == "archived":
        active_runs = (
            (
                await session.execute(
                    select(Run).where(
                        Run.organization_id == event.organization_id,
                        Run.user_id == event.user_id,
                        Run.event_id == event.id,
                        col(Run.status).in_(["pending", "running"]),
                    )
                )
            )
            .scalars()
            .all()
        )
        for run in active_runs:
            run.cancel_requested = True
            session.add(run)

    verb = "Archived" if lifecycle == "archived" else "Restored"
    session.add(event)
    session.add(
        EventTrailEntry(
            organization_id=event.organization_id,
            user_id=event.user_id,
            event_id=event.id,
            actor_id=actor_id,
            kind=f"event_{lifecycle}",
            summary=f"{verb} Event {event.title}",
            payload={"lifecycle": lifecycle},
        )
    )
    return event


async def permanently_delete_event(
    session: AsyncSession,
    *,
    event: Event,
    confirmation: str,
) -> dict[str, object]:
    """Erase an archived Event aggregate and best-effort remove its blobs."""

    if event.is_life:
        raise EventLifecycleError("Life cannot be permanently deleted")
    if event.lifecycle != "archived":
        raise EventLifecycleError("Archive this Event before permanently deleting it")
    if confirmation != event.title:
        raise EventLifecycleError(
            "Type the exact Event name to confirm permanent deletion",
            status_code=422,
        )

    active_run = (
        (
            await session.execute(
                select(Run.id)
                .where(
                    Run.organization_id == event.organization_id,
                    Run.user_id == event.user_id,
                    Run.event_id == event.id,
                    col(Run.status).in_(["pending", "running"]),
                )
                .limit(1)
            )
        )
        .scalars()
        .first()
    )
    if active_run is not None:
        raise EventLifecycleError(
            "Aloy is still stopping work for this Event. Try permanent deletion again shortly."
        )

    tenant = (
        event.organization_id,
        event.user_id,
    )
    conversation_ids = list(
        (
            await session.execute(
                select(Conversation.id).where(
                    Conversation.organization_id == tenant[0],
                    Conversation.user_id == tenant[1],
                    Conversation.event_id == event.id,
                )
            )
        ).scalars()
    )
    run_ids = list(
        (
            await session.execute(
                select(Run.id).where(
                    Run.organization_id == tenant[0],
                    Run.user_id == tenant[1],
                    Run.event_id == event.id,
                )
            )
        ).scalars()
    )
    timeline_run_ids = list(
        (
            await session.execute(
                select(RunTimelineEvent.run_id).where(
                    RunTimelineEvent.organization_id == tenant[0],
                    RunTimelineEvent.user_id == tenant[1],
                    RunTimelineEvent.event_id == event.id,
                )
            )
        ).scalars()
    )
    all_run_ids = list(dict.fromkeys([*run_ids, *timeline_run_ids]))
    draft_ids = list(
        (
            await session.execute(
                select(EventSetupDraft.id).where(
                    EventSetupDraft.organization_id == tenant[0],
                    EventSetupDraft.user_id == tenant[1],
                    EventSetupDraft.created_event_id == event.id,
                )
            )
        ).scalars()
    )

    stored_files = list(
        (
            await session.execute(
                select(StoredFile).where(
                    StoredFile.organization_id == tenant[0],
                    StoredFile.user_id == tenant[1],
                    StoredFile.event_id == event.id,
                )
            )
        ).scalars()
    )
    builds = list(
        (
            await session.execute(
                select(SurfaceBuild).where(
                    SurfaceBuild.organization_id == tenant[0],
                    SurfaceBuild.user_id == tenant[1],
                    SurfaceBuild.event_id == event.id,
                )
            )
        ).scalars()
    )
    evidence = list(
        (
            await session.execute(
                select(SurfaceEvidenceArtifact).where(
                    SurfaceEvidenceArtifact.organization_id == tenant[0],
                    SurfaceEvidenceArtifact.user_id == tenant[1],
                    SurfaceEvidenceArtifact.event_id == event.id,
                )
            )
        ).scalars()
    )
    setup_items = list(
        (
            await session.execute(
                select(EventSetupContextItem).where(
                    EventSetupContextItem.organization_id == tenant[0],
                    EventSetupContextItem.user_id == tenant[1],
                    or_(
                        col(EventSetupContextItem.event_id) == event.id,
                        col(EventSetupContextItem.draft_id).in_(draft_ids),
                    ),
                )
            )
        ).scalars()
    )
    cover = dict((event.metadata_ or {}).get("cover") or {})
    storage_keys = {
        str(key)
        for key in [
            cover.get("storage_key"),
            *(row.storage_key for row in stored_files),
            *(row.bundle_key for row in builds),
            *(row.storage_key for row in evidence),
            *(row.storage_key for row in setup_items),
        ]
        if key
    }

    direct_event_models: list[Any] = [
        SurfaceCommandAttempt,
        SurfaceEvidenceArtifact,
        SurfaceInspection,
        SurfaceInteraction,
        SurfacePublication,
        SurfaceEvolutionProposal,
        SurfaceDataRecord,
        SurfaceBuild,
        SurfaceRevision,
        SurfaceProject,
        RunEventLog,
        RunTimelineEvent,
        TraceRecord,
        ActionProposal,
        EventTrailEntry,
        ContextArtifact,
        StoredFile,
        KnowledgeEntry,
        EventBrief,
    ]
    for model in direct_event_models:
        await session.execute(
            delete(model).where(
                col(model.organization_id) == tenant[0],
                col(model.user_id) == tenant[1],
                col(model.event_id) == event.id,
            )
        )

    if all_run_ids:
        await session.execute(
            delete(RunTimelineCursor).where(
                col(RunTimelineCursor.run_id).in_(all_run_ids)
            )
        )
        await session.execute(
            delete(UsageRecord).where(
                col(UsageRecord.organization_id) == tenant[0],
                col(UsageRecord.user_id) == tenant[1],
                col(UsageRecord.run_id).in_(all_run_ids),
            )
        )
    if conversation_ids:
        await session.execute(
            delete(Message).where(col(Message.conversation_id).in_(conversation_ids))
        )
        await session.execute(
            delete(UsageRecord).where(
                col(UsageRecord.organization_id) == tenant[0],
                col(UsageRecord.user_id) == tenant[1],
                col(UsageRecord.conversation_id).in_(conversation_ids),
            )
        )

    await session.execute(
        delete(Run).where(
            col(Run.organization_id) == tenant[0],
            col(Run.user_id) == tenant[1],
            col(Run.event_id) == event.id,
        )
    )
    await session.execute(
        delete(Task).where(
            col(Task.organization_id) == tenant[0],
            col(Task.user_id) == tenant[1],
            col(Task.event_id) == event.id,
        )
    )
    await session.execute(
        delete(EventContextSnapshot).where(
            col(EventContextSnapshot.organization_id) == tenant[0],
            col(EventContextSnapshot.user_id) == tenant[1],
            col(EventContextSnapshot.event_id) == event.id,
        )
    )
    await session.execute(
        delete(CronJob).where(
            col(CronJob.organization_id) == tenant[0],
            col(CronJob.user_id) == tenant[1],
            col(CronJob.event_id) == event.id,
        )
    )
    await session.execute(
        delete(Conversation).where(
            col(Conversation.organization_id) == tenant[0],
            col(Conversation.user_id) == tenant[1],
            col(Conversation.event_id) == event.id,
        )
    )
    await session.execute(
        delete(EventConnectionGrant).where(
            col(EventConnectionGrant.organization_id) == tenant[0],
            col(EventConnectionGrant.user_id) == tenant[1],
            col(EventConnectionGrant.event_id) == event.id,
        )
    )
    await session.execute(
        delete(EventTemplateInstallation).where(
            col(EventTemplateInstallation.organization_id) == tenant[0],
            col(EventTemplateInstallation.user_id) == tenant[1],
            col(EventTemplateInstallation.event_id) == event.id,
        )
    )
    await session.execute(
        delete(EventSetupContextItem).where(
            col(EventSetupContextItem.organization_id) == tenant[0],
            col(EventSetupContextItem.user_id) == tenant[1],
            or_(
                col(EventSetupContextItem.event_id) == event.id,
                col(EventSetupContextItem.draft_id).in_(draft_ids),
            ),
        )
    )
    if draft_ids:
        await session.execute(
            delete(EventSetupDraft).where(col(EventSetupDraft.id).in_(draft_ids))
        )
    await session.delete(event)
    await session.commit()

    cleanup_failures = 0
    store = get_object_store()
    for key in storage_keys:
        try:
            await run_in_threadpool(store.delete, key)
        except Exception:  # noqa: BLE001 -- the durable pointer is already gone
            cleanup_failures += 1
            logger.exception("Could not remove an unreachable Event blob")

    return {
        "deleted": True,
        "event_id": event.id,
        "storage_objects": len(storage_keys),
        "storage_cleanup": "complete" if cleanup_failures == 0 else "pending",
    }


__all__ = [
    "EventLifecycleError",
    "permanently_delete_event",
    "set_event_lifecycle",
]
