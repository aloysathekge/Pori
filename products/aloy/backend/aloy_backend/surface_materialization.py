"""Model-free publication of an already persisted Surface revision.

Reviewed template source and future trusted imports enter the ordinary host
pipeline here. The job owns no source-writing or model authority: it can only
build, inspect, and atomically publish the exact revision frozen in its receipt.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from .database import async_session
from .models import Event, EventTrailEntry, Run, SurfaceProject, SurfaceRevision
from .runtime import authenticated_run_context
from .surface_authoring import SurfaceConflictError
from .surface_builds import SurfaceBuildHandler
from .surface_pipeline import SurfacePipelineResult, SurfaceRevisionHostPipeline
from .surface_requests import verified_surface_publication
from .tenancy import OrganizationPolicy

SURFACE_MATERIALIZATION_RUN_KIND = "surface_materialization"
SURFACE_MATERIALIZATION_AGENT_ID = "surface-host"
SURFACE_MATERIALIZATION_POLICY_VERSION = "aloy-surface-materialization@1"
SURFACE_MATERIALIZATION_RECEIPT_KIND = "surface_materialization_request"
SURFACE_MATERIALIZATION_PROFILE = {
    "profile_id": SURFACE_MATERIALIZATION_POLICY_VERSION,
    "model_tools": [],
    "host_pipeline": "surface-host-v1",
}

logger = logging.getLogger("aloy_backend.surface_materialization")


class SurfaceMaterializationBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(min_length=1, max_length=200)
    revision_id: str = Field(min_length=1, max_length=200)
    source_checksum: str = Field(pattern=r"^[a-f0-9]{64}$")
    expected_published_revision_id: str | None = None
    expected_published_build_id: str | None = None
    trigger: str = Field(min_length=1, max_length=100)


class SurfaceMaterializationSourceError(RuntimeError):
    """The frozen source cannot become live without a reviewed replacement."""


class SurfaceMaterializationHostError(RuntimeError):
    """The trusted host pipeline was temporarily unavailable."""

    def __init__(self, diagnostics: list[dict[str, Any]]):
        self.diagnostics = diagnostics
        message = (
            str(diagnostics[0].get("message") or "Surface host pipeline failed")
            if diagnostics
            else "Surface host pipeline failed"
        )
        super().__init__(message)


class SurfaceMaterializationCancelled(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _binding_fingerprint(binding: SurfaceMaterializationBinding) -> str:
    return hashlib.sha256(_canonical(binding.model_dump(mode="json"))).hexdigest()


async def _record_initial_source_failure(
    session: AsyncSession,
    *,
    run: Run,
    reason: str,
    revision_id: str | None = None,
) -> bool:
    now = datetime.now(timezone.utc)
    run.status = "failed"
    run.success = False
    run.final_answer = "The reviewed starting Surface could not be verified."
    run.reasoning = reason[:4000]
    run.progress = {
        **(run.progress or {}),
        "stage": "failed",
        "error": reason[:1000],
        "updated_at": now.isoformat(),
    }
    run.completed_at = now
    run.lease_owner = None
    run.lease_expires_at = None
    session.add(run)
    session.add(
        EventTrailEntry(
            organization_id=run.organization_id,
            user_id=run.user_id,
            event_id=run.event_id,
            actor_id="worker:surface-host",
            kind="surface_build_failed",
            summary="The starting Surface source could not be verified",
            run_id=run.id,
            evidence_refs=(
                [{"surface_revision_id": revision_id}] if revision_id else []
            ),
            payload={"mode": "persisted_source", "terminal": True},
        )
    )
    await session.commit()
    return True


def _materialization_binding(run: Run) -> SurfaceMaterializationBinding:
    receipts = [
        dict(item)
        for item in run.execution_receipts or []
        if item.get("kind") == SURFACE_MATERIALIZATION_RECEIPT_KIND
    ]
    if len(receipts) != 1:
        raise SurfaceMaterializationSourceError(
            "Surface materialization has no unique source binding"
        )
    receipt = receipts[0]
    if receipt.pop("kind", None) != SURFACE_MATERIALIZATION_RECEIPT_KIND:
        raise SurfaceMaterializationSourceError("Surface source binding is invalid")
    policy_version = receipt.pop("policy_version", None)
    fingerprint = receipt.pop("fingerprint", None)
    binding = SurfaceMaterializationBinding.model_validate(receipt)
    if (
        policy_version != SURFACE_MATERIALIZATION_POLICY_VERSION
        or fingerprint != _binding_fingerprint(binding)
    ):
        raise SurfaceMaterializationSourceError(
            "Surface source binding failed integrity validation"
        )
    return binding


async def queue_surface_revision_materialization(
    session: AsyncSession,
    *,
    event: Event,
    project: SurfaceProject,
    revision: SurfaceRevision,
    policy: OrganizationPolicy,
    trigger: str,
    actor_id: str,
    origin_evidence: list[dict[str, str]] | None = None,
) -> tuple[Run, bool]:
    """Queue one exact source revision without granting source-write authority."""

    if event.lifecycle == "archived":
        raise SurfaceMaterializationSourceError(
            "Archived Events cannot publish a starting Surface"
        )
    if (
        project.event_id != event.id
        or revision.event_id != event.id
        or revision.project_id != project.id
        or project.draft_revision_id != revision.id
    ):
        raise SurfaceMaterializationSourceError(
            "Surface source is not the Event's current draft"
        )
    binding = SurfaceMaterializationBinding(
        project_id=project.id,
        revision_id=revision.id,
        source_checksum=revision.checksum,
        expected_published_revision_id=project.published_revision_id,
        expected_published_build_id=project.published_build_id,
        trigger=trigger,
    )
    fingerprint = _binding_fingerprint(binding)
    idempotency_key = f"surface-materialize:{revision.id}:{fingerprint[:24]}"
    existing = (
        (
            await session.execute(
                select(Run).where(
                    Run.organization_id == event.organization_id,
                    Run.user_id == event.user_id,
                    Run.event_id == event.id,
                    Run.run_kind == SURFACE_MATERIALIZATION_RUN_KIND,
                    Run.idempotency_key == idempotency_key,
                )
            )
        )
        .scalars()
        .first()
    )
    if existing is not None:
        if _materialization_binding(existing) != binding:
            raise SurfaceMaterializationSourceError(
                "Surface materialization identity was reused for different source"
            )
        return existing, True

    run = Run(
        organization_id=event.organization_id,
        user_id=event.user_id,
        event_id=event.id,
        agent_id=SURFACE_MATERIALIZATION_AGENT_ID,
        session_id=event.id,
        conversation_id=None,
        idempotency_key=idempotency_key,
        run_kind=SURFACE_MATERIALIZATION_RUN_KIND,
        run_profile={
            **SURFACE_MATERIALIZATION_PROFILE,
            **binding.model_dump(mode="json"),
        },
        task="Build, inspect, and publish the Event's reviewed starting Surface",
        max_steps=1,
        max_tool_calls=3,
        timeout_seconds=min(300, policy.run_timeout_seconds),
        max_attempts=min(3, policy.max_attempts),
        isolation_profile="worker-process",
        progress={"stage": "queued", "pipeline_attempt": 1},
        execution_receipts=[
            {
                "kind": SURFACE_MATERIALIZATION_RECEIPT_KIND,
                "policy_version": SURFACE_MATERIALIZATION_POLICY_VERSION,
                **binding.model_dump(mode="json"),
                "fingerprint": fingerprint,
            }
        ],
    )
    now = datetime.now(timezone.utc)
    event.updated_at = now
    session.add(event)
    session.add(run)
    session.add(
        EventTrailEntry(
            organization_id=event.organization_id,
            user_id=event.user_id,
            event_id=event.id,
            actor_id=actor_id,
            kind="surface_build_queued",
            summary="Queued the Event's starting Surface",
            run_id=run.id,
            evidence_refs=[
                {"surface_revision_id": revision.id},
                *(origin_evidence or []),
            ],
            payload={
                "mode": "persisted_source",
                "trigger": trigger,
                "project_id": project.id,
                "revision_id": revision.id,
                "source_checksum": revision.checksum,
            },
        )
    )
    await session.flush()
    return run, False


async def execute_claimed_surface_materialization(
    run_id: str,
    worker_id: str,
    *,
    session_factory: Any = async_session,
    handler_factory: Any = SurfaceBuildHandler,
    pipeline_factory: Any = SurfaceRevisionHostPipeline,
) -> bool:
    """Publish one frozen revision through the ordinary trusted host pipeline."""

    async with session_factory() as session:
        run = await session.get(Run, run_id)
        if (
            run is None
            or run.run_kind != SURFACE_MATERIALIZATION_RUN_KIND
            or run.status != "running"
            or run.lease_owner != worker_id
        ):
            return False
        try:
            binding = _materialization_binding(run)
        except Exception as exc:
            return await _record_initial_source_failure(
                session,
                run=run,
                reason=str(exc),
            )
        project = await session.get(SurfaceProject, binding.project_id)
        revision = await session.get(SurfaceRevision, binding.revision_id)
        if (
            project is None
            or revision is None
            or project.organization_id != run.organization_id
            or project.user_id != run.user_id
            or project.event_id != run.event_id
            or revision.project_id != project.id
            or revision.organization_id != run.organization_id
            or revision.user_id != run.user_id
            or revision.event_id != run.event_id
            or revision.checksum != binding.source_checksum
            or project.draft_revision_id != revision.id
        ):
            return await _record_initial_source_failure(
                session,
                run=run,
                reason="The frozen Surface revision is unavailable or was superseded",
                revision_id=binding.revision_id,
            )
        pipeline_attempt = max(
            1, int((run.progress or {}).get("pipeline_attempt") or 1)
        )
        run_context = authenticated_run_context(
            user_id=run.user_id,
            organization_id=run.organization_id,
            run_id=run.id,
            session_id=run.session_id,
            event_id=run.event_id,
            workspace_id=run.event_id,
            agent_id=run.agent_id,
            max_steps=run.max_steps,
            max_tool_calls=run.max_tool_calls,
            max_duration_seconds=float(run.timeout_seconds),
            isolation_profile=run.isolation_profile,
        )
        run.progress = {
            **(run.progress or {}),
            "stage": "validating_source",
            "pipeline_attempt": pipeline_attempt,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        session.add(run)
        session.add(
            EventTrailEntry(
                organization_id=run.organization_id,
                user_id=run.user_id,
                event_id=run.event_id,
                actor_id="worker:surface-host",
                kind="surface_build_started",
                summary="Started preparing the Event's starting Surface",
                run_id=run.id,
                evidence_refs=[{"surface_revision_id": revision.id}],
                payload={
                    "mode": "persisted_source",
                    "attempt": run.attempt_count,
                    "pipeline_attempt": pipeline_attempt,
                },
            )
        )
        await session.commit()

    async def observe_stage(stage: str) -> None:
        async with session_factory() as progress_session:
            current = await progress_session.get(Run, run_id)
            if current is None or current.lease_owner != worker_id:
                raise SurfaceMaterializationCancelled(
                    "Surface materialization no longer owns its worker lease"
                )
            if current.cancel_requested:
                raise SurfaceMaterializationCancelled(
                    "Surface materialization was cancelled"
                )
            current.progress = {
                **(current.progress or {}),
                "stage": stage,
                "pipeline_attempt": pipeline_attempt,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            progress_session.add(current)
            await progress_session.commit()

    try:
        handler = handler_factory(
            run_context=run_context,
            session_factory=session_factory,
        )
        outcome: SurfacePipelineResult = await pipeline_factory(
            run_id=run_id,
            build_handler=handler,
            stage_observer=observe_stage,
        ).execute(
            revision_id=binding.revision_id,
            source_fingerprint=binding.source_checksum,
            expected_published_revision_id=binding.expected_published_revision_id,
            expected_published_build_id=binding.expected_published_build_id,
            attempt=pipeline_attempt,
        )
        if outcome.status == "repair_required":
            raise SurfaceMaterializationSourceError(
                str(outcome.diagnostics[0].get("message") or "Surface checks failed")
                if outcome.diagnostics
                else "Surface checks failed"
            )
        if outcome.status != "published":
            raise SurfaceMaterializationHostError(outcome.diagnostics)

        async with session_factory() as session:
            run = await session.get(Run, run_id)
            if run is None or run.lease_owner != worker_id:
                return False
            publication = await verified_surface_publication(session, run=run)
            if publication is None:
                raise SurfaceMaterializationHostError(
                    [{"message": "The host produced no verified live publication"}]
                )
            now = datetime.now(timezone.utc)
            run.status = "completed"
            run.success = True
            run.steps_taken = 1
            run.final_answer = "The Event's starting Surface is ready."
            run.reasoning = "Model-free trusted Surface pipeline completed."
            run.metrics = {"pipeline_timings_ms": outcome.timings_ms}
            run.tool_surface_fingerprint = hashlib.sha256(
                _canonical(SURFACE_MATERIALIZATION_PROFILE)
            ).hexdigest()
            run.execution_receipts = [
                *(run.execution_receipts or []),
                {
                    "kind": "surface_materialization_result",
                    "status": "published",
                    "revision_id": outcome.revision_id,
                    "build_id": outcome.build_id,
                    "publication": publication,
                    "timings_ms": outcome.timings_ms,
                },
            ]
            run.progress = {
                **(run.progress or {}),
                "stage": "published",
                "surface_receipt": publication,
                "updated_at": now.isoformat(),
            }
            run.completed_at = now
            run.lease_owner = None
            run.lease_expires_at = None
            session.add(run)
            session.add(
                EventTrailEntry(
                    organization_id=run.organization_id,
                    user_id=run.user_id,
                    event_id=run.event_id,
                    actor_id="worker:surface-host",
                    kind="surface_materialization_finished",
                    summary="Prepared the Event's starting Surface",
                    run_id=run.id,
                    evidence_refs=[
                        {"surface_revision_id": str(outcome.revision_id or "")},
                        {"surface_build_id": str(outcome.build_id or "")},
                        {"surface_publication_id": publication["publication_id"]},
                    ],
                    payload={"mode": "persisted_source", "status": "published"},
                )
            )
            await session.commit()
            return True
    except Exception as exc:
        logger.exception("Surface materialization Run %s failed", run_id)
        async with session_factory() as session:
            run = await session.get(Run, run_id)
            if run is None or run.lease_owner != worker_id:
                return False
            cancelled = isinstance(exc, SurfaceMaterializationCancelled) or bool(
                run.cancel_requested
            )
            source_failed = isinstance(
                exc,
                (SurfaceMaterializationSourceError, SurfaceConflictError),
            )
            terminal = (
                cancelled or source_failed or run.attempt_count >= run.max_attempts
            )
            retry_host = (
                isinstance(exc, SurfaceMaterializationHostError) and not terminal
            )
            now = datetime.now(timezone.utc)
            run.status = (
                "cancelled" if cancelled else "failed" if terminal else "pending"
            )
            run.success = False
            run.final_answer = (
                "The starting Surface was cancelled."
                if cancelled
                else (
                    "The reviewed starting Surface did not pass trusted checks."
                    if source_failed
                    else (
                        "The Surface host was unavailable. The starting Surface will retry."
                        if not terminal
                        else "The Surface host could not prepare the starting Surface."
                    )
                )
            )
            run.reasoning = str(exc)[:4000]
            run.progress = {
                **(run.progress or {}),
                "stage": "cancelled"
                if cancelled
                else "failed"
                if terminal
                else "retry_scheduled",
                "pipeline_attempt": pipeline_attempt + 1
                if retry_host
                else pipeline_attempt,
                "error": str(exc)[:1000],
                "updated_at": now.isoformat(),
            }
            run.completed_at = now if terminal else None
            run.lease_owner = None
            run.lease_expires_at = None
            session.add(run)
            session.add(
                EventTrailEntry(
                    organization_id=run.organization_id,
                    user_id=run.user_id,
                    event_id=run.event_id,
                    actor_id="worker:surface-host",
                    kind=(
                        "surface_build_failed"
                        if terminal
                        else "surface_build_retry_scheduled"
                    ),
                    summary=(
                        "The starting Surface could not be prepared"
                        if terminal
                        else "The starting Surface will retry safely"
                    ),
                    run_id=run.id,
                    evidence_refs=[{"surface_revision_id": binding.revision_id}],
                    payload={
                        "mode": "persisted_source",
                        "terminal": terminal,
                        "attempt": run.attempt_count,
                        "max_attempts": run.max_attempts,
                    },
                )
            )
            await session.commit()
            return True


__all__ = [
    "SURFACE_MATERIALIZATION_AGENT_ID",
    "SURFACE_MATERIALIZATION_RUN_KIND",
    "SurfaceMaterializationBinding",
    "SurfaceMaterializationSourceError",
    "execute_claimed_surface_materialization",
    "queue_surface_revision_materialization",
]
