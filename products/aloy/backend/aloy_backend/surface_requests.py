"""Model-planned Event Surface requests and verified builder receipts.

The ordinary Event agent never receives source-authoring or publication tools.
It can only describe the durable interactive experience it believes will help
the user.  The host persists that judgment as a purpose-scoped Run; the worker
then resolves the Surface Builder profile, skill, workspace, and toolchain.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from pori import RunContext

from .database import async_session
from .model_roles import (
    ModelAssignment,
    ModelRole,
    resolve_model_assignment,
)
from .models import (
    Event,
    EventTrailEntry,
    Organization,
    Run,
    SurfaceProject,
    SurfacePublication,
)
from .run_profiles import SURFACE_BUILDER_RUN_PROFILE
from .skills import SURFACE_BUILDER_SKILL_ID
from .tenancy import OrganizationPolicy

SURFACE_BUILDER_RUN_KIND = "surface_builder"
SURFACE_BUILDER_AGENT_ID = "surface-builder"


class SurfacePublicationRequiredError(RuntimeError):
    """A builder attempt ended without making its candidate live."""


class SurfaceBuilderCompletionGuard:
    """Allow a builder to terminate only after its exact publication is live."""

    def __init__(
        self,
        *,
        run_context: RunContext,
        session_factory: Any = async_session,
    ) -> None:
        self._run_context = run_context
        self._session_factory = session_factory

    async def require_publication(self) -> dict[str, Any]:
        async with self._session_factory() as session:
            run = await session.get(Run, self._run_context.run_id)
            if (
                run is None
                or run.organization_id != self._run_context.organization_id
                or run.user_id != self._run_context.user_id
                or run.event_id != self._run_context.event_id
                or run.run_kind != SURFACE_BUILDER_RUN_KIND
            ):
                raise ValueError("Surface Builder Run is unavailable")
            receipt = await verified_surface_publication(session, run=run)
        if receipt is None:
            raise ValueError(
                "Cannot finish this Surface Builder Run yet: no verified live "
                "publication belongs to this Run. Continue by persisting source "
                "with surface_write_files, then call surface_build, "
                "surface_preview, and surface_publish before answering."
            )
        return receipt


class SurfaceRequestParams(BaseModel):
    """The conversation model's product judgment, not generated UI source."""

    goal: str = Field(min_length=3, max_length=1000)
    experience: str = Field(min_length=3, max_length=3000)
    jobs: list[str] = Field(default_factory=list, max_length=20)
    source_refs: list[str] = Field(default_factory=list, max_length=30)
    interaction_notes: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("goal", "experience")
    @classmethod
    def _clean_text(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("jobs", "source_refs", "interaction_notes")
    @classmethod
    def _clean_list(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            item = " ".join(str(value).split())[:500]
            if item and item not in cleaned:
                cleaned.append(item)
        return cleaned


def _request_fingerprint(params: SurfaceRequestParams) -> str:
    payload = json.dumps(
        params.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _builder_task(params: SurfaceRequestParams) -> str:
    request = json.dumps(params.model_dump(mode="json"), indent=2, sort_keys=True)
    return (
        "Create or revise the Event's durable interactive Surface for the "
        "product request below. Use the trusted Event context and current draft "
        "supplied by the host. Return one complete React candidate containing "
        "every source file required by the experience; do not describe or claim "
        "a publication. Aloy's host persists, validates, builds, previews, and "
        "publishes the candidate. Never convert schedule rows, display sections, "
        "or navigation "
        "items into canonical Tasks unless the request explicitly requires real "
        "actionable work.\n\nSurface request:\n" + request
    )


class SurfaceRequestHandler:
    """Queue a dedicated builder Run from one authenticated Event Run."""

    def __init__(
        self,
        *,
        run_context: RunContext,
        session_factory: Any = async_session,
        owner_loop: asyncio.AbstractEventLoop | None = None,
        model_assignment_resolver: Any = resolve_model_assignment,
    ) -> None:
        self._run_context = run_context
        self._session_factory = session_factory
        self._owner_loop = owner_loop
        self._model_assignment_resolver = model_assignment_resolver

    async def _request(self, params: SurfaceRequestParams) -> dict[str, Any]:
        event_id = self._run_context.event_id
        if not event_id:
            raise ValueError("An Event is required to request a Surface")
        fingerprint = _request_fingerprint(params)
        idempotency_key = f"surface-request:{self._run_context.run_id}:{fingerprint}"
        async with self._session_factory() as session:
            event = await session.get(Event, event_id)
            if (
                event is None
                or event.organization_id != self._run_context.organization_id
                or event.user_id != self._run_context.user_id
            ):
                raise ValueError("Event is unavailable")
            if event.lifecycle == "archived":
                raise ValueError("Event is archived")
            organization = await session.get(Organization, event.organization_id)
            if organization is None:
                raise ValueError("Event organization is unavailable")
            policy = OrganizationPolicy.model_validate(organization.policy or {})

            existing = (
                (
                    await session.execute(
                        select(Run).where(
                            Run.organization_id == event.organization_id,
                            Run.user_id == event.user_id,
                            Run.event_id == event.id,
                            Run.run_kind == SURFACE_BUILDER_RUN_KIND,
                            Run.idempotency_key == idempotency_key,
                        )
                    )
                )
                .scalars()
                .first()
            )
            if existing is not None:
                return {
                    "status": existing.status,
                    "run_id": existing.id,
                    "ready": False,
                    "replayed": True,
                    "message": (
                        "The Surface request already exists. Do not claim the "
                        "Surface is ready until the host publishes it."
                    ),
                }

            assignment: ModelAssignment = self._model_assignment_resolver(
                ModelRole.SURFACE_BUILDER,
                required_capabilities=(
                    SURFACE_BUILDER_RUN_PROFILE.required_model_capabilities
                ),
                expected_skill_id=SURFACE_BUILDER_SKILL_ID,
                allowed_provider_profiles=(policy.allowed_provider_profiles or None),
                allowed_models=policy.allowed_models or None,
            )

            conversation_id = event.primary_conversation_id
            run = Run(
                organization_id=event.organization_id,
                user_id=event.user_id,
                event_id=event.id,
                agent_id=SURFACE_BUILDER_AGENT_ID,
                session_id=conversation_id or event.id,
                conversation_id=conversation_id,
                parent_run_id=self._run_context.run_id,
                root_run_id=self._run_context.run_id,
                idempotency_key=idempotency_key,
                run_kind=SURFACE_BUILDER_RUN_KIND,
                run_profile=SURFACE_BUILDER_RUN_PROFILE.descriptor(),
                model_assignment=assignment.descriptor(),
                task=_builder_task(params),
                max_steps=min(40, policy.max_steps_per_run),
                timeout_seconds=min(900, policy.run_timeout_seconds),
                max_attempts=min(3, policy.max_attempts),
                isolation_profile="worker-process",
            )
            event.updated_at = datetime.now(timezone.utc)
            session.add(event)
            session.add(run)
            session.add(
                EventTrailEntry(
                    organization_id=event.organization_id,
                    user_id=event.user_id,
                    event_id=event.id,
                    actor_id=self._run_context.agent_id,
                    kind="surface_build_queued",
                    summary="Queued a new Event Surface experience",
                    run_id=run.id,
                    evidence_refs=[
                        {"conversation_run_id": self._run_context.run_id},
                        *([{"source_ref": ref} for ref in params.source_refs[:10]]),
                    ],
                    payload={
                        "goal": params.goal,
                        "experience": params.experience,
                        "profile": SURFACE_BUILDER_RUN_PROFILE.descriptor(),
                        "model_assignment": {
                            "role": assignment.role.value,
                            "provider": assignment.provider,
                            "model": assignment.model,
                            "skill_id": assignment.skill_id,
                            "config_fingerprint": assignment.config_fingerprint,
                        },
                    },
                )
            )
            await session.commit()
            return {
                "status": "queued",
                "run_id": run.id,
                "ready": False,
                "replayed": False,
                "message": (
                    "A dedicated Surface Builder has been queued. Tell the user "
                    "it is being built; do not say it exists or is live until the "
                    "host reports a verified publication."
                ),
            }

    async def request(self, params: SurfaceRequestParams) -> dict[str, Any]:
        current = asyncio.get_running_loop()
        if self._owner_loop is None or self._owner_loop is current:
            return await self._request(params)
        future = asyncio.run_coroutine_threadsafe(
            self._request(params),
            self._owner_loop,
        )
        return await asyncio.wrap_future(future)


async def record_surface_builder_started(
    session: AsyncSession,
    *,
    run: Run,
) -> None:
    session.add(
        EventTrailEntry(
            organization_id=run.organization_id,
            user_id=run.user_id,
            event_id=run.event_id,
            actor_id=SURFACE_BUILDER_AGENT_ID,
            kind="surface_build_started",
            summary="Started building the Event Surface",
            run_id=run.id,
            evidence_refs=[{"surface_request_run_id": run.id}],
            payload={"attempt": run.attempt_count},
        )
    )


async def verified_surface_publication(
    session: AsyncSession,
    *,
    run: Run,
) -> dict[str, Any] | None:
    """Return a receipt only when this exact Run owns the live publication."""
    result = await session.execute(
        select(SurfacePublication, SurfaceProject)
        .join(
            SurfaceProject,
            col(SurfaceProject.id) == col(SurfacePublication.project_id),
        )
        .where(
            SurfacePublication.organization_id == run.organization_id,
            SurfacePublication.user_id == run.user_id,
            SurfacePublication.event_id == run.event_id,
            SurfacePublication.run_id == run.id,
            SurfaceProject.organization_id == run.organization_id,
            SurfaceProject.user_id == run.user_id,
            SurfaceProject.event_id == run.event_id,
            col(SurfaceProject.published_revision_id)
            == col(SurfacePublication.revision_id),
            col(SurfaceProject.published_build_id) == col(SurfacePublication.build_id),
        )
        .order_by(col(SurfacePublication.created_at).desc())
    )
    row = result.first()
    if row is None:
        return None
    publication, project = row
    return {
        "project_id": project.id,
        "publication_id": publication.id,
        "revision_id": publication.revision_id,
        "build_id": publication.build_id,
    }


async def record_surface_builder_failure(
    session: AsyncSession,
    *,
    run: Run,
    terminal: bool,
) -> None:
    session.add(
        EventTrailEntry(
            organization_id=run.organization_id,
            user_id=run.user_id,
            event_id=run.event_id,
            actor_id=SURFACE_BUILDER_AGENT_ID,
            kind=(
                "surface_build_failed" if terminal else "surface_build_retry_scheduled"
            ),
            summary=(
                "The Event Surface could not be published"
                if terminal
                else "The Event Surface build will retry safely"
            ),
            run_id=run.id,
            evidence_refs=[{"surface_request_run_id": run.id}],
            payload={
                "attempt": run.attempt_count,
                "max_attempts": run.max_attempts,
            },
        )
    )


__all__ = [
    "SURFACE_BUILDER_AGENT_ID",
    "SURFACE_BUILDER_RUN_KIND",
    "SurfaceBuilderCompletionGuard",
    "SurfaceRequestHandler",
    "SurfaceRequestParams",
    "SurfacePublicationRequiredError",
    "record_surface_builder_failure",
    "record_surface_builder_started",
    "verified_surface_publication",
]
