"""Trusted host boundary for Surface reads and exactly-once interactions."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from pori import stable_fingerprint

from .config import settings
from .event_context import refresh_event_context_snapshot
from .event_presenters import (
    event_payload,
    file_payload,
    proposal_payload,
    task_payload,
    trail_payload,
)
from .models import (
    ActionProposal,
    Conversation,
    Event,
    EventTrailEntry,
    Message,
    Run,
    StoredFile,
    SurfaceBuild,
    SurfaceDataRecord,
    SurfaceInteraction,
    SurfaceProject,
    SurfaceRevision,
    Task,
)
from .proposal_executor import proposal_tool_registry
from .surface_commands import (
    SURFACE_COMMAND_CONTRACT_VERSION,
    ResolvedSurfaceCommand,
    SurfaceCommandError,
    apply_state_command,
    resolve_surface_command,
)
from .surface_lifecycle import stage_surface_action_message
from .surface_manifest import (
    SURFACE_PROTOCOL_VERSION,
    SurfaceIntentDeclaration,
    SurfaceManifest,
    validate_intent_payload,
)
from .surface_state import surface_record_payload
from .tenancy import OrganizationContext

MAX_INTERACTION_PAYLOAD_BYTES = 128 * 1024


class SurfaceInteractionError(ValueError):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class SurfaceInteractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    build_id: str = Field(min_length=1, max_length=200)
    code_revision_id: str = Field(min_length=1, max_length=200)
    data_revision: int = Field(ge=0)
    method: Literal["command", "dispatch", "ask_aloy", "request_action"]
    name: str = Field(min_length=1, max_length=128)
    component_id: str = Field(default="surface", min_length=1, max_length=200)
    payload: dict[str, Any] = Field(default_factory=dict)
    message: str | None = Field(default=None, max_length=50_000)
    reason: str | None = Field(default=None, max_length=4000)
    idempotency_key: str = Field(min_length=8, max_length=200)

    @model_validator(mode="after")
    def validate_method_shape(self) -> "SurfaceInteractionRequest":
        if self.method == "ask_aloy":
            if self.name != "aloy.ask" or not (self.message or "").strip():
                raise ValueError("ask_aloy requires name aloy.ask and a message")
        elif self.message is not None:
            raise ValueError("message is only valid for ask_aloy")
        if (
            len(
                json.dumps(self.payload, sort_keys=True, separators=(",", ":")).encode()
            )
            > MAX_INTERACTION_PAYLOAD_BYTES
        ):
            raise ValueError("Surface interaction payload is too large")
        return self


def _fingerprint(request: SurfaceInteractionRequest) -> str:
    encoded = json.dumps(
        request.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _record_payload(record: SurfaceDataRecord) -> dict[str, Any]:
    return surface_record_payload(record)


def interaction_payload(interaction: SurfaceInteraction) -> dict[str, Any]:
    return {
        "id": interaction.id,
        "protocol_version": SURFACE_PROTOCOL_VERSION,
        "command_contract_version": SURFACE_COMMAND_CONTRACT_VERSION,
        "event_id": interaction.event_id,
        "build_id": interaction.build_id,
        "code_revision_id": interaction.code_revision_id,
        "base_data_revision": interaction.base_data_revision,
        "data_revision": interaction.result_data_revision,
        "name": interaction.name,
        "interaction_class": interaction.interaction_class,
        "component_id": interaction.component_id,
        "status": interaction.status,
        "handling_run_id": interaction.handling_run_id,
        "proposal_id": interaction.proposal_id,
        "request_message_id": interaction.request_message_id,
        "outcome_message_id": interaction.outcome_message_id,
        "result": interaction.result,
        "error": interaction.error,
        "created_at": interaction.created_at,
        "updated_at": interaction.updated_at,
    }


async def _runtime_scope(
    session: AsyncSession,
    *,
    context: OrganizationContext,
    event_id: str,
    build_id: str,
    require_published: bool = True,
) -> tuple[Event, SurfaceProject, SurfaceRevision, SurfaceBuild, SurfaceManifest]:
    event = await session.get(Event, event_id)
    if (
        event is None
        or event.organization_id != context.organization_id
        or event.user_id != context.user_id
    ):
        raise SurfaceInteractionError(404, "Event not found")
    query = (
        select(SurfaceBuild, SurfaceProject, SurfaceRevision)
        .join(SurfaceProject, col(SurfaceProject.id) == col(SurfaceBuild.project_id))
        .join(SurfaceRevision, col(SurfaceRevision.id) == col(SurfaceBuild.revision_id))
        .where(
            SurfaceBuild.id == build_id,
            SurfaceBuild.status == "succeeded",
            col(SurfaceBuild.bundle_key).is_not(None),
            SurfaceBuild.organization_id == context.organization_id,
            SurfaceBuild.user_id == context.user_id,
            SurfaceBuild.event_id == event.id,
            SurfaceProject.organization_id == context.organization_id,
            SurfaceProject.user_id == context.user_id,
            SurfaceProject.event_id == event.id,
            SurfaceRevision.organization_id == context.organization_id,
            SurfaceRevision.user_id == context.user_id,
            SurfaceRevision.event_id == event.id,
        )
    )
    if require_published:
        query = query.where(
            SurfaceProject.published_build_id == SurfaceBuild.id,
            SurfaceProject.published_revision_id == SurfaceRevision.id,
        )
    result = await session.execute(query)
    row = result.first()
    if row is None:
        raise SurfaceInteractionError(404, "Renderable Surface build not found")
    build, project, revision = row
    try:
        manifest = SurfaceManifest.model_validate(revision.manifest)
    except ValueError as exc:
        raise SurfaceInteractionError(
            409, "Surface manifest is not executable"
        ) from exc
    return event, project, revision, build, manifest


async def surface_runtime_context(
    session: AsyncSession,
    *,
    context: OrganizationContext,
    event_id: str,
    build_id: str,
    require_published: bool = True,
) -> dict[str, Any]:
    event, project, revision, build, manifest = await _runtime_scope(
        session,
        context=context,
        event_id=event_id,
        build_id=build_id,
        require_published=require_published,
    )
    capabilities = set(manifest.capabilities)
    interactions = list(
        (
            await session.execute(
                select(SurfaceInteraction)
                .where(
                    SurfaceInteraction.organization_id == context.organization_id,
                    SurfaceInteraction.user_id == context.user_id,
                    SurfaceInteraction.event_id == event.id,
                    SurfaceInteraction.project_id == project.id,
                )
                .order_by(col(SurfaceInteraction.created_at).desc())
                .limit(100)
            )
        )
        .scalars()
        .all()
    )
    data: dict[str, Any] = {
        "interactions": [interaction_payload(row) for row in interactions]
    }
    if "event" in capabilities:
        data["event"] = event_payload(event)
    if "tasks" in capabilities:
        tasks = list(
            (
                await session.execute(
                    select(Task)
                    .where(
                        Task.organization_id == context.organization_id,
                        Task.user_id == context.user_id,
                        Task.event_id == event.id,
                    )
                    .order_by(col(Task.order), col(Task.created_at))
                    .limit(200)
                )
            )
            .scalars()
            .all()
        )
        data["tasks"] = [task_payload(row) for row in tasks]
    proposals: list[ActionProposal] | None = None
    if {"proposals", "receipts"}.intersection(capabilities):
        proposals = list(
            (
                await session.execute(
                    select(ActionProposal)
                    .where(
                        ActionProposal.organization_id == context.organization_id,
                        ActionProposal.user_id == context.user_id,
                        ActionProposal.event_id == event.id,
                    )
                    .order_by(col(ActionProposal.created_at).desc())
                    .limit(100)
                )
            )
            .scalars()
            .all()
        )
    if "proposals" in capabilities:
        data["proposals"] = [proposal_payload(row) for row in proposals or []]
    if "receipts" in capabilities:
        data["receipts"] = [
            {
                "proposal_id": row.id,
                "tool": row.tool,
                "receipt": row.receipt,
                "status": row.status,
                "updated_at": row.updated_at,
            }
            for row in proposals or []
            if row.receipt is not None
        ]
    if "files" in capabilities:
        files = list(
            (
                await session.execute(
                    select(StoredFile)
                    .where(
                        StoredFile.organization_id == context.organization_id,
                        StoredFile.user_id == context.user_id,
                        StoredFile.event_id == event.id,
                    )
                    .order_by(col(StoredFile.created_at).desc())
                    .limit(200)
                )
            )
            .scalars()
            .all()
        )
        data["files"] = [file_payload(row) for row in files]
    if "trail" in capabilities:
        trail = list(
            (
                await session.execute(
                    select(EventTrailEntry)
                    .where(
                        EventTrailEntry.organization_id == context.organization_id,
                        EventTrailEntry.user_id == context.user_id,
                        EventTrailEntry.event_id == event.id,
                    )
                    .order_by(col(EventTrailEntry.created_at).desc())
                    .limit(100)
                )
            )
            .scalars()
            .all()
        )
        data["trail"] = [trail_payload(row) for row in trail]

    namespaces = sorted(
        capability.removeprefix("data:")
        for capability in capabilities
        if capability.startswith("data:")
    )
    if namespaces:
        records = list(
            (
                await session.execute(
                    select(SurfaceDataRecord)
                    .where(
                        SurfaceDataRecord.organization_id == context.organization_id,
                        SurfaceDataRecord.user_id == context.user_id,
                        SurfaceDataRecord.event_id == event.id,
                        SurfaceDataRecord.project_id == project.id,
                        col(SurfaceDataRecord.namespace).in_(namespaces),
                    )
                    .order_by(
                        col(SurfaceDataRecord.namespace),
                        col(SurfaceDataRecord.record_key),
                    )
                    .limit(1000)
                )
            )
            .scalars()
            .all()
        )
        namespaced: dict[str, list[dict[str, Any]]] = {name: [] for name in namespaces}
        for row in records:
            namespaced[row.namespace].append(_record_payload(row))
        data["surface"] = namespaced

    return {
        "protocol_version": SURFACE_PROTOCOL_VERSION,
        "command_contract_version": SURFACE_COMMAND_CONTRACT_VERSION,
        "sdk_version": manifest.sdk_version,
        "event_id": event.id,
        "project_id": project.id,
        "build_id": build.id,
        "code_revision_id": revision.id,
        "data_revision": project.data_revision,
        "capabilities": manifest.capabilities,
        "widgets": manifest.widgets,
        "data": data,
    }


async def _existing_interaction(
    session: AsyncSession,
    *,
    project_id: str,
    idempotency_key: str,
) -> SurfaceInteraction | None:
    return (
        (
            await session.execute(
                select(SurfaceInteraction).where(
                    SurfaceInteraction.project_id == project_id,
                    SurfaceInteraction.idempotency_key == idempotency_key,
                )
            )
        )
        .scalars()
        .first()
    )


def _declaration_for(
    manifest: SurfaceManifest, request: SurfaceInteractionRequest
) -> SurfaceIntentDeclaration | None:
    if request.method == "ask_aloy":
        if "ask_aloy" not in manifest.capabilities:
            raise SurfaceInteractionError(403, "Surface lacks ask_aloy capability")
        return None
    declaration = manifest.intents.get(request.name)
    if declaration is None:
        raise SurfaceInteractionError(403, "Surface intent is not declared")
    try:
        command = resolve_surface_command(request.name, declaration)
    except SurfaceCommandError as exc:
        raise SurfaceInteractionError(exc.status_code, exc.detail) from exc
    expected_effect = {
        "dispatch": "state",
        "request_action": "external_action",
    }.get(request.method)
    if expected_effect is not None and command.effect != expected_effect:
        raise SurfaceInteractionError(422, "Surface intent uses the wrong SDK method")
    if request.method == "command" and command.effect == "local":
        raise SurfaceInteractionError(
            422, "Local Surface controls must stay inside the iframe"
        )
    try:
        validate_intent_payload(declaration.schema_, request.payload)
    except ValueError as exc:
        raise SurfaceInteractionError(422, str(exc)) from exc
    return declaration


async def handle_surface_interaction(
    session: AsyncSession,
    *,
    context: OrganizationContext,
    event_id: str,
    request: SurfaceInteractionRequest,
) -> dict[str, Any]:
    event, project, revision, build, manifest = await _runtime_scope(
        session, context=context, event_id=event_id, build_id=request.build_id
    )
    project_id = project.id
    if request.code_revision_id != revision.id:
        raise SurfaceInteractionError(409, "Surface code revision changed")
    fingerprint = _fingerprint(request)
    replay = await _existing_interaction(
        session, project_id=project.id, idempotency_key=request.idempotency_key
    )
    if replay is not None:
        if replay.request_fingerprint != fingerprint:
            raise SurfaceInteractionError(
                409, "idempotency_key was already used for another interaction"
            )
        payload = interaction_payload(replay)
        payload["replayed"] = True
        return payload

    declaration = _declaration_for(manifest, request)
    now = datetime.now(timezone.utc)
    if request.method == "ask_aloy":
        command = ResolvedSurfaceCommand(
            name="aloy.ask", effect="reasoning", wake_policy="immediate"
        )
    else:
        assert declaration is not None
        try:
            command = resolve_surface_command(request.name, declaration)
        except SurfaceCommandError as exc:
            raise SurfaceInteractionError(exc.status_code, exc.detail) from exc
    interaction = SurfaceInteraction(
        organization_id=context.organization_id,
        user_id=context.user_id,
        event_id=event.id,
        project_id=project.id,
        build_id=build.id,
        code_revision_id=revision.id,
        conversation_id=event.primary_conversation_id,
        name=request.name,
        interaction_class=command.effect,
        component_id=request.component_id,
        payload=request.payload,
        actor_id=context.user_id,
        idempotency_key=request.idempotency_key,
        request_fingerprint=fingerprint,
        base_data_revision=request.data_revision,
        status="pending",
        created_at=now,
        updated_at=now,
    )
    session.add(interaction)

    if command.effect == "state":
        if request.data_revision != project.data_revision:
            raise SurfaceInteractionError(409, "Surface data revision changed")
        assert declaration is not None
        try:
            mutation = await apply_state_command(
                session,
                project=project,
                interaction=interaction,
                declaration=declaration,
                command=command,
                payload=request.payload,
                actor_id=context.user_id,
                code_revision_id=revision.id,
                build_id=build.id,
                now=now,
            )
        except SurfaceCommandError as exc:
            raise SurfaceInteractionError(exc.status_code, exc.detail) from exc
        interaction.status = "committed"
        interaction.result_data_revision = mutation.data_revision
        interaction.result = {
            "command": command.payload(),
            "entity_refs": [mutation.entity_ref],
            "record": (
                _record_payload(mutation.record)
                if mutation.record is not None
                else None
            ),
        }
        operation_label = command.operation or "state"
        session.add(
            EventTrailEntry(
                organization_id=context.organization_id,
                user_id=context.user_id,
                event_id=event.id,
                actor_id=context.user_id,
                kind="surface_interaction_committed",
                summary=(
                    f"{operation_label.title()}d {command.namespace} "
                    "from the Event Surface"
                ),
                evidence_refs=[{"surface_interaction_id": interaction.id}],
                payload={
                    "interaction_id": interaction.id,
                    "name": request.name,
                    "command": command.payload(),
                    "entity_refs": [mutation.entity_ref],
                    "data_revision": mutation.data_revision,
                },
            )
        )

    elif command.effect == "reasoning":
        conversation = await session.get(Conversation, event.primary_conversation_id)
        if (
            conversation is None
            or conversation.event_id != event.id
            or conversation.organization_id != context.organization_id
            or conversation.user_id != context.user_id
        ):
            raise SurfaceInteractionError(409, "Event Conversation is unavailable")
        active = (
            await session.execute(
                select(func.count())
                .select_from(Run)
                .where(
                    Run.organization_id == context.organization_id,
                    Run.conversation_id == conversation.id,
                    col(Run.status).in_(["pending", "running"]),
                )
            )
        ).scalar_one()
        if active:
            raise SurfaceInteractionError(
                409, "Event Conversation already has active work"
            )
        account_active = (
            await session.execute(
                select(func.count())
                .select_from(Run)
                .where(
                    Run.organization_id == context.organization_id,
                    Run.user_id == context.user_id,
                    col(Run.status).in_(["pending", "running"]),
                )
            )
        ).scalar_one()
        account_cap = min(
            context.policy.max_concurrent_runs,
            max(1, settings.max_concurrent_runs),
        )
        if account_active >= account_cap:
            raise SurfaceInteractionError(429, "Account Run limit reached")
        snapshot, _pack, _created = await refresh_event_context_snapshot(
            session,
            organization_id=context.organization_id,
            user_id=context.user_id,
            event_id=event.id,
        )
        message = (request.message or "").strip()
        if not message:
            message = f"Carry out the {request.name} reasoning command from this Event Surface."
        trigger = {
            "contract_version": SURFACE_COMMAND_CONTRACT_VERSION,
            "event_id": event.id,
            "interaction_id": interaction.id,
            "command": command.payload(),
            "data_revision": project.data_revision,
            "context_snapshot_id": snapshot.id,
            "context_snapshot_fingerprint": snapshot.fingerprint,
        }
        user_message = Message(
            conversation_id=conversation.id,
            role="user",
            content=message,
            metadata_={
                "kind": "surface_interaction",
                "surface_interaction_id": interaction.id,
                "surface_command": trigger,
                "surface_input": request.payload,
                "code_revision_id": revision.id,
                "data_revision": project.data_revision,
            },
        )
        run = Run(
            organization_id=context.organization_id,
            user_id=context.user_id,
            event_id=event.id,
            agent_id="default_agent",
            session_id=conversation.id,
            conversation_id=conversation.id,
            idempotency_key=f"surface:{interaction.id}",
            task=(
                "<trusted-surface-command>\n"
                f"{json.dumps(trigger, sort_keys=True)}\n"
                "</trusted-surface-command>\n"
                f"User request: {message}"
            ),
            max_steps=context.policy.max_steps_per_run,
            max_attempts=context.policy.max_attempts,
            timeout_seconds=context.policy.run_timeout_seconds,
            status="pending",
        )
        session.add(user_message)
        session.add(run)
        interaction.status = "queued"
        interaction.handling_run_id = run.id
        interaction.result = {
            "command": command.payload(),
            "trigger": trigger,
            "run_id": run.id,
            "conversation_id": conversation.id,
        }
        session.add(
            EventTrailEntry(
                organization_id=context.organization_id,
                user_id=context.user_id,
                event_id=event.id,
                actor_id=context.user_id,
                kind="surface_reasoning_requested",
                summary="Asked Aloy from the Event Surface",
                run_id=run.id,
                evidence_refs=[{"surface_interaction_id": interaction.id}],
                payload={
                    "interaction_id": interaction.id,
                    "name": request.name,
                    "command": command.payload(),
                    "trigger": trigger,
                },
            )
        )

    elif command.effect == "external_action":
        assert declaration is not None and declaration.tool is not None
        if declaration.tool in context.policy.denied_tools or (
            context.policy.allowed_tools
            and declaration.tool not in context.policy.allowed_tools
        ):
            raise SurfaceInteractionError(
                403, "Surface action tool is denied by policy"
            )
        try:
            tool = proposal_tool_registry().get_tool(declaration.tool)
            normalized = tool.param_model.model_validate(request.payload).model_dump(
                mode="json"
            )
        except ValueError as exc:
            raise SurfaceInteractionError(
                422, f"Invalid Surface action: {exc}"
            ) from exc
        proposal = ActionProposal(
            organization_id=context.organization_id,
            user_id=context.user_id,
            event_id=event.id,
            origin_session_id=event.primary_conversation_id,
            origin_run_id=None,
            tool=declaration.tool,
            args=normalized,
            tool_schema_fingerprint=stable_fingerprint(
                tool.param_model.model_json_schema()
            ),
            reason=request.reason or "Requested from the Event Surface.",
            impact="Will perform an external action if approved and committed.",
            risk="high",
            routing="ask",
            status="pending",
            expires_at=now + timedelta(days=7),
            safe_default={"decision": "reject"},
        )
        session.add(proposal)
        interaction.status = "waiting_approval"
        interaction.proposal_id = proposal.id
        interaction.result = {
            "command": command.payload(),
            "proposal_id": proposal.id,
        }
        await stage_surface_action_message(
            session,
            interaction=interaction,
            proposal=proposal,
        )
        session.add(
            EventTrailEntry(
                organization_id=context.organization_id,
                user_id=context.user_id,
                event_id=event.id,
                actor_id=context.user_id,
                kind="proposal_staged",
                summary=f"Staged {declaration.tool} from the Event Surface",
                proposal_id=proposal.id,
                evidence_refs=[{"surface_interaction_id": interaction.id}],
                payload={
                    "interaction_id": interaction.id,
                    "command": command.payload(),
                    "tool": declaration.tool,
                    "status": proposal.status,
                    "routing": proposal.routing,
                },
            )
        )

    else:
        raise SurfaceInteractionError(
            501,
            f"Surface {command.effect} commands are declared but not enabled in this phase",
        )

    event.updated_at = now
    interaction.updated_at = now
    session.add(event)
    session.add(interaction)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        replay = await _existing_interaction(
            session,
            project_id=project_id,
            idempotency_key=request.idempotency_key,
        )
        if replay is not None and replay.request_fingerprint == fingerprint:
            payload = interaction_payload(replay)
            payload["replayed"] = True
            return payload
        raise SurfaceInteractionError(
            409, "Surface interaction conflicted with another update"
        ) from exc
    await session.refresh(interaction)
    payload = interaction_payload(interaction)
    payload["replayed"] = False
    return payload


__all__ = [
    "SurfaceInteractionError",
    "SurfaceInteractionRequest",
    "handle_surface_interaction",
    "interaction_payload",
    "surface_runtime_context",
]
