"""Versioned host-owned command semantics for generated Event Surfaces.

Generated React declares intent and supplies typed input. This module owns the
meaning of that intent: effect classification, wake policy, optimistic
concurrency, and exact entity mutation semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from .models import SurfaceDataRecord, SurfaceInteraction, SurfaceProject
from .surface_manifest import SurfaceIntentDeclaration

SURFACE_COMMAND_CONTRACT_VERSION: Literal["1"] = "1"
SurfaceCommandEffect = Literal[
    "local", "state", "reasoning", "external_action", "automation", "source_change"
]
SurfaceStateOperation = Literal["create", "replace", "merge", "upsert", "delete"]
SurfaceWakePolicy = Literal["never", "immediate", "approval", "scheduled", "builder"]

_LEGACY_EFFECTS: dict[str, SurfaceCommandEffect] = {
    "durable_selection": "state",
    "reasoning_request": "reasoning",
    "external_action": "external_action",
    "application_change": "source_change",
}
_WAKE_POLICIES: dict[SurfaceCommandEffect, SurfaceWakePolicy] = {
    "local": "never",
    "state": "never",
    "reasoning": "immediate",
    "external_action": "approval",
    "automation": "scheduled",
    "source_change": "builder",
}


class SurfaceCommandError(ValueError):
    """A deterministic command rejection safe to return across the SDK."""

    def __init__(
        self, status_code: int, code: str, detail: str, *, retryable: bool = False
    ):
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.detail = detail
        self.retryable = retryable


@dataclass(frozen=True)
class ResolvedSurfaceCommand:
    name: str
    effect: SurfaceCommandEffect
    wake_policy: SurfaceWakePolicy
    operation: SurfaceStateOperation | None = None
    namespace: str | None = None
    legacy_upsert: bool = False

    def payload(self) -> dict[str, Any]:
        return {
            "contract_version": SURFACE_COMMAND_CONTRACT_VERSION,
            "name": self.name,
            "effect": self.effect,
            "wake_policy": self.wake_policy,
            "operation": self.operation,
            "namespace": self.namespace,
            "legacy_compatibility": self.legacy_upsert,
        }


@dataclass(frozen=True)
class SurfaceMutationResult:
    data_revision: int
    entity_ref: dict[str, str]
    record: SurfaceDataRecord | None


def resolve_surface_command(
    name: str, declaration: SurfaceIntentDeclaration
) -> ResolvedSurfaceCommand:
    raw_effect = declaration.interaction_class
    effect = _LEGACY_EFFECTS.get(raw_effect, raw_effect)
    if effect not in _WAKE_POLICIES:
        raise SurfaceCommandError(
            422, "unsupported_effect", "Unsupported Surface command effect"
        )
    operation: SurfaceStateOperation | None = None
    namespace: str | None = None
    legacy_upsert = False
    if effect == "state":
        if declaration.write is None:
            raise SurfaceCommandError(
                422, "missing_mutation", "State command lacks a mutation declaration"
            )
        operation = declaration.write.operation
        namespace = declaration.write.namespace
        # Published V1 manifests used ``durable_selection`` and had implicit
        # upsert behavior. Parsing and reserializing those manifests may add a
        # default operation, so the legacy class—not Pydantic field presence—
        # is the durable compatibility discriminator.
        legacy_upsert = raw_effect == "durable_selection"
    return ResolvedSurfaceCommand(
        name=name,
        effect=effect,  # type: ignore[arg-type]
        wake_policy=_WAKE_POLICIES[effect],  # type: ignore[index]
        operation=operation,
        namespace=namespace,
        legacy_upsert=legacy_upsert,
    )


def _record_key(declaration: SurfaceIntentDeclaration, payload: dict[str, Any]) -> str:
    assert declaration.write is not None
    value: Any = declaration.write.key
    if declaration.write.key_field:
        value = payload.get(declaration.write.key_field)
    if not isinstance(value, str) or not value.strip():
        raise SurfaceCommandError(
            422, "invalid_entity_key", "Surface entity key is invalid"
        )
    return value.strip()[:200]


async def apply_state_command(
    session: AsyncSession,
    *,
    project: SurfaceProject,
    interaction: SurfaceInteraction,
    declaration: SurfaceIntentDeclaration,
    command: ResolvedSurfaceCommand,
    payload: dict[str, Any],
    actor_id: str,
    code_revision_id: str,
    build_id: str,
    now: datetime,
) -> SurfaceMutationResult:
    """Apply one strict entity command behind a project-wide revision CAS."""
    if (
        command.effect != "state"
        or command.operation is None
        or command.namespace is None
    ):
        raise SurfaceCommandError(
            422, "not_state_command", "Command does not mutate Surface state"
        )
    key = _record_key(declaration, payload)
    current = (
        (
            await session.execute(
                select(SurfaceDataRecord).where(
                    SurfaceDataRecord.project_id == project.id,
                    SurfaceDataRecord.namespace == command.namespace,
                    SurfaceDataRecord.record_key == key,
                )
            )
        )
        .scalars()
        .first()
    )
    operation = command.operation
    if operation == "create" and current is not None:
        raise SurfaceCommandError(409, "entity_exists", "Surface entity already exists")
    if (
        operation in {"replace", "merge", "delete"}
        and current is None
        and not command.legacy_upsert
    ):
        raise SurfaceCommandError(
            409, "entity_missing", "Surface entity no longer exists"
        )

    next_revision = project.data_revision + 1
    claimed = await session.execute(
        update(SurfaceProject)
        .where(
            col(SurfaceProject.id) == project.id,
            col(SurfaceProject.data_revision) == project.data_revision,
        )
        .values(data_revision=next_revision, updated_at=now)
    )
    if claimed.rowcount != 1:  # type: ignore[attr-defined]
        raise SurfaceCommandError(
            409,
            "stale_data_revision",
            "Surface data changed concurrently",
            retryable=True,
        )

    provenance = {
        "surface_interaction_id": interaction.id,
        "command_contract_version": SURFACE_COMMAND_CONTRACT_VERSION,
        "command_name": command.name,
        "command_operation": operation,
        "code_revision_id": code_revision_id,
        "build_id": build_id,
    }
    record = current
    if operation == "delete":
        assert record is not None
        await session.delete(record)
        record = None
    else:
        if current is None:
            record = SurfaceDataRecord(
                organization_id=project.organization_id,
                user_id=project.user_id,
                event_id=project.event_id,
                project_id=project.id,
                namespace=command.namespace,
                record_key=key,
                data=dict(payload),
                revision=next_revision,
                posture=(
                    declaration.write.posture if declaration.write else "user_reported"
                ),
                actor_id=actor_id,
                provenance=provenance,
                created_at=now,
                updated_at=now,
            )
        else:
            current.data = (
                {**current.data, **payload} if operation == "merge" else dict(payload)
            )
            current.revision = next_revision
            current.posture = (
                declaration.write.posture if declaration.write else "user_reported"
            )
            current.actor_id = actor_id
            current.provenance = provenance
            current.updated_at = now
            record = current
        session.add(record)
    await session.flush()
    return SurfaceMutationResult(
        data_revision=next_revision,
        entity_ref={"namespace": command.namespace, "key": key},
        record=record,
    )


__all__ = [
    "SURFACE_COMMAND_CONTRACT_VERSION",
    "ResolvedSurfaceCommand",
    "SurfaceCommandError",
    "SurfaceMutationResult",
    "apply_state_command",
    "resolve_surface_command",
]
