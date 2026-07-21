"""Fail-closed operator workflow for staging and publishing template releases."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import delete, or_, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from .config import settings
from .event_templates import (
    DISCOVERY_GROUPS,
    MAX_TEMPLATE_SNAPSHOT_BYTES,
    TEMPLATE_SCHEMA_VERSION,
    AssetPayload,
    CompatibilityPayload,
    EventTemplateError,
    GuidedJobPayload,
    compute_template_release_checksum,
    load_template_release_rows,
    template_release_content,
    validate_template_release,
)
from .models import (
    EventTemplate,
    EventTemplateAsset,
    EventTemplateCompatibility,
    EventTemplateGuidedJob,
    EventTemplateOperatorReceipt,
    EventTemplateRelease,
    EventTemplateSeed,
)
from .storage import ObjectStore, get_object_store
from .tenancy import OrganizationContext

_INTENT_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,199}$"
_SLUG_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"


class CatalogIdentityInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    slug: str = Field(min_length=1, max_length=100, pattern=_SLUG_PATTERN)
    title: str = Field(min_length=1, max_length=300)
    summary: str = Field(default="", max_length=2_000)
    discovery_group: Literal[
        "student", "individual", "professional", "team", "business"
    ]


class CompatibilityInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    requirement_key: str = Field(min_length=1, max_length=100)
    requirement: CompatibilityPayload
    required: bool = True


class SeedInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    seed_key: str = Field(min_length=1, max_length=100)
    kind: Literal["event", "context", "surface", "surface_data"]
    ordinal: int = Field(default=0, ge=0, le=10_000)
    payload: dict[str, Any] = Field(default_factory=dict)


class OperatorMutationInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    intent_id: str = Field(min_length=8, max_length=200, pattern=_INTENT_PATTERN)
    idempotency_key: str = Field(min_length=8, max_length=200, pattern=_INTENT_PATTERN)
    reason: str = Field(min_length=3, max_length=500)


class TemplateReleaseImportInput(OperatorMutationInput):
    template: CatalogIdentityInput
    version: int = Field(ge=1, le=1_000_000)
    release_notes: str = Field(default="", max_length=10_000)
    expected_draft_checksum: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    assets: list[AssetPayload] = Field(default_factory=list, max_length=100)
    compatibility: list[CompatibilityInput] = Field(default_factory=list, max_length=50)
    seeds: list[SeedInput] = Field(min_length=1, max_length=250)
    guided_jobs: list[GuidedJobPayload] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def reject_duplicate_keys(self) -> "TemplateReleaseImportInput":
        keyed = (
            ("asset", [item.asset_key for item in self.assets]),
            (
                "compatibility",
                [item.requirement_key for item in self.compatibility],
            ),
            ("seed", [item.seed_key for item in self.seeds]),
            ("guided job", [item.job_key for item in self.guided_jobs]),
        )
        for label, values in keyed:
            if len(values) != len(set(values)):
                raise ValueError(f"Template {label} keys must be unique")
        return self


class TemplateReleasePublishInput(OperatorMutationInput):
    expected_checksum: str = Field(pattern=r"^[a-f0-9]{64}$")
    expected_current_release_id: str | None = Field()


def ensure_catalog_operator(context: OrganizationContext) -> None:
    """Require deployment-authorized global catalog authority in addition to RBAC."""
    subjects = settings.event_template_catalog_operator_subjects_set
    if not subjects:
        raise EventTemplateError(
            "Event template catalog authority is not configured",
            status_code=503,
        )
    if context.user_id not in subjects:
        raise EventTemplateError(
            "Event template catalog operator not found", status_code=404
        )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _verify_asset_blob(store: ObjectStore, asset: EventTemplateAsset) -> None:
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        with store.open(asset.storage_key) as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size_bytes += len(chunk)
    except (FileNotFoundError, OSError) as exc:
        raise EventTemplateError(
            f"Template asset {asset.asset_key!r} is unavailable"
        ) from exc
    if size_bytes != asset.size_bytes or digest.hexdigest() != asset.sha256:
        raise EventTemplateError(
            f"Template asset {asset.asset_key!r} failed integrity validation"
        )


async def _verify_release_assets(assets: tuple[EventTemplateAsset, ...]) -> None:
    """Verify stored bytes only at publication, not on latency-sensitive reads."""
    if not assets:
        return
    store = get_object_store()
    for asset in assets:
        await asyncio.to_thread(_verify_asset_blob, store, asset)


def _fingerprint(action: str, value: Any) -> str:
    return hashlib.sha256(
        _canonical_bytes({"action": action, "request": value})
    ).hexdigest()


async def _matching_receipt(
    session: AsyncSession,
    *,
    context: OrganizationContext,
    action: str,
    intent_id: str,
    idempotency_key: str,
    request_fingerprint: str,
) -> EventTemplateOperatorReceipt | None:
    receipt = (
        (
            await session.execute(
                select(EventTemplateOperatorReceipt).where(
                    or_(
                        col(EventTemplateOperatorReceipt.intent_id) == intent_id,
                        col(EventTemplateOperatorReceipt.idempotency_key)
                        == idempotency_key,
                    )
                )
            )
        )
        .scalars()
        .first()
    )
    if receipt is None:
        return None
    if (
        receipt.organization_id != context.organization_id
        or receipt.user_id != context.user_id
        or receipt.action != action
        or receipt.request_fingerprint != request_fingerprint
    ):
        raise EventTemplateError(
            "Operator intent or idempotency key was already used for another request"
        )
    return receipt


def _receipt_response(
    receipt: EventTemplateOperatorReceipt, *, replayed: bool
) -> dict[str, Any]:
    return {**dict(receipt.receipt), "replayed": replayed}


def _new_receipt(
    *,
    context: OrganizationContext,
    action: str,
    body: OperatorMutationInput,
    request_fingerprint: str,
    template_id: str,
    release_id: str,
    status: str,
    result: dict[str, Any],
) -> EventTemplateOperatorReceipt:
    receipt = EventTemplateOperatorReceipt(
        organization_id=context.organization_id,
        user_id=context.user_id,
        action=action,
        intent_id=body.intent_id,
        idempotency_key=body.idempotency_key,
        request_fingerprint=request_fingerprint,
        template_id=template_id,
        release_id=release_id,
        reason=body.reason,
        status=status,
    )
    audit = {
        "id": receipt.id,
        "action": action,
        "intent_id": body.intent_id,
        "actor_id": context.user_id,
        "organization_id": context.organization_id,
        "reason": body.reason,
        "request_fingerprint": request_fingerprint,
        "created_at": receipt.created_at.isoformat(),
    }
    receipt.receipt = {**result, "audit_receipt": audit}
    return receipt


async def import_template_release(
    session: AsyncSession,
    *,
    context: OrganizationContext,
    body: TemplateReleaseImportInput,
) -> dict[str, Any]:
    """Create or optimistic-concurrency replace one unpublished release draft."""
    ensure_catalog_operator(context)
    request = body.model_dump(mode="json")
    if len(_canonical_bytes(request)) > MAX_TEMPLATE_SNAPSHOT_BYTES:
        raise EventTemplateError("Template import exceeds the host limit")
    request_fingerprint = _fingerprint("event_template.release.imported", request)
    replay = await _matching_receipt(
        session,
        context=context,
        action="event_template.release.imported",
        intent_id=body.intent_id,
        idempotency_key=body.idempotency_key,
        request_fingerprint=request_fingerprint,
    )
    if replay is not None:
        return _receipt_response(replay, replayed=True)

    template = (
        (
            await session.execute(
                select(EventTemplate).where(EventTemplate.slug == body.template.slug)
            )
        )
        .scalars()
        .first()
    )
    if template is None:
        template = EventTemplate(
            slug=body.template.slug,
            title=body.template.title,
            summary=body.template.summary,
            discovery_group=body.template.discovery_group,
            status="draft",
        )
        session.add(template)
        await session.flush()

    release = (
        (
            await session.execute(
                select(EventTemplateRelease).where(
                    EventTemplateRelease.template_id == template.id,
                    EventTemplateRelease.version == body.version,
                )
            )
        )
        .scalars()
        .first()
    )
    if release is not None:
        if release.status != "draft":
            raise EventTemplateError("Published template releases are immutable")
        if body.expected_draft_checksum is None:
            raise EventTemplateError(
                "Replacing a draft requires expected_draft_checksum"
            )
        if release.checksum != body.expected_draft_checksum:
            raise EventTemplateError("Template draft changed; reload it and retry")
        await session.execute(
            delete(EventTemplateAsset).where(
                col(EventTemplateAsset.release_id) == release.id
            )
        )
        await session.execute(
            delete(EventTemplateCompatibility).where(
                col(EventTemplateCompatibility.release_id) == release.id
            )
        )
        await session.execute(
            delete(EventTemplateSeed).where(
                col(EventTemplateSeed.release_id) == release.id
            )
        )
        await session.execute(
            delete(EventTemplateGuidedJob).where(
                col(EventTemplateGuidedJob.release_id) == release.id
            )
        )
        await session.flush()
    else:
        if body.expected_draft_checksum is not None:
            raise EventTemplateError("Template draft does not exist")
        release = EventTemplateRelease(
            template_id=template.id,
            version=body.version,
        )
        session.add(release)
        await session.flush()

    catalog_snapshot = body.template.model_dump(mode="json")
    release.schema_version = TEMPLATE_SCHEMA_VERSION
    release.status = "draft"
    release.release_notes = body.release_notes
    release.catalog_snapshot = catalog_snapshot
    release.checksum = ""
    session.add(release)
    session.add_all(
        [
            EventTemplateAsset(
                release_id=release.id,
                asset_key=item.asset_key,
                kind=item.kind,
                storage_key=item.storage_key,
                content_type=item.content_type,
                sha256=item.sha256,
                size_bytes=item.size_bytes,
                metadata_=item.metadata,
            )
            for item in body.assets
        ]
    )
    session.add_all(
        [
            EventTemplateCompatibility(
                release_id=release.id,
                requirement_key=item.requirement_key,
                requirement=item.requirement.model_dump(mode="json"),
                required=item.required,
            )
            for item in body.compatibility
        ]
    )
    session.add_all(
        [
            EventTemplateSeed(
                release_id=release.id,
                seed_key=item.seed_key,
                kind=item.kind,
                ordinal=item.ordinal,
                payload=item.payload,
            )
            for item in body.seeds
        ]
    )
    session.add_all(
        [
            EventTemplateGuidedJob(
                release_id=release.id,
                job_key=item.job_key,
                title=item.title,
                instructions=item.instructions,
                definition_of_done=item.definition_of_done,
                priority=item.priority,
                execution_profile=item.execution_profile,
                ordinal=item.ordinal,
                materialize_task=item.materialize_task,
            )
            for item in body.guided_jobs
        ]
    )
    await session.flush()
    release.checksum = await compute_template_release_checksum(session, release.id)
    session.add(release)
    await session.flush()
    validated = await validate_template_release(session, release.id)
    result = {
        "template": {
            "id": template.id,
            **catalog_snapshot,
            "status": template.status,
        },
        "release": {
            "id": release.id,
            "version": release.version,
            "status": release.status,
            "checksum": validated.checksum,
            "schema_version": release.schema_version,
            "validation": "passed",
        },
    }
    receipt = _new_receipt(
        context=context,
        action="event_template.release.imported",
        body=body,
        request_fingerprint=request_fingerprint,
        template_id=template.id,
        release_id=release.id,
        status="draft_imported",
        result=result,
    )
    session.add(receipt)
    await session.flush()
    return _receipt_response(receipt, replayed=False)


async def publish_template_release(
    session: AsyncSession,
    *,
    context: OrganizationContext,
    release_id: str,
    body: TemplateReleasePublishInput,
) -> dict[str, Any]:
    """Publish one exact validated draft and atomically advance the catalog."""
    ensure_catalog_operator(context)
    request = {"release_id": release_id, **body.model_dump(mode="json")}
    request_fingerprint = _fingerprint("event_template.release.published", request)
    replay = await _matching_receipt(
        session,
        context=context,
        action="event_template.release.published",
        intent_id=body.intent_id,
        idempotency_key=body.idempotency_key,
        request_fingerprint=request_fingerprint,
    )
    if replay is not None:
        return _receipt_response(replay, replayed=True)

    release = await session.get(EventTemplateRelease, release_id)
    if release is None:
        raise EventTemplateError("Template release not found", status_code=404)
    if release.status != "draft":
        raise EventTemplateError("Only a draft template release can be published")
    if release.checksum != body.expected_checksum:
        raise EventTemplateError("Template draft changed; reload it and retry")
    validated = await validate_template_release(session, release.id)
    await _verify_release_assets(validated.assets)
    catalog = CatalogIdentityInput.model_validate(release.catalog_snapshot)
    if catalog.discovery_group not in DISCOVERY_GROUPS:
        raise EventTemplateError("Template has an invalid discovery group")
    template = await session.get(EventTemplate, release.template_id)
    if template is None:
        raise EventTemplateError("Event template not found", status_code=404)
    current_pointer_matches = (
        col(EventTemplate.current_release_id).is_(None)
        if body.expected_current_release_id is None
        else col(EventTemplate.current_release_id) == body.expected_current_release_id
    )
    advanced = cast(
        CursorResult[Any],
        await session.execute(
            update(EventTemplate)
            .where(
                col(EventTemplate.id) == template.id,
                current_pointer_matches,
            )
            .values(
                slug=catalog.slug,
                title=catalog.title,
                summary=catalog.summary,
                discovery_group=catalog.discovery_group,
                status="published",
                current_release_id=release.id,
                updated_at=datetime.now(timezone.utc),
            )
        ),
    )
    if advanced.rowcount != 1:
        raise EventTemplateError(
            "Template catalog changed; reload it before publishing"
        )
    release.status = "published"
    release.published_at = datetime.now(timezone.utc)
    session.add(release)
    await session.refresh(template)
    result = {
        "template": {
            "id": template.id,
            **catalog.model_dump(mode="json"),
            "status": template.status,
        },
        "release": {
            "id": release.id,
            "version": release.version,
            "status": release.status,
            "checksum": validated.checksum,
            "schema_version": release.schema_version,
            "published_at": release.published_at.isoformat(),
        },
    }
    receipt = _new_receipt(
        context=context,
        action="event_template.release.published",
        body=body,
        request_fingerprint=request_fingerprint,
        template_id=template.id,
        release_id=release.id,
        status="published",
        result=result,
    )
    session.add(receipt)
    await session.flush()
    return _receipt_response(receipt, replayed=False)


async def operator_release_payload(
    session: AsyncSession, release: EventTemplateRelease
) -> dict[str, Any]:
    assets, compatibility, seeds, guided_jobs = await load_template_release_rows(
        session, release.id
    )
    validation = "passed"
    validation_error: str | None = None
    try:
        await validate_template_release(session, release.id)
    except EventTemplateError as exc:
        validation = "failed"
        validation_error = exc.detail
    return {
        "id": release.id,
        "template_id": release.template_id,
        "version": release.version,
        "schema_version": release.schema_version,
        "status": release.status,
        "checksum": release.checksum,
        "release_notes": release.release_notes,
        "catalog": release.catalog_snapshot,
        "content": template_release_content(
            release, assets, compatibility, seeds, guided_jobs
        ),
        "validation": validation,
        "validation_error": validation_error,
        "published_at": (
            release.published_at.isoformat() if release.published_at else None
        ),
        "created_at": release.created_at.isoformat(),
    }


async def list_operator_releases(session: AsyncSession) -> list[dict[str, Any]]:
    releases = list(
        (
            await session.execute(
                select(EventTemplateRelease).order_by(
                    col(EventTemplateRelease.created_at).desc(),
                    col(EventTemplateRelease.id).desc(),
                )
            )
        )
        .scalars()
        .all()
    )
    return [await operator_release_payload(session, release) for release in releases]


__all__ = [
    "TemplateReleaseImportInput",
    "TemplateReleasePublishInput",
    "ensure_catalog_operator",
    "import_template_release",
    "list_operator_releases",
    "operator_release_payload",
    "publish_template_release",
]
