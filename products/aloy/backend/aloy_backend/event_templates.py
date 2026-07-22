"""Generic, database-backed Event template discovery and installation.

The runtime knows template contracts, never template domains. A published
release is validated and snapshotted before one transaction materializes
ordinary tenant-owned Event, context, Task, and Surface records.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from .event_context import refresh_event_context_snapshot
from .events import ensure_event_conversation
from .models import (
    Event,
    EventTemplate,
    EventTemplateAsset,
    EventTemplateCompatibility,
    EventTemplateGuidedJob,
    EventTemplateInstallation,
    EventTemplateRelease,
    EventTemplateSeed,
    EventTrailEntry,
    KnowledgeEntry,
    Organization,
    Run,
    SurfaceDataRecord,
    SurfaceProject,
    SurfaceRevision,
    Task,
)
from .surface_authoring import (
    MAX_SURFACE_FILES,
    MAX_SURFACE_SOURCE_BYTES,
    SurfaceFilePatch,
    SurfaceWriteFilesParams,
    surface_source_path,
)
from .surface_manifest import SURFACE_SDK_VERSION, parse_surface_manifest
from .surface_materialization import (
    SURFACE_MATERIALIZATION_RUN_KIND,
    queue_surface_revision_materialization,
)
from .tenancy import OrganizationPolicy

TEMPLATE_SCHEMA_VERSION = 1
MAX_TEMPLATE_SNAPSHOT_BYTES = 2 * 1024 * 1024
DISCOVERY_GROUPS = frozenset(
    {"student", "individual", "professional", "team", "business"}
)
SEED_KINDS = frozenset({"event", "context", "surface", "surface_data"})
HOST_COMPATIBILITY: dict[str, str | int | bool] = {
    "event_template_schema": TEMPLATE_SCHEMA_VERSION,
    "surface_sdk": SURFACE_SDK_VERSION,
}


class EventTemplateError(ValueError):
    def __init__(self, detail: str, *, status_code: int = 409) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class EventSeedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=300)
    summary: str = Field(default="", max_length=2_000)
    phase: str = Field(default="", max_length=200)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TemplateKnowledgeEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    key: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=1, max_length=16_000)
    tags: list[str] = Field(default_factory=list, max_length=30)


class ContextSeedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entries: list[TemplateKnowledgeEntry] = Field(default_factory=list, max_length=50)
    setup_gaps: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("setup_gaps")
    @classmethod
    def validate_setup_gaps(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value or len(value) > 500 for value in cleaned):
            raise ValueError("Template setup gaps must be 1-500 characters")
        return cleaned


class SurfaceSeedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sdk_version: Literal["1"] = SURFACE_SDK_VERSION
    files: dict[str, str] = Field(min_length=1, max_length=MAX_SURFACE_FILES)


class SurfaceDataSeedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    namespace: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_.-]*$")
    key: str = Field(min_length=1, max_length=200)
    data: dict[str, Any] = Field(default_factory=dict)
    posture: Literal["sample", "setup_gap"] = "sample"


class CompatibilityPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operator: Literal["eq"] = "eq"
    value: str | int | bool


class AssetPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    asset_key: str = Field(min_length=1, max_length=200)
    kind: str = Field(min_length=1, max_length=100)
    storage_key: str = Field(min_length=1, max_length=1_000)
    content_type: str = Field(min_length=1, max_length=200)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(ge=0, le=100 * 1024 * 1024)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GuidedJobPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    job_key: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=1_000)
    instructions: str = Field(default="", max_length=50_000)
    definition_of_done: str = Field(default="", max_length=10_000)
    priority: Literal["low", "normal", "high", "urgent"] = "normal"
    execution_profile: Literal["general", "sourced_research"] = "general"
    ordinal: int = Field(ge=0, le=10_000)
    materialize_task: bool = True


@dataclass(frozen=True)
class TemplateReleaseBundle:
    template: EventTemplate
    release: EventTemplateRelease
    assets: tuple[EventTemplateAsset, ...]
    compatibility: tuple[EventTemplateCompatibility, ...]
    seeds: tuple[EventTemplateSeed, ...]
    guided_jobs: tuple[EventTemplateGuidedJob, ...]
    snapshot: dict[str, Any]


@dataclass(frozen=True)
class TemplateInstallResult:
    installation: EventTemplateInstallation
    event: Event
    surface_run_id: str | None
    replayed: bool


@dataclass(frozen=True)
class ValidatedTemplateRelease:
    release: EventTemplateRelease
    assets: tuple[EventTemplateAsset, ...]
    compatibility: tuple[EventTemplateCompatibility, ...]
    seeds: tuple[EventTemplateSeed, ...]
    guided_jobs: tuple[EventTemplateGuidedJob, ...]
    checksum: str


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def template_release_content(
    release: EventTemplateRelease,
    assets: tuple[EventTemplateAsset, ...],
    compatibility: tuple[EventTemplateCompatibility, ...],
    seeds: tuple[EventTemplateSeed, ...],
    guided_jobs: tuple[EventTemplateGuidedJob, ...],
) -> dict[str, Any]:
    return {
        "schema_version": release.schema_version,
        "template_id": release.template_id,
        "version": release.version,
        "release_notes": release.release_notes,
        "catalog": release.catalog_snapshot,
        "assets": [
            {
                "asset_key": item.asset_key,
                "kind": item.kind,
                "storage_key": item.storage_key,
                "content_type": item.content_type,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
                "metadata": item.metadata_,
            }
            for item in assets
        ],
        "compatibility": [
            {
                "requirement_key": item.requirement_key,
                "requirement": item.requirement,
                "required": item.required,
            }
            for item in compatibility
        ],
        "seeds": [
            {
                "seed_key": item.seed_key,
                "kind": item.kind,
                "ordinal": item.ordinal,
                "payload": item.payload,
            }
            for item in seeds
        ],
        "guided_jobs": [
            {
                "job_key": item.job_key,
                "title": item.title,
                "instructions": item.instructions,
                "definition_of_done": item.definition_of_done,
                "priority": item.priority,
                "execution_profile": item.execution_profile,
                "ordinal": item.ordinal,
                "materialize_task": item.materialize_task,
            }
            for item in guided_jobs
        ],
    }


async def load_template_release_rows(session: AsyncSession, release_id: str) -> tuple[
    tuple[EventTemplateAsset, ...],
    tuple[EventTemplateCompatibility, ...],
    tuple[EventTemplateSeed, ...],
    tuple[EventTemplateGuidedJob, ...],
]:
    async def rows(model, *order):
        return tuple(
            (
                await session.execute(
                    select(model).where(model.release_id == release_id).order_by(*order)
                )
            )
            .scalars()
            .all()
        )

    return (
        await rows(EventTemplateAsset, col(EventTemplateAsset.asset_key)),
        await rows(
            EventTemplateCompatibility,
            col(EventTemplateCompatibility.requirement_key),
        ),
        await rows(
            EventTemplateSeed,
            col(EventTemplateSeed.ordinal),
            col(EventTemplateSeed.seed_key),
        ),
        await rows(
            EventTemplateGuidedJob,
            col(EventTemplateGuidedJob.ordinal),
            col(EventTemplateGuidedJob.job_key),
        ),
    )


async def compute_template_release_checksum(
    session: AsyncSession, release_id: str
) -> str:
    """Compute the immutable fingerprint an authoring plane signs off on."""
    release = await session.get(EventTemplateRelease, release_id)
    if release is None:
        raise EventTemplateError("Template release not found", status_code=404)
    assets, compatibility, seeds, guided_jobs = await load_template_release_rows(
        session, release.id
    )
    content = template_release_content(
        release, assets, compatibility, seeds, guided_jobs
    )
    return hashlib.sha256(_canonical_bytes(content)).hexdigest()


def _validate_compatibility(
    compatibility: tuple[EventTemplateCompatibility, ...],
) -> None:
    for row in compatibility:
        requirement = CompatibilityPayload.model_validate(row.requirement)
        actual = HOST_COMPATIBILITY.get(row.requirement_key)
        if actual is None:
            if row.required:
                raise EventTemplateError(
                    f"Host does not support required template capability: "
                    f"{row.requirement_key}"
                )
            continue
        if actual != requirement.value and row.required:
            raise EventTemplateError(
                f"Template requires {row.requirement_key}={requirement.value}; "
                f"host provides {actual}"
            )


def _validated_seed_payloads(
    seeds: tuple[EventTemplateSeed, ...],
) -> tuple[
    EventSeedPayload,
    ContextSeedPayload | None,
    SurfaceSeedPayload | None,
    tuple[SurfaceDataSeedPayload, ...],
]:
    unknown = sorted({item.kind for item in seeds}.difference(SEED_KINDS))
    if unknown:
        raise EventTemplateError(
            "Template release contains unsupported seed kinds: " + ", ".join(unknown)
        )
    event_rows = [item for item in seeds if item.kind == "event"]
    context_rows = [item for item in seeds if item.kind == "context"]
    surface_rows = [item for item in seeds if item.kind == "surface"]
    if len(event_rows) != 1:
        raise EventTemplateError("Template release must contain exactly one event seed")
    if len(context_rows) > 1 or len(surface_rows) > 1:
        raise EventTemplateError(
            "Template release may contain at most one context and one Surface seed"
        )
    event_seed = EventSeedPayload.model_validate(event_rows[0].payload)
    context_seed = (
        ContextSeedPayload.model_validate(context_rows[0].payload)
        if context_rows
        else None
    )
    surface_seed = (
        SurfaceSeedPayload.model_validate(surface_rows[0].payload)
        if surface_rows
        else None
    )
    data_seeds = tuple(
        SurfaceDataSeedPayload.model_validate(item.payload)
        for item in seeds
        if item.kind == "surface_data"
    )
    data_keys = [(item.namespace, item.key) for item in data_seeds]
    if len(data_keys) != len(set(data_keys)):
        raise EventTemplateError("Template Surface data keys must be unique")
    if (
        context_seed is not None
        and context_seed.setup_gaps
        and ("setup", "gaps") in data_keys
    ):
        raise EventTemplateError("Template setup gaps reserve Surface key setup/gaps")
    return event_seed, context_seed, surface_seed, data_seeds


def _validate_surface_source(seed: SurfaceSeedPayload) -> tuple[dict[str, str], dict]:
    patches = [
        SurfaceFilePatch(path=f"/workspace{path}", content=content)
        for path, content in seed.files.items()
    ]
    SurfaceWriteFilesParams(
        idempotency_key="template-source-validation",
        patches=patches,
    )
    files = {
        surface_source_path(f"/workspace{path}"): content
        for path, content in seed.files.items()
    }
    if "/src/App.tsx" not in files:
        raise EventTemplateError("Template Surface source requires /src/App.tsx")
    total_bytes = sum(len(value.encode("utf-8")) for value in files.values())
    if total_bytes > MAX_SURFACE_SOURCE_BYTES:
        raise EventTemplateError("Template Surface source exceeds the host limit")
    manifest = parse_surface_manifest(files, sdk_version=seed.sdk_version)
    return files, manifest.model_dump(mode="json", by_alias=True)


async def validate_template_release(
    session: AsyncSession, release_id: str
) -> ValidatedTemplateRelease:
    """Validate one stored release without granting it publication authority."""
    release = await session.get(EventTemplateRelease, release_id)
    if release is None:
        raise EventTemplateError("Template release not found", status_code=404)
    if release.schema_version != TEMPLATE_SCHEMA_VERSION:
        raise EventTemplateError("Template release schema is unsupported")
    if release.version < 1:
        raise EventTemplateError("Template release version must be positive")
    assets, compatibility, seeds, guided_jobs = await load_template_release_rows(
        session, release.id
    )
    expected = hashlib.sha256(
        _canonical_bytes(
            template_release_content(release, assets, compatibility, seeds, guided_jobs)
        )
    ).hexdigest()
    if not release.checksum or release.checksum != expected:
        raise EventTemplateError("Template release failed integrity validation")
    try:
        _validate_compatibility(compatibility)
        _, _, surface_seed, data_seeds = _validated_seed_payloads(seeds)
        for asset in assets:
            AssetPayload.model_validate(
                {
                    "asset_key": asset.asset_key,
                    "kind": asset.kind,
                    "storage_key": asset.storage_key,
                    "content_type": asset.content_type,
                    "sha256": asset.sha256,
                    "size_bytes": asset.size_bytes,
                    "metadata": asset.metadata_,
                }
            )
        for job in guided_jobs:
            GuidedJobPayload.model_validate(
                {
                    "job_key": job.job_key,
                    "title": job.title,
                    "instructions": job.instructions,
                    "definition_of_done": job.definition_of_done,
                    "priority": job.priority,
                    "execution_profile": job.execution_profile,
                    "ordinal": job.ordinal,
                    "materialize_task": job.materialize_task,
                }
            )
        if surface_seed is not None:
            _validate_surface_source(surface_seed)
        if data_seeds and surface_seed is None:
            raise EventTemplateError("Surface data seeds require a Surface source seed")
    except EventTemplateError:
        raise
    except ValueError as exc:
        raise EventTemplateError(
            f"Template release failed contract validation: {exc}"
        ) from exc
    return ValidatedTemplateRelease(
        release=release,
        assets=assets,
        compatibility=compatibility,
        seeds=seeds,
        guided_jobs=guided_jobs,
        checksum=expected,
    )


async def load_published_release(
    session: AsyncSession,
    *,
    template_id: str,
    release_id: str | None = None,
) -> TemplateReleaseBundle:
    template = await session.get(EventTemplate, template_id)
    if template is None or template.status != "published":
        raise EventTemplateError("Event template not found", status_code=404)
    if template.discovery_group not in DISCOVERY_GROUPS:
        raise EventTemplateError("Template has an invalid discovery group")
    selected_release_id = release_id or template.current_release_id
    if not selected_release_id:
        raise EventTemplateError("Template has no published release")
    release = await session.get(EventTemplateRelease, selected_release_id)
    if (
        release is None
        or release.template_id != template.id
        or release.status != "published"
    ):
        raise EventTemplateError("Template release not found", status_code=404)
    validated = await validate_template_release(session, release.id)
    assets = validated.assets
    compatibility = validated.compatibility
    seeds = validated.seeds
    guided_jobs = validated.guided_jobs
    snapshot = {
        "template": {
            "id": template.id,
            "slug": template.slug,
            "title": template.title,
            "summary": template.summary,
            "discovery_group": template.discovery_group,
        },
        "release_id": release.id,
        "release_checksum": release.checksum,
        **template_release_content(release, assets, compatibility, seeds, guided_jobs),
    }
    if len(_canonical_bytes(snapshot)) > MAX_TEMPLATE_SNAPSHOT_BYTES:
        raise EventTemplateError("Template release snapshot exceeds the host limit")
    return TemplateReleaseBundle(
        template=template,
        release=release,
        assets=assets,
        compatibility=compatibility,
        seeds=seeds,
        guided_jobs=guided_jobs,
        snapshot=snapshot,
    )


async def find_template_installation(
    session: AsyncSession,
    *,
    organization_id: str,
    user_id: str,
    idempotency_key: str,
) -> EventTemplateInstallation | None:
    return (
        (
            await session.execute(
                select(EventTemplateInstallation).where(
                    EventTemplateInstallation.organization_id == organization_id,
                    EventTemplateInstallation.user_id == user_id,
                    EventTemplateInstallation.idempotency_key == idempotency_key,
                )
            )
        )
        .scalars()
        .first()
    )


async def _replay_installation(
    session: AsyncSession,
    *,
    installation: EventTemplateInstallation,
    template_id: str,
    release_id: str | None,
) -> TemplateInstallResult:
    if installation.template_id != template_id or (
        release_id is not None and installation.release_id != release_id
    ):
        raise EventTemplateError(
            "idempotency_key was already used for another template installation"
        )
    event = await session.get(Event, installation.event_id)
    if event is None:
        raise EventTemplateError("Installed Event provenance is incomplete")
    surface_run_id = await _ensure_installed_surface_materialization(
        session,
        event=event,
        installation=installation,
    )
    return TemplateInstallResult(
        installation=installation,
        event=event,
        surface_run_id=surface_run_id,
        replayed=True,
    )


async def _ensure_installed_surface_materialization(
    session: AsyncSession,
    *,
    event: Event,
    installation: EventTemplateInstallation,
) -> str | None:
    existing = (
        (
            await session.execute(
                select(Run)
                .where(
                    Run.organization_id == event.organization_id,
                    Run.user_id == event.user_id,
                    Run.event_id == event.id,
                    Run.run_kind == SURFACE_MATERIALIZATION_RUN_KIND,
                )
                .order_by(col(Run.created_at))
                .limit(1)
            )
        )
        .scalars()
        .first()
    )
    if existing is not None:
        return existing.id
    project = (
        (
            await session.execute(
                select(SurfaceProject).where(
                    SurfaceProject.organization_id == event.organization_id,
                    SurfaceProject.user_id == event.user_id,
                    SurfaceProject.event_id == event.id,
                )
            )
        )
        .scalars()
        .first()
    )
    if (
        project is None
        or project.published_build_id is not None
        or project.draft_revision_id is None
    ):
        return None
    revision = await session.get(SurfaceRevision, project.draft_revision_id)
    if revision is None:
        raise EventTemplateError("Installed Surface source is incomplete")
    organization = await session.get(Organization, event.organization_id)
    policy = OrganizationPolicy.model_validate(
        organization.policy if organization is not None else {}
    )
    run, _ = await queue_surface_revision_materialization(
        session,
        event=event,
        project=project,
        revision=revision,
        policy=policy,
        trigger="event_template_installation",
        actor_id=event.user_id,
        origin_evidence=[
            {"template_installation_id": installation.id},
            {"template_release_id": installation.release_id},
        ],
    )
    return run.id


async def install_event_template(
    session: AsyncSession,
    *,
    organization_id: str,
    user_id: str,
    template_id: str,
    idempotency_key: str,
    release_id: str | None = None,
    title: str | None = None,
) -> TemplateInstallResult:
    """Atomically materialize one published release as ordinary Event state."""
    existing = await find_template_installation(
        session,
        organization_id=organization_id,
        user_id=user_id,
        idempotency_key=idempotency_key,
    )
    if existing is not None:
        return await _replay_installation(
            session,
            installation=existing,
            template_id=template_id,
            release_id=release_id,
        )

    bundle = await load_published_release(
        session,
        template_id=template_id,
        release_id=release_id,
    )
    event_seed, context_seed, surface_seed, data_seeds = _validated_seed_payloads(
        bundle.seeds
    )
    installation = EventTemplateInstallation(
        organization_id=organization_id,
        user_id=user_id,
        template_id=bundle.template.id,
        release_id=bundle.release.id,
        event_id="pending",
        idempotency_key=idempotency_key,
        release_snapshot=bundle.snapshot,
    )
    event_metadata = dict(event_seed.metadata)
    event_metadata["template"] = {
        "installation_id": installation.id,
        "template_id": bundle.template.id,
        "release_id": bundle.release.id,
        "version": bundle.release.version,
        "checksum": bundle.release.checksum,
    }
    event = Event(
        organization_id=organization_id,
        user_id=user_id,
        type="project",
        title=title or event_seed.title,
        lifecycle="active",
        phase=event_seed.phase,
        summary=event_seed.summary,
        metadata_=event_metadata,
    )
    session.add(event)
    await session.flush()
    conversation = await ensure_event_conversation(session, event=event)
    installation.event_id = event.id
    session.add(installation)

    if context_seed is not None:
        for entry in context_seed.entries:
            session.add(
                KnowledgeEntry(
                    organization_id=organization_id,
                    user_id=user_id,
                    event_id=event.id,
                    content=entry.content,
                    tags=[*entry.tags, "event-template"],
                    # Template context is declarative knowledge. Its origin is
                    # carried by tags/provenance rather than widening the
                    # kernel's semantic/episodic/procedural memory taxonomy.
                    kind="semantic",
                    confidence=1.0,
                    sensitivity="internal",
                    source="admin",
                    provenance={
                        "kind": "event_template",
                        "installation_id": installation.id,
                        "release_id": bundle.release.id,
                        "seed_key": entry.key,
                    },
                    conflict_key=f"template:{installation.id}:{entry.key}",
                )
            )

    project: SurfaceProject | None = None
    if surface_seed is not None:
        files, manifest = _validate_surface_source(surface_seed)
        source_checksum = hashlib.sha256(
            _canonical_bytes({"manifest": manifest, "files": files})
        ).hexdigest()
        total_bytes = sum(len(value.encode("utf-8")) for value in files.values())
        project = SurfaceProject(
            organization_id=organization_id,
            user_id=user_id,
            event_id=event.id,
            sdk_version=surface_seed.sdk_version,
            data_revision=0,
            lifecycle="draft",
        )
        session.add(project)
        await session.flush()
        revision = SurfaceRevision(
            organization_id=organization_id,
            user_id=user_id,
            event_id=event.id,
            project_id=project.id,
            revision_number=1,
            idempotency_key=f"template:{installation.id}:source",
            request_fingerprint=bundle.release.checksum,
            manifest=manifest,
            files=files,
            checksum=source_checksum,
            file_count=len(files),
            total_bytes=total_bytes,
        )
        session.add(revision)
        await session.flush()
        project.draft_revision_id = revision.id
        session.add(project)

    if data_seeds and project is None:
        raise EventTemplateError("Surface data seeds require a Surface source seed")
    for index, data_seed in enumerate(data_seeds, start=1):
        assert project is not None
        session.add(
            SurfaceDataRecord(
                organization_id=organization_id,
                user_id=user_id,
                event_id=event.id,
                project_id=project.id,
                namespace=data_seed.namespace,
                record_key=data_seed.key,
                data=data_seed.data,
                revision=1,
                posture=f"template_{data_seed.posture}",
                actor_id=user_id,
                provenance={
                    "kind": "event_template",
                    "installation_id": installation.id,
                    "release_id": bundle.release.id,
                    "template_posture": data_seed.posture,
                },
            )
        )
        project.data_revision = index
    if project is not None:
        session.add(project)

    for job in bundle.guided_jobs:
        if not job.materialize_task:
            continue
        session.add(
            Task(
                organization_id=organization_id,
                user_id=user_id,
                event_id=event.id,
                origin_conversation_id=conversation.id,
                title=job.title,
                instructions=job.instructions,
                definition_of_done=job.definition_of_done,
                priority=job.priority,
                execution_profile=job.execution_profile,
                order=job.ordinal,
                created_by=user_id,
            )
        )

    if context_seed is not None and context_seed.setup_gaps and project is not None:
        session.add(
            SurfaceDataRecord(
                organization_id=organization_id,
                user_id=user_id,
                event_id=event.id,
                project_id=project.id,
                namespace="setup",
                record_key="gaps",
                data={"items": context_seed.setup_gaps},
                revision=1,
                posture="template_setup_gap",
                actor_id=user_id,
                provenance={
                    "kind": "event_template",
                    "installation_id": installation.id,
                    "release_id": bundle.release.id,
                    "template_posture": "setup_gap",
                },
            )
        )
        project.data_revision += 1
        session.add(project)

    session.add(
        EventTrailEntry(
            organization_id=organization_id,
            user_id=user_id,
            event_id=event.id,
            actor_id=user_id,
            kind="event_template_installed",
            summary=f"Started {event.title} from {bundle.template.title}",
            evidence_refs=[
                {
                    "template_id": bundle.template.id,
                    "release_id": bundle.release.id,
                    "release_checksum": bundle.release.checksum,
                }
            ],
            payload={
                "installation_id": installation.id,
                "template_id": bundle.template.id,
                "release_id": bundle.release.id,
                "version": bundle.release.version,
            },
        )
    )
    await session.flush()
    await refresh_event_context_snapshot(
        session,
        organization_id=organization_id,
        user_id=user_id,
        event_id=event.id,
    )
    await session.flush()
    surface_run_id = await _ensure_installed_surface_materialization(
        session,
        event=event,
        installation=installation,
    )
    return TemplateInstallResult(
        installation=installation,
        event=event,
        surface_run_id=surface_run_id,
        replayed=False,
    )


__all__ = [
    "DISCOVERY_GROUPS",
    "EventTemplateError",
    "TemplateInstallResult",
    "compute_template_release_checksum",
    "find_template_installation",
    "install_event_template",
    "load_template_release_rows",
    "load_published_release",
    "template_release_content",
    "validate_template_release",
]
