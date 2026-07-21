"""Durable orchestration for deterministic validation and isolated builds."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from pori import RunContext

from .database import async_session
from .models import (
    Event,
    EventTrailEntry,
    SurfaceBuild,
    SurfaceEvidenceArtifact,
    SurfaceInspection,
    SurfaceProject,
    SurfaceRevision,
)
from .storage import (
    ObjectStore,
    get_object_store,
    surface_bundle_key,
    surface_preview_artifact_key,
)
from .surface_authoring import SurfaceAuthoringError, SurfaceConflictError
from .surface_build_runner import (
    MAX_SURFACE_BUILD_LOG_CHARS,
    MAX_SURFACE_BUNDLE_BYTES,
    SurfaceBuildRunner,
    SurfaceBuildRunnerResult,
    configured_surface_build_runner,
    validate_surface_source,
)
from .surface_inspection_transport import (
    LocalSurfaceInspectionTransport,
    SurfaceInspectionArtifact,
    SurfaceInspectionRequest,
    SurfaceInspectionTransport,
    configured_surface_inspection_transport,
    new_surface_inspection_binding,
    validate_surface_inspection_result,
)
from .surface_manifest import SurfaceManifest
from .surface_publication import (
    SurfacePublicationParams,
    change_surface_publication,
)
from .surface_quality import (
    SURFACE_QUALITY_RECEIPT_KEY,
    create_surface_quality_receipt,
    surface_quality_receipt_error,
)
from .surface_runtime import build_surface_runtime_document


class SurfaceBuildParams(BaseModel):
    revision_id: str = Field(min_length=1, max_length=200)
    idempotency_key: str = Field(min_length=8, max_length=200)

    @field_validator("revision_id", "idempotency_key")
    @classmethod
    def validate_trimmed(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("Surface build identifiers must be trimmed")
        return value


class SurfacePreviewParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    build_id: str | None = Field(default=None, max_length=200)


def _build_fingerprint(revision: SurfaceRevision, toolchain_version: str) -> str:
    encoded = json.dumps(
        {
            "revision_id": revision.id,
            "source_checksum": revision.checksum,
            "toolchain_version": toolchain_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _public_preview_artifacts(values: list[dict]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for value in values[:50]:
        if not isinstance(value, dict):
            continue
        try:
            size_bytes = max(0, int(value.get("size_bytes") or 0))
        except (TypeError, ValueError):
            size_bytes = 0
        artifact = {
            "kind": str(value.get("kind") or "preview"),
            "name": str(value.get("name") or "")[:200],
            "content_type": str(value.get("content_type") or "")[:200],
            "sha256": str(value.get("sha256") or "")[:128],
            "size_bytes": size_bytes,
        }
        artifacts.append(artifact)
    return artifacts


def surface_build_payload(
    build: SurfaceBuild,
    *,
    include_log: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": build.id,
        "event_id": build.event_id,
        "project_id": build.project_id,
        "revision_id": build.revision_id,
        "creator_run_id": build.creator_run_id,
        "status": build.status,
        "source_checksum": build.source_checksum,
        "toolchain_version": build.toolchain_version,
        "validation_result": build.validation_result,
        "diagnostics": build.diagnostics,
        "bundle_available": bool(build.bundle_key),
        "bundle_sha256": build.bundle_sha256,
        "bundle_size_bytes": build.bundle_size_bytes,
        "preview_artifacts": build.preview_artifacts,
        "resource_metrics": build.resource_metrics,
        "created_at": build.created_at,
        "started_at": build.started_at,
        "completed_at": build.completed_at,
    }
    if include_log:
        payload["build_log"] = build.build_log
    return payload


async def _owned_project(
    session: AsyncSession,
    *,
    organization_id: str,
    user_id: str,
    event_id: str,
) -> SurfaceProject | None:
    result = await session.execute(
        select(SurfaceProject).where(
            SurfaceProject.organization_id == organization_id,
            SurfaceProject.user_id == user_id,
            SurfaceProject.event_id == event_id,
        )
    )
    return result.scalars().first()


async def _owned_build(
    session: AsyncSession,
    build_id: str,
    *,
    organization_id: str,
    user_id: str,
    event_id: str,
    project_id: str,
) -> SurfaceBuild | None:
    result = await session.execute(
        select(SurfaceBuild).where(
            SurfaceBuild.id == build_id,
            SurfaceBuild.organization_id == organization_id,
            SurfaceBuild.user_id == user_id,
            SurfaceBuild.event_id == event_id,
            SurfaceBuild.project_id == project_id,
        )
    )
    return result.scalars().first()


class SurfaceBuildHandler:
    """Build immutable Surface source on behalf of one authenticated Run."""

    def __init__(
        self,
        *,
        run_context: RunContext,
        runner: SurfaceBuildRunner | None = None,
        object_store: ObjectStore | None = None,
        inspection_transport: SurfaceInspectionTransport | None = None,
        session_factory: Any = async_session,
        owner_loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._run_context = run_context
        self._runner = runner or configured_surface_build_runner()
        # Resolve storage only if a successful build actually has a bundle to
        # retain. Merely assembling a Surface Builder run stays side-effect free.
        self._object_store = object_store
        self._inspection_transport = inspection_transport
        self._session_factory = session_factory
        self._owner_loop = owner_loop

    async def _on_owner_loop(self, coroutine):
        current = asyncio.get_running_loop()
        if self._owner_loop is None or self._owner_loop is current:
            return await coroutine
        future = asyncio.run_coroutine_threadsafe(coroutine, self._owner_loop)
        return await asyncio.wrap_future(future)

    async def _event_and_project(
        self,
        session: AsyncSession,
    ) -> tuple[Event, SurfaceProject]:
        event_id = self._run_context.event_id
        if not event_id:
            raise SurfaceAuthoringError("Event identity is required")
        event = await session.get(Event, event_id)
        if (
            event is None
            or event.organization_id != self._run_context.organization_id
            or event.user_id != self._run_context.user_id
        ):
            raise SurfaceAuthoringError("Event is unavailable")
        project = await _owned_project(
            session,
            organization_id=event.organization_id,
            user_id=event.user_id,
            event_id=event.id,
        )
        if project is None:
            raise SurfaceAuthoringError("Surface project is unavailable")
        return event, project

    async def _build(self, params: SurfaceBuildParams) -> dict[str, Any]:
        async with self._session_factory() as session:
            event, project = await self._event_and_project(session)
            revision_result = await session.execute(
                select(SurfaceRevision).where(
                    SurfaceRevision.id == params.revision_id,
                    SurfaceRevision.organization_id == event.organization_id,
                    SurfaceRevision.user_id == event.user_id,
                    SurfaceRevision.event_id == event.id,
                    SurfaceRevision.project_id == project.id,
                )
            )
            revision = revision_result.scalars().first()
            if revision is None:
                raise SurfaceAuthoringError("Surface revision is unavailable")
            request_fingerprint = _build_fingerprint(
                revision, self._runner.toolchain_version
            )
            replay_result = await session.execute(
                select(SurfaceBuild).where(
                    SurfaceBuild.project_id == project.id,
                    SurfaceBuild.organization_id == event.organization_id,
                    SurfaceBuild.user_id == event.user_id,
                    SurfaceBuild.idempotency_key == params.idempotency_key,
                )
            )
            replay = replay_result.scalars().first()
            if replay is not None:
                if replay.request_fingerprint != request_fingerprint:
                    raise SurfaceConflictError(
                        "idempotency_key was already used for a different build"
                    )
                payload = surface_build_payload(replay, include_log=True)
                payload["replayed"] = True
                return payload

            files = {
                str(path): str(content)
                for path, content in dict(revision.files).items()
            }
            manifest = dict(revision.manifest)
            diagnostics = validate_surface_source(files, manifest)
            now = datetime.now(timezone.utc)
            build = SurfaceBuild(
                organization_id=event.organization_id,
                user_id=event.user_id,
                event_id=event.id,
                project_id=project.id,
                revision_id=revision.id,
                creator_run_id=self._run_context.run_id,
                idempotency_key=params.idempotency_key,
                request_fingerprint=request_fingerprint,
                status="failed" if diagnostics else "building",
                source_checksum=revision.checksum,
                toolchain_version=self._runner.toolchain_version,
                validation_result={
                    "passed": not diagnostics,
                    "checked_at": now.isoformat(),
                },
                diagnostics=diagnostics,
                started_at=now,
                completed_at=now if diagnostics else None,
            )
            session.add(build)
            if diagnostics:
                session.add(
                    self._trail_entry(
                        event=event,
                        build=build,
                        summary="Surface build rejected by deterministic validation",
                    )
                )
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise SurfaceConflictError(
                    "Surface build was created concurrently; inspect builds and retry"
                ) from exc
            if diagnostics:
                payload = surface_build_payload(build, include_log=True)
                payload["replayed"] = False
                return payload

        try:
            runner_result = await self._runner.build(
                build_id=build.id,
                files=files,
                manifest=manifest,
            )
        except Exception as exc:
            runner_result = SurfaceBuildRunnerResult(
                status="failed",
                diagnostics=[
                    {
                        "code": "builder_exception",
                        "severity": "error",
                        "message": f"Isolated builder failed: {exc}",
                        "path": None,
                        "line": None,
                    }
                ],
            )

        status = runner_result.status
        result_diagnostics = list(runner_result.diagnostics)
        bundle_key: str | None = None
        bundle_sha256: str | None = None
        bundle_size = 0
        if status == "succeeded":
            if runner_result.bundle is None:
                status = "failed"
                result_diagnostics.append(
                    {
                        "code": "missing_bundle",
                        "severity": "error",
                        "message": "The isolated builder returned no bundle",
                        "path": None,
                        "line": None,
                    }
                )
            elif len(runner_result.bundle) > MAX_SURFACE_BUNDLE_BYTES:
                status = "failed"
                result_diagnostics.append(
                    {
                        "code": "bundle_too_large",
                        "severity": "error",
                        "message": (
                            "Surface bundle exceeds "
                            f"{MAX_SURFACE_BUNDLE_BYTES} bytes"
                        ),
                        "path": None,
                        "line": None,
                    }
                )
            else:
                bundle_size = len(runner_result.bundle)
                bundle_sha256 = hashlib.sha256(runner_result.bundle).hexdigest()
                bundle_key = surface_bundle_key(
                    event.organization_id,
                    event.id,
                    build.id,
                    bundle_sha256,
                )
                try:
                    object_store = self._object_store or get_object_store()
                    await asyncio.to_thread(
                        object_store.put,
                        bundle_key,
                        io.BytesIO(runner_result.bundle),
                        content_type="application/zip",
                    )
                except Exception as exc:
                    status = "failed"
                    bundle_key = None
                    bundle_sha256 = None
                    bundle_size = 0
                    result_diagnostics.append(
                        {
                            "code": "bundle_storage_failed",
                            "severity": "error",
                            "message": f"Surface bundle could not be retained: {exc}",
                            "path": None,
                            "line": None,
                        }
                    )

        async with self._session_factory() as session:
            event, project = await self._event_and_project(session)
            persisted = await _owned_build(
                session,
                build.id,
                organization_id=event.organization_id,
                user_id=event.user_id,
                event_id=event.id,
                project_id=project.id,
            )
            if persisted is None:
                raise SurfaceAuthoringError("Surface build record disappeared")
            persisted.status = status
            persisted.diagnostics = result_diagnostics
            persisted.build_log = runner_result.build_log[-MAX_SURFACE_BUILD_LOG_CHARS:]
            persisted.bundle_key = bundle_key
            persisted.bundle_sha256 = bundle_sha256
            persisted.bundle_size_bytes = bundle_size
            persisted.preview_artifacts = _public_preview_artifacts(
                runner_result.preview_artifacts
            )
            persisted.resource_metrics = dict(runner_result.resource_metrics)
            persisted.completed_at = datetime.now(timezone.utc)
            session.add(persisted)
            session.add(
                self._trail_entry(
                    event=event,
                    build=persisted,
                    summary=(
                        "Surface build completed"
                        if status == "succeeded"
                        else f"Surface build {status}"
                    ),
                )
            )
            await session.commit()
            await session.refresh(persisted)
            payload = surface_build_payload(persisted, include_log=True)
            payload["replayed"] = False
            return payload

    def _trail_entry(
        self,
        *,
        event: Event,
        build: SurfaceBuild,
        summary: str,
    ) -> EventTrailEntry:
        evidence = [{"surface_build_id": build.id}]
        if build.bundle_sha256:
            evidence.append({"bundle_sha256": build.bundle_sha256})
        return EventTrailEntry(
            organization_id=event.organization_id,
            user_id=event.user_id,
            event_id=event.id,
            actor_id=self._run_context.agent_id,
            kind="surface_build_finished",
            summary=summary,
            run_id=self._run_context.run_id,
            evidence_refs=evidence,
            payload={
                "project_id": build.project_id,
                "revision_id": build.revision_id,
                "build_id": build.id,
                "status": build.status,
            },
        )

    async def build(self, params: SurfaceBuildParams) -> dict[str, Any]:
        return await self._on_owner_loop(self._build(params))

    async def _preview(self, params: SurfacePreviewParams) -> dict[str, Any]:
        async with self._session_factory() as session:
            event, project = await self._event_and_project(session)
            if params.build_id:
                build = await _owned_build(
                    session,
                    params.build_id,
                    organization_id=event.organization_id,
                    user_id=event.user_id,
                    event_id=event.id,
                    project_id=project.id,
                )
            else:
                result = await session.execute(
                    select(SurfaceBuild)
                    .where(
                        SurfaceBuild.organization_id == event.organization_id,
                        SurfaceBuild.user_id == event.user_id,
                        SurfaceBuild.event_id == event.id,
                        SurfaceBuild.project_id == project.id,
                    )
                    .order_by(col(SurfaceBuild.created_at).desc())
                    .limit(1)
                )
                build = result.scalars().first()
            if build is None:
                raise SurfaceAuthoringError("Surface build is unavailable")
            revision = await session.get(SurfaceRevision, build.revision_id)
            if revision is None or revision.project_id != project.id:
                raise SurfaceAuthoringError("Surface revision is unavailable")
            manifest = SurfaceManifest.model_validate(revision.manifest)
            payload = surface_build_payload(build, include_log=True)
            preview_ready = build.status == "succeeded"
            runtime_diagnostics: list[dict[str, Any]] = []
            runtime_proven = False
            inspection_evidence: dict[str, Any] = {}
            capture_values: list[SurfaceInspectionArtifact] = []
            inspection_transport: dict[str, Any] = {}
            reusable_quality_receipt = dict(
                (build.resource_metrics or {}).get(SURFACE_QUALITY_RECEIPT_KEY) or {}
            )
            quality_receipt_reused = bool(
                preview_ready
                and reusable_quality_receipt
                and surface_quality_receipt_error(build) is None
            )
            if quality_receipt_reused:
                runtime_proven = True
                inspection_evidence = dict(
                    reusable_quality_receipt.get("evidence") or {}
                )
            elif preview_ready and build.resource_metrics.get("backend") in {
                "local_dev",
                "isolated",
            }:
                from .surface_interactions import surface_runtime_context
                from .tenancy import OrganizationContext, OrganizationPolicy

                if not build.bundle_key:
                    runtime_diagnostics = [
                        {
                            "stage": "runtime",
                            "code": "runtime_bundle_unavailable",
                            "severity": "error",
                            "message": "The Surface runtime bundle is unavailable",
                        }
                    ]
                else:
                    object_store = self._object_store or get_object_store()

                    def read_bundle() -> bytes:
                        with object_store.open(build.bundle_key or "") as stream:
                            return stream.read()

                    try:
                        bundle = await asyncio.to_thread(read_bundle)
                        document = build_surface_runtime_document(bundle)
                        runtime_context = await surface_runtime_context(
                            session,
                            context=OrganizationContext(
                                organization_id=event.organization_id,
                                user_id=event.user_id,
                                role="member",
                                permissions=(),
                                policy=OrganizationPolicy(),
                            ),
                            event_id=event.id,
                            build_id=build.id,
                            require_published=False,
                        )
                        binding = new_surface_inspection_binding(
                            build_id=build.id,
                            revision_id=build.revision_id,
                            source_checksum=build.source_checksum,
                            bundle_sha256=build.bundle_sha256 or "",
                        )
                        transport = self._inspection_transport
                        if transport is None:
                            transport = (
                                LocalSurfaceInspectionTransport()
                                if build.resource_metrics.get("backend") == "local_dev"
                                else configured_surface_inspection_transport()
                            )
                        inspection = await asyncio.to_thread(
                            transport.inspect,
                            SurfaceInspectionRequest(
                                binding=binding,
                                document=document,
                                runtime_context=runtime_context,
                                manifest=manifest,
                            ),
                        )
                        runtime_diagnostics = [
                            *inspection.diagnostics,
                            *validate_surface_inspection_result(
                                inspection,
                                expected=binding,
                            ),
                        ]
                        inspection_evidence = dict(inspection.evidence)
                        inspection_transport = dict(inspection.transport)
                        inspection_evidence["transport"] = inspection_transport
                        capture_values = list(inspection.artifacts)
                    except Exception as exc:
                        runtime_diagnostics = [
                            {
                                "stage": "runtime",
                                "code": "runtime_inspection_unavailable",
                                "severity": "error",
                                "message": str(exc),
                            }
                        ]
                preview_ready = not runtime_diagnostics
                runtime_proven = preview_ready
            elif (
                preview_ready
                and build.resource_metrics.get("runtime_inspection") == "passed"
                and build.resource_metrics.get("viewport_inspection") == "passed"
                and build.resource_metrics.get("accessibility_inspection") == "passed"
                and build.resource_metrics.get("state_inspection") == "passed"
                and build.resource_metrics.get("focus_inspection") == "passed"
                and build.resource_metrics.get("contrast_inspection") == "passed"
                and (
                    not manifest.interaction_checks
                    or build.resource_metrics.get("interaction_inspection") == "passed"
                )
                and (
                    not manifest.primary_jobs
                    or build.resource_metrics.get("primary_job_inspection") == "passed"
                )
            ):
                runtime_proven = True
                inspection_evidence = dict(
                    build.resource_metrics.get("inspection_evidence") or {}
                )
            elif preview_ready:
                runtime_diagnostics = [
                    {
                        "stage": "runtime",
                        "code": "runtime_inspector_unavailable",
                        "severity": "error",
                        "message": (
                            "The build produced no trusted browser runtime inspection receipt"
                        ),
                    }
                ]
                preview_ready = False
            retained_captures: list[dict[str, Any]] = []
            if preview_ready and not runtime_diagnostics and capture_values:
                object_store = self._object_store or get_object_store()
                for capture in capture_values:
                    data = capture.data
                    checksum = capture.sha256
                    name = capture.name
                    content_type = capture.content_type
                    key = surface_preview_artifact_key(
                        event.organization_id,
                        event.id,
                        build.id,
                        checksum,
                        name,
                    )
                    try:
                        await asyncio.to_thread(
                            object_store.put,
                            key,
                            io.BytesIO(data),
                            content_type=content_type,
                        )
                    except Exception as exc:
                        runtime_diagnostics.append(
                            {
                                "stage": "runtime",
                                "code": "viewport_capture_storage_failed",
                                "severity": "error",
                                "message": (
                                    "Trusted viewport capture could not be retained: "
                                    f"{exc}"
                                ),
                            }
                        )
                        continue
                    retained_captures.append(
                        {
                            "kind": capture.kind,
                            "name": name,
                            "content_type": content_type,
                            "sha256": checksum,
                            "size_bytes": len(data),
                        }
                    )
            if runtime_diagnostics:
                preview_ready = False
                runtime_proven = False
            if retained_captures:
                existing_artifacts = [
                    item
                    for item in list(build.preview_artifacts or [])
                    if item.get("kind") != "viewport_capture"
                ]
                build.preview_artifacts = _public_preview_artifacts(
                    [*existing_artifacts, *retained_captures]
                )
            quality_receipt = (
                reusable_quality_receipt
                if quality_receipt_reused
                else create_surface_quality_receipt(
                    build_id=build.id,
                    revision_id=build.revision_id,
                    source_checksum=build.source_checksum,
                    bundle_sha256=build.bundle_sha256,
                    validation_passed=build.validation_result.get("passed") is True,
                    manifest=manifest,
                    runtime_proven=runtime_proven,
                    runtime_diagnostics=runtime_diagnostics,
                    inspection_evidence=inspection_evidence,
                )
            )
            if quality_receipt.get("passed") is not True:
                preview_ready = False
                if not runtime_diagnostics:
                    runtime_diagnostics.append(
                        {
                            "stage": "quality",
                            "code": "surface_quality_evidence_incomplete",
                            "severity": "error",
                            "message": (
                                "The trusted build produced incomplete viewport, "
                                "state, focus, contrast, accessibility, or primary-job evidence"
                            ),
                        }
                    )
                    quality_receipt = create_surface_quality_receipt(
                        build_id=build.id,
                        revision_id=build.revision_id,
                        source_checksum=build.source_checksum,
                        bundle_sha256=build.bundle_sha256,
                        validation_passed=(
                            build.validation_result.get("passed") is True
                        ),
                        manifest=manifest,
                        runtime_proven=runtime_proven,
                        runtime_diagnostics=runtime_diagnostics,
                        inspection_evidence=inspection_evidence,
                    )
            build.resource_metrics = {
                **dict(build.resource_metrics or {}),
                SURFACE_QUALITY_RECEIPT_KEY: quality_receipt,
            }
            if inspection_transport and build.bundle_key and build.bundle_sha256:
                inspection_record = SurfaceInspection(
                    organization_id=event.organization_id,
                    user_id=event.user_id,
                    event_id=event.id,
                    project_id=project.id,
                    build_id=build.id,
                    revision_id=build.revision_id,
                    bundle_key=build.bundle_key,
                    bundle_sha256=build.bundle_sha256,
                    inspection_kind="runtime",
                    inspector_version=(
                        f"{inspection_transport.get('id', 'unknown')}@"
                        f"{inspection_transport.get('version', 'unknown')}"
                    ),
                    status=(
                        "passed" if quality_receipt.get("passed") is True else "failed"
                    ),
                    receipt_sha256=str(quality_receipt["fingerprint"]),
                    policy_versions={
                        "quality": str(quality_receipt.get("policy_version") or ""),
                        "transport": str(inspection_transport.get("version") or ""),
                    },
                    summary={
                        "diagnostic_count": len(runtime_diagnostics),
                        "quality_passed": quality_receipt.get("passed") is True,
                    },
                    timings=dict(inspection_evidence.get("timings") or {}),
                )
                session.add(inspection_record)
                for retained_capture in retained_captures:
                    checksum = str(retained_capture["sha256"])
                    session.add(
                        SurfaceEvidenceArtifact(
                            organization_id=event.organization_id,
                            user_id=event.user_id,
                            event_id=event.id,
                            project_id=project.id,
                            inspection_id=inspection_record.id,
                            build_id=build.id,
                            revision_id=build.revision_id,
                            bundle_key=build.bundle_key,
                            bundle_sha256=build.bundle_sha256,
                            artifact_kind=str(retained_capture["kind"]),
                            storage_key=surface_preview_artifact_key(
                                event.organization_id,
                                event.id,
                                build.id,
                                checksum,
                                str(retained_capture["name"]),
                            ),
                            content_type=str(retained_capture["content_type"]),
                            content_sha256=checksum,
                            size_bytes=int(retained_capture["size_bytes"]),
                        )
                    )
            session.add(build)
            await session.commit()
            payload["preview_ready"] = preview_ready
            payload["execution_available"] = runtime_proven
            payload["quality_gate"] = quality_receipt
            payload["quality_gate_reused"] = quality_receipt_reused
            payload["resource_metrics"] = build.resource_metrics
            payload["preview_artifacts"] = build.preview_artifacts
            payload["runtime_diagnostics"] = runtime_diagnostics
            payload["diagnostics"] = [
                *list(payload.get("diagnostics") or []),
                *runtime_diagnostics,
            ]
            payload["message"] = (
                "Surface mounted successfully with current Event context"
                if preview_ready and payload["execution_available"]
                else (
                    "Surface runtime inspection failed"
                    if runtime_diagnostics
                    else "Executable runtime inspection is unavailable"
                )
            )
            return payload

    async def preview(self, params: SurfacePreviewParams) -> dict[str, Any]:
        return await self._on_owner_loop(self._preview(params))

    async def _change_publication(
        self,
        params: SurfacePublicationParams,
        *,
        action: Literal["publish", "rollback"],
    ) -> dict[str, Any]:
        event_id = self._run_context.event_id
        if not event_id:
            raise SurfaceAuthoringError("Event identity is required")
        async with self._session_factory() as session:
            return await change_surface_publication(
                session,
                organization_id=self._run_context.organization_id,
                user_id=self._run_context.user_id,
                event_id=event_id,
                actor_id=self._run_context.agent_id,
                run_id=self._run_context.run_id,
                params=params,
                action=action,
                object_store=self._object_store or get_object_store(),
            )

    async def publish(self, params: SurfacePublicationParams) -> dict[str, Any]:
        return await self._on_owner_loop(
            self._change_publication(params, action="publish")
        )

    async def rollback(self, params: SurfacePublicationParams) -> dict[str, Any]:
        return await self._on_owner_loop(
            self._change_publication(params, action="rollback")
        )


__all__ = [
    "SurfaceBuildHandler",
    "SurfaceBuildParams",
    "SurfacePreviewParams",
    "SurfacePublicationParams",
    "surface_build_payload",
]
