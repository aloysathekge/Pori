"""Host-owned lifecycle for one complete model-authored Surface candidate.

The model proposes source. Aloy owns every mechanical and authority-bearing
step after that: immutable persistence, deterministic validation, isolated
build, preview inspection, and atomic publication. A failed candidate never
changes the live Surface pointer and returns bounded repair diagnostics.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .surface_authoring import (
    MAX_SURFACE_FILE_BYTES,
    MAX_SURFACE_FILES,
    SurfaceAuthoringError,
    SurfaceConflictError,
    SurfaceFilePatch,
    SurfaceWriteFilesParams,
    surface_source_path,
)
from .surface_builds import (
    SurfaceBuildHandler,
    SurfaceBuildParams,
    SurfacePreviewParams,
)
from .surface_publication import SurfacePublicationParams
from .surface_manifest import parse_surface_manifest

MAX_CANDIDATE_SUBMISSIONS = 2

# These diagnostics describe Aloy's build infrastructure, not model-authored
# source. Sending them back to the model wastes latency and tokens because no
# source rewrite can repair storage, sandbox acquisition, or runner contracts.
HOST_FAILURE_CODES = frozenset(
    {
        "builder_exception",
        "builder_protocol_error",
        "bundle_storage_failed",
        "isolated_builder_unavailable",
        "local_toolchain_unavailable",
        "missing_bundle",
        "runtime_bundle_invalid",
        "runtime_bundle_unavailable",
        "runtime_inspector_failed",
        "runtime_inspector_unavailable",
        "surface_quality_evidence_incomplete",
    }
)


class SurfaceCandidateFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(
        min_length=1,
        max_length=500,
        pattern=r"^/workspace/",
        description="Absolute generated source path below /workspace/.",
    )
    content: str = Field(
        max_length=MAX_SURFACE_FILE_BYTES,
        description="The complete UTF-8 file content, never a patch or diff.",
    )

    @field_validator("path", mode="before")
    @classmethod
    def normalize_model_path(cls, value: Any) -> Any:
        """Canonicalize safe project-root shorthand before authority checks.

        Provider-enforced structured output does not reliably preserve the
        virtual ``/workspace`` prefix even when it appears in the schema. The
        host can add that prefix without changing which project file the model
        selected. All other paths still fail the existing workspace/toolchain
        allow-list validation below.
        """
        if not isinstance(value, str):
            return value
        normalized = value.replace("\\", "/")
        if normalized == "/surface.json" or normalized.startswith("/src/"):
            return f"/workspace{normalized}"
        if normalized == "surface.json" or normalized.startswith("src/"):
            return f"/workspace/{normalized}"
        if normalized.startswith("workspace/"):
            return f"/{normalized}"
        return normalized

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        surface_source_path(value)
        return value

    @model_validator(mode="after")
    def validate_encoded_size(self) -> "SurfaceCandidateFile":
        # Field max_length counts characters; authoring limits encoded bytes.
        SurfaceFilePatch(path=self.path, content=self.content)
        return self

    @property
    def source_path(self) -> str:
        return surface_source_path(self.path)


class SurfaceCandidateEnvelopeFile(BaseModel):
    """Shape-only provider output; host authority checks happen after parsing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(
        description=(
            "Absolute model-owned source path below /workspace/. Use only React, "
            "TypeScript, JavaScript, CSS, JSON, Markdown, or SVG source. Do not "
            "return index.html, package files, lockfiles, compiler configuration, "
            "dependencies, or other host-owned toolchain files."
        )
    )
    content: str = Field(
        description="The complete UTF-8 file content, never a patch or diff."
    )


class SurfaceCandidateEnvelope(BaseModel):
    """Provider-facing shape separated from Aloy's authoritative validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str
    primary_jobs: list[str]
    files: list[SurfaceCandidateEnvelopeFile]


class SurfaceCandidate(BaseModel):
    """One complete, replacement-safe React candidate returned by the model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = Field(
        min_length=1,
        max_length=1000,
        description="What the candidate helps this Event's user accomplish.",
    )
    primary_jobs: list[str] = Field(
        min_length=1,
        max_length=20,
        description="Concrete user jobs implemented by the candidate.",
    )
    files: list[SurfaceCandidateFile] = Field(
        min_length=1,
        max_length=MAX_SURFACE_FILES,
    )

    @field_validator("summary", mode="before")
    @classmethod
    def normalize_summary(cls, value: Any) -> Any:
        """Bound descriptive metadata without spending a model repair.

        The summary is display metadata, not executable source or authority.
        Collapsing whitespace and applying the documented limit is therefore a
        deterministic host normalization; an empty summary still fails closed.
        """
        if not isinstance(value, str):
            return value
        return " ".join(value.split())[:1000]

    @field_validator("primary_jobs")
    @classmethod
    def validate_jobs(cls, values: list[str]) -> list[str]:
        jobs = [" ".join(value.split())[:300] for value in values if value.strip()]
        if not jobs:
            raise ValueError("At least one primary Surface job is required")
        if len(jobs) != len(set(jobs)):
            raise ValueError("Surface primary jobs must be unique")
        return jobs

    @model_validator(mode="after")
    def validate_complete_file_set(self) -> "SurfaceCandidate":
        paths = [item.source_path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("Surface candidate file paths must be unique")
        if "/surface.json" not in paths:
            raise ValueError("Surface candidate must include /workspace/surface.json")
        return self

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class SurfacePipelineDiagnostic(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    stage: str
    code: str
    severity: str = "error"
    message: str
    path: str | None = None
    line: int | None = None


class SurfacePipelineResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["published", "repair_required", "host_failed"]
    candidate_fingerprint: str
    revision_id: str | None = None
    build_id: str | None = None
    publication: dict[str, Any] | None = None
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)


def _idempotency_key(run_id: str, submission: int, stage: str, digest: str) -> str:
    return f"surface-pipeline:{run_id}:{submission}:{stage}:{digest[:24]}"


def _diagnostics_for_stage(
    stage: str,
    values: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for value in values[:100]:
        item = dict(value) if isinstance(value, dict) else {"message": str(value)}
        item["stage"] = stage
        item.setdefault("code", f"{stage}_failed")
        item.setdefault("severity", "error")
        item.setdefault("message", f"Surface {stage} failed")
        normalized.append(
            SurfacePipelineDiagnostic.model_validate(item).model_dump(mode="json")
        )
    return normalized


def _failed_build_status(
    diagnostics: list[dict[str, Any]],
) -> Literal["repair_required", "host_failed"]:
    codes = {str(item.get("code") or "") for item in diagnostics}
    return "host_failed" if codes & HOST_FAILURE_CODES else "repair_required"


class SurfaceHostPipeline:
    """Execute the trusted lifecycle for a candidate without model control."""

    def __init__(
        self,
        *,
        run_id: str,
        authoring_handler: Any,
        build_handler: SurfaceBuildHandler,
        stage_observer: Callable[[str], Awaitable[None]] | None = None,
        required_primary_jobs: list[dict[str, str]] | None = None,
    ) -> None:
        self._run_id = run_id
        self._authoring = authoring_handler
        self._builds = build_handler
        self._stage_observer = stage_observer
        self._required_primary_jobs = [dict(item) for item in required_primary_jobs or []]

    async def _stage(self, stage: str) -> None:
        if self._stage_observer is not None:
            await self._stage_observer(stage)

    async def execute(
        self,
        candidate: SurfaceCandidate,
        *,
        submission: int,
    ) -> SurfacePipelineResult:
        digest = candidate.fingerprint
        timings: dict[str, float] = {}

        await self._stage("validating_candidate")
        started = perf_counter()
        if self._required_primary_jobs:
            required_descriptions = [
                item["description"] for item in self._required_primary_jobs
            ]
            diagnostics: list[dict[str, Any]] = []
            if candidate.primary_jobs != required_descriptions:
                diagnostics.append(
                    {
                        "code": "primary_job_contract_mismatch",
                        "message": (
                            "Candidate primary_jobs must exactly match the host-frozen "
                            "job descriptions in their original order"
                        ),
                        "path": "/surface.json",
                    }
                )
            try:
                manifest = parse_surface_manifest(
                    {item.source_path: item.content for item in candidate.files}
                )
                declared = [
                    {"id": job.id, "description": job.description}
                    for job in manifest.primary_jobs
                ]
                if declared != self._required_primary_jobs:
                    diagnostics.append(
                        {
                            "code": "primary_job_manifest_mismatch",
                            "message": (
                                "surface.json must declare every host-issued primary job "
                                "id and description exactly once and in order"
                            ),
                            "path": "/surface.json",
                        }
                    )
            except ValueError as exc:
                diagnostics.append(
                    {
                        "code": "primary_job_manifest_invalid",
                        "message": str(exc),
                        "path": "/surface.json",
                    }
                )
            if diagnostics:
                timings["contract"] = round((perf_counter() - started) * 1000, 3)
                return SurfacePipelineResult(
                    status="repair_required",
                    candidate_fingerprint=digest,
                    diagnostics=_diagnostics_for_stage("validation", diagnostics),
                    timings_ms=timings,
                )
        before = await self._authoring.read()
        draft = dict((before.get("draft") or {}).get("files") or {})
        candidate_by_path = {item.source_path: item for item in candidate.files}
        patches = [
            SurfaceFilePatch(path=item.path, content=item.content)
            for item in candidate.files
        ]
        patches.extend(
            SurfaceFilePatch(path=f"/workspace{path}", operation="delete")
            for path in sorted(set(draft) - set(candidate_by_path))
        )
        try:
            persisted = await self._authoring.write(
                SurfaceWriteFilesParams(
                    expected_revision=before.get("expected_revision"),
                    idempotency_key=_idempotency_key(
                        self._run_id, submission, "persist", digest
                    ),
                    patches=patches,
                )
            )
        except SurfaceConflictError:
            # Concurrency and idempotency conflicts require fresh host truth;
            # they are not source problems the model can repair.
            raise
        except SurfaceAuthoringError as exc:
            timings["persist"] = round((perf_counter() - started) * 1000, 3)
            return SurfacePipelineResult(
                status="repair_required",
                candidate_fingerprint=digest,
                diagnostics=_diagnostics_for_stage(
                    "validation",
                    [{"code": "source_rejected", "message": str(exc)}],
                ),
                timings_ms=timings,
            )
        timings["persist"] = round((perf_counter() - started) * 1000, 3)
        revision_id = str((persisted.get("draft") or {}).get("id") or "")
        if not revision_id:
            raise ValueError("Host persisted no Surface revision")

        await self._stage("building_bundle")
        started = perf_counter()
        build = await self._builds.build(
            SurfaceBuildParams(
                revision_id=revision_id,
                idempotency_key=_idempotency_key(
                    self._run_id, submission, "build", digest
                ),
            )
        )
        timings["build"] = round((perf_counter() - started) * 1000, 3)
        build_id = str(build.get("id") or "")
        if build.get("status") != "succeeded":
            diagnostics = _diagnostics_for_stage(
                "build",
                list(build.get("diagnostics") or []),
            )
            if not diagnostics:
                diagnostics = _diagnostics_for_stage(
                    "build",
                    [{"message": "The candidate did not produce a valid bundle"}],
                )
            return SurfacePipelineResult(
                status=_failed_build_status(diagnostics),
                candidate_fingerprint=digest,
                revision_id=revision_id,
                build_id=build_id or None,
                diagnostics=diagnostics,
                timings_ms=timings,
            )

        await self._stage("inspecting_preview")
        started = perf_counter()
        preview = await self._builds.preview(SurfacePreviewParams(build_id=build_id))
        timings["preview"] = round((perf_counter() - started) * 1000, 3)
        if preview.get("preview_ready") is not True:
            diagnostics = _diagnostics_for_stage(
                "preview",
                list(preview.get("diagnostics") or [])
                or [{"message": "The retained preview is not ready"}],
            )
            return SurfacePipelineResult(
                status=_failed_build_status(diagnostics),
                candidate_fingerprint=digest,
                revision_id=revision_id,
                build_id=build_id,
                diagnostics=diagnostics,
                timings_ms=timings,
            )

        project = dict(persisted.get("project") or {})
        await self._stage("publishing_surface")
        started = perf_counter()
        publication = await self._builds.publish(
            SurfacePublicationParams(
                build_id=build_id,
                expected_published_revision_id=project.get("published_revision_id"),
                expected_published_build_id=project.get("published_build_id"),
                idempotency_key=_idempotency_key(
                    self._run_id, submission, "publish", digest
                ),
            )
        )
        timings["publish"] = round((perf_counter() - started) * 1000, 3)
        return SurfacePipelineResult(
            status="published",
            candidate_fingerprint=digest,
            revision_id=revision_id,
            build_id=build_id,
            publication=publication,
            timings_ms=timings,
        )


__all__ = [
    "HOST_FAILURE_CODES",
    "MAX_CANDIDATE_SUBMISSIONS",
    "SurfaceCandidate",
    "SurfaceCandidateEnvelope",
    "SurfaceCandidateEnvelopeFile",
    "SurfaceCandidateFile",
    "SurfaceHostPipeline",
    "SurfacePipelineDiagnostic",
    "SurfacePipelineResult",
]
