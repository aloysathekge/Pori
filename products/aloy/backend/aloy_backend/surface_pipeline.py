"""Host-owned lifecycle for one materialized model-authored Surface candidate.

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
from .surface_manifest import parse_surface_manifest
from .surface_publication import SurfacePublicationParams

# One original candidate plus one deliberate repair. The trusted host must
# return every independent diagnostic it can observe in the first pass; it may
# not turn sequential quality gates into additional paid model submissions.
# One product generation plus two bounded, diagnostic-driven repairs. Repairs
# use the exact rejected source as their base and omit unrelated Event context,
# so a source-contract rebase cannot consume the only compiler-repair chance.
MAX_CANDIDATE_SUBMISSIONS = 3
_UNSET_BASE_REVISION = object()

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


def _normalize_candidate_path(value: Any) -> Any:
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
        return _normalize_candidate_path(value)

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
    files: list[SurfaceCandidateEnvelopeFile]


class SurfaceCandidateEditEnvelopeFile(BaseModel):
    """One bounded source mutation proposed against a frozen host revision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(
        description=(
            "Absolute model-owned source path below /workspace/. Return only a "
            "path that must be written or deleted for this requested revision."
        )
    )
    operation: Literal["write", "delete", "replace_text"]
    content: str | None = Field(
        default=None,
        description=(
            "Complete UTF-8 file content for write; null for delete or replace_text."
        ),
    )
    match: str | None = Field(
        default=None,
        description=(
            "For replace_text, the exact non-empty source fragment that must "
            "occur once in the current transaction state. Changes execute in "
            "order, so this includes earlier changes to the same file."
        ),
    )
    replacement: str | None = Field(
        default=None,
        description=(
            "For replace_text, the complete replacement for match. It may be "
            "empty when removing a fragment."
        ),
    )

    @field_validator("path", mode="before")
    @classmethod
    def normalize_model_path(cls, value: Any) -> Any:
        return _normalize_candidate_path(value)

    @model_validator(mode="after")
    def validate_operation_payload(self) -> "SurfaceCandidateEditEnvelopeFile":
        if self.operation == "write":
            if (
                self.content is None
                or self.match is not None
                or self.replacement is not None
            ):
                raise ValueError(
                    "write requires content and cannot contain match or replacement"
                )
        elif self.operation == "delete":
            if any(
                value is not None
                for value in (self.content, self.match, self.replacement)
            ):
                raise ValueError("delete cannot contain content, match, or replacement")
        elif not self.match or self.replacement is None or self.content is not None:
            raise ValueError(
                "replace_text requires a non-empty match and replacement, "
                "and cannot contain content"
            )
        return self


class SurfaceCandidateEditEnvelope(BaseModel):
    """Provider-facing incremental revision for an existing Surface."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str
    changes: list[SurfaceCandidateEditEnvelopeFile] = Field(
        min_length=1,
        max_length=MAX_SURFACE_FILES * 2,
    )


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


def materialize_surface_candidate_edit(
    envelope: SurfaceCandidateEditEnvelope,
    *,
    base_files: dict[str, str],
    primary_jobs: list[str],
) -> SurfaceCandidate:
    """Apply a model's bounded file edits to frozen host-owned source.

    The model never writes a revision directly. Every operation is normalized
    through the existing authoring contract, then the resulting complete file
    set passes the same authoritative candidate validation as a new Surface.
    """
    if not base_files:
        raise ValueError("Incremental Surface generation requires existing source")
    materialized = {str(path): str(content) for path, content in base_files.items()}
    frozen_base = dict(materialized)
    for change in envelope.changes:
        path = surface_source_path(change.path)
        if change.operation == "delete":
            if path not in materialized:
                raise ValueError(f"Cannot delete missing Surface source file: {path}")
            del materialized[path]
            continue
        if change.operation == "replace_text":
            if path not in materialized:
                raise ValueError(
                    f"Cannot replace text in missing Surface source file: {path}"
                )
            assert change.match is not None
            assert change.replacement is not None
            occurrences = materialized[path].count(change.match)
            if occurrences != 1:
                raise ValueError(
                    "replace_text match must occur exactly once in "
                    f"{path}; found {occurrences} occurrences"
                )
            content = materialized[path].replace(
                change.match,
                change.replacement,
                1,
            )
        else:
            assert change.content is not None
            content = change.content
        # Treat an idempotent operation as transaction noise, not as a reason
        # to discard other valid changes. Provider output can legitimately
        # restate an already-satisfied manifest edit while changing another
        # file. The host still rejects an entirely unchanged transaction below.
        if materialized.get(path) == content:
            continue
        patch = SurfaceFilePatch(
            path=change.path,
            operation="write",
            content=content,
        )
        assert patch.content is not None
        materialized[path] = patch.content
    if materialized == frozen_base:
        raise ValueError("Surface edit transaction does not change source")
    return SurfaceCandidate.model_validate(
        {
            "summary": envelope.summary,
            "primary_jobs": primary_jobs,
            "files": [
                {"path": f"/workspace{path}", "content": content}
                for path, content in sorted(materialized.items())
            ],
        }
    )


def bind_surface_manifest_primary_jobs(
    candidate: SurfaceCandidate,
    *,
    required_primary_jobs: list[dict[str, str]],
) -> SurfaceCandidate:
    """Bind model-authored job proofs to host-issued ids and descriptions.

    The Builder owns the browser steps and assertions because those depend on
    the UI it created. It never owns the acceptance contract identity. This
    boundary replaces copied or stale ids/descriptions deterministically while
    leaving the executable proof unchanged. A missing or ambiguous proof set
    is left untouched so the normal fail-closed pipeline emits diagnostics.
    """
    if not required_primary_jobs:
        return candidate
    by_path = {item.source_path: item for item in candidate.files}
    manifest_file = by_path.get("/surface.json")
    if manifest_file is None:
        return candidate
    try:
        manifest_value = json.loads(manifest_file.content)
    except (TypeError, ValueError):
        return candidate
    if not isinstance(manifest_value, dict):
        return candidate
    declared_value = manifest_value.get("primary_jobs")
    if not isinstance(declared_value, list):
        return candidate
    declared = [item for item in declared_value if isinstance(item, dict)]
    if len(declared) != len(declared_value):
        return candidate

    required_descriptions = [item["description"] for item in required_primary_jobs]
    matched: list[dict[str, Any]] = []
    for description in required_descriptions:
        matches = [
            item
            for item in declared
            if str(item.get("description") or "") == description
        ]
        if len(matches) != 1:
            matched = []
            break
        matched.append(dict(matches[0]))
    if not matched:
        # A proof may only be rebound when it was authored for this request's
        # jobs: matched above by exact description, or here by exact host-issued
        # id with a paraphrased description. A same-count set with foreign ids
        # is a previous revision's stale contract; rebinding would attach old
        # browser proofs to new job identities, so it must fail the gate.
        if len(declared) != len(required_primary_jobs):
            return candidate
        if [str(item.get("id") or "") for item in declared] != [
            item["id"] for item in required_primary_jobs
        ]:
            return candidate
        matched = [dict(item) for item in declared]

    manifest_value["primary_jobs"] = [
        {
            **proof,
            "id": required["id"],
            "description": required["description"],
        }
        for proof, required in zip(matched, required_primary_jobs, strict=True)
    ]
    rebound_content = json.dumps(
        manifest_value,
        ensure_ascii=False,
        indent=2,
    )
    rebound_files = [
        (
            SurfaceCandidateFile(
                path=item.path,
                content=rebound_content,
            )
            if item.source_path == "/surface.json"
            else item
        )
        for item in candidate.files
    ]
    return SurfaceCandidate.model_validate(
        {
            "summary": candidate.summary,
            "primary_jobs": required_descriptions,
            "files": [item.model_dump(mode="python") for item in rebound_files],
        }
    )


def surface_primary_job_contract_diagnostics(
    candidate: SurfaceCandidate,
    *,
    required_primary_jobs: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Check one candidate against the host-frozen primary-job contract.

    This is the single authority for the contract; the publication pipeline
    enforces it, and the development workspace evaluates the same function at
    ``finish_candidate`` so a Builder never spends a paid host submission on a
    deterministically rejectable manifest.
    """
    if not required_primary_jobs:
        return []
    required_descriptions = [item["description"] for item in required_primary_jobs]
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
        if declared != required_primary_jobs:
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
    return diagnostics


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
        expected_base_revision_id: str | None | object = _UNSET_BASE_REVISION,
    ) -> None:
        self._run_id = run_id
        self._authoring = authoring_handler
        self._builds = build_handler
        self._stage_observer = stage_observer
        self._required_primary_jobs = [
            dict(item) for item in required_primary_jobs or []
        ]
        self._expected_base_revision_id = expected_base_revision_id

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
            diagnostics = surface_primary_job_contract_diagnostics(
                candidate,
                required_primary_jobs=self._required_primary_jobs,
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
        if (
            self._expected_base_revision_id is not _UNSET_BASE_REVISION
            and before.get("expected_revision") != self._expected_base_revision_id
        ):
            raise SurfaceConflictError(
                "Surface source changed after Builder context was frozen"
            )
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

        project = dict(persisted.get("project") or {})
        return await SurfaceRevisionHostPipeline(
            run_id=self._run_id,
            build_handler=self._builds,
            stage_observer=self._stage_observer,
        ).execute(
            revision_id=revision_id,
            source_fingerprint=digest,
            expected_published_revision_id=project.get("published_revision_id"),
            expected_published_build_id=project.get("published_build_id"),
            attempt=submission,
            timings_ms=timings,
        )


class SurfaceRevisionHostPipeline:
    """Build, inspect, and publish one already-persisted immutable revision.

    Model-authored candidates enter here after source persistence. Reviewed
    template source enters at the same boundary, so there is one compiler,
    quality gate, publication authority, and idempotency contract.
    """

    def __init__(
        self,
        *,
        run_id: str,
        build_handler: SurfaceBuildHandler,
        stage_observer: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._run_id = run_id
        self._builds = build_handler
        self._stage_observer = stage_observer

    async def _stage(self, stage: str) -> None:
        if self._stage_observer is not None:
            await self._stage_observer(stage)

    async def execute(
        self,
        *,
        revision_id: str,
        source_fingerprint: str,
        expected_published_revision_id: str | None,
        expected_published_build_id: str | None,
        attempt: int = 1,
        timings_ms: dict[str, float] | None = None,
    ) -> SurfacePipelineResult:
        timings = dict(timings_ms or {})

        await self._stage("building_bundle")
        started = perf_counter()
        build = await self._builds.build(
            SurfaceBuildParams(
                revision_id=revision_id,
                idempotency_key=_idempotency_key(
                    self._run_id, attempt, "build", source_fingerprint
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
                candidate_fingerprint=source_fingerprint,
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
                candidate_fingerprint=source_fingerprint,
                revision_id=revision_id,
                build_id=build_id,
                diagnostics=diagnostics,
                timings_ms=timings,
            )

        await self._stage("publishing_surface")
        started = perf_counter()
        publication = await self._builds.publish(
            SurfacePublicationParams(
                build_id=build_id,
                expected_published_revision_id=expected_published_revision_id,
                expected_published_build_id=expected_published_build_id,
                idempotency_key=_idempotency_key(
                    self._run_id, attempt, "publish", source_fingerprint
                ),
            )
        )
        timings["publish"] = round((perf_counter() - started) * 1000, 3)
        return SurfacePipelineResult(
            status="published",
            candidate_fingerprint=source_fingerprint,
            revision_id=revision_id,
            build_id=build_id,
            publication=publication,
            timings_ms=timings,
        )


__all__ = [
    "HOST_FAILURE_CODES",
    "MAX_CANDIDATE_SUBMISSIONS",
    "SurfaceCandidate",
    "SurfaceCandidateEditEnvelope",
    "SurfaceCandidateEditEnvelopeFile",
    "SurfaceCandidateEnvelope",
    "SurfaceCandidateEnvelopeFile",
    "SurfaceCandidateFile",
    "SurfaceHostPipeline",
    "SurfaceRevisionHostPipeline",
    "SurfacePipelineDiagnostic",
    "SurfacePipelineResult",
    "bind_surface_manifest_primary_jobs",
    "materialize_surface_candidate_edit",
    "surface_primary_job_contract_diagnostics",
]
