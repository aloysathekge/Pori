"""Durable Surface Builder execution over a trusted development workspace.

Production uses a bounded edit/check loop against a temporary Git-tracked
project. The older schema-bound executor remains private for focused lifecycle
compatibility tests while v4 Runs use the workspace path. Neither path grants
model code publication, Event-state, shell, or network authority.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ValidationError
from sqlmodel import select

from pori import (
    BudgetExceeded,
    BudgetLedger,
    SystemMessage,
    TokenUsage,
    UserMessage,
    ainvoke_structured,
    create_llm,
    ensure_budgeted_chat_model,
    estimate_llm_call_cost,
    normalize_usage,
    stable_fingerprint,
)

from .database import async_session
from .model_roles import ModelAssignment, ModelRole
from .models import (
    Conversation,
    EventTrailEntry,
    Message,
    Organization,
    OrganizationMembership,
    Run,
)
from .run_budgets import budget_ledger_for_run, remaining_run_seconds
from .run_outcome import make_usage_record
from .run_profiles import SURFACE_BUILDER_RUN_PROFILE
from .runtime import authenticated_run_context
from .skills import SURFACE_BUILDER_SKILL_ID, surface_builder_instructions
from .surface_builder_loop import run_surface_builder_loop
from .surface_development_workspace import (
    LocalGitSurfaceWorkspace,
    SurfaceDevelopmentWorkspace,
)
from .surface_lifecycle import mark_surface_run_started, reconcile_surface_run
from .surface_pipeline import (
    MAX_CANDIDATE_SUBMISSIONS,
    SurfaceCandidate,
    SurfaceCandidateEditEnvelope,
    SurfaceCandidateEditEnvelopeFile,
    SurfaceCandidateEnvelope,
    SurfaceCandidateEnvelopeFile,
    SurfaceHostPipeline,
    SurfacePipelineResult,
    bind_surface_manifest_primary_jobs,
    materialize_surface_candidate_edit,
)
from .surface_requests import (
    SURFACE_BUILDER_RUN_KIND,
    record_surface_builder_failure,
    record_surface_builder_started,
    surface_request_primary_jobs,
    verified_surface_publication,
)
from .surface_workspace import resolve_surface_authoring_runtime
from .tenancy import OrganizationPolicy

logger = logging.getLogger("aloy_backend.surface_builder")

MAX_SURFACE_PROMPT_CONTEXT_CHARS = 480_000
MAX_SURFACE_PROMPT_TRAIL_ENTRIES = 40
MAX_SURFACE_PROMPT_FILE_EXCERPTS = 8
MAX_SURFACE_PROMPT_FILE_EXCERPT_CHARS = 8_000
SURFACE_PROGRESS_HEARTBEAT_SECONDS = 5.0
MAX_REJECTED_OUTPUT_EXCERPT_CHARS = 24_000
SURFACE_STREAM_IDLE_TIMEOUT_SECONDS = 60.0


class SurfaceCandidateExhaustedError(RuntimeError):
    """All bounded candidate submissions failed deterministic host checks."""


class SurfaceCandidateOutputError(ValueError):
    """The provider returned output that did not validate as a candidate."""

    def __init__(self, diagnostic: dict[str, Any]) -> None:
        self.diagnostic = diagnostic
        super().__init__(str(diagnostic.get("error") or "Invalid Surface candidate"))


class SurfaceGenerationTimeoutError(TimeoutError):
    """One model submission exceeded its frozen role-level latency budget."""


class SurfacePipelineInfrastructureError(RuntimeError):
    """The trusted host failed in a way model-authored source cannot repair."""

    def __init__(
        self,
        diagnostics: list[dict[str, Any]],
        *,
        revision_id: str | None,
        build_id: str | None,
    ) -> None:
        self.diagnostics = diagnostics
        self.revision_id = revision_id
        self.build_id = build_id
        message = next(
            (str(item.get("message")) for item in diagnostics if item.get("message")),
            "Surface host pipeline failed",
        )
        super().__init__(message)


@dataclass
class _GenerationProgress:
    """Non-sensitive evidence that a streaming Builder call is advancing."""

    output_chars: int = 0
    output_chunks: int = 0
    first_output_at: datetime | None = None
    last_output_at: datetime | None = None
    first_output_monotonic: float | None = None
    last_output_monotonic: float | None = None

    def on_delta(self, value: str) -> None:
        if not value:
            return
        now = datetime.now(timezone.utc)
        monotonic_now = perf_counter()
        self.output_chars += len(value)
        self.output_chunks += 1
        if self.first_output_at is None:
            self.first_output_at = now
            self.first_output_monotonic = monotonic_now
        self.last_output_at = now
        self.last_output_monotonic = monotonic_now

    def payload(self) -> dict[str, Any]:
        return {
            "generation_phase": (
                "receiving_output" if self.output_chunks else "waiting_for_output"
            ),
            "output_chars": self.output_chars,
            "output_chunks": self.output_chunks,
            "first_output_at": (
                self.first_output_at.isoformat() if self.first_output_at else None
            ),
            "last_output_at": (
                self.last_output_at.isoformat() if self.last_output_at else None
            ),
        }


def _json(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=True, sort_keys=True)


def _rejected_output_diagnostic(response: Any) -> dict[str, Any]:
    payload = response if isinstance(response, dict) else {}
    raw = payload.get("raw")
    if isinstance(raw, str):
        raw_text = raw
    elif raw is None:
        raw_text = ""
    else:
        raw_text = _json(raw)
    excerpt_size = MAX_REJECTED_OUTPUT_EXCERPT_CHARS // 2
    truncated = len(raw_text) > MAX_REJECTED_OUTPUT_EXCERPT_CHARS
    return {
        "stage": "structured_output",
        "code": "surface_candidate_parse_failed",
        "error": str(payload.get("error") or "Provider returned no parsed candidate"),
        "raw_type": type(raw).__name__,
        "raw_chars": len(raw_text),
        "raw_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "raw_truncated": truncated,
        "raw_head": raw_text[:excerpt_size],
        "raw_tail": raw_text[-excerpt_size:] if truncated else "",
        "contains_react_source": any(
            marker in raw_text
            for marker in ("App.tsx", "from 'react'", 'from "react"', "export default")
        ),
    }


def _candidate_contract_diagnostics(
    exc: ValidationError,
) -> list[dict[str, Any]]:
    """Convert authoritative candidate validation into bounded repair feedback."""
    diagnostics: list[dict[str, Any]] = []
    for error in exc.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    )[:50]:
        location = error.get("loc") or ()
        path = ".".join(str(part) for part in location)
        diagnostics.append(
            {
                "stage": "candidate_validation",
                "code": "invalid_surface_candidate",
                "severity": "error",
                "message": str(error.get("msg") or "Invalid Surface candidate"),
                "path": path or None,
                "validation_type": str(error.get("type") or "value_error"),
            }
        )
    return diagnostics or [
        {
            "stage": "candidate_validation",
            "code": "invalid_surface_candidate",
            "severity": "error",
            "message": "The candidate violates Aloy's Surface source contract.",
        }
    ]


def _surface_builder_prompt_projection(
    value: dict[str, Any],
    *,
    compact: bool = False,
) -> dict[str, Any]:
    """Project only the trusted context required to author one Surface revision.

    The editable draft must remain complete. Published source duplicates that
    draft in the common case and is authority evidence rather than model input,
    so only its metadata crosses the generation boundary. Trail payloads are
    operational receipts; compact semantic entries are enough to orient design.
    """
    surface = dict(value.get("surface") or {})
    published = dict(surface.get("published") or {})
    published.pop("files", None)
    surface["published"] = published
    trail_limit = 15 if compact else MAX_SURFACE_PROMPT_TRAIL_ENTRIES
    excerpt_count = 3 if compact else MAX_SURFACE_PROMPT_FILE_EXCERPTS
    excerpt_chars = 3_000 if compact else MAX_SURFACE_PROMPT_FILE_EXCERPT_CHARS
    trail = []
    for entry in list(value.get("trail") or [])[:trail_limit]:
        if not isinstance(entry, dict):
            continue
        trail.append(
            {
                key: entry[key]
                for key in ("id", "kind", "summary", "created_at")
                if key in entry
            }
        )
    return {
        "event": value.get("event") or {},
        "brief": value.get("brief") or {},
        "tasks": list(value.get("tasks") or [])[: (50 if compact else 100)],
        "proposals": list(value.get("proposals") or [])[: (25 if compact else 50)],
        "files": list(value.get("files") or [])[: (50 if compact else 100)],
        "file_excerpts": {
            path: content[:excerpt_chars]
            for path, content in list(dict(value.get("file_excerpts") or {}).items())[
                :excerpt_count
            ]
        },
        "trail": trail,
        "surface": surface,
    }


def _render_prompt_context(value: dict[str, Any]) -> str:
    """Keep structured generation purpose-scoped and refuse partial source."""
    rendered = _json(_surface_builder_prompt_projection(value))
    if len(rendered) <= MAX_SURFACE_PROMPT_CONTEXT_CHARS:
        return rendered
    # Keep the complete editable draft while reducing optional canonical context.
    rendered = _json(_surface_builder_prompt_projection(value, compact=True))
    if len(rendered) > MAX_SURFACE_PROMPT_CONTEXT_CHARS:
        raise ValueError(
            "Surface Builder context exceeds the safe structured-generation limit"
        )
    return rendered


def _validate_assignment(run: Run, policy: OrganizationPolicy) -> ModelAssignment:
    if run.model_assignment is None:
        raise ValueError("Surface Builder model assignment is unavailable")
    assignment = ModelAssignment.model_validate(run.model_assignment)
    assignment.verify_fingerprint()
    if assignment.role != ModelRole.SURFACE_BUILDER:
        raise ValueError("Surface Builder model assignment has the wrong role")
    missing = SURFACE_BUILDER_RUN_PROFILE.required_model_capabilities.difference(
        assignment.capabilities
    )
    if missing:
        raise ValueError(
            "Surface Builder model assignment lacks capabilities: "
            + ", ".join(sorted(missing))
        )
    if assignment.skill_id != SURFACE_BUILDER_SKILL_ID:
        raise ValueError("Surface Builder skill assignment is unavailable")
    if policy.allowed_provider_profiles and (
        assignment.provider not in policy.allowed_provider_profiles
    ):
        raise PermissionError(
            "Surface Builder provider is denied by organization policy"
        )
    if policy.allowed_models and assignment.model not in policy.allowed_models:
        raise PermissionError("Surface Builder model is denied by organization policy")
    if run.team_config_id is not None:
        raise ValueError("Surface Builder Runs cannot use a TeamConfig")
    if run.run_profile != SURFACE_BUILDER_RUN_PROFILE.descriptor():
        raise ValueError("Surface Builder Run profile is unavailable or stale")
    return assignment


def _usage_metrics(
    assignment: ModelAssignment,
    usage: TokenUsage,
    *,
    generation_ms: float,
    pipeline_timings: list[dict[str, float]],
    started_at: datetime | None,
) -> dict[str, Any]:
    model_id = assignment.model.rsplit("/", 1)[-1]
    cost = estimate_llm_call_cost(assignment.model, usage)
    if cost is None:
        cost = estimate_llm_call_cost(model_id, usage)
    normalized_started = started_at
    if normalized_started is not None and normalized_started.tzinfo is None:
        normalized_started = normalized_started.replace(tzinfo=timezone.utc)
    return {
        "model": f"{assignment.provider}/{assignment.model}",
        "structured_output": True,
        "tool_calls": 0,
        "tokens": {
            "input": usage.input_tokens,
            "output": usage.output_tokens,
            "total": usage.total_tokens,
            "cache_read": usage.cache_read_tokens,
            "cache_write": usage.cache_write_tokens,
        },
        "cost_usd": f"${cost:.4f}" if cost is not None else None,
        "surface_pipeline": {
            "generation_ms": round(generation_ms, 3),
            "submissions": pipeline_timings,
        },
        "aloy_model_assignment": assignment.descriptor(),
        "aloy_run_elapsed_ms": (
            round(
                (datetime.now(timezone.utc) - normalized_started).total_seconds()
                * 1000,
                3,
            )
            if normalized_started is not None
            else None
        ),
    }


def _messages(
    *,
    task: str,
    context: str,
    instructions: str,
    previous_candidate: BaseModel | None,
    diagnostics: list[dict[str, Any]],
    repair_files: dict[str, str] | None,
    candidate_mode: str,
    required_primary_jobs: list[dict[str, str]],
) -> list[Any]:
    if candidate_mode == "edit":
        contract = (
            "\n\nRevision contract: an existing Surface is frozen in the supplied "
            "context. Return only the smallest source transactions required for "
            "this request. Prefer replace_text with an exact fragment that occurs "
            "once; use a whole-file write only when the file truly needs a broad "
            "rewrite. Transactions execute in listed order and may make multiple "
            "exact changes to the same file. Preserve every unmentioned file and "
            "never repeat the complete project."
        )
    else:
        contract = (
            "\n\nCreation contract: no usable Surface source exists yet. Return one "
            "complete candidate containing every required model-owned source file."
        )
    repair = ""
    if previous_candidate is not None:
        repair_source = _json({"files": repair_files or {}})
        if len(repair_source) > MAX_SURFACE_PROMPT_CONTEXT_CHARS:
            raise ValueError("Surface repair source exceeds the safe generation limit")
        repair = (
            "\n\nThe host rejected the previous candidate. The exact current "
            "rejected source below is now the sole editing base; do not use an "
            "older Event-context draft or stale match. Return a corrected "
            "candidate in the same revision mode that repairs the entire "
            "diagnostic bundle. For an incremental revision, return only minimal "
            "replace_text, write, or delete transactions; ordered transactions "
            "may target the same file.\nDiagnostics:\n"
            + _json(_compact_repair_diagnostics(diagnostics))
            + "\nExact current rejected source:\n"
            + repair_source
            + "\nPrevious submitted transaction:\n"
            + previous_candidate.model_dump_json()
        )
    host_contract = (
        "\n\nHost-owned acceptance jobs:\n"
        + _json(required_primary_jobs)
        + "\nDo not copy these jobs into provider metadata; the host binds their "
        "ids and descriptions. In surface.json, provide exactly one ordered "
        "browser proof per job using the exact id and description. Use "
        'method "openResource" with name "event.resource.open" to prove a '
        "trusted Event file viewer action; never invent host_ui."
    )
    return [
        SystemMessage(
            content=(
                SURFACE_BUILDER_RUN_PROFILE.system_prompt
                + "\n\nApply this exact Aloy Builder skill:\n\n"
                + instructions
            )
        ),
        UserMessage(
            content=(
                task
                + (
                    "\n\nTrusted Event and current Surface context:\n" + context
                    if previous_candidate is None
                    else ""
                )
                + contract
                + host_contract
                + repair
            )
        ),
    ]


def _compact_repair_diagnostics(
    values: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse repeated composition failures into one actionable bundle.

    Exact per-composition evidence remains in the retained pipeline receipt.
    The paid repair prompt receives one entry per underlying problem, with
    bounded examples and the affected states/viewports, instead of dozens of
    nearly identical diagnostics.
    """
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for raw in values:
        item = dict(raw)
        key = (
            item.get("stage"),
            item.get("code"),
            item.get("severity"),
            item.get("path"),
            item.get("line"),
        )
        group = groups.setdefault(
            key,
            {
                "stage": item.get("stage"),
                "code": item.get("code"),
                "severity": item.get("severity", "error"),
                "message": str(item.get("message") or "Surface validation failed"),
                "path": item.get("path"),
                "line": item.get("line"),
                "occurrences": 0,
                "viewports": [],
                "states": [],
                "examples": [],
            },
        )
        group["occurrences"] += 1
        viewport = item.get("viewport")
        if viewport is not None and viewport not in group["viewports"]:
            group["viewports"].append(viewport)
        state = item.get("state")
        if state is not None and state not in group["states"]:
            group["states"].append(state)
        message = str(item.get("message") or "")
        if message and message != group["message"] and message not in group["examples"]:
            if len(group["examples"]) < 4:
                group["examples"].append(message)
    return list(groups.values())[:40]


async def _progress_heartbeat(
    run_id: str,
    worker_id: str,
    *,
    stage: str,
    submission: int,
    generation_progress: _GenerationProgress | None = None,
) -> None:
    """Keep long non-streaming model generation visibly alive."""
    try:
        while True:
            await asyncio.sleep(SURFACE_PROGRESS_HEARTBEAT_SECONDS)
            async with async_session() as heartbeat_session:
                heartbeat_run = await heartbeat_session.get(Run, run_id)
                if (
                    heartbeat_run is None
                    or heartbeat_run.status != "running"
                    or heartbeat_run.lease_owner != worker_id
                ):
                    return
                heartbeat_run.progress = {
                    **(heartbeat_run.progress or {}),
                    "stage": stage,
                    "submission": submission,
                    **(
                        generation_progress.payload()
                        if generation_progress is not None
                        else {}
                    ),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                heartbeat_session.add(heartbeat_run)
                await heartbeat_session.commit()
    except asyncio.CancelledError:
        return


async def _stop_progress_heartbeat(heartbeat: asyncio.Task[None]) -> None:
    """Cancel and join a heartbeat even when cancellation wins task startup."""
    heartbeat.cancel()
    try:
        await heartbeat
    except asyncio.CancelledError:
        # A task cancelled before its coroutine starts cannot handle the
        # cancellation inside ``_progress_heartbeat``. Cancellation is the
        # expected shutdown path here, so the owner must absorb it as well.
        return


async def _await_candidate_generation(
    invocation: Any,
    *,
    progress: _GenerationProgress,
    first_output_timeout_seconds: float,
    deadline: float,
) -> Any:
    """Await a structured stream with first-output, idle, and Run deadlines."""
    task = asyncio.create_task(invocation)
    started = perf_counter()
    try:
        while True:
            now = perf_counter()
            remaining = deadline - now
            if remaining <= 0:
                raise BudgetExceeded(
                    "Duration budget exceeded",
                    code="max_duration_seconds",
                )
            if progress.first_output_monotonic is None:
                progress_timeout = first_output_timeout_seconds - (now - started)
                if progress_timeout <= 0:
                    raise SurfaceGenerationTimeoutError(
                        "Surface generation produced no output before the "
                        f"{first_output_timeout_seconds:g}-second deadline"
                    )
            else:
                last_output = progress.last_output_monotonic or now
                progress_timeout = SURFACE_STREAM_IDLE_TIMEOUT_SECONDS - (
                    now - last_output
                )
                if progress_timeout <= 0:
                    raise SurfaceGenerationTimeoutError(
                        "Surface generation stopped producing output for "
                        f"{SURFACE_STREAM_IDLE_TIMEOUT_SECONDS:g} seconds"
                    )
            done, _ = await asyncio.wait(
                {task},
                timeout=min(1.0, remaining, progress_timeout),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if done:
                return await task
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


async def _execute_claimed_surface_builder_legacy(
    run_id: str,
    worker_id: str,
    *,
    llm_factory: Callable[[Any], Any] = create_llm,
    structured_invoker: Callable[..., Any] = ainvoke_structured,
    runtime_resolver: Callable[..., Any] = resolve_surface_authoring_runtime,
    pipeline_factory: Callable[..., SurfaceHostPipeline] = SurfaceHostPipeline,
) -> bool:
    """Generate, repair if necessary, and publish one trusted Surface Run."""
    async with async_session() as session:
        run = await session.get(Run, run_id)
        if (
            run is None
            or run.run_kind != SURFACE_BUILDER_RUN_KIND
            or run.status != "running"
            or run.lease_owner != worker_id
        ):
            return False
        claimed_run: Run = run
        assignment: ModelAssignment | None = None
        total_usage = TokenUsage()
        generation_ms = 0.0
        pipeline_timings: list[dict[str, float]] = []
        receipts: list[dict[str, Any]] = [
            dict(receipt)
            for receipt in run.execution_receipts or []
            if receipt.get("kind") != "model_assignment"
        ]
        submissions = 0
        deadline = perf_counter()
        budget_ledger: BudgetLedger | None = None
        try:
            organization = await session.get(Organization, run.organization_id)
            membership = (
                (
                    await session.execute(
                        select(OrganizationMembership).where(
                            OrganizationMembership.organization_id
                            == run.organization_id,
                            OrganizationMembership.user_id == run.user_id,
                            OrganizationMembership.status == "active",
                        )
                    )
                )
                .scalars()
                .first()
            )
            if organization is None or membership is None:
                raise PermissionError("Run owner no longer has organization access")
            policy = OrganizationPolicy.model_validate(organization.policy or {})
            assignment = _validate_assignment(run, policy)
            budget_ledger = budget_ledger_for_run(run)
            budget_ledger.start_clock()
            remaining_timeout = remaining_run_seconds(run)
            if remaining_timeout <= 0:
                raise BudgetExceeded(
                    "Duration budget exceeded",
                    code="max_duration_seconds",
                )
            deadline = perf_counter() + remaining_timeout
            if await mark_surface_run_started(session, run=run):
                await session.commit()
            await record_surface_builder_started(session, run=run)
            run.progress = {
                **(run.progress or {}),
                "stage": "generating_candidate",
                "submission": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            session.add(run)
            await session.commit()

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
                max_tokens=run.max_tokens,
                max_cost_usd=run.max_cost_usd,
                max_duration_seconds=float(run.timeout_seconds),
                isolation_profile="worker-process",
            )
            runtime = await runtime_resolver(
                session,
                run_context=run_context,
                session_factory=async_session,
            )
            project_snapshot = dict(
                getattr(runtime, "project_snapshot", None)
                or runtime.prompt_context.get("surface")
                or {}
            )
            draft_snapshot = dict(project_snapshot.get("draft") or {})
            base_files = {
                str(path): str(content)
                for path, content in dict(draft_snapshot.get("files") or {}).items()
            }
            base_revision_id = project_snapshot.get("expected_revision")
            candidate_mode = (
                "edit" if base_revision_id is not None and base_files else "complete"
            )
            context = _render_prompt_context(runtime.prompt_context)
            instructions = surface_builder_instructions()
            prompt_fingerprint = stable_fingerprint(
                {
                    "profile": SURFACE_BUILDER_RUN_PROFILE.fingerprint,
                    "skill_id": SURFACE_BUILDER_SKILL_ID,
                    "task": run.task,
                    "context": context,
                    "candidate_mode": candidate_mode,
                    "base_revision_id": base_revision_id,
                    "model_assignment": assignment.config_fingerprint,
                }
            )
            llm = ensure_budgeted_chat_model(
                llm_factory(assignment.llm_config()),
                budget_ledger,
            )
            pipeline_progress: dict[str, Any] = {
                "submission": 1,
                "candidate_fingerprint": None,
            }

            async def observe_stage(stage: str) -> None:
                claimed_run.progress = {
                    **(claimed_run.progress or {}),
                    "stage": stage,
                    "submission": pipeline_progress["submission"],
                    "candidate_fingerprint": pipeline_progress["candidate_fingerprint"],
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                session.add(claimed_run)
                await session.commit()

            previous_candidate: BaseModel | None = None
            diagnostics: list[dict[str, Any]] = []
            published: SurfacePipelineResult | None = None
            seen_candidate_fingerprints: set[str] = set()
            required_primary_jobs = surface_request_primary_jobs(run)
            required_primary_job_descriptions = [
                item["description"] for item in required_primary_jobs
            ]

            for submission in range(1, MAX_CANDIDATE_SUBMISSIONS + 1):
                submissions = submission
                prior_diagnostics = list(diagnostics)
                budget_ledger.consume_step()
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    raise BudgetExceeded(
                        "Duration budget exceeded",
                        code="max_duration_seconds",
                    )
                started = perf_counter()
                generation_progress = _GenerationProgress()
                claimed_run.progress = {
                    **(claimed_run.progress or {}),
                    "stage": "generating_candidate",
                    "submission": submission,
                    "candidate_mode": candidate_mode,
                    "base_revision_id": base_revision_id,
                    **generation_progress.payload(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                session.add(claimed_run)
                await session.commit()
                heartbeat = asyncio.create_task(
                    _progress_heartbeat(
                        run.id,
                        worker_id,
                        stage="generating_candidate",
                        submission=submission,
                        generation_progress=generation_progress,
                    )
                )
                try:
                    generation_timeout = min(
                        remaining,
                        float(assignment.generation_timeout_seconds),
                    )
                    output_schema = (
                        SurfaceCandidateEditEnvelope
                        if candidate_mode == "edit"
                        else SurfaceCandidateEnvelope
                    )
                    llm_calls_before = budget_ledger.snapshot()["llm_calls_used"]
                    try:
                        response = await _await_candidate_generation(
                            structured_invoker(
                                llm,
                                output_schema,
                                _messages(
                                    task=run.task,
                                    context=context,
                                    instructions=instructions,
                                    previous_candidate=previous_candidate,
                                    diagnostics=diagnostics,
                                    repair_files=(
                                        base_files
                                        if previous_candidate is not None
                                        else None
                                    ),
                                    candidate_mode=candidate_mode,
                                    required_primary_jobs=required_primary_jobs,
                                ),
                                include_raw=True,
                                on_delta=generation_progress.on_delta,
                                meta={
                                    "run_id": run.id,
                                    "run_kind": SURFACE_BUILDER_RUN_KIND,
                                    "submission": submission,
                                    "candidate_mode": candidate_mode,
                                    "base_revision_id": base_revision_id,
                                    "profile": SURFACE_BUILDER_RUN_PROFILE.profile_id,
                                    "generation_timeout_seconds": generation_timeout,
                                },
                            ),
                            progress=generation_progress,
                            first_output_timeout_seconds=generation_timeout,
                            deadline=deadline,
                        )
                    except SurfaceGenerationTimeoutError:
                        receipts.append(
                            {
                                "kind": "surface_generation_timeout",
                                "worker_attempt": run.attempt_count,
                                "submission": submission,
                                "candidate_mode": candidate_mode,
                                "timeout_seconds": generation_timeout,
                                **generation_progress.payload(),
                            }
                        )
                        raise
                    # Production's structured invoker calls the budgeted model
                    # and charges itself. Test/custom invokers may return a
                    # provider result directly; charge that completed call once
                    # at this host boundary instead of leaving an escape hatch.
                    if budget_ledger.snapshot()["llm_calls_used"] == llm_calls_before:
                        llm.charge_completed_call()
                finally:
                    await _stop_progress_heartbeat(heartbeat)
                generation_ms += (perf_counter() - started) * 1000
                raw_usage = normalize_usage(getattr(llm, "last_usage", None))
                total_usage += TokenUsage(
                    input_tokens=raw_usage.input_tokens,
                    output_tokens=raw_usage.output_tokens,
                    total_tokens=raw_usage.total_tokens,
                    cache_read_tokens=raw_usage.cache_read_tokens,
                    cache_write_tokens=raw_usage.cache_write_tokens,
                )
                parsed = response.get("parsed") if isinstance(response, dict) else None
                if parsed is None:
                    diagnostic = _rejected_output_diagnostic(response)
                    receipts.append(
                        {
                            "kind": "surface_candidate_rejected",
                            "worker_attempt": run.attempt_count,
                            "submission": submission,
                            "diagnostic": diagnostic,
                        }
                    )
                    raise SurfaceCandidateOutputError(diagnostic)
                parsed_value = (
                    parsed.model_dump(mode="python")
                    if isinstance(parsed, BaseModel)
                    else parsed
                )
                envelope: BaseModel | None = None
                changed_file_count = 0
                try:
                    if candidate_mode == "edit":
                        edit_envelope = (
                            parsed
                            if isinstance(parsed, SurfaceCandidateEditEnvelope)
                            else SurfaceCandidateEditEnvelope.model_validate(
                                parsed_value
                            )
                        )
                        envelope = edit_envelope
                        changed_file_count = len(edit_envelope.changes)
                        candidate = materialize_surface_candidate_edit(
                            edit_envelope,
                            base_files=base_files,
                            primary_jobs=required_primary_job_descriptions,
                        )
                    else:
                        complete_envelope = (
                            parsed
                            if isinstance(parsed, SurfaceCandidateEnvelope)
                            else SurfaceCandidateEnvelope.model_validate(parsed_value)
                        )
                        envelope = complete_envelope
                        changed_file_count = len(complete_envelope.files)
                        candidate = SurfaceCandidate.model_validate(
                            {
                                "summary": complete_envelope.summary,
                                "primary_jobs": required_primary_job_descriptions,
                                "files": [
                                    item.model_dump(mode="python")
                                    for item in complete_envelope.files
                                ],
                            }
                        )
                    candidate = bind_surface_manifest_primary_jobs(
                        candidate,
                        required_primary_jobs=required_primary_jobs,
                    )
                except ValidationError as exc:
                    diagnostics = _candidate_contract_diagnostics(exc)
                except ValueError as exc:
                    diagnostics = [
                        {
                            "stage": "candidate_validation",
                            "code": "invalid_surface_candidate",
                            "severity": "error",
                            "message": str(exc),
                        }
                    ]
                else:
                    diagnostics = []
                if diagnostics:
                    receipts.append(
                        {
                            "kind": "surface_candidate_contract_rejected",
                            "worker_attempt": run.attempt_count,
                            "submission": submission,
                            "candidate_mode": candidate_mode,
                            "diagnostics": diagnostics,
                        }
                    )
                    previous_candidate = envelope or (
                        parsed if isinstance(parsed, BaseModel) else None
                    )
                    if submission >= MAX_CANDIDATE_SUBMISSIONS:
                        run.progress = {
                            **(run.progress or {}),
                            "stage": "candidate_rejected",
                            "submission": submission,
                            "diagnostics": diagnostics,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        }
                        session.add(run)
                        await session.commit()
                        break
                    run.progress = {
                        **(run.progress or {}),
                        "stage": "repairing_candidate",
                        "submission": submission + 1,
                        "diagnostics": diagnostics,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    session.add(run)
                    await session.commit()
                    continue

                if candidate.fingerprint in seen_candidate_fingerprints:
                    diagnostics = [
                        {
                            "stage": "candidate_validation",
                            "code": "duplicate_surface_candidate",
                            "severity": "error",
                            "message": (
                                "The candidate is byte-identical to an already "
                                "rejected submission and cannot repair the listed "
                                "diagnostics. Change the smallest relevant source "
                                "file before resubmitting."
                            ),
                        },
                        *prior_diagnostics,
                    ]
                    receipts.append(
                        {
                            "kind": "surface_candidate_duplicate",
                            "worker_attempt": run.attempt_count,
                            "submission": submission,
                            "candidate_mode": candidate_mode,
                            "candidate_fingerprint": candidate.fingerprint,
                            "diagnostics": diagnostics,
                        }
                    )
                    previous_candidate = envelope
                    run.progress = {
                        **(run.progress or {}),
                        "stage": "candidate_rejected",
                        "submission": submission,
                        "candidate_fingerprint": candidate.fingerprint,
                        "diagnostics": diagnostics,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    session.add(run)
                    await session.commit()
                    # A byte-identical repair proves that the assignment did
                    # not respond to the exhaustive diagnostic bundle. A
                    # further paid call would be a blind retry.
                    break
                seen_candidate_fingerprints.add(candidate.fingerprint)

                run.progress = {
                    **(run.progress or {}),
                    "stage": "validating_candidate",
                    "submission": submission,
                    "candidate_fingerprint": candidate.fingerprint,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                session.add(run)
                await session.commit()
                pipeline_progress.update(
                    submission=submission,
                    candidate_fingerprint=candidate.fingerprint,
                )
                pipeline = pipeline_factory(
                    run_id=run.id,
                    authoring_handler=runtime.authoring_handler,
                    build_handler=runtime.build_handler,
                    stage_observer=observe_stage,
                    required_primary_jobs=required_primary_jobs,
                    expected_base_revision_id=base_revision_id,
                )
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    raise BudgetExceeded(
                        "Duration budget exceeded",
                        code="max_duration_seconds",
                    )
                try:
                    outcome = await asyncio.wait_for(
                        pipeline.execute(candidate, submission=submission),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError as exc:
                    raise BudgetExceeded(
                        "Duration budget exceeded",
                        code="max_duration_seconds",
                    ) from exc
                pipeline_timings.append(outcome.timings_ms)
                receipts.append(
                    {
                        "kind": "surface_candidate",
                        "submission": submission,
                        "candidate_mode": candidate_mode,
                        "changed_file_count": changed_file_count,
                        **generation_progress.payload(),
                        **outcome.model_dump(mode="json"),
                    }
                )
                if outcome.status == "published":
                    published = outcome
                    break
                if outcome.status == "host_failed":
                    raise SurfacePipelineInfrastructureError(
                        outcome.diagnostics,
                        revision_id=outcome.revision_id,
                        build_id=outcome.build_id,
                    )
                previous_candidate = envelope
                diagnostics = outcome.diagnostics
                if outcome.revision_id:
                    base_revision_id = outcome.revision_id
                    base_files = {
                        item.source_path: item.content for item in candidate.files
                    }
                    candidate_mode = "edit"
                final_submission = submission >= MAX_CANDIDATE_SUBMISSIONS
                run.progress = {
                    **(run.progress or {}),
                    "stage": (
                        "candidate_rejected"
                        if final_submission
                        else "repairing_candidate"
                    ),
                    "submission": submission if final_submission else submission + 1,
                    "diagnostics": diagnostics,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                session.add(run)
                await session.commit()

            if published is None:
                raise SurfaceCandidateExhaustedError(
                    "Surface candidate repair budget was exhausted"
                )
            async with async_session() as receipt_session:
                surface_receipt = await verified_surface_publication(
                    receipt_session,
                    run=run,
                )
            if surface_receipt is None:
                raise ValueError("Host pipeline produced no verified live publication")

            await session.refresh(run, attribute_names=["cancel_requested"])
            run.status = "cancelled" if run.cancel_requested else "completed"
            run.success = not run.cancel_requested
            budget_usage = budget_ledger.snapshot()
            run.steps_taken = int(budget_usage["steps_used"])
            run.final_answer = (
                "Your Event Surface is ready. Open it beside this conversation "
                "to use the new visual workspace."
            )
            run.reasoning = "Host-owned structured Surface pipeline completed."
            run.metrics = _usage_metrics(
                assignment,
                total_usage,
                generation_ms=generation_ms,
                pipeline_timings=pipeline_timings,
                started_at=run.started_at,
            )
            run.metrics["budget_usage"] = budget_usage
            run.prompt_fingerprint = prompt_fingerprint
            run.tool_surface_fingerprint = stable_fingerprint(
                {"model_tools": [], "host_pipeline": "surface-host-v1"}
            )
            run.execution_receipts = [
                {"kind": "model_assignment", **assignment.descriptor()},
                *receipts,
                {"kind": "surface_publication", **surface_receipt},
            ]
            run.selected_skills = [SURFACE_BUILDER_SKILL_ID]
            run.artifacts = []
            run.plan = []
            run.progress = {
                **(run.progress or {}),
                "stage": "published",
                "submission": submissions,
                "surface_receipt": surface_receipt,
                "budget_usage": budget_usage,
                "budget_accounted_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            run.completed_at = datetime.now(timezone.utc)
            run.lease_owner = None
            run.lease_expires_at = None
            session.add(run)
            usage_record = make_usage_record(
                organization_id=run.organization_id,
                user_id=run.user_id,
                run_id=run.id,
                conversation_id=run.conversation_id,
                metrics=run.metrics,
            )
            if usage_record is not None:
                session.add(usage_record)
            outcome_message: Message | None = None
            if run.conversation_id and run.status == "completed":
                conversation = await session.get(Conversation, run.conversation_id)
                if (
                    conversation is not None
                    and conversation.organization_id == run.organization_id
                    and conversation.user_id == run.user_id
                ):
                    outcome_message = Message(
                        conversation_id=conversation.id,
                        role="assistant",
                        content=run.final_answer,
                        metadata_={
                            "kind": "surface_build_result",
                            "status": "published",
                            "run_id": run.id,
                            "surface_receipt": surface_receipt,
                            "metrics": run.metrics,
                        },
                    )
                    session.add(outcome_message)
                    conversation.updated_at = datetime.now(timezone.utc)
                    session.add(conversation)
            await reconcile_surface_run(
                session,
                run=run,
                outcome_message=outcome_message,
            )
            await session.commit()
            return True
        except Exception as exc:
            logger.exception("Surface Builder Run %s failed", run_id)
            await session.rollback()
            run = await session.get(Run, run_id)
            if run is None or run.lease_owner != worker_id:
                return False
            budget_exhaustion = exc if isinstance(exc, BudgetExceeded) else None
            terminal = (
                budget_exhaustion is not None
                or isinstance(
                    exc,
                    (
                        SurfaceCandidateExhaustedError,
                        SurfaceCandidateOutputError,
                        SurfaceGenerationTimeoutError,
                        SurfacePipelineInfrastructureError,
                    ),
                )
                or (run.attempt_count >= run.max_attempts)
            )
            run.status = "failed" if terminal else "pending"
            run.success = False
            run.steps_taken = submissions
            if assignment is not None:
                run.metrics = _usage_metrics(
                    assignment,
                    total_usage,
                    generation_ms=generation_ms,
                    pipeline_timings=pipeline_timings,
                    started_at=run.started_at,
                )
                run.execution_receipts = [
                    {"kind": "model_assignment", **assignment.descriptor()},
                    *receipts,
                ]
            if budget_ledger is not None:
                budget_usage = budget_ledger.snapshot()
                run.steps_taken = int(budget_usage["steps_used"])
                run.metrics = {
                    **(run.metrics or {}),
                    "budget_usage": budget_usage,
                }
                if budget_exhaustion is not None:
                    budget_receipt = {
                        "kind": "run_budget",
                        "status": "exhausted",
                        "reason": budget_exhaustion.code,
                        "error": str(budget_exhaustion),
                        "usage": budget_usage,
                    }
                    run.metrics["budget_gate"] = budget_receipt
                    run.execution_receipts = [
                        *(run.execution_receipts or []),
                        budget_receipt,
                    ]
            if budget_exhaustion is not None:
                run.final_answer = (
                    "The Event Surface build stopped at its configured execution "
                    "limit. Your last working Surface is unchanged."
                )
            elif isinstance(exc, SurfacePipelineInfrastructureError):
                run.final_answer = (
                    "Aloy generated the Surface source, but the trusted host "
                    "could not retain or publish its build. Your last working "
                    "Surface is unchanged."
                )
            else:
                run.final_answer = (
                    "I could not safely publish the new Event Surface. Your last "
                    "working Surface is unchanged."
                    if terminal
                    else "The Event Surface build will retry safely."
                )
            rejected = next(
                (
                    receipt.get("diagnostic")
                    for receipt in reversed(receipts)
                    if receipt.get("kind") == "surface_candidate_rejected"
                ),
                None,
            )
            host_diagnostics = (
                exc.diagnostics
                if isinstance(exc, SurfacePipelineInfrastructureError)
                else None
            )
            run.reasoning = (
                str(budget_exhaustion)
                if budget_exhaustion is not None
                else (
                    str(host_diagnostics[0].get("message"))
                    if host_diagnostics
                    else (
                        str(rejected.get("error"))
                        if isinstance(rejected, dict) and rejected.get("error")
                        else "The structured candidate or trusted pipeline failed."
                    )
                )
            )
            run.progress = {
                **(run.progress or {}),
                "stage": "failed" if terminal else "retry_scheduled",
                "submissions": submissions,
                "diagnostic": (
                    host_diagnostics[0]
                    if host_diagnostics
                    else (
                        {
                            key: value
                            for key, value in rejected.items()
                            if key not in {"raw_head", "raw_tail"}
                        }
                        if isinstance(rejected, dict)
                        else None
                    )
                ),
                **(
                    {
                        "budget_usage": budget_ledger.snapshot(),
                        "budget_accounted_at": datetime.now(timezone.utc).isoformat(),
                    }
                    if budget_ledger is not None
                    else {}
                ),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            run.completed_at = datetime.now(timezone.utc) if terminal else None
            run.lease_owner = None
            run.lease_expires_at = None
            session.add(run)
            if budget_exhaustion is not None:
                session.add(
                    EventTrailEntry(
                        organization_id=run.organization_id,
                        user_id=run.user_id,
                        event_id=run.event_id,
                        actor_id="aloy:surface-builder",
                        kind="run_budget_exhausted",
                        summary="Stopped a Surface build at its execution limit",
                        run_id=run.id,
                        payload={"reason": budget_exhaustion.code},
                    )
                )
            await record_surface_builder_failure(session, run=run, terminal=terminal)
            if terminal and run.conversation_id:
                conversation = await session.get(Conversation, run.conversation_id)
                if conversation is not None:
                    session.add(
                        Message(
                            conversation_id=conversation.id,
                            role="assistant",
                            content=run.final_answer,
                            metadata_={
                                "kind": "surface_build_result",
                                "status": "failed",
                                "run_id": run.id,
                            },
                        )
                    )
                    conversation.updated_at = datetime.now(timezone.utc)
                    session.add(conversation)
            if terminal:
                usage_record = make_usage_record(
                    organization_id=run.organization_id,
                    user_id=run.user_id,
                    run_id=run.id,
                    conversation_id=run.conversation_id,
                    metrics=run.metrics,
                )
                if usage_record is not None:
                    session.add(usage_record)
                await reconcile_surface_run(
                    session,
                    run=run,
                    error="The Surface candidate exhausted its safe repair budget.",
                )
            await session.commit()
            return True


async def execute_claimed_surface_builder(
    run_id: str,
    worker_id: str,
    *,
    llm_factory: Callable[[Any], Any] = create_llm,
    structured_invoker: Callable[..., Any] | None = None,
    runtime_resolver: Callable[..., Any] = resolve_surface_authoring_runtime,
    pipeline_factory: Callable[..., SurfaceHostPipeline] = SurfaceHostPipeline,
) -> bool:
    """Execute the workspace Builder, retaining v3 injection compatibility.

    Focused lifecycle tests may inject the schema-bound invoker explicitly.
    Normal worker calls omit it and use the provider-neutral development
    workspace.
    """
    if structured_invoker is not None:
        return await _execute_claimed_surface_builder_legacy(
            run_id,
            worker_id,
            llm_factory=llm_factory,
            structured_invoker=structured_invoker,
            runtime_resolver=runtime_resolver,
            pipeline_factory=pipeline_factory,
        )

    captured_runtime: Any | None = None
    workspace: SurfaceDevelopmentWorkspace | None = None
    workspace_receipts: list[dict[str, Any]] = []
    previous_files: dict[str, str] | None = None
    invocation = 0

    async with async_session() as preflight_session:
        preflight_run = await preflight_session.get(Run, run_id)
        if (
            preflight_run is None
            or preflight_run.run_kind != SURFACE_BUILDER_RUN_KIND
            or preflight_run.status != "running"
            or preflight_run.lease_owner != worker_id
        ):
            return False
        task = preflight_run.task
        assignment_value = ModelAssignment.model_validate(
            preflight_run.model_assignment
        )
        capabilities = set(assignment_value.capabilities)
        required_jobs = surface_request_primary_jobs(preflight_run)
        primary_job_descriptions = [item["description"] for item in required_jobs]

    async def capture_runtime(*args: Any, **kwargs: Any) -> Any:
        nonlocal captured_runtime, previous_files
        captured_runtime = await runtime_resolver(*args, **kwargs)
        project = dict(getattr(captured_runtime, "project_snapshot", None) or {})
        draft = dict(project.get("draft") or {})
        previous_files = {
            str(path): str(content)
            for path, content in dict(draft.get("files") or {}).items()
        }
        return captured_runtime

    async def workspace_invoker(
        llm: Any,
        output_model: type[BaseModel],
        legacy_messages: list[Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        nonlocal workspace, previous_files, invocation
        if captured_runtime is None or previous_files is None:
            raise RuntimeError("Surface development runtime was not resolved")
        if workspace is None:
            workspace = await LocalGitSurfaceWorkspace.create(
                workspace_id=run_id,
                base_files=previous_files,
                build_runner=captured_runtime.workspace_build_runner,
            )
        invocation += 1
        if invocation == 1:
            user_context = (
                task
                + "\n\nTrusted Event and current Surface context:\n"
                + _render_prompt_context(captured_runtime.prompt_context)
                + "\n\nHost-owned acceptance jobs:\n"
                + _json(required_jobs)
            )
        else:
            legacy_text = str(legacy_messages[-1].content)
            diagnostic_text = legacy_text
            marker = "\nDiagnostics:\n"
            source_marker = "\nExact current rejected source:\n"
            if marker in legacy_text:
                diagnostic_text = legacy_text.split(marker, 1)[1]
            if source_marker in diagnostic_text:
                diagnostic_text = diagnostic_text.split(source_marker, 1)[0]
            user_context = (
                task
                + "\n\nThe full host gate rejected the checked workspace. Repair every "
                "diagnostic below against the current workspace, then finish again.\n"
                + diagnostic_text[:120_000]
            )

        on_delta = kwargs.get("on_delta")

        async def progress(update: dict[str, Any]) -> None:
            if callable(on_delta):
                on_delta(
                    f"workspace turn {update.get('turn', 0)}; "
                    f"tool calls {update.get('tool_calls', 0)}\n"
                )

        ledger = getattr(llm, "budget_ledger", None)
        remaining_turns = 20
        if isinstance(ledger, BudgetLedger):
            usage = ledger.snapshot()
            max_steps = usage.get("max_steps")
            if isinstance(max_steps, int):
                remaining_turns = max(1, min(20, max_steps - int(usage["steps_used"])))
        loop_result = await run_surface_builder_loop(
            llm=llm,
            workspace=workspace,
            messages=[
                SystemMessage(
                    content=(
                        SURFACE_BUILDER_RUN_PROFILE.system_prompt
                        + "\n\nApply this exact Aloy Builder skill:\n\n"
                        + surface_builder_instructions()
                    )
                ),
                UserMessage(content=user_context),
            ],
            primary_jobs=primary_job_descriptions,
            capabilities=capabilities,
            budget_ledger=ledger if isinstance(ledger, BudgetLedger) else None,
            max_turns=remaining_turns,
            on_progress=progress,
        )
        current_files = {
            item.source_path: item.content
            for item in loop_result.finished.candidate.files
        }
        if invocation > 1 and current_files == previous_files:
            raise SurfaceCandidateExhaustedError(
                "Workspace repair did not change the rejected source"
            )
        receipt = loop_result.finished.receipt
        workspace_receipts.append(
            {
                "kind": "surface_development_workspace",
                "submission": invocation,
                "workspace_id": receipt.workspace_id,
                "base_commit": receipt.base_commit,
                "candidate_commit": receipt.candidate_commit,
                "source_fingerprint": receipt.source_fingerprint,
                "changed_paths": receipt.changed_paths,
                "diff_sha256": hashlib.sha256(
                    receipt.diff_excerpt.encode("utf-8")
                ).hexdigest(),
                "diff_chars": len(receipt.diff_excerpt),
                "turns": loop_result.turns,
                "tool_calls": loop_result.tool_calls,
                "protocol": loop_result.protocol,
            }
        )
        summary = loop_result.finished.candidate.summary
        if output_model is SurfaceCandidateEditEnvelope:
            changes: list[SurfaceCandidateEditEnvelopeFile] = []
            for path, content in sorted(current_files.items()):
                if previous_files.get(path) != content:
                    changes.append(
                        SurfaceCandidateEditEnvelopeFile(
                            path=f"/workspace{path}",
                            operation="write",
                            content=content,
                        )
                    )
            changes.extend(
                SurfaceCandidateEditEnvelopeFile(
                    path=f"/workspace{path}",
                    operation="delete",
                )
                for path in sorted(set(previous_files) - set(current_files))
            )
            parsed: BaseModel = SurfaceCandidateEditEnvelope(
                summary=summary,
                changes=changes,
            )
        elif output_model is SurfaceCandidateEnvelope:
            parsed = SurfaceCandidateEnvelope(
                summary=summary,
                files=[
                    SurfaceCandidateEnvelopeFile(
                        path=f"/workspace{path}",
                        content=content,
                    )
                    for path, content in sorted(current_files.items())
                ],
            )
        else:
            raise SurfaceCandidateOutputError(
                {
                    "error": "Workspace Builder received an unsupported output contract",
                    "output_model": output_model.__name__,
                }
            )
        previous_files = current_files
        return {
            "parsed": parsed,
            "raw": {
                "workspace_commit": receipt.candidate_commit,
                "source_fingerprint": receipt.source_fingerprint,
            },
        }

    try:
        completed = await _execute_claimed_surface_builder_legacy(
            run_id,
            worker_id,
            llm_factory=llm_factory,
            structured_invoker=workspace_invoker,
            runtime_resolver=capture_runtime,
            pipeline_factory=pipeline_factory,
        )
        if workspace_receipts:
            async with async_session() as receipt_session:
                persisted = await receipt_session.get(Run, run_id)
                if persisted is not None:
                    persisted.execution_receipts = [
                        *(persisted.execution_receipts or []),
                        *workspace_receipts,
                    ]
                    metrics = dict(persisted.metrics or {})
                    metrics["structured_output"] = False
                    metrics["tool_calls"] = sum(
                        int(item["tool_calls"]) for item in workspace_receipts
                    )
                    metrics["surface_workspace"] = {
                        "version": "1",
                        "submissions": len(workspace_receipts),
                        "protocols": [
                            str(item["protocol"]) for item in workspace_receipts
                        ],
                    }
                    persisted.metrics = metrics
                    persisted.tool_surface_fingerprint = stable_fingerprint(
                        {
                            "model_tools": [
                                "list_files",
                                "read_file",
                                "search_source",
                                "write_file",
                                "replace_text",
                                "delete_file",
                                "run_typecheck",
                                "run_preview_check",
                                "read_diagnostics",
                                "finish_candidate",
                            ],
                            "host_pipeline": "surface-host-v1",
                            "workspace": "surface-development-workspace-v1",
                        }
                    )
                    if persisted.status == "completed":
                        persisted.reasoning = (
                            "The Git-tracked Surface workspace passed its local "
                            "compiler loop and the full host publication gate."
                        )
                    receipt_session.add(persisted)
                    await receipt_session.commit()
        return completed
    finally:
        if workspace is not None:
            await workspace.close()


__all__ = [
    "MAX_SURFACE_PROMPT_CONTEXT_CHARS",
    "SurfaceCandidateExhaustedError",
    "SurfaceGenerationTimeoutError",
    "execute_claimed_surface_builder",
]
