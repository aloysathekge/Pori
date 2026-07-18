"""Dedicated no-tool Surface Builder execution.

The Builder is a single structured-output model role. It returns complete
candidate source; the trusted host executes the lifecycle and may provide
deterministic diagnostics for one bounded repair call. The model never receives
authoring, build, preview, publication, filesystem, or answer tools.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ValidationError
from sqlmodel import select

from pori import (
    SystemMessage,
    TokenUsage,
    UserMessage,
    ainvoke_structured,
    create_llm,
    estimate_llm_call_cost,
    normalize_usage,
    stable_fingerprint,
)

from .database import async_session
from .model_roles import ModelAssignment, ModelRole
from .models import (
    Conversation,
    Message,
    Organization,
    OrganizationMembership,
    Run,
)
from .run_outcome import make_usage_record
from .run_profiles import SURFACE_BUILDER_RUN_PROFILE
from .runtime import authenticated_run_context
from .skills import SURFACE_BUILDER_SKILL_ID, surface_builder_instructions
from .surface_lifecycle import mark_surface_run_started, reconcile_surface_run
from .surface_pipeline import (
    MAX_CANDIDATE_SUBMISSIONS,
    SurfaceCandidate,
    SurfaceCandidateEnvelope,
    SurfaceHostPipeline,
    SurfacePipelineResult,
)
from .surface_requests import (
    SURFACE_BUILDER_RUN_KIND,
    record_surface_builder_failure,
    record_surface_builder_started,
    verified_surface_publication,
)
from .surface_workspace import resolve_surface_authoring_runtime
from .tenancy import OrganizationPolicy

logger = logging.getLogger("aloy_backend.surface_builder")

MAX_SURFACE_PROMPT_CONTEXT_CHARS = 700_000
SURFACE_PROGRESS_HEARTBEAT_SECONDS = 5.0
MAX_REJECTED_OUTPUT_EXCERPT_CHARS = 24_000


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


def _render_prompt_context(value: dict[str, Any]) -> str:
    """Keep structured generation bounded and refuse unsafe partial revisions."""
    rendered = _json(value)
    if len(rendered) <= MAX_SURFACE_PROMPT_CONTEXT_CHARS:
        return rendered
    # Event file excerpts and old Trail entries are useful but less important
    # than complete current source. Trim those before refusing the request.
    reduced = {
        **value,
        "file_excerpts": {
            path: content[:12_000]
            for path, content in list(dict(value.get("file_excerpts") or {}).items())[
                :20
            ]
        },
        "trail": list(value.get("trail") or [])[:100],
    }
    rendered = _json(reduced)
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
) -> list[Any]:
    repair = ""
    if previous_candidate is not None:
        repair = (
            "\n\nThe host rejected the previous complete candidate. Return a new "
            "complete replacement that repairs every diagnostic.\nDiagnostics:\n"
            + _json(diagnostics)
            + "\nPrevious candidate:\n"
            + previous_candidate.model_dump_json()
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
                + "\n\nTrusted Event and current Surface context:\n"
                + context
                + repair
            )
        ),
    ]


async def _progress_heartbeat(
    run_id: str,
    worker_id: str,
    *,
    stage: str,
    submission: int,
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
                    "stage": stage,
                    "submission": submission,
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


async def execute_claimed_surface_builder(
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
        deadline = perf_counter() + run.timeout_seconds
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
            if await mark_surface_run_started(session, run=run):
                await session.commit()
            await record_surface_builder_started(session, run=run)
            run.progress = {
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
                max_steps=MAX_CANDIDATE_SUBMISSIONS,
                isolation_profile="worker-process",
            )
            runtime = await runtime_resolver(
                session,
                run_context=run_context,
                session_factory=async_session,
            )
            context = _render_prompt_context(runtime.prompt_context)
            instructions = surface_builder_instructions()
            prompt_fingerprint = stable_fingerprint(
                {
                    "profile": SURFACE_BUILDER_RUN_PROFILE.fingerprint,
                    "skill_id": SURFACE_BUILDER_SKILL_ID,
                    "task": run.task,
                    "context": context,
                    "model_assignment": assignment.config_fingerprint,
                }
            )
            llm = llm_factory(assignment.llm_config())
            pipeline_progress: dict[str, Any] = {
                "submission": 1,
                "candidate_fingerprint": None,
            }

            async def observe_stage(stage: str) -> None:
                claimed_run.progress = {
                    "stage": stage,
                    "submission": pipeline_progress["submission"],
                    "candidate_fingerprint": pipeline_progress["candidate_fingerprint"],
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                session.add(claimed_run)
                await session.commit()

            pipeline = pipeline_factory(
                run_id=run.id,
                authoring_handler=runtime.authoring_handler,
                build_handler=runtime.build_handler,
                stage_observer=observe_stage,
            )
            previous_candidate: BaseModel | None = None
            diagnostics: list[dict[str, Any]] = []
            published: SurfacePipelineResult | None = None

            for submission in range(1, MAX_CANDIDATE_SUBMISSIONS + 1):
                submissions = submission
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    raise TimeoutError("Surface Builder Run exhausted its time budget")
                started = perf_counter()
                heartbeat = asyncio.create_task(
                    _progress_heartbeat(
                        run.id,
                        worker_id,
                        stage="generating_candidate",
                        submission=submission,
                    )
                )
                try:
                    generation_timeout = min(
                        remaining,
                        float(assignment.generation_timeout_seconds),
                    )
                    try:
                        response = await asyncio.wait_for(
                            structured_invoker(
                                llm,
                                SurfaceCandidateEnvelope,
                                _messages(
                                    task=run.task,
                                    context=context,
                                    instructions=instructions,
                                    previous_candidate=previous_candidate,
                                    diagnostics=diagnostics,
                                ),
                                include_raw=True,
                                meta={
                                    "run_id": run.id,
                                    "run_kind": SURFACE_BUILDER_RUN_KIND,
                                    "submission": submission,
                                    "profile": SURFACE_BUILDER_RUN_PROFILE.profile_id,
                                    "generation_timeout_seconds": generation_timeout,
                                },
                            ),
                            timeout=generation_timeout,
                        )
                    except TimeoutError as exc:
                        receipts.append(
                            {
                                "kind": "surface_generation_timeout",
                                "worker_attempt": run.attempt_count,
                                "submission": submission,
                                "timeout_seconds": generation_timeout,
                            }
                        )
                        raise SurfaceGenerationTimeoutError(
                            "Surface candidate generation exceeded "
                            f"{generation_timeout:g} seconds"
                        ) from exc
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
                envelope = (
                    parsed
                    if isinstance(parsed, SurfaceCandidateEnvelope)
                    else SurfaceCandidateEnvelope.model_validate(
                        parsed.model_dump(mode="python")
                        if isinstance(parsed, BaseModel)
                        else parsed
                    )
                )
                try:
                    candidate = SurfaceCandidate.model_validate(
                        envelope.model_dump(mode="python")
                    )
                except ValidationError as exc:
                    diagnostics = _candidate_contract_diagnostics(exc)
                    receipts.append(
                        {
                            "kind": "surface_candidate_contract_rejected",
                            "worker_attempt": run.attempt_count,
                            "submission": submission,
                            "diagnostics": diagnostics,
                        }
                    )
                    previous_candidate = envelope
                    run.progress = {
                        "stage": "repairing_candidate",
                        "submission": submission + 1,
                        "diagnostics": diagnostics,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    session.add(run)
                    await session.commit()
                    continue

                run.progress = {
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
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    raise TimeoutError("Surface Builder Run exhausted its time budget")
                outcome = await asyncio.wait_for(
                    pipeline.execute(candidate, submission=submission),
                    timeout=remaining,
                )
                pipeline_timings.append(outcome.timings_ms)
                receipts.append(
                    {
                        "kind": "surface_candidate",
                        "submission": submission,
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
                previous_candidate = candidate
                diagnostics = outcome.diagnostics
                run.progress = {
                    "stage": "repairing_candidate",
                    "submission": submission + 1,
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
            run.steps_taken = submissions
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
                "stage": "published",
                "submission": submissions,
                "surface_receipt": surface_receipt,
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
            terminal = isinstance(
                exc,
                (
                    SurfaceCandidateExhaustedError,
                    SurfaceCandidateOutputError,
                    SurfaceGenerationTimeoutError,
                    SurfacePipelineInfrastructureError,
                ),
            ) or (run.attempt_count >= run.max_attempts)
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
            if isinstance(exc, SurfacePipelineInfrastructureError):
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
                str(host_diagnostics[0].get("message"))
                if host_diagnostics
                else (
                    str(rejected.get("error"))
                    if isinstance(rejected, dict) and rejected.get("error")
                    else "The structured candidate or trusted pipeline failed."
                )
            )
            run.progress = {
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
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            run.completed_at = datetime.now(timezone.utc) if terminal else None
            run.lease_owner = None
            run.lease_expires_at = None
            session.add(run)
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
                await reconcile_surface_run(
                    session,
                    run=run,
                    error="The Surface candidate exhausted its safe repair budget.",
                )
            await session.commit()
            return True


__all__ = [
    "MAX_SURFACE_PROMPT_CONTEXT_CHARS",
    "SurfaceCandidateExhaustedError",
    "SurfaceGenerationTimeoutError",
    "execute_claimed_surface_builder",
]
