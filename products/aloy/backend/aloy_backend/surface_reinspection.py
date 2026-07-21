"""Model-free background reinspection of the currently published Surface."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from .database import async_session
from .models import (
    Event,
    EventTrailEntry,
    Organization,
    Run,
    SurfaceBuild,
    SurfaceInspection,
    SurfaceProject,
)
from .runtime import authenticated_run_context
from .surface_builds import SurfaceBuildHandler, SurfacePreviewParams
from .surface_evolution import SurfaceEvolutionSignal
from .surface_evolution_proposals import record_surface_evolution_signal
from .surface_quality import SURFACE_QUALITY_RECEIPT_KEY
from .tenancy import OrganizationContext, OrganizationPolicy

SURFACE_REINSPECTION_RUN_KIND = "surface_reinspection"
SURFACE_REINSPECTION_AGENT_ID = "surface-inspector"
SURFACE_REINSPECTION_PROFILE = {
    "profile_id": "aloy-surface-reinspection@1",
    "model_tools": [],
    "host_pipeline": "trusted-surface-inspection",
}

logger = logging.getLogger("aloy_backend.surface_reinspection")

_USER_CHECK_LABELS = {
    "runtime_execution": "Opens correctly",
    "declared_interactions": "Buttons and actions",
    "primary_job_simulation": "Main task flow",
    "viewport_matrix": "Desktop and mobile layout",
    "state_matrix": "Loading and error states",
    "accessibility_audit": "Basic accessibility",
    "focus_indicator_audit": "Keyboard navigation",
    "contrast_audit": "Readable text",
}


class SurfaceReinspectionError(ValueError):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _health_checks(receipt: dict[str, Any]) -> list[dict[str, str]]:
    checks = dict(receipt.get("checks") or {})
    return [
        {
            "id": check_id,
            "label": label,
            "status": str(dict(checks.get(check_id) or {}).get("status") or "unknown"),
        }
        for check_id, label in _USER_CHECK_LABELS.items()
        if check_id in checks
    ]


async def surface_health_snapshot(
    session: AsyncSession,
    *,
    context: OrganizationContext,
    event: Event,
) -> dict[str, Any]:
    """Return one user-safe health state for the currently published Surface."""

    project = (
        (
            await session.execute(
                select(SurfaceProject).where(
                    SurfaceProject.organization_id == context.organization_id,
                    SurfaceProject.user_id == context.user_id,
                    SurfaceProject.event_id == event.id,
                )
            )
        )
        .scalars()
        .first()
    )
    if project is None or not project.published_build_id:
        return {
            "status": "no_surface",
            "message": "This Event does not have a live Surface yet.",
            "build_id": None,
            "run_id": None,
            "inspection_id": None,
            "checked_at": None,
            "checks": [],
        }

    build_id = project.published_build_id
    recent_runs = list(
        (
            await session.execute(
                select(Run)
                .where(
                    Run.organization_id == context.organization_id,
                    Run.user_id == context.user_id,
                    Run.event_id == event.id,
                    Run.run_kind == SURFACE_REINSPECTION_RUN_KIND,
                )
                .order_by(col(Run.created_at).desc())
                .limit(25)
            )
        )
        .scalars()
        .all()
    )
    latest_run = next(
        (
            run
            for run in recent_runs
            if str(dict(run.run_profile or {}).get("build_id") or "") == build_id
        ),
        None,
    )
    inspection = (
        (
            await session.execute(
                select(SurfaceInspection)
                .where(
                    SurfaceInspection.organization_id == context.organization_id,
                    SurfaceInspection.user_id == context.user_id,
                    SurfaceInspection.event_id == event.id,
                    SurfaceInspection.build_id == build_id,
                )
                .order_by(col(SurfaceInspection.created_at).desc())
                .limit(1)
            )
        )
        .scalars()
        .first()
    )
    build = await session.get(SurfaceBuild, build_id)
    publication_receipt = dict(
        ((build.resource_metrics if build is not None else {}) or {}).get(
            SURFACE_QUALITY_RECEIPT_KEY
        )
        or {}
    )
    receipt_checks = _health_checks(publication_receipt)
    inspection_checks = (
        [
            {
                "id": str(item.get("id") or ""),
                "label": _USER_CHECK_LABELS.get(
                    str(item.get("id") or ""),
                    str(item.get("id") or "Check").replace("_", " ").title(),
                ),
                "status": str(item.get("status") or "unknown"),
            }
            for item in list(dict(inspection.summary or {}).get("checks") or [])
            if isinstance(item, dict) and item.get("id")
        ]
        if inspection
        else []
    )
    checks = inspection_checks or receipt_checks

    if latest_run is not None and latest_run.status in {"pending", "running"}:
        retrying = dict(latest_run.progress or {}).get("stage") == "retry_scheduled"
        return {
            "status": "checking",
            "message": (
                "The trusted check is temporarily unavailable and will retry."
                if retrying
                else "Aloy is checking the live Surface in the background."
            ),
            "build_id": build_id,
            "run_id": latest_run.id,
            "inspection_id": inspection.id if inspection else None,
            "checked_at": (
                inspection.completed_at or inspection.created_at if inspection else None
            ),
            "checks": checks,
        }
    if latest_run is not None and latest_run.status == "failed":
        return {
            "status": "unavailable",
            "message": "Aloy could not complete the latest check. The current Surface remains live.",
            "build_id": build_id,
            "run_id": latest_run.id,
            "inspection_id": inspection.id if inspection else None,
            "checked_at": latest_run.completed_at,
            "checks": checks,
        }

    if latest_run is not None and latest_run.status == "completed":
        stage = str(dict(latest_run.progress or {}).get("stage") or "")
        if stage in {"passed", "quality_regression"}:
            passed = stage == "passed"
            return {
                "status": "ready" if passed else "needs_improvement",
                "message": (
                    "The live Surface passed its trusted checks."
                    if passed
                    else "Aloy found a problem and prepared a safe improvement."
                ),
                "build_id": build_id,
                "run_id": latest_run.id,
                "inspection_id": dict(latest_run.progress or {}).get("inspection_id"),
                "checked_at": latest_run.completed_at,
                "checks": checks,
            }

    if inspection is not None:
        passed = inspection.status == "passed"
        return {
            "status": "ready" if passed else "needs_improvement",
            "message": (
                "The live Surface passed its trusted checks."
                if passed
                else "Aloy found a problem and prepared a safe improvement."
            ),
            "build_id": build_id,
            "run_id": latest_run.id if latest_run else None,
            "inspection_id": inspection.id,
            "checked_at": inspection.completed_at or inspection.created_at,
            "checks": checks,
        }

    publication_passed = publication_receipt.get("passed") is True
    return {
        "status": "ready" if publication_passed else "unavailable",
        "message": (
            "The live Surface passed its publication checks."
            if publication_passed
            else "Aloy has not completed a trusted check for this Surface yet."
        ),
        "build_id": build_id,
        "run_id": latest_run.id if latest_run else None,
        "inspection_id": None,
        "checked_at": publication_receipt.get("inspected_at"),
        "checks": checks,
    }


async def queue_surface_reinspection(
    session: AsyncSession,
    *,
    context: OrganizationContext,
    event: Event,
    reason: str,
    intent_id: str | None = None,
    actor_id: str | None = None,
) -> tuple[Run, bool]:
    """Queue one model-free inspection, deduplicated while work is active."""

    if event.lifecycle == "archived":
        raise SurfaceReinspectionError(409, "Archived Events cannot be reinspected")
    project_query = select(SurfaceProject).where(
        SurfaceProject.organization_id == context.organization_id,
        SurfaceProject.user_id == context.user_id,
        SurfaceProject.event_id == event.id,
    )
    if session.bind and session.bind.dialect.name == "postgresql":
        project_query = project_query.with_for_update()
    project = (await session.execute(project_query)).scalars().first()
    if (
        project is None
        or not project.published_revision_id
        or not project.published_build_id
    ):
        raise SurfaceReinspectionError(409, "A published Surface is required")
    request_key = (
        f"surface-reinspection:{project.published_build_id}:{intent_id}"
        if intent_id
        else None
    )
    base_query = select(Run).where(
        Run.organization_id == context.organization_id,
        Run.user_id == context.user_id,
        Run.event_id == event.id,
        Run.run_kind == SURFACE_REINSPECTION_RUN_KIND,
    )
    existing = None
    if request_key:
        existing = (
            (
                await session.execute(
                    base_query.where(Run.idempotency_key == request_key).order_by(
                        col(Run.created_at).desc()
                    )
                )
            )
            .scalars()
            .first()
        )
    if existing is None:
        existing = (
            (
                await session.execute(
                    base_query.where(
                        col(Run.status).in_(["pending", "running"])
                    ).order_by(col(Run.created_at).desc())
                )
            )
            .scalars()
            .first()
        )
    if existing is not None:
        return existing, True

    run = Run(
        organization_id=context.organization_id,
        user_id=context.user_id,
        event_id=event.id,
        agent_id=SURFACE_REINSPECTION_AGENT_ID,
        session_id=event.primary_conversation_id or event.id,
        conversation_id=event.primary_conversation_id,
        idempotency_key=request_key
        or (
            f"surface-reinspection:{project.published_build_id}:"
            f"{datetime.now(timezone.utc).isoformat()}"
        ),
        run_kind=SURFACE_REINSPECTION_RUN_KIND,
        run_profile={
            **SURFACE_REINSPECTION_PROFILE,
            "reason": reason,
            "build_id": project.published_build_id,
            "revision_id": project.published_revision_id,
            "data_revision": project.data_revision,
        },
        task="Reinspect the currently published Event Surface with trusted host checks",
        max_steps=1,
        max_tool_calls=1,
        timeout_seconds=min(300, context.policy.run_timeout_seconds),
        max_attempts=min(3, context.policy.max_attempts),
        isolation_profile="worker-process",
        execution_receipts=[
            {
                "kind": "surface_reinspection_request",
                "policy_version": SURFACE_REINSPECTION_PROFILE["profile_id"],
                "build_id": project.published_build_id,
                "revision_id": project.published_revision_id,
                "data_revision": project.data_revision,
                "reason": reason,
                "intent_id": intent_id,
            }
        ],
    )
    session.add(run)
    session.add(
        EventTrailEntry(
            organization_id=context.organization_id,
            user_id=context.user_id,
            event_id=event.id,
            actor_id=actor_id or context.user_id,
            kind="surface_reinspection_queued",
            summary="Queued a trusted check of the live Event Surface",
            run_id=run.id,
            evidence_refs=[{"surface_build_id": project.published_build_id}],
            payload={
                "reason": reason,
                "intent_id": intent_id,
                "build_id": project.published_build_id,
                "revision_id": project.published_revision_id,
                "data_revision": project.data_revision,
            },
        )
    )
    await session.commit()
    await session.refresh(run)
    return run, False


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


async def queue_due_surface_reinspections(
    *,
    now: datetime | None = None,
    interval_seconds: int = 86_400,
    limit: int = 10,
    session_factory: Any = async_session,
) -> int:
    """Queue bounded stale live-build checks; safe for concurrent worker ticks."""

    moment = _aware(now or datetime.now(timezone.utc))
    cutoff = moment - timedelta(seconds=interval_seconds)
    enqueued = 0
    async with session_factory() as session:
        statement = (
            select(SurfaceProject, Event)
            .join(Event, col(Event.id) == col(SurfaceProject.event_id))
            .where(
                SurfaceProject.lifecycle == "published",
                col(SurfaceProject.published_build_id).is_not(None),
                col(SurfaceProject.published_revision_id).is_not(None),
                Event.lifecycle == "active",
            )
            .order_by(col(SurfaceProject.updated_at))
            .limit(max(1, min(limit, 100)))
        )
        if session.bind and session.bind.dialect.name == "postgresql":
            statement = statement.with_for_update(skip_locked=True)
        rows = list((await session.execute(statement)).all())
        for project, event in rows:
            latest_run = (
                (
                    await session.execute(
                        select(Run)
                        .where(
                            Run.organization_id == event.organization_id,
                            Run.user_id == event.user_id,
                            Run.event_id == event.id,
                            Run.run_kind == SURFACE_REINSPECTION_RUN_KIND,
                        )
                        .order_by(col(Run.created_at).desc())
                        .limit(1)
                    )
                )
                .scalars()
                .first()
            )
            if latest_run is not None:
                latest_build_id = str(
                    dict(latest_run.run_profile or {}).get("build_id") or ""
                )
                if latest_build_id == project.published_build_id and (
                    latest_run.status in {"pending", "running"}
                    or _aware(latest_run.created_at) >= cutoff
                ):
                    continue
            latest_receipt = (
                (
                    await session.execute(
                        select(SurfaceInspection)
                        .where(
                            SurfaceInspection.organization_id == event.organization_id,
                            SurfaceInspection.user_id == event.user_id,
                            SurfaceInspection.event_id == event.id,
                            SurfaceInspection.build_id == project.published_build_id,
                            SurfaceInspection.inspection_kind == "reinspection",
                        )
                        .order_by(col(SurfaceInspection.created_at).desc())
                        .limit(1)
                    )
                )
                .scalars()
                .first()
            )
            if (
                latest_receipt is not None
                and _aware(latest_receipt.created_at) >= cutoff
            ):
                continue
            organization = await session.get(Organization, event.organization_id)
            policy = OrganizationPolicy.model_validate(
                organization.policy if organization is not None else {}
            )
            _, replayed = await queue_surface_reinspection(
                session,
                context=OrganizationContext(
                    organization_id=event.organization_id,
                    user_id=event.user_id,
                    role="member",
                    permissions=(),
                    policy=policy,
                ),
                event=event,
                reason="automatic_health_check",
                actor_id="worker:surface-inspector-planner",
            )
            if not replayed:
                enqueued += 1
    return enqueued


def surface_reinspection_run_payload(run: Run, *, replayed: bool = False) -> dict:
    return {
        "run_id": run.id,
        "event_id": run.event_id,
        "status": run.status,
        "build_id": dict(run.run_profile or {}).get("build_id"),
        "reason": dict(run.run_profile or {}).get("reason"),
        "progress": run.progress or {},
        "created_at": run.created_at,
        "completed_at": run.completed_at,
        "replayed": replayed,
    }


async def execute_claimed_surface_reinspection(
    run_id: str,
    worker_id: str,
    *,
    session_factory: Any = async_session,
    handler_factory: Any = SurfaceBuildHandler,
) -> bool:
    """Run fresh host inspection and propose evolution only for trusted failures."""

    async with session_factory() as session:
        run = await session.get(Run, run_id)
        if (
            run is None
            or run.run_kind != SURFACE_REINSPECTION_RUN_KIND
            or run.status != "running"
            or run.lease_owner != worker_id
        ):
            return False
        profile = dict(run.run_profile or {})
        build_id = str(profile.get("build_id") or "")
        if not build_id:
            raise SurfaceReinspectionError(409, "Reinspection build binding is missing")
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
            max_duration_seconds=float(run.timeout_seconds),
            isolation_profile=run.isolation_profile,
        )

    try:
        preview = await handler_factory(
            run_context=run_context,
            session_factory=session_factory,
        ).preview(
            SurfacePreviewParams(
                build_id=build_id,
                force_reinspection=True,
                inspection_kind="reinspection",
            )
        )
        inspection_id = str(preview.get("inspection_id") or "")
        if not inspection_id:
            raise RuntimeError("Trusted Surface reinspection produced no receipt")
        quality = dict(preview.get("quality_gate") or {})
        passed = quality.get("passed") is True

        async with session_factory() as session:
            run = await session.get(Run, run_id)
            if run is None or run.lease_owner != worker_id:
                return False
            event = await session.get(Event, run.event_id)
            project = (
                (
                    await session.execute(
                        select(SurfaceProject).where(
                            SurfaceProject.event_id == run.event_id,
                            SurfaceProject.organization_id == run.organization_id,
                            SurfaceProject.user_id == run.user_id,
                        )
                    )
                )
                .scalars()
                .first()
            )
            still_live = bool(project and project.published_build_id == build_id)
            now = datetime.now(timezone.utc)
            run.status = "completed"
            run.success = True
            run.steps_taken = 1
            run.final_answer = (
                "The live Surface passed trusted reinspection."
                if passed
                else "Trusted reinspection found a Surface quality regression."
            )
            run.reasoning = "Model-free host inspection completed."
            run.progress = {
                "stage": "passed" if passed else "quality_regression",
                "inspection_id": inspection_id,
                "build_id": build_id,
                "still_live": still_live,
                "updated_at": now.isoformat(),
            }
            run.execution_receipts = [
                *(run.execution_receipts or []),
                {
                    "kind": "surface_reinspection_result",
                    "inspection_id": inspection_id,
                    "build_id": build_id,
                    "quality_passed": passed,
                    "still_live": still_live,
                    "quality_fingerprint": quality.get("fingerprint"),
                },
            ]
            run.completed_at = now
            run.lease_owner = None
            run.lease_expires_at = None
            session.add(run)
            session.add(
                EventTrailEntry(
                    organization_id=run.organization_id,
                    user_id=run.user_id,
                    event_id=run.event_id,
                    actor_id="worker:surface-inspector",
                    kind="surface_reinspection_finished",
                    summary=(
                        "Live Surface passed trusted reinspection"
                        if passed
                        else "Trusted reinspection found a Surface quality regression"
                    ),
                    run_id=run.id,
                    evidence_refs=[
                        {"surface_inspection_id": inspection_id},
                        {"surface_build_id": build_id},
                    ],
                    payload={"quality_passed": passed, "still_live": still_live},
                )
            )
            if not passed and still_live and event is not None:
                organization = await session.get(Organization, run.organization_id)
                policy = OrganizationPolicy.model_validate(
                    organization.policy if organization is not None else {}
                )
                await record_surface_evolution_signal(
                    session,
                    context=OrganizationContext(
                        organization_id=run.organization_id,
                        user_id=run.user_id,
                        role="member",
                        permissions=(),
                        policy=policy,
                    ),
                    event=event,
                    signal=SurfaceEvolutionSignal(
                        trigger="quality_failure",
                        goal=(
                            "Repair live Surface quality regressions detected by "
                            "trusted reinspection"
                        ),
                        evidence_refs=[inspection_id],
                    ),
                )
            else:
                await session.commit()
            return True
    except Exception as exc:
        logger.exception("Surface reinspection Run %s failed", run_id)
        async with session_factory() as session:
            run = await session.get(Run, run_id)
            if run is None or run.lease_owner != worker_id:
                return False
            terminal = run.attempt_count >= run.max_attempts
            now = datetime.now(timezone.utc)
            run.status = "failed" if terminal else "pending"
            run.success = False
            run.final_answer = "Trusted Surface reinspection was unavailable."
            run.reasoning = str(exc)[:4000]
            run.progress = {
                "stage": "failed" if terminal else "retry_scheduled",
                "error": str(exc)[:1000],
                "updated_at": now.isoformat(),
            }
            run.completed_at = now if terminal else None
            run.lease_owner = None
            run.lease_expires_at = None
            session.add(run)
            session.add(
                EventTrailEntry(
                    organization_id=run.organization_id,
                    user_id=run.user_id,
                    event_id=run.event_id,
                    actor_id="worker:surface-inspector",
                    kind="surface_reinspection_unavailable",
                    summary="Trusted Surface reinspection was unavailable",
                    run_id=run.id,
                    payload={"terminal": terminal},
                )
            )
            await session.commit()
            return False


__all__ = [
    "SURFACE_REINSPECTION_AGENT_ID",
    "SURFACE_REINSPECTION_RUN_KIND",
    "SurfaceReinspectionError",
    "execute_claimed_surface_reinspection",
    "queue_due_surface_reinspections",
    "queue_surface_reinspection",
    "surface_health_snapshot",
    "surface_reinspection_run_payload",
]
