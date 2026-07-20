"""Fail-closed completion gate and durable artifact indexing for research Runs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from pori import VIRTUAL_PREFIX, get_workspace_data, replace_virtual_path

from .config import settings
from .database import async_session
from .models import EventTrailEntry, KnowledgeEntry, Run


@dataclass(frozen=True)
class ResearchGateResult:
    accepted: bool
    errors: tuple[str, ...]
    report_file_ids: tuple[str, ...]
    evidence_count: int
    record_count: int

    def receipt(self) -> dict[str, Any]:
        return {
            "kind": "research_quality_gate",
            "accepted": self.accepted,
            "errors": list(self.errors),
            "report_file_ids": list(self.report_file_ids),
            "evidence_count": self.evidence_count,
            "record_count": self.record_count,
        }


def _artifact_path(run: Run, raw: str) -> Path | None:
    if not raw:
        return None
    thread = get_workspace_data(run.event_id, run.id, settings.sandbox_base_dir)
    try:
        if raw.startswith(VIRTUAL_PREFIX):
            candidate = Path(replace_virtual_path(raw, thread))
        else:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = Path(thread.workspace_path) / candidate
        resolved = candidate.resolve()
        resolved.relative_to(Path(thread.workspace_path).parent.resolve())
        return resolved
    except ValueError:
        return None


async def gate_and_index_research_run(
    session: AsyncSession,
    *,
    run: Run,
    artifacts: list[dict[str, Any]],
    evidence_ids: tuple[str, ...],
    record_ids: tuple[str, ...],
    session_factory: Any = async_session,
) -> ResearchGateResult:
    """Validate research deliverables and stage report indexes in one transaction."""
    errors: list[str] = []
    unique_evidence = tuple(dict.fromkeys(evidence_ids))
    unique_records = tuple(dict.fromkeys(record_ids))

    evidence_rows: list[KnowledgeEntry] = []
    record_rows: list[KnowledgeEntry] = []
    async with session_factory() as read_session:
        candidates = list(
            (
                await read_session.execute(
                    select(KnowledgeEntry).where(
                        KnowledgeEntry.organization_id == run.organization_id,
                        KnowledgeEntry.user_id == run.user_id,
                        KnowledgeEntry.event_id == run.event_id,
                        KnowledgeEntry.status == "active",
                    )
                )
            )
            .scalars()
            .all()
        )
        # Recorder ids are the fast path. Run provenance is the crash-recovery
        # path: checkpoint resume must see evidence and records committed by a
        # prior worker process even though its in-memory collectors are gone.
        evidence_id_set = set(unique_evidence)
        record_id_set = set(unique_records)
        for row in candidates:
            metadata = row.metadata_ or {}
            provenance = row.provenance or {}
            belongs_to_run = (
                provenance.get("run_id") == run.id or metadata.get("run_id") == run.id
            )
            if "web_evidence" in (row.tags or []) and (
                row.id in evidence_id_set or belongs_to_run
            ):
                evidence_rows.append(row)
            if metadata.get("record_type") == "event_record" and (
                row.id in record_id_set or belongs_to_run
            ):
                record_rows.append(row)
    if not evidence_rows:
        errors.append("No committed web evidence was produced")

    committed_evidence_ids = {row.id for row in evidence_rows}
    grounded_record_rows = [
        row
        for row in record_rows
        if (row.metadata_ or {}).get("posture") in {"observed", "inferred"}
        and any(
            reference.get("evidence_id") in committed_evidence_ids
            for reference in (row.metadata_ or {}).get("evidence_refs") or []
            if isinstance(reference, dict)
        )
    ]
    if not grounded_record_rows:
        errors.append("No canonical evidence-backed Event records were produced")

    source_urls = {
        str((row.metadata_ or {}).get("url"))
        for row in evidence_rows
        if (row.metadata_ or {}).get("url")
    }
    report_entries: list[tuple[dict[str, Any], Path, str]] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or artifact.get("kind") != "file":
            continue
        file_id = str(artifact.get("file_id") or "")
        path = _artifact_path(run, str(artifact.get("path") or ""))
        if (
            not file_id
            or path is None
            or path.suffix.lower() not in {".md", ".markdown"}
        ):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        report_entries.append((artifact, path, text))
    if not report_entries:
        errors.append("No stored Markdown research report artifact was produced")
    elif source_urls and not any(
        any(url in text for url in source_urls) for _, _, text in report_entries
    ):
        errors.append("The research report does not cite any committed source URL")

    accepted = not errors
    report_file_ids: list[str] = []
    if accepted:
        evidence_refs = [
            {
                "evidence_id": row.id,
                "url": (row.metadata_ or {}).get("url"),
                "title": (row.metadata_ or {}).get("title"),
                "retrieved_at": (row.metadata_ or {}).get("retrieved_at"),
            }
            for row in evidence_rows
        ]
        for artifact, path, _ in report_entries:
            file_id = str(artifact["file_id"])
            report_file_ids.append(file_id)
            conflict_key = f"research_report:{run.task_id or run.id}:{path.name}"
            index_id = (
                "rpt_"
                + hashlib.sha256(
                    f"{run.organization_id}\x1f{run.event_id}\x1f{run.id}\x1f{file_id}".encode(
                        "utf-8"
                    )
                ).hexdigest()[:32]
            )
            if await session.get(KnowledgeEntry, index_id) is None:
                previous = list(
                    (
                        await session.execute(
                            select(KnowledgeEntry).where(
                                KnowledgeEntry.organization_id == run.organization_id,
                                KnowledgeEntry.user_id == run.user_id,
                                KnowledgeEntry.event_id == run.event_id,
                                KnowledgeEntry.conflict_key == conflict_key,
                                KnowledgeEntry.status == "active",
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                for old_report in previous:
                    old_report.status = "superseded"
                    old_report.superseded_by = index_id
                    session.add(old_report)
                session.add(
                    KnowledgeEntry(
                        id=index_id,
                        organization_id=run.organization_id,
                        user_id=run.user_id,
                        event_id=run.event_id,
                        agent_id=run.agent_id,
                        session_id=run.session_id,
                        content=f"Sourced research report: {path.name}",
                        tags=["research_report", "event_artifact"],
                        importance=4,
                        kind="semantic",
                        confidence=1.0,
                        sensitivity="internal",
                        source="agent",
                        provenance={
                            "source": "agent",
                            "source_id": file_id,
                            "actor_id": run.agent_id,
                            "conversation_id": run.conversation_id,
                            "run_id": run.id,
                            "metadata": {"task_id": run.task_id},
                        },
                        conflict_key=conflict_key,
                        metadata_={
                            "record_type": "research_report",
                            "file_id": file_id,
                            "file_name": path.name,
                            "task_id": run.task_id,
                            "run_id": run.id,
                            "evidence_refs": evidence_refs,
                            "record_ids": [row.id for row in record_rows],
                        },
                    )
                )
                session.add(
                    EventTrailEntry(
                        organization_id=run.organization_id,
                        user_id=run.user_id,
                        event_id=run.event_id,
                        actor_id=run.agent_id,
                        kind="research_report_indexed",
                        summary=f"Indexed sourced report {path.name}",
                        run_id=run.id,
                        task_id=run.task_id,
                        evidence_refs=[{"file_id": file_id}, *evidence_refs],
                        payload={
                            "file_id": file_id,
                            "record_ids": [row.id for row in record_rows],
                        },
                    )
                )

    return ResearchGateResult(
        accepted=accepted,
        errors=tuple(errors),
        report_file_ids=tuple(report_file_ids),
        evidence_count=len(evidence_rows),
        record_count=len(record_rows),
    )


__all__ = ["ResearchGateResult", "gate_and_index_research_run"]
