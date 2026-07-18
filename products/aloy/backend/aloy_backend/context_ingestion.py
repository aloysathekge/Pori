"""Durable, model-independent ingestion for Event setup context sources."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import logging
import socket
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from io import BytesIO
from typing import Any, Awaitable, Callable
from urllib.parse import urljoin, urlsplit

import httpx
from pypdf import PdfReader
from sqlalchemy import or_
from sqlmodel import col, select

from .config import settings
from .database import async_session
from .doc_extract import ExtractionError, extract_docx_text, extract_xlsx_text
from .event_bootstrap import queue_event_bootstrap_if_ready
from .models import (
    Event,
    EventSetupContextItem,
    EventTrailEntry,
    KnowledgeEntry,
)
from .storage import get_object_store

logger = logging.getLogger("aloy_backend.context_ingestion")

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_TEXT_TYPES = {
    "application/json",
    "application/xml",
    "application/x-yaml",
    "text/csv",
    "text/html",
    "text/markdown",
    "text/plain",
    "text/xml",
    "text/yaml",
}


class ContextIngestionError(Exception):
    def __init__(self, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class IngestedSource:
    text: str
    content_type: str
    retrieved_at: datetime
    sha256: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden_depth = 0
        self.title_depth = 0
        self.parts: list[str] = []
        self.title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self.hidden_depth += 1
        if tag == "title":
            self.title_depth += 1
        if tag in {"br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self.hidden_depth:
            self.hidden_depth -= 1
        if tag == "title" and self.title_depth:
            self.title_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.title_depth:
            self.title_parts.append(data)
        if not self.hidden_depth:
            self.parts.append(data)

    def result(self) -> tuple[str, str]:
        title = " ".join(" ".join(self.title_parts).split())
        lines = [" ".join(line.split()) for line in "".join(self.parts).splitlines()]
        text = "\n".join(line for line in lines if line)
        return title, text


def _bounded_text(text: str) -> str:
    text = text.replace("\x00", "").strip()
    if not text:
        raise ContextIngestionError(
            "The source contains no extractable text", retryable=False
        )
    return text[: settings.context_ingestion_max_text_chars]


def _decode_text(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def extract_file_source(item: EventSetupContextItem, raw: bytes) -> IngestedSource:
    content_type = (item.content_type or "application/octet-stream").split(";", 1)[0]
    suffix = item.label.lower().rsplit(".", 1)[-1] if "." in item.label else ""
    try:
        if content_type == DOCX_MIME or suffix == "docx":
            text = extract_docx_text(raw)
        elif content_type == XLSX_MIME or suffix == "xlsx":
            text = extract_xlsx_text(raw)
        elif content_type == "application/pdf" or suffix == "pdf":
            reader = PdfReader(BytesIO(raw))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        elif content_type == "text/html" or suffix in {"html", "htm"}:
            parser = _VisibleTextParser()
            parser.feed(_decode_text(raw))
            title, text = parser.result()
            return IngestedSource(
                text=_bounded_text(text),
                title=title,
                content_type=content_type,
                retrieved_at=datetime.now(timezone.utc),
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        elif (
            content_type.startswith("text/")
            or content_type in _TEXT_TYPES
            or suffix
            in {
                "csv",
                "json",
                "md",
                "txt",
                "xml",
                "yaml",
                "yml",
            }
        ):
            text = _decode_text(raw)
        else:
            raise ContextIngestionError(
                f"Text extraction is not available for {content_type}",
                retryable=False,
            )
    except ExtractionError as exc:
        raise ContextIngestionError(str(exc), retryable=False) from exc
    except Exception as exc:
        if isinstance(exc, ContextIngestionError):
            raise
        raise ContextIngestionError(
            f"Could not extract {item.label}: {exc}", retryable=False
        ) from exc
    return IngestedSource(
        text=_bounded_text(text),
        content_type=content_type,
        retrieved_at=datetime.now(timezone.utc),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


async def _assert_public_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ContextIngestionError(
            "Only public HTTP and HTTPS links can be ingested", retryable=False
        )
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise ContextIngestionError(f"Could not resolve the link host: {exc}") from exc
    addresses = {ipaddress.ip_address(info[4][0]) for info in infos}
    if not addresses or any(not address.is_global for address in addresses):
        raise ContextIngestionError(
            "Private or local network links are not allowed", retryable=False
        )


def _assert_public_peer(response: httpx.Response) -> None:
    stream = response.extensions.get("network_stream")
    peer = stream.get_extra_info("server_addr") if stream is not None else None
    if peer:
        address = ipaddress.ip_address(peer[0])
        if not address.is_global:
            raise ContextIngestionError(
                "The link connected to a private network address", retryable=False
            )


async def fetch_public_link(url: str) -> IngestedSource:
    current = url
    timeout = httpx.Timeout(settings.context_ingestion_link_timeout_seconds)
    headers = {
        "User-Agent": "Aloy-Context-Ingestion/1.0",
        "Accept": "text/html,text/plain,application/json,application/xml;q=0.8",
    }
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        for redirect_count in range(4):
            await _assert_public_url(current)
            try:
                async with client.stream("GET", current, headers=headers) as response:
                    _assert_public_peer(response)
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise ContextIngestionError(
                                "The link redirected without a destination",
                                retryable=False,
                            )
                        current = urljoin(current, location)
                        continue
                    response.raise_for_status()
                    raw = bytearray()
                    async for chunk in response.aiter_bytes():
                        raw.extend(chunk)
                        if len(raw) > settings.context_ingestion_max_link_bytes:
                            raise ContextIngestionError(
                                "The linked page exceeds the ingestion size limit",
                                retryable=False,
                            )
                    content_type = (
                        response.headers.get("content-type", "text/html")
                        .split(";", 1)[0]
                        .lower()
                    )
                    if not (
                        content_type.startswith("text/") or content_type in _TEXT_TYPES
                    ):
                        raise ContextIngestionError(
                            f"Linked content type {content_type} is not supported",
                            retryable=False,
                        )
                    decoded = _decode_text(bytes(raw))
                    title = ""
                    if content_type == "text/html":
                        parser = _VisibleTextParser()
                        parser.feed(decoded)
                        title, decoded = parser.result()
                    return IngestedSource(
                        text=_bounded_text(decoded),
                        title=title,
                        content_type=content_type,
                        retrieved_at=datetime.now(timezone.utc),
                        sha256=hashlib.sha256(raw).hexdigest(),
                        metadata={
                            "final_url": str(response.url),
                            "etag": response.headers.get("etag"),
                            "last_modified": response.headers.get("last-modified"),
                        },
                    )
            except httpx.HTTPStatusError as exc:
                retryable = (
                    exc.response.status_code >= 500
                    or exc.response.status_code in {408, 429}
                )
                raise ContextIngestionError(
                    f"The link returned HTTP {exc.response.status_code}",
                    retryable=retryable,
                ) from exc
            except httpx.HTTPError as exc:
                raise ContextIngestionError(
                    f"Could not retrieve the link: {exc}"
                ) from exc
        raise ContextIngestionError(
            "The link redirected too many times", retryable=False
        )


LinkFetcher = Callable[[str], Awaitable[IngestedSource]]


async def claim_next_context_item(worker_id: str) -> str | None:
    now = datetime.now(timezone.utc)
    async with async_session() as session:
        statement = (
            select(EventSetupContextItem.id)
            .where(
                col(EventSetupContextItem.event_id).is_not(None),
                col(EventSetupContextItem.kind).in_(["file", "link"]),
                EventSetupContextItem.attempt_count
                < EventSetupContextItem.max_attempts,
                or_(
                    col(EventSetupContextItem.status) == "pending",
                    (col(EventSetupContextItem.status) == "failed")
                    & (
                        col(EventSetupContextItem.next_attempt_at).is_(None)
                        | (col(EventSetupContextItem.next_attempt_at) <= now)
                    ),
                    (col(EventSetupContextItem.status) == "ingesting")
                    & (
                        col(EventSetupContextItem.lease_expires_at).is_(None)
                        | (col(EventSetupContextItem.lease_expires_at) < now)
                    ),
                ),
            )
            .order_by(
                col(EventSetupContextItem.updated_at),
                col(EventSetupContextItem.created_at),
            )
            .limit(50)
        )
        candidate_ids = list((await session.execute(statement)).scalars().all())
        dialect = session.bind.dialect.name if session.bind else ""
        for item_id in candidate_ids:
            lock = select(EventSetupContextItem).where(
                EventSetupContextItem.id == item_id
            )
            if dialect == "postgresql":
                lock = lock.with_for_update(skip_locked=True)
            item = (await session.execute(lock)).scalars().first()
            if (
                item is None
                or item.event_id is None
                or item.kind not in {"file", "link"}
            ):
                continue
            lease = item.lease_expires_at
            if lease is not None and lease.tzinfo is None:
                lease = lease.replace(tzinfo=timezone.utc)
            next_attempt = item.next_attempt_at
            if next_attempt is not None and next_attempt.tzinfo is None:
                next_attempt = next_attempt.replace(tzinfo=timezone.utc)
            eligible = (
                item.status == "pending"
                or (
                    item.status == "failed"
                    and (next_attempt is None or next_attempt <= now)
                )
                or (item.status == "ingesting" and (lease is None or lease < now))
            )
            if not eligible or item.attempt_count >= item.max_attempts:
                continue
            item.status = "ingesting"
            item.error = None
            item.attempt_count += 1
            item.lease_owner = worker_id
            item.lease_expires_at = now + timedelta(
                seconds=settings.context_ingestion_lease_seconds
            )
            item.next_attempt_at = None
            item.updated_at = now
            session.add(item)
            await session.commit()
            return item.id
        await session.rollback()
        return None


async def _read_source(
    item: EventSetupContextItem, *, link_fetcher: LinkFetcher
) -> IngestedSource:
    if item.kind == "link" and item.source_url:
        return await link_fetcher(item.source_url)
    if item.kind == "file" and item.storage_key:
        try:
            with get_object_store().open(item.storage_key) as stream:
                raw = stream.read(settings.storage_max_file_mb * 1024 * 1024 + 1)
        except FileNotFoundError as exc:
            raise ContextIngestionError(
                "The staged file is no longer available", retryable=False
            ) from exc
        if len(raw) > settings.storage_max_file_mb * 1024 * 1024:
            raise ContextIngestionError(
                "The staged file exceeds the ingestion limit", retryable=False
            )
        return extract_file_source(item, raw)
    raise ContextIngestionError("The context source is incomplete", retryable=False)


async def _record_failure(
    item_id: str, worker_id: str, exc: ContextIngestionError
) -> None:
    now = datetime.now(timezone.utc)
    async with async_session() as session:
        item = await session.get(EventSetupContextItem, item_id)
        if item is None or item.lease_owner != worker_id or item.status != "ingesting":
            return
        retryable = exc.retryable and item.attempt_count < item.max_attempts
        if not exc.retryable:
            item.attempt_count = item.max_attempts
        item.status = "failed"
        item.error = str(exc)[:2000]
        item.next_attempt_at = (
            now + timedelta(seconds=min(300, 5 * (2 ** max(0, item.attempt_count - 1))))
            if retryable
            else None
        )
        item.lease_owner = None
        item.lease_expires_at = None
        item.updated_at = now
        metadata = dict(item.metadata_ or {})
        metadata["ingestion"] = {
            "status": "failed",
            "retryable": retryable,
            "attempt": item.attempt_count,
        }
        item.metadata_ = metadata
        session.add(item)
        if item.event_id:
            session.add(
                EventTrailEntry(
                    organization_id=item.organization_id,
                    user_id=item.user_id,
                    event_id=item.event_id,
                    actor_id="aloy:context-ingestion",
                    kind="context_ingestion_failed",
                    summary=f"Could not ingest {item.label}",
                    evidence_refs=[{"context_item_id": item.id}],
                    payload={
                        "kind": item.kind,
                        "attempt": item.attempt_count,
                        "retryable": retryable,
                        "next_attempt_at": (
                            item.next_attempt_at.isoformat()
                            if item.next_attempt_at
                            else None
                        ),
                        "error": item.error,
                    },
                )
            )
        await session.commit()


async def execute_claimed_context_item(
    item_id: str,
    worker_id: str,
    *,
    link_fetcher: LinkFetcher = fetch_public_link,
) -> bool:
    async with async_session() as session:
        item = await session.get(EventSetupContextItem, item_id)
        if item is None or item.status != "ingesting" or item.lease_owner != worker_id:
            return False
        source_item = item.model_copy(deep=True)
    try:
        source = await _read_source(source_item, link_fetcher=link_fetcher)
    except ContextIngestionError as exc:
        await _record_failure(item_id, worker_id, exc)
        return True
    except Exception as exc:
        logger.exception("Unexpected Event context ingestion failure for %s", item_id)
        await _record_failure(
            item_id,
            worker_id,
            ContextIngestionError(f"Unexpected ingestion failure: {exc}"),
        )
        return True

    now = datetime.now(timezone.utc)
    async with async_session() as session:
        item = await session.get(EventSetupContextItem, item_id)
        if (
            item is None
            or item.status != "ingesting"
            or item.lease_owner != worker_id
            or not item.event_id
        ):
            return False
        event = await session.get(Event, item.event_id)
        if (
            event is None
            or event.organization_id != item.organization_id
            or event.user_id != item.user_id
        ):
            await session.rollback()
            await _record_failure(
                item_id,
                worker_id,
                ContextIngestionError(
                    "The owning Event is unavailable", retryable=False
                ),
            )
            return True
        provenance = {
            "source": "event_context_ingestion",
            "context_item_id": item.id,
            "draft_id": item.draft_id,
            "kind": item.kind,
            "source_url": item.source_url,
            "input_sha256": item.sha256 or None,
            "content_sha256": source.sha256,
            "retrieved_at": source.retrieved_at.isoformat(),
            "freshness": source.metadata,
        }
        heading = source.title or item.label
        prefix = f"{heading}\n"
        if item.source_url:
            prefix += f"Source: {item.source_url}\n"
        content = f"{prefix}\n{source.text}".strip()
        entry = (
            await session.get(KnowledgeEntry, item.knowledge_entry_id)
            if item.knowledge_entry_id
            else None
        )
        if entry is None:
            entry = KnowledgeEntry(
                organization_id=item.organization_id,
                user_id=item.user_id,
                event_id=event.id,
                session_id=event.primary_conversation_id,
                content=content,
            )
        entry.content = content
        entry.tags = ["event-context", item.kind, "ingested"]
        entry.importance = 3
        entry.kind = "semantic"
        entry.confidence = 1.0
        entry.sensitivity = item.sensitivity
        entry.source = "user"
        entry.provenance = provenance
        entry.retention = {"mode": "event_lifecycle"}
        entry.scope_level = "personal"
        entry.status = "active"
        entry.metadata_ = {
            "event_scoped": True,
            "content_type": source.content_type,
            "context_item_id": item.id,
            "ingestion_status": "ready",
        }
        entry.updated_at = now
        entry.event_at = source.retrieved_at
        session.add(entry)
        await session.flush()
        item.knowledge_entry_id = entry.id
        item.status = "ready"
        item.error = None
        item.retrieved_at = source.retrieved_at
        item.ingested_at = now
        item.lease_owner = None
        item.lease_expires_at = None
        item.next_attempt_at = None
        item.updated_at = now
        metadata = dict(item.metadata_ or {})
        metadata["ingestion"] = {
            "status": "ready",
            "attempt": item.attempt_count,
            "content_chars": len(source.text),
            "content_sha256": source.sha256,
            "title": source.title,
            **source.metadata,
        }
        item.metadata_ = metadata
        session.add(item)
        session.add(
            EventTrailEntry(
                organization_id=item.organization_id,
                user_id=item.user_id,
                event_id=event.id,
                actor_id="aloy:context-ingestion",
                kind="context_ingestion_completed",
                summary=f"Ingested {item.label}",
                evidence_refs=[
                    {"context_item_id": item.id},
                    {"knowledge_entry_id": entry.id},
                ],
                payload={
                    "kind": item.kind,
                    "content_type": source.content_type,
                    "content_chars": len(source.text),
                    "retrieved_at": source.retrieved_at.isoformat(),
                },
            )
        )
        await queue_event_bootstrap_if_ready(
            session,
            organization_id=item.organization_id,
            user_id=item.user_id,
            event_id=event.id,
        )
        await session.commit()
    return True


async def run_next_context_ingestion(worker_id: str) -> bool:
    item_id = await claim_next_context_item(worker_id)
    if item_id is None:
        return False
    await execute_claimed_context_item(item_id, worker_id)
    return True
