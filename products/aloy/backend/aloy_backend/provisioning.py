"""Provisioning IN: materialize durable uploads into a conversation's sandbox.

The object store is the source of truth; the sandbox uploads dir
(``/mnt/user-data/uploads``) is where the agent's tools can actually reach
the bytes. Provisioning is idempotent and hash-skipped: a ``.provisioned.json``
manifest in the uploads dir records each materialized file's sha256, so
re-runs, continues, and eager-at-upload-time calls never copy twice (the
FileSyncManager discipline from the Hermes mining, simplified).

Latency contract (spec: "Latency & context budget"): this runs at upload time
(eager, off the send path) and again at run setup as a cheap verify — never
inside the LLM-call loop.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional

from pori import get_thread_data

from .config import settings
from .models import StoredFile
from .storage import get_object_store, safe_name

logger = logging.getLogger("aloy_backend")

_MANIFEST = ".provisioned.json"


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024
    return f"{size:.1f}GB"


def provision_conversation_uploads(
    conversation_id: str, records: Iterable[StoredFile]
) -> List[dict]:
    """Ensure every upload is present in the conversation's sandbox uploads
    dir. Returns manifest entries [{name, size_bytes, content_type}] for the
    task block — an entry appears even when the copy was skipped (already
    present, same hash). A missing blob is logged and omitted, never fatal.
    """
    thread = get_thread_data(conversation_id, settings.sandbox_base_dir)
    uploads_dir = Path(thread.uploads_path)
    manifest_path = uploads_dir / _MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}

    store = get_object_store()
    entries: List[dict] = []
    used_names = set(manifest.keys())
    changed = False

    for rec in sorted(records, key=lambda r: r.created_at.isoformat()):
        name = _claim_name(rec, manifest, used_names)
        target = uploads_dir / name
        if manifest.get(name) != rec.sha256 or not target.is_file():
            try:
                _copy_from_store(store, rec.storage_key, target)
            except FileNotFoundError:
                logger.warning(
                    "Upload %s (%s) missing from the object store; skipping",
                    rec.id,
                    rec.name,
                )
                continue
            manifest[name] = rec.sha256
            changed = True
        entries.append(
            {
                "name": name,
                "size_bytes": rec.size_bytes,
                "content_type": rec.content_type,
            }
        )

    if changed:
        tmp = manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        os.replace(tmp, manifest_path)
    return entries


def _claim_name(rec: StoredFile, manifest: dict, used_names: set) -> str:
    """A stable on-disk name per record: the safe filename, or an id-prefixed
    variant when a DIFFERENT file already claimed it (two 'data.csv' uploads
    must not overwrite each other)."""
    name = safe_name(rec.name)
    existing = manifest.get(name)
    fresh = existing is None and name not in used_names
    if fresh or existing == rec.sha256:
        used_names.add(name)
        return name
    prefixed = f"{rec.id[:8]}_{name}"
    used_names.add(prefixed)
    return prefixed


def _copy_from_store(store, key: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, store.open(key) as src:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def uploads_task_block(entries: List[dict]) -> str:
    """The reference block the model sees INSTEAD of file bytes: ~1 line per
    file regardless of size, plus sampled-access guidance (spec: references
    in, samples out)."""
    if not entries:
        return ""
    lines = "\n".join(
        f"- /mnt/user-data/uploads/{e['name']} "
        f"({_human_size(e['size_bytes'])}, {e['content_type']})"
        for e in entries
    )
    return (
        "\n\n<uploaded-files>\n"
        "The user has uploaded these files. They are on disk, NOT in this "
        "conversation:\n"
        f"{lines}\n"
        "Work with them through your file/bash tools. Inspect a sample first "
        "(head, wc -l, a dataframe .head()) rather than reading a whole file "
        "into your context.\n"
        "</uploaded-files>"
    )


def resolve_upload_refs(
    records: Iterable[Optional[StoredFile]],
    *,
    organization_id: str,
    conversation_id: str,
) -> List[StoredFile]:
    """Filter looked-up rows to real uploads of THIS org + conversation —
    a ref to someone else's file id is silently dropped, never an oracle."""
    return [
        r
        for r in records
        if r is not None
        and r.kind == "upload"
        and r.organization_id == organization_id
        and r.conversation_id == conversation_id
    ]
