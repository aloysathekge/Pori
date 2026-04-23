"""@path file references in CLI prompts.

Lets users attach files by typing ``@main.py`` or ``@~/docs/cv.docx`` directly
in the task prompt. Each matched token is read once at input time and inlined
as a ``<attached path="...">`` block appended to the user's message, so the
LLM sees the content in the same turn — no extra tool roundtrip needed.

Safety reuses the same ``fs_config.is_path_safe`` policy as the ``read_file``
tool (path must sit under the user's home or the current working directory).
Large files are truncated; ``.docx`` / ``.pdf`` need optional deps
(``python-docx`` / ``pypdf``) and degrade gracefully if missing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from pori.tools.standard.filesystem_tools import format_bytes, fs_config

# Match @"quoted path" or @unquoted_path. The negative lookbehind on [\w/]
# prevents matching emails (``foo@bar``) or URLs with trailing @-fragments.
_TOKEN = re.compile(r'(?<![\w/])@(?:"([^"]+)"|([^\s]+))')

MAX_INLINE_BYTES = 200_000  # cap per-file to avoid blowing the context window


def _parse_refs(text: str) -> List[str]:
    """Return ordered, deduplicated list of raw path strings from @tokens."""
    seen: List[str] = []
    for match in _TOKEN.finditer(text):
        raw = match.group(1) or match.group(2)
        if raw and raw not in seen:
            seen.append(raw)
    return seen


def _resolve(raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.resolve()


def _extract(path: Path) -> Tuple[str, str]:
    """Return (content, status_note). Empty content means skipped."""
    ext = path.suffix.lower()
    size = path.stat().st_size

    if ext == ".docx":
        try:
            from docx import Document  # type: ignore
        except ImportError:
            return "", "skipped — .docx needs `pip install python-docx`"
        try:
            doc = Document(str(path))
            text = "\n".join(p.text for p in doc.paragraphs if p.text)
            return text, f"{format_bytes(size)}, .docx extracted"
        except Exception as e:
            return "", f"skipped — .docx extraction failed: {e}"

    if ext == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError:
            return "", "skipped — .pdf needs `pip install pypdf`"
        try:
            reader = PdfReader(str(path))
            text = "\n".join((p.extract_text() or "") for p in reader.pages)
            return text, f"{format_bytes(size)}, .pdf extracted"
        except Exception as e:
            return "", f"skipped — .pdf extraction failed: {e}"

    # Plain text / code / markdown / config: read with size cap.
    try:
        if size > MAX_INLINE_BYTES:
            with open(path, "rb") as f:
                raw = f.read(MAX_INLINE_BYTES)
            text = raw.decode("utf-8", errors="replace")
            return (
                text,
                f"truncated to {format_bytes(MAX_INLINE_BYTES)} of "
                f"{format_bytes(size)}",
            )
        return path.read_text(encoding="utf-8"), format_bytes(size)
    except UnicodeDecodeError:
        try:
            return (
                path.read_text(encoding="utf-8", errors="replace"),
                f"{format_bytes(size)} (replaced invalid bytes)",
            )
        except Exception as e:
            return "", f"skipped — read failed: {e}"


def expand_file_refs(task: str) -> Tuple[str, List[str]]:
    """Parse @path tokens in ``task``, inline file contents, return enriched task.

    Returns ``(enriched_task, notes)`` where ``notes`` are human-facing status
    lines to surface in the CLI so the user knows what was attached (or why
    something was skipped). If no references are found, the task is returned
    unchanged with empty notes.
    """
    refs = _parse_refs(task)
    if not refs:
        return task, []

    blocks: List[str] = []
    notes: List[str] = []

    for raw in refs:
        try:
            path = _resolve(raw)
        except Exception as e:
            notes.append(f"  @{raw}: resolve failed — {e}")
            continue

        if not path.exists():
            notes.append(f"  @{raw}: not found")
            continue
        if not path.is_file():
            notes.append(f"  @{raw}: not a file")
            continue
        if not fs_config.is_path_safe(path):
            notes.append(f"  @{raw}: path not allowed (must live under home or cwd)")
            continue

        content, status = _extract(path)
        if not content:
            notes.append(f"  @{raw}: {status}")
            continue

        blocks.append(f'<attached path="{path}">\n{content}\n</attached>')
        notes.append(f"  @{raw} -> {path} ({status})")

    if blocks:
        enriched = task + "\n\n" + "\n\n".join(blocks)
    else:
        enriched = task
    return enriched, notes
