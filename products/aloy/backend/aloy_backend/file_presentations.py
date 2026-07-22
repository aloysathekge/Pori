"""Trusted file presentation classification and bounded Office previews.

Original bytes remain immutable in object storage. This module never executes
HTML: it returns typed JSON which the Aloy host maps to fixed viewer
components. The host may place an HTML artifact in its isolated preview frame;
generated Surfaces do not participate in this path.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import PurePosixPath
from typing import Any

from .doc_extract import (
    ExtractionError,
    extract_docx_blocks,
    extract_pptx_slides,
    extract_xlsx_sheets,
)

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

_DOCX_SUFFIXES = {".docx"}
_SHEET_SUFFIXES = {".xlsx", ".xlsm"}
_SLIDE_SUFFIXES = {".pptx"}
_MARKDOWN_SUFFIXES = {".md", ".mdx"}
_HTML_SUFFIXES = {".htm", ".html"}
_CODE_SUFFIXES = {
    ".astro",
    ".bash",
    ".c",
    ".cjs",
    ".conf",
    ".cpp",
    ".cs",
    ".css",
    ".dart",
    ".diff",
    ".env",
    ".fish",
    ".go",
    ".gql",
    ".graphql",
    ".h",
    ".hpp",
    ".ini",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".kts",
    ".less",
    ".lua",
    ".mjs",
    ".patch",
    ".php",
    ".prisma",
    ".ps1",
    ".py",
    ".r",
    ".rb",
    ".rs",
    ".sass",
    ".scala",
    ".scss",
    ".sh",
    ".sql",
    ".svelte",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".vue",
    ".xml",
    ".yaml",
    ".yml",
    ".zsh",
}
_CODE_MEDIA_TYPES = {
    "application/javascript",
    "application/json",
    "application/ld+json",
    "application/typescript",
    "application/xml",
    "text/css",
    "text/html",
    "text/javascript",
    "text/typescript",
    "text/xml",
}
_IMAGE_SUFFIXES = {
    ".avif",
    ".bmp",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".png",
    ".svg",
    ".webp",
}
_VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".ogv", ".webm"}
_AUDIO_SUFFIXES = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
}
_TEXT_SUFFIXES = {
    ".txt",
    ".log",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
}

_MAX_OFFICE_BYTES = 25 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 10_000
_MAX_ARCHIVE_EXPANDED_BYTES = 100 * 1024 * 1024
_MAX_PREVIEW_CHARS = 500_000
_MAX_PREVIEW_CELLS = 20_000
_MAX_PREVIEW_SLIDES = 200
TEXT_PREVIEW_READ_LIMIT = 2 * 1024 * 1024 + 1


def presentation_kind(name: str, content_type: str) -> str:
    """Resolve one stable host renderer from verified metadata and suffix."""
    suffix = PurePosixPath(name).suffix.lower()
    media_type = content_type.split(";", 1)[0].strip().lower()

    # A recognized filename suffix wins over ambiguous platform MIME metadata.
    # Windows commonly uploads TypeScript `.ts` files as `video/mp2t`.
    if suffix == ".pdf":
        return "pdf"
    if suffix in _IMAGE_SUFFIXES:
        return "image"
    if suffix in _VIDEO_SUFFIXES:
        return "video"
    if suffix in _AUDIO_SUFFIXES:
        return "audio"
    if suffix in _DOCX_SUFFIXES:
        return "document"
    if suffix in _SHEET_SUFFIXES:
        return "spreadsheet"
    if suffix in _SLIDE_SUFFIXES:
        return "slides"
    if suffix in _MARKDOWN_SUFFIXES:
        return "markdown"
    if suffix in _HTML_SUFFIXES:
        return "html"
    if suffix in _CODE_SUFFIXES:
        return "code"
    if suffix in _TEXT_SUFFIXES:
        return "text"

    if media_type == "application/pdf":
        return "pdf"
    if media_type.startswith("image/"):
        return "image"
    if media_type.startswith("video/"):
        return "video"
    if media_type.startswith("audio/"):
        return "audio"
    if media_type == DOCX_MIME:
        return "document"
    if media_type == XLSX_MIME:
        return "spreadsheet"
    if media_type == PPTX_MIME:
        return "slides"
    if media_type == "text/markdown":
        return "markdown"
    if media_type == "text/html":
        return "html"
    if media_type in _CODE_MEDIA_TYPES:
        return "code"
    if media_type.startswith("text/"):
        return "text"
    return "unknown"


def build_text_preview(raw: bytes) -> dict[str, Any]:
    """Decode a bounded, inert source/text preview without executing content."""
    byte_truncated = len(raw) >= TEXT_PREVIEW_READ_LIMIT
    bounded = raw[: TEXT_PREVIEW_READ_LIMIT - 1]
    try:
        text = bounded.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = bounded.decode("utf-8", errors="replace")
    char_truncated = len(text) > _MAX_PREVIEW_CHARS
    if char_truncated:
        text = text[:_MAX_PREVIEW_CHARS]
    return {"text": text, "truncated": byte_truncated or char_truncated}


def build_office_preview(kind: str, raw: bytes) -> dict[str, Any]:
    """Create bounded, inert JSON for one supported OOXML file."""
    _validate_office_archive(raw)
    if kind == "document":
        blocks = extract_docx_blocks(raw)
        document_blocks, truncated = _bounded_strings(blocks)
        return {"blocks": document_blocks, "truncated": truncated}
    if kind == "spreadsheet":
        return _bounded_sheets(extract_xlsx_sheets(raw))
    if kind == "slides":
        slides = extract_pptx_slides(raw)
        slide_previews: list[dict[str, object]] = []
        used = 0
        truncated = len(slides) > _MAX_PREVIEW_SLIDES
        for slide in slides[:_MAX_PREVIEW_SLIDES]:
            text = str(slide.get("text", ""))
            remaining = _MAX_PREVIEW_CHARS - used
            if remaining <= 0:
                truncated = True
                break
            if len(text) > remaining:
                text = text[:remaining]
                truncated = True
            slide_previews.append(
                {
                    "number": slide.get("number", len(slide_previews) + 1),
                    "text": text,
                }
            )
            used += len(text)
        return {"slides": slide_previews, "truncated": truncated}
    raise ExtractionError(f"No Office previewer for {kind}")


def _validate_office_archive(raw: bytes) -> None:
    if len(raw) > _MAX_OFFICE_BYTES:
        raise ExtractionError("Office preview exceeds the 25 MB preview limit")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
            if len(infos) > _MAX_ARCHIVE_MEMBERS:
                raise ExtractionError("Office archive contains too many members")
            expanded = sum(info.file_size for info in infos)
            if expanded > _MAX_ARCHIVE_EXPANDED_BYTES:
                raise ExtractionError("Office archive expands beyond the preview limit")
            if any(info.flag_bits & 0x1 for info in infos):
                raise ExtractionError(
                    "Password-protected Office files cannot be previewed"
                )
    except zipfile.BadZipFile as exc:
        raise ExtractionError(f"Invalid Office archive: {exc}") from exc


def _bounded_strings(values: list[str]) -> tuple[list[str], bool]:
    out: list[str] = []
    used = 0
    truncated = False
    for value in values:
        remaining = _MAX_PREVIEW_CHARS - used
        if remaining <= 0:
            truncated = True
            break
        if len(value) > remaining:
            value = value[:remaining]
            truncated = True
        out.append(value)
        used += len(value)
    return out, truncated


def _bounded_sheets(sheets: list[dict[str, object]]) -> dict[str, Any]:
    out: list[dict[str, object]] = []
    used_chars = 0
    used_cells = 0
    truncated = False
    for sheet in sheets:
        next_rows: list[list[str]] = []
        rows = sheet.get("rows", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, list):
                continue
            next_row: list[str] = []
            for value in row:
                if used_cells >= _MAX_PREVIEW_CELLS or used_chars >= _MAX_PREVIEW_CHARS:
                    truncated = True
                    break
                text = str(value)
                remaining = _MAX_PREVIEW_CHARS - used_chars
                if len(text) > remaining:
                    text = text[:remaining]
                    truncated = True
                next_row.append(text)
                used_cells += 1
                used_chars += len(text)
            next_rows.append(next_row)
            if truncated:
                break
        out.append({"name": str(sheet.get("name", "Sheet")), "rows": next_rows})
        if truncated:
            break
    return {"sheets": out, "truncated": truncated}


__all__ = [
    "DOCX_MIME",
    "PPTX_MIME",
    "XLSX_MIME",
    "build_office_preview",
    "build_text_preview",
    "presentation_kind",
    "TEXT_PREVIEW_READ_LIMIT",
]
