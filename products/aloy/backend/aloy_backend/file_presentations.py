"""Trusted file presentation classification and bounded Office previews.

Original bytes remain immutable in object storage. This module never renders
untrusted HTML: it returns typed JSON which the Aloy host maps to fixed viewer
components. Generated Surfaces do not participate in this path.
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
    ".yaml",
    ".yml",
    ".toml",
    ".xml",
    ".html",
    ".css",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".py",
    ".sql",
    ".sh",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".cpp",
}

_MAX_OFFICE_BYTES = 25 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 10_000
_MAX_ARCHIVE_EXPANDED_BYTES = 100 * 1024 * 1024
_MAX_PREVIEW_CHARS = 500_000
_MAX_PREVIEW_CELLS = 20_000
_MAX_PREVIEW_SLIDES = 200


def presentation_kind(name: str, content_type: str) -> str:
    """Resolve one stable host renderer from verified metadata and suffix."""
    suffix = PurePosixPath(name).suffix.lower()
    media_type = content_type.split(";", 1)[0].strip().lower()

    if media_type == "application/pdf" or suffix == ".pdf":
        return "pdf"
    if media_type.startswith("image/") or suffix in _IMAGE_SUFFIXES:
        return "image"
    if media_type.startswith("video/") or suffix in _VIDEO_SUFFIXES:
        return "video"
    if media_type.startswith("audio/") or suffix in _AUDIO_SUFFIXES:
        return "audio"
    if media_type == DOCX_MIME or suffix in _DOCX_SUFFIXES:
        return "document"
    if media_type == XLSX_MIME or suffix in _SHEET_SUFFIXES:
        return "spreadsheet"
    if media_type == PPTX_MIME or suffix in _SLIDE_SUFFIXES:
        return "slides"
    if media_type == "text/markdown" or suffix in _MARKDOWN_SUFFIXES:
        return "markdown"
    if media_type.startswith("text/") or suffix in _TEXT_SUFFIXES:
        return "text"
    return "unknown"


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
                raise ExtractionError("Password-protected Office files cannot be previewed")
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
    "presentation_kind",
]
