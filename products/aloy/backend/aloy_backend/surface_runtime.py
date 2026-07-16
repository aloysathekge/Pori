"""Validate immutable Surface bundles and wrap them in an isolated document.

The bundle is untrusted even though the compiler is Aloy-owned.  Runtime HTML
is always constructed by the host so generated code cannot choose its CSP,
iframe privileges, or trusted chrome.
"""

from __future__ import annotations

import io
import re
import secrets
import zipfile
from dataclasses import dataclass

MAX_RUNTIME_ENTRY_BYTES = 4 * 1024 * 1024
MAX_RUNTIME_UNCOMPRESSED_BYTES = 6 * 1024 * 1024
_ALLOWED_ENTRIES = frozenset({"surface.js", "surface.css"})


class InvalidSurfaceBundle(ValueError):
    """Raised when an immutable build artifact is not safe to host."""


@dataclass(frozen=True)
class SurfaceRuntimeDocument:
    html: str
    content_security_policy: str


def _read_entry(archive: zipfile.ZipFile, name: str) -> str:
    info = archive.getinfo(name)
    if info.is_dir() or info.flag_bits & 0x1:
        raise InvalidSurfaceBundle(f"Surface bundle entry {name!r} is invalid")
    if info.file_size > MAX_RUNTIME_ENTRY_BYTES:
        raise InvalidSurfaceBundle(f"Surface bundle entry {name!r} is too large")
    try:
        raw = archive.read(info)
        return raw.decode("utf-8")
    except (RuntimeError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        raise InvalidSurfaceBundle(
            f"Surface bundle entry {name!r} cannot be read as UTF-8"
        ) from exc


def _escape_raw_text_end_tag(value: str, tag: str) -> str:
    # HTML parsers terminate raw-text elements before JavaScript or CSS gets a
    # chance to interpret string literals. Escape every casing of the end tag.
    return re.sub(rf"</{tag}", rf"<\\/{tag}", value, flags=re.IGNORECASE)


def build_surface_runtime_document(bundle: bytes) -> SurfaceRuntimeDocument:
    """Return host-owned HTML for the fixed ``surface.js`` bundle contract."""
    try:
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos if not info.is_dir()]
            if len(names) != len(set(names)):
                raise InvalidSurfaceBundle("Surface bundle contains duplicate entries")
            unknown = set(names) - _ALLOWED_ENTRIES
            if unknown:
                raise InvalidSurfaceBundle(
                    "Surface bundle contains unsupported entries"
                )
            if "surface.js" not in names:
                raise InvalidSurfaceBundle("Surface bundle is missing surface.js")
            if any(
                name.startswith(("/", "\\")) or "\\" in name or ".." in name.split("/")
                for name in names
            ):
                raise InvalidSurfaceBundle("Surface bundle contains an unsafe path")
            if sum(info.file_size for info in infos) > MAX_RUNTIME_UNCOMPRESSED_BYTES:
                raise InvalidSurfaceBundle(
                    "Surface bundle expands beyond the runtime limit"
                )
            script = _read_entry(archive, "surface.js")
            style = (
                _read_entry(archive, "surface.css") if "surface.css" in names else ""
            )
    except zipfile.BadZipFile as exc:
        raise InvalidSurfaceBundle("Surface bundle is not a valid ZIP archive") from exc

    nonce = secrets.token_urlsafe(24)
    policy = "; ".join(
        (
            "default-src 'none'",
            f"script-src 'nonce-{nonce}'",
            f"style-src 'nonce-{nonce}'",
            "connect-src 'none'",
            "img-src data: blob:",
            "font-src data:",
            "media-src data: blob:",
            "object-src 'none'",
            "frame-src 'none'",
            "worker-src 'none'",
            "base-uri 'none'",
            "form-action 'none'",
        )
    )
    safe_script = _escape_raw_text_end_tag(script, "script")
    safe_style = _escape_raw_text_end_tag(style, "style")
    style_element = f'<style nonce="{nonce}">{safe_style}</style>' if style else ""
    document = (
        "<!doctype html><html><head>"
        '<meta charset="utf-8">'
        f'<meta http-equiv="Content-Security-Policy" content="{policy}">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"{style_element}"
        '</head><body><div id="root"></div>'
        f'<script nonce="{nonce}">{safe_script}</script>'
        "</body></html>"
    )
    return SurfaceRuntimeDocument(html=document, content_security_policy=policy)


__all__ = [
    "InvalidSurfaceBundle",
    "MAX_RUNTIME_ENTRY_BYTES",
    "MAX_RUNTIME_UNCOMPRESSED_BYTES",
    "SurfaceRuntimeDocument",
    "build_surface_runtime_document",
]
