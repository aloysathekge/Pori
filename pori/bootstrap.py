"""Startup bootstrap for the Pori CLI (CLI-3).

Windows-only UTF-8 setup so unicode in agent output (emoji, accented text) and
spawned child processes don't crash on the legacy cp1252 console. POSIX is
untouched by design. Idempotent, and opt-out-able via ``PYTHONUTF8=0``.
"""

from __future__ import annotations

import os
import sys

_bootstrap_applied = False


def apply_windows_utf8_bootstrap() -> None:
    """On Windows, make the console + child processes speak UTF-8. No-op elsewhere."""
    global _bootstrap_applied
    if _bootstrap_applied:
        return
    _bootstrap_applied = True
    if sys.platform != "win32":
        return
    # Child processes inherit UTF-8 (setdefault so PYTHONUTF8=0 opts out).
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    # Reconfigure the current process's streams so print() of unicode doesn't
    # crash on cp1252, without needing a re-exec.
    for stream in (sys.stdout, sys.stderr, sys.stdin):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
