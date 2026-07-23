"""The bundled baseline Surface every custom Event starts from.

See ``docs/aloy-baseline-surface-spec.md``: the Builder never sees an empty
workspace. This module only loads the reviewed template source; delivery uses
the ordinary template-release and model-free materialization boundaries.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

_BASELINE_DIR = Path(__file__).resolve().parent / "product_surfaces" / "baseline"


@lru_cache(maxsize=1)
def baseline_surface_files() -> dict[str, str]:
    """Return the baseline source as canonical workspace paths -> content."""
    files: dict[str, str] = {}
    for path in sorted(_BASELINE_DIR.rglob("*")):
        if not path.is_file():
            continue
        files["/" + path.relative_to(_BASELINE_DIR).as_posix()] = path.read_text(
            encoding="utf-8"
        )
    if "/surface.json" not in files or "/src/App.tsx" not in files:
        raise RuntimeError("The bundled baseline Surface template is incomplete")
    return dict(files)


def baseline_surface_fingerprint() -> str:
    """Stable content fingerprint used by release/materialization identity."""
    encoded = json.dumps(
        baseline_surface_files(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["baseline_surface_files", "baseline_surface_fingerprint"]
