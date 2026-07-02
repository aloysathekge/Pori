#!/usr/bin/env python
"""Fail if any runtime/extra dependency uses an unbounded ``>=`` (no upper bound).

Bounded ranges (``>=floor,<next_major``) stop a malicious or breaking new major
from slipping in via a fresh install before ``uv.lock`` pins it (a Shai-Hulud-
style supply-chain slip). Build-system ``requires`` are intentionally not checked
— they pin the build tool and are conventionally unbounded.

Usage: ``python tools/ci/check_dep_bounds.py`` (exit 1 on any violation).
"""

from __future__ import annotations

import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore


def _check(specs, where: str, out: list) -> None:
    for spec in specs:
        s = spec.replace(" ", "")
        if ">=" in s and "<" not in s and "==" not in s and "~=" not in s:
            out.append(f"{where}: {spec!r} has no upper bound")


def main() -> int:
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    project = data.get("project", {})
    out: list = []
    _check(project.get("dependencies", []), "dependencies", out)
    for extra, specs in (project.get("optional-dependencies", {}) or {}).items():
        _check(specs, f"optional-dependencies.{extra}", out)
    if out:
        print("Unbounded dependency ranges (add ',<next_major'):")
        for v in out:
            print("  -", v)
        return 1
    print("All project dependencies are bounded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
