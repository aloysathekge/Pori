"""Supply-chain invariant (INF-2): every project dependency has an upper bound.

Behavior contract, not a snapshot: it asserts the *invariant* (no unbounded
``>=``) rather than freezing specific versions, so it stays green as pins move but
fails the moment an unbounded dependency is added.
"""

import importlib.util
import pathlib

import pytest

pytestmark = [pytest.mark.unit]


def _load_checker():
    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "tools"
        / "ci"
        / "check_dep_bounds.py"
    )
    spec = importlib.util.spec_from_file_location("check_dep_bounds", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_project_dependencies_are_bounded():
    checker = _load_checker()
    # main() reads pyproject.toml from the repo root (pytest's working dir) and
    # returns 0 only when every runtime/extra dependency is bounded.
    assert checker.main() == 0
