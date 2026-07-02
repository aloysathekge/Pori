"""Per-tool check_fn gating in the registry snapshot (SK-6)."""

import pytest
from pydantic import BaseModel

pytestmark = [pytest.mark.tools]


class _NoParams(BaseModel):
    pass


def _noop(params, context):
    return {}


def test_check_fn_gates_tool_in_snapshot(tool_registry):
    tool_registry.register_tool(
        "gated_off", _NoParams, _noop, "off", check_fn=lambda: False
    )
    tool_registry.register_tool(
        "gated_on", _NoParams, _noop, "on", check_fn=lambda: True
    )

    snap = tool_registry.snapshot()
    names = {name for name, _ in snap.tool_items}

    assert "gated_on" in names  # predicate True -> visible
    assert "gated_off" not in names  # predicate False -> dropped
    assert ("gated_off", "check_fn:false") in snap.excluded


def test_check_fn_failure_is_treated_as_unavailable(tool_registry):
    def boom():
        raise RuntimeError("probe failed")

    tool_registry.register_tool("gated_err", _NoParams, _noop, "err", check_fn=boom)
    snap = tool_registry.snapshot()
    assert "gated_err" not in {name for name, _ in snap.tool_items}
