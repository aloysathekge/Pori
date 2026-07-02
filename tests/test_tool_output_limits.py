"""Global tool-output ceiling for uncapped tools."""

import pytest
from pydantic import BaseModel

from pori.tools.registry import DEFAULT_MAX_OUTPUT_CHARS, ToolExecutor

pytestmark = [pytest.mark.tools]


class _NoParams(BaseModel):
    pass


def test_ungrouped_tool_output_is_capped(tool_registry):
    big = "x" * (DEFAULT_MAX_OUTPUT_CHARS + 5000)
    tool_registry.register_tool(
        "bigout", _NoParams, lambda params, ctx: {"data": big}, "big output"
    )
    result = ToolExecutor(tool_registry).execute_tool("bigout", {}, {})
    assert result["success"] is True
    assert result["result"].get("truncated") is True
    assert result["result"]["max_output_chars"] == DEFAULT_MAX_OUTPUT_CHARS


def test_small_output_is_untouched(tool_registry):
    tool_registry.register_tool(
        "smallout", _NoParams, lambda params, ctx: {"ok": True}, "small output"
    )
    result = ToolExecutor(tool_registry).execute_tool("smallout", {}, {})
    assert result["result"] == {"ok": True}  # not wrapped/truncated
