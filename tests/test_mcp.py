"""Kernel MCP client — schema translation, session-scoped registration/dispatch,
and per-run isolation. Uses an injected fake connection, so no `mcp` SDK needed.
"""

from contextlib import asynccontextmanager

import pytest

from pori.mcp import McpServerConfig, McpSessionSet
from pori.mcp.schema import normalize_mcp_input_schema
from pori.tools.registry import ToolRegistry

pytestmark = pytest.mark.mcp


# --- schema translation ---------------------------------------------------


class TestSchemaTranslation:
    def test_definitions_become_defs(self):
        out = normalize_mcp_input_schema(
            {
                "type": "object",
                "definitions": {"X": {"type": "string"}},
                "properties": {"a": {"$ref": "#/definitions/X"}},
            }
        )
        assert "$defs" in out and "definitions" not in out
        assert out["properties"]["a"]["$ref"] == "#/$defs/X"

    def test_missing_object_type_coerced(self):
        out = normalize_mcp_input_schema({"properties": {"a": {"type": "string"}}})
        assert out["type"] == "object"

    def test_dangling_required_pruned(self):
        out = normalize_mcp_input_schema(
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a", "ghost"],
            }
        )
        assert out["required"] == ["a"]

    def test_nullable_union_collapsed(self):
        out = normalize_mcp_input_schema(
            {
                "type": "object",
                "properties": {"a": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
            }
        )
        assert out["properties"]["a"].get("type") == "string"
        assert "anyOf" not in out["properties"]["a"]

    def test_non_dict_is_safe(self):
        assert normalize_mcp_input_schema(None) == {"type": "object", "properties": {}}


# --- fake connection ------------------------------------------------------


class FakeConnection:
    def __init__(self, tools, on_call=None):
        self._tools = tools
        self._on_call = on_call
        self.entered = False
        self.exited = False

    async def list_tools(self):
        from pori.mcp import McpToolDesc

        return [McpToolDesc(**t) for t in self._tools]

    async def call_tool(self, name, arguments):
        if self._on_call:
            return self._on_call(name, arguments)
        return f"called {name} with {arguments}"


def make_factory(conn: FakeConnection):
    @asynccontextmanager
    async def factory(config):
        conn.entered = True
        try:
            yield conn
        finally:
            conn.exited = True

    return factory


SERVER = McpServerConfig(name="acme", url="http://x", transport="http")


# --- session set ----------------------------------------------------------


class TestSessionSet:
    def test_registers_namespaced_tools_in_group(self):
        conn = FakeConnection(
            [
                {"name": "search", "description": "Search", "input_schema": {}},
                {"name": "fetch-page", "description": "Fetch", "input_schema": {}},
            ]
        )
        reg = ToolRegistry()
        s = McpSessionSet([SERVER], connection_factory=make_factory(conn))
        try:
            n = s.connect_and_register(reg)
            assert n == 2
            assert "mcp__acme__search" in reg.tools
            assert "mcp__acme__fetch_page" in reg.tools  # sanitized
            assert "mcp:acme" in reg.groups
        finally:
            s.close()

    def test_call_dispatches_to_server(self):
        seen = {}

        def on_call(name, args):
            seen["name"] = name
            seen["args"] = args
            return "OK-RESULT"

        conn = FakeConnection(
            [{"name": "ping", "description": "", "input_schema": {}}], on_call=on_call
        )
        reg = ToolRegistry()
        s = McpSessionSet([SERVER], connection_factory=make_factory(conn))
        try:
            s.connect_and_register(reg)
            handler = reg.tools["mcp__acme__ping"].function
            params = reg.tools["mcp__acme__ping"].param_model(x=1)
            out = handler(params, {})
            assert out == {"result": "OK-RESULT"}
            assert seen["name"] == "ping"
            assert seen["args"]["x"] == 1
        finally:
            s.close()

    def test_tools_include_filter(self):
        conn = FakeConnection(
            [
                {"name": "keep", "description": "", "input_schema": {}},
                {"name": "drop", "description": "", "input_schema": {}},
            ]
        )
        reg = ToolRegistry()
        cfg = McpServerConfig(name="acme", url="http://x", tools_include=["keep"])
        s = McpSessionSet([cfg], connection_factory=make_factory(conn))
        try:
            s.connect_and_register(reg)
            assert "mcp__acme__keep" in reg.tools
            assert "mcp__acme__drop" not in reg.tools
        finally:
            s.close()

    def test_failed_server_is_skipped_not_fatal(self):
        @asynccontextmanager
        async def bad_factory(config):
            raise RuntimeError("connection refused")
            yield  # pragma: no cover

        reg = ToolRegistry()
        s = McpSessionSet([SERVER], connection_factory=bad_factory)
        try:
            n = s.connect_and_register(reg)
            assert n == 0
            assert not any(t.startswith("mcp__") for t in reg.tools)
        finally:
            s.close()

    def test_check_fn_false_after_close(self):
        conn = FakeConnection([{"name": "ping", "description": "", "input_schema": {}}])
        reg = ToolRegistry()
        s = McpSessionSet([SERVER], connection_factory=make_factory(conn))
        s.connect_and_register(reg)
        check = reg.tools["mcp__acme__ping"].check_fn
        assert check() is True
        s.close()
        assert check() is False  # tool vanishes from the surface after teardown
        assert conn.exited is True  # transport exited in its own task

    def test_no_leak_into_source_registry(self):
        """Registering into a per-run copy must not touch the base registry."""
        conn = FakeConnection([{"name": "ping", "description": "", "input_schema": {}}])
        base = ToolRegistry()
        run_registry = base.filtered()  # the per-run copy the orchestrator uses
        s = McpSessionSet([SERVER], connection_factory=make_factory(conn))
        try:
            s.connect_and_register(run_registry)
            assert "mcp__acme__ping" in run_registry.tools
            assert "mcp__acme__ping" not in base.tools  # base untouched
        finally:
            s.close()


class TestOrchestratorWiring:
    async def test_run_connects_and_tears_down_without_leaking(
        self, mock_llm, tool_registry
    ):
        from pori.orchestrator.core import Orchestrator

        conn = FakeConnection([{"name": "ping", "description": "", "input_schema": {}}])
        orch = Orchestrator(
            llm=mock_llm,
            tools_registry=tool_registry,
            mcp_connection_factory=make_factory(conn),
        )
        await orch.execute_task("say hi", mcp_servers=[SERVER])
        # Session-scoped: connected during the run, torn down after.
        assert conn.entered is True
        assert conn.exited is True
        # The orchestrator's shared registry never gains the run's MCP tools.
        assert "mcp__acme__ping" not in tool_registry.tools
