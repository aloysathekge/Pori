"""Session-scoped MCP client — the Claude-Code model, not Hermes' global daemon.

An :class:`McpSessionSet` is created PER RUN, connects that run's servers,
registers their tools into the RUN's tool registry, and tears down at run end.
Nothing is process-global, so there is no shared MCP state to leak across
tenants — a product just supplies a different server list per run.

The kernel tool loop is sync; MCP is async. Each set owns one background event
loop (a daemon thread that dies with the set). Each server is held open by a
long-lived coroutine that enters the transport and awaits a close event, then
exits it IN THE SAME TASK (required by anyio cancel scopes); sync tool calls
marshal over via ``run_coroutine_threadsafe``. Connections are injectable so
tests need no ``mcp`` SDK.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
import threading
from typing import Any, Callable, Dict, List, Optional, Protocol

from pydantic import BaseModel, Field

from ..tools.registry import CapabilityGroup, ToolRegistry
from .config import McpServerConfig
from .schema import normalize_mcp_input_schema

logger = logging.getLogger(__name__)

_SANITIZE = re.compile(r"[^A-Za-z0-9_]")
_SECRET = re.compile(
    r"(bearer\s+[\w.\-]+|sk-[\w\-]+|ghp_[\w]+|xox[bap]-[\w\-]+)", re.IGNORECASE
)


def _sanitize(name: str) -> str:
    return _SANITIZE.sub("_", name)


def _scrub(text: str) -> str:
    return _SECRET.sub("[redacted]", text)


class McpToolDesc(BaseModel):
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = Field(default_factory=dict)


class McpConnection(Protocol):
    """A live connection to one MCP server (SDK-backed or a test fake)."""

    async def list_tools(self) -> List[McpToolDesc]: ...

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str: ...


# A connection factory: (config) -> async context manager yielding McpConnection.
ConnectionFactory = Callable[[McpServerConfig], "Any"]


class _ServerRuntime:
    def __init__(self, config: McpServerConfig, connection: McpConnection):
        self.config = config
        self.connection = connection
        self.tools: List[McpToolDesc] = []
        self.alive = True
        self.close_event: Optional[asyncio.Event] = None
        self.serve_future: Optional[concurrent.futures.Future] = None


class McpSessionSet:
    """Connect a run's MCP servers, register their tools, tear down at run end."""

    def __init__(
        self,
        servers: List[McpServerConfig],
        connection_factory: Optional[ConnectionFactory] = None,
    ):
        self._servers = servers
        self._factory = connection_factory or _default_connection_factory
        self._runtimes: Dict[str, _ServerRuntime] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle --------------------------------------------------------

    def _start_loop(self) -> None:
        ready = threading.Event()

        def run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        ready.wait()

    def connect_and_register(self, registry: ToolRegistry) -> int:
        """Connect all servers and register their tools into ``registry``.

        Returns the number of tools registered. A server that fails to connect
        is skipped (its tools don't appear) — never fatal to the run.
        """
        if not self._servers:
            return 0
        self._start_loop()
        registered = 0
        for config in self._servers:
            runtime = self._connect_one(config)
            if runtime is None:
                continue
            self._runtimes[config.name] = runtime
            registered += self._register_server(registry, runtime)
        return registered

    @property
    def connected_server_names(self) -> List[str]:
        """Names of servers that actually connected (products use this to
        distinguish 'connected, zero tools' from 'failed to connect')."""
        return list(self._runtimes.keys())

    def _connect_one(self, config: McpServerConfig) -> Optional[_ServerRuntime]:
        assert self._loop is not None
        ready: concurrent.futures.Future = concurrent.futures.Future()
        serve_future = asyncio.run_coroutine_threadsafe(
            self._serve(config, ready), self._loop
        )
        try:
            runtime = ready.result(timeout=config.timeout_seconds)
        except Exception as exc:
            logger.warning(
                "MCP server %s failed to connect: %s", config.name, _scrub(str(exc))
            )
            return None
        runtime.serve_future = serve_future
        return runtime

    async def _serve(
        self, config: McpServerConfig, ready: concurrent.futures.Future
    ) -> None:
        """Hold one server's connection open (enter + exit in this same task)."""
        try:
            cm = self._factory(config)
            connection = await cm.__aenter__()
        except Exception as exc:  # connect failed
            if not ready.done():
                ready.set_exception(exc)
            return
        runtime = _ServerRuntime(config, connection)
        runtime.close_event = asyncio.Event()
        try:
            runtime.tools = await connection.list_tools()
        except Exception as exc:
            await cm.__aexit__(None, None, None)
            if not ready.done():
                ready.set_exception(exc)
            return
        ready.set_result(runtime)
        try:
            await runtime.close_event.wait()
        finally:
            runtime.alive = False
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                logger.debug("MCP teardown error for %s", config.name, exc_info=True)

    def _register_server(self, registry: ToolRegistry, runtime: _ServerRuntime) -> int:
        group = f"mcp:{_sanitize(runtime.config.name)}"
        tool_names: List[str] = []
        for desc in runtime.tools:
            if not runtime.config.allows_tool(desc.name):
                continue
            full = f"mcp__{_sanitize(runtime.config.name)}__{_sanitize(desc.name)}"
            if full in registry.tools:  # preserve a built-in of the same name
                continue
            model = _params_model_for(normalize_mcp_input_schema(desc.input_schema))
            registry.register_tool(
                name=full,
                param_model=model,
                function=self._make_handler(runtime, desc.name),
                description=desc.description or f"MCP tool {desc.name}",
                check_fn=self._make_check_fn(runtime),
            )
            tool_names.append(full)
        if tool_names:
            try:
                registry.define_group(
                    CapabilityGroup(
                        name=group,
                        description=f"Tools from MCP server '{runtime.config.name}'.",
                        tool_names=frozenset(tool_names),
                    )
                )
            except ValueError:
                pass
        return len(tool_names)

    @staticmethod
    def _make_check_fn(runtime: _ServerRuntime) -> Callable[[], bool]:
        """Liveness gate: the tool leaves the surface once its server is gone."""
        return lambda: runtime.alive

    def _make_handler(self, runtime: _ServerRuntime, tool_name: str):
        def handler(params: Any, context: Dict[str, Any]) -> Dict[str, Any]:
            if not runtime.alive or self._loop is None:
                return {
                    "error": f"MCP server '{runtime.config.name}' is not connected."
                }
            args = (
                params.model_dump() if hasattr(params, "model_dump") else dict(params)
            )
            try:
                future = asyncio.run_coroutine_threadsafe(
                    runtime.connection.call_tool(tool_name, args), self._loop
                )
                result = future.result(timeout=runtime.config.timeout_seconds)
            except Exception as exc:
                return {"error": _scrub(f"MCP call failed: {exc}")}
            return {"result": result}

        return handler

    def close(self) -> None:
        for runtime in self._runtimes.values():
            runtime.alive = False
            if runtime.close_event is not None and self._loop is not None:
                self._loop.call_soon_threadsafe(runtime.close_event.set)
        for runtime in self._runtimes.values():  # wait for clean teardown
            if runtime.serve_future is not None:
                try:
                    runtime.serve_future.result(timeout=10)
                except Exception:
                    logger.debug("MCP serve task did not exit cleanly", exc_info=True)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)


def _params_model_for(input_schema: Dict[str, Any]) -> type[BaseModel]:
    """A permissive Pydantic model that exposes the server's JSON Schema verbatim.

    MCP arg shapes are open-ended: validate loosely (extra allowed) and surface
    the real JSON Schema to the LLM via the tool's ``input_schema``.
    """

    class _McpParams(BaseModel):
        model_config = {"extra": "allow"}

        @classmethod
        def model_json_schema(cls, *a: Any, **k: Any) -> Dict[str, Any]:
            return input_schema

    return _McpParams


def _default_connection_factory(config: McpServerConfig):
    from .sdk_transport import sdk_connection

    return sdk_connection(config)


__all__ = ["McpConnection", "McpSessionSet", "McpToolDesc"]
