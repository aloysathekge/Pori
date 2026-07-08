"""MCP server configuration (kernel-side, tenancy-blind).

A product (or the CLI) supplies a list of these per run. Auth is a RESOLVED
token/header passed in — the kernel does not own OAuth; a product resolves the
credential (e.g. via its connect-engine) and hands over a ready header.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class McpServerConfig(BaseModel):
    """One MCP server to connect for a run."""

    name: str = Field(
        ..., description="Short id; namespaces the tools as mcp__<name>__<tool>"
    )
    transport: Literal["http", "sse"] = "http"
    url: str = Field("", description="Endpoint for http/sse transports")
    # Resolved auth headers (e.g. {'Authorization': 'Bearer ...'}). The kernel
    # does not perform OAuth — a product injects the ready credential.
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = 300.0
    # Optional per-server tool filtering (context-cost discipline).
    tools_include: Optional[List[str]] = None
    tools_exclude: List[str] = Field(default_factory=list)

    def allows_tool(self, raw_tool_name: str) -> bool:
        if raw_tool_name in self.tools_exclude:
            return False
        if self.tools_include is not None:
            return raw_tool_name in self.tools_include
        return True
