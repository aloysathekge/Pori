"""Kernel MCP client — connect external MCP tool servers per run (session-scoped).

Gated capability: the SDK transport needs the optional ``pori[mcp]`` extra, but
importing this package never requires it (the SDK is lazy-imported). A product
supplies per-run ``McpServerConfig`` list (with resolved auth headers); the
kernel connects them, registers their tools into the run's registry, and tears
down at run end. See docs/pori-mcp-spec.md.
"""

from .config import McpServerConfig
from .session import McpConnection, McpSessionSet, McpToolDesc

__all__ = [
    "McpServerConfig",
    "McpSessionSet",
    "McpConnection",
    "McpToolDesc",
]
