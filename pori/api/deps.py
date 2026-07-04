from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import Request

from pori.llm import ChatAnthropic
from pori.memory import AgentMemory, MemoryStore, create_memory_store
from pori.orchestrator import Orchestrator
from pori.tools.registry import tool_registry
from pori.tools.standard import register_all_tools

load_dotenv()


def build_memory_store() -> MemoryStore:
    """Build the shared, namespace-keyed persistent memory store.

    One store is shared across all API requests. Per-request ``AgentMemory``
    instances (see :func:`get_request_memory`) isolate callers by namespace, so
    the store is the *only* shared piece — never a caller's transcript.
    """
    return create_memory_store(
        backend=os.getenv("PORI_MEMORY_BACKEND", "memory"),
        sqlite_path=os.getenv("PORI_MEMORY_SQLITE_PATH"),
    )


def build_orchestrator() -> Orchestrator:
    """Build the shared orchestrator (LLM + tools).

    Deliberately built **without** a ``shared_memory``: API requests must not
    share a single transcript. Each request gets its own ``AgentMemory`` via
    :func:`get_request_memory`, sharing only the app-level ``MemoryStore``.
    """
    registry = tool_registry()
    register_all_tools(registry)

    model_name = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    llm = ChatAnthropic(
        model=model_name,
        temperature=0.0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    return Orchestrator(
        llm=llm,
        tools_registry=registry,
    )


def get_orchestrator(request: Request) -> Orchestrator:
    """Dependency to retrieve the pre-configured (shared) orchestrator."""
    return request.app.state.orchestrator


def get_clarify_bridges(request: Request) -> set:
    """The app-level set of active per-stream clarify bridges."""
    return request.app.state.clarify_bridges


def get_request_memory(request: Request) -> AgentMemory:
    """Per-request ``AgentMemory`` — the isolation boundary.

    Each request (hence each task) gets its own ``AgentMemory`` with a unique,
    auto-generated session, sharing only the app-level ``MemoryStore``. Because
    the store is namespace-keyed (``organization:user:agent:session``), distinct
    sessions never collide, so concurrent callers cannot read or overwrite each
    other's transcript.

    Tenant identity (org/user/agent) currently comes from the deployment
    environment (single-tenant). When request-scoped identity lands
    (auth -> tenant), resolve it here — the isolation seam is already in place.
    A fixed ``PORI_MEMORY_SESSION_ID`` is intentionally NOT honored here: pinning
    one session across requests is exactly the transcript-bleed this prevents.
    """
    return AgentMemory(
        organization_id=os.getenv("PORI_MEMORY_ORGANIZATION_ID", "default_org"),
        user_id=os.getenv("PORI_MEMORY_USER_ID", "api_user"),
        agent_id=os.getenv("PORI_MEMORY_AGENT_ID", "api_agent"),
        # No fixed session_id: each request gets a fresh, isolated session.
        store=request.app.state.memory_store,
    )
