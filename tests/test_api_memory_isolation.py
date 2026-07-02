"""Isolation guarantees for the API's per-request memory (Phase 0, fix #1).

Before this fix, the API built one ``AgentMemory`` at startup and handed the same
instance to every request, so concurrent callers shared one transcript. Now each
request gets its own ``AgentMemory`` (unique session/namespace) sharing only the
app-level ``MemoryStore``. These tests lock that in.

The ``get_request_memory`` test needs ``fastapi`` (imported by ``pori.api``) and is
skipped where it is absent; the orchestrator-level test exercises the underlying
isolation mechanism (``execute_task`` honoring an injected memory) and always runs.
"""

from types import SimpleNamespace

import pytest

from pori.memory import AgentMemory, create_memory_store

pytestmark = [pytest.mark.orchestrator]


def _fake_request(store):
    """A minimal stand-in for a FastAPI Request exposing app.state.memory_store."""
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(memory_store=store))
    )


def test_get_request_memory_isolates_concurrent_callers():
    """Two requests get distinct sessions sharing one store — no transcript bleed."""
    pytest.importorskip("fastapi")
    from pori.api.deps import get_request_memory

    store = create_memory_store(backend="memory")
    req = _fake_request(store)

    m1 = get_request_memory(req)
    m2 = get_request_memory(req)

    # Distinct instances and distinct namespaces (unique session per request)...
    assert m1 is not m2
    assert m1.namespace != m2.namespace
    # ...but the SAME shared store is the only thing in common.
    assert m1.store is store
    assert m2.store is store

    # A message written and persisted by caller 1 stays in caller 1's namespace;
    # caller 2 (a different namespace) never sees it.
    m1.add_message("user", "secret-from-caller-1")
    m1.persist()

    contents_2 = [getattr(msg, "content", None) for msg in m2.messages]
    assert "secret-from-caller-1" not in contents_2
    assert m2.messages == []  # a fresh caller starts empty

    # And the store persisted caller 1's namespace only (round-trips by namespace).
    assert store.load(m1.namespace) is not None
    assert store.load(m2.namespace) is None


async def test_execute_task_uses_injected_memory(orchestrator_with_tool_calls):
    """execute_task uses the per-call memory and does not hijack shared_memory."""
    orch = orchestrator_with_tool_calls
    assert orch.shared_memory is None  # fixture builds it without shared memory

    injected = AgentMemory(store=create_memory_store(backend="memory"))
    result = await orch.execute_task(task="isolated task", memory=injected)

    # The agent used exactly the injected memory...
    assert result["agent"].memory is injected
    # ...and the orchestrator's shared_memory was NOT set as a side effect.
    assert orch.shared_memory is None
