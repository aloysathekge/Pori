"""The single run-outcome finalizer persists EVERYTHING, once.

Both the streaming and non-streaming request paths build one RunOutcome and call
persist_run_outcome — so proving this one finalizer writes the full row set (and
is idempotent) is the structural guard against the "streaming path forgot to
save X" drift that produced three separate bugs.
"""

from types import SimpleNamespace

import pytest
from sqlmodel import select

from aloy_backend.models import (
    Conversation,
    CoreMemoryBlock,
    Run,
    RunEventLog,
    TraceRecord,
    UsageRecord,
)
from aloy_backend.run_outcome import (
    build_run_outcome,
    make_usage_record,
    persist_run_outcome,
)
from aloy_backend.tenancy import OrganizationContext, OrganizationPolicy
from aloy_backend.worker import claim_next_run
from pori import AgentMemory

pytestmark = pytest.mark.asyncio

ORG = "user:test-user"
USER = "test-user"


def _context() -> OrganizationContext:
    return OrganizationContext(
        organization_id=ORG,
        user_id=USER,
        role="owner",
        permissions=("run:create", "run:read"),
        policy=OrganizationPolicy(),
    )


def _agent_result() -> dict:
    return {
        "final_answer": "Hello Aloy!",
        "reasoning": "greeted the user",
        "success": True,
        "steps_taken": 2,
        "metrics": {
            "model": "anthropic/claude-sonnet-4-5",
            "tokens": {"input": 100, "output": 20, "total": 120},
            "cost_usd": "$0.0123",
        },
        "trace": {
            "prompt_fingerprint": "pf-1",
            "tool_surface_fingerprint": "ts-1",
            "execution_receipts": [{"tool": "answer"}],
            "duration": "2.5s",
            "total_spans": 4,
            "status": "ok",
        },
        "artifacts": [{"path": "notes.md"}],
        "plan": [{"step": "answer"}],
        "selected_skills": ["greet@1"],
    }


def _run_context():
    return SimpleNamespace(
        run_id="run-parity-1",
        organization_id=ORG,
        session_id="sess-1",
        agent_id="agent-1",
    )


async def _make_conv(session) -> Conversation:
    conv = Conversation(
        organization_id=ORG, user_id=USER, event_id="evt-run-outcome", title="t"
    )
    session.add(conv)
    await session.commit()
    await session.refresh(conv)
    return conv


class TestFinalizer:
    async def test_usage_prefers_complete_budget_metering(self):
        usage = make_usage_record(
            organization_id=ORG,
            user_id=USER,
            run_id="run-budget-usage",
            conversation_id=None,
            metrics={
                "model": "openai/gpt-4o-mini",
                "tokens": {"input": 10, "output": 5, "total": 15},
                "cost_usd": "$0.0001",
                "budget_usage": {
                    "input_tokens_used": 30,
                    "output_tokens_used": 20,
                    "tokens_used": 50,
                    "cost_used_usd": 0.0002,
                },
            },
        )

        assert usage is not None
        assert usage.input_tokens == 30
        assert usage.output_tokens == 20
        assert usage.total_tokens == 50
        assert usage.estimated_cost == 0.0002

    async def test_persists_the_full_row_set(self, db_session_maker):
        async with db_session_maker() as session:
            conv = await _make_conv(session)
            memory = AgentMemory()
            memory.core_memory.get_block("human").set_value("Name: Aloy")

            outcome = build_run_outcome(
                _agent_result(),
                memory,
                _run_context(),
                task="hi",
                fallback_org=ORG,
                events=[{"type": "run_end", "step": 2}],
            )
            msg = await persist_run_outcome(session, conv, _context(), outcome)

            # Message
            assert msg.content == "Hello Aloy!"
            assert msg.metadata_["run_id"] == "run-parity-1"
            assert msg.metadata_["steps_taken"] == 2
            # Run
            run = await session.get(Run, "run-parity-1")
            assert run is not None and run.success is True
            assert run.event_id == conv.event_id
            assert run.prompt_fingerprint == "pf-1"
            assert run.execution_receipts == [{"tool": "answer"}]
            # Usage
            usage = (
                (
                    await session.execute(
                        select(UsageRecord).where(UsageRecord.run_id == "run-parity-1")
                    )
                )
                .scalars()
                .first()
            )
            assert usage is not None
            assert usage.total_tokens == 120 and usage.input_tokens == 100
            assert abs(usage.estimated_cost - 0.0123) < 1e-9
            assert usage.model == "claude-sonnet-4-5" and usage.provider == "anthropic"
            # Trace
            trace = (
                (
                    await session.execute(
                        select(TraceRecord).where(TraceRecord.run_id == "run-parity-1")
                    )
                )
                .scalars()
                .first()
            )
            assert trace is not None and trace.total_spans == 4
            assert trace.event_id == conv.event_id
            assert abs(trace.duration_seconds - 2.5) < 1e-9
            # Replay event log
            log = (
                (
                    await session.execute(
                        select(RunEventLog).where(RunEventLog.run_id == "run-parity-1")
                    )
                )
                .scalars()
                .first()
            )
            assert log is not None and log.event_count == 1
            assert log.event_id == conv.event_id
            # Core memory persisted
            block = (
                (
                    await session.execute(
                        select(CoreMemoryBlock).where(
                            CoreMemoryBlock.organization_id == ORG,
                            CoreMemoryBlock.user_id == USER,
                            CoreMemoryBlock.label == "human",
                        )
                    )
                )
                .scalars()
                .first()
            )
            assert block is not None and block.value == "Name: Aloy"

    async def test_idempotent_by_run_id(self, db_session_maker):
        """A disconnect-finally firing after a normal finish (same run_id) must
        not double-write."""
        async with db_session_maker() as session:
            conv = await _make_conv(session)
            outcome = build_run_outcome(
                _agent_result(),
                AgentMemory(),
                _run_context(),
                task="hi",
                fallback_org=ORG,
            )
            await persist_run_outcome(session, conv, _context(), outcome)
            await persist_run_outcome(session, conv, _context(), outcome)

            runs = (
                (await session.execute(select(Run).where(Run.id == "run-parity-1")))
                .scalars()
                .all()
            )
            assert len(runs) == 1
            usages = (
                (
                    await session.execute(
                        select(UsageRecord).where(UsageRecord.run_id == "run-parity-1")
                    )
                )
                .scalars()
                .all()
            )
            assert len(usages) == 1

    async def test_persists_non_json_serializable_metrics(self, db_session_maker):
        """Regression: the raw agent result carries rich objects (e.g. a
        TokenUsage pydantic model) in metrics; persisting must JSON-normalize
        them, not throw on flush (which silently dropped the assistant message)."""
        from pydantic import BaseModel as _PBase

        class _TokenUsage(_PBase):
            input: int = 10
            output: int = 5
            total: int = 15

        async with db_session_maker() as session:
            conv = await _make_conv(session)
            result = {
                "final_answer": "hi",
                "success": True,
                "steps_taken": 1,
                "metrics": {
                    "model": "anthropic/claude",
                    "tokens": _TokenUsage(),  # NOT a plain dict
                    "cost_usd": "$0.02",
                },
            }
            outcome = build_run_outcome(
                result, AgentMemory(), _run_context(), task="hi", fallback_org=ORG
            )
            msg = await persist_run_outcome(session, conv, _context(), outcome)
            assert msg.content == "hi"
            usage = (
                (
                    await session.execute(
                        select(UsageRecord).where(UsageRecord.run_id == "run-parity-1")
                    )
                )
                .scalars()
                .first()
            )
            assert usage is not None and usage.total_tokens == 15


class TestTerminalStatusKeepsInlineRunsOutOfTheQueue:
    """Regression: the finalizer records an ALREADY-EXECUTED inline run. It must
    stamp a terminal status — the durable worker claims status=='pending' (the
    Run default), so an unset status made the worker re-execute a finished run
    with max_steps=0 → ExecutionBudget(ge=1) crash, failing every inline run."""

    def _rc(self, run_id: str):
        return SimpleNamespace(
            run_id=run_id,
            organization_id=ORG,
            session_id="sess-1",
            agent_id="agent-1",
        )

    async def test_success_is_completed_and_unclaimable(
        self, db_session_maker, monkeypatch
    ):
        async with db_session_maker() as session:
            conv = await _make_conv(session)
            outcome = build_run_outcome(
                _agent_result(),
                AgentMemory(),
                self._rc("ok-1"),
                task="hi",
                fallback_org=ORG,
            )
            await persist_run_outcome(session, conv, _context(), outcome)
            run = await session.get(Run, "ok-1")
            assert run.status == "completed" and run.max_steps == 0

        # The actual regression: a finished inline run must not be picked up.
        monkeypatch.setattr("aloy_backend.worker.async_session", db_session_maker)
        assert await claim_next_run("worker-x") is None

    async def test_failure_is_failed(self, db_session_maker):
        async with db_session_maker() as session:
            conv = await _make_conv(session)
            result = {**_agent_result(), "success": False}
            outcome = build_run_outcome(
                result,
                AgentMemory(),
                self._rc("fail-1"),
                task="hi",
                fallback_org=ORG,
            )
            await persist_run_outcome(session, conv, _context(), outcome)
            assert (await session.get(Run, "fail-1")).status == "failed"

    async def test_stopped_is_cancelled(self, db_session_maker):
        async with db_session_maker() as session:
            conv = await _make_conv(session)
            result = {**_agent_result(), "success": False, "stopped": True}
            outcome = build_run_outcome(
                result,
                AgentMemory(),
                self._rc("stop-1"),
                task="hi",
                fallback_org=ORG,
            )
            await persist_run_outcome(session, conv, _context(), outcome)
            assert (await session.get(Run, "stop-1")).status == "cancelled"


class TestFinalizerAbsentData:
    async def test_no_trace_no_usage_when_absent(self, db_session_maker):
        """A run with no metrics/trace still persists the message + run cleanly."""
        async with db_session_maker() as session:
            conv = await _make_conv(session)
            result = {"final_answer": "hi", "success": True, "steps_taken": 1}
            outcome = build_run_outcome(
                result, AgentMemory(), _run_context(), task="hi", fallback_org=ORG
            )
            msg = await persist_run_outcome(session, conv, _context(), outcome)
            assert msg.content == "hi"
            usages = (
                (
                    await session.execute(
                        select(UsageRecord).where(UsageRecord.run_id == "run-parity-1")
                    )
                )
                .scalars()
                .all()
            )
            assert usages == []
