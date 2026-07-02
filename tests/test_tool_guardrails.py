"""Cross-step tool loop guardrail (AC-5): pori/tool_guardrails.py."""

import pytest

from pori.tool_guardrails import ToolCallGuardrailController

pytestmark = [pytest.mark.unit]


def test_exact_failure_warns_then_halts():
    g = ToolCallGuardrailController()
    args = {"path": "a"}
    assert g.after_call("read_file", args, success=False, result={}) is None  # 1st
    d2 = g.after_call("read_file", args, success=False, result={})  # 2nd -> warn
    assert d2 and d2.action == "warn" and d2.reason == "exact_failure"
    d3 = g.after_call("read_file", args, success=False, result={})  # 3rd -> halt
    assert d3 and d3.action == "halt" and d3.reason == "exact_failure_loop"


def test_same_tool_failure_halts_across_different_args():
    g = ToolCallGuardrailController()
    # Distinct args each time keeps per-signature counts at 1, but the same-tool
    # counter accumulates to the halt threshold (6).
    decision = None
    for i in range(6):
        decision = g.after_call("terminal", {"cmd": f"c{i}"}, success=False, result={})
    assert decision and decision.action == "halt"
    assert decision.reason == "same_tool_failure_loop"


def test_idempotent_no_progress_warns_then_halts():
    g = ToolCallGuardrailController()
    args = {"path": "x"}
    res = {"success": True, "output": "same"}
    decisions = [
        g.after_call("read_file", args, success=True, result=res) for _ in range(5)
    ]
    # 1st,2nd -> None; 3rd -> warn; 4th -> None; 5th -> halt
    assert decisions[0] is None and decisions[1] is None
    assert decisions[2] and decisions[2].reason == "no_progress"
    assert decisions[4] and decisions[4].action == "halt"
    assert decisions[4].reason == "no_progress_loop"


def test_mutating_tool_not_checked_for_no_progress():
    g = ToolCallGuardrailController()
    args = {"path": "x", "content": "y"}
    res = {"success": True, "output": "ok"}
    for _ in range(6):
        assert g.after_call("write_file", args, success=True, result=res) is None


def test_success_resets_exact_failure_counter():
    g = ToolCallGuardrailController()
    args = {"path": "a"}
    g.after_call("read_file", args, success=False, result={})
    g.after_call("read_file", args, success=False, result={})  # would warn next
    g.after_call("read_file", args, success=True, result={"output": "ok"})  # reset
    # Back to a clean slate: a single fresh failure is not yet a warn.
    assert g.after_call("read_file", args, success=False, result={}) is None
