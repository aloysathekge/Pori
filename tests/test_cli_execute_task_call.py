"""Regression: the CLI's execute_task call must pass `task`.

The interactive REPL in pori/main.py isn't exercised at runtime by the suite, so a
call-site edit that drops the required `task` argument (as happened when the
background-delegation prepend was added) would otherwise only surface at runtime.
This static check guards that specific class of breakage.
"""

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]


def test_main_execute_task_calls_pass_task():
    src = Path(__file__).resolve().parent.parent / "pori" / "main.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "execute_task"
    ]
    assert calls, "expected at least one execute_task call in main.py"
    for call in calls:
        has_task = bool(call.args) or any(kw.arg == "task" for kw in call.keywords)
        assert has_task, f"execute_task at main.py:{call.lineno} is missing `task`"
