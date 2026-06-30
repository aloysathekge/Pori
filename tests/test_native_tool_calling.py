"""Phase B.1: native tool-calling type/schema foundations (additive)."""

from pori.llm import AssistantMessage, ToolCall, ToolResultMessage, ToolTurn
from pori.tools.registry import ToolRegistry
from pori.tools.standard import register_all_tools


def test_tool_call_defaults():
    tc = ToolCall(name="write_file")
    assert tc.name == "write_file"
    assert tc.id == ""
    assert tc.arguments == {}


def test_assistant_message_carries_tool_calls():
    msg = AssistantMessage(
        content="Writing the file",
        tool_calls=[ToolCall(id="t1", name="write_file", arguments={"path": "a"})],
    )
    assert msg.role == "assistant"
    assert msg.tool_calls[0].name == "write_file"
    # Legacy plain-text assistant messages still work (content-only).
    assert AssistantMessage(content="hi").tool_calls == []


def test_tool_result_message():
    msg = ToolResultMessage(tool_call_id="t1", content="wrote a")
    assert msg.role == "tool"
    assert msg.tool_call_id == "t1"


def test_tool_turn_holds_text_and_calls():
    turn = ToolTurn(
        text="Saving the report",
        tool_calls=[ToolCall(name="write_file", arguments={"path": "r.md"})],
    )
    assert turn.text == "Saving the report"
    assert turn.tool_calls[0].arguments["path"] == "r.md"
    assert ToolTurn().tool_calls == []


def test_registry_tool_schemas_shape():
    registry = ToolRegistry()
    register_all_tools(registry)
    schemas = registry.tool_schemas()
    by_name = {s["name"]: s for s in schemas}

    assert "write_file" in by_name
    wf = by_name["write_file"]
    assert set(wf) == {"name", "description", "input_schema"}
    assert wf["description"]
    # input_schema is a JSON Schema with the tool's parameters.
    assert wf["input_schema"]["type"] == "object"
    assert "file_path" in wf["input_schema"]["properties"]
