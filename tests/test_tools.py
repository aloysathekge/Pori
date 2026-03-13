from pydantic import BaseModel, Field

from pori.memory import AgentMemory


def test_tool_registry_register_and_execute(tool_registry):
    """Register a tiny tool and execute it through ToolExecutor."""
    from pori.tools.registry import ToolExecutor

    # Tool is provided by the fixture (test_tool)
    executor = ToolExecutor(tool_registry)

    result = executor.execute_tool(
        tool_name="test_tool",
        params={"param1": "abc", "param2": 7},
        context={},
    )

    assert result["success"] is True
    assert "Test result: abc, 7" in result["result"]


def test_tool_registry_decorator_registration():
    """Use the decorator to register a tool and then run it."""
    from pori.tools.registry import ToolExecutor, ToolRegistry

    registry = ToolRegistry()

    class EchoParams(BaseModel):
        text: str = Field(..., description="Text to echo")

    @registry.tool(name="echo", param_model=EchoParams, description="Echo text")
    def echo_tool(params: EchoParams, context):
        return params.text.upper()

    executor = ToolExecutor(registry)
    result = executor.execute_tool("echo", {"text": "hello"}, context={})

    assert result["success"] is True
    assert result["result"] == "HELLO"


def test_core_memory_insert_and_rethink_tools():
    from pori.tools.registry import ToolExecutor, tool_registry
    from pori.tools.standard import register_all_tools

    registry = tool_registry()
    register_all_tools(registry)
    executor = ToolExecutor(registry)
    memory = AgentMemory()
    ctx = {"memory": memory, "state": {}}

    ins = executor.execute_tool(
        "memory_insert",
        {"label": "notes", "new_str": "line1", "insert_line": 0},
        context=ctx,
    )
    assert ins["success"] is True
    assert "line1" in memory.core_memory.get_block("notes").value

    rethink = executor.execute_tool(
        "memory_rethink",
        {"label": "notes", "new_memory": "fresh memory"},
        context=ctx,
    )
    assert rethink["success"] is True
    assert memory.core_memory.get_block("notes").value == "fresh memory"
