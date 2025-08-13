from pydantic import BaseModel, Field


def test_tool_registry_register_and_execute(tool_registry):
    """Register a tiny tool and execute it through ToolExecutor."""
    from pori.tools import ToolExecutor

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
    from pori.tools import ToolRegistry, ToolExecutor

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
