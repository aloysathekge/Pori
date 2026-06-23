from pydantic import BaseModel, Field

from pori.evolution import EvolutionRepository
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


def test_propose_evolution_tool_submits_inert_proposal():
    from pori.tools.registry import ToolExecutor, tool_registry
    from pori.tools.standard import register_all_tools

    registry = tool_registry()
    register_all_tools(registry)
    repository = EvolutionRepository()
    executor = ToolExecutor(registry)

    result = executor.execute_tool(
        "propose_evolution",
        {
            "artifact_kind": "skill",
            "target": "skills/brainstorming",
            "title": "Improve brainstorming",
            "summary": "Ask questions before implementation.",
            "rationale": "Repeated build tasks need design-before-build behavior.",
            "current_version": "0",
            "proposed_version": "1",
            "proposed_content": "Ask one clarifying question first.",
            "eval_cases": [
                {
                    "name": "asks-before-coding",
                    "input": "Build a sync workflow",
                    "expected": "A clarifying question first",
                    "criteria": "The answer asks a clarifying question first.",
                }
            ],
        },
        context={"evolution_repository": repository},
    )

    assert result["success"] is True
    proposal_id = result["result"]["proposal_id"]
    proposal = repository.get(proposal_id)
    assert proposal.status.value == "proposed"
    assert proposal.target == "skills/brainstorming"


def test_propose_evolution_tool_requires_repository():
    from pori.tools.registry import ToolExecutor, tool_registry
    from pori.tools.standard import register_all_tools

    registry = tool_registry()
    register_all_tools(registry)
    executor = ToolExecutor(registry)

    result = executor.execute_tool(
        "propose_evolution",
        {
            "artifact_kind": "skill",
            "target": "skills/brainstorming",
            "title": "Improve brainstorming",
            "summary": "Ask questions before implementation.",
            "rationale": "Repeated build tasks need design-before-build behavior.",
            "proposed_version": "1",
            "proposed_content": "Ask one clarifying question first.",
            "eval_cases": [
                {
                    "name": "asks-before-coding",
                    "input": "Build a sync workflow",
                    "criteria": "The answer asks a clarifying question first.",
                }
            ],
        },
        context={},
    )

    assert result["success"] is True
    assert result["result"]["success"] is False
    assert "repository not available" in result["result"]["error"]
