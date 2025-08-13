import asyncio


async def run_one_step(agent):
    # Helper to run exactly one async step
    await agent.step()
    return agent


def test_agent_initialization(test_agent):
    """Agent should set up memory, state, and tools."""
    agent = test_agent
    assert agent.state.n_steps == 0
    # System message and initial task message should be in working memory
    msgs = agent.memory.get_recent_messages_structured(5)
    assert len(msgs) >= 2
    # Tool registry should contain our test tools
    assert len(agent.tools_registry.tools) >= 1


def test_agent_executes_tool_calls(async_context, test_agent_with_tool_calls):
    """Agent should execute LLM-suggested tool calls and update memory state."""
    loop = async_context
    agent = test_agent_with_tool_calls

    # Run a single step
    loop.run_until_complete(run_one_step(agent))
    assert agent.state.n_steps == 1
    assert len(agent.memory.tool_call_history) >= 1

    # Run a second step: should populate final_answer via the 'answer' tool
    loop.run_until_complete(run_one_step(agent))
    fa = agent.memory.get_final_answer()
    assert isinstance(fa, dict)
    assert fa.get("final_answer") == "The test was successful"
