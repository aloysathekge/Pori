import asyncio
import logging
import os
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from tools import ToolRegistry, ToolExecutor
from agent import Agent, AgentSettings
from orchestrator import Orchestrator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# Define some example tools
class CalculateParams(BaseModel):
    expression: str = Field(..., description="Math expression to calculate")


def calculate_tool(params: CalculateParams, context: dict):
    """Calculate the result of a math expression."""
    try:
        # Warning: eval is unsafe for production use, this is just for demonstration
        result = eval(params.expression)
        return result
    except Exception as e:
        return {"error": str(e)}


class WeatherParams(BaseModel):
    location: str = Field(..., description="City name or location")


def weather_tool(params: WeatherParams, context: dict):
    """Get the current weather for a location."""
    # This is a mock implementation
    locations = {
        "new york": {"temp": 72, "condition": "Sunny"},
        "london": {"temp": 62, "condition": "Rainy"},
        "tokyo": {"temp": 80, "condition": "Clear"},
        "sydney": {"temp": 70, "condition": "Partly Cloudy"},
    }

    location = params.location.lower()
    if location in locations:
        return locations[location]
    else:
        return {"temp": 65, "condition": "Unknown location, using default weather"}


class AnswerParams(BaseModel):
    final_answer: str = Field(
        ..., description="The final answer to the user's question"
    )
    reasoning: str = Field(
        ..., description="Brief explanation of how you arrived at this answer"
    )


def answer_tool(params: AnswerParams, context: dict) -> dict:
    """Provide a final answer to the user's question."""
    answer = {
        "final_answer": params.final_answer,
        "reasoning": params.reasoning or "No additional reasoning provided.",
    }

    logging.info(f"Answer tool called with final answer: {params.final_answer}")

    # If context has memory, store this as the final answer
    if context and "memory" in context:
        context["memory"].update_state("final_answer", answer)
        logging.info("Final answer stored in memory")
    else:
        logging.warning(
            "Could not store final answer - memory not available in context"
        )

    return answer


class DoneParams(BaseModel):
    success: bool = Field(
        True, description="Whether the task was completed successfully"
    )
    message: str = Field(
        "Task completed", description="Final message about task completion"
    )


def done_tool(params: DoneParams, context: dict):
    """Mark the task as done."""
    return {"final_message": params.message, "success": params.success}


async def main():
    # Set up the tool registry
    registry = ToolRegistry()

    # Register tools
    registry.register_tool(
        name="calculate",
        param_model=CalculateParams,
        function=calculate_tool,
        description="Calculate the result of a mathematical expression",
    )

    registry.register_tool(
        name="weather",
        param_model=WeatherParams,
        function=weather_tool,
        description="Get the current weather for a location",
    )

    registry.register_tool(
        name="answer",
        param_model=AnswerParams,
        function=answer_tool,
        description="REQUIRED: Provide your final answer to the user's question with reasoning",
    )

    registry.register_tool(
        name="done",
        param_model=DoneParams,
        function=done_tool,
        description="Mark the task as complete",
    )

    # Create LLM - uses ANTHROPIC_API_KEY from environment
    llm = ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    # Create orchestrator
    orchestrator = Orchestrator(llm=llm, tools_registry=registry)

    # Define steps callback for monitoring
    def on_step_end(agent: Agent):
        print(f"Completed step {agent.state.n_steps}")

    # Execute a task
    task = "What's the weather like in Tokyo? Then calculate 10 * 5 and tell me if that number is less than Tokyo's temperature."

    try:
        result = await orchestrator.execute_task(
            task=task,
            agent_settings=AgentSettings(max_steps=10),
            on_step_end=on_step_end,
        )

        print("\n=== Task Execution Summary ===")
        print(f"Task: {task}")
        print(f"Success: {result['success']}")
        print(f"Steps taken: {result['steps_taken']}")

        # Show the final answer if available
        agent = result.get("agent")

        if agent:
            # Get final answer from memory
            final_answer = agent.memory.get_state("final_answer")

            if final_answer:
                print("\n📝 FINAL ANSWER:")
                print(f"  {final_answer['final_answer']}")
                if final_answer.get("reasoning"):
                    print(f"\n  Reasoning: {final_answer['reasoning']}")
            else:
                print("\n⚠️ NO FINAL ANSWER FOUND")

            print("\nTool Calls:")
            for i, tool_call in enumerate(agent.memory.tool_call_history):
                print(
                    f"{i+1}. {tool_call.tool_name}({tool_call.parameters}) → {'✓' if tool_call.success else '✗'}"
                )

    except Exception as e:
        print(f"Error executing task: {e}")


if __name__ == "__main__":
    asyncio.run(main())
