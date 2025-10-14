"""
Demonstration of Pori's filesystem tools capabilities.

This example shows how the agent can interact with the filesystem using the new tools.
"""

import asyncio
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from pori import Agent, AgentSettings, Orchestrator
from pori.tools_builtin import register_all_tools
from pori.tools import tool_registry

# Load environment variables
load_dotenv()


async def main():
    """Demonstrate filesystem tools with various tasks."""

    # Set up the tool registry with filesystem tools
    registry = tool_registry()
    register_all_tools(registry)
    
    print(f"🔧 Registered {len(registry.tools)} tools including filesystem operations")
    print("Available filesystem tools:")
    fs_tools = [name for name in registry.tools.keys() if any(fs_op in name for fs_op in 
                ['read_file', 'write_file', 'list_directory', 'file_info', 'create_directory', 
                 'copy_file', 'move_file', 'delete_file', 'search_files'])]
    for tool in sorted(fs_tools):
        print(f"  - {tool}")
    print()

    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    # Create orchestrator
    orchestrator = Orchestrator(llm=llm, tools_registry=registry)

    # Demo tasks showcasing filesystem capabilities
    filesystem_tasks = [
        # Basic file operations
        "List all files in the current directory and show their sizes",
        
        # File creation and reading
        "Create a new text file called 'agent_test.txt' with the content 'Hello from Pori agent!' and then read it back to verify",
        
        # Directory operations
        "Create a directory called 'test_workspace' and then list its contents",
        
        # File search
        "Search for all Python files (*.py) in the current directory and show their basic information",
        
        # File analysis
        "Find the README.md file and provide detailed information about it including a preview of its contents",
    ]

    print("🚀 Starting Filesystem Tools Demonstration")
    print("=" * 60)

    for i, task in enumerate(filesystem_tasks, 1):
        print(f"\n📋 Task {i}: {task}")
        print("-" * 50)

        try:
            # Execute the task
            result = await orchestrator.execute_task(
                task=task,
                agent_settings=AgentSettings(max_steps=8),
            )

            if result["success"]:
                print("✅ Task completed successfully!")

                # Get the agent and final answer
                agent = result.get("agent")
                if agent:
                    final_answer = agent.memory.get_final_answer()
                    if final_answer:
                        print(f"\n📝 Result:")
                        print(f"   {final_answer['final_answer']}")
                        if final_answer.get("reasoning"):
                            print(f"\n💭 How it was done:")
                            print(f"   {final_answer['reasoning']}")

                    print(f"\n📊 Execution Summary:")
                    print(f"   Steps taken: {result['steps_taken']}")
                    print(f"   Tool calls: {len(agent.memory.tool_call_history)}")

                    # Show which filesystem tools were used
                    fs_tool_calls = [
                        tc for tc in agent.memory.tool_call_history 
                        if tc.tool_name in fs_tools
                    ]
                    if fs_tool_calls:
                        print(f"   Filesystem tools used: {[tc.tool_name for tc in fs_tool_calls]}")
            else:
                print("❌ Task failed or incomplete")

        except Exception as e:
            print(f"❌ Error: {e}")

        print("\n" + "=" * 60)
        
        # Small delay between tasks
        await asyncio.sleep(1)

    print("\n🎉 Filesystem tools demonstration completed!")
    print("\nThe agent can now:")
    print("✓ Read and write files safely")
    print("✓ Navigate and list directories") 
    print("✓ Search for files by name and content")
    print("✓ Get detailed file information")
    print("✓ Create, copy, move, and delete files/directories")
    print("✓ All with built-in security measures and path validation")


if __name__ == "__main__":
    asyncio.run(main())