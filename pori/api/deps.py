from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import Request
from pori.llm import ChatAnthropic

from pori.orchestrator import Orchestrator
from pori.tools.registry import tool_registry
from pori.tools.standard import register_all_tools

load_dotenv()


def build_orchestrator() -> Orchestrator:
    """Builds the orchestrator with all its dependencies."""
    registry = tool_registry()
    register_all_tools(registry)

    model_name = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    llm = ChatAnthropic(
        model=model_name,
        temperature=0.0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    return Orchestrator(llm=llm, tools_registry=registry)


def get_orchestrator(request: Request) -> Orchestrator:
    """Dependency to retrieve the pre-configured orchestrator instance."""
    return request.app.state.orchestrator
