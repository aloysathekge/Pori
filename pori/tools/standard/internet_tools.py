"""
Internet search tools for retrieving public web information.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from ..registry import tool_registry

try:
    from tavily import TavilyClient  # type: ignore
except ImportError:
    TavilyClient = None  # type: ignore[misc, assignment]

Registry = tool_registry()
logger = logging.getLogger("pori.internet_tools")


class WebSearchParams(BaseModel):
    query: str = Field(..., description="Search query to run on the web")
    max_results: int = Field(
        5, ge=1, le=20, description="Maximum number of results to return (1-20)"
    )
    topic: str = Field(
        "general",
        description="Search topic: 'general', 'news', or 'finance'",
    )


@Registry.tool(
    name="web_search",
    description="Search the public web for information and return concise result snippets. Use for current events, news, facts, and general knowledge.",
)
def web_search_tool(params: WebSearchParams, context: Dict[str, Any]) -> Dict[str, Any]:
    """Search the web using Tavily API."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "error": "Tavily API not configured. Set TAVILY_API_KEY in your environment.",
        }

    if TavilyClient is None:
        return {
            "error": "Tavily API requires the tavily-python package. Install with: pip install tavily-python",
        }

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=params.query,
            max_results=params.max_results,
            search_depth="basic",
            topic=params.topic or "general",
        )
    except Exception as exc:
        logger.warning("Web search failed: %s", exc)
        return {"error": f"Web search failed: {exc}"}

    results_raw = response.get("results", [])
    results: List[Dict[str, str]] = []

    for r in results_raw[: params.max_results]:
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")
        if not url:
            continue
        results.append(
            {
                "title": title or "Untitled",
                "url": url,
                "snippet": content or "",
                "source": "tavily",
            }
        )

    return {
        "query": params.query,
        "results": results,
        "total_found": len(results),
        "answer": response.get("answer"),
    }


def register_internet_tools(registry=None):
    """Tools auto-register on import; kept for compatibility."""
    return None
