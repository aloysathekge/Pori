"""
Tests for internet search tools.
"""

from unittest.mock import Mock, patch

from pori.tools.standard.internet_tools import WebSearchParams, web_search_tool


@patch("pori.tools.standard.internet_tools.TavilyClient")
@patch.dict("os.environ", {"TAVILY_API_KEY": "tvly-test-key"})
def test_web_search_tool_success(mock_client_class):
    mock_client = Mock()
    mock_client.search.return_value = {
        "query": "python language",
        "results": [
            {
                "title": "Python",
                "url": "https://example.com/python",
                "content": "Python is a programming language.",
            },
            {
                "title": "Python syntax",
                "url": "https://example.com/syntax",
                "content": "Python syntax basics.",
            },
        ],
        "answer": "Python is a programming language.",
    }
    mock_client_class.return_value = mock_client

    params = WebSearchParams(query="python language", max_results=2)
    result = web_search_tool(params, context={})

    assert "error" not in result
    assert result["query"] == "python language"
    assert result["total_found"] == 2
    assert len(result["results"]) == 2
    assert result["results"][0]["title"] == "Python"
    assert result["results"][0]["snippet"] == "Python is a programming language."
    assert result["answer"] == "Python is a programming language."


@patch("pori.tools.standard.internet_tools.os.getenv")
def test_web_search_tool_missing_api_key(mock_getenv):
    mock_getenv.return_value = None

    params = WebSearchParams(query="something")
    result = web_search_tool(params, context={})

    assert "error" in result
    assert "TAVILY_API_KEY" in result["error"]


@patch("pori.tools.standard.internet_tools.TavilyClient")
@patch.dict("os.environ", {"TAVILY_API_KEY": "tvly-test-key"})
def test_web_search_tool_request_error(mock_client_class):
    mock_client = Mock()
    mock_client.search.side_effect = Exception("API error")
    mock_client_class.return_value = mock_client

    params = WebSearchParams(query="something")
    result = web_search_tool(params, context={})

    assert "error" in result
    assert "failed" in result["error"].lower()
