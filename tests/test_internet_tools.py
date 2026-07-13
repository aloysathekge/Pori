"""
Tests for internet search tools.
"""

import json
from unittest.mock import MagicMock, Mock, patch

from pori.capabilities import CapabilityPrerequisites
from pori.tools.standard.internet_tools import (
    WebSearchParams,
    _select_search_backend,
    web_search_tool,
)

# Isolate every test from the real environment's search keys / override.
_NO_KEYS = {
    "TAVILY_API_KEY": "",
    "SERPER_API_KEY": "",
    "SERPAPI_API_KEY": "",
    "WEB_SEARCH_BACKEND": "",
}


@patch("pori.tools.standard.internet_tools.TavilyClient")
@patch.dict("os.environ", {**_NO_KEYS, "TAVILY_API_KEY": "tvly-test-key"})
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


@patch.dict("os.environ", _NO_KEYS, clear=False)
def test_web_search_tool_no_backend_configured():
    result = web_search_tool(WebSearchParams(query="something"), context={})
    assert "error" in result
    # The message names both keys so an operator knows either enables it.
    assert "TAVILY_API_KEY" in result["error"]
    assert "SERPER_API_KEY" in result["error"]


@patch("pori.tools.standard.internet_tools.TavilyClient")
@patch.dict("os.environ", {**_NO_KEYS, "TAVILY_API_KEY": "tvly-test-key"})
def test_web_search_tool_request_error(mock_client_class):
    mock_client = Mock()
    mock_client.search.side_effect = Exception("API error")
    mock_client_class.return_value = mock_client

    params = WebSearchParams(query="something")
    result = web_search_tool(params, context={})

    assert "error" in result
    assert "failed" in result["error"].lower()


# ── Google (Serper) backend ──────────────────────────────────────────────────

_SERPER_RESPONSE = {
    "organic": [
        {"title": "Python", "link": "https://python.org", "snippet": "The language."},
        {"title": "Docs", "link": "https://docs.python.org", "snippet": "Reference."},
    ],
    "answerBox": {"answer": "Python is a programming language."},
}


@patch("pori.tools.standard.internet_tools.urllib.request.urlopen")
@patch.dict("os.environ", {**_NO_KEYS, "SERPER_API_KEY": "serper-test-key"})
def test_web_search_google_backend(mock_urlopen):
    resp = MagicMock()
    resp.read.return_value = json.dumps(_SERPER_RESPONSE).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = resp

    result = web_search_tool(WebSearchParams(query="python", max_results=2), context={})

    assert "error" not in result
    assert result["total_found"] == 2
    assert result["results"][0]["url"] == "https://python.org"
    assert result["results"][0]["source"] == "serper"
    assert result["answer"] == "Python is a programming language."
    # It really hit Serper's endpoint with the API key header.
    request = mock_urlopen.call_args[0][0]
    assert request.full_url == "https://google.serper.dev/search"
    assert request.headers.get("X-api-key") == "serper-test-key"


_SERPAPI_RESPONSE = {
    "organic_results": [
        {
            "title": "Tokyo",
            "link": "https://en.wikipedia.org/wiki/Tokyo",
            "snippet": "Capital of Japan.",
        },
    ],
    "answer_box": {"answer": "Tokyo"},
}


@patch("pori.tools.standard.internet_tools.urllib.request.urlopen")
@patch.dict("os.environ", {**_NO_KEYS, "SERPAPI_API_KEY": "serpapi-test-key"})
def test_web_search_serpapi_backend(mock_urlopen):
    resp = MagicMock()
    resp.read.return_value = json.dumps(_SERPAPI_RESPONSE).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = resp

    result = web_search_tool(WebSearchParams(query="capital of japan"), context={})

    assert "error" not in result
    assert result["results"][0]["source"] == "serpapi"
    assert result["results"][0]["url"] == "https://en.wikipedia.org/wiki/Tokyo"
    assert result["answer"] == "Tokyo"
    # Hit serpapi.com with the key as a query param (their API design).
    request = mock_urlopen.call_args[0][0]
    assert request.full_url.startswith("https://serpapi.com/search?")
    assert "api_key=serpapi-test-key" in request.full_url


@patch.dict(
    "os.environ",
    {**_NO_KEYS, "SERPER_API_KEY": "s", "SERPAPI_API_KEY": "sa", "TAVILY_API_KEY": "t"},
)
def test_backend_selection_priority_and_override():
    # All keys present → Serper wins by default.
    assert _select_search_backend() == "serper"
    # Explicit override selects a specific provider.
    with patch.dict("os.environ", {"WEB_SEARCH_BACKEND": "serpapi"}):
        assert _select_search_backend() == "serpapi"
    with patch.dict("os.environ", {"WEB_SEARCH_BACKEND": "tavily"}):
        assert _select_search_backend() == "tavily"


@patch.dict("os.environ", _NO_KEYS, clear=False)
def test_backend_selection_none_without_keys():
    assert _select_search_backend() is None


def test_internet_capability_satisfied_by_any_key():
    prereq = CapabilityPrerequisites(
        environment_any=("TAVILY_API_KEY", "SERPER_API_KEY", "SERPAPI_API_KEY")
    )
    assert prereq.missing(environ={"SERPER_API_KEY": "x"}) == ()
    assert prereq.missing(environ={"SERPAPI_API_KEY": "x"}) == ()
    assert prereq.missing(environ={"TAVILY_API_KEY": "x"}) == ()
    missing = prereq.missing(environ={})
    assert missing == ("environment_any:TAVILY_API_KEY|SERPER_API_KEY|SERPAPI_API_KEY",)
