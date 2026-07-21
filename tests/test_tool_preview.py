"""Tests for activity-descriptor tool previews."""

import pytest

from pori.observability import build_tool_preview, build_tool_result_preview


@pytest.mark.parametrize(
    "tool,params,expected",
    [
        (
            "write_file",
            {"file_path": "notes/report.md", "content": "x"},
            "Writing notes/report.md",
        ),
        ("sandbox_write_file", {"path": "/ws/cv.docx"}, "Writing /ws/cv.docx"),
        ("read_file", {"file_path": "a.py"}, "Reading a.py"),
        ("create_directory", {"directory_path": "lessons"}, "Creating folder lessons"),
        (
            "search_files",
            {"pattern": "*.py", "content_search": "write_file"},
            "Searching for write_file",
        ),
        ("bash", {"command": "npm run build"}, "Running: npm run build"),
        ("web_search", {"query": "weather"}, "Searching the web: weather"),
        (
            "update_plan",
            {"todos": [{"content": "a"}, {"content": "b"}]},
            "Updating the plan (2 step(s))",
        ),
        ("answer", {"final_answer": "hi"}, "Writing the answer"),
        ("done", {}, "Finishing up"),
    ],
)
def test_known_tools_have_friendly_previews(tool, params, expected):
    assert build_tool_preview(tool, params) == expected


def test_unknown_tool_falls_back_to_name():
    assert build_tool_preview("mystery_tool", {"x": 1}) == "mystery_tool"


def test_unknown_tool_with_primary_arg():
    assert build_tool_preview("ask_user", {"question": "Your name?"}) == (
        "ask_user: Your name?"
    )


def test_no_raw_json_and_truncates_long_values():
    preview = build_tool_preview("bash", {"command": "x" * 500})
    assert "{" not in preview
    assert len(preview) <= 80
    assert preview.endswith("...")


def test_handles_missing_params():
    assert build_tool_preview("write_file", None) == "Writing a file"


@pytest.mark.parametrize(
    "tool,params,expected",
    [
        ("write_file", {"file_path": "notes/report.md"}, "Wrote notes/report.md"),
        ("read_file", {"path": "a.py"}, "Read a.py"),
        ("bash", {"command": "npm test"}, "Ran: npm test"),
        ("web_search", {"query": "jobs"}, "Searched the web: jobs"),
        ("answer", {}, "Wrote the answer"),
    ],
)
def test_completed_tools_have_safe_result_previews(tool, params, expected):
    assert build_tool_result_preview(tool, params) == expected
