"""edit_file tool: targeted replace + read-before-edit staleness (Hermes semantics)."""

import pytest

from pori.tools.standard.filesystem_tools import (
    EditFileParams,
    ReadFileParams,
    edit_file_tool,
    read_file_tool,
)

pytestmark = [pytest.mark.tools]


def _edit(path, old, new, **kw):
    return edit_file_tool(
        EditFileParams(file_path=str(path), old_string=old, new_string=new, **kw), {}
    )


def test_unique_replace_applies_once(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello world", encoding="utf-8")
    read_file_tool(ReadFileParams(file_path=str(f)), {})  # read first -> no warning
    res = _edit(f, "world", "there")
    assert res["success"] is True
    assert res["replacements"] == 1
    assert "warning" not in res
    assert f.read_text(encoding="utf-8") == "hello there"


def test_non_unique_requires_replace_all(tmp_path):
    f = tmp_path / "b.txt"
    f.write_text("x x x", encoding="utf-8")
    res = _edit(f, "x", "y")
    assert res["success"] is False and "not unique" in res["error"]
    res2 = _edit(f, "x", "y", replace_all=True)
    assert res2["success"] is True and res2["replacements"] == 3
    assert f.read_text(encoding="utf-8") == "y y y"


@pytest.mark.parametrize(
    "old,new",
    [("zzz", "q"), ("abc", "abc"), ("", "q")],  # not found / identical / empty
)
def test_edit_error_cases(tmp_path, old, new):
    f = tmp_path / "c.txt"
    f.write_text("abc", encoding="utf-8")
    assert _edit(f, old, new)["success"] is False


def test_edit_without_reading_warns_but_applies(tmp_path):
    f = tmp_path / "d.txt"
    f.write_text("foo bar", encoding="utf-8")  # created out-of-band, never read
    res = _edit(f, "bar", "baz")
    assert res["success"] is True
    assert "warning" in res  # read-before-edit nudge
    assert f.read_text(encoding="utf-8") == "foo baz"


def test_edit_refuses_protected_file(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("hitl:\n  enabled: true\n", encoding="utf-8")
    read_file_tool(ReadFileParams(file_path=str(f)), {})
    res = _edit(f, "true", "false")
    assert res["success"] is False and "protected" in res["error"].lower()
    assert "true" in f.read_text(encoding="utf-8")  # never modified
