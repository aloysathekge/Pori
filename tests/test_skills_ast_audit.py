"""Opt-in AST audit for skill/plugin code (SK-7)."""

import pytest

from pori.skills_ast_audit import audit_path, audit_source

pytestmark = [pytest.mark.unit]


def test_flags_dynamic_and_dangerous_patterns():
    src = "import subprocess\nx = eval(user_input)\nv = getattr(obj, attr_name)\n"
    cats = {f.category for f in audit_source(src)}
    assert "import" in cats  # subprocess
    assert "call" in cats  # eval
    assert "dynamic-attr" in cats  # getattr(obj, <computed>)


def test_clean_code_has_no_findings():
    assert audit_source("def add(a, b):\n    return a + b\n") == []


def test_literal_getattr_is_not_flagged():
    # A constant attribute name is static and safe — not a review hint.
    assert audit_source("v = getattr(obj, 'name')\n") == []


def test_syntax_error_is_reported():
    findings = audit_source("def (:\n")
    assert findings and findings[0].category == "syntax"


def test_audit_path_scans_a_directory(tmp_path):
    (tmp_path / "a.py").write_text("import pickle\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("x = 1\n", encoding="utf-8")
    findings = audit_path(tmp_path)
    assert any("pickle" in f.message for f in findings)
