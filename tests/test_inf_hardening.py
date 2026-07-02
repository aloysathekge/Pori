"""INF hardening cluster: ${VAR} config expansion (INF-3), symlink-safe sandbox
paths (INF-4), and the sensitive-config write gate (INF-6)."""

import pytest

from pori.config import _expand_env_vars
from pori.sandbox.path_resolution import _safe_join
from pori.tools.standard.filesystem_tools import (
    WriteFileParams,
    is_sensitive_write_target,
    write_file_tool,
)

pytestmark = [pytest.mark.unit]


# --- INF-3: ${VAR} config expansion ----------------------------------------
def test_expand_env_vars_substitutes_and_recurses(monkeypatch):
    monkeypatch.setenv("PORI_TEST_SECRET", "s3cr3t")
    data = {
        "key": "${PORI_TEST_SECRET}",
        "nested": {"list": ["${PORI_TEST_SECRET}", "plain"]},
    }
    out = _expand_env_vars(data)
    assert out["key"] == "s3cr3t"
    assert out["nested"]["list"] == ["s3cr3t", "plain"]


def test_expand_env_vars_leaves_unset_verbatim(monkeypatch):
    monkeypatch.delenv("PORI_DEFINITELY_UNSET", raising=False)
    assert _expand_env_vars("${PORI_DEFINITELY_UNSET}") == "${PORI_DEFINITELY_UNSET}"


# --- INF-4: symlink-safe sandbox path join ---------------------------------
def test_safe_join_blocks_dotdot_traversal(tmp_path):
    with pytest.raises(ValueError):
        _safe_join(str(tmp_path), "../../etc/passwd")


def test_safe_join_allows_within(tmp_path):
    assert _safe_join(str(tmp_path), "sub/file.txt").endswith("file.txt")


def test_safe_join_blocks_symlink_escape(tmp_path):
    base = tmp_path / "sandbox"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (base / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted (e.g. Windows without privilege)")
    with pytest.raises(ValueError):
        _safe_join(str(base), "link/secret.txt")


# --- INF-6: sensitive-config write gate ------------------------------------
@pytest.mark.parametrize(
    "p",
    ["config.yaml", "config.yml", ".env", ".env.local", ".pori/memory.db", "a/.pori/x"],
)
def test_sensitive_write_target_true(p):
    assert is_sensitive_write_target(p) is True


@pytest.mark.parametrize("p", ["notes.txt", "src/main.py", "config.txt", "envfile"])
def test_sensitive_write_target_false(p):
    assert is_sensitive_write_target(p) is False


def test_write_file_tool_refuses_config_yaml(tmp_path):
    target = tmp_path / "config.yaml"
    res = write_file_tool(
        WriteFileParams(file_path=str(target), content="hitl:\n  enabled: false\n"), {}
    )
    assert res["success"] is False
    assert "protected" in res["error"].lower()
    assert not target.exists()  # never written
