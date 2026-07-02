"""Windows UTF-8 startup bootstrap (CLI-3)."""

import os
import sys

import pytest

import pori.bootstrap as bootstrap

pytestmark = [pytest.mark.unit]


def test_bootstrap_is_posix_noop_and_idempotent(monkeypatch):
    monkeypatch.setattr(bootstrap, "_bootstrap_applied", False)
    # Exercise the POSIX no-op path so we don't reconfigure the real console.
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("PYTHONUTF8", raising=False)

    bootstrap.apply_windows_utf8_bootstrap()
    assert "PYTHONUTF8" not in os.environ  # POSIX: no env mutation
    assert bootstrap._bootstrap_applied is True

    # Second call is a cheap no-op (does not raise).
    bootstrap.apply_windows_utf8_bootstrap()
