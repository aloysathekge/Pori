"""Sandbox: secrets blocklist + the E2B remote provider (resume-or-create).

The E2B tests use injected create/connect fns (fake handles) so they run
without the e2b package or an API key — the real SDK path is exercised only
when the backend is actually selected."""

import os

import pytest

from pori.sandbox import create_sandbox_provider
from pori.sandbox.e2b import E2BSandboxProvider
from pori.sandbox.env_safety import sanitized_subprocess_env


class TestEnvSafety:
    def test_strips_secret_shaped_names(self):
        env = {
            "ANTHROPIC_API_KEY": "sk-ant",
            "OPENAI_API_KEY": "sk-oai",
            "MY_SECRET": "x",
            "GITHUB_TOKEN": "ghp",
            "DB_PASSWORD": "pw",
            "SUPABASE_URL": "https://x",
            "DATABASE_URL": "postgres://x",
            "VIRTUAL_ENV": "/venv",
        }
        safe = sanitized_subprocess_env(env)
        assert safe == {}

    def test_keeps_ordinary_vars(self):
        env = {"PATH": "/usr/bin", "HOME": "/home/x", "LANG": "en_US.UTF-8"}
        assert sanitized_subprocess_env(env) == env

    def test_path_allowlisted_even_if_pattern_would_match(self):
        # AUTHORITY contains "AUTH" but is display plumbing, not a secret.
        env = {"XAUTHORITY": "/x", "PATH": "/bin"}
        safe = sanitized_subprocess_env(env)
        assert "PATH" in safe
        # XAUTHORITY matches AUTH -> stripped (conservative default is correct)
        assert "XAUTHORITY" not in safe

    def test_defaults_to_os_environ(self, monkeypatch):
        monkeypatch.setenv("ZZ_TEST_APIKEY", "leak")
        monkeypatch.setenv("ZZ_TEST_PLAIN", "keep")
        safe = sanitized_subprocess_env()
        assert "ZZ_TEST_APIKEY" not in safe
        assert safe.get("ZZ_TEST_PLAIN") == "keep"

    def test_local_sandbox_runs_without_secrets(self, monkeypatch):
        from pori.sandbox import LocalSandbox

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
        sandbox = LocalSandbox()
        # printenv on the sanitized env must not contain the key
        out = sandbox.execute_command("printenv || set")
        assert "sk-should-not-leak" not in out


class FakeCommands:
    def __init__(self, sandbox):
        self._s = sandbox

    def run(self, command, timeout=None, cwd=None):
        self._s.commands_run.append((command, cwd))

        class R:
            stdout = f"ran: {command}"
            stderr = ""
            exit_code = 0

        return R()


class FakeFiles:
    def __init__(self, sandbox):
        self._s = sandbox

    def read(self, path):
        return f"contents of {path}"

    def write(self, path, content):
        self._s.written[path] = content


class FakeHandle:
    def __init__(self, sandbox_id):
        self.sandbox_id = sandbox_id
        self.commands = FakeCommands(self)
        self.files = FakeFiles(self)
        self.commands_run = []
        self.written = {}
        self.killed = False

    def kill(self):
        self.killed = True


class FakeE2B:
    """Stand-in for the e2b SDK: tracks created + connectable sandboxes."""

    def __init__(self):
        self.created = []
        self.live = {}  # sandbox_id -> handle (reconnectable)
        self._counter = 0

    def create(self, ttl):
        self._counter += 1
        sid = f"sbx_{self._counter}"
        handle = FakeHandle(sid)
        self.created.append(sid)
        self.live[sid] = handle
        return handle

    def connect(self, sandbox_id):
        if sandbox_id not in self.live:
            raise RuntimeError(f"sandbox {sandbox_id} not found")
        return self.live[sandbox_id]


@pytest.fixture
def fake_e2b():
    return FakeE2B()


def _provider(fake_e2b, ledger_path):
    return E2BSandboxProvider(
        api_key="test",
        ledger_path=ledger_path,
        create_fn=fake_e2b.create,
        connect_fn=fake_e2b.connect,
    )


class TestE2BProvider:
    def test_acquire_creates_once_per_thread(self, fake_e2b, tmp_path):
        provider = _provider(fake_e2b, tmp_path / "ledger.json")
        first = provider.acquire("session-A")
        second = provider.acquire("session-A")
        assert first == second
        assert fake_e2b.created == [first]  # created exactly once

    def test_distinct_threads_get_distinct_sandboxes(self, fake_e2b, tmp_path):
        provider = _provider(fake_e2b, tmp_path / "ledger.json")
        a = provider.acquire("session-A")
        b = provider.acquire("session-B")
        assert a != b
        assert len(fake_e2b.created) == 2

    def test_reconnects_same_sandbox_after_restart(self, fake_e2b, tmp_path):
        ledger = tmp_path / "ledger.json"
        # First "process": create a sandbox for the session
        p1 = _provider(fake_e2b, ledger)
        sid = p1.acquire("resume-me")

        # Second "process": fresh provider, same ledger -> reconnect, no create
        p2 = _provider(fake_e2b, ledger)
        reconnected = p2.acquire("resume-me")
        assert reconnected == sid
        assert fake_e2b.created == [sid]  # still only one ever created

    def test_recreates_when_remote_sandbox_is_gone(self, fake_e2b, tmp_path):
        ledger = tmp_path / "ledger.json"
        p1 = _provider(fake_e2b, ledger)
        sid = p1.acquire("gone")
        # Remote sandbox expired between processes
        del fake_e2b.live[sid]

        p2 = _provider(fake_e2b, ledger)
        new_sid = p2.acquire("gone")
        assert new_sid != sid
        assert new_sid in fake_e2b.created

    def test_execute_and_file_ops(self, fake_e2b, tmp_path):
        provider = _provider(fake_e2b, tmp_path / "ledger.json")
        sid = provider.acquire("s")
        sandbox = provider.get(sid)
        assert sandbox is not None
        assert "ran: echo hi" in sandbox.execute_command("echo hi")
        assert sandbox.read_file("/a.txt") == "contents of /a.txt"
        sandbox.write_file("/b.txt", "data")
        assert sandbox._handle.written["/b.txt"] == "data"

    def test_release_kills_and_forgets(self, fake_e2b, tmp_path):
        ledger = tmp_path / "ledger.json"
        provider = _provider(fake_e2b, ledger)
        sid = provider.acquire("s")
        handle = fake_e2b.live[sid]
        provider.release(sid)
        assert handle.killed is True
        assert provider.get(sid) is None
        # A later provider from the same ledger won't reconnect the released id
        provider2 = _provider(fake_e2b, ledger)
        new_sid = provider2.acquire("s")
        assert new_sid != sid

    def test_missing_api_key_is_a_clear_error(self, monkeypatch):
        monkeypatch.delenv("E2B_API_KEY", raising=False)
        with pytest.raises(ValueError, match="E2B_API_KEY"):
            E2BSandboxProvider()


class TestFactory:
    def test_local_backend(self):
        from pori.sandbox import LocalSandboxProvider

        assert isinstance(create_sandbox_provider("local"), LocalSandboxProvider)

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError, match="Unknown sandbox backend"):
            create_sandbox_provider("nope")
