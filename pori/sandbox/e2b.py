"""E2B remote sandbox provider — true isolation for agent commands.

Modeled on the smallest complete remote backend in the Hermes reference
(Daytona, see references/hermes-agent-deep-dives/execution-backends.md):
the provider implements only the small ``Sandbox``/``SandboxProvider``
surface; lifecycle is resume-or-create — sandbox ids persist in a JSON
ledger keyed by thread_id, so a restarted (or resumed) session reattaches
the SAME sandbox with its filesystem intact instead of starting over.

Gated capability: requires the ``e2b`` package (optional extra
``pori[sandbox-e2b]``) and ``E2B_API_KEY``. Construction fails with a clear
message when either is missing; nothing else in the kernel changes.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .base import Sandbox, SandboxProvider

logger = logging.getLogger(__name__)

#: Per-command timeout (matches the local sandbox's 300s).
COMMAND_TIMEOUT_SECONDS = 300
#: Sandbox time-to-live; refreshed on acquire. Override via PORI_E2B_TTL.
DEFAULT_TTL_SECONDS = 1800


def _default_ledger_path() -> Path:
    return Path(os.getenv("PORI_E2B_LEDGER", ".pori/e2b_sandboxes.json"))


class E2BSandbox(Sandbox):
    """One E2B cloud sandbox (a Firecracker microVM)."""

    def __init__(self, handle: Any):
        self._handle = handle

    @property
    def sandbox_id(self) -> str:
        return str(self._handle.sandbox_id)

    def execute_command(self, command: str, cwd: Optional[str] = None) -> str:
        try:
            result = self._handle.commands.run(
                command,
                timeout=COMMAND_TIMEOUT_SECONDS,
                cwd=cwd or None,
            )
        except Exception as exc:  # SDK raises on nonzero exit in some modes
            exit_code = getattr(exc, "exit_code", None)
            stdout = getattr(exc, "stdout", "") or ""
            stderr = getattr(exc, "stderr", "") or str(exc)
            if exit_code is not None:
                return f"stdout:\n{stdout}\nstderr:\n{stderr}\n exit_code={exit_code}"
            return f"Error: {exc}"
        out = getattr(result, "stdout", "") or ""
        err = getattr(result, "stderr", "") or ""
        exit_code = getattr(result, "exit_code", 0) or 0
        if exit_code != 0:
            return f"stdout:\n{out}\nstderr:\n{err}\n exit_code={exit_code}"
        return out + ("\n" + err if err else "")

    def read_file(self, path: str) -> str:
        return str(self._handle.files.read(path))

    def write_file(self, path: str, content: str, append: bool = False) -> None:
        if append:
            # The SDK write is whole-file; route appends through the shell
            # (Hermes insight: file ops as shell commands work on any backend).
            quoted = shlex.quote(content)
            self._handle.commands.run(
                f"printf '%s' {quoted} >> {shlex.quote(path)}",
                timeout=COMMAND_TIMEOUT_SECONDS,
            )
            return
        self._handle.files.write(path, content)

    def list_dir(self, path: str, max_depth: int = 2) -> list[str]:
        # find is universally present in E2B images; trailing / marks dirs.
        out = self.execute_command(
            f"find {shlex.quote(path)} -maxdepth {int(max_depth)} " "-mindepth 1 | sort"
        )
        entries: list[str] = []
        prefix = path.rstrip("/") + "/"
        for line in out.splitlines():
            line = line.strip()
            if not line or line.startswith(("Error:", "stdout:", "stderr:")):
                continue
            entries.append(line[len(prefix) :] if line.startswith(prefix) else line)
        return entries


class E2BSandboxProvider(SandboxProvider):
    """Resume-or-create E2B sandboxes, one per thread/session.

    The thread→sandbox mapping is persisted to a JSON ledger (atomic
    temp+rename writes), so a new process — including a worker resuming an
    interrupted run — reconnects to the sandbox the session was using.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        ledger_path: Optional[Path] = None,
        # Injection seams for tests: (ttl)->handle and (sandbox_id)->handle
        create_fn: Optional[Callable[[int], Any]] = None,
        connect_fn: Optional[Callable[[str], Any]] = None,
    ):
        self._api_key = api_key or os.getenv("E2B_API_KEY", "")
        if create_fn is None or connect_fn is None:
            if not self._api_key:
                raise ValueError(
                    "E2B sandbox backend requires E2B_API_KEY (get one at e2b.dev)"
                )
            try:
                from e2b import Sandbox as E2BSdkSandbox
            except ImportError as exc:
                raise ImportError(
                    "E2B sandbox backend requires the 'e2b' package: "
                    "pip install 'pori[sandbox-e2b]'"
                ) from exc
            create_fn = create_fn or (
                lambda ttl: E2BSdkSandbox.create(timeout=ttl, api_key=self._api_key)
            )
            connect_fn = connect_fn or (
                lambda sid: E2BSdkSandbox.connect(sid, api_key=self._api_key)
            )
        self._create_fn = create_fn
        self._connect_fn = connect_fn
        self._ttl = int(os.getenv("PORI_E2B_TTL", str(DEFAULT_TTL_SECONDS)))
        self._ledger_path = ledger_path or _default_ledger_path()
        self._sandboxes: Dict[str, E2BSandbox] = {}
        self._threads: Dict[str, str] = self._load_ledger()

    # -- ledger (file-state discipline: atomic write, reconcile on load) ----

    def _load_ledger(self) -> Dict[str, str]:
        try:
            data = json.loads(self._ledger_path.read_text(encoding="utf-8"))
            return {str(k): str(v) for k, v in dict(data).items()}
        except (OSError, ValueError):
            return {}

    def _save_ledger(self) -> None:
        try:
            self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(self._ledger_path.parent), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._threads, f, indent=2)
            os.replace(tmp, self._ledger_path)
        except OSError as exc:  # ledger is an optimization, never fatal
            logger.warning("Could not persist E2B sandbox ledger: %s", exc)

    # -- SandboxProvider ----------------------------------------------------

    def acquire(self, thread_id: Optional[str] = None) -> str:
        key = thread_id or "default"
        known_id = self._threads.get(key)
        if known_id and known_id in self._sandboxes:
            return known_id
        if known_id:
            try:
                handle = self._connect_fn(known_id)
                self._sandboxes[known_id] = E2BSandbox(handle)
                logger.info("Reconnected E2B sandbox %s (thread %s)", known_id, key)
                return known_id
            except Exception as exc:
                logger.info(
                    "E2B sandbox %s gone (%s); creating a fresh one", known_id, exc
                )
        handle = self._create_fn(self._ttl)
        sandbox = E2BSandbox(handle)
        self._sandboxes[sandbox.sandbox_id] = sandbox
        self._threads[key] = sandbox.sandbox_id
        self._save_ledger()
        logger.info("Created E2B sandbox %s (thread %s)", sandbox.sandbox_id, key)
        return sandbox.sandbox_id

    def get(self, sandbox_id: str) -> Optional[Sandbox]:
        return self._sandboxes.get(sandbox_id)

    def release(self, sandbox_id: str) -> None:
        sandbox = self._sandboxes.pop(sandbox_id, None)
        if sandbox is not None:
            try:
                sandbox._handle.kill()
            except Exception as exc:
                logger.warning("E2B sandbox %s release failed: %s", sandbox_id, exc)
        self._threads = {k: v for k, v in self._threads.items() if v != sandbox_id}
        self._save_ledger()


__all__ = ["E2BSandbox", "E2BSandboxProvider"]
