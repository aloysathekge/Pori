"""Provenance-gated autonomy for skills (SK-2).

Distinguishes user-authored skills from agent-grown ones so an autonomous curator
(SK-1) can only ever touch what the agent itself created — never a user's
hand-written skill. A ContextVar marks the active write-origin during a run, and a
small JSON ledger (``.pori/skill_usage.json``) records which skills were written
under an agent origin. The destructive ceiling for curation is *archive*, and it
is scoped strictly to agent-created rows.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator, List, Optional, Set

# Origins that count as "the agent wrote this autonomously".
_AGENT_ORIGINS = frozenset({"background_review", "autonomous", "curator"})

_write_origin: ContextVar[Optional[str]] = ContextVar(
    "skill_write_origin", default=None
)


def current_write_origin() -> Optional[str]:
    """The write-origin bound for the current run, or None (a user edit)."""
    return _write_origin.get()


@contextmanager
def use_write_origin(origin: Optional[str]) -> Iterator[None]:
    """Bind the active write-origin for a block (e.g. a background-review fork)."""
    token = _write_origin.set(origin)
    try:
        yield
    finally:
        _write_origin.reset(token)


def is_agent_origin(origin: Optional[str] = None) -> bool:
    """True if the given (or current) write-origin is an autonomous agent origin."""
    resolved = origin if origin is not None else _write_origin.get()
    return resolved in _AGENT_ORIGINS


def _ledger_path(base: Optional[Path] = None) -> Path:
    return (base or Path(".pori")) / "skill_usage.json"


def _load(base: Optional[Path] = None) -> dict:
    try:
        return json.loads(_ledger_path(base).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def mark_agent_created(skill_id: str, *, base: Optional[Path] = None) -> None:
    """Record that ``skill_id`` was written under an autonomous agent origin."""
    path = _ledger_path(base)
    data = _load(base)
    created: List[str] = data.setdefault("agent_created", [])
    if skill_id not in created:
        created.append(skill_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        pass


def is_agent_created(skill_id: str, *, base: Optional[Path] = None) -> bool:
    """True only for skills the agent authored — the curation ceiling for SK-1."""
    return skill_id in set(_load(base).get("agent_created", []))


def agent_created_skills(*, base: Optional[Path] = None) -> Set[str]:
    """The full set of agent-created skill ids (what a curator may touch)."""
    return set(_load(base).get("agent_created", []))
