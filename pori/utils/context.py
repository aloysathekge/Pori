"""
Context variables for the Pori framework.

This module contains context variables that can be used across different
parts of the application without creating circular imports.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional

# A context variable to hold the request ID.
# This allows the ID to be accessed by any part of the application
# during the lifecycle of a single request.
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

# Per-turn identity (GW-5). Set once per run and cleared in finally, so the
# current session/user/org is available concurrency-safely deep in the call tree
# (tools, guardrails, tracing) without threading it through every constructor.
session_id_var: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
org_id_var: ContextVar[Optional[str]] = ContextVar("org_id", default=None)


@dataclass(frozen=True)
class Identity:
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    org_id: Optional[str] = None


def current_identity() -> Identity:
    """Read the current per-turn identity from context (fields default to None)."""
    return Identity(
        session_id=session_id_var.get(),
        user_id=user_id_var.get(),
        org_id=org_id_var.get(),
    )


@contextmanager
def use_identity(
    *,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    org_id: Optional[str] = None,
) -> Iterator[Identity]:
    """Bind per-turn identity contextvars for the duration of the block.

    Tokens are reset in ``finally`` so nested or concurrent turns never leak
    identity into one another.
    """
    tokens = (
        session_id_var.set(session_id),
        user_id_var.set(user_id),
        org_id_var.set(org_id),
    )
    try:
        yield current_identity()
    finally:
        session_id_var.reset(tokens[0])
        user_id_var.reset(tokens[1])
        org_id_var.reset(tokens[2])
