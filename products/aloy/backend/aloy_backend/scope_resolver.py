"""Layered-inheritance knowledge scope resolver — the Aloy moat.

Knowledge lives at three scope levels, least- to most-specific:

    ORG  (shared org-wide)  <  TEAM  (a team)  <  PERSONAL  (the user's own)

When the *same* fact exists at more than one level — identified by a shared
``conflict_key`` — the **most-specific level wins** (personal over team over org),
mirroring "local overrides global". Records with no ``conflict_key`` don't collide
and are all kept. On a same-level tie the first record seen wins (callers pass the
newest first).

This is a pure function over records (duck-typed on ``scope_level`` +
``conflict_key``); the DB query and prompt injection live in
``conversation_runtime``. For the first milestone only the personal layer is
populated — org/team rows slot in with no change here.
"""

from __future__ import annotations

from typing import Any, Iterable, List

ORG = "org"
TEAM = "team"
PERSONAL = "personal"

SCOPE_LEVELS = (ORG, TEAM, PERSONAL)

# higher = more specific = wins on a conflict_key collision
_SPECIFICITY = {ORG: 0, TEAM: 1, PERSONAL: 2}


def specificity(level: str) -> int:
    """How specific a scope level is (higher wins). Unknown -> personal."""
    return _SPECIFICITY.get(level, _SPECIFICITY[PERSONAL])


def resolve_layered(
    records: Iterable[Any], *, event_id: str | None = None
) -> List[Any]:
    """Merge org/team/personal records into the effective set for a user.

    On a shared ``conflict_key`` the most-specific ``scope_level`` wins;
    ``conflict_key``-less records all pass through. Original input order is
    preserved among the survivors.
    """
    # key -> ((Event-over-global, ownership specificity), order, record)
    winners: dict[str, tuple[tuple[int, int], int, Any]] = {}
    passthrough: List[tuple[int, Any]] = []

    for order, record in enumerate(records):
        key = (getattr(record, "conflict_key", None) or "").strip() or None
        if key is None:
            passthrough.append((order, record))
            continue
        record_event_id = getattr(record, "event_id", None)
        situation = 1 if event_id is not None and record_event_id == event_id else 0
        spec = (
            situation,
            specificity(getattr(record, "scope_level", PERSONAL) or PERSONAL),
        )
        current = winners.get(key)
        if current is None or spec > current[0]:
            winners[key] = (spec, order, record)

    combined = passthrough + [(order, rec) for _spec, order, rec in winners.values()]
    combined.sort(key=lambda item: item[0])
    return [rec for _order, rec in combined]
