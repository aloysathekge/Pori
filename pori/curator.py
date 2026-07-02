"""Deterministic skill curator (SK-1, layer 3).

The librarian half of the learning loop: skills the agent authored age from
active -> stale -> archived by inactivity, so an auto-growing library stays tidy.
Strictly deterministic (no LLM), scoped to agent-created skills only (SK-2), and
archive-only — an archived skill is MOVED to a recoverable ``.archive/`` dir,
never deleted. A user's hand-written skills are never touched.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import FrozenSet, List, Optional, Union

from .skill_provenance import agent_created_skills

STALE_AFTER_DAYS = 30
ARCHIVE_AFTER_DAYS = 90
GRACE_DAYS = 7  # never transition a skill younger than this
MIN_RUN_INTERVAL_HOURS = 24


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ledger_path(base: Optional[Path]) -> Path:
    return (base or Path(".pori")) / "skill_usage.json"


def _load(base: Optional[Path]) -> dict:
    try:
        return json.loads(_ledger_path(base).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save(base: Optional[Path], data: dict) -> None:
    path = _ledger_path(base)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        pass


def _parse(ts: object) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts) if isinstance(ts, str) else None
    except ValueError:
        return None


@dataclass
class CuratorResult:
    scanned: int = 0
    marked_stale: List[str] = field(default_factory=list)
    archived: List[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.marked_stale or self.archived)


def record_skill_use(
    skill_id: str, *, base: Optional[Path] = None, now: Optional[datetime] = None
) -> None:
    """Mark a skill used now, reactivating it to ``active`` if it had gone stale."""
    stamp = (now or _now()).isoformat()
    data = _load(base)
    usage = data.setdefault("usage", {})
    entry = usage.setdefault(skill_id, {"first_seen": stamp})
    entry["last_used"] = stamp
    entry["state"] = "active"
    _save(base, data)


def should_run_now(
    *,
    base: Optional[Path] = None,
    now: Optional[datetime] = None,
    min_interval_hours: int = MIN_RUN_INTERVAL_HOURS,
) -> bool:
    """Inactivity trigger: True if curation has not run within the interval."""
    now = now or _now()
    last = _parse(_load(base).get("last_curated"))
    if last is None:
        return True
    return (now - last).total_seconds() >= min_interval_hours * 3600


def run_curation(
    skills_dir: Union[str, Path],
    *,
    base: Optional[Path] = None,
    now: Optional[datetime] = None,
    pinned: FrozenSet[str] = frozenset(),
) -> CuratorResult:
    """Apply active -> stale -> archived transitions to agent-created skills."""
    now = now or _now()
    data = _load(base)
    usage = data.setdefault("usage", {})
    created = agent_created_skills(base=base)
    result = CuratorResult(scanned=len(created))
    skills_root = Path(skills_dir)

    for skill_id in sorted(created):
        entry = usage.setdefault(
            skill_id, {"first_seen": now.isoformat(), "state": "active"}
        )
        first_seen = _parse(entry.get("first_seen")) or now
        last_active = _parse(entry.get("last_used")) or first_seen
        state = entry.get("state", "active")
        slug = skill_id.split("@", 1)[0]

        if skill_id in pinned or slug in pinned:
            continue
        if (now - first_seen).days < GRACE_DAYS:
            continue  # too new to touch

        inactive_days = (now - last_active).days
        if inactive_days >= ARCHIVE_AFTER_DAYS and state != "archived":
            if _archive_skill(skills_root, slug):
                entry["state"] = "archived"
                result.archived.append(skill_id)
        elif inactive_days >= STALE_AFTER_DAYS and state == "active":
            entry["state"] = "stale"
            result.marked_stale.append(skill_id)

    data["last_curated"] = now.isoformat()
    _save(base, data)
    return result


def _archive_skill(skills_dir: Path, slug: str) -> bool:
    """Move a skill dir into ``{skills_dir}/.archive/{slug}`` (recoverable)."""
    source = skills_dir / slug
    if not source.is_dir():
        return False
    destination = skills_dir / ".archive" / slug
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)  # supersede a prior archive of the same slug
        shutil.move(str(source), str(destination))
        return True
    except OSError:
        return False
