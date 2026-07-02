"""Deterministic skill curator (SK-1, layer 3)."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pori.curator import record_skill_use, run_curation, should_run_now
from pori.skill_provenance import mark_agent_created

pytestmark = [pytest.mark.unit]

NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _make_skill(skills_dir: Path, slug: str) -> None:
    skill = skills_dir / slug
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {slug}\ndescription: x.\n---\nbody", encoding="utf-8"
    )


def _write_ledger(base: Path, payload: dict) -> None:
    base.mkdir(parents=True, exist_ok=True)
    (base / "skill_usage.json").write_text(json.dumps(payload), encoding="utf-8")


def test_inactive_agent_skill_is_archived_by_move(tmp_path):
    base, skills = tmp_path / ".pori", tmp_path / "skills"
    _make_skill(skills, "auto-skill")
    mark_agent_created("auto-skill@0.1.0", base=base)
    record_skill_use("auto-skill@0.1.0", base=base, now=NOW - timedelta(days=100))

    result = run_curation(skills, base=base, now=NOW)

    assert "auto-skill@0.1.0" in result.archived
    assert not (skills / "auto-skill").exists()  # moved, not left in place
    assert (skills / ".archive" / "auto-skill" / "SKILL.md").exists()  # recoverable


def test_moderately_inactive_skill_is_marked_stale_not_archived(tmp_path):
    base, skills = tmp_path / ".pori", tmp_path / "skills"
    _make_skill(skills, "midskill")
    mark_agent_created("midskill@0.1.0", base=base)
    record_skill_use("midskill@0.1.0", base=base, now=NOW - timedelta(days=45))

    result = run_curation(skills, base=base, now=NOW)

    assert "midskill@0.1.0" in result.marked_stale
    assert "midskill@0.1.0" not in result.archived
    assert (skills / "midskill").exists()  # still present


def test_grace_protects_recently_created_skills(tmp_path):
    base, skills = tmp_path / ".pori", tmp_path / "skills"
    _make_skill(skills, "fresh")
    _write_ledger(
        base,
        {
            "agent_created": ["fresh@0.1.0"],
            "usage": {
                "fresh@0.1.0": {
                    "first_seen": (NOW - timedelta(days=2)).isoformat(),  # within grace
                    "last_used": (NOW - timedelta(days=100)).isoformat(),
                    "state": "active",
                }
            },
        },
    )
    result = run_curation(skills, base=base, now=NOW)
    assert not result.changed
    assert (skills / "fresh").exists()


def test_user_authored_skills_are_never_touched(tmp_path):
    base, skills = tmp_path / ".pori", tmp_path / "skills"
    _make_skill(skills, "user-made")  # never marked agent-created

    result = run_curation(skills, base=base, now=NOW)

    assert result.scanned == 0  # out of the curation ceiling entirely
    assert (skills / "user-made").exists()


def test_use_reactivates_a_stale_skill(tmp_path):
    base = tmp_path / ".pori"
    _write_ledger(
        base,
        {"usage": {"s@1": {"first_seen": NOW.isoformat(), "state": "stale"}}},
    )
    record_skill_use("s@1", base=base, now=NOW)
    data = json.loads((base / "skill_usage.json").read_text(encoding="utf-8"))
    assert data["usage"]["s@1"]["state"] == "active"


def test_should_run_now_respects_the_interval(tmp_path):
    base, skills = tmp_path / ".pori", tmp_path / "skills"
    assert should_run_now(base=base, now=NOW) is True  # never run
    run_curation(skills, base=base, now=NOW)  # stamps last_curated
    assert should_run_now(base=base, now=NOW) is False
    assert should_run_now(base=base, now=NOW + timedelta(hours=25)) is True
