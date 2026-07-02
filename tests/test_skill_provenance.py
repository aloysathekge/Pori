"""Provenance-gated autonomy for skills (SK-2)."""

import pytest

from pori.skill_provenance import (
    agent_created_skills,
    current_write_origin,
    is_agent_created,
    is_agent_origin,
    mark_agent_created,
    use_write_origin,
)

pytestmark = [pytest.mark.unit]


def test_write_origin_contextvar_binds_and_clears():
    assert current_write_origin() is None
    with use_write_origin("background_review"):
        assert current_write_origin() == "background_review"
        assert is_agent_origin() is True
    assert current_write_origin() is None
    assert is_agent_origin("user") is False  # a user edit is never an agent origin


def test_agent_created_ledger_round_trip(tmp_path):
    assert is_agent_created("skill-a", base=tmp_path) is False  # user skill by default
    mark_agent_created("skill-a", base=tmp_path)
    mark_agent_created("skill-a", base=tmp_path)  # idempotent
    mark_agent_created("skill-b", base=tmp_path)
    assert is_agent_created("skill-a", base=tmp_path) is True
    assert agent_created_skills(base=tmp_path) == {"skill-a", "skill-b"}
    # A hand-written skill stays outside the curation ceiling.
    assert is_agent_created("user-skill", base=tmp_path) is False
