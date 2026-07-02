"""Session-key lane primitive + repository lineage (GW-2)."""

import pytest

from pori.sessions import (
    SessionMessage,
    SessionRecord,
    SQLiteSessionRepository,
    build_session_key,
)

pytestmark = [pytest.mark.unit]


def test_build_session_key_is_stable_and_scoped():
    assert build_session_key("org1", "user1", "agentA") == "org1:user1:agentA"
    assert build_session_key("org1", "user1") == "org1:user1:default"
    # Different user -> different lane; the lane is independent of any instance.
    assert build_session_key("org1", "u1") != build_session_key("org1", "u2")


def test_branch_forks_a_new_session_id_under_the_lineage(tmp_path):
    repo = SQLiteSessionRepository(tmp_path / "s.db")
    parent = repo.create(SessionRecord(organization_id="o", user_id="u", agent_id="a"))
    msg = repo.add_message(
        SessionMessage(session_id=parent.id, role="user", content="hi")
    )

    child = repo.branch("o", "u", parent.id, through_message_id=msg.id, title="fork")

    assert child.id != parent.id  # new instance...
    assert child.parent_session_id == parent.id  # ...same lineage
    assert child.branched_from_message_id == msg.id
    export = repo.export("o", "u", child.id)
    assert [m.content for m in export.messages] == ["hi"]  # history copied into branch
