"""Per-turn identity contextvars (GW-5)."""

import pytest

from pori.utils.context import Identity, current_identity, use_identity

pytestmark = [pytest.mark.unit]


def test_use_identity_sets_and_clears():
    assert current_identity() == Identity()  # unset outside the block
    with use_identity(session_id="s1", user_id="u1", org_id="o1") as ident:
        assert ident.session_id == "s1"
        assert current_identity().user_id == "u1"
        assert current_identity().org_id == "o1"
    assert current_identity() == Identity()  # reset in finally


def test_use_identity_nesting_restores_outer():
    with use_identity(session_id="outer"):
        with use_identity(session_id="inner"):
            assert current_identity().session_id == "inner"
        assert current_identity().session_id == "outer"  # inner reset restores outer
