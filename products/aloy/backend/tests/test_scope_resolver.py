"""Layered knowledge scope resolver — the Aloy moat (org -> team -> personal)."""

from dataclasses import dataclass

from pori_cloud.scope_resolver import ORG, PERSONAL, TEAM, resolve_layered, specificity


@dataclass
class Rec:
    id: str
    scope_level: str
    conflict_key: str | None = None


def ids(records):
    return [r.id for r in records]


def test_specificity_order():
    assert specificity(PERSONAL) > specificity(TEAM) > specificity(ORG)


def test_personal_wins_conflict():
    out = resolve_layered(
        [Rec("p", PERSONAL, "k"), Rec("t", TEAM, "k"), Rec("o", ORG, "k")]
    )
    assert ids(out) == ["p"]


def test_team_wins_over_org():
    assert ids(resolve_layered([Rec("o", ORG, "k"), Rec("t", TEAM, "k")])) == ["t"]


def test_records_without_conflict_key_all_kept():
    out = resolve_layered([Rec("a", ORG), Rec("b", PERSONAL), Rec("c", TEAM)])
    assert set(ids(out)) == {"a", "b", "c"}


def test_org_only_key_kept():
    assert ids(resolve_layered([Rec("o", ORG, "k")])) == ["o"]


def test_input_order_preserved_among_survivors():
    out = resolve_layered(
        [
            Rec("x1", ORG),
            Rec("x2", PERSONAL, "k"),
            Rec("x3", ORG, "k"),  # loses to x2 on key k
            Rec("x4", ORG),
        ]
    )
    assert ids(out) == ["x1", "x2", "x4"]


def test_same_level_tie_keeps_first_seen():
    # callers pass newest first, so first-seen (newest) wins a same-level tie
    out = resolve_layered([Rec("new", PERSONAL, "k"), Rec("old", PERSONAL, "k")])
    assert ids(out) == ["new"]
