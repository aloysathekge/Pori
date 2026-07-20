from aloy_backend.surface_run_gate import evaluate_surface_interaction_context


def test_surface_interaction_context_gate_requires_the_exact_interaction():
    rejected = evaluate_surface_interaction_context(
        required_interaction_id="sint-required",
        observed_interaction_ids={"sint-other"},
    )
    accepted = evaluate_surface_interaction_context(
        required_interaction_id="sint-required",
        observed_interaction_ids={"sint-other", "sint-required"},
    )

    assert rejected.accepted is False
    assert rejected.receipt()["required_interaction_id"] == "sint-required"
    assert rejected.receipt()["observed_interaction_ids"] == ["sint-other"]
    assert accepted.accepted is True
    assert accepted.errors == ()
