from aloy_backend.surface_evolution import (
    SurfaceEvolutionSignal,
    evaluate_surface_evolution,
)


def test_explicit_user_and_surface_requests_queue_builder_work():
    for trigger in ("explicit_user_request", "surface_source_change"):
        decision = evaluate_surface_evolution(
            SurfaceEvolutionSignal(
                trigger=trigger,
                goal="Add a grade calculator",
                base_revision_id="srev_live",
                base_build_id="sbuild_live",
                base_data_revision=7,
            )
        )

        assert decision.outcome == "queue"
        assert decision.reason_codes == ["explicit_source_change"]
        assert decision.receipt()["kind"] == "surface_evolution_decision"


def test_inferred_signals_propose_instead_of_silently_redesigning():
    decision = evaluate_surface_evolution(
        SurfaceEvolutionSignal(
            trigger="negative_feedback",
            goal="The current timetable view is not useful",
        )
    )

    assert decision.outcome == "propose"
    assert decision.reason_codes == ["negative_feedback"]


def test_single_job_failure_is_insufficient_but_repeated_failure_proposes():
    first = evaluate_surface_evolution(
        SurfaceEvolutionSignal(
            trigger="primary_job_failure",
            goal="Adding an application fails",
        )
    )
    repeated = evaluate_surface_evolution(
        SurfaceEvolutionSignal(
            trigger="primary_job_failure",
            goal="Adding an application fails",
            occurrence_count=2,
        )
    )

    assert first.outcome == "ignore"
    assert repeated.outcome == "propose"


def test_archived_event_or_active_builder_deduplicates_queueing():
    archived = evaluate_surface_evolution(
        SurfaceEvolutionSignal(
            trigger="explicit_user_request",
            goal="Add another view",
            event_archived=True,
        )
    )
    active = evaluate_surface_evolution(
        SurfaceEvolutionSignal(
            trigger="surface_source_change",
            goal="Add another view",
            active_builder=True,
        )
    )

    assert archived.outcome == "ignore"
    assert archived.reason_codes == ["event_archived"]
    assert active.outcome == "ignore"
    assert active.reason_codes == ["builder_already_active"]


def test_decision_fingerprint_binds_current_publication_and_data_revision():
    original = evaluate_surface_evolution(
        SurfaceEvolutionSignal(
            trigger="explicit_user_request",
            goal="Add a map",
            base_revision_id="srev_1",
            base_build_id="sbuild_1",
            base_data_revision=3,
        )
    )
    changed = evaluate_surface_evolution(
        SurfaceEvolutionSignal(
            trigger="explicit_user_request",
            goal="Add a map",
            base_revision_id="srev_1",
            base_build_id="sbuild_1",
            base_data_revision=4,
        )
    )

    assert original.fingerprint != changed.fingerprint
