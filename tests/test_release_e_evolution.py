import pytest

from pori import (
    EvolutionArtifactKind,
    EvolutionEvalCase,
    EvolutionEvalResult,
    EvolutionProposal,
    EvolutionProposalStatus,
    EvolutionRepository,
    FileEvolutionRepository,
    run_local_evolution_evals,
)
from pori.main import _handle_evolution_command


def _proposal(
    *,
    target: str = "skills/brainstorming",
    version: str = "1",
    proposal_id: str = "evo_test",
) -> EvolutionProposal:
    return EvolutionProposal(
        proposal_id=proposal_id,
        artifact_kind=EvolutionArtifactKind.SKILL,
        target=target,
        title="Improve brainstorming skill",
        summary="Ask clarifying questions before implementation.",
        rationale="Repeated coding tasks need stronger design-before-build behavior.",
        current_version="0",
        proposed_version=version,
        proposed_content="Ask one clarifying question at a time before implementation.",
        eval_cases=(
            EvolutionEvalCase(
                name="asks-before-coding",
                input="Build a new sync workflow",
                expected="A clarifying question before implementation details",
                criteria="The response must ask a clarifying question first.",
            ),
        ),
    )


def test_evolution_proposals_require_eval_cases():
    with pytest.raises(ValueError, match="at least one eval case"):
        EvolutionProposal(
            artifact_kind=EvolutionArtifactKind.SKILL,
            target="skills/brainstorming",
            title="Unsafe proposal",
            summary="No evals",
            rationale="Should fail",
            proposed_version="1",
            proposed_content="Change behavior without checks.",
            eval_cases=(),
        )


def test_failed_evaluations_block_approval():
    repo = EvolutionRepository()
    proposal = repo.submit(_proposal())

    evaluated = repo.record_evaluations(
        proposal.proposal_id,
        (
            EvolutionEvalResult(
                case_name="asks-before-coding",
                passed=False,
                reason="The response started implementing immediately.",
            ),
        ),
    )

    assert evaluated.status == EvolutionProposalStatus.EVALUATED
    with pytest.raises(ValueError, match="all eval cases pass"):
        repo.approve(proposal.proposal_id, reviewer="human")


def test_review_summary_reports_eval_readiness():
    repo = EvolutionRepository()
    proposal = repo.submit(_proposal())

    initial_summary = proposal.review_summary()

    assert initial_summary.eval_case_count == 1
    assert initial_summary.eval_result_count == 0
    assert initial_summary.passed_eval_count == 0
    assert initial_summary.missing_eval_cases == ("asks-before-coding",)
    assert initial_summary.evaluations_passed is False

    evaluated = repo.record_evaluations(
        proposal.proposal_id,
        (
            EvolutionEvalResult(
                case_name="asks-before-coding",
                passed=True,
                score=1.0,
                reason="The response asked a clarifying question first.",
            ),
        ),
    )
    ready_summary = evaluated.review_summary()

    assert ready_summary.eval_result_count == 1
    assert ready_summary.passed_eval_count == 1
    assert ready_summary.missing_eval_cases == ()
    assert ready_summary.evaluations_passed is True


def test_local_evolution_eval_runner_checks_expected_text():
    proposal = _proposal()

    results = run_local_evolution_evals(proposal)

    assert len(results) == 1
    assert results[0].case_name == "asks-before-coding"
    assert results[0].passed is False
    assert results[0].score == 0.0
    assert results[0].evaluator == "local-eval-runner"
    assert "not found" in results[0].reason


def test_governed_activation_requires_eval_and_review():
    repo = EvolutionRepository()
    proposal = repo.submit(_proposal())

    with pytest.raises(ValueError, match="expected one of"):
        repo.activate(proposal.proposal_id, activated_by="human")

    repo.record_evaluations(
        proposal.proposal_id,
        (
            EvolutionEvalResult(
                case_name="asks-before-coding",
                passed=True,
                score=1.0,
                reason="The response asked a clarifying question first.",
            ),
        ),
    )
    approved = repo.approve(proposal.proposal_id, reviewer="human")
    activation = repo.activate(proposal.proposal_id, activated_by="human")

    assert approved.status == EvolutionProposalStatus.APPROVED
    assert activation.target == "skills/brainstorming"
    assert activation.version == "1"
    assert repo.active("skills/brainstorming") == activation
    assert repo.get(proposal.proposal_id).status == EvolutionProposalStatus.ACTIVATED


def test_rollback_restores_previous_active_version():
    repo = EvolutionRepository()
    first = repo.submit(_proposal(version="1", proposal_id="evo_first"))
    second = repo.submit(_proposal(version="2", proposal_id="evo_second"))

    for proposal in (first, second):
        repo.record_evaluations(
            proposal.proposal_id,
            (
                EvolutionEvalResult(
                    case_name="asks-before-coding",
                    passed=True,
                    reason="Passed",
                ),
            ),
        )
        repo.approve(proposal.proposal_id, reviewer="human")
        repo.activate(proposal.proposal_id, activated_by="human")

    restored = repo.rollback("skills/brainstorming", rolled_back_by="human")

    assert restored is not None
    assert restored.proposal_id == first.proposal_id
    assert restored.version == "1"
    assert repo.get(second.proposal_id).status == EvolutionProposalStatus.ROLLED_BACK


def test_file_evolution_repository_persists_state(tmp_path):
    path = tmp_path / "evolution.json"
    repo = FileEvolutionRepository(path)
    proposal = repo.submit(_proposal())
    repo.record_evaluations(
        proposal.proposal_id,
        (
            EvolutionEvalResult(
                case_name="asks-before-coding",
                passed=True,
                reason="Passed",
            ),
        ),
    )
    repo.approve(proposal.proposal_id, reviewer="human")
    repo.activate(proposal.proposal_id, activated_by="human")

    reloaded = FileEvolutionRepository(path)

    assert (
        reloaded.get(proposal.proposal_id).status == EvolutionProposalStatus.ACTIVATED
    )
    assert reloaded.active("skills/brainstorming").proposal_id == proposal.proposal_id


def test_evolution_cli_flow_from_json_files(tmp_path, capsys):
    repo = FileEvolutionRepository(tmp_path / "evolution.json")
    proposal_path = tmp_path / "proposal.json"
    results_path = tmp_path / "results.json"
    proposal_path.write_text(
        _proposal().model_dump_json(),
        encoding="utf-8",
    )
    results_path.write_text(
        (
            "["
            '{"case_name":"asks-before-coding","passed":true,'
            '"score":1.0,"reason":"Passed"}'
            "]"
        ),
        encoding="utf-8",
    )

    _handle_evolution_command(f"/evolution propose {proposal_path}", repo)
    _handle_evolution_command(f"/evolution eval evo_test {results_path}", repo)
    _handle_evolution_command("/evolution approve evo_test reviewer", repo)
    _handle_evolution_command("/evolution activate evo_test reviewer", repo)
    _handle_evolution_command("/evolution active skills/brainstorming", repo)

    output = capsys.readouterr().out
    assert "Submitted proposal evo_test" in output
    assert "Recorded 1 eval result" in output
    assert "Approved proposal evo_test" in output
    assert "Activated skills/brainstorming" in output
    assert "skills/brainstorming: 1 (evo_test)" in output


def test_evolution_cli_review_output_shows_missing_eval_state(tmp_path, capsys):
    repo = FileEvolutionRepository(tmp_path / "evolution.json")
    repo.submit(_proposal())

    _handle_evolution_command("/evolution list", repo)
    _handle_evolution_command("/evolution show evo_test", repo)

    output = capsys.readouterr().out
    assert "evo_test [proposed] skills/brainstorming -> 1 evals=0/1" in output
    assert "Passed evals: 0/1" in output
    assert "Missing evals: asks-before-coding" in output


def test_evolution_cli_local_eval_can_unlock_approval(tmp_path, capsys):
    repo = FileEvolutionRepository(tmp_path / "evolution.json")
    proposal = _proposal()
    proposal = proposal.model_copy(
        update={
            "eval_cases": (
                EvolutionEvalCase(
                    name="mentions-clarifying-question",
                    input="Build a new sync workflow",
                    expected="clarifying question",
                    criteria="The proposed content must preserve clarification.",
                ),
            )
        }
    )
    repo.submit(proposal)

    _handle_evolution_command("/evolution eval-local evo_test", repo)
    _handle_evolution_command("/evolution approve evo_test reviewer", repo)

    output = capsys.readouterr().out
    assert "Recorded 1 local eval result(s) for evo_test; passed 1/1" in output
    assert "Approved proposal evo_test" in output
    assert repo.get("evo_test").evaluations_passed is True
