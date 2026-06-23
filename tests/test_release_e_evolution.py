import pytest

from pori import (
    EvolutionArtifactKind,
    EvolutionEvalCase,
    EvolutionEvalResult,
    EvolutionProposal,
    EvolutionProposalStatus,
    EvolutionRepository,
    FileEvolutionRepository,
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
