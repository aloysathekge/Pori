"""Offline Builder model evaluation harness.

Replays fixture Surface change requests through the real workspace loop and
the real build + browser quality gate against the bundled baseline, then
reports the numbers model qualification should rest on: gate pass rate,
turns and calls to finish, cached vs uncached tokens, estimated cost, and
wall time. No database, worker, or catalog is touched; a run costs only the
provider tokens it measures.

Usage (spends real tokens):

    uv run python -m aloy_backend.builder_eval \
        --provider fireworks --model accounts/fireworks/models/kimi-k2p6 \
        --cases revision-small

The harness mechanics are covered by tests with scripted models; CI never
spends.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from pori import (
    BudgetLedger,
    ExecutionBudget,
    LLMConfig,
    create_llm,
    ensure_budgeted_chat_model,
)

from .baseline_surface import baseline_surface_files
from .run_profiles import SURFACE_BUILDER_RUN_PROFILE
from .skills import surface_builder_instructions
from .surface_build_runner import LocalDevelopmentSurfaceBuildRunner
from .surface_builder_loop import SurfaceBuilderLoopError, run_surface_builder_loop
from .surface_manifest import parse_surface_manifest
from .surface_pipeline import (
    SurfaceCandidate,
    bind_surface_manifest_primary_jobs,
    surface_primary_job_contract_diagnostics,
)
from .surface_runtime import build_surface_runtime_document
from .surface_runtime_inspection import inspect_surface_runtime

MAX_GATE_ATTEMPTS = 3
MAX_TURNS_PER_ATTEMPT = 20

# Serverless list prices per million tokens; override per model on the CLI.
DEFAULT_PRICES = {"input": 0.95, "cached": 0.16, "output": 4.00}


@dataclass(frozen=True)
class EvalCase:
    name: str
    task: str
    required_primary_jobs: list[dict[str, str]]
    base_files: dict[str, str] = field(default_factory=baseline_surface_files)


def _job(job_id_hex: str, description: str) -> dict[str, str]:
    return {"id": f"job_{job_id_hex}", "description": description}


CASES: dict[str, Callable[[], EvalCase]] = {
    "revision-small": lambda: EvalCase(
        name="revision-small",
        task=(
            "Revise the Event's existing Surface: add a Notes view alongside "
            "the current views. It shows a heading named Notes, one textbox "
            "labeled 'New note', and an 'Add note' button that appends the "
            "note to a visible list using local component state. Copy the "
            "host-issued primary job ids and descriptions exactly into "
            "surface.json, replacing any previously declared jobs, and give "
            "each a browser-provable path."
        ),
        required_primary_jobs=[
            _job("ea11ca5e00000001", "See this Event's work at a glance"),
            _job("ea11ca5e00000002", "Capture a quick note about this Event"),
        ],
    ),
    "revision-jobtracker": lambda: EvalCase(
        name="revision-jobtracker",
        task=(
            "Revise the Event's existing Surface into a lightweight job "
            "application tracker: a pipeline board with stages Wishlist, "
            "Applied, Interview, Offer, and Rejected, a form to add a job "
            "with company and role, a select on each job card to move it "
            "between stages using local component state, and a top panel "
            "showing total applications. Copy the host-issued primary job "
            "ids and descriptions exactly into surface.json, replacing any "
            "previously declared jobs, and give each a browser-provable path."
        ),
        required_primary_jobs=[
            _job("ea11ca5e00000011", "View job applications grouped by stage"),
            _job("ea11ca5e00000012", "Add a new job application"),
            _job("ea11ca5e00000013", "See the total number of applications"),
        ],
    ),
}


def _smoke_context() -> dict[str, Any]:
    return {
        "protocol_version": "1",
        "sdk_version": "1",
        "event_id": "event-eval",
        "project_id": "project-eval",
        "build_id": "build-eval",
        "code_revision_id": "revision-eval",
        "data_revision": 0,
        "capabilities": ["tasks", "files", "ask_aloy"],
        "widgets": [],
        "data": {
            "event": {
                "title": "Builder Evaluation Event",
                "summary": "A fixture Event used to measure Builder models.",
            },
            "tasks": [],
            "files": [],
            "interactions": [],
        },
    }


def _initial_user_message(case: EvalCase) -> str:
    context = {
        "event": _smoke_context()["data"]["event"],
        "surface": {
            "draft": {
                "files": sorted(case.base_files),
                "source_access": (
                    "Full source lives in your workspace; use list_files and "
                    "read_file instead of expecting it here."
                ),
            }
        },
    }
    return (
        case.task
        + "\n\nTrusted Event and current Surface context:\n"
        + json.dumps(context, ensure_ascii=False, sort_keys=True)
        + "\n\nHost-owned acceptance jobs:\n"
        + json.dumps(case.required_primary_jobs, ensure_ascii=False)
    )


def _repair_message(case: EvalCase, diagnostics: list[dict[str, Any]]) -> str:
    return (
        case.task
        + "\n\nThe full host gate rejected the checked workspace. Repair every "
        "diagnostic below against the current workspace, then finish again.\n"
        + json.dumps(diagnostics[:40], ensure_ascii=False)
    )


async def _full_gate(
    candidate: SurfaceCandidate,
    *,
    required_primary_jobs: list[dict[str, str]],
    build_runner: Any,
) -> list[dict[str, Any]]:
    """The same authority order production uses: contract, compile, browser."""
    bound = bind_surface_manifest_primary_jobs(
        candidate, required_primary_jobs=required_primary_jobs
    )
    diagnostics = surface_primary_job_contract_diagnostics(
        bound, required_primary_jobs=required_primary_jobs
    )
    if diagnostics:
        return [{"stage": "validation", **item} for item in diagnostics]
    files = {item.source_path: item.content for item in bound.files}
    manifest = parse_surface_manifest(files)
    build = await build_runner.build(
        build_id=f"builder-eval-{uuid.uuid4().hex[:12]}",
        files=files,
        manifest=manifest.model_dump(mode="json", by_alias=True),
    )
    if build.status != "succeeded" or build.bundle is None:
        return list(build.diagnostics) or [
            {"stage": "build", "code": "build_failed", "message": "No bundle"}
        ]
    return inspect_surface_runtime(
        build_surface_runtime_document(build.bundle),
        _smoke_context(),
        manifest=manifest,
    )


async def run_case(
    case: EvalCase,
    *,
    llm_factory: Callable[[], Any],
    build_runner: Any | None = None,
    prices: dict[str, float] = DEFAULT_PRICES,
    max_tokens_budget: int = 900_000,
) -> dict[str, Any]:
    from .surface_development_workspace import LocalGitSurfaceWorkspace

    runner = build_runner or LocalDevelopmentSurfaceBuildRunner()
    ledger = BudgetLedger(
        ExecutionBudget(
            max_steps=MAX_GATE_ATTEMPTS * MAX_TURNS_PER_ATTEMPT + 10,
            max_tool_calls=400,
            max_tokens=max_tokens_budget,
            max_duration_seconds=3600.0,
        )
    )
    ledger.start_clock()
    raw_llm = llm_factory()
    raw_llm.session_affinity = f"builder-eval-{uuid.uuid4().hex[:12]}"
    llm = ensure_budgeted_chat_model(raw_llm, ledger)

    from pori import SystemMessage, UserMessage

    system = SystemMessage(
        content=(
            SURFACE_BUILDER_RUN_PROFILE.system_prompt
            + "\n\nApply this exact Aloy Builder skill:\n\n"
            + surface_builder_instructions()
        )
    )
    workspace = await LocalGitSurfaceWorkspace.create(
        workspace_id=f"eval-{case.name}",
        base_files=case.base_files,
        build_runner=runner,
    )
    started = time.perf_counter()
    status = "gate_failed"
    failure: list[dict[str, Any]] = []
    turns = 0
    tool_calls = 0
    gate_attempts = 0
    try:
        message = _initial_user_message(case)
        previous_fingerprint: str | None = None
        for attempt in range(1, MAX_GATE_ATTEMPTS + 1):
            gate_attempts = attempt
            try:
                result = await run_surface_builder_loop(
                    llm=llm,
                    workspace=workspace,
                    messages=[system, UserMessage(content=message)],
                    primary_jobs=[
                        item["description"] for item in case.required_primary_jobs
                    ],
                    capabilities={"tools"},
                    required_primary_jobs=case.required_primary_jobs,
                    budget_ledger=ledger,
                    max_turns=MAX_TURNS_PER_ATTEMPT,
                    provider_call_timeout_seconds=285.0,
                )
            except SurfaceBuilderLoopError as exc:
                status = "loop_error"
                transcript = getattr(exc, "transcript", [])
                counts: dict[str, dict[str, int]] = {}
                for entry in transcript:
                    bucket = counts.setdefault(
                        str(entry.get("action")), {"ok": 0, "error": 0}
                    )
                    bucket["ok" if entry.get("ok") else "error"] += 1
                failure = [
                    {
                        "stage": "loop",
                        "message": str(exc),
                        "action_counts": counts,
                        "transcript_tail": [
                            {
                                "turn": entry.get("turn"),
                                "action": entry.get("action"),
                                "ok": entry.get("ok"),
                                "error": (
                                    (str(entry.get("error"))[:160] or None)
                                    if entry.get("error")
                                    else None
                                ),
                            }
                            for entry in transcript[-25:]
                        ],
                    }
                ]
                break
            turns += result.turns
            tool_calls += result.tool_calls
            fingerprint = result.finished.receipt.source_fingerprint
            if fingerprint == previous_fingerprint:
                status = "no_progress"
                failure = failure or [
                    {"stage": "loop", "message": "Repair repeated identical source"}
                ]
                break
            previous_fingerprint = fingerprint
            candidate = SurfaceCandidate.model_validate(
                {
                    "summary": result.finished.candidate.summary,
                    "primary_jobs": [
                        item["description"] for item in case.required_primary_jobs
                    ],
                    "files": [
                        item.model_dump(mode="python")
                        for item in result.finished.candidate.files
                    ],
                }
            )
            diagnostics = await _full_gate(
                candidate,
                required_primary_jobs=case.required_primary_jobs,
                build_runner=runner,
            )
            if not diagnostics:
                status = "published"
                failure = []
                break
            failure = diagnostics
            message = _repair_message(case, diagnostics)
    except Exception as exc:  # budget exhaustion or infrastructure
        status = "error"
        failure = [{"stage": "harness", "message": f"{type(exc).__name__}: {exc}"}]
    finally:
        await workspace.close()

    usage = ledger.snapshot()
    cached = int(usage.get("cache_read_tokens_used") or 0)
    fresh_input = max(0, int(usage.get("input_tokens_used") or 0) - cached)
    output = int(usage.get("output_tokens_used") or 0)
    cost = (
        fresh_input * prices["input"]
        + cached * prices["cached"]
        + output * prices["output"]
    ) / 1_000_000
    return {
        "case": case.name,
        "status": status,
        "gate_attempts": gate_attempts,
        "turns": turns,
        "tool_calls": tool_calls,
        "llm_calls": int(usage.get("llm_calls_used") or 0),
        "input_tokens": fresh_input,
        "cached_tokens": cached,
        "output_tokens": output,
        "estimated_cost_usd": round(cost, 4),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "failure": failure[:10],
    }


def _print_report(model: str, results: list[dict[str, Any]]) -> None:
    print(f"\nBuilder eval — {model}")
    header = (
        f"{'case':<22} {'status':<12} {'gates':>5} {'turns':>5} "
        f"{'calls':>5} {'fresh':>8} {'cached':>8} {'out':>7} "
        f"{'$est':>7} {'sec':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['case']:<22} {r['status']:<12} {r['gate_attempts']:>5} "
            f"{r['turns']:>5} {r['llm_calls']:>5} {r['input_tokens']:>8} "
            f"{r['cached_tokens']:>8} {r['output_tokens']:>7} "
            f"{r['estimated_cost_usd']:>7.3f} {r['wall_seconds']:>6.1f}"
        )
        for item in r["failure"][:3]:
            print(f"    ! {item.get('stage')}: {str(item.get('message'))[:110]}")
            if item.get("action_counts"):
                print(f"      actions: {json.dumps(item['action_counts'])[:160]}")
    passed = sum(1 for r in results if r["status"] == "published")
    print(f"\npass rate: {passed}/{len(results)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Builder model eval")
    parser.add_argument("--provider", default="fireworks")
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    parser.add_argument("--input-price", type=float, default=DEFAULT_PRICES["input"])
    parser.add_argument("--cached-price", type=float, default=DEFAULT_PRICES["cached"])
    parser.add_argument("--output-price", type=float, default=DEFAULT_PRICES["output"])
    parser.add_argument("--out", default="builder-eval-results.json")
    args = parser.parse_args()

    prices = {
        "input": args.input_price,
        "cached": args.cached_price,
        "output": args.output_price,
    }

    def llm_factory() -> Any:
        return create_llm(
            LLMConfig(
                provider=args.provider,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_output_tokens,
            )
        )

    async def run_all() -> list[dict[str, Any]]:
        results = []
        for name in args.cases:
            if name not in CASES:
                raise SystemExit(f"Unknown case {name!r}; known: {sorted(CASES)}")
            results.append(
                await run_case(CASES[name](), llm_factory=llm_factory, prices=prices)
            )
        return results

    results = asyncio.run(run_all())
    _print_report(args.model, results)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"model": args.model, "results": results}, handle, indent=1)
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
