# Current State

## Active Task

Hermes Release A is implemented on `feat/hermes-release-a` and awaiting review.
It adds immutable `RunContext`, deterministic prompt/tool fingerprints, typed
tool execution receipts, trace/result propagation, memory-scope enforcement,
and parent/child identity propagation for teams and nested teams.

Verification: 186 full-suite tests passed; focused Release A tests passed;
mypy passed for 62 source files; Black/isort passed.

## Release A Boundaries

- Local orchestrators preserve their existing shared-memory behavior.
- Explicit Cloud contexts create exactly scoped memory and reject mismatches.
- Approvals and scanners remain defense in depth, not OS containment.
- Changes are uncommitted on `feat/hermes-release-a`.

## Previously Active Task

Codebase-review follow-ups. Full suite now 180 passed; black/isort clean; no
new mypy errors introduced (all remaining mypy errors are pre-existing).

Security (done earlier this session):
- Replaced unsafe `eval()` in `pori/tools/standard/math_tools.py` with an AST-based
  safe arithmetic evaluator (`evaluate_expression`), with an exponent guard.
- Hardened `pori/sandbox/path_resolution.py` against path traversal: added
  `_safe_join`, so `replace_virtual_path` now rejects `../` escapes and
  `replace_virtual_paths_in_command` leaves traversal tokens unrewritten.
- Added `tests/test_math_tools.py` and traversal tests in
  `tests/test_sandbox_integration.py`.

Quality / maintainability (this round):
- `pori/agent.py`: extracted `Agent._reject_action(...)` helper and replaced 5
  duplicated rejection blocks in `execute_actions` (behavior-preserving).
- `pori/agent.py`: replaced 6 silent `except Exception: pass` with `logger.debug`.
- `pori/memory.py`: added module logger; extracted `_SEMANTIC_WEIGHT`/
  `_LEXICAL_WEIGHT` + `AgentMemory._blend_scores` and replaced the 3 hardcoded
  `0.75*sem + 0.25*lex` sites; added debug logging on embedding fallbacks.
- Added `tests/test_config.py` (12 tests) covering config resolution order,
  validation failures, back-compat aliases, and the `create_llm` factory.

Not done (deferred, larger/riskier): full decomposition of the 486-line
`execute_actions` and the Agent god-object; LLM-provider and API-layer tests;
enabling mypy enforcement in CI (currently `|| true`).

## Stabilization (in progress — before adding Agno features)

Decision: stabilize the framework before building new primitives. Gates chosen:
**strict** (failing coverage floor, blocking mypy, Python 3.10). First test-backfill
target: **LLM providers**.

Done this round:
- **mypy: 0 errors** (was ~71). Fixed across config/agent/team/sandbox_tools/
  filesystem_tools/llm providers/memory/registry/orchestrator/main/api/context/
  math_tools. Mostly safe annotations + correct `cast`s + `BaseException` narrowing
  in `asyncio.gather` result handling + a None-guard in `Team._run_nested_team`.
- Added `types-PyYAML` to test extra; removed dead `ExecuteScriptParams`.
- **CI made strict**: dropped `|| true` on mypy; added Python 3.10 to matrix;
  added `fail_under = 65` coverage floor (no-regression baseline; ratchet up as
  tests grow). Verified locally: 66.16% ≥ 65, gate passes.
- Verified the `safe_path_operation` "destination-bypass" from the original review
  is a FALSE POSITIVE — both branches validate.
- 180 tests pass; black/isort clean.

Still TODO for stability:
- **S3 LLM provider tests** (chosen priority): mock anthropic/openai/google SDK
  clients; cover message conversion, tool-call parsing, error handling. Currently
  22–41%. Then ratchet `fail_under` upward.
- CLI/main.py (0%), hitl.py (48%) backfill.
- Resolve remaining ~11 `except: pass` outside already-hardened files.

## Reference study: Agno

Studied `references/agno` (Agno SDK, Apache-2.0) + its Pori-authored notes.
Findings: Pori already implements the Team modes (ROUTER/BROADCAST/DELEGATE);
the clearest capability gap is a **Workflow** primitive (deterministic step DAG),
for which a Pori design note already exists. Wrote
`.agent/reference-studies/workflow-adaptation.md` — maps Agno's Workflow design
onto Pori's *actual* APIs (task-at-construction, dict run-result, blueprint
workers via Orchestrator, Trace/MemoryStore reuse). No product code changed.
Next-ranked gaps if pursued: structured RunOutput + `print_response`, then
first-class Knowledge/RAG with pluggable vectordb backends.

Deeper Agno study (db/session/run, knowledge/vectordb/reasoning/context/learn,
os/tools/guardrails/tracing) produced a phased adoption roadmap:
`.agent/reference-studies/agno-adoption-roadmap.md`. Recommended order:
Phase 0 structured RunOutput + event stream (foundation, unblocks streaming/
pause-resume) → Phase 4 PII + prompt-injection guardrails (quick win) →
Phase 1 Workflow → Phase 2 session→run→message persistence → Phase 3 reasoning
(native extended thinking) → Phase 5 RAG (RRF + reranker) → Phase 6+ AgentOS/OTel
(deferred). Each phase: own branch, tests-first, CI green, progress updated.
No product code changed yet — awaiting go-ahead on Phase 0.

## Previously: Active Task

Enabled multi-agent team mode in `config.yaml` for CLI use.

## Decisions Made

- AGENTS.md is the canonical repo instruction file.
- `.agent/` holds Aloy repo-local rules, commands, skills, and progress memory.
- CLI team mode: `research-team` with `delegate` mode and three members (researcher, analyst, writer).

## Important Discoveries

- CLI only uses team mode when `config.yaml` has a non-empty `team.members` list.
- Stale SQLite session memory can cause single-agent CLI to answer wrong prior questions; `/new` clears transient context.

## Blockers

- None.

## Next Session Should Start With

Restart `uv run pori` to pick up team config. Run `/new` before unrelated tasks to avoid memory contamination.
