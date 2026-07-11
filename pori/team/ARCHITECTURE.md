# pori/team — multi-agent coordination

## What this package owns
`Team`: a coordinator LLM that distributes one task across member
`Agent`s and synthesizes a result. This is peer-level *coordination* —
distinct from the `delegate_task` sub-agent tool (`pori/subagents.py`),
where a running agent spawns its own children.

## Files
- `core.py` — the `Team` class and its single entrypoint `run()`.
  Builds member agents from `MemberConfig`s, runs the mode-specific
  coordination (coordinator LLM calls use structured output into the
  models below), and returns the combined result with trace/metrics.
- `models.py` — the Pydantic shapes: `TeamMode`, `MemberConfig`,
  `TeamConfig`, plus the coordinator's structured outputs
  (`RoutingDecision`, `DelegationPlan`/`DelegationStep`,
  `BroadcastSummary`) and `MemberRunResult`.
- `__init__.py` — re-exports; import from `pori.team` (or `pori`), not
  the submodules.

## The three modes (`TeamMode`)
- `ROUTER` — coordinator picks exactly one member (`RoutingDecision`)
  and that member runs the task.
- `BROADCAST` — all members run in parallel (`asyncio`), coordinator
  synthesizes a `BroadcastSummary` from their answers.
- `DELEGATE` — coordinator emits a `DelegationPlan`: a DAG of
  `DelegationStep`s with member assignments; the executor runs steps
  respecting `depends_on`, feeding dependency outputs into later steps.

## Key contracts
- Members do **not** share memory by default; they communicate only
  through the coordinator's inputs/outputs. Shared state must be passed
  explicitly.
- Each member is a full `Agent` with its own settings/tools (from
  `MemberConfig`, which can carry its own `LLMConfig`), so budgets,
  HITL, and cancellation (`RunContext`, `BudgetLedger`,
  `CancellationToken`) thread through from the team level.
- Coordinator decisions are structured-output calls — changing a model
  in `models.py` changes the coordinator's schema and prompt contract.

## Change X → look at Y
- New mode or coordination behavior → `core.py` (`Team.run` dispatch)
  plus a structured-output model in `models.py`.
- Member selection quality → the routing/delegation prompts in
  `core.py`.
- Parallelism/budget issues in BROADCAST/DELEGATE → the asyncio
  gathering in `core.py` and `pori/runtime.py` primitives.
- Agent-spawns-agent behavior → not here; see `pori/subagents.py` and
  `Orchestrator.run_subagent`.
