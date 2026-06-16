# Reference Study: Workflow Abstraction for Pori

Source studied: `references/agno` (Agno SDK) and the Pori-authored notes
`references/agno/workflow_implementation.md` and `references/agno/run_agent_workflow_team.md`.

Status: **proposal / design** — no Pori product code has been changed.
License note: Agno is **Apache-2.0** (`references/agno/LICENSE`). Per the reference
rules we extract *principles and interfaces*, not code. Nothing here is copied from
Agno source; the skeletons below are written against Pori's own APIs.

---

## 1. Why

Pori has two execution primitives:

- **`Agent`** (`pori/agent.py`) — single Plan→Act→Reflect→Evaluate loop.
- **`Team`** (`pori/team/core.py`) — LLM-coordinated multi-agent (ROUTER / BROADCAST / DELEGATE).

Both put orchestration decisions in the hands of an LLM. Agno adds a third primitive —
**`Workflow`** — for **explicit, programmatic control flow**: a DAG of steps you define in
code (`Step → Step`, branch, loop, parallel), where Agents/Teams are the workers *inside*
steps but the topology is fixed and testable.

This is the single clearest capability gap between Pori and Agno (Agno's `workflow`
package is ~22k LOC; Pori has no equivalent). It is also the gap with an existing Pori
design note. Team and Workflow are complementary, not competing:

| | Orchestration decided by | Determinism | Best for |
|---|---|---|---|
| **Team** | LLM coordinator | Low | open-ended delegation, "agents talking" |
| **Workflow** | your code | High | fixed pipelines: Research→Draft→Review→Finalize; `if score<t: revise` |

---

## 2. The constraint the generic note misses

`references/agno/workflow_implementation.md` assumes the worker API is
`agent.run(input_text, session_id) -> str`. **Pori's API is different**, and the design
must respect it:

- **Task is supplied at construction, not at `run()`.**
  `Agent(task=..., llm=..., tools_registry=..., settings=..., memory=..., hitl_handler=...)`
  then `await agent.run(on_step_start=?, on_step_end=?)`. Same for `Team(task=..., coordinator_llm=..., members=[MemberConfig], mode=...)` → `await team.run()`.
- **Workers return a dict, not a string.** `Agent.run()` / `Team.run()` return
  `{"task", "completed", "steps_taken", "trace", "metrics", ...}`; the final answer is
  surfaced via `agent.result_summary()["final_answer"]`.
- **Workers are created fresh per execution.** `Team` already documents this: "Members
  are *blueprints* (`MemberConfig`), not live agents. A fresh `Agent` is created for every
  execution, matching `Orchestrator.execute_task()`." A Workflow step should follow the
  same blueprint pattern rather than holding a long-lived Agent.
- **`Orchestrator.execute_task(task, ...)`** already wraps "build a fresh Agent for a task
  string, run it, return a normalized dict." A Workflow `AgentStep` should delegate to the
  Orchestrator instead of re-implementing Agent construction.

Implication: in Pori, a step's worker is best modeled as **(blueprint + a function that
derives the task string from the workflow context)**. The step constructs the worker,
runs it, and writes the result back into the context.

---

## 3. Proposed design (mapped to Pori)

Target location: `pori/workflow/` (new package), exported from `pori/__init__.py`
alongside `Agent`, `Team`, `Orchestrator`.

### 3.1 Shared context

```python
# pori/workflow/context.py
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class WorkflowContext:
    """Single source of truth threaded through every step."""
    data: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None  # maps to Pori task_id / memory session_id

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)
```

### 3.2 Step protocol — async-first (Pori convention)

Pori's public agent/team APIs are async-only, so the Step protocol is async-first
(no sync `run`, unlike the generic note which is sync-first):

```python
# pori/workflow/steps.py
from typing import Protocol
from .context import WorkflowContext

class WorkflowStep(Protocol):
    name: str
    async def arun(self, ctx: WorkflowContext) -> WorkflowContext: ...
```

### 3.3 Worker steps — delegate to Orchestrator / Team

```python
@dataclass
class AgentStep:
    """Run a fresh single Agent on a task derived from context."""
    name: str
    orchestrator: Orchestrator        # reuse execute_task — don't rebuild Agent here
    input_key: str = "input"
    output_key: str = "output"
    agent_settings: Optional[AgentSettings] = None

    async def arun(self, ctx: WorkflowContext) -> WorkflowContext:
        task = ctx.get(self.input_key, "")
        result = await self.orchestrator.execute_task(task, self.agent_settings)
        ctx.set(self.output_key, result["final_answer"])
        ctx.set(f"{self.output_key}__meta", {
            "steps_taken": result["steps_taken"],
            "success": result["success"],
            "trace": result["trace"],
            "metrics": result["metrics"],
        })
        return ctx

@dataclass
class TeamStep:
    """Run a Team (blueprint members) on a task derived from context."""
    name: str
    coordinator_llm: BaseChatModel
    members: List[MemberConfig]
    mode: TeamMode = TeamMode.DELEGATE
    input_key: str = "input"
    output_key: str = "output"

    async def arun(self, ctx: WorkflowContext) -> WorkflowContext:
        task = ctx.get(self.input_key, "")
        team = Team(task=task, coordinator_llm=self.coordinator_llm,
                    members=self.members, mode=self.mode)
        result = await team.run()
        ctx.set(self.output_key, result.get("final_answer") or result.get("result"))
        return ctx

@dataclass
class FunctionStep:
    """Pure-Python step: transforms, DB ops, formatting — no LLM."""
    name: str
    afn: Callable[[WorkflowContext], Awaitable[WorkflowContext]]
    async def arun(self, ctx): return await self.afn(ctx)
```

### 3.4 Control-flow steps

`ConditionStep` (predicate → then/else), `LoopStep` (predicate + body + `max_iterations`,
mandatory bound — mirrors `AgentSettings.max_steps`), `ParallelStep` (run sub-steps on
deep-copied contexts via `asyncio.gather`, then merge — later keys win, same as the note).
These are unchanged from the generic note except async-only.

### 3.5 Workflow class — return Pori's run-dict shape

To stay interchangeable with `Agent.run()` / `Team.run()`, `Workflow.arun()` should return
the **same dict shape** (`{"task", "completed", "steps_taken", "trace", ...}`) rather than a
bare context. Internally it threads the context; externally it looks like any other Pori
runnable.

```python
@dataclass
class Workflow:
    name: str
    steps: List[WorkflowStep] = field(default_factory=list)

    async def arun(self, initial: WorkflowContext) -> Dict[str, Any]:
        ctx = initial
        for step in self.steps:
            ctx = await step.arun(ctx)
        return {
            "task": ctx.get("input"),
            "completed": True,
            "steps_taken": len(self.steps),
            "final_answer": ctx.get("output"),
            "context": ctx.data,
        }
```

---

## 4. Reuse what Pori already has (don't reinvent)

- **Persistence:** the note proposes a new `WorkflowDatabase` protocol. Pori already has
  `MemoryStore` (`pori/memory.py`) and `TraceStore` (`pori/observability/`). Prefer
  emitting workflow/step spans into the existing `Trace` tree (add `SpanType.workflow` /
  `SpanType.step`) over a parallel persistence layer.
- **Metrics:** aggregate child `StepMetrics`/`RunMetrics` from each worker result rather
  than inventing a new metrics shape.
- **HITL:** `AgentStep`/`TeamStep` can forward `hitl_handler`/`hitl_config` straight into
  `execute_task` / `Team`, so human-approval works inside workflows for free.
- **Config:** add an optional `WorkflowConfig` to `pori/config.py` (mirroring `TeamConfig`)
  so workflows can be declared in `config.yaml` like teams are today.

---

## 5. Divergences from the generic note (decisions to confirm)

1. **Async-only** Step protocol (Pori convention) — drop the sync `run`.
2. **Workers via blueprints + Orchestrator**, not long-lived Agent objects.
3. **Return the Pori run-dict**, not a bare `WorkflowContext`, for interchangeability.
4. **Trace/metrics reuse** instead of a new `WorkflowDatabase`.
5. **Loop bounds are mandatory** (`max_iterations`), consistent with `max_steps`/`max_failures`.

Open questions for the maintainer:
- Should `Workflow` be runnable through `Orchestrator` (e.g. `execute_workflow`) for a
  single entry point, or stand alone?
- Do steps share one `AgentMemory` (cross-step memory) or stay isolated like Team members
  (which do **not** share memory by default)? Recommendation: isolated by default, opt-in
  shared memory via context — matches Team semantics.
- Streaming: Agno exposes `print_response(stream=True)`. Worth a follow-up study on a
  structured `RunOutput` + `print_response` convenience across Agent/Team/Workflow.

---

## 6. Incremental implementation plan

1. `pori/workflow/context.py` + `steps.py` (`FunctionStep`, `AgentStep`, `TeamStep`) +
   sequential `Workflow`. Unit-test each step with a mocked Orchestrator/Team (the existing
   `conftest.py` `MockLLM` pattern covers this).
2. Control-flow steps (`ConditionStep`, `LoopStep`, `ParallelStep`) with isolated tests.
3. Trace integration (`SpanType.workflow`/`step`) + metrics aggregation.
4. `WorkflowConfig` in `config.py` + CLI wiring, mirroring how Team is configured.
5. End-to-end test: the note's Research→Draft→Critique→Loop example against `MockLLM`.

Each phase is independently shippable and testable; phase 1 alone delivers a usable
sequential pipeline.

---

## 7. Verdict

Highest-leverage Agno adaptation for Pori. It fills a real gap, has an existing design
note, composes cleanly with the current `Agent`/`Team`/`Orchestrator`, and — if it returns
the standard run-dict and reuses `Trace`/`MemoryStore` — adds a primitive without
fragmenting the architecture. Recommend proceeding with phase 1 when prioritized.
```
