# pori/tools — tool registration, gating, and execution

## What this package owns
Everything between "a Python function exists" and "the model can call it":
schema generation, capability gating, side-effect metadata, authorization
policy, and validated execution. The agent loop calls into this package; it
never dispatches tools itself.

## Files
- `registry.py` — the core. `ToolRegistry` holds `ToolInfo` entries
  (Pydantic-validated params, `SideEffect` metadata, capability group,
  optional per-tool `check_fn`). `snapshot()` resolves gating into a
  `CapabilitySnapshot`; `filtered()` derives a restricted registry (used for
  sub-agents and org policy); `tool_schemas` is the exact surface shipped on
  every LLM call. `ToolExecutor.execute_tool` validates params and runs the
  function. `tool_registry()` returns the process-wide default registry.
- `policy.py` — declarative authorization for side-effecting calls.
  `ToolAuthorizationPolicy` + `AuthorizationDecision`; keeps intent
  heuristics out of the agent loop (`agent/dispatch.py` consumes it).
- `standard/__init__.py` — `register_all_tools()` installs the built-in
  domains plus `pori.tools` entry-point plugins. `STANDARD_KERNEL_TOOLS`
  lists the always-on kernel tools.
- `standard/core_tools.py` — `answer`, `done`, `ask_user`, `think`, memory
  tools. `answer`/`done` are the loop's terminal signals — never remove.
- `standard/filesystem_tools.py`, `standard/internet_tools.py`,
  `standard/planning_tools.py`, `standard/skills_tools.py` — the other
  built-in domains (internet is capability-gated on `TAVILY_API_KEY`).

## Key contracts
- Register via `@registry.tool(...)` or `register_tool(...)`; params are a
  Pydantic model, results a `{"success": bool, ...}` dict.
- Every registered tool's schema ships on **every** LLM call — a new core
  tool is a permanent context tax. Climb the Footprint Ladder (CLAUDE.md)
  before adding one; gate with a `CapabilityGroup` prerequisite or
  `check_fn` so unusable tools vanish from the model surface.
- `CollisionPolicy` governs duplicate names (error/keep/replace);
  protected (kernel) tools cannot be replaced.
- Sandbox-aware variants of file tools live in `pori/sandbox/`, not here.

## Change X → look at Y
- New built-in tool → the right `standard/*_tools.py` + gating; check
  `STANDARD_KERNEL_TOOLS` only if it must be always-on.
- Tool schemas look wrong to the model → `ToolRegistry.tool_schemas` /
  `create_tools_model` in `registry.py`.
- Sub-agent or org tool restriction bugs → `filtered()` / `snapshot()`.
- "Side effect blocked/authorized" behavior → `policy.py`, then its caller
  in `pori/agent/dispatch.py` (HITL gating is `pori/hitl.py`, wired into
  `ToolExecutor`).
- Tool output truncation → `output_limit_for` in `registry.py`.
