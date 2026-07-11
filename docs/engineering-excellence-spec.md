# Engineering Excellence — spec: the world-class + agent-ingestible bar

_Design spec (2026-07-11). The user's directive: before the wedge arc (tasks/
events, living artifacts, triggers, canvas), the code and architecture must be
world-class — big-corp best practices — and easily ingestible by AI coding
agents. Three parallel structural audits (kernel, backend, frontend+tooling)
back every finding here with file:line evidence; this spec is the synthesis
and the phased plan._

## The verdict, honestly

The kernel is **already near the bar** — the auditors' words: "well above a
typical OSS bar" (zero bare excepts, curated `__all__` with a documented
product-seam contract, import-linter boundaries in CI, dependency bounds
enforced, disciplined test markers, excellent CLAUDE.md). The gaps are
**concentrated, not smeared**:

1. **Three god-files** hold most of the structural debt:
   `routes/conversations.py` (1398 lines, 16 endpoints, 5 concerns —
   `send_message` alone is 516 lines fusing 4 execution modes),
   `agent/core.py` (1918 lines — `execute_actions` is ~625 of them),
   `ChatPage.tsx` (765 lines, 16 useState).
2. **Typing is decorative.** No mypy config exists anywhere (CI runs it
   flagless on the kernel only); 93/99 backend route handlers lack return
   types; the Agent's most important public param is `memory: Optional[Any]`.
3. **The frontend has zero gates and zero tests.** tsc/eslint/build don't run
   in CI; no test runner exists — on the most stateful client code we own.
4. **Docs mislead agents.** `aloy_backend_api.md` describes the pre-tenancy
   single-user system (zero mentions of org/permission/RBAC); 6 of 17 route
   modules are undocumented; `aloy_backend_implementation.md` is frozen at
   the "one Run table" era but reads as current. For an agent-ingestible
   codebase, wrong docs are worse than no docs.
5. **Two real drift findings** (structure with behavior consequences):
   - The worker path builds runs WITHOUT connections/MCP/library — a durable
     run silently has no Gmail tools and no file library, and the
     `build_orchestrator` wiring is duplicated between `send_message` and
     `execute_claimed_run` (they WILL drift further).
   - `send_message` — the endpoint that spends money — is the only
     conversation endpoint gated by rate-limit alone, with no
     `require_permission(RUN_CREATE)`.

## The standards we adopt (the bar, stated once)

- **Size**: no module > ~500 lines without a documented reason; no function
  > ~80 lines; a route module owns ONE resource.
- **One implementation per concern**: run assembly, persistence, ownership
  checks each exist exactly once (the single-finalizer rule, generalized).
- **Typing as a gate**: mypy with real strictness flags on both Python
  packages, ratcheted per-module; public seams never `Any`.
- **Every surface gated in CI**: kernel, backend, app — format, lint, types,
  tests, build; pre-commit mirrors CI so nothing fails only remotely.
- **Docs an agent can trust**: every module opens with a contract docstring;
  architecture docs carry a "verified against" stamp; decisions live in ADRs;
  one START-HERE index routes humans and agents.
- **Library code never prints; failures are never silent** (`fail_open`
  helper makes intentional swallows explicit and greppable).

## The phases

### Phase 0 — Gates (near-free, land first; locks the bar before refactors)
- CI: add **app job** (tsc, eslint, vite build); add **mypy to the backend
  job** (+ dev dep); coverage floor on backend (`--cov-fail-under`, start
  where we are and ratchet); kernel floor already exists (65).
- Pre-commit ↔ CI parity: add mypy + standard hooks (eof-fixer,
  trailing-whitespace, check-yaml, check-merge-conflict); document eslint/tsc
  as the app's pre-push.
- TS strictness: `noUnusedLocals`, `noUnusedParameters`,
  `noUncheckedIndexedAccess`; eslint → type-checked config; fix
  `ecmaVersion` 2020→2023.
- `[tool.mypy]` in both pyprojects: start `check_untyped_defs` +
  `disallow_incomplete_defs`, per-module overrides ratchet.
- Hygiene: gitignore backend runtime droppings (`aloy.db`, `uvicorn.*.log`,
  `stdin.empty`), remove `products/aloy.txt` (content preserved in the
  vision docs/memory), fix CLAUDE.md CI-matrix drift, index docs/README.

### Phase 1 — Structural decomposition (the heart of the arc)
**Backend** (fixes both drift findings as a by-product):
- Extract `run_surface.py::resolve_run_surface()` (connections + MCP +
  library + denied-tools) and a shared orchestrator-assembly helper — used
  by BOTH `send_message` and the worker. Worker gains connections/MCP/library
  (or explicitly, documentedly, not).
- Add `require_permission(RUN_CREATE)` to `send_message` + clarify (compose
  with rate-limit, not instead of).
- Decompose `send_message` → ~40-line dispatcher + 4 mode handlers
  (durable / team / streaming / blocking) + `assemble_task()` +
  `StreamPersister` (replaces the 3 nested closures).
- Split `routes/conversations.py` → `routes/conversations/` package:
  crud, search, artifacts, files, live, messaging (+ shared `_helpers`).
- Shared `load_owned(model, id, context)` dependency (collapses ~46
  copy-pasted ownership checks); move `_build_team_from_config` out of
  routes into `team_execution.py`.
- Wrap blocking store/shutil I/O in async handlers with
  `run_in_threadpool`.
**Kernel** (the loop stays whole — the project rule holds):
- Extract `agent/dispatch.py` (execute_actions + closures, ~625 lines) and
  `agent/completion.py` (the 7 answer-gate predicates); metrics-glue helper
  out of `step`. core.py → ~1150 lines with a readable loop.
- Split `memory.py` → `memory/` package (stores / models / agent_memory) —
  the same move that worked for `agent/`.
- Remove the 6 `print()`s from the run loop (info already logged + emitted);
  sweep hitl/clarify/exporters prints toward events.
**Frontend**:
- ChatPage → `useConversations` / `useAttachments` / `useStreamingRun` hooks
  + `MessageList` / `ClarifyPrompt` components; named `PendingFile` type.
  ChatPage becomes a ~150-line composition.

### Phase 2 — Typing ratchet (after structure, so types land on final shapes)
- Backend: return types on all route handlers; type the private helpers;
  `TypedDict`/aliases where dict-shapes are known.
- Kernel: type the Agent constructor seams (`memory`, `guardrails`);
  ratchet `disallow_untyped_defs` module-by-module starting at leaves.
- `fail_open(logger)` helper; migrate the 42 `except Exception → debug/pass`
  sites (16 in core.py) so intent is explicit.

### Phase 3 — Docs & agent onboarding (documents the NEW shape)
- **Regenerate `aloy_backend_api.md`** around org-RBAC covering all 17 route
  modules; retire `aloy_backend_implementation.md` to `docs/history/` with a
  banner. Both get "verified against commit X" stamps.
- Module docstrings: the 21 backend + 8 kernel modules that open cold.
- `docs/adr/` seeded with the load-bearing decisions (no-LangChain,
  loop-stays-whole, sibling-binding, fail-open policy, footprint ladder,
  memory-as-index, single-finalizer, front-door-only imports).
- `START-HERE.md`: one map — "new human read X; new agent read Y; to change
  Z, read W." Per-package ARCHITECTURE notes for `pori/tools`, `pori/llm`,
  `pori/memory`, `pori/team`.
- Frontend architecture note (state conventions, api-layer contract, hook
  patterns).

### Phase 4 — Test hardening
- Vitest + RTL for the app; first targets: the attachment ladder and
  streaming/re-attach hooks (extracted in Phase 1, so now testable).
- Backend coverage to a real floor; split the 800-line kernel test files
  along the Phase-1 seams as they're touched.

### Phase 5 — Polish (scheduled, not blocking)
- Logger naming → dotted `__name__` convention; hoist non-cycle-breaking
  inline imports.
- Design-token/`cva` layer for the app (146 duplicated Tailwind class
  triplets across 30 files).
- `schemas.py`/`models.py` split by aggregate when next touched.

## Sequencing & risk rules

- Every phase = one or more PRs, each green on the FULL suite before merge;
  behavior-neutral refactors and behavior fixes (the two drift findings)
  land in SEPARATE commits so review stays honest.
- Phase 0 lands before Phase 1 so the new gates protect the refactors.
- The two Phase-1 kernel/backend decompositions are independent — can be
  parallel PRs.
- The wedge arc does not start until Phases 0–3 are merged (user directive:
  world-class first). Phase 4–5 can overlap the wedge.
