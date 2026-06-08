# deepagent_copy.md — Pori migration plan toward a deepagents-grade architecture

This document is a **concrete, sequenced migration plan** for porting the
high-leverage architectural ideas from LangChain's `deepagents` into Pori
**without** taking on LangChain or LangGraph as dependencies. Each phase is
independently shippable, has explicit file paths, public-API impact notes, and
a testable acceptance bar.

> **Guiding constraints**
> 1. No LangChain / LangGraph dependency. Direct SDK calls (`pori/llm/*`) stay.
> 2. Existing public surface (`pori.Agent`, `pori.Team`, `pori.Orchestrator`,
>    `pori.AgentMemory`, `pori.ToolRegistry`) must keep working through every
>    phase. New features land behind keyword-only kwargs with sensible defaults.
> 3. Letta-style `CoreMemory` (persona/human/notes blocks) is **kept**. It is
>    superior to deepagents' "load AGENTS.md into the prompt" memory.
> 4. Pori's eval/guardrail framework and span-based tracing are **kept**. They
>    are first-class wins over deepagents.
> 5. Phases ship as separate PRs. Every phase keeps `uv run pytest tests/ -v`
>    green plus adds new tests for the new surface.

---

## Phase 0 — Repo prep (≤1 day, pure plumbing)

**Goal:** create the directories and stub modules every later phase will land
into, so phase PRs touch only their own area.

### Changes

```
pori/
├── backends/                # NEW (Phase 1)
│   ├── __init__.py
│   ├── protocol.py          # stub: BackendProtocol Protocol class
│   └── state.py             # stub
├── middleware/              # NEW (Phase 2)
│   ├── __init__.py
│   ├── base.py              # stub: AgentMiddleware base class
│   └── types.py             # stub: ModelRequest, ModelResponse, ToolCallRequest
├── subagents/               # NEW (Phase 3)
│   ├── __init__.py
│   └── spec.py              # stub: SubAgent TypedDict
├── permissions/             # NEW (Phase 4)
│   ├── __init__.py
│   └── types.py             # stub: FilesystemPermission dataclass
├── skills/                  # NEW (Phase 5)
│   ├── __init__.py
│   └── loader.py            # stub
├── profiles/                # NEW (Phase 6)
│   ├── __init__.py
│   └── harness.py           # stub
└── runtime/                 # NEW (Phase 7)
    ├── __init__.py
    └── tool_runtime.py      # stub: ToolRuntime dataclass
```

### Acceptance

- `from pori.backends import BackendProtocol` imports.
- `pyproject.toml` package list updated.
- All existing tests still pass.

---

## Phase 1 — Backend protocol (the cornerstone)

**Goal:** unify `pori/sandbox/` and the OS-touching code in
`pori/tools/standard/filesystem_tools.py` behind one async-aware protocol.
Every later phase depends on this.

### Public API

```python
# pori/backends/protocol.py
from typing import Protocol, runtime_checkable, Optional, List
from dataclasses import dataclass

@dataclass
class FileInfo:
    path: str
    is_dir: bool = False
    size: Optional[int] = None
    modified_at: Optional[str] = None

@dataclass
class GrepMatch:
    path: str
    line: int
    text: str

@dataclass
class ReadResult:
    content: Optional[str] = None
    error: Optional[str] = None

@dataclass
class WriteResult:
    path: Optional[str] = None
    error: Optional[str] = None

@dataclass
class EditResult:
    path: Optional[str] = None
    occurrences: Optional[int] = None
    error: Optional[str] = None

@dataclass
class ExecuteResponse:
    output: str
    exit_code: Optional[int] = None
    truncated: bool = False

@runtime_checkable
class BackendProtocol(Protocol):
    def ls(self, path: str) -> List[FileInfo]: ...
    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult: ...
    def write(self, file_path: str, content: str) -> WriteResult: ...
    def edit(self, file_path: str, old: str, new: str, replace_all: bool = False) -> EditResult: ...
    def glob(self, pattern: str, path: str = "/") -> List[FileInfo]: ...
    def grep(self, pattern: str, path: str = "/", glob: Optional[str] = None) -> List[GrepMatch]: ...

    # Async pairs (default impl: asyncio.to_thread on the sync method)
    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult: ...
    async def awrite(self, file_path: str, content: str) -> WriteResult: ...
    async def aedit(self, file_path: str, old: str, new: str, replace_all: bool = False) -> EditResult: ...
    async def als(self, path: str) -> List[FileInfo]: ...
    async def aglob(self, pattern: str, path: str = "/") -> List[FileInfo]: ...
    async def agrep(self, pattern: str, path: str = "/", glob: Optional[str] = None) -> List[GrepMatch]: ...

@runtime_checkable
class SandboxBackendProtocol(BackendProtocol, Protocol):
    @property
    def id(self) -> str: ...
    def execute(self, command: str, *, timeout: Optional[int] = None) -> ExecuteResponse: ...
    async def aexecute(self, command: str, *, timeout: Optional[int] = None) -> ExecuteResponse: ...
```

### Implementations

| Class | File | Backed by |
|---|---|---|
| `StateBackend` | `pori/backends/state.py` | `AgentMemory.files: dict[str, FileData]` (new field) |
| `LocalFsBackend` | `pori/backends/local_fs.py` | wraps the existing path-validated logic from `filesystem_tools.py` |
| `SandboxBackend` | `pori/backends/sandbox.py` | wraps `pori/sandbox/local.py` (`LocalSandbox`) |
| `CompositeBackend` | `pori/backends/composite.py` | `read_backend` / `write_backend` split |

### Refactor

- Move the path-validation logic (`DEFAULT_ALLOWED_EXTENSIONS`,
  `DEFAULT_FORBIDDEN_DIRS`, traversal checks) out of every tool function and
  into `LocalFsBackend.__init__`.
- Replace the `try/except ImportError` shim at
  `pori/tools/standard/filesystem_tools.py:22-31` (sandbox path resolution)
  with a `Backend` lookup on the runtime — see Phase 7.
- `pori/tools/standard/filesystem_tools.py` tools now delegate to
  `runtime.backend.read(...)` instead of touching `pathlib` directly. The
  legacy direct-OS path is kept as a fallback only when no backend is
  configured (deprecate in Phase 8).

### Public-API impact

- `Agent.__init__` gains `backend: BackendProtocol | None = None` (defaults
  to `LocalFsBackend(root=os.getcwd())` to preserve current behavior).
- `Orchestrator.execute_task` gains the same kwarg.
- No removed APIs.

### Tests

- `tests/test_backends/test_state_backend.py` — round-trip read/write/edit/ls.
- `tests/test_backends/test_local_fs_backend.py` — path traversal denial.
- `tests/test_backends/test_composite_backend.py` — reads from A, writes to B.
- `tests/test_backends/test_sandbox_backend.py` — execute returns
  `ExecuteResponse`.
- Existing `tests/test_tools.py` filesystem tests must still pass after the
  refactor; add a fixture that spins up `LocalFsBackend` against `tmp_path`.

### Acceptance

- All existing filesystem tests pass.
- `isinstance(backend, BackendProtocol)` is True for all four implementations.
- `pori/sandbox/path_resolution.py`'s virtual-prefix logic is reachable
  exclusively through `SandboxBackend`.

---

## Phase 2 — Middleware interface (the structural unlock)

**Goal:** move planning, reflection, summarization, semantic recall,
guardrails, and HITL out of `Agent.step()` into composable middleware. Cut
`pori/agent.py` from 1,601 lines to under 400.

### Public API

```python
# pori/middleware/types.py
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from pori.llm.messages import BaseMessage

@dataclass
class ModelRequest:
    messages: List[BaseMessage]
    system_prompt: str
    tools: List[Any]                 # tool schemas
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelResponse:
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    raw: Any = None

@dataclass
class ToolCallRequest:
    tool_name: str
    tool_args: Dict[str, Any]
    tool_call_id: str

@dataclass
class ToolCallResult:
    content: str
    error: Optional[str] = None
    state_update: Optional[Dict[str, Any]] = None
```

```python
# pori/middleware/base.py
from typing import List, Optional, Awaitable, Callable

class AgentMiddleware:
    name: str  # for excluded_middleware lookup; defaults to class name

    @property
    def tools(self) -> List["ToolInfo"]:
        return []

    async def pre_model_request(self, request: ModelRequest) -> ModelRequest:
        return request

    async def post_model_response(self, response: ModelResponse) -> ModelResponse:
        return response

    async def wrap_tool_call(
        self,
        request: ToolCallRequest,
        next_: Callable[[ToolCallRequest], Awaitable[ToolCallResult]],
    ) -> ToolCallResult:
        return await next_(request)
```

### Concrete middleware (each ports one feature out of `agent.py`)

| Middleware | Replaces | File |
|---|---|---|
| `PlanningMiddleware` | `_should_plan` / `_plan_if_needed` | `pori/middleware/planning.py` |
| `ReflectionMiddleware` | `_should_reflect` / `_reflect_and_update_plan` | `pori/middleware/reflection.py` |
| `SummarizationMiddleware` | `summary_interval` block + `create_summary` call | `pori/middleware/summarization.py` |
| `SemanticRecallMiddleware` | the `memory.recall(...)` block in `_build_messages` | `pori/middleware/recall.py` |
| `CoreMemoryMiddleware` | the `core_memory.compile()` injection in `_build_messages` | `pori/middleware/core_memory.py` |
| `GuardrailsMiddleware` | the existing `Agent.guardrails` list (pre/post checks) | `pori/middleware/guardrails.py` |
| `HITLMiddleware` | wires `HITLConfig` + `HITLHandler` into `wrap_tool_call` | `pori/middleware/hitl.py` |
| `MetricsMiddleware` | the `LLMCallMetrics` / `StepMetrics` block | `pori/middleware/metrics.py` |
| `TracingMiddleware` | the span-emission block in `step()` | `pori/middleware/tracing.py` |

### `Agent` after the refactor

```python
class Agent:
    def __init__(
        self,
        task,
        llm,
        tools_registry,
        *,
        settings=AgentSettings(),
        memory=None,
        backend=None,
        middleware=(),                          # NEW: user-supplied middleware
        hitl_handler=None,
        hitl_config=None,
        guardrails=None,
        system_prompt=None,
    ):
        self._stack = self._build_default_stack(
            settings, hitl_handler, hitl_config, guardrails,
        ) + list(middleware)
        ...

    async def step(self):
        request = self._build_request()
        for mw in self._stack:
            request = await mw.pre_model_request(request)
        response = await self.llm.acomplete(request)
        for mw in reversed(self._stack):
            response = await mw.post_model_response(response)
        for tc in response.tool_calls:
            await self._dispatch_tool(tc)
```

`_dispatch_tool` walks the middleware chain via `wrap_tool_call` (onion-style;
HITLMiddleware is naturally implemented as `wrap_tool_call`).

### Public-API impact

- `Agent.__init__` gains keyword-only `middleware: Sequence[AgentMiddleware] = ()`.
- `Agent.guardrails` and `Agent.hitl_*` parameters still work — internally they
  add `GuardrailsMiddleware` / `HITLMiddleware` to the default stack.
- `AgentSettings.planning_mode` / `reflection_mode` still work — they configure
  the default `PlanningMiddleware` / `ReflectionMiddleware` instances.
- No breaking changes.

### Tests

- One unit test per middleware (`tests/test_middleware/test_*.py`) using a
  stub LLM that records the `ModelRequest` it received.
- `tests/test_middleware/test_stack_order.py` — assert the order
  `core_memory → recall → summarization → planning → reflection → metrics`.
- Re-run `tests/test_agent.py` unchanged. It must pass without edits.

### Acceptance

- `pori/agent.py` is under 400 lines (currently 1,601).
- Removing all of Pori's middleware leaves a working tool-calling loop.

---

## Phase 3 — Sub-agents as data + the `task` tool

**Goal:** let the main agent spawn isolated sub-agents mid-step (each with its
own `AgentMemory`, optional model override, optional sub-permissions) and get a
single string result back. This is the killer feature.

### Public API

```python
# pori/subagents/spec.py
from typing import TypedDict, NotRequired, Sequence, List, Optional
from pori.llm.base import BaseChatModel
from pori.middleware.base import AgentMiddleware
from pori.tools.registry import ToolInfo

class SubAgent(TypedDict):
    name: str
    description: str
    system_prompt: str
    tools: NotRequired[Sequence[ToolInfo]]
    model: NotRequired[BaseChatModel]
    middleware: NotRequired[List[AgentMiddleware]]
    permissions: NotRequired[List["FilesystemPermission"]]   # Phase 4
```

```python
# pori/middleware/subagents.py
class SubAgentMiddleware(AgentMiddleware):
    def __init__(self, subagents: List[SubAgent], backend, default_model):
        ...

    @property
    def tools(self) -> List[ToolInfo]:
        return [self._build_task_tool()]

    def _build_task_tool(self) -> ToolInfo:
        # Tool name: "task". Params: subagent_type: str, instructions: str.
        # Implementation: instantiate a fresh Agent with the named subagent's
        # system_prompt, tools, model, middleware. Run to completion. Return
        # the final answer string. Sub-agent gets its own AgentMemory.
        ...
```

### `Agent` integration

```python
agent = Agent(
    task="Research and summarize three claims about X",
    llm=anthropic_sonnet,
    tools_registry=tr,
    subagents=[
        {
            "name": "researcher",
            "description": "Performs deep web research on a single claim",
            "system_prompt": "...",
            "tools": [internet_search, read_url],
            "model": anthropic_haiku,            # cheap for parallel research
        }
    ],
)
```

A default `general-purpose` sub-agent is auto-injected unless the caller passes
one with that name (mirrors deepagents).

### Relationship to `Team`

- `Team` (router/broadcast/delegate) keeps its current role: **operator-level**
  multi-agent coordination invoked by user code.
- `subagents` on `Agent` is **agent-level** delegation invoked by the LLM
  through the `task` tool.
- Both can coexist. `Team` members can themselves declare `subagents`.

### Public-API impact

- `Agent.__init__` gains `subagents: Sequence[SubAgent] = ()`.
- New module `pori/subagents/` with `SubAgent` TypedDict.
- No breaking changes.

### Tests

- `tests/test_subagents/test_task_tool.py` — main agent calls `task("researcher", "...")`,
  asserts the sub-agent ran with its own memory and returned a string.
- `tests/test_subagents/test_isolation.py` — sub-agent mutations to its own
  `AgentMemory` are not visible to the parent.
- `tests/test_subagents/test_default_general_purpose.py` — `general-purpose` is
  auto-added.

### Acceptance

- Main agent can call `task("general-purpose", "Compute X")` and receive a
  string result.
- Two parallel `task` calls run concurrently (asyncio.gather).

---

## Phase 4 — Filesystem permissions as data

**Goal:** declarative path/operation rules enforced by middleware that always
sits last in the stack.

### Public API

```python
# pori/permissions/types.py
from dataclasses import dataclass
from typing import List, Literal

FilesystemOperation = Literal["read", "write"]

@dataclass
class FilesystemPermission:
    operations: List[FilesystemOperation]
    paths: List[str]                          # glob patterns, "/workspace/**"
    mode: Literal["allow", "deny"] = "allow"
```

```python
# pori/middleware/permissions.py
class PermissionMiddleware(AgentMiddleware):
    def __init__(self, rules: List[FilesystemPermission], backend):
        ...

    async def wrap_tool_call(self, request, next_):
        if self._is_filesystem_tool(request.tool_name):
            decision = self._evaluate(request)
            if decision == "deny":
                return ToolCallResult(content="", error="permission_denied")
        return await next_(request)
```

### Stack invariant

`PermissionMiddleware` is **always appended last** by `Agent._build_default_stack`
so it sees every tool other middleware injected (filesystem tools, sub-agent
tools, user tools).

### Public-API impact

- `Agent.__init__` gains `permissions: List[FilesystemPermission] = ()`.
- Sub-agents inherit unless they declare their own `permissions`.

### Tests

- `tests/test_permissions/test_deny_write.py`.
- `tests/test_permissions/test_allow_read_only_subtree.py`.
- `tests/test_permissions/test_subagent_inheritance.py`.

### Acceptance

- A rule `[FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")]`
  causes `write_file` and `edit_file` to return `permission_denied` regardless
  of which middleware exposed the tool.

---

## Phase 5 — Skills (Anthropic agent-skills pattern)

**Goal:** load `SKILL.md` files (YAML frontmatter + markdown) from configurable
backend paths, layer them (base → user → project), inject summaries into the
system prompt with progressive disclosure.

### Public API

```python
# pori/middleware/skills.py
class SkillsMiddleware(AgentMiddleware):
    def __init__(self, backend: BackendProtocol, sources: List[str]):
        # sources: ["/skills/base/", "/skills/user/", "/skills/project/"]
        ...

    async def pre_model_request(self, request):
        skills_block = self._render_skills_section()
        request.system_prompt = request.system_prompt + "\n\n" + skills_block
        return request
```

`SKILL.md` shape:
```markdown
---
name: web-research
description: Structured approach for thorough web research
---
# Web Research Skill
## When to Use
- ...
```

`Agent.__init__` gains `skills: List[str] | None = None`.

### Tests

- `tests/test_skills/test_loader.py` — round-trip frontmatter parse.
- `tests/test_skills/test_layering.py` — last-source-wins on name collision.
- `tests/test_skills/test_prompt_injection.py` — block lands in the system
  prompt.

### Acceptance

- `Agent(skills=["/skills/user/", "/skills/project/"])` injects both sets,
  with project overriding user on name conflict.

---

## Phase 6 — Harness profiles per model

**Goal:** different models get different system-prompt suffixes, tool
descriptions, and middleware. Encoded as data, keyed off model identity.

### Public API

```python
# pori/profiles/harness.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from pori.middleware.base import AgentMiddleware

@dataclass
class HarnessProfile:
    name: str
    matches: Callable[[str], bool]                          # model id predicate
    system_prompt_suffix: Optional[str] = None
    tool_description_overrides: Dict[str, str] = field(default_factory=dict)
    extra_middleware: List[Callable[[], AgentMiddleware]] = field(default_factory=list)
    excluded_tools: List[str] = field(default_factory=list)
```

Built-in profiles:

- `pori/profiles/_anthropic.py` — adds Anthropic prompt-cache breakpoint
  middleware, "use thinking blocks for hard problems" suffix.
- `pori/profiles/_openai.py` — strips verbose tool descriptions; adds
  "respond concisely" suffix.
- `pori/profiles/_google.py` — Gemini-specific tool-schema massaging.

`Agent.__init__` resolves profile automatically from `llm.model_id`. Caller can
override with `harness_profile=HarnessProfile(...)`.

### Tests

- `tests/test_profiles/test_resolution.py` — Anthropic model id → Anthropic
  profile.
- `tests/test_profiles/test_suffix_applied.py` — system prompt ends with the
  suffix.

### Acceptance

- Switching `llm` from Anthropic to OpenAI changes the assembled stack
  without any caller code change.

---

## Phase 7 — `ToolRuntime` injection

**Goal:** stop tools from importing global state via `try/except ImportError`.
Inject everything they need via a runtime parameter.

### Public API

```python
# pori/runtime/tool_runtime.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pori.memory import AgentMemory
from pori.backends.protocol import BackendProtocol

@dataclass
class ToolRuntime:
    memory: AgentMemory
    backend: BackendProtocol
    task_id: str
    metadata: Dict[str, Any]
```

### Tool signature change

Today:
```python
@Registry.tool(name="read_file", description="...")
def read_file(params: ReadFileParams) -> Dict[str, Any]:
    ...
```

After:
```python
@Registry.tool(name="read_file", description="...")
def read_file(runtime: ToolRuntime, params: ReadFileParams) -> Dict[str, Any]:
    return runtime.backend.read(params.path).__dict__
```

`ToolRegistry.register_tool` introspects the signature: if the first parameter
is annotated `ToolRuntime`, the executor injects it. Older tools without it
still work (back-compat).

### Refactor scope

- All tools in `pori/tools/standard/filesystem_tools.py` and
  `pori/sandbox/sandbox_tools.py` move to runtime injection.
- Delete the `try/except ImportError` shim at the top of `filesystem_tools.py`.

### Tests

- `tests/test_tools/test_runtime_injection.py` — tool with `runtime` param
  receives the right `AgentMemory` instance.
- `tests/test_tools.py` continues to pass.

### Acceptance

- `pori/tools/standard/filesystem_tools.py` no longer imports from
  `pori.sandbox.path_resolution`.

---

## Phase 8 — Async tool path

**Goal:** make the tool execution path natively async; sync tools run via
`asyncio.to_thread`.

### Changes

- `ToolExecutor.execute` becomes `async def aexecute`.
- Sync `execute` kept as a thin wrapper (`asyncio.run(self.aexecute(...))`)
  for back-compat.
- `Agent._dispatch_tool` and `wrap_tool_call` are already async after Phase 2;
  they now `await` the tool.

### Acceptance

- `await asyncio.gather(executor.aexecute(...), executor.aexecute(...))` runs
  two tools concurrently.
- Existing sync tool registration still works.

---

## Phase 9 — Excluded middleware contract + required-middleware invariant

**Goal:** if you let users disable middleware, they shouldn't be able to break
core invariants silently.

### Implementation

```python
# pori/middleware/_required.py
_REQUIRED_MIDDLEWARE: tuple[type[AgentMiddleware], ...] = (
    PermissionMiddleware,
    SubAgentMiddleware,
)

def apply_excluded(stack, excluded):
    matched = set()
    out = []
    for mw in stack:
        if mw.__class__ in excluded or mw.name in excluded:
            if mw.__class__ in _REQUIRED_MIDDLEWARE:
                raise ValueError(
                    f"Cannot exclude required middleware: {mw.name}"
                )
            matched.add(mw.name)
            continue
        out.append(mw)
    unmatched = set(_to_names(excluded)) - matched
    if unmatched:
        raise ValueError(f"excluded_middleware did not match anything: {unmatched}")
    return out
```

### Tests

- `tests/test_middleware/test_excluded_required.py` — raises on
  `excluded=["PermissionMiddleware"]`.
- `tests/test_middleware/test_excluded_typo.py` — raises on entry that
  matches nothing in the stack.

### Acceptance

- An `excluded_middleware` typo fails loudly at construction time.

---

## Phase 10 — Monorepo split

**Goal:** mirror deepagents' `libs/{deepagents,cli,evals,acp,partners/*}` so
the CLI and provider extras can version independently of the SDK.

### Target layout

```
Workspace/Pori/
├── packages/
│   ├── pori-core/           # was Pori/pori (agent, middleware, backends, memory, eval, observability)
│   ├── pori-cli/            # was Pori/pori/cli.py + main.py — separately versioned
│   ├── pori-evals/          # was Pori/pori/eval/ — separately versioned
│   ├── pori-providers-anthropic/
│   ├── pori-providers-openai/
│   ├── pori-providers-google/
│   ├── pori-providers-fireworks/
│   ├── pori-providers-openrouter/
│   └── pori-api/            # was pori_cloud/ — already separate, fold under packages/
└── pori_website/            # untouched
```

`pori-cli` pins `pori-core==X.Y.Z` exactly; CI fails on drift, mirroring the
deepagents `dangerous-skip-sdk-pin-check` mechanism.

### Acceptance

- `uv run pori` still launches the CLI with `pori-cli` installed.
- `pip install pori-core` works without pulling provider extras.
- `pip install "pori-core[anthropic,openai]"` installs only those provider
  packages.

---

## What we are explicitly **not** doing

- **No LangChain/LangGraph adoption.** `Command(update={...})` style updates,
  `BaseChatModel` from langchain-core, and the graph runtime are out. Pori's
  direct-SDK design is a feature.
- **No replacement of `CoreMemory`.** Pori's persona/human/notes blocks +
  `memory_insert` / `memory_rethink` tools are *better* than deepagents' simple
  AGENTS.md loading. They become a `CoreMemoryMiddleware` (Phase 2) but the
  primitive stays.
- **No replacement of the eval/guardrail framework.** Pori's "same `BaseEval`
  interface for offline eval and runtime guardrail" stays. `GuardrailsMiddleware`
  in Phase 2 wires the runtime side; offline `pori-evals` keeps its own CLI.
- **No replacement of span-based tracing.** `agent.run() -> {trace, metrics}`
  stays. `TracingMiddleware` and `MetricsMiddleware` (Phase 2) take over the
  emission.

---

## Sequencing rationale

```
Phase 1 (Backends) ──► Phase 2 (Middleware) ──► Phase 3 (Sub-agents)
                                          │
                                          ├─► Phase 4 (Permissions)
                                          ├─► Phase 5 (Skills)
                                          ├─► Phase 6 (Profiles)
                                          ├─► Phase 7 (ToolRuntime)
                                          └─► Phase 8 (Async tools)
                                                       │
                                                       └─► Phase 9 (Excluded)

Phase 10 (Monorepo) — orthogonal, can land anytime after Phase 7.
```

Phases 1 and 2 are blocking. Phases 3–9 can land in any order after 1+2.
Phase 10 is independent.

---

## Risk register

| Risk | Mitigation |
|---|---|
| Phase 2 refactor breaks `tests/test_agent.py` | Land middleware behind a `_use_middleware_stack=True` feature flag for one release; flip default in the next. |
| `BackendProtocol.read` returning `ReadResult` is a behavior change for filesystem tools | Tools wrap the result and emit the same dict shape they emit today; assertions in `tests/test_tools.py` stay green. |
| `task` tool causes recursive sub-agent infinite loops | Sub-agent inherits `max_steps` minus the parent's used budget; sub-agents cannot themselves declare further sub-agents in v1. |
| Permissions deny too aggressively | `mode="deny"` rule with no matching rule above it = allow. Same precedence as deepagents — first match wins. |
| Profile auto-resolution picks the wrong profile | Caller can always pass `harness_profile=...` explicitly. Auto-resolution emits a debug log. |
| Monorepo split breaks `pip install pori` | Keep a thin `pori` meta-package that depends on `pori-core[all]` for one major version. |

---

## Definition of done

The migration is complete when:

1. `pori/agent.py` is under 400 lines and contains no planning/reflection/
   summarization/recall/HITL/guardrail/metrics/tracing logic — all of those
   live in middleware.
2. `from pori.backends import StateBackend, LocalFsBackend, SandboxBackend,
   CompositeBackend` works and all four pass the `BackendProtocol` runtime
   check.
3. An `Agent` can call `task("general-purpose", "...")` and receive a string.
4. `permissions=[FilesystemPermission(...)]` and `skills=[...]` both work.
5. Switching the LLM provider auto-applies a different harness profile.
6. All tools receive a `ToolRuntime` and the `try/except ImportError` shim in
   `filesystem_tools.py` is gone.
7. `excluded_middleware=["PermissionMiddleware"]` raises.
8. `pori-core`, `pori-cli`, and at least one `pori-providers-*` package are
   installable independently.
9. `uv run pytest tests/ -v` is green and adds at least one test file per new
   module.
