# Reference Study: Native Tool-Calling (Phase B)

Scope for migrating Pori from its hand-rolled JSON-envelope output contract to
native provider tool-calling. Companion to
`system-prompt-architecture.md` (Phase A is complete; this is Phase B).

Status: **scope / design** — no code changed yet.

---

## 1. Where Pori is today (grounded)

Every provider is handed a Pydantic `output_model` and asked to return one JSON
object matching:

```python
# pori/agent.py:122
class AgentOutput(BaseModel):
    current_state: Dict[str, str]      # next_goal/evaluation/memory — prompt-enforced only
    action: List[Dict[str, Any]]       # [{tool_name: params}], single-key dicts
```

- The **real tool registry is never sent as `tools=`.** Tool schemas are rendered
  as *text* via `ToolRegistry.get_tool_descriptions()` (registry.py:293) into
  `{tool_descriptions}` in `agent_core.md`.
- Structured output is achieved differently per provider:
  - **Anthropic** (`llm/anthropic.py:86`) — builds *one synthetic tool* = the
    output schema and forces `tool_choice` to it (native API used only to force
    JSON). Closest to native.
  - **OpenAI** (`llm/openai.py:80`) — JSON mode (`response_format=json_schema`).
  - **Google** (`llm/google.py:63`) — JSON mode (`response_schema`).
  - **OpenRouter/Fireworks** — subclass OpenAI, inherit JSON mode.
- `get_next_action()` (agent.py:803-940) carries a large
  `_coerce_to_output_json()` recovery heuristic + a one-shot retry on malformed
  JSON — the brittleness native tool-calling removes.
- Tool dispatch: `execute_actions` assumes single-key dicts
  (`tool_name = list(action.keys())[0]`, agent.py:1650) → `ToolExecutor.execute_tool`.
- **No message types** for tool calls/results (`llm/messages.py` has only
  System/User/Assistant, all `{role, content: str}`). Tool results re-enter as
  plain `system` text.
- **No streaming** anywhere.
- `create_tools_model` (registry.py:327) exists but is **dead/unused** — an
  intended hook for real tool schemas.

## 2. What native tool-calling changes

Pass real tool JSON schemas to the provider's function-calling API; read back
structured `tool_use`/`tool_calls`/`functionCall` blocks (+ optional assistant
text). Wins: reliability (models are trained on it), provider-validated args,
parallel calls, cleaner/cacheable prompt, fewer tokens, deletable recovery code.

### The one hard design problem: re-homing `current_state`

Native output is `text + tool_calls`, with **no sibling JSON object**, so
`next_goal`/`evaluation_previous_goal`/`memory` lose their carrier. Decision:

- **`next_goal` (the activity line) ← the assistant's text content** that
  accompanies the tool calls. Prompt the model to lead with a one-line,
  present-tense status; map that text to `state.current_activity`.
- `evaluation_previous_goal` / `memory` were informational only (never
  structurally consumed) — drop them in native mode.

This keeps the activity-descriptor feature working (now sourced from real
assistant text instead of an envelope field).

---

## 3. Design

### 3.1 Flag-gated, envelope stays default until proven
- `config.llm.tool_calling: "envelope" | "native"` (default `envelope`).
- Thread a `tool_calling` mode to the LLM wrappers and the agent. The envelope
  path is untouched; native is a parallel branch. Flip the default only in B.5.

### 3.2 New message types (`llm/messages.py`)
- `ToolCall { id, name, arguments: dict }`.
- `AssistantMessage` gains optional `tool_calls: list[ToolCall]` and keeps `content`.
- `ToolResultMessage { role="tool", tool_call_id, content }`.
- Each provider's `ainvoke` gains conversion for these (Anthropic `tool_use`/
  `tool_result` blocks; OpenAI `tool_calls`/`role:"tool"`; Google `functionCall`/
  `functionResponse`).

### 3.3 Tool schemas from the registry
- Add `ToolRegistry.tool_schemas()` (or reuse the snapshot surface,
  registry.py:274) → provider-agnostic `[{name, description, input_schema}]` from
  each `param_model.model_json_schema()`. Each provider maps to its tool format.
- Retire/replace the dead `create_tools_model`.

### 3.4 New LLM entry point (`llm/base.py`)
- `async ainvoke_tools(messages, tools) -> ToolTurn` where
  `ToolTurn { text: str, tool_calls: list[ToolCall], usage }`.
- Keep `with_structured_output` **as-is** for genuine extraction calls
  (`PlanOutput`, `ReflectOutput`, `CompletionValidation` — agent.py:1403/1472/1530);
  those are not tool dispatch and should stay JSON/structured.

### 3.5 Agent native branch (`agent.py`)
- In `get_next_action()`, when mode == native: call `ainvoke_tools(messages,
  registry tools)`; translate each `ToolCall(name,args)` → the existing
  `{name: args}` action dict (so `execute_actions` is unchanged); set
  `current_activity` from `ToolTurn.text`.
- Tool results go back as `ToolResultMessage` (native) instead of `system` text.
- The recovery/`_coerce_to_output_json` path is **envelope-only**; native needs none.

### 3.6 Prompt (native variant)
- In native mode, drop the "JSON Output Format" + `{tool_descriptions}` + JSON
  `CRITICAL RULES` from the stable tier; keep identity + workflow + non-JSON
  rules. Smaller, more cacheable prompt.

---

## 4. Phasing (each step ships green)

- **B.1** — additive: `ToolCall`/`ToolResultMessage` types, `tool_schemas()`,
  `ainvoke_tools` on the base Protocol (+ a `ToolTurn`). No behaviour change;
  unit tests for types/schemas.
- **B.2 (done)** — `ChatAnthropic.ainvoke_tools` (real native path) +
  `Agent(tool_calling="native")` branch (`_get_next_action_native`): maps
  `ToolCall -> {name: args}`, sets `next_goal` from assistant text, skips the
  envelope/recovery. Native mock + tests (incl. end-to-end answer). Envelope
  stays default → suite green. `tests/test_native_tool_calling.py`.
- **B.3 (done)** — `ChatOpenAI.ainvoke_tools` (OpenAI tool format; OpenRouter/
  Fireworks inherit it) and `ChatGoogle.ainvoke_tools` (Gemini
  `function_declarations` + a schema sanitizer for unsupported keys). Tested:
  OpenAI parsing, Fireworks/Kimi inheritance, Gemini sanitizer. **Note:** the
  Kimi/OpenAI path is the verified one (B.1 spike + unit tests); the Google path
  is implemented but not yet live-verified (no Gemini key) — envelope remains its
  fallback via the flag.
- **B.4 (done)** — `agent_core_native.md` (no JSON envelope / `{tool_descriptions}`;
  adds native tool-call + one-line status guidance); agent picks the prompt by
  mode. `config.llm.tool_calling` threaded through orchestrator -> CLI ->
  `Agent`. **Live-verified end-to-end on Kimi/Fireworks**: native tool calls →
  actions → execution → receipts → activity line from assistant text. Tests +
  live run green; envelope still the default.
- **B.5** — flip default to native; migrate `conftest.py` mock + ~90 hardcoded
  `AgentOutput` literals; delete the envelope path, `_coerce_to_output_json`, and
  the retry logic.

## 5. Blast radius & risks

- **Core:** `llm/messages.py`, `llm/base.py`, `llm/anthropic.py`/`openai.py`/
  `google.py` (+ inherited openrouter/fireworks), `agent.py`
  (`get_next_action`, `AgentOutput`, `current_state` plumbing), `tools/registry.py`,
  `prompts/system/agent_core.md`, `config.py`.
- **Tests:** ~106 occurrences across 10 files; the heavy one is `conftest.py`
  (`MockLLM`/`MockLLMResponse`). Strategy: keep envelope tests green through
  B.1-B.4; do the literal migration in B.5 when the default flips.
- **Risks:**
  - **Fireworks/Kimi tool-calling support** — **verified working** (B.1 spike):
    `kimi-k2p6` via Fireworks returned a proper OpenAI-style `tool_call`
    (`get_weather {"city":"Paris"}`, empty content). The envelope fallback via the
    flag remains for any provider/model that doesn't.
  - **Re-homing reasoning** — `next_goal` from assistant text may be terser/absent
    for some models; the prompt must steer it.
  - **Provider quirks** — Google `functionCall` parts, OpenAI `tool_choice`
    defaults, Anthropic parallel `tool_use` blocks + interleaved text.

## 6. Non-goals (Phase B)
- Streaming tool calls (separate follow-on).
- Migrating `PlanOutput`/`ReflectOutput`/`CompletionValidation` off structured
  output — they stay JSON/structured.
- Removing the envelope before native is proven across providers (B.5 only).
