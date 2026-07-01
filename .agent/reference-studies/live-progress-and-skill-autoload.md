# Live Progress & Skill Auto-Loading — Reference Study + Fix Scope

**Status:** study + scope (no code yet).
**Motivation:** two UX gaps surfaced in a real CLI session (`docs/a.txt`):

1. **Opaque progress.** The CLI prints `--> Step N starting…` then goes silent
   during the LLM call. For a generation-heavy task ("write a poem", the 24–35s
   MoE lessons) the user stares at a step counter with zero feedback — looks hung.
2. **Skills don't auto-load from the task.** The `teach` skill only ran when
   forced with `/teach@1`, and the model then narrated a false, self-contradictory
   explanation ("I can't use it, it has `model_invocation_disabled`") *while
   actually using it*.

This documents how **Hermes** and **Claude Code** solve both, what **Pori** does
today, and scopes the two fixes. References are to the vendored copies under
`references/hermes-agent/` and `references/claude-code/`.

---

## Cause 1 — Live progress (never surface "Step N")

### Hermes
Token-by-token streaming through a callback → queue → consumer bridge
(`.plans/streaming-support.md`, `agent/conversation_loop.py`):

- A `stream_callback(text_delta: str)` is injected into the agent. The LLM
  streams deltas into a thread-safe `queue.Queue()`.
- **Each consumer renders in its own context:** CLI prints to the terminal live,
  the Gateway *edits* the Telegram/Discord message progressively, the API server
  emits SSE to the client.
- Feature-flagged (`streaming.enabled`, default off); non-streaming path stays
  intact as fallback; graceful degradation if the provider can't stream.
- Handles mid-stream timeouts (`PARTIAL_STREAM_STUB_ID` + a continuation prompt)
  so a long generation degrades instead of hanging.

### Claude Code
Streams assistant text token-by-token **and** renders each tool call as it fires
(`Bash(npm test)`, `Read(x.ts)`) under a live spinner showing elapsed time +
token count + "esc to interrupt".

### The shared principle
The unit shown to the user is **content/action streaming out**, not the internal
loop index. "Step N" is an implementation detail they never surface.

### Pori today (the gap)
- `on_step_start` prints `--> Step {n} starting…` (`pori/main.py:1068`).
- The agent then blocks on `get_next_action()` → `ainvoke_tools()` — a **single
  non-streaming** `chat.completions.create(...)` (`pori/llm/openai.py`, no
  `stream=True`) — with **no output until it returns** (`pori/agent.py:637`).
- No explicit request timeout: a genuine stall relies on the SDK default and is
  indistinguishable from "slow"; `retry_async` only fires on a raised error.

Pori shows the loop counter and goes dark during generation — the opposite of
both references.

---

## Cause 2 — Skills load from the task (no `/skill`)

### Hermes
A **skill index** in the system prompt + progressive disclosure
(`agent/system_prompt.py:280`, `build_skills_system_prompt`):

- Every skill's **name + description/category** is always injected into the
  *stable* tier (cheap, names-only — the full `SKILL.md` is not preloaded).
- The model reads the index and, when the task matches, calls **`skill_view`** to
  load the full instructions on demand. `/skill-name` is only a manual override.

### Claude Code
Same progressive disclosure via `SKILL.md` frontmatter (confirmed in-repo):

```yaml
name: frontend-design
description: Guidance for distinctive visual design … Helps with … when building new UI …
```

The `name` + `description` are always in context and act as the **trigger**; the
model reads the body + `references/` only when the task fits
(`plugins/plugin-dev/agents/skill-reviewer.md:73`). No slash command.

### Pori today (already ~matched — one config knob blocks it)
Pori **already has this architecture**: `_render_available_skills_prompt`
(`pori/agent.py:379`) builds the same Hermes-style index, and
`skills_list`/`skill_view` are the same progressive-disclosure tools.

The reason `teach` didn't auto-load: it is flagged **`disable-model-invocation`
(`model_invocation_disabled: True`)**, and Pori deliberately:
- **excludes such skills from the auto-index** (`pori/agent.py:392`), and
- **never auto-nudges them** (`pori/agent.py:1213`).

Neither reference has a "manual-only" flag as the default — a skill's description
*is* its auto-trigger. So the flag is opting `teach` out of the exact behavior we
want. Two distinct problems fall out:

- **(2a) Auto-load:** the flag hides the skill from the model's task-driven path.
- **(2b) Narration leak:** `skills_list` dumps the *whole* entry incl.
  `model_invocation_disabled` to the model (`pori/tools/standard/skills_tools.py:81,93`).
  The model reads it literally as "forbidden", so even after a forced load it
  narrates "I answered directly" — false and self-contradictory.

---

## Fix A — Live streaming progress

Adopt Hermes's callback→consumer pattern. Feature-flagged, default off, so it
ships behavior-neutral.

- **A.1 — Provider streaming.** Add a streaming variant to the LLM wrappers:
  `ainvoke_tools(..., on_delta: Callable[[str], None] | None = None)` that sets
  `stream=True`, accumulates text + `tool_use`/`tool_calls` deltas, and calls
  `on_delta(text)` per chunk. Anthropic and OpenAI both stream tool calls; keep
  the non-streaming path when `on_delta` is None.
- **A.2 — Thread the callback.** `Agent.get_next_action` accepts an
  `on_text_delta` hook (from `AgentSettings`/run options) and forwards it to the
  provider. Non-streaming stays default.
- **A.3 — CLI consumer.** Replace the dead air after `Step N starting…` with the
  streamed text (and tool-call previews as they arrive via `build_tool_preview`).
  Prefer showing the *activity/content*, demote the raw step counter.
- **A.4 — Cloud consumer.** Add a `token`/`delta` SSE event to
  `stream_agent_execution` (the consumer scaffolding already exists) so the web
  UI types the answer out live; the client reader already handles event types.
- **A.5 — Explicit timeout.** Pass a per-call request timeout so a true stall
  raises (and retries) instead of hanging silently — closes the "is it slow or
  dead?" ambiguity independent of streaming.
- **Flag:** `config.llm.streaming` (or `AgentSettings.stream`), default off;
  enable in the CLI first, verify on Kimi, then default on.

**Blast radius:** `pori/llm/*` (streaming path), `pori/agent.py` (hook),
`pori/main.py` (CLI render), `pori_cloud/streaming.py` + client SSE. Behind a flag
→ every phase green.

## Fix B — Skill auto-load + honest narration

Two small, independent changes; (B.1/B.2) are safe and high-value, (B.3) is a
policy decision.

- **B.1 — Stop leaking the routing flag to the model.** `skills_list` should
  return a *model-facing* view (name, description, commands, readiness,
  eligibility) and **omit `model_invocation_disabled`** (and other internal
  routing fields). It's an orchestration guard, not model data. Kills the
  false-narration bug at the source. (`pori/tools/standard/skills_tools.py`)
- **B.2 — Make a loaded skill obviously usable.** When a skill is force-loaded
  (slash command / `skill_view`), the tool result / prompt should state that its
  instructions are now in effect and must be followed — so the model can't claim
  it "can't use" a skill it just loaded.
- **B.3 — Reconsider the default (decision).** Hermes/Claude Code make skills
  auto-discoverable by default; the description is the trigger. Options:
  - keep `disable-model-invocation` but reserve it for genuinely
    dangerous/expensive skills and **document** it, or
  - flip the default so skills are auto-indexed unless explicitly hidden.
  Either way, `teach` should almost certainly **not** carry the flag — dropping
  it makes Pori auto-load it from the task, matching the references.
- **B.4 — Prompt guidance.** One line in the skills guidance: `skill_view` loads
  instructions on demand; once loaded, follow them regardless of how discovery
  happened.

**Blast radius:** `pori/tools/standard/skills_tools.py`, the skills-guidance
prompt, and (B.3) the `teach` skill config + docs. No architecture change — Pori
already has the index + progressive disclosure.

---

## Sequencing

1. **B.1 + B.2** first — tiny, no flag, immediately removes the confusing
   narration and is independently correct.
2. **B.3** — quick config/doc decision; unblocks task-driven skill loading.
3. **A.5** (timeout) — small, removes the "hang vs slow" ambiguity now.
4. **A.1–A.4** (streaming) — the larger effort; flag-gated, CLI first, then Cloud
   SSE + client. This is the one genuinely missing capability vs the references.

## Verdict
- **Skills:** architecturally Pori already matches Hermes/Claude Code; the gap is
  a config flag + a `skills_list` leak, not missing machinery.
- **Progress:** streaming is a real missing capability; the callback→consumer
  pattern from Hermes is the direct, proven adoption, and Pori's Cloud SSE
  consumer is already in place to receive it.
