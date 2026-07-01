# Streaming Event Architecture — "Feels Alive" Blueprint

**Status:** blueprint (design agreed; not yet built).
**Supersedes/extends:** [[live-progress-and-skill-autoload]] (the earlier study of
the same problem). This is the concrete build plan.

## 1. The problem

Pori buffers a whole turn, then dumps it — 10s of silence, then everything at
once. Two structural causes:

1. **Prose is trapped in JSON.** The model answers via
   `answer({"final_answer": "...poem...", "reasoning": "...", "artifact_references": [...]})`.
   Half a JSON string (`{"final_answer": "In Pretoria's he`) isn't renderable, so
   nothing can show until the tool-call JSON closes.
2. **No intermediate state is exposed.** Pori logs only at step boundaries; a
   tool call isn't surfaced until *after* it runs, and a 10s pause can't be told
   apart from a stalled/retrying API call.

Agentic tools "feel alive" because they stream **raw model events as they
happen** and expose real intermediate state (tool names, reasoning) — not faked
spinners.

## 2. The core principle — split prose from data

The `answer` tool bundles three things, but only one of them is actually data:

| Field | Kind | Belongs as |
|---|---|---|
| `final_answer` | prose | **streamed text** |
| `reasoning` | prose | **streamed text** (a "thinking" block) |
| `artifact_references` | data (receipt IDs) | **stays in the tool** |

> **The rule:** prose (answer + reasoning) streams as text; the tool carries only
> **structured, verifiable data** (the receipts). The tool stops being a
> container for everything the model wants to say.

**Why this is safe (no rip-out):** Pori already has the bridge — PR #38
(`fix/native-text-only-answer`) makes "model replied with text and **no** tool
call" become the final answer. So the machinery to accept a plain-text answer
already exists and is tested. If the model keeps calling `answer`, behaviour is
exactly as today → **zero regression**; if it writes text, the prose streams.
The only structural thing to preserve is receipt-backed artifacts (keep `answer`
for file-producing tasks).

## 3. The event model — one normalized stream

A per-turn **accumulator** consumes provider deltas as blocks
(`start → delta(s) → stop`) and emits **normalized `PoriEvent`s**. Everything
downstream (renderers, logger) only ever sees this one shape.

```
Provider stream → Provider Adapter → PoriEvent → accumulator/renderers/JSONL
```

Event types (illustrative):

```
run_start | step_start | llm_start
text_delta        # visible answer prose, streamed
thinking_delta    # reasoning prose, streamed (may be dimmed/collapsed)
tool_call_start   # emitted the INSTANT the tool name is known (before args finish)
tool_call_end     # after execution: success + result summary
llm_retry         # API is retrying/rate-limited (NOT "still thinking")
step_end | run_end
```

Accumulator construction rule (per block):

| block | on delta | on stop |
|---|---|---|
| **text** | emit `text_delta`, render live | — |
| **thinking** | emit `thinking_delta`, render live | — |
| **tool_use** | **buffer silently** (partial JSON isn't parseable) | parse JSON, emit `tool_call_end` |

Parsing tool JSON **only at stop** (never mid-stream) also fixes the large-payload
(HTML) truncation bug. The `tool_call_start` fires at the block start — before
args finish — which is what makes tool calls feel instant.

## 4. The provider adapter layer (reasoning is not portable)

Providers disagree on reasoning, in three tiers. Pori is provider-agnostic, so a
thin adapter normalizes each into the same `PoriEvent` shape. Each provider/model
declares a `reasoning_mode`:

| `reasoning_mode` | Provider behaviour | Adapter action |
|---|---|---|
| `native` | separate thinking channel (e.g. Anthropic extended thinking) | map `thinking_delta` straight through |
| `tagged` | reasoning inline as `<think>…</think>` in the text stream (DeepSeek-R1 style; maybe Kimi per config) | run a streaming scrubber: split on the tags, emit `thinking_delta` for the tagged span, `text_delta` for the rest (tag convention is per-provider) |
| `none` | no reasoning trace | emit `text_delta` for everything; no thinking row |

**Why tiers don't break the prose/data split:** the split holds regardless of
tier. The tier only decides whether streamed prose gets **sub-split** into
thinking vs answer — not whether it streams. Worst case (`none`): no
dimmed/collapsible thinking UI for that model, but you still get live token
streaming of the answer — the main win.

Define the adapter **interface first** (the `reasoning_mode` enum + the three
code paths), so adding a provider later is a config entry + (if `tagged`) a
regex, not new plumbing. **Action item:** verify the actual `reasoning_mode` of
each provider we support (Fireworks/Kimi, Anthropic, OpenAI, OpenRouter) and the
tag convention for any `tagged` ones before locking the shape.

## 5. The label layer (start/done)

A dumb, fast lookup (no LLM) that turns a tool call into a human line, with
**start** and **done** states:

```
→ Creating why-agents-feel-alive.md…      (on tool_call_start)
✓ Created why-agents-feel-alive.md         (on tool_call_end)
```

Pori already has the seed: `pori/observability/tool_preview.py::build_tool_preview`
(start-style only). Extend it to start/done, ideally as an optional
`label_template` on `ToolInfo` so all ~35 tools get sensible defaults and new
tools opt in. Start-labels must tolerate missing args (args stream in after the
block starts) — key off the tool name, get specific once parsed.

## 6. Renderers (all off the one stream)

- **CLI clean (end-user):** `text_delta` streams; `→/✓` tool lines; no `Step N`.
- **CLI verbose (dev, `PORI_VERBOSE=1`):** full event/log detail.
- **Cloud SSE:** map `PoriEvent` → SSE (`tool`, `step`, `message` already exist).
- **JSONL sink (`.pori/events.jsonl`):** the same stream logged for replay/audit
  — a diff-able trail for the validator false-positive work.

## 7. Kill the ambiguous gap

Emit `llm_retry` from `retry_async` + apply an explicit per-provider request
timeout, so a pause renders as "thinking…" vs "rate-limited, retrying in Ns"
instead of unreadable silence.

## 8. What Pori already has (don't rebuild)

- **Text-as-answer bridge** — PR #38 (`fix/native-text-only-answer`). The safety
  net for §2.
- **`build_tool_preview`** — the label-table seed (§5).
- **Skill-as-data** — `_render_available_skills_prompt` injects a name+desc index;
  `skill_view` loads on demand. Matches Claude Code's "skills are files" pattern.
- **Text streaming A.1–A.3** (`feat/llm-streaming`) — streams `content` deltas,
  but with no accumulator/typed-events and no prose/data split, so answers (tool
  args) still don't stream. This blueprint replaces the ad-hoc callbacks.
- **Clean CLI + `PORI_VERBOSE`** — the two renderer modes exist; they need to
  consume events instead of `print()`.

## 9. Phasing

- **P1 — accumulator + normalized events + adapter interface.** Per-block
  start/delta/stop; parse tool JSON only at stop; emit `PoriEvent`s; define the
  `reasoning_mode` enum with the three paths (implement `none` + `tagged` first —
  Kimi). *Fixes the truncation bug as a side effect.*
- **P2 — prose/data split + labels.** Prompt nudge: write direct answers (and
  reasoning) as text; call `answer` only to attach receipts. Start/done labels on
  `tool_call_start`/`end`.
- **P3 — renderers.** CLI clean/verbose consume events; add the JSONL sink;
  Cloud SSE maps from the same events.
- **P4 — reasoning tiers.** Streaming `<think>` scrubber for `tagged`; passthrough
  for `native`; verify each provider's mode.
- **P5 — kill the gap.** `llm_retry` event + request timeout.
- **P6 (deferred) — lazy tool-loading.** ~28k tokens/step today (all 35 schemas
  sent every call). Send a task-relevant subset via capability groups. Independent
  cost/latency win.
- **P7 (deferred) — grouping/collapse.** Group tool calls under phase headers,
  collapse verbose output ("Edited X.tsx ⌄"). Pure presentation, last.

## 10. Risks & mitigations

- **Reliability of prose-as-text** — depends on the model choosing text over
  `answer` (prompt-tuning; Kimi has a habit). *Mitigation:* the PR #38 bridge
  means non-cooperation degrades to today's behaviour, never to broken.
- **Losing artifact receipts** — *Mitigation:* keep the `answer` tool for
  file/artifact tasks; prose-only tasks use text.
- **Reasoning tier detection wrong** — *Mitigation:* default any unknown model to
  `none` (still streams the answer); upgrade per verified provider.
- **`text = done` ambiguity** — a text-only turn is treated as the final answer;
  fine for native tool-calling (models act via tools), bounded by PR #38.

## 11. Open decisions (settle before P1 locks)

1. Confirm each provider's `reasoning_mode` + tag convention (Anthropic, OpenAI,
   OpenRouter). **Verified: Kimi K2 on Fireworks = `native`** — it streams
   reasoning in the OpenAI-style `reasoning_content` delta field (no `<think>`
   tags). `config.yaml` sets `reasoning_mode: native` for it.
2. Prompt wording for the prose/data split (how hard to push "answer as text").
3. Fold **P6 lazy-tools** in now or defer (recommended: defer; it's independent).
