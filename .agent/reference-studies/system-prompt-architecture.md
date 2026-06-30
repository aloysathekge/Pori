# Reference Study: System-Prompt Architecture Redesign

Source studied: `references/hermes-agent/agent/system_prompt.py` (Hermes's
3-tier prompt assembler) and `references/claude-code` (layered context +
system-reminders), against Pori's `pori/prompts/system/agent_core.md`.

Status: **proposal / design** — no prompt/runtime code is changed by this
document. Sequenced after the Phase 1 planning work (`planning-architecture.md`).

License note: principles and interfaces only; nothing is copied from Hermes or
Claude Code source.

---

## 1. Goal

Adopt the best of both reference harnesses **without losing Pori's identity**:

- from **Hermes** — a **cache-tiered prompt assembler** (stable → context →
  volatile) and a **`SOUL.md` persona** as the identity source;
- from **Claude Code** — **layered project context** (`AGENTS.md`/`CLAUDE.md`)
  and a **system-reminder channel** that re-injects volatile state each turn;

and, as the headline follow-on, migrate from Pori's hand-parsed **JSON envelope**
to **native provider tool-calling**.

Sequencing (agreed): **Phase A — prompt architecture first; Phase B — native
tool-calling.**

## 2. Where Pori is today

`agent_core.md` is one static Markdown file with a `{tool_descriptions}` slot,
assembled by ad-hoc string concatenation in `pori/agent.py` (`__init__`,
≈ lines 311–334): base template → optional custom prompt → "# Available Skills"
→ "# Selected Skills". Memory is fenced as `<memory-context>` and the current
task is repeated as the final user message.

Limitations vs the references:
- **Not cache-ordered** — volatile additions (skills, memory) are mixed into the
  prompt, so the cacheable prefix shifts between runs/turns.
- **Identity is one inline line** — no persona, not user-overridable.
- **No project-context discovery** — `AGENTS.md`/`CLAUDE.md` aren't loaded into
  the prompt.
- **No reminder channel** — the live plan (`PlanStore`) isn't re-injected each
  turn; it sits in one static block.
- **Mandated JSON envelope** — robust-but-brittle; needs structured-output
  recovery.

---

## 3. Target architecture

### 3.1 Three cache-ordered tiers (Hermes)

A new assembler (`pori/prompts/assembler.py`) returns the prompt as ordered
tiers joined `\n\n`, built **stable → context → volatile** so the prefix cache
stays warm:

- **stable** (built once per session; rarely changes)
  - **Default identity** — the neutral, baked-in "who am I" block (see 3.3).
  - Core **workflow + operating-rule blocks** (today's `agent_core.md`, minus the
    volatile and envelope-specific parts; behavior lives here, not in SOUL.md).
  - **Tool guidance** + `{tool_descriptions}` (fixed once the run's tool surface
    is resolved).
  - **Available-skills index** (metadata only).
- **context** (semi-stable; caller/project)
  - The caller's `system_prompt` (`_custom_system_prompt`).
  - **Project context files** discovered under CWD: `AGENTS.md`, `CLAUDE.md`,
    `.cursorrules` (Hermes/Claude Code pattern). Bounded + injection-scanned like
    skills.
- **volatile** (rebuilt every turn; placed last)
  - **SOUL.md** user persona (hot-reloaded per turn; empty by default — see 3.3).
  - `<memory-context>` snapshot.
  - The **live plan/todo** rendered from `PlanStore` (active items only).
  - **Selected-skill** full instructions.
  - Timestamp / session / model-provider line.

Build the stable+context tiers once and reuse; rebuild only the volatile tier
per turn. (Hermes rebuilds the whole prompt only on context compression.)

### 3.2 System-reminder channel (Claude Code)

Generalize Pori's existing memory-fencing + task-repetition into a single
**per-turn reminder channel**: ephemeral `<system-reminder>` blocks carried in a
dedicated message *below* the cached system prompt, not baked into it. It carries
the volatile tier's turn-specific items — most importantly the **live plan**
(re-injected each turn so the model always sees current todo state, completed
items dropped), plus truncation warnings and mode notes. This is the robust home
for the plan the Phase 1 work introduced, and it keeps the cached prefix stable.

### 3.3 Identity & SOUL.md (Hermes-faithful)

Hermes makes a clean three-way split, and we mirror it exactly (verified against
`references/hermes-agent/docker/SOUL.md` and `agent/prompt_builder.py:123`
`DEFAULT_AGENT_IDENTITY`):

1. **Default identity — neutral, baked in code/prompt.** A concise, professional
   "who am I" block that is the fallback when no SOUL.md content is present. It is
   **not** quirky or opinionated. Lives first in the stable tier.

   Proposed Pori default (refines today's one-liner):

   > *"I am Pori, an open-source AI agent. I am helpful, knowledgeable, and
   > direct. I assist with answering questions, writing and editing code,
   > analyzing information, and executing actions through my tools. I communicate
   > clearly, admit uncertainty when appropriate, and prioritize being genuinely
   > useful over being verbose. I am targeted and efficient in exploration."*

2. **SOUL.md — ships EMPTY, user-owned, hot-reloaded.** A template
   (`pori/prompts/system/SOUL.md`) containing only guidance comments + examples,
   like Hermes's. The **user** supplies any personality; Pori imposes none.
   Re-read on each turn (no restart) and resolved override-first: project
   `./SOUL.md` → `config.agent.soul_path` → shipped (empty) template. When empty,
   the default identity (1) is used. Because it is re-read per turn it belongs in
   the **volatile** tier (or reminder channel), not the cached stable prefix.

3. **Operating rules — behavior, kept OUT of the persona.** Pori's behavioral
   commitments (tool-grounding: never invent tool-retrievable facts;
   receipt-honest artifacts; plan-before-acting via `update_plan`; clarify only
   when truly blocked) live in named operating-rule blocks in the **stable** tier
   — analogous to Hermes's `MEMORY_GUIDANCE`/`SKILLS_GUIDANCE`/etc. They hold
   **regardless** of what a user writes in SOUL.md.

This keeps "Pori's identity" as a stable, neutral default and behavioral contract,
while giving users the same persona customization Hermes users rely on.

SOUL.md is distinct from and layered above `AGENTS.md`: **SOUL = personality/tone
(user-owned)**; **AGENTS = the project's rules/context (context tier)**; **default
identity + operating rules = the framework's fixed contract (stable tier)**.

---

## 4. Phase B — native tool-calling (headline follow-on)

After the prompt architecture lands:

- Provider wrappers (`pori/llm/anthropic.py`, `openai.py`, `google.py`) emit tool
  **schemas** and read back native tool calls (`tool_use` / `tool_calls` /
  `functionCall`).
- `pori/llm/messages.py` gains `tool_use` / `tool_result` message types.
- The agent loop consumes native tool calls instead of parsing
  `AgentOutput.action`; the evaluator adapts.
- Gate behind `config.llm.tool_calling: native | envelope` for A/B and rollback.
- Once native is stable: **delete** the "JSON Output Format" + JSON `CRITICAL
  RULES` sections from the prompt and retire the structured-output recovery path.

Native tool-calling removes the largest source of malformed output and is what
every reference harness uses; it is sequenced second only because it touches the
LLM spine, not the prompt.

---

## 5. Mapping to existing Pori code

| Concern | Today | After redesign |
| --- | --- | --- |
| Prompt build | string concat in `agent.py __init__` | `prompts/assembler.py` (tiered) |
| Identity | one inline line | neutral default (stable) + empty user `SOUL.md` (volatile, hot-reloaded) |
| Project context | not loaded | `AGENTS.md`/`CLAUDE.md`/`.cursorrules` (context tier) |
| Plan in prompt | one static block | live re-inject via reminder channel |
| Memory | `<memory-context>` fence | volatile tier / reminder channel |
| Caching | unstructured | stable→context→volatile, prefix-stable |
| Output contract | JSON envelope | native tool-calling (Phase B) |

### Non-goals
- No change to `AgentMemory`, `PlanStore`, or tool semantics — only how the prompt
  is assembled and how volatile state is surfaced.
- SOUL.md is user persona/tone only; behavioral rules stay in the operating-rule
  blocks, and project rules stay in `AGENTS.md`. SOUL.md ships empty.
- Phase A keeps the JSON envelope; only Phase B removes it.

---

## 6. Phasing & verification

- **Phase A.1** — `assembler.py` with the three tiers; move `agent_core.md` content
  into the stable default-identity + operating-rule blocks; keep behaviour
  identical. Tests: prompt contains identity/workflow/tools; tier order.
- **Phase A.2** — reminder channel + volatile tier: hot-reloaded `SOUL.md`
  (empty template + override-first resolution), re-inject the live `PlanStore`
  each turn, move memory + selected skills into volatile. Tests: empty SOUL.md
  falls back to default identity; project SOUL.md overrides; plan re-injected with
  completed items dropped; stable prefix unchanged across turns.
- **Phase A.3** — project-context discovery (`AGENTS.md`/`CLAUDE.md`).
- **Phase B** — native tool-calling behind a flag; then remove envelope.
- Standard gates each phase: `uv run pytest`, `black`, `isort`, `mypy`.
