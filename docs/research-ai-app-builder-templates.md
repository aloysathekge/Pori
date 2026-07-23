# How AI app builders make models produce working React apps

_Research report, 2026-07-23. Sources: leaked/official system prompts
(x1xhlol/system-prompts-and-models-of-ai-tools, ~137k stars; the open-sourced
`stackblitz/bolt.new` repo), engineering blogs from Replit, Vercel (v0),
StackBlitz, Lovable, and Manus. Compiled to inform
[`aloy-baseline-surface-spec.md`](./aloy-baseline-surface-spec.md). Claims
below survived adversarial verification except where caveated._

## 1. Headline answer

**No serious product lets the model scaffold a project from scratch.** Every
mature builder pins a curated, pre-wired stack and has the model *modify* it:

- **v0 (Vercel):** fixed Next.js App Router + shadcn/ui + Tailwind + lucide
  scaffold. The model is *prohibited from regenerating baseline files*
  (`app/layout.tsx`, `globals.css`, all `components/ui/*`, `lib/utils.ts`,
  configs) — they exist, are assumed, and are never emitted.
- **Lovable:** hard-codes React + Vite + TypeScript + Tailwind + shadcn in the
  system prompt and declares Angular/Vue/Svelte/Next/native *unsupported*.
  The model never chooses a stack.
- **Bolt (StackBlitz):** historically a "blank canvas" (see §5 caveat), but
  starter templates fork with **zero LLM tokens** — the template is
  infrastructure, not generation. Modern Bolt ingests design systems once
  (~53 min autonomous run) so each subsequent prototype takes ~5 min.
- **Replit Agent:** curated environment + multi-agent loop; the model edits
  and a dedicated verifier exercises the running app.

The template is where these products encode quality: design tokens instead of
raw colors (Lovable bans hardcoded Tailwind color utilities and inline styles
outright, machine-checked), mandatory responsive design, fixed icon/dependency
allow-lists, semantic tokens in a known CSS file. The model inherits the
patterns; prompts only have to say "preserve them."

## 2. Edit formats (how the model changes code)

The industry split, in order of adoption:

| Product | Format |
| --- | --- |
| Lovable (current) | targeted **search-replace** edits, batched operations |
| v0 (current) | **lazy partial rewrite** ("QuickEdit") with the literal marker `// ... existing code ...` for unchanged regions; full-file only on request; structured CodeProject container with one stable project id across turns |
| Replit (repair model) | **numbered line diffs with sentinel tokens** — chosen over unified diff because explicit line numbers stopped diff-application hallucination |
| Bolt (2024 OSS prompt) | **full-file rewrites**, ellipsis placeholders forbidden, XML `boltArtifact/boltAction` protocol parsed deterministically |

Common invariants regardless of format: the model must **read before
editing** (v0's SearchRepo/ReadFile are mandatory pre-edit), output is a
**machine-parseable protocol** (never free-form code blocks), and file
operations are explicit typed actions (delete/move as dedicated ops).

## 3. Verification and repair loops

- **v0 "autofixers":** deterministic repair that runs *while the model output
  streams* — fixing wrong imports and common JSX/TS errors mid-stream — plus
  post-stream AST-based multi-file fixes and automatic `package.json`
  dependency completion by scanning imports. Engineered to run <250 ms.
  Lesson: burn deterministic compute before model tokens on mechanical
  errors.
- **Replit's verifier:** a separate agent that interacts with the *running*
  application. Their named failure mode is the **"Potemkin interface"** — UI
  that renders beautifully with nothing wired up; only behavioral testing
  catches it. Agent 3 dropped screenshot/computer-use testing for **injected
  Playwright helpers in a persistent REPL** (stripped DOM + ARIA labels),
  which extended autonomous productive work ~10× (20 → 200+ min) at a median
  $0.20 per verification session.
- **Isolation:** Replit runs testing in a **subagent so the verification
  transcript never pollutes the builder's context** — the builder sends a
  test plan, gets back only a summary.
- **Lovable:** console-log and network-request reading tools the model must
  consult before touching code, plus a silent post-wiring verification pass
  and machine-checked design-system violation scans.
- **Replit repair economics:** a fine-tuned 7B model (DeepSeek-Coder base)
  applies fixes from LSP diagnostics — a small model because repair is
  latency/cost-bound; real-world repair scored notably below benchmark
  repair, so gate claims on real evals.

## 4. Token and latency economics

- **Prompt caching is the business model.** Manus calls KV-cache hit rate the
  single most important production metric (10× price gap cached vs uncached
  input on Claude). Bolt reports ~90% cache efficiency on its Claude Agent
  SDK workflows as its primary cost lever.
- Cache discipline: **byte-stable prompt prefix** (one changed token — even a
  timestamp — invalidates everything after), **append-only context**, and a
  **fixed tool set** for the whole run (Manus masks logits instead of
  adding/removing tools mid-run).
- v0 keeps its knowledge injection *identical across requests* specifically
  for cache hits, and injects curated code samples selected by embeddings
  instead of doing generation-time web search.
- **Keep failures in context:** Manus found leaving error traces (failed
  builds, stack traces) in context measurably improves behavior — the model
  stops repeating mistakes. Do not scrub errors.
- Replit: reliability degrades sharply past ~50 agent steps; they frontload
  important work and compress memory between sub-agents.

## 5. Caveats and gaps

- The famous Bolt "full-file rewrites, no diffs" prompt is the **October-2024
  open-sourced `stackblitz/bolt.new` version**; adversarial verification
  refuted labeling it *current production* behavior. Treat as historical
  evidence of a starting point the industry (including Bolt) moved past.
- **Rork:** essentially no engineering material surfaced publicly (an
  Expo/React-Native builder; its template-first UX is visible in the product
  but undocumented). No load-bearing claims included.
- Leaked prompts describe *what the model is told*, not the full server
  architecture; blogs fill some of that gap for Replit/v0/StackBlitz.

## 6. Implications for the Aloy baseline Surface

Validations of the existing spec:

1. **Warmed template, zero-token delivery** — exactly Bolt's zero-token
   template fork; universal industry pattern. Aloy's host-owned toolchain
   makes its template *smaller* than anyone's (no configs in model space).
2. **`replace_text` + full-file `write_file` + typed delete** — matches the
   modern edit-format consensus (Lovable search-replace / v0 QuickEdit) and
   Aloy already requires exactly-once matches, which addresses the
   hallucinated-diff problem Replit solved with line numbers.
3. **Trusted browser gate with behavioral primary-job proofs** — this is
   Replit's Potemkin-interface answer, and Aloy already runs it host-side
   with the results returned as one compact bundle (their subagent-isolation
   lesson, already embodied).
4. **Design tokens + machine-checked source rules** — Lovable's approach;
   Aloy's `theme.css` tokens + `validate_surface_source` forbidden patterns
   are the same shape.

Adjustments worth adopting:

1. **Protect the baseline primitives like v0 protects its scaffold.** Mark
   `primitives.tsx` / `theme.css` as preserve-don't-rewrite in the skill (or
   enforce write-protection in the workspace for a designated set), so edits
   anchor to stable code and `replace_text` matches stay reliable.
2. **Deterministic autofixers before paid repairs.** Pre-gate normalization
   (import fixes, missing-manifest-field completion, token-violation
   auto-substitution) in the workspace `check()` — v0 proves a <250 ms
   deterministic pass absorbs most mechanical failures before any model turn.
3. **Cache discipline in the Builder loop.** Byte-stable system
   prompt + projection prefix, append-only history, fixed tool schemas
   (already true), and cache-aware budget accounting — then the aggregate
   token cap becomes meaningful on cached providers and the uncached-provider
   penalty is an explicit, measured cost.
4. **Keep rejected-source diagnostics verbatim in the loop context** (Manus):
   the compact-bundle design should compress, not sanitize — error text is
   training signal in-context.
5. **A future lightweight repair tier**: mechanical TS/manifest fixes don't
   need the frontier Builder model; a fast small model (or pure AST fixes)
   mirrors Replit's 7B repair economics. Rung-3 (gated) capability, later.
