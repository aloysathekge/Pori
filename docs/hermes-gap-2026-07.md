# Hermes Gap Analysis — What Pori Is Still Missing (2026-07-07)

_Second full mining pass over `references/hermes-agent` (source-level sweep),
crossed against Pori's current monorepo. Excludes everything already harvested
(the ALIGNMENT.md rows, the marathon/long-running work) — this is the NEW gap
list. Pattern-harvest only; the kernel never pastes Hermes code._

## The headline

After two harvest passes, Pori's **loop-quality** is at parity or better
(durability/resume now exceeds several Hermes behaviors). The remaining gap is
**breadth of surface**: Hermes meets users where they are (~20 chat platforms,
voice, images, browser, integrations) while Pori has one web app and a CLI, and
Pori's message layer literally cannot carry an image. The gaps below are ranked
by leverage toward the Aloy vision (personal + org OS agent).

## Tier 1 — foundations that gate everything else

### 1. Multimodal message plumbing (images first) — KERNEL, prerequisite
`pori/llm/messages.py` types every message `content` as plain `str`. No image /
media content blocks, no attachment plumbing, no upload UI. This single
constraint blocks: vision analysis, screenshots (browser/computer-use),
photo-in-chat (every messaging platform), document images. Hermes routes vision
through a dedicated analyzer + router (`tools/vision_tools.py`,
`agent/vision_routing.py`). **Harvest**: content-block message types in
`llm/messages.py` + provider adapters, then a gated `vision_analyze` tool.

### 2. Messaging gateway + channels — THE Aloy product gap
Hermes: `BasePlatformAdapter` (4 abstract methods: connect/disconnect/send/
get_chat_info, defaulted media/typing/buttons/chunking), ~20 platform plugins
(telegram, slack, discord, whatsapp, email, sms, signal, imessage, teams,
matrix, …), a **DeliveryRouter** (`gateway/delivery.py`) that routes any
outbound (cron results, notifications) to the right platform/chat, and a
dial-out WebSocket **relay** so a hosted gateway needs no public inbound port.
Pori/Aloy: zero channels (grep-verified); the web app is the only surface.
Aloy's own README already names the plan (Telegram personal, Slack org).
**Harvest**: the adapter ABC + platform registry + DeliveryRouter shape into an
Aloy `gateway/` service; start with **Telegram only**. This also unblocks
delivery of cron/background results beyond conversations (Phase 3's last mile)
and is the natural home for Hermes's group-chat semantics (per-user session
lanes in groups, mention gating) when a second user appears.
ALIGNMENT GW-7's "iff multi-surface is a real goal" gate is now satisfied by
the product plan — treat GW-7 as unblocked.

### 3. Provider failover chains + credential pool — resilience for a daily driver
Pori classifies errors (`error_classifier.py`) but only retries the SAME
provider/model; there is no cross-provider failover chain, no multi-credential
rotation/cooldown. Hermes: `agent/credential_pool.py` (persistent pool,
rotation, cooldown), provider profiles declaring fallback models, ~35 provider
plugins. Pori already has the substrate (provider profiles + `FailoverReason`
+ model tiers from delegation). **Harvest**: a failover chain in `create_llm`
consuming the existing classifications; credential pool later.

## Tier 2 — capability breadth (each is a gated tool / provider registry)

### 4. Perception & media provider registries
Hermes's repeated architecture: per-modality **provider registry + routing +
plugins** (image_gen, video_gen, tts, transcription, vision — each an
`agent/*_provider.py` + `*_registry.py` + `plugins/<modality>/`). For Aloy the
order of value is:
- **STT (transcription)** — voice notes are the #1 personal-agent input on
  chat platforms; Hermes auto-transcribes inbound voice on every platform.
- **Vision analyze** (after Tier-1 #1).
- **TTS** — reply-as-voice; local engines exist (Piper/Kitten) for zero-cost.
- Image/video generation — later, product-dependent.

### 5. Document extraction folded into read_file — CHEAP WIN
`tools/read_extract.py`: stdlib-only extraction of `.docx`, `.xlsx`, `.ipynb`
folded into the existing `read_file` tool (no new tool surface — Footprint
Ladder rung 1). Pori's `read_file` is text-only today. Small effort, real
daily-driver value.

### 6. Browser automation — LARGE, choose one backend
Hermes ships four backends (accessibility-tree CLI, anti-detect Camoufox, raw
CDP, CDP supervisor). Pori has none. Start with a single gated tool behind a
capability prerequisite; don't port the zoo.

### 7. Session search (FTS5) — zero-LLM long-term recall
`tools/session_search_tool.py`: SQLite FTS5 over all transcripts with three
inferred modes (discovery/scroll/browse). Complements Pori's semantic archival
memory with exact recall ("what did we decide about X in March"). Pori's
SQLite store makes this a natural extension.

### 8. Blueprints — skills × cron, nearly free now
`tools/blueprints.py`: a shareable automation = a SKILL.md with a `blueprint:`
schedule in its frontmatter — consent-first install ("this skill wants to run
every morning"). Pori now has BOTH halves (skills + the new cron engine);
blueprints are a thin joining layer and a genuinely differentiating UX.

### 9. Large tool-result spill — CHEAP WIN
`tools/tool_result_storage.py`: oversized tool output spills to a sandbox temp
file (with a pointer in the transcript) instead of truncation. Pairs perfectly
with Pori's AC-3 compression; protects long runs from losing big results.

### 10. External memory providers
Hermes ships 8 pluggable backends (honcho, mem0, supermemory, …) behind a
`MemoryProvider` ABC. Pori has the entry-point seam (`pori.memory_stores`) but
zero implementations beyond memory/sqlite. Low urgency — layered knowledge is
Aloy's own moat — but one reference implementation (e.g. a Postgres/pgvector
store for the backend) would prove the seam.

## Tier 3 — operator polish & later bets

- **`pori doctor`** — `diagnose_provider()` already exists internally; expose
  it as a CLI command + env probe (Hermes `hermes_cli/doctor.py`). CHEAP.
- **Insights** — Hermes `agent/insights.py` (token/cost/tool/model breakdowns
  over the state DB). Aloy's UsagePage covers part; the CLI has nothing.
- **Onboarding hints** — `agent/onboarding.py`: contextual one-time first-touch
  tips tracked in config (not a wizard). Cheap polish for CLI + app.
- **MCP client/server** — parked by standing rule, but note two facts for the
  decision: (a) Hermes's MCP client pattern (dynamic `mcp-*` toolsets with
  `check_fn`, vanish on disconnect) drops straight onto Pori's gated-tool
  ladder; (b) the MCP **server** (`mcp_serve.py` exposing channels/sessions as
  tools) would let Claude Code/Cursor drive Aloy — a real differentiator once
  the gateway exists.
- **Computer use** (cua-driver over MCP stdio), **LSP client** (diagnostics/
  definitions for coding tasks), **Microsoft Graph / Google Meet realtime /
  Spotify / Home Assistant** integrations — product-dependent; revisit after
  the gateway ships.
- **Secret sources** — Bitwarden Secrets Manager pull-at-startup
  (`agent/secret_sources/`) as a pluggable alternative to `.env`.

## Non-gaps (deliberate)

- **Composio**: Hermes doesn't use it either (native plugins + MCP). Not a gap.
- **Virtual pet / achievements**: product flavor, not framework capability.
- **Toolset distributions / batch SWE runners**: eval-data infrastructure for
  Nous's model training, orthogonal to Pori.
- **Web search provider zoo** (8 providers): Pori's Tavily-gated `web_search`
  is fine; add a provider registry only when a second provider is actually
  wanted.

## Suggested sequence

1. **Multimodal content blocks** (kernel) — the prerequisite everything
   perceptual sits on.
2. **Telegram gateway slice** (Aloy): adapter ABC + registry + DeliveryRouter,
   one platform, results delivery wired to it.
3. **Cheap-wins batch**: read_extract into `read_file`, tool-result spill,
   `pori doctor`, blueprints (skills×cron).
4. **Provider failover chain** on the existing classifier.
5. **STT + vision registries** (needs 1).
6. Then by product pull: session search, browser (one backend), memory
   provider reference impl, MCP decision.
