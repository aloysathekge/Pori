# Pori MCP Integration — spec (kernel MCP client)

_Design spec (2026-07-08). SPEC ONLY — no code. Adds an MCP **client** to the
Pori kernel so any agent can use external MCP tool servers. Reverses the earlier
"MCP parked" stance by explicit decision. The design is **session-scoped, not
process-global** — which is what makes it multi-tenant-safe by construction, so
Aloy can layer per-user servers on top without the kernel knowing about tenancy._

_Sources: `references/hermes-agent-deep-dives/mcp-integration.md` (Hermes client),
and Ken Huang "Chapter 13: MCP Integration" (the Claude-Code per-session model)._

## The decision that shapes everything: session-scoped, not global

Hermes runs ONE process-global background loop with connections shared across all
conversations — great for a single-user CLI, **fatal for multi-tenant** (one
tenant's servers/tokens would bleed to another). Claude Code instead instantiates
MCP clients **per session** and passes them into the engine. **Pori adopts the
Claude-Code model:**

- An MCP client set is created **per run/session**, connects the run's servers,
  registers their tools into **that run's tool registry** (never the global
  singleton), and tears down at run end.
- Because connections and tools are scoped to the run, **no cross-tenant bleed is
  possible** — the kernel never holds a shared MCP state. Aloy just supplies a
  different server list (with the user's resolved tokens) per run.

## Principle: MCP obeys the Footprint Ladder (Pori's improvement over Hermes)

Hermes registers MCP tools with **no gating or approval flag** — they run like any
built-in. That violates Pori's whole discipline. In Pori:

- MCP is a **gated capability** (rung 4): the `mcp` client needs the optional
  `pori[mcp]` extra (the official `mcp` SDK). No SDK / no servers → zero surface,
  zero cost. Importing `pori` never requires `mcp`.
- MCP tools register into a **capability group** (`mcp`, or per-server
  `mcp:{server}`) so a run can include/exclude them like any group.
- **Write/dangerous MCP tools are HITL-gatable by name** (the deployment's
  `hitl.interrupt_on`), exactly as Aloy's `gmail_send` is. The registration
  boundary is where Pori interposes what Hermes omits.
- `check_fn` liveness so tools vanish when their server disconnects.

## Kernel architecture (`pori/mcp/`)

### Data / config
- `McpServerConfig`: `name`, `transport` ("stdio"|"http"|"sse"), `url` or
  `command`+`args`+`env`, `auth` (none | bearer-token | headers), `timeout`,
  `keepalive`, `tools_include`/`tools_exclude`. **Auth is a resolved token/header
  passed IN** — the kernel does not own OAuth (that's the product's job; same as
  Gmail tools receive an injected token). Static-token = a Bearer header.
- Built on the official **`mcp` Python SDK** (optional; lazy import; no-op when
  absent).

### Session lifecycle (`McpSessionSet`)
- Constructed with a list of `McpServerConfig` for one run. `connect_all()`
  connects (remote in parallel), `initialize()`, `list_tools()`, registers tools
  into the run's registry; `close_all()` tears down at run end.
- **Sync/async bridge:** kernel tools run sync; MCP is async. A per-set background
  loop (or thread pool) marshals calls via `run_coroutine_threadsafe` — the same
  pattern already used for the E2B sandbox and browser bridges. NOT a global
  daemon; owned by the set, dies with the run.
- **Resilience (harvest from Hermes):** reconnect with exponential backoff,
  per-server timeout, a circuit breaker, and `ping` keepalive (never `list_tools`
  — 1MB/cycle on big servers). `tools/list_changed` → refresh + re-register.

### Tool surfacing
- Namespacing: **`mcp__{server}__{tool}`** (double underscore — Claude Code
  convention; makes provenance visible in traces/receipts). Collisions with a
  built-in → skip the MCP tool (preserve the built-in).
- **Schema translation** (harvest Hermes `_normalize_mcp_input_schema`): the
  single most reusable, provider-portability-critical piece — `#/definitions`→
  `#/$defs`, coerce `type:object`, prune dangling `required`, collapse nullable
  unions. Makes any server's schemas work across Anthropic/OpenAI/Google.
- Tool call → `session.call_tool`; text/image/structured results normalized;
  errors credential-sanitized before they reach the model (harvest
  `_sanitize_error`).

### Wiring into the run
- `Agent` / `Orchestrator.execute_task` accept `mcp_servers: list[McpServerConfig]`.
  The Orchestrator builds a **per-run registry** (a copy of the base, NOT the
  global singleton), connects the servers, registers their tools into it, and the
  Agent snapshots that. Teardown on run completion. Session-scoped = tenant-safe.
- Delegation: a sub-agent can inherit or be scoped to a subset of servers
  (Claude-Code `requiredMcpServers` idea) — later.

### Kernel config (single-user / CLI)
- `config.yaml` `mcp.servers.<name>` (transport/url/command/env/auth/timeout),
  mirroring the durable-CLI story. The CLI connects them per session.
- Optional extra: `pori[mcp]` → the `mcp` SDK.

## Aloy multi-tenant layer (separate — sketched, its own spec later)

The kernel client is tenancy-blind; Aloy makes it multi-tenant by **supplying a
per-user server list per run** — nothing more:
- Aloy stores per-user `McpServer` rows (name, transport, url, auth ref) and, for
  OAuth servers, **reuses the connect-engine** (`OAuthConnection`, encrypted
  tokens, the web redirect) — an MCP server is "just another provider."
- At run-setup (exactly where Gmail tokens are resolved today), Aloy resolves the
  user's enabled servers + fresh tokens and passes them as `mcp_servers` into the
  orchestrator. Per-user gating falls out for free.
- **Transports for hosted:** remote **HTTP/SSE first** (per-user, feasible now).
  **stdio servers run inside the user's E2B sandbox** (never as a shared-host
  subprocess) — later, ties into the sandbox work.
- Because the kernel is session-scoped, there is no shared MCP state to leak.
- (Full Aloy multi-tenant design — server CRUD, catalog, connection pooling for
  latency — is a follow-up spec: "we'll figure the tenant way".)

## Context cost (must respect the Footprint Ladder)
Every registered tool's schema ships on every LLM call. So: per-user/per-run
server enablement, `tools_include`/`exclude` filtering, and gating so unconnected
servers cost nothing. A user with 5 MCP servers × 20 tools each is 100 tool
schemas — real budget. Enablement + filtering are not optional.

## Phasing
1. **Kernel MCP client, remote HTTP/SSE only**, session-scoped, gated (`pori[mcp]`),
   with the Footprint-Ladder registration hook + schema translation + resilience.
   Harvest: transports, `MCPServerTask` shape, schema translator, error sanitize.
2. **Kernel stdio transport** (for CLI/local).
3. **Aloy: per-user server config + OAuth via the connect-engine + run-setup
   injection** (mirrors Gmail). Remote servers.
4. **Aloy: stdio servers inside the E2B sandbox.**
5. **(Skip unless needed) MCP *server* side** — expose Pori/Aloy over MCP for
   Claude Code/Cursor. Self-contained, low code reuse; only if IDE consumption is
   a goal.

## Open decisions
1. **Kernel vs extension:** does the MCP client live in `pori/mcp/` (in-kernel,
   gated) or an `extensions/pori-mcp` package? (Lean: in-kernel gated — it's
   generic agent runtime and the user wants it in the kernel; the extra keeps it
   zero-cost when unused.)
2. **Per-run connect vs per-session pool:** connect servers every run (simplest,
   matches Gmail) or keep a per-user/session pool warm for latency? (Lean: per-run
   for v1; pool later if remote-connect latency hurts.)
3. **Sampling / elicitation:** support server→LLM sampling and server-initiated
   elicitation (route to HITL) now, or defer? (Lean: defer; tools-only v1.)
4. **Namespacing depth** when the same server is configured by multiple users in
   one process — moot given session-scoping, but confirm.
5. **Roots / resources / prompts:** v1 = tools only; add resource/prompt utility
   tools (capability-gated) later.
