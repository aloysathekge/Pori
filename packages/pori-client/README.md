# `@pori/client` — Aloy transport

The single typed client for the Aloy backend. The **web** (`products/aloy/app`) and
**desktop** (`products/aloy/desktop`) surfaces both import it, so the wire protocol lives
in exactly one place.

**Owns:**

- **`AloyClient`** — REST + SSE client for the `/v1` surface: `submitTask`,
  `streamTask` (SSE over a POST body), `submitClarification` (the `ask_user`
  button bridge), `getTaskStatus`, `getTaskResult`.
- **`PoriEvent` types** — a 1:1 mirror of the kernel's event contract
  (`text_delta`, `thinking_delta`, `tool_call_start/end`, `clarification_request`,
  `run_end`, …).
- **`parseSseStream`** — incremental SSE decoder for the event stream.

```ts
import { AloyClient } from "@pori/client";

const client = new AloyClient({ baseUrl: "http://localhost:8000", apiKey });

await client.streamTask("Compare Postgres and SQLite for our use case", {
  onText: (chunk) => appendAnswer(chunk),
  onThinking: (chunk) => appendThinking(chunk),
  onToolStart: (e) => showToolChip(e.payload),
  onClarification: (req) => renderButtons(req), // answer via submitClarification(req.id, value)
  onRunEnd: () => markDone(),
});
```

## Provenance

Harvested from Hermes `packages/pori-client` (**MIT**) — the package structure and
`tsconfig` were kept; the PTY / JSON-RPC-over-WebSocket gateway was **stripped**
and replaced with a REST + SSE (`PoriEvent`) client targeting the Pori/Aloy
backend. Rebranded `@hermes/shared` → `@pori/client`. See `references/HARVEST.md`.

## Auth

The backend expects the API key in the `X-API-Key` header (`PORI_API_KEY` on the
server). Pass it as `apiKey` to `AloyClient`.
