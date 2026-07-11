# Aloy app — frontend architecture

React 19 + Vite + Tailwind v4 SPA (`@aloy/app`), talking to the Aloy backend's
`/v1` HTTP+SSE API. Shared event constants/types come from `@pori/client`
(workspace package).

## Layout (`src/`)
- `pages/` — one component per route (13 routes in `App.tsx`): ChatPage plus
  resource pages (AgentConfigs, Teams, Skills, Memory, Files, Schedules,
  Traces, Usage, Connections, Settings) and Login/Signup.
- `components/` — `chat/` (Composer, MessageList/MessageBubble, Markdown,
  ArtifactDrawer, ClarifyPrompt, RunReplay, StreamingIndicator, ...),
  `layout/AppLayout.tsx`, `settings/`, and `ui/` (small in-house primitives:
  Button, Card, Modal, ... — no component library).
- `hooks/` — the three chat hooks (below).
- `api/` — one module per backend resource; all go through `client.ts`.
- `contexts/` — auth only (`AuthContext.tsx` + `useAuth`), Supabase-backed.
- `types/index.ts` — app-side response/wire types; `lib/` — tiny utilities.

## State conventions (deliberate)
No global store (no Redux/Zustand). State is local `useState` owned by the
page that renders it, with reusable stateful logic extracted into hooks. The
only context is auth. Cross-page data is refetched, not cached globally.

## API layer contract
`api/client.ts` owns `BASE_URL` (`VITE_API_BASE_URL`), attaches the Supabase
bearer token via an injected token getter (`setTokenGetter`, wired by auth),
and exposes three transports:
- `apiFetch<T>` — JSON in/out; non-2xx throws `ApiError { status, message }`.
- `apiStreamFetch` — returns the raw `Response` for SSE consumption.
- `apiUploadFile` — multipart via XHR so upload progress is observable.

Every other `api/*.ts` module is a typed wrapper over these — components
never call `fetch` directly, and error handling can always `instanceof
ApiError`.

## The three chat hooks (`src/hooks/`)
- `useConversations` — the conversation list + active selection. Page-
  agnostic: routing (navigate-on-create/delete) stays with the caller.
- `useStreamingRun` — one streaming run: the SSE lifecycle (send → frames →
  final message), streaming text/thinking/tool/plan state, clarify
  (`ClarifyState`) pause/resume, stop, and re-attach to a live run. Appends
  into the caller's message list via a passed `setMessages`.
- `useAttachments` — the composer's pending images/files and the attachment
  ladder (inline text → native doc → durable upload to object storage), with
  client-side caps mirroring the backend's.

ChatPage composes all three; the hooks own behavior, the page owns routing
and layout.

## SSE streaming model
`api/sse.ts` consumes the backend's kernel `PoriEvent` stream (see
`aloy_backend/streaming.py`): `text_delta` / `thinking_delta` /
`tool_call_start|end` / step frames / `clarification_request`, then a flat
final `message` frame. Consumers pass an `SSECallbacks` object; the reader
enforces a 90s idle timeout (server keepalives ~15s, so silence means a
stalled stream) and supports aborting via `AbortSignal`. Clarify answers go
back over plain POST (`submitClarification`); `attachLiveRun` re-joins an
in-flight run after reload.

## Gates
TypeScript is maximally strict: `strict`, `noUnusedLocals`,
`noUnusedParameters`, `noUncheckedIndexedAccess` (tsconfig.app.json). CI
(`.github/workflows/ci.yml`, "Aloy app" job) runs `tsc -b`, then
`eslint . --max-warnings 0` — warnings fail the build — then `vite build`.
Keep changes warning-clean, not just error-clean.
