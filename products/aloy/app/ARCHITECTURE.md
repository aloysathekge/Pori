# Aloy app architecture

The Aloy app is a React 19, Vite, TypeScript, and Tailwind host for the durable
product model. It talks to the backend only through `/v1` REST and SSE and uses
`@pori/client` for shared transport contracts.

## Product shell

- **Today** is the attention lens across Life and active Events.
- **New conversation** creates a transcript-isolated Conversation inside the
  permanent Life Event.
- **New Event** creates a dedicated Event and its one canonical Conversation.
- **Event Workbench** composes Conversation, generated Surface, opened files,
  and Event context without making any pane the source of truth.
- **Connections** and **Schedules** expose user capabilities and durable wakes.
  Legacy Agent configuration remains operator-owned and absent from ordinary
  customer navigation.

## Source layout

- `pages/` owns route-level composition.
- `components/workbench/` owns flexible Event workspace layout.
- `components/surfaces/` owns the generated-app iframe host and bridge.
- `components/chat/` owns Conversation rendering and input.
- `components/layout/` owns the global shell and responsive navigation.
- `components/ui/` contains host-owned primitives.
- `hooks/` owns reusable stateful behavior such as streaming and attachments.
- `api/` contains typed backend wrappers; components do not call `fetch`
  directly.
- `contexts/` contains Supabase-backed auth.

There is no global Redux/Zustand store. Page state stays local, durable truth is
refetched from the backend, and live invalidation arrives through SSE.

## Transport contract

`api/client.ts` attaches the Supabase bearer token and exposes JSON, streaming,
and upload transports. Non-success JSON responses become typed `ApiError`s.
`api/sse.ts` handles Pori event frames, CRLF/trailing-buffer correctness,
caller cancellation, reconnect, and an idle watchdog.

The app never treats a transport acknowledgement as completed work. Task and
Surface interaction status is rendered from the durable lifecycle returned by
the host.

## Surface isolation and bridge

A generated Surface runs in a sandboxed iframe and communicates through a
session-bound `MessageChannel`. The host validates protocol version, Event,
publication, capabilities, command schemas, and interaction identity. The
Surface cannot call the API directly, read host files, hold provider tokens, or
execute a protected action.

Presentation-only changes remain inside the generated app. Durable state,
reasoning, external action, automation, and source-change commands cross the
bridge as distinct typed effects. The host persists the exact interaction and
returns its queued/running/approval/terminal lifecycle to the Surface.

Conversation, Surface, files, and Event context remain peer workspace regions.
Opening or updating one must not erase local state or steal focus in another.

## Responsive and accessibility contract

The global sidebar becomes mobile navigation; Workbench panes collapse into
focused views; context tabs remain operable without overlap; generated
Surfaces fill their pane and adapt internally. Keyboard operation, visible
focus, semantic names, contrast, reduced motion, touch targets, modal focus,
and overflow are release gates at desktop, tablet, and phone widths. Static
source review cannot substitute for signed-in interaction evidence.

## Verification

TypeScript uses strict checking, including unused and unchecked-index gates.
From this directory run:

```bash
npm run test
npm run lint -- --max-warnings 0
npm run build
```

The unit bridge tests and build checks do not replace real generated-Surface or
manual viewport acceptance.
