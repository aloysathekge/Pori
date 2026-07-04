# `@aloy/app` — Aloy web app

The Aloy web SPA. **Adopted from the existing `pori_cloud_client`** (our own
Vite/React frontend) rather than rebuilt — it already ships the screens we need:
Chat, **Memory**, **Teams**, **Traces**, **Usage** (charts), AgentConfigs,
Settings, and Login/Signup. Rebranded to Aloy; talks to the backend over
REST + SSE.

## Stack

React 19 · Vite · TypeScript · Tailwind · react-router · lucide-react · recharts ·
`@supabase/supabase-js` (auth). A 12-module API layer (`src/api/*`) with an
`sse.ts` streaming client.

## Run

```bash
cd products/aloy/app
npm install
cp .env.example .env.local   # Supabase auth + backend URL
npm run dev
```

## Migration status (docs/Aloy.md — "adopt pori_cloud_client, unify on PoriEvent")

- [x] **Stage 1** — copy `pori_cloud_client` → `products/aloy/app`, rebrand identity to Aloy, builds clean (`tsc -b && vite build`).
- [x] **Stage 2** — `src/api/sse.ts` now consumes the kernel **`PoriEvent`** stream via `@pori/client` (alias wired in `vite.config`/`tsconfig`): `text_delta` streams **live** into the assistant bubble, `tool_call_start/end` drive tool chips, `run_end`/final `message` finalize. Supabase **Bearer** auth kept (`apiStreamFetch`). `onClarification` handler is ready for backend 3.3b. Builds clean.
- [x] **Stage 3.1** — backend adopted (`products/aloy/backend`). **3.2 boot** (Postgres/Supabase env) and **3.3b clarify** remain.

### Visual rebrand TODO
Logo assets in `public/` are still `pori-*.svg`/`.png` (referenced from
`AppLayout`/`Login`). Swap for Aloy marks and tune the accent palette (currently
Tailwind zinc/indigo) in the visual pass.
