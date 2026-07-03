# `@aloy/web` — Aloy web app

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
cd apps/web
npm install
cp .env.example .env.local   # Supabase auth + backend URL
npm run dev
```

## Migration status (docs/Aloy.md — "adopt pori_cloud_client, unify on PoriEvent")

- [x] **Stage 1** — copy `pori_cloud_client` → `apps/web`, rebrand identity to Aloy, builds clean (`tsc -b && vite build`).
- [ ] **Stage 2** — retarget `src/api/*` + `src/api/sse.ts` to **`@aloy/shared`** so the app speaks the kernel's **`PoriEvent`** contract, gaining **clarify buttons + delegation** (today it uses its own `status/tool/step/message` events).
- [ ] **Stage 3** — reconcile the backend (`pori_cloud` → `products/aloy/backend`, vs `pori/api`); keep Supabase **Bearer** auth (add Bearer support to the backend + `@aloy/shared`).

### Visual rebrand TODO
Logo assets in `public/` are still `pori-*.svg`/`.png` (referenced from
`AppLayout`/`Login`). Swap for Aloy marks and tune the accent palette (currently
Tailwind zinc/indigo) in the visual pass.
