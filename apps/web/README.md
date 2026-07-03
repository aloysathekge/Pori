# `@aloy/web` — Aloy web app

The Aloy chat SPA. Talks to the Pori/Aloy backend over REST + SSE through
[`@aloy/shared`](../shared) — no other backend coupling.

> **Reconciliation note:** an external `pori_cloud_client` (a Vite/React frontend)
> also exists — see `docs/Aloy.md` §11.1. Per the agreed *harvest-Hermes*
> direction this app was scaffolded fresh from Hermes `web/` (below); if
> `pori_cloud_client` has screens worth keeping, fold them in over this shell.

## Stack

React 19 · Vite 6 · TypeScript · Tailwind 4 · lucide-react. This is the **build
scaffold + chat structure harvested from Hermes `web/`** (MIT) and rewired: the
`@nous-research/ui` design system, `@xterm` terminal / PTY bridge, the
dashboard-session-token plugin, `/api` proxy, and the dozen Hermes-only pages
were **not** copied — Aloy has its own theme and talks plain REST + SSE. See
`docs/Aloy.md` §3a and `references/HARVEST.md`.

## Run

```bash
cd apps/web
npm install
cp .env.example .env.local   # set your backend URL + API key
npm run dev                  # http://localhost:5173
```

The backend is the kernel/Aloy API (`uvicorn`), which streams `PoriEvent`s from
`POST /v1/tasks/stream`. Set `PORI_API_KEY` on the server and the matching
`VITE_ALOY_API_KEY` here.

## Scripts

- `npm run dev` — Vite dev server
- `npm run typecheck` — `tsc --noEmit`
- `npm run build` — production build
- `npm run preview` — preview the build

## What's here (v1)

Chat with live streaming (answer + collapsible **thinking**), tool-call chips,
and **clarify buttons** (the `ask_user` bridge). Next: conversations sidebar,
delegation sub-threads, memory panel, auth screen.
