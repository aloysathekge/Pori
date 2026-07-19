# Boot Aloy locally — quickstart

Bring the whole stack up on your machine: **backend** (`products/aloy/backend`) +
**web app** (`products/aloy/app`). ~10 minutes. No Postgres or Docker required for dev — the
backend defaults to **SQLite**.

## What you need

- **Python 3.11+** and **[uv](https://docs.astral.sh/uv/)**
- **Node 20+** (`npm`)
- A free **Supabase** project (used for login/auth only) — <https://supabase.com>
- An **LLM API key** — Anthropic *or* Google (whichever your default model uses)

## 1. Supabase (auth) — ~3 min

Create a free project, then from **Project Settings → API** copy:

- **Project URL** → `SUPABASE_URL` (backend) and `VITE_SUPABASE_URL` (web)
- **anon public key** → `VITE_SUPABASE_ANON_KEY` (web)

Email auth is on by default — that's all you need. *(For a real multi-user
Postgres instead of SQLite, also grab the DB connection string from
Settings → Database.)*

## 2. Backend

```bash
cd products/aloy/backend
cp .env.example .env
```

Edit `.env` — the minimum:

```env
SUPABASE_URL=https://YOURPROJECT.supabase.co
DATABASE_URL=sqlite+aiosqlite:///./aloy.db        # dev default; or your Supabase Postgres URL
CORS_ORIGINS=http://localhost:5173
ANTHROPIC_API_KEY=sk-ant-...                       # or GOOGLE_API_KEY=... to match your model
```

For model-authored Event Surfaces, copy the credential-free specialist-role
example beside `.env`:

```bash
cp aloy.models.example.yaml aloy.models.yaml
```

Set the Builder and future Critic provider/model IDs there and keep their API
keys in `.env`. The legacy Conversation AgentConfig seam remains independent
for existing Conversations but is operator-only and has no customer-facing
Agents page. New ordinary Conversations use Aloy's configured default runtime.
Surface generation fails closed while `surface_builder.qualification.status` is
`unqualified`; change it to `qualified` only with the evaluation suite and
evidence that justified promotion. Restart the API and worker after changing
the role file. Set `ALOY_MODEL_ROLES_PATH` only when the file lives elsewhere.

Surface compilation requires an isolated sandbox in hosted environments. For
single-developer local testing without E2B, start both API and worker with
`SURFACE_BUILD_BACKEND=local_dev`. This uses only Aloy's pinned Vite, React,
and Surface SDK toolchain in an ephemeral directory; generated source cannot
supply commands, dependencies, plugins, configuration, or the HTML shell.
Never enable `local_dev` on a hosted or multi-user deployment.

Then install, migrate, run:

```bash
uv sync                                            # deps + the Pori kernel (editable, from ../../..)
uv run python -m alembic -c alembic.example.ini upgrade head  # create/update tables
uv run uvicorn aloy_backend.api:app --reload --port 8000
```

Backend is up at <http://localhost:8000> (OpenAPI docs at `/docs`).

## 3. Web app

In a second terminal:

```bash
cd products/aloy/app
cp .env.example .env.local
```

Edit `.env.local`:

```env
VITE_SUPABASE_URL=https://YOURPROJECT.supabase.co
VITE_SUPABASE_ANON_KEY=YOUR-ANON-KEY
VITE_API_BASE_URL=http://localhost:8000/v1
```

Then:

```bash
npm install
npm run dev                                        # http://localhost:5173
```

## 4. Use it

Open <http://localhost:5173> → **Sign up** (Supabase) → **Chat**. You should see:

- answers **streaming** in live, a collapsible **thinking** view, **tool chips**;
- **delegation** as `delegate_task` tool events for big/parallel asks;
- **clarify buttons** when the agent calls `ask_user` with options;
- **Memory / Skills / Teams / Traces / Usage** tabs.

## Gotchas

- **Run the backend from `products/aloy/backend`** — its `pyproject` pulls the kernel from `../../..` (the repo root), installed editable by `uv sync`.
- **Model ↔ key:** set the LLM key for whichever provider your default model uses (check `config.yaml` / the agent config). Gemini → `GOOGLE_API_KEY`, Claude → `ANTHROPIC_API_KEY`.
- **CORS:** keep `http://localhost:5173` in `CORS_ORIGINS`, or the browser blocks `/v1` calls.
- **Auth:** the web app logs in via Supabase; the backend verifies that JWT — **both must point at the same Supabase project.**
- **SQLite vs Postgres:** SQLite is zero-setup for one-machine dev. For multi-user / production, point `DATABASE_URL` at Postgres (your Supabase DB) and re-run `alembic upgrade head`.

## Migration status this exercises

This is **Stage 3.2 (boot)** from `docs/Aloy.md` — the first time the adopted
backend + web app run end to end on the unified `PoriEvent` contract (streaming,
tools, delegation, clarify) with the layered-knowledge scope resolver.
