# Boot and operate Aloy locally

This is the local quickstart, operator checklist, and V1 demonstration guide for
the Aloy product. It starts the API, durable worker, and React app together. A
local SQLite database and local blob storage are enough for one-developer use;
Postgres and remote object storage are deployment choices, not boot
requirements.

## 1. Prerequisites

- Python 3.10 or newer and [uv](https://docs.astral.sh/uv/)
- Node.js 20 or newer and npm; Bun for the current app unit suite
- `make` plus Bash (Git Bash is the supported Windows shell for the Makefile)
- a Supabase project for authentication
- a provider API key only when testing model-backed Conversation or Surface
  work

The shell can boot and inspect Aloy without model credits. Model-backed answers,
Task Runs, Event Brief generation, and Surface generation will fail closed until
their selected provider is available.

## 2. Configure the backend

From `products/aloy/backend`:

```bash
cp .env.example .env
```

For local development, the minimum useful values are:

```env
SUPABASE_URL=https://YOURPROJECT.supabase.co
DATABASE_URL=sqlite+aiosqlite:///./aloy_backend.db
CORS_ORIGINS=http://localhost:5173
SURFACE_BUILD_BACKEND=local_dev
```

Add only the provider key selected by the Pori model configuration, for example
`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, or `FIREWORKS_API_KEY`. Public web
research additionally needs `SERPER_API_KEY`, `SERPAPI_API_KEY`, or
`TAVILY_API_KEY`. Credentials stay in `.env`; never put them in YAML or commit
them.

`SURFACE_BUILD_BACKEND=local_dev` runs Aloy's fixed Surface compiler on this
developer workstation. It accepts generated source files but not model-supplied
commands, dependencies, plugins, configuration, or HTML shells. Never enable it
in a hosted or multi-user environment; those environments require the isolated
builder backend.

### Surface Builder and Critic roles

Copy the credential-free role example beside `.env`:

```bash
cp aloy.models.example.yaml aloy.models.yaml
```

The operator owns the Builder and Critic provider, model, skill, limits, and
qualification evidence. A role marked `unqualified` cannot build or judge a
Surface. Promote it to `qualified` only after the named evaluation suite has
passed, then restart both API and worker. `ALOY_MODEL_ROLES_PATH` is needed only
when the file lives somewhere else.

## 3. Configure the app

From `products/aloy/app`:

```bash
cp .env.example .env.local
```

Set:

```env
VITE_SUPABASE_URL=https://YOURPROJECT.supabase.co
VITE_SUPABASE_ANON_KEY=YOUR-ANON-KEY
VITE_API_BASE_URL=http://localhost:8000/v1
```

The app and backend must point at the same Supabase project.

## 4. Start everything

Run these commands from `products/aloy` in Git Bash:

```bash
make install
make dev
```

`make dev` applies Alembic migrations before starting all three processes:

| Process | Address | Responsibility |
| --- | --- | --- |
| API | `http://127.0.0.1:8000` | auth, REST, SSE, foreground Conversation Runs |
| Worker | background process | durable Tasks, Schedules, Event setup, Surface builds, watchdogs, reconciliation |
| Web app | `http://localhost:5173` | Today, Life Conversations, Events, Workbench, Connections, Schedules |

Useful commands:

```bash
make check     # call /v1/health
make stop      # stop port holders and stray Aloy workers
make api       # API only
make worker    # worker only
make web       # app only
make migrate   # migrations only
```

The API documentation is available at `http://127.0.0.1:8000/docs`.

## 5. First product check

1. Open `http://localhost:5173` and sign in.
2. Choose **New conversation**. It must create a fresh Conversation inside the
   permanent Life Event and must not add a dedicated Event to the Event rail.
3. Choose **New Event**, enter a name, and create it. Aloy must create the Event
   and its one canonical Conversation immediately; background setup must not
   block entry.
4. Leave the Event and reopen it. The same canonical Conversation, Tasks,
   files, memory, Trail, and Surface publication must still be there.

With no model credits, these topology, navigation, file, memory, connection,
settings, and host-owned state checks remain valid. Do not treat model failure
as evidence that Event persistence failed.

## 6. Operator behavior and recovery

### The API is healthy but work is stale

Verify that the worker process is running. The API accepts and streams
foreground requests; the worker claims durable Task Runs, setup ingestion,
Surface builds, Schedules, and Proposal reconciliation. Restarting the worker is
safe: each loop first repairs expired Run leases and orphaned Task projections,
then resumes eligible work from durable checkpoints.

Do not manually change a stale Run from `running` to `pending`. The watchdog
owns that transition and writes the recovery or terminal Trail evidence.

### A Run reaches a budget

Every producer freezes host-owned limits for steps, tool calls, tokens, cost,
and active duration. Exhaustion is a terminal, non-retryable outcome with actual
usage recorded. Increase a governed policy for a future Run; never mutate the
frozen budget of an existing Run to hide an overage.

### A provider accepted an action but the receipt is missing

Leave the Proposal `indeterminate`. The worker performs bounded, read-only
provider reconciliation using the deterministic operation identity. It never
repeats the write just because the database commit was lost. A recovered
provider result becomes a receipt-backed committed outcome; an unprovable
result remains visibly indeterminate for review.

### A long Event reopens slowly or loses older detail

Current prompts hydrate the latest verified Conversation summary plus a bounded
recent tail. Older history is retrieved on demand through the Event-scoped
history tool. The full transcript remains durable; prompt compaction must never
delete it. A fresh Life Conversation receives accepted Life memory but not a
sibling transcript automatically.

### A Surface build fails

The last verified publication remains active. Inspect the build status and
diagnostics in the Event rather than assuming a queued request is live. Builder
submission, host validation, compilation, runtime inspection, and publication
are separate stages. A failed candidate must not replace the working Surface.

## 7. The 60-second Career OS demonstration

Prepare one Event named **Career OS** with the Task **Research US companies for
startup jobs**, a qualified Conversation model, working public-search tooling,
and a qualified Surface Builder.

1. Open Career OS. The canonical Conversation and current Surface reopen.
2. Open the research Task and choose **Work on this**.
3. Show queued/running progress arriving live without leaving the Event.
4. Open the cited Markdown report under Files and inspect at least one source
   evidence record.
5. Open the Career Surface and show the same host-owned company records there.
6. Trigger one state-only Surface update; it persists without waking a model.
7. Trigger one reasoning interaction; its lifecycle appears in Surface,
   Conversation, Today, and Trail.
8. Stage a protected external action. Show that no provider call occurs before
   approval, then show the decision and receipt-backed terminal state.

The demo fails if unsupported companies are presented as facts, a generated
Surface owns canonical records, `accepted` is displayed as `completed`, an
external write occurs before approval, or reopening loses continuity.

## 8. Manual release checks

Browser-based visual acceptance is deliberately manual when browser automation
is unavailable. Check the Event Workbench and a generated Surface at desktop,
tablet, and phone widths. Verify keyboard-only operation, visible focus, useful
landmark and control names, contrast, reduced motion, no horizontal page
overflow, touch targets, pane collapse/restore, modal focus trapping, and a
recoverable degraded Surface state. Record this evidence separately; static
source inspection is not a passing visual result.

## 9. Safe verification

From the repository root:

```bash
uv run --no-sync pytest tests/ -q --basetemp .pytest_tmp_release_kernel
uv run mypy pori/ --ignore-missing-imports
```

From `products/aloy/backend`:

```bash
uv run --no-sync pytest tests/ -q --basetemp .pytest_tmp_release_backend
uv run --no-sync mypy aloy_backend/ --ignore-missing-imports
```

From `products/aloy/app`:

```bash
npm run test
npm run lint -- --max-warnings 0
npm run build
```

These automated gates do not replace the signed-in, real-provider, or manual
viewport checks above.
