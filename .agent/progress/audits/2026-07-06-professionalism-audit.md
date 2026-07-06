# Professionalism Audit — 2026-07-06

Full-monorepo audit (kernel, backend, web app + @pori/client, website, desktop, repo root)
run via 4 parallel read-only audit agents. This file is the canonical findings list;
work through it top-down. Tags: HIGH / MEDIUM / LOW = impact on professionalism.

## Cross-cutting blockers (fix first)

1. **HIGH — No LICENSE file exists**, yet MIT is advertised in `README.md:7` (badge),
   `README.md:489`, and `pyproject.toml:10`. Legal/OSS adoption blocker; three dead references.
2. **HIGH — Release story stale** (CORRECTED 2026-07-06: v1.4.0 *IS* on PyPI, published
   2026-04-23 — the audit agent's "never published" claim was wrong): but the published
   package is **219 commits behind main** (monorepo migration, agent-package split, delegation
   all unreleased); `CHANGELOG.md` also stops at 1.4.0. Fix = cut a 1.5.0 release + changelog.
   `publish.yml`'s stale "# DISABLED (billing)" header fixed this session; workflow uses
   Trusted Publishing (user must configure repo as Trusted Publisher on pypi.org, or the
   original token path).
3. **HIGH — TS workspace has zero tests and zero CI.** `.github/workflows/ci.yml` is
   Python-only; no `tsc -b`, no eslint gate, no vitest anywhere in app or pori-client.
   Highest-risk untested code: the hand-rolled SSE frame parser (`products/aloy/app/src/api/sse.ts`).
4. **HIGH — Repo-root junk tracked in git**: `debug.log` (Chromium crashpad noise),
   `config.yaml` (contradicts CONTRIBUTING's "create from example" flow), `deepagent_copy.md`
   (28 KB internal planning note at root). Stale `.gitignore` rule `apps/**/package-lock.json`.

## Kernel (`pori/`)

- MEDIUM — mypy runs with **no config** (`ci.yml:40`, no `[tool.mypy]`): loose mode; untyped
  defs pass silently. 239 `Any`/`Dict[str, Any]` across 45 files hollow the "Pydantic-validated" pitch.
- MEDIUM — **No exception hierarchy** (`pori/exceptions.py` missing). `FatalAgentError`
  (`agent/schemas.py:23`), `StructuredOutputParseError` (`llm/openai.py:23`),
  `CapabilityResolutionError` (`tools/registry.py:46`), `BudgetExceeded` (`runtime.py`) are
  scattered with inconsistent bases; no `except pori.PoriError` for consumers.
- MEDIUM — `setup_logging()` mutates the **root logger** and streams to **stdout**
  (`utils/logging_config.py:147-148, 69, 99`) — classic library sins.
- MEDIUM — README is kernel-only; never mentions the monorepo (`products/`, `packages/`,
  bun workspace). "Pori Cloud" described as external, though it's `products/aloy` in-repo.
  Architecture tree omits `main.py`, `cli.py`, `curator.py`, `evolution.py`, etc.
- MEDIUM — 5 god-files >800 loc: `agent/core.py` (1701), `main.py` (1699), `skills.py` (1222),
  `memory.py` (1170), `tools/standard/filesystem_tools.py` (967). `Agent` manually rebinds ~10
  sibling-module methods (`agent/core.py:123-133`).
- MEDIUM — 45 broad `except Exception:` across 18 files; `sandbox/local.py:78` swallows all
  subprocess errors into a string. `sandbox/local.py:63-65` runs model-supplied strings with
  `shell=True` (mitigated by hardline floor + HITL, but the riskiest line in the kernel).
- MEDIUM — `docs/` is internal planning (PRD, implementation plan), not adopter docs; no
  Sphinx/MkDocs API reference despite ~120 exported symbols.
- LOW — `print()` in importable code: `pori/clarify.py:45,50,52`. Stale comment
  `pyproject.toml:152-154` claims package lives under `packages/pori/` (it doesn't).
- LOW — pre-commit lacks ruff + mypy (black/isort only); coverage floor 65%.
- Positives: zero bare excepts, zero TODO/FIXME, bounded deps with rationale, strong CI matrix,
  marker taxonomy, only 1 legitimate skip, SECURITY.md with real SLA.

## Backend (`products/aloy/backend`)

- **HIGH — The durable worker is never started in deployment.** `worker.py` + the
  `pori-cloud-worker` entrypoint exist, but `docker-compose.yml` defines only `api` and
  RUNBOOK.md never starts it → every `POST /runs` and durable `send_message`
  (`conversations.py:680-705`) enqueues runs that nothing executes. Release-blocking.
- **HIGH — In-process state breaks under `--workers 2`** (`Dockerfile:59`):
  `CLARIFY_BRIDGES` module-global set (`streaming.py:31`) means clarify resolution 404s when
  routed to the other worker; the in-memory rate limiter (`rate_limit.py`) is per-process and
  never evicts. Move to Redis/shared store or pin to 1 worker.
- MEDIUM — No startup config validation: `Settings` defaults `supabase_url=""` and SQLite URL
  (`config.py:8,11`); misconfig fails per-request, not at boot.
- MEDIUM — `init_db()` runs `create_all` at every boot alongside Alembic (`database.py:20-25`)
  → schema-drift risk; gate to dev/SQLite.
- MEDIUM — SSE gaps: no overall timeout on the interactive streaming path; assistant message
  persisted only after loop completes (`conversations.py:856-909`) so client disconnect loses
  the answer; `BaseHTTPMiddleware` (`middleware.py:16`) wraps StreamingResponse (buffering risk).
- MEDIUM — No unified error schema (3 shapes coexist); `list_runs` unpaginated
  (`runs.py:102-112`); `list_conversations` limit uncapped (`conversations.py:88`).
- MEDIUM — Plaintext logs; `request_id` contextvar never reaches the formatter. No JWT `iss`
  check; no timeout on JWKS fetch (`auth.py:20-23`); python-jose unmaintained → consider pyjwt.
- LOW — Health check is static (no DB ping); graceful shutdown is a no-op; 500 handler
  f-string-interpolates client `X-Request-ID` into JSON (`middleware.py:22,31`);
  `RunRequest.task` unbounded length; prod origin hardcoded as default (`config.py:14-16`).
- Positives: JWT signature+exp+aud properly verified with rotation retry; tenant isolation
  consistently enforced AND tested; RBAC guards owner-minting; 24 real Alembic migrations;
  multi-stage non-root Dockerfile; `scope_resolver.py` is pure, deterministic, fully tested
  (note: team layer not wired yet; knowledge load capped LIMIT 200, `conversation_runtime.py:98`).

## Web app (`products/aloy/app`) + `packages/pori-client`

- **HIGH — ~20 silent `catch {}` blocks** ("// handle silently") across ChatPage, MemoryPage,
  TracesPage, TeamsPage, SettingsPage, UsagePage. No toast system, no Error Boundary
  (render throw = white screen).
- **HIGH — SSE not resilient**: no reconnect/backoff; on drop, accumulated `streamText` is
  discarded (`ChatPage.tsx:215-220`) — partial answer silently lost. No 401 interceptor →
  expired session never redirects to /login.
- **HIGH — Missing env = white screen**: `AuthContext.tsx:23-26` calls `createClient` at module
  load; undefined Supabase env throws during import with no boundary to catch it.
- **HIGH — Rebrand unfinished**: `pori-icon.svg`/`pori-logo.svg` wired into `index.html:5`,
  `AppLayout.tsx:51,100`, `LoginPage.tsx:33`, `SignupPage.tsx:38`. Dead Vite template assets remain.
- **HIGH — `@pori/client` README documents `AloyClient`; exported class is `PoriClient`**
  (`client.ts:69`) — the copy-paste example doesn't compile. App also duplicates the SSE parser
  (`app/src/api/sse.ts` ≈ `pori-client/src/sse.ts`) instead of using `PoriClient` — two parsers
  to keep in sync; the package's headline class is effectively dead code.
- MEDIUM — No Prettier; `noUnusedLocals/Parameters` off; eslint not enforced anywhere;
  icon-only buttons lack `aria-label` (0 `aria-` hits in app/src); bare meta tags in index.html.
- LOW — pori-client has no build/dist (exports raw `./src/index.ts`) — not independently
  publishable despite README framing; no retry logic in client either; app `.gitignore`
  doesn't cover plain `.env`.
- Positives: strict TS, zero `any`, zero console.log, all components <400 loc, real
  loading/empty states, semantic layout, clarify chips are real buttons.

## Website (`products/aloy/website`)

- **HIGH — Conversion funnel dead-ends**: `index.html:340` (quickstart), `:343` (kernel),
  `:362` (GitHub) are `href="#"`; footer Docs (`:361`) loops back into the page.
- MEDIUM — No `og:image`/Twitter card, no `og:url`/canonical; no deploy config anywhere
  (no vercel/netlify/pages workflow) and no real domain referenced.
- LOW — No analytics; check WCAG contrast on `--muted`/`--faint` captions.
- Positives: clean copy (no typos/lorem), 22 KB fully inline, strong a11y
  (aria-hidden, focus-visible, prefers-reduced-motion), responsive, good base meta.

## Desktop (`products/aloy/desktop`)

- Stub (README only). **MEDIUM — root `package.json:9` workspace glob `products/*/desktop`
  matches a dir with no package.json** → dead/broken workspace entry. Either scaffold a minimal
  Electron shell (package.json, main + preload with contextIsolation, load app/ build,
  electron-builder config, signing later) or drop the glob until then.

## Workspace siblings (parent dir)

`pori_cloud`, `pori_cloud_client`, `pori_website` (separate git repos) + `pori_docs` are all
untouched since ≤2026-06-30 (pre-harvest) → archive candidates to avoid confusion about the
source of truth.

## Suggested order of attack

**Day-1 quick wins (mostly mechanical):** add LICENSE (MIT); untrack `debug.log` +
`config.yaml` + move `deepagent_copy.md`; fix pori-client README class name; wire the
4 real URLs on the landing page; drop/scaffold the desktop workspace glob; fix stale
pyproject/gitignore comments.

**Week-1 (credibility):** TS CI job (tsc + eslint + first vitest tests on the SSE parser);
compose `worker` service + startup config validation in the backend; Error Boundary +
toast channel in the app; refresh CHANGELOG; decide the PyPI publish story.

**Then:** boot the stack end-to-end (still the #1 milestone per current.md), SSE resilience
(both sides), single-worker or Redis for clarify/rate-limit, finish the app rebrand,
mypy strictness + `pori/exceptions.py`, og:image + deploy config for the website,
monorepo-aware README, MkDocs API reference.
