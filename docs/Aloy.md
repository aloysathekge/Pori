# Aloy — Product Plan (website · webapp · desktop · API)

**Status:** Draft v0.1
**Date:** 2026-07-03
**Scope:** Aloy as the flagship **product** on the Pori kernel — its four delivery clients (**API/backend**, **webapp**, **desktop/Electron**, **website**) and the cross-cutting planes (auth/tenancy, knowledge, streaming, receipts) that bind them.
**Companion:** [`Pori.md`](./Pori.md) — the kernel PRD. Aloy *consumes* Pori; Pori never depends on Aloy.

> This plans the surfaces and the glue. It does **not** re-specify the kernel (that's `Pori.md`). Genuinely undecided points are marked **⚠ OPEN** and collected in §11.

---

## 1. Summary

**Aloy is a personal *and* org OS agent** — Hermes-class capabilities as the baseline, plus governance (audit, policy, cost attribution, layered org knowledge) that falls out of Pori's receipt/validator/memory contract. It ships as:

- **API / backend** — the single source of truth: auth, tenancy, conversations, memory, org policy, receipts/audit, cost — composing `pori` (kernel) + `extensions/pori-*`, streaming over **SSE / `PoriEvent`**.
- **Webapp** — the browser SPA daily-driver (chat, memory, delegation view, org admin).
- **Desktop (Electron)** — the same app in a native shell, plus local system reach (files, notifications, tray, offline-ish).
- **Website** — the public marketing/onboarding front door.

**One rule governs all of it:** surfaces talk to the backend **only over REST + SSE** — never a Python import. The backend imports `extensions` and `pori`; the kernel imports nothing upward. This is the single safeguard against the Hermes-monolith trap.

**Build order:** *Personal Aloy first* (a Hermes-grade daily driver on the hardened kernel), org plane after — but **tenancy-aware from day one** so isolation is never retrofitted.

---

## 2. What Aloy is (positioning)

| | Aloy |
|---|---|
| **For whom** | individuals (personal OS agent) → teams/orgs (governed OS agent) |
| **Baseline** | Hermes-class: portability, cost discipline, breadth, delegation |
| **Beyond** | eval-native governance — every action is a receipt, every policy is a validator, memory is org→team→personal |
| **Moat** | the **layered knowledge plane** (org-shared + team + personal, personal wins) + provable/auditable work |
| **Not** | a chat wrapper, a workflow-graph builder, or a single-tenant script — governance and multi-tenancy are structural |

The differentiator and the enterprise requirement are the *same thing*: audit, governance, policy, and cost attribution are exactly what a receipts+validators kernel produces for free.

---

## 3. Topology

```
 products/aloy/appsite ──signup──▶  products/aloy/app (SPA)     products/aloy/desktop (Electron)
                                        │                          │
                                        └──── packages/pori-client (typed REST + SSE client, TS) ────┘
                                                          │
                                                          ▼   REST + SSE  (PoriEvent stream)
                                    products/aloy/backend  (FastAPI)
                                      auth · tenancy · RBAC · conversations · memory ·
                                      org policy · receipts/audit · cost attribution · gateway
                                                          │   Python import (one-way, CI-enforced)
                                                          ▼
                                    extensions/pori-*   (memory-scope/tenancy · skills · learning · gateway · providers)
                                                          │
                                                          ▼
                                    pori/  (KERNEL: Plan→Act→Reflect→Evaluate loop · delegation ·
                                            memory engine · validators · receipts · llm · tools · sandbox)
```

**Dependency direction:** `surfaces (products/aloy/{web,desktop,website}) → (REST+SSE) → products/aloy/backend → extensions → pori`. Never upward.

---

### Terminology

- **Client surface** means a delivery client such as the webapp, desktop shell,
  website, or API.
- **Event Surface** means the typed, composable Event workspace defined in
  [`aloy-surface-spec.md`](./aloy-surface-spec.md).

These are different architectural concepts. Product specifications use
“Surface” unqualified only for the Event Surface.

## 3a. Client-surface strategy — copy Hermes, then make it Aloy

Hermes's web, desktop, shared-transport, and gateway are **MIT-licensed**, so for the **surfaces** we **copy the actual code, then make it ours** — not a clean-room rewrite. Every surface starts as a Hermes copy:

1. **Copy** the Hermes shell (`web/`, `apps/desktop`, `apps/shared`, gateway) into `packages/` (the shared client) and `products/aloy/` (the surfaces).
2. **Retarget the transport** — strip the PTY / JSON-RPC bridge to `hermes --tui` (Pori has no TUI); point everything at Pori's **REST + SSE / `PoriEvent`** via `packages/pori-client`.
3. **Rebrand fully** — name (**Aloy**), logo + assets, color palette + theme tokens, typography, all copy/strings; remove every Hermes brand mark.
4. **License hygiene** — retain the MIT notice + attribution on copied files; log each copy in `references/HARVEST.md` (source → license → destination → what changed).

**⚠ Boundary (do not blur):** copy-then-rebrand is for **surfaces** (MIT UI shells). It does **not** override two standing rules — the **kernel (`pori`) stays pattern-harvest** ("never paste"; this is what keeps it from rotting into a god-file), and **Claude Code-derived ideas stay clean-room** (non-permissive — design/behavior only). Copy freely from *Hermes UI*; adapt *patterns* for the kernel.

---

## 4. API / backend — the keystone

**Home:** `products/aloy/backend/` — evolves from the repo's `pori/api` (and the external `pori_cloud` sibling; **⚠ OPEN §11.1** to reconcile). It is the *only* thing every surface depends on.

### 4.1 Stack
- **FastAPI** (async) + **Uvicorn**, **SSE** streaming over the `PoriEvent` contract (already built in the kernel API — GW-4).
- **PostgreSQL** + **SQLAlchemy** + **Alembic** migrations; **Docker** packaging.
- Composes `pori` (agent loop, delegation, memory engine, validators, receipts) + `extensions/pori-*` (tenancy, memory-scope, learning, gateway).

### 4.2 Responsibilities (what the backend owns that the kernel does not)
- **Auth** — API keys + user sessions (JWT/opaque); **fail-closed** by default (already done in the kernel API), OAuth/SSO later for org.
- **Tenancy & RBAC** — `org → team → user` identity on every request; per-request memory isolation (Phase-0 fix is the first brick); role→permission checks.
- **Conversations** — create/resume/list/branch/delete; per-user persistent memory (Postgres-backed `MemoryStore`).
- **The knowledge plane** — the org→team→personal **scope resolver** at prompt-build time (§8).
- **Org policy engine** — org/role policy expressed as **scoped kernel validators** (pre-gate / post-check); the safety floor stays in the kernel.
- **Receipts / audit / cost** — persist the receipt chain per tenant → audit log + per-tenant/per-user **cost attribution** dashboards.
- **Gateway** — messaging adapters (Slack for org, Telegram for personal) as a thin adapter ABC over `PoriEvent` (later; harvest Hermes *architecture*, not its 19K-line `run.py`).

### 4.3 API endpoints (illustrative REST + SSE)
```
POST   /v1/auth/token                         # exchange creds → session
GET    /v1/conversations                      # list (scoped to identity)
POST   /v1/conversations                      # create
POST   /v1/conversations/{id}/messages        # send a turn
GET    /v1/conversations/{id}/stream          # SSE: PoriEvent (text/thinking/tool/
                                              #   delegation-progress/clarification_request)
POST   /v1/clarify/{request_id}               # answer a structured ask_user (button bridge)
GET    /v1/memory                             # scoped memory blocks (org/team/personal)
GET    /v1/receipts/{conversation_id}         # audit chain + cost
GET    /v1/org/{org}/policy   /members  ...    # org admin (later)
```

### 4.4 Data model (first cut)
`orgs`, `teams`, `users`, `memberships(role)`, `conversations`, `messages`, `memory_blocks(scope, owner)`, `receipts(hash, parent_hash, actor, tenant, cost)`, `api_keys`, `usage`.

### 4.5 Backend milestones
- **B0** — extract `pori/api` → `products/aloy/backend/`; declare fastapi/starlette deps; per-request memory isolation; fail-closed auth *(mostly done in-kernel; port + reconcile with `pori_cloud`)*.
- **B1** — Postgres `MemoryStore`; users + conversations + persistent memory; SSE end-to-end incl. clarify buttons.
- **B2** — receipts persistence + audit/cost read APIs.
- **B3** — org/team tenancy + RBAC + policy-validators; org admin APIs.
- **B4** — gateway adapters (Slack/Telegram).

---

## 5. `packages/pori-client` — the typed transport package

A small **TypeScript** package both the webapp and the desktop app import — the single client for the backend so surfaces never duplicate protocol logic.

- **Owns:** REST client, **SSE `PoriEvent` decoder** (text / thinking / tool-call / delegation-progress / `clarification_request`), the **clarify-button** round-trip, auth/token handling, and shared TS types generated from the backend schema.
- **Harvest:** **copy** Hermes `packages/pori-client` (a TS transport pkg) → **strip the PTY/JSON-RPC bridge** → retarget to Pori REST + SSE (see §3a).
- **Why:** one protocol implementation, two shells. Changes to the event contract land in one place.

---

## 6. Webapp (`products/aloy/app`)

**Purpose:** the primary daily-driver UI in the browser.

### 6.1 Stack (matches the harvest source)
**React 19 + Vite + TypeScript + Tailwind 4 + react-router + lucide-react** — the Hermes `web/` shell, retargeted. Consumes `packages/pori-client`.

### 6.2 v1 scope (personal)
- **Chat** with live streaming (text + separate **thinking** block), tool-call previews, and **artifact/receipt** affordances.
- **Delegation view** — show `delegate_task` children (single/batch/background) as live sub-threads with their summaries; surface background completions.
- **Clarify UI** — render `ask_user` options as **buttons** (the bridge), submit back over REST.
- **Memory panel** — view/edit CoreMemory blocks (persona/human/notes); later scoped org/team/personal.
- **Conversations** — list, resume, branch, delete.
- **Settings** — model/provider, `.pori/agents` specialists, `llm.tiers`.

### 6.3 Later (org)
- Org/team switcher; **admin** (members, roles, policy); **audit & cost** dashboards (from receipts); shared org knowledge editor.

### 6.4 Harvest
**Copy** Hermes `web/` (React 19 + Vite + Tailwind SPA) → keep the shell + chat components + streaming UX → **strip the PTY bridge** → point the transport at `packages/pori-client` → **rebrand** (Aloy name, color/theme tokens, typography, logo/assets, copy). Per §3a.

---

## 7. Desktop (`products/aloy/desktop`, Electron)

**Purpose:** the same webapp in a native shell, plus local system reach a browser can't offer.

### 7.1 Stack
**Electron** wrapping the `products/aloy/app` build, sharing `packages/pori-client`. **Copy** Hermes `products/aloy/desktop` → strip the PTY bridge → retarget to REST + SSE → **rebrand** (app name, icon, window chrome, tray assets). Per §3a.

### 7.2 Why desktop (what it adds over the webapp)
- **Local filesystem / workspace** access for the agent (real local files, not uploads).
- **OS integration** — tray, global hotkey, native notifications (esp. for **background delegation** completions), launch-on-login.
- **Local execution** — talk to a **local backend** or a local sandbox for private/offline work; local skills in `.pori/`.
- **Credential vault** — OS keychain for API keys/tokens.

### 7.3 Modes
- **Thin:** shell over the hosted backend (same as web, native niceties).
- **Local-first:** bundles/points to a **local** `products/aloy/backend` for privacy — the personal-Aloy sweet spot.
**⚠ OPEN §11.4:** which mode is v1 (recommend thin-first, local-first fast-follow).

### 7.4 v1 scope
The webapp feature set + tray/notifications (background-delegation done → native toast) + local file access + keychain. Auto-update later.

---

## 8. The knowledge plane (the moat) — how it threads all surfaces

Layered inheritance **org → team → personal**; on collision, **personal wins** (mirrors Hermes local-over-external). A **scope resolver** runs at prompt-build time in the backend, merging the three layers into the agent's context; memory writes ride the kernel's `write → receipt → validate → commit` rails, tagged with scope + actor.

- **Kernel** owns the block model, recall→inject, write lifecycle, `MemoryStore` interface.
- **Backend (`extensions/pori-*` + aloy)** owns the **scope resolver**, RBAC on writes, concrete Postgres stores, and the learning-loop curator.
- **Client surfaces** just render scoped memory and let users edit what their role allows.

**Milestone rule (already decided):** build the scope resolver *now*, populate only the **personal** layer first; org/team slot in later with **no refactor**.

---

## 9. Cross-cutting contracts

- **Streaming = `PoriEvent` over SSE.** One event contract for every surface (text, thinking, tool_call_start/end, delegation progress, `clarification_request`). Already live in the kernel API; `packages/pori-client` is its only decoder.
- **Auth/identity.** `org:team:user` on every request → the isolation boundary *and* the audit/cost tag. Personal = a degenerate single-member org, so nothing is retrofitted for orgs.
- **Receipts everywhere.** Audit log, replay, and per-tenant cost all read the same receipt chain the kernel emits — surfaces just visualize it.
- **Delegation is first-class in the UI.** `delegate_task` (single/batch/background) surfaces as sub-threads + background toasts; specialists (`.pori/agents/`) are editable in settings.

---

## 10. Build order (phased, personal-first, tenancy-aware from day 1)

1. **P0 — Backend foundation.** Extract/reconcile the API into `products/aloy/backend`; per-request isolation; fail-closed auth; SSE + clarify buttons end-to-end. *(Most kernel pieces exist; this is port + Postgres + reconcile.)*
2. **P1 — `packages/pori-client`.** Typed REST + SSE client (harvest Hermes shared; strip PTY).
3. **P2 — Webapp v1 (personal).** Chat + streaming + delegation view + clarify UI + memory panel + conversations, on `packages/pori-client`.
4. **P3 — Desktop v1 (thin).** Electron shell around the webapp + tray/notifications + local files + keychain.
5. **P4 — Website.** Marketing/onboarding front door → signup → webapp.
6. **P5 — Org plane.** Tenancy/RBAC/policy-validators; org admin + audit/cost dashboards; scope resolver populated for org/team.
7. **P6 — Gateway.** Slack (org) + Telegram (personal) adapters over `PoriEvent`.

Website can be built anytime (it's decoupled); it's sequenced late only because it markets a working product.

---

## 10a. Website (`website/`)

**Purpose:** the public front door — product story, positioning, pricing, docs links, **signup/waitlist** → hands off to the webapp.

- **Stack:** if Hermes ships a marketing/landing site, **copy + rebrand** it (per §3a); otherwise a static site — recommend **Astro** (or reuse the external `pori_website` sibling, **⚠ OPEN §11.1**). Either way, rebrand fully (name/colors/theme/copy).
- **Scope:** landing (the "personal + org OS agent" story), features, pricing, docs/changelog links, auth entry (sign in / sign up → webapp), waitlist for org.
- **Decoupled:** talks to the backend only for signup/waitlist; everything else is static. Not on the critical path to a working product.

---

## 11. Open questions & risks

**Open:**
1. **`pori_cloud` / `pori_website` reconciliation.** External sibling repos (`Pori/pori_cloud`, `pori_website`, `pori_cloud_client`) vs. in-repo `products/aloy/{backend,web,website}` + `packages/pori-client`. Decide: copy-in vs. submodule vs. keep external. (Backend README already flags copy-in.)
2. **`pori/api` vs `products/aloy/backend`.** The small in-kernel API vs. the product backend — one must become canonical (recommend: backend is canonical; kernel keeps only a trivial reference server, if any).
3. **Backend auth for org.** API-key + session now; OAuth/SSO (Google/Microsoft) for org — when.
4. **Desktop v1 mode.** Thin (hosted) vs. local-first backend — recommend thin-first, local-first fast-follow (§7.3).
5. **Website stack.** Astro vs. Next vs. reuse `pori_website`.
6. **✓ RESOLVED — TS workspace.** Bun workspace over `packages/*` + `products/*/{web,desktop,website}`; the shared client is `@pori/client` (`packages/pori-client`), linked via `workspace:*`. Client surfaces live under their product; only product-neutral libs sit in `packages/`. See [`MONOREPO.md`](../MONOREPO.md).
7. **Hosting/deploy.** Backend (Fly/Render/K8s?), Postgres, SSE at scale (keep-alive, proxies).
8. **Real-server SSE e2e** for the clarify-button flow (noted follow-up from the kernel API work).

**Risks:**
- **Client/backend coupling drift** → mitigated by the REST+SSE-only rule and `packages/pori-client` as the single client.
- **Kernel rot** (backend logic leaking into `pori`) → mitigated by the CI-enforced one-way dependency.
- **Org retrofit** → mitigated by tenancy-aware-from-day-1 + the scope resolver built before it's needed.
- **Harvest Frankenstein** (web/desktop from Hermes) → keep the shells + components, strip the PTY bridge, retarget to `PoriEvent`; log in `HARVEST.md`.

---

## 12. Glossary

- **Event Surface** — an Event-owned, versioned composition of allowlisted
  typed blocks rendered by trusted React components; never arbitrary UI code.
- **Client surface** — a delivery client (webapp, desktop, website) that talks
  to the backend over REST + SSE only.
- **`packages/pori-client`** — the one TS transport package (REST + SSE `PoriEvent` decoder + clarify bridge) both shells consume.
- **Scope resolver** — backend component merging org→team→personal knowledge at prompt-build time (personal wins).
- **Knowledge plane** — the layered-inheritance memory that is Aloy's moat.
- **`PoriEvent`** — the kernel's typed streaming event contract carried over SSE.
- **Policy = validators** — org/role policy is a set of scoped kernel validators, not a separate engine.
