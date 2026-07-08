# Aloy Connections — spec: the connect-engine + Gmail (first app)

_Design spec (2026-07-08). SPEC ONLY — no code yet. Lets an Aloy user connect
their own accounts (Gmail first) so the agent can act on them, with Aloy owning
the tokens and the data path (no Composio). Native web OAuth; the Pori kernel is
untouched — this is all Aloy product, added through the kernel's tool seam._

## Goal & principle

- A user clicks **Connect Gmail**, approves a Google consent screen, and the
  agent can then read/search/send their mail — with the OAuth token stored
  **encrypted in Aloy's own DB**, scoped to that user, never leaving Aloy.
- **Two layers, built once vs per-app:**
  - **Layer A — the connect-engine** (reusable): the OAuth web flow + token
    custody + refresh. Written once; every future app reuses it.
  - **Layer B — Gmail** (per-app): a `ProviderSpec` (scopes, endpoints) + a few
    tools that call the Gmail API with the stored token.
- **Non-goals (for v1):** 100-app breadth (that's Composio's game; we do the few
  that matter), MCP (parked), per-org shared connections (start per-user).

## Why Aloy makes this easier than Hermes

Hermes is a local CLI → localhost-callback + terminal-paste OAuth. Aloy is a
hosted web app with a real backend + URL → a **standard web OAuth redirect**.
The reference to borrow from Hermes is the *token lifecycle* handling
(`tools/mcp_oauth.py`: refresh, 401 re-auth, atomic storage) — delivered via a
normal redirect instead of a localhost server. See
`references/hermes-agent-deep-dives/external-connections.md`.

---

## Layer A — the connect-engine (reusable)

### Data model (new tables, `aloy_backend/models.py` + migration)

`OAuthConnection` — one per (user, provider):
- `id`, `organization_id` (idx), `user_id` (idx), `provider` (e.g. "google")
- `access_token_enc`, `refresh_token_enc` — **encrypted at rest** (see security)
- `scopes: list[str]`, `expires_at: datetime | None`
- `account_email: str | None` (the connected account, for display)
- `status: "active" | "revoked" | "error"`, `created_at`, `updated_at`
- unique (organization_id, user_id, provider)

`OAuthFlowState` — short-lived, one per in-flight connect:
- `state` (PK, random) , `organization_id`, `user_id`, `provider`,
  `pkce_verifier`, `redirect_after` (app URL to return to), `expires_at` (10 min)

### Provider abstraction (`aloy_backend/connections/providers.py`)

A `ProviderSpec` so adding an app = add a spec (+ its tools), not new OAuth code:
```
ProviderSpec(
  name="google",
  authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
  token_url="https://oauth2.googleapis.com/token",
  revoke_url="https://oauth2.googleapis.com/revoke",
  scopes=[...],                      # per-app (see Gmail below)
  client_id_env="GOOGLE_OAUTH_CLIENT_ID",
  client_secret_env="GOOGLE_OAUTH_CLIENT_SECRET",
  extra_authorize_params={"access_type": "offline", "prompt": "consent"},  # get a refresh_token
  account_email_from="id_token",     # how to extract the connected email
)
```
Registry `PROVIDERS = {"google": ...}`. A provider is "available" only when its
client-id/secret env vars are set (gated, like every other Aloy capability).

### The OAuth web flow (endpoints, `aloy_backend/routes/connections.py`)

All tenant-scoped (the user from `OrganizationContext`):

1. `POST /v1/connections/{provider}/start` → mint `state` + PKCE verifier, persist
   `OAuthFlowState`, return `{ authorize_url }` (authorize_url includes
   `redirect_uri = {BACKEND}/v1/connections/{provider}/callback`, `state`,
   `code_challenge`, `scope`). The app opens it (popup or full redirect).
2. Google shows **Aloy's** consent screen; user approves; Google redirects to
   the backend callback with `?code=…&state=…`.
3. `GET /v1/connections/{provider}/callback` (no auth header — identity comes
   from the `state` row): validate + consume `state`, exchange `code`+verifier
   for tokens at `token_url`, extract `account_email`, **encrypt + upsert**
   `OAuthConnection` (status=active), then redirect the browser back to
   `redirect_after` (the app's Connections screen) with a success flag.
4. `GET /v1/connections` → list the user's connections (provider, account_email,
   scopes, status — never the tokens).
5. `DELETE /v1/connections/{provider}` → call `revoke_url`, mark revoked/delete.

### Token lifecycle (`aloy_backend/connections/store.py`)

- `get_access_token(session, org, user, provider) -> str | None`: load the
  connection; if `expires_at` is near, **refresh** using `refresh_token`
  (persist the new access token + expiry); on refresh failure mark `error` and
  return None. This is the single choke point every tool goes through.
- 401 from the provider API → the tool reports "reconnect needed"; the app shows
  a Reconnect button (re-runs `/start`). (Hermes's 401-dedup is a nice-to-have
  later; v1 can be simpler.)

### Security & trust (the whole point)

- **Encryption at rest:** tokens encrypted with an app secret
  (`CONNECTIONS_ENC_KEY`, a Fernet/AES key from env; KMS/SSM in prod). The DB
  never holds plaintext tokens. Decrypted only in-process to call the provider.
- **Tenant scoping:** every query filters `organization_id` + `user_id`; a user
  can only ever reach their own connection.
- **Scope minimization:** request the least Gmail scope that works (start
  read-only; add send only when needed).
- **The consent screen shows Aloy** (our registered OAuth app) — the user is
  authorizing *Aloy*, not a third party. That is the trust story Composio can't
  give.
- **Revoke = delete + remote-revoke**, so "disconnect" actually severs access.
- **Audit:** connect/disconnect are surfaced in the run replay / logs.

---

## Layer B — Gmail (the first app)

### Google Cloud setup (one-time, operator) — the real cost

- Register an **OAuth 2.0 Web application** in Google Cloud Console; set the
  authorized redirect URI to the backend callback; put the client id/secret in
  the backend env (`GOOGLE_OAUTH_CLIENT_ID/SECRET`).
- **The timeline item to plan around:** Gmail scopes are *restricted*. Serving
  real users at scale requires Google's **OAuth verification + a CASA security
  assessment** (weeks; possibly a paid third-party audit). Until verified, the
  app works for **test users** you allowlist in the consent screen — which is
  perfect for dogfooding and a private beta. Plan the verification as a parallel
  track, not a blocker to building.

### Scopes (start minimal)
- v1: `gmail.readonly` (search/read) + `gmail.send` (send). Add `gmail.modify`
  (label/archive) only when a feature needs it. Fewer restricted scopes = faster
  verification.

### Tools (Aloy product tools, `aloy_backend/tools/gmail.py`)
Registered onto the kernel registry in a **`google` capability group** (so they
gate as a unit). Suggested v1 set:
- `gmail_search(query, max_results)` → message list (id, from, subject, snippet, date)
- `gmail_read(message_id)` → full message (headers + body text)
- `gmail_send(to, subject, body, cc?)` → send
- (later) `gmail_list_labels`, `gmail_modify`

### How a tool gets the user's token (the key wiring)

Kernel tools are **sync** and can't do async DB lookups, and `check_fn` can't see
*which* user. So resolve connections **at run-setup time**, where Aloy has the
async DB + the authenticated user, and inject:

1. In `send_message` / the worker, before building the orchestrator: look up the
   user's active connections; for each, resolve a **fresh** access token via the
   store (refreshing if needed).
2. Pass them into the run via `tool_context_extra={"connections": {"google":
   {"access_token": "...", "account_email": "..."}}}` (the same injection seam
   already used for `clarify_handler`). The Gmail tool reads
   `context["connections"]["google"]["access_token"]` — no async in the tool.
3. **Per-user gating:** in `build_orchestrator`, include the `google` capability
   group **only if** the user has an active Google connection. So a user who
   hasn't connected Gmail never sees the Gmail tools (zero surface), and one who
   has, does. This uses the existing `allowed_capability_groups` filter — no new
   gating machinery.

Result: the kernel stays product-agnostic (a tool just reads a token from its
context), Aloy owns the token custody + refresh, and the tools appear per-user.

---

## App UX (`products/aloy/app`)

- A **Connections** screen (new nav item, or a section in Settings): a card per
  available provider (Gmail, later Slack/Calendar) showing status —
  **Connect** (not connected) / **Connected as alice@gmail.com · Disconnect**.
- **Connect** → `POST /connections/google/start` → open `authorize_url` in a
  popup (or full-page redirect); the backend callback returns to the app with a
  success flag; the screen refreshes the list.
- **Disconnect** → `DELETE /connections/google` with a confirm.
- **Reconnect** affordance when a connection is in `error`.
- `api/connections.ts` mirrors the endpoints. Styled in the current design system.

---

## Effort & phasing

- **Phase 1 (the engine + Gmail read):** OAuthConnection/OAuthFlowState tables +
  migration, ProviderSpec + Google spec, the 5 connect endpoints, encryption,
  the store/refresh, `gmail_search`/`gmail_read`, the run-setup injection +
  capability-group gating, the Connections screen, tests. ~2–3 focused days.
- **Phase 2:** `gmail_send` (+ scope), Reconnect UX, 401 handling polish.
- **Phase 3:** second provider (Calendar or Slack) — proves the ProviderSpec
  abstraction; mostly a new spec + tools, the engine unchanged.
- **Parallel track (not code):** Google OAuth verification / CASA assessment for
  going past test-users. Start early.

## Open decisions (for you)

1. **Encryption key management:** env-var Fernet key for dev → KMS/SSM for prod?
   (Recommend: env now, note KMS for prod.)
2. **Popup vs full-page redirect** for the connect flow. (Recommend: popup —
   keeps the app state; fall back to redirect.)
3. **Per-user vs per-org connections.** (Recommend: per-user for v1; org-shared
   later via the scope model.)
4. **Which first scope** — readonly-only for v1, or readonly+send together?
5. **Who owns the Google verification** track and its timeline.
6. **Long tail:** once native covers the top apps, is MCP (parked) the substrate
   for the rest, or keep hand-building? (Decide when the top few are done.)
