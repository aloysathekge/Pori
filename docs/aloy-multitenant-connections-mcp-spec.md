# Aloy Multi-Tenant Connections & MCP — spec

_Design spec (2026-07-08). SPEC ONLY — no code. Makes connections (Gmail today,
MCP servers next) work for both of Aloy's tenant shapes — **personal** and
**business** — by adding one dimension: connection **scope**. Feeds the kernel
MCP client (`mcp_servers`, session-scoped, tenancy-blind) built in #117. Extends
the connect-engine (#114) and Gmail (#115)._

## The two tenant shapes (already in the code)

- **Personal** — every user has an auto-created personal org `user:<id>`
  (`ensure_personal_organization`); they own it.
- **Business** — a real org with members + roles `viewer → member → admin →
  owner` (`ROLE_PERMISSIONS`), each with a `Permission` set.

## The core idea: connection **scope**

Every connection/server gets `scope ∈ {user, org}`:

- **user-scoped** (what exists today): a person connects *their own* account.
  Keyed `(organization_id, user_id, provider)`. Works in both tenants —
  personal (`org = user:<id>`) or a member's **private** connection inside a
  business (`org = <biz>`, `user = <member>`), invisible to other members and to
  admins.
- **org-scoped / shared** (new): the business connects a resource **once**;
  permitted members' agents all use it. Keyed `(organization_id, provider)` with
  `user_id = NULL`. The credential lives at the org. Creation is role-gated; use
  is permission-gated.

The old "per-user vs per-org" open question resolves to **both**, told apart by
`scope`.

## Data model changes

### `OAuthConnection` (extend)
- Add `scope: str = "user"` (`"user" | "org"`), `created_by: str | None` (which
  user set up an org-shared connection).
- `user_id` becomes nullable for org-scoped rows.
- Uniqueness: `(org, user, provider)` for user-scope; `(org, provider)` for
  org-scope (partial/dual unique, or a computed key column `user_id_or_shared`).
- Migration backfills existing rows to `scope="user"`.

### `McpServer` (new)
One row per configured MCP server (personal or org-shared):
- `id`, `organization_id` (idx), `user_id: str | None`, `scope` (`user`|`org`),
  `name`, `transport` (`http`|`sse`; `stdio` later), `url`,
  `auth_kind` (`none` | `static` | `oauth`),
  `oauth_connection_id: str | None` (an MCP OAuth server reuses an
  `OAuthConnection` — "just another provider"),
  `static_secret_enc: str | None` (encrypted Bearer, same Fernet as tokens),
  `tools_include` / `tools_exclude`, `enabled`, `created_by`, timestamps.
- Same `scope` semantics: user-scoped = a member's private server; org-scoped =
  admin-provisioned for everyone.

### Permission (extend)
- Add `CONNECTION_MANAGE = "connection:manage"` for creating/removing
  **org-shared** connections and MCP servers. Grant to `admin` + `owner`.
- **Managing your own user-scoped** connection needs no special permission (you
  own it) beyond `RUN_CREATE`. **Using** an org-shared connection needs only
  `RUN_CREATE` (any member) in v1 — finer per-connection role/member allowlists
  are a later phase.

## The union resolver (the one behavioral change)

`resolve_run_connections` / a new `resolve_run_mcp_servers`, at run-setup, return
the **union**:

> **(the user's own user-scoped rows) + (the org's org-scoped rows the user's
> permissions allow to use)**

- Personal org: only user-scoped exist → identical to today.
- Business: a member gets their private connections **plus** the org-shared ones.
- Tokens are freshly resolved (refresh-on-use) exactly as today; org-shared OAuth
  refresh uses the org-scoped `OAuthConnection`.
- The merged MCP list is passed straight into `Orchestrator.execute_task(
  mcp_servers=…)` — **the kernel never learns about tenancy**; it just receives a
  per-run server list. Same for Gmail token injection into `tool_context_extra`.

## Security & privacy (what the split forces)

- **Personal stays private even from admins.** User-scoped rows are only ever
  queried by their owner; an org admin managing shared connections can never see
  or use a member's personal token. Enforced in every query by `user_id`.
- **Org-shared credentials are powerful** (one token → many users): creation
  gated by `CONNECTION_MANAGE`; **write tools (`gmail_send`, MCP writes) should be
  HITL-gated** for org-shared use; audit who created/used them.
- **Billing follows scope**: org-shared usage bills the org; personal bills the
  user (usage records already carry org + user).
- **Cross-context isolation**: a connection made in the personal org is NOT
  auto-available in a business org (keyed by `organization_id`). Connecting is
  per-org. (Opt-in sharing of a personal connection into an org is a possible
  later feature — see open decisions.)

## Transports for hosted (unchanged from the MCP spec)
- **Remote HTTP/SSE MCP servers**: per-user or org-shared, feasible now.
- **stdio MCP servers**: run inside the **user's E2B sandbox**, never a
  shared-host subprocess — later phase.

## App UX
- **Personal tenant:** the existing Connections screen — your own connections +
  your own MCP servers.
- **Business tenant, every member:** Connections shows **My connections**
  (user-scoped) and **Organization connections** (org-shared, read-only status —
  "provided by your org").
- **Business admin (`CONNECTION_MANAGE`):** an **Organization → Connections**
  management area to add/remove org-shared connections (run the OAuth as the org)
  and org-shared MCP servers (URL + auth).

## Phasing
1. **`scope` on `OAuthConnection` + migration + the union resolver** for Gmail;
   the org-shared connect flow (admin, `CONNECTION_MANAGE`); app shows
   org-shared. → **business Gmail works.**
2. **`McpServer` table** (user + org scope), OAuth-via-connect-engine +
   static-token; `resolve_run_mcp_servers` → per-run `mcp_servers` into
   `execute_task`; per-user + org-shared gating. → **MCP works, personal +
   business.**
3. **stdio MCP servers inside the E2B sandbox.**
4. **Fine-grained use-gating** (per-connection allowed roles/member allowlist),
   audit log, connection health/reconnect UX.

## Open decisions
1. **Use-gating granularity for org-shared:** any member (v1) vs per-connection
   allowed-roles vs per-member allowlist. (Lean: any member in v1.)
2. **Personal-into-business opt-in:** may a user expose a personal connection
   inside a business context, or must they reconnect per-org? (Lean: per-org for
   isolation; opt-in later.)
3. **Org-shared Google consent model:** an admin's individual consent for a
   shared Workspace vs domain-wide delegation / service account. (Affects the
   Google verification track.)
4. **One org-shared connection per provider, or several** (e.g. two company Slack
   workspaces)? (Lean: allow several, name them.)
5. **HITL default for org-shared writes:** force-on by default, or per-org policy?
   (Lean: default-on; the blast radius is the whole org.)
