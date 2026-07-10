# Aloy Object Storage + Sandbox File Provisioning — spec

_Design spec (2026-07-10). SPEC ONLY — no code yet. Gives Aloy durable file
storage (uploads + agent artifacts survive redeploys/replicas) and provisions
files into per-run sandboxes so the agent can work on data too big for
context — the architecture primer's "rung 4" file→LLM path (retrieval-based:
object storage + sandbox, `docs/architecture-primer.md` §7)._

## Goal & principle

- A user attaches a 200MB CSV; the agent analyzes it with pandas in a sandbox
  and produces a chart + a cleaned dataset; both are downloadable a week
  later, from any replica. None of that works today.
- **Durable files live in object storage; compute is provisioned per run**
  (primer §5). The sandbox filesystem is a scratchpad with a lifecycle;
  the object store is the source of truth. Files flow: store → sandbox at
  run setup (provisioning IN), sandbox → store at run end (extraction OUT).
- **Two layers:**
  - **Layer A — the object store** (net-new): a small `ObjectStore` seam +
    a `StoredFile` table. Hermes has *no precedent* for this layer — its file
    tools assume a real local filesystem ("no upload path needed",
    `execution-backends.md:37-40`), the exact assumption a hosted product
    breaks. We design it fresh, keeping Hermes's *transfer discipline*
    (SHA-256 diffing, size caps, atomic writes).
  - **Layer B — sandbox provisioning** (mostly wiring): the kernel already
    has the sandbox seam end-to-end — `Sandbox`/`SandboxProvider` ABCs,
    virtual paths (`/mnt/user-data/{workspace,uploads,outputs}`), symlink-safe
    path jailing, artifact detection (`files_written` diffing), and a working
    **E2B provider with a resume-or-create ledger** (`pori/sandbox/e2b.py:100`).
    What's missing is that the Aloy backend never plumbs `sandbox_base_dir`
    into runs, so none of it activates (see "the cwd bug").
- **Non-goals (v1):** desktop/local-bridge file access; stdio MCP servers
  inside the sandbox (parked in the MCP spec — *this* spec is its
  prerequisite); Redis/cache tier; content-addressed dedup across tenants;
  virus scanning (note a hook, don't build it).

## Current state (what the spec builds on)

- **Attachments today** (`routes/conversations.py:665`): text files inlined
  ≤200KB (`<attached-file>`); images/PDFs ride natively as kernel
  `ImageBlock`/`DocumentBlock` ≤~10MB; DOCX/XLSX text-extracted server-side.
  All bytes live in `Message.metadata_` JSON or nowhere. Rungs 1–3 of the
  primer's §7 ladder — rung 4 absent.
- **Agent-written files** land in the **backend process cwd** and are served
  by the artifacts endpoint reading from `Path.cwd()` with a per-conversation
  allowlist (`routes/conversations.py:614-645`). Ephemeral, shared across
  orgs' runs (same cwd), lost on redeploy. This produced the stray
  `about_aloy.md` / `ai_*_analysis.md` files we've been deleting from the
  repo (PRs #132, #133).
- **The cwd bug, precisely**: `worker.py:_configure_sandbox` sets the sandbox
  *provider* when `SANDBOX_ENABLED`, but `build_orchestrator`
  (`orchestrator.py:83`) passes no `sandbox_base_dir`, and no tool context
  carries it — so `_ensure_sandbox_and_thread_dirs`
  (`pori/sandbox/sandbox_tools.py:87-93`) can never build per-thread dirs,
  `bash` cwd falls through to `None` → host cwd (`pori/sandbox/local.py:62`),
  and artifact observation is disabled. Fixing this plumbing is Phase 1's
  first commit and needs no new infrastructure.
- **No blob storage anywhere**: no S3/GCS/bucket code in the backend;
  Supabase is used for auth JWKS + Postgres only. `docker-compose.yml` has no
  storage service.

## Layer A — the object store

### The seam (`aloy_backend/storage.py`, new)

```python
class ObjectStore(Protocol):
    def put(self, key: str, data: BinaryIO, *, content_type: str) -> int: ...
    def open(self, key: str) -> BinaryIO: ...          # streaming read
    def delete(self, key: str) -> None: ...
    def url(self, key: str, *, expires_s: int) -> str | None: ...
        # presigned GET when the backend supports it; None → stream through us
```

Two implementations, selected by config:

- **`LocalDiskObjectStore`** (dev, default): rooted at
  `settings.storage_dir` (default `.aloy_storage/`, gitignored). Atomic
  temp+rename writes (same discipline as the kernel's E2B ledger). `url()`
  returns None.
- **`S3ObjectStore`** (prod): any S3-compatible endpoint — AWS S3, Cloudflare
  R2, MinIO, **Supabase Storage via its S3 protocol** (we already run
  Supabase; zero new vendors). `boto3` as an optional dependency, config:
  `STORAGE_BACKEND=s3`, `STORAGE_S3_ENDPOINT`, `STORAGE_S3_BUCKET`,
  `STORAGE_S3_ACCESS_KEY`, `STORAGE_S3_SECRET_KEY`.

**Key scheme (tenancy lives in the key):**
`org/{organization_id}/conv/{conversation_id}/{uploads|outputs}/{file_id}/{filename}`
— every access path re-derives the prefix from the authenticated org, so a
leaked key string alone never crosses tenants (same rule as the artifacts
allowlist today).

### Data model (`models.py` + migration)

`StoredFile` — one row per durable blob:
- `id` (uuid), `organization_id` (idx), `user_id`, `conversation_id` (idx)
- `kind`: `"upload"` | `"artifact"`
- `name`, `content_type`, `size_bytes`, `sha256`
- `storage_key` (the object-store key), `created_at`
- `run_id: str | None` (artifacts: which run produced it)

Pointers, not bytes: `Message.metadata_` keeps rendering chips/artifact lists
but gains `file_id` on each entry so the UI can download. `Run.artifacts`
entries likewise. **Artifact rows are written inside the same finalizer
transaction as the message/run** (`persist_run_outcome`) — the
single-finalizer rule extends to files so a disconnect can't orphan a blob
(mirrors Hermes's one-transaction persistence discipline,
`streaming-persistence-architecture.md:91-99`).

### Endpoints

- `POST /conversations/{id}/files` — multipart upload (streamed to the store,
  never buffered whole in memory), returns `{file_id, name, size}`.
  Per-file cap `STORAGE_MAX_FILE_MB` (default 100), per-org quota
  `STORAGE_ORG_QUOTA_MB` (default 2048) enforced against `sum(size_bytes)`.
  Permission: `RUN_CREATE` (same as sending a message).
- `GET /files/{file_id}` — download: org-checked lookup → presigned redirect
  when `url()` gives one, else `StreamingResponse` from `open()`.
- `GET /conversations/{id}/artifacts` + `/content` — unchanged shapes, but
  served from `StoredFile`/object store instead of `Path.cwd()` (the cwd read
  path is deleted).

## Layer B — sandbox file provisioning

### Fix the wiring first (no new infra)

1. Backend `Settings` gains `sandbox_base_dir` (default `.aloy_sandbox/`,
   mirrors the kernel's `SandboxConfig.base_dir`, `pori/config.py:183`).
2. `build_orchestrator` → `Orchestrator(sandbox_base_dir=…)`; run setup puts
   `thread_id = conversation_id` into tool context — one workspace per
   conversation, reused across turns (Hermes: one env per task, subagents
   share it, `execution-backends.md:44-47`; our subagents already inherit the
   parent's tool context).
3. With that, the kernel's existing machinery activates: `bash` runs jailed in
   `{base}/threads/{conversation_id}/user-data/workspace`, virtual paths map,
   `files_written` artifact observation turns on. **This alone fixes the cwd
   bug for the local backend** and is shippable as its own PR.

### Provisioning IN (store → sandbox)

At run setup, when the turn references uploads (and on first run of a
conversation with prior uploads):

- Materialize each `StoredFile` into the sandbox at
  `/mnt/user-data/uploads/{name}` — local backend: write into the thread dir;
  E2B: `sandbox.write_file` (the SDK path Hermes prescribes for
  non-bind-mount remotes, `execution-backends.md:58-64`).
- **Skip-if-present by SHA-256** (a `.provisioned` manifest in the thread dir)
  so re-runs and continues don't re-copy — Hermes `FileSyncManager`'s
  mtime+hash change detection, simplified.
- The task message gains a system-authored block, not the file bytes:
  `Uploaded files are available in /mnt/user-data/uploads/: data.csv (198MB,
  text/csv). Use the bash/read tools to work with them.` — rung 4: tell the
  model *where*, let it extract with tools.
- Attachment routing ladder (extends primer §7): text ≤200KB → inline
  (unchanged); images/PDF ≤10MB → native blocks (unchanged); **anything
  larger, or any type we can't inline/extract → object store + provisioning**.
  The composer stops rejecting big files; it uploads them.

### Extraction OUT (sandbox → store)

At run end (in the pump's finalizer path, where artifacts already flow):

- For each `files_written` entry under `/mnt/user-data/{outputs,workspace}`:
  read from the sandbox, `put()` to the object store, insert `StoredFile`
  (kind=artifact, run_id) — all inside `persist_run_outcome`'s transaction.
- Caps: per-file `STORAGE_MAX_ARTIFACT_MB` (default 25), per-run total
  (default 100MB); oversize files keep their pointer entry with
  `too_large: true` instead of silently vanishing (Hermes: no silent caps).
- E2B: extraction must happen **before** the sandbox TTL reaps the VM — run
  end is inside the run's own lifecycle, so ordering is natural; the resume
  ledger (`pori/sandbox/e2b.py`) keeps the VM warm for continues.

### Security notes

- Path containment for every store↔sandbox copy goes through the kernel's
  `_safe_join` (`pori/sandbox/path_resolution.py:77`), which is already
  symlink-safe (`resolve()` + `relative_to` — the Hermes
  `validate_within_dir` pattern; the older string-only check the deep-dives
  flagged has been fixed).
- Subprocess env is already secret-stripped (`pori/sandbox/env_safety.py`) and
  the command floor exists (`command_safety.py`) — the two "port regardless"
  items from the Hermes mining are done; nothing new required here.
- Object-store credentials never enter the sandbox. If a tenant sandbox must
  fetch from the store directly (future), use short-lived presigned URLs, and
  scope sandbox egress to the store endpoint (Hermes egress-isolation note,
  `infrastructure-security.md:363-371`).

## Frontend

- Composer: files over the inline/native limits upload via
  `POST /files` with a progress chip, then send carries `file_refs:
  [file_id]` (new `SendMessageRequest` field) instead of bytes.
- Artifacts drawer + message chips: download via `GET /files/{file_id}`.
- Conversation view shows uploads as chips from `StoredFile` metadata (no
  content re-render needed).

## Phases

1. **Fix the wiring + durable artifacts** — `sandbox_base_dir` plumbing
   (cwd bug dies); `ObjectStore` seam + `LocalDiskObjectStore`; `StoredFile`;
   artifact extraction OUT in the finalizer; artifacts endpoints served from
   the store. _No user-visible upload changes yet; agent outputs become
   durable and downloadable._
2. **Big uploads + provisioning IN** — upload endpoint + quotas; composer
   upload flow; provisioning into the sandbox; the task-reference block.
   _Rung 4 unlocked: "analyze this 200MB CSV" works._
3. **Prod hardening** — `S3ObjectStore` (Supabase Storage first target);
   presigned downloads; retention policy (e.g. orphan cleanup for
   conversations deleted); E2B end-to-end smoke test with provisioning.

Each phase is a PR with tests (storage seam unit tests run on
`LocalDiskObjectStore`; no cloud creds in CI — same rule as everything else).

## Open questions (decide at build time, defaults chosen)

- **Multipart vs base64-JSON upload**: multipart (streams; base64 inflates
  33% and buffers). Default: multipart.
- **Provision prior-turn uploads on every run, or only when referenced?**
  Default: all of the conversation's uploads (simple, correct); revisit if
  provisioning cost shows up.
- **Supabase Storage vs R2/MinIO first**: Supabase (already a dependency);
  the seam makes this a config choice, not a code choice.
