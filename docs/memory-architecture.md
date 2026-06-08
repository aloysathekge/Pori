# Memory Architecture

Pori uses one storage-neutral `MemoryRecord` contract in core and cloud.
`CoreMemory` remains the small curated context block; searchable long-term
memory uses typed records.

## Scope

Every record carries `organization_id`, `user_id`, and optional `agent_id` and
`session_id`. Retrieval requires an exact organization and user match. A record
with no agent or session is intentionally visible to narrower scopes belonging
to that same user. Broader scopes cannot read another user's records.

Pori Cloud currently derives `organization_id` as `user:{user_id}` from the
authenticated identity. This is a safe personal tenant boundary until an
organization membership and RBAC model is introduced.

## Record Policy

Records include:

- semantic, episodic, or procedural kind
- provenance and source identifiers
- confidence and sensitivity
- creation, update, event, deletion, and expiry timestamps
- retention policy and optional legal hold
- conflict key, status, and supersession link

Conflict policies are `keep_both`, `reject`, and `supersede`. Deletion is soft
by default so lifecycle actions remain auditable; hard deletion is explicit.
Expired records are excluded before scoring.

## Retrieval Evaluation

`evaluate_retrieval` reports recall at k, precision at k, reciprocal rank, and
any records that violate the request scope. Isolation tests should always
assert that `leaked_record_ids` is empty.
