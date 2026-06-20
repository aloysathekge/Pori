# Runtime Trust Model

Pori treats approvals, prompt instructions, validation, and output scanning as
defense in depth. They are not containment boundaries.

Every run has an immutable `RunContext` carrying organization, user, agent,
session, and run identity. Local Pori creates an explicit local scope. Hosted
systems must derive tenant identity from authenticated authorization context,
not model text or request-provided identifiers.

Tool actions produce typed execution receipts. A model statement that a file
was created, a search ran, or another external effect occurred is not execution
evidence without a successful receipt from the responsible tool boundary.

## Isolation Profiles

- `local`: tools execute with the host process privileges. Appropriate only for
  a trusted local user.
- `sandbox`: filesystem or command tools are restricted by an executor, but the
  Python host process and trusted plugins remain privileged.
- `worker`: the complete run executes in an isolated worker with scoped
  credentials, filesystem, process, and network policy. Multi-tenant hosted
  execution should use this profile.

In-process Python plugins share the worker's privileges. Untrusted plugins,
skills, MCP servers, shell commands, and generated code require a separate
isolation boundary and must still use the run's authorization and receipts.
