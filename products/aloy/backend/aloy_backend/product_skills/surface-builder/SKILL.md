---
name: surface-builder
description: Build or revise a model-authored Aloy Event Surface as a safe, useful React application. Use for new Surface generation, layout or behavior changes, diagnostic repair, and revision review before publication.
---

# Surface Builder

Create a Surface that is specific to the Event without turning the example into
a hardcoded product template. University, travel, career, and other Events must
all use the same runtime and safety contract.

## Workflow

1. Inspect the supplied Event brief, permanent Session context, canonical
   records, current Surface revision, diagnostics, and user request.
2. State the Surface's primary jobs, important entities, evidence sources,
   uncertainty, required states, viewports, and interaction intents.
3. Return one schema-valid, complete replacement candidate containing every
   required source file. Use only the provided Surface SDK and approved
   dependencies. Do not access host APIs, ambient credentials, arbitrary
   network endpoints, or parent-frame internals.
   Aloy owns the application shell and fixed compiler: never return
   `index.html`, package manifests, lockfiles, compiler configuration,
   dependencies, or other toolchain files. Return only model-owned React,
   TypeScript, JavaScript, CSS, JSON, Markdown, and SVG source.
4. Do not call authoring, filesystem, build, preview, publication, rollback, or
   answer tools. Aloy's trusted host owns those operations and the model-visible
   tool surface is intentionally empty.
5. When the host returns deterministic diagnostics, return a new complete
   candidate that repairs every finding. Never return a partial patch. The host
   grants only a bounded number of candidate submissions.
6. Bind displayed facts to canonical Event data. Label each important value as
   user-reported, verified, estimated, pending, or indeterminate; never present
   a plan or estimate as completed reality.
7. Declare every interaction intent and classify it:
   - local UI state can execute inside the Surface;
   - durable data changes go through validated Aloy commands;
   - reasoning requests return to the permanent Event Session;
   - consequential actions create a Proposal and require the applicable rail;
   - source-changing actions create a new Surface revision.
8. Implement loading, empty, populated, partial, stale, error, and permission-
   denied states at the required desktop and compact viewports.
9. Repair deterministic build, SDK, accessibility, responsiveness, and intent
   diagnostics in the next complete candidate.
   Every declared intent must include an accessible executable
   `interaction_checks` path in `surface.json`. Aloy's host runs those paths in
   a real browser against the exact Event context and refuses publication when
   a visible control is missing, disabled, throws, sends the wrong SDK method,
   or produces a payload that violates the declared schema.
10. Never describe a candidate as built, previewed, published, or live. Only
    Aloy's host may make those claims after a verified publication receipt.
11. Preserve the last-good design and Event truth when revising an existing
    Surface. A failed candidate must remain safe to discard.

## SDK contract

Write `/workspace/surface.json` whenever the Surface needs host data or a
meaningful interaction. The host rejects undeclared capabilities, intents,
payload fields, data namespaces, and tools. A minimal durable-selection
manifest is:

```json
{
  "format": "aloy-react-surface",
  "entrypoint": "/src/App.tsx",
  "sdk_version": "1",
  "capabilities": ["event", "tasks", "data:academic", "ask_aloy"],
  "intents": {
    "academic.course_selected": {
      "class": "state",
      "schema": {
        "type": "object",
        "properties": {"courseId": {"type": "string", "maxLength": 100}},
        "required": ["courseId"],
        "additionalProperties": false
      },
      "write": {
        "namespace": "academic",
        "operation": "create",
        "key_field": "courseId",
        "posture": "user_reported"
      }
    }
  },
  "interaction_checks": [
    {
      "name": "Select a course",
      "steps": [
        {"action": "click", "role": "button", "name": "Select Algorithms"}
      ],
      "expect": {"method": "command", "name": "academic.course_selected"}
    }
  ],
  "widgets": []
}
```

Use `useEvent`, `useTasks`, `useSurfaceData(namespace)`, and `useInteractions`
for reactive reads. Interaction records are the durable way to render queued,
running, approval, execution, committed, rejected, failed, or indeterminate
outcomes after the original SDK Promise has resolved; never invent completion
from local component state.
Use `command(name, input)` for every host-owned state or reasoning command.
State intents must declare exactly one of `create`, `replace`, `merge`, or
`delete`; choose the real entity lifecycle operation instead of simulating an
upsert in component state. `dispatch(name, payload)` remains compatibility-only
for already-published V1 Surfaces. Use `askAloy(message, context)` for a legacy
free-form explicit reasoning turn, and
`requestAction({name, payload, reason})` only for a declared external action
whose manifest entry names the exact host tool. Local sorting, filtering,
tabs, disclosure, and temporary form state stay local and require no intent.
Import these APIs directly from `@aloy/surface`. Await every durable SDK
Promise, show a pending state while it is in flight, show an actionable error
when it rejects, and reconcile the UI from refreshed canonical Surface data.
Never wrap SDK writes in a `void` helper, swallow `.catch(...)`, clear a form
before persistence succeeds, or claim success from optimistic local state.

SDK writes are revision-bound and idempotent. Reuse an idempotency key only
when retrying the exact same user action. A user-originated Surface write may
record `user_reported`; generated code cannot claim `committed`, `verified`, or
receipt-backed posture.

The host owns runtime connectivity. On every connection the SDK acknowledges
the exact bridge session, answers host heartbeats, and accepts context or
interaction responses only from that session. Interaction Promises are
bounded and may be replayed once after a retryable failure or host reconnect;
the SDK preserves the original idempotency key across that replay. Do not add
custom parent-window messaging, retry loops, network fallbacks, fake offline
success, or a second persistence layer. Use `useSurfaceRuntime()` when the
generated application needs to show a local degraded state, while the trusted
host remains responsible for reconnect and recovery controls.

## Quality bar

- Make the Event's next useful decision obvious without flattening the whole
  Event into a dashboard.
- Prefer clear hierarchy, calm density, responsive composition, accessible
  controls, and informative empty or degraded states.
- Keep the Trail, Tasks, Proposals, files, evidence, and Session canonical even
  when the Surface chooses not to display all of them.
- Send user selections and actions back as typed intent payloads with enough
  context for Aloy to reason and respond in the same permanent Session.
- Never mutate durable Event truth by editing generated source.
- Never bypass proposal, approval, receipt, validation, or publication rails.
- Never overwrite the last-good revision in place.
- Never describe generated source as live; the host communicates publication
  only after it verifies the exact retained build and atomic live pointer.
