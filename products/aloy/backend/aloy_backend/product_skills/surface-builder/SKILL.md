---
name: surface-builder
description: Build or revise a model-authored Aloy Event Surface as a safe, useful React application. Use for new Surface generation, layout or behavior changes, diagnostic repair, and revision review before publication.
---

# Surface Builder

Create a Surface that is specific to the Event without turning the example into
a hardcoded product template. University, travel, career, and other Events must
all use the same runtime and safety contract.

## Workflow

1. Inspect the Event brief, permanent Session context, canonical records,
   current Surface revision, diagnostics, and user request before editing.
2. State the Surface's primary jobs, important entities, evidence sources,
   uncertainty, required states, viewports, and interaction intents.
3. Generate or patch the React source using only the provided Surface SDK and
   approved dependencies. Do not access host APIs, ambient credentials,
   arbitrary network endpoints, or parent-frame internals.
4. Call `surface_build` for the immutable revision with a fresh idempotency
   key. Treat `failed` diagnostics as source/toolchain problems to repair and
   `blocked` as unavailable isolation—not permission to execute locally.
5. Call `surface_preview` to inspect the retained build log and preview
   artifact metadata. A successful build is not permission to publish or to
   claim that the application has passed visual review.
6. Bind displayed facts to canonical Event data. Label each important value as
   user-reported, verified, estimated, pending, or indeterminate; never present
   a plan or estimate as completed reality.
7. Declare every interaction intent and classify it:
   - local UI state can execute inside the Surface;
   - durable data changes go through validated Aloy commands;
   - reasoning requests return to the permanent Event Session;
   - consequential actions create a Proposal and require the applicable rail;
   - source-changing actions create a new Surface revision.
8. Preview loading, empty, populated, partial, stale, error, and permission-
   denied states at the required desktop and compact viewports.
9. Repair deterministic build, SDK, accessibility, responsiveness, and intent
   diagnostics before responding to visual or usefulness critique.
10. Publish only through the revision service after required checks pass.
   Preserve the immutable last-good revision on every failure.

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
      "class": "durable_selection",
      "schema": {
        "type": "object",
        "properties": {"courseId": {"type": "string", "maxLength": 100}},
        "required": ["courseId"],
        "additionalProperties": false
      },
      "write": {
        "namespace": "academic",
        "key_field": "courseId",
        "posture": "user_reported"
      }
    }
  },
  "widgets": []
}
```

Use `useEvent`, `useTasks`, `useSurfaceData(namespace)`, and `useInteractions`
for reactive reads. Interaction records are the durable way to render queued,
running, approval, execution, committed, rejected, failed, or indeterminate
outcomes after the original SDK Promise has resolved; never invent completion
from local component state.
Use `dispatch(name, payload)` only for declared durable selections,
`askAloy(message, context)` for an explicit reasoning turn, and
`requestAction({name, payload, reason})` only for a declared external action
whose manifest entry names the exact host tool. Local sorting, filtering,
tabs, disclosure, and temporary form state stay local and require no intent.

SDK writes are revision-bound and idempotent. Reuse an idempotency key only
when retrying the exact same user action. A user-originated Surface write may
record `user_reported`; generated code cannot claim `committed`, `verified`, or
receipt-backed posture.

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
