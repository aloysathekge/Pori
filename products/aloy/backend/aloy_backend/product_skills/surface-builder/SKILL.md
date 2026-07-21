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
   uncertainty, required states, viewports, and interaction intents. Treat
   phone widths around 360 and 390 pixels, tablet, a narrow desktop Workbench
   pane, and wide desktop as distinct required compositions rather than scaled
   copies of one layout.
   The request includes a host-issued `primary_job_contract`. Copy every job id
   and description exactly into `surface.json`; never drop, rename, reorder, or
   replace them with easier jobs.
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
   denied states at the required desktop and compact viewports. On compact
   widths, navigation, tables, maps, timelines, kanban boards, forms, and charts
   must recompose without page-level horizontal overflow, clipped actions, or
   pointer-only controls. Keep primary touch actions at least 44 pixels.
   Use exactly one visible `main` landmark. Give every visible interactive
   control an accessible name, keep custom controls keyboard reachable, give
   every image an `alt` attribute, and never emit duplicate DOM ids. Aloy's
   host renders wide 1440px, split 640px, tablet 768px, mobile 390px, and narrow
   mobile 360px compositions and rejects page overflow, clipped controls, or
   missing deterministic accessibility evidence.
   For each data-driven primary region, call `useSurfaceResourceState` with its
   exact capability name and spread `feedbackProps` onto the visible region.
   Render honest loading, empty, stale, error, permission-denied, pending, and
   indeterminate views from that host-owned value. Never infer failure from an
   empty array and never add inspection-only branches. Treat long titles,
   descriptions, lists, tables, and evidence as ordinary input: wrap prose,
   constrain internal scrollers, virtualize or paginate dense collections when
   appropriate, and never solve density with page-level horizontal overflow.
   Preserve a visible keyboard focus indicator on every control; never ship
   `outline: none` without a stronger replacement. Use deterministic solid
   backdrops behind text so the host can prove at least 4.5:1 contrast for
   normal text and 3:1 for large text. Treat browser-reported focus and contrast
   failures as blocking diagnostics, not visual preferences.
9. Repair deterministic build, SDK, accessibility, responsiveness, and intent
   diagnostics in the next complete candidate.
   Every declared intent must include an accessible executable
   `interaction_checks` path in `surface.json`. Aloy's host runs those paths in
   a real browser against the exact Event context and refuses publication when
   a visible control is missing, disabled, throws, sends the wrong SDK method,
   or produces a payload that violates the declared schema.
   Also declare a complete semantic path for every host-issued `primary_jobs`
   entry. Each job has accessible steps followed by one or more observable
   assertions: a visible named region/control, exactly one typed SDK request,
   a committed Surface-data value, or an approval state. The host resets Event
   context, executes each job in a real browser, and rejects the build unless
   the complete job succeeds. Do not use CSS selectors, arbitrary scripts, or
   inspection-only UI.
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
  "primary_jobs": [
    {
      "id": "job_0123456789abcdef",
      "description": "Select a course and preserve that choice",
      "steps": [
        {"action": "click", "role": "button", "name": "Select Algorithms"}
      ],
      "assertions": [
        {"kind": "request", "method": "command", "name": "academic.course_selected"},
        {"kind": "state", "namespace": "academic", "key": "CS301", "field": "courseId", "equals": "CS301"}
      ]
    }
  ],
  "widgets": ["table", "form"]
}
```

The sample job id is illustrative. Always use the exact ids supplied by the
host request. A read-only job may have no steps, but still needs at least one
`visible` assertion using an exact accessible role and name. Interactive jobs
should prove both the typed request and the resulting host-owned state or
approval outcome when applicable.

The host-reviewed V1 widget registry currently accepts only these generic
widgets: `table`, `form`, `timeline`, `chart`, and `kanban`. `approval` also
requires the `proposals` capability, and `file_viewer` requires `files`.
Unknown widget ids and missing capability grants fail closed before source is
persisted. Provider-backed widgets such as maps are not yet in the registry;
do not invent or declare them.

Use `useEvent`, `useTasks`, `useSurfaceData(namespace)`,
`useEventRecords(namespace)`, `useInteractions`, and
`useCommandAttempts` for reactive reads. Their exact V1 shapes are:

```ts
useEvent<T>(): T | null
useTasks(): Array<Record<string, unknown>>
useSurfaceData<T>(namespace: string): Array<SurfaceDataRecord<T>>
useEventRecords<T>(namespace: string): Array<EventRecord<T>>
useInteractions(): SurfaceInteraction[]
useSurfaceInteraction(id: string | null | undefined): SurfaceInteraction | null
useLatestSurfaceInteraction(name: string, componentId?: string): SurfaceInteraction | null
useProposals(): SurfaceProposal[]
usePendingApprovals(): SurfaceProposal[]
useSurfaceApprovalState(): SurfaceApprovalState
useReceipts(): SurfaceReceipt[]
useTrail(): SurfaceTrailEntry[]
useCommandAttempts(): SurfaceCommandAttempt[]
useSurfaceRuntime(): { status: 'disconnected' | 'healthy' | 'degraded'; message?: string }
```

`useSurfaceData` returns an array directly. Read each entity from
`record.data`, use `record.key` as its canonical identity, and never destructure
it as `{data, status, error}`. `useCommandAttempts` takes no namespace argument;
filter its returned records by `name` only when the view needs a subset.
`useEventRecords` is the read-only projection for evidence-backed canonical
Event knowledge produced by sourced research. Declare `records:<namespace>` in
the manifest. Each record includes posture, confidence, and inspectable source
references. Never copy it into `data:<namespace>` merely to display it, and
never present `unverified` fields as confirmed facts.
Interaction records are the durable
way to render accepted commands through queued, running, approval, execution,
committed, rejected, failed, or indeterminate outcomes. Command-attempt records
retain host conflicts and policy/validation rejections even when no interaction
was accepted. Match a caught `SurfaceRequestError.attemptId` when present and
use its `serverCode`/`retryable` fields for specific recovery copy; never invent
completion from local component state.
Use `useSurfaceCommand(name, {componentId})` for every user-facing host-owned
state or reasoning control. Call its `execute(input)` method, disable repeated
submission with `pending`, and render one visible non-empty status/error element
with the returned `feedbackProps`. `feedbackProps` contains DOM attributes
(`role`, `aria-live`, `data-aloy-command-name`, and
`data-aloy-command-status`); it is not a status object. Read lifecycle state
from the controller's `status`, `pending`, `error`, and `retry` members, and
spread `feedbackProps` onto a persistently mounted element:

```tsx
const save = useSurfaceCommand<ApplicationInput>(
  'career.application_created',
  { componentId: 'add-application' },
);
const applications = useSurfaceData<Application>('career');

async function submit(input: ApplicationInput) {
  try {
    await save.execute(input);
  } catch {
    // Keep the form input. The persistent region below renders save.error.
  }
}

async function retry() {
  try {
    await save.retry();
  } catch {
    // The same persistent region continues to render the actionable error.
  }
}

return <>
  {applications.map(record => <ApplicationRow
    key={record.key}
    application={record.data}
  />)}
  <div {...save.feedbackProps}>
    {save.pending
      ? 'Saving…'
      : save.status === 'committed'
        ? 'Application saved.'
        : save.error?.message ?? 'Ready'}
    {save.error?.retryable && <button onClick={retry}>Retry</button>}
  </div>
</>;
```

For reasoning and external actions, `save.status === 'accepted'` means only
that the host durably accepted the request. Continue with
`save.lifecycleStatus` and `save.interaction` (or
`useSurfaceInteraction(receipt.id)`) to render queued, running,
waiting-approval, executing, and terminal feedback. Never label an accepted
request complete. Approval UI itself remains host-owned even when the generated
Surface links to or summarizes the pending Proposal.
Use `useSurfaceApprovalState()` to explain what is waiting and direct the user
to the host approval region. Spread its `feedbackProps` onto the persistently
visible summary region so the trusted host can verify the pending state. Its
`proposals` and `interactions` identify the work, but it never grants decision
authority. Use `useReceipts()` to render confirmed external outcomes and
`useTrail()` for a bounded semantic history. Never put Approve, Reject,
credential, payment-confirmation, or provider-execution authority in generated
code; those controls remain in Aloy's trusted host.

Keep that feedback element mounted when a dialog closes or a canonical record
changes identity. The host publication gate exercises each
declared command and rejects a Surface unless refreshed canonical context is
delivered before the committed/accepted feedback becomes visible. Use the
lower-level `command(name, input)` only outside React hooks or for carefully
composed internal helpers that provide the same lifecycle UI.
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
When an event handler catches `execute()` solely to prevent an unhandled React
Promise, keep the hook's visible error output mounted and offer `retry()` for a
retryable failure; never replace that error with silent local success.
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
  controls, mobile-safe touch targets, and informative empty or degraded
  states.
- Keep the Trail, Tasks, Proposals, files, evidence, and Session canonical even
  when the Surface chooses not to display all of them.
- Send user selections and actions back as typed intent payloads with enough
  context for Aloy to reason and respond in the same permanent Session.
- Never mutate durable Event truth by editing generated source.
- Never bypass proposal, approval, receipt, validation, or publication rails.
- Never overwrite the last-good revision in place.
- Never describe generated source as live; the host communicates publication
  only after it verifies the exact retained build and atomic live pointer.
