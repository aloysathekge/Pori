# Aloy vision v2 and V1 reset-plan handoff

## Why the plan changed

Phase 5 proved the durable Event workspace, continuous Session, initial Task
state, Proposal system, Surface, and Today. It also exposed the missing product
bridge: an `open` Task such as "Research US companies for startup jobs" is a
checklist row and does not durably cause Aloy to work on it.

The revised model makes Task the executable contract and Run the bounded unit
of agent execution. V1 starts work only through an explicit **Work on this**
action. Protected external consequences still require a durable Proposal,
human decision, executor, and receipt.

A second review corrected the Conversation topology. The original universal
"one Event, one Session" rule made casual chat unusable. Life is now the
deliberate exception: it is one permanent personal Event with many user-started
Conversations. Dedicated Events keep one canonical continuous Conversation.
This adds R1 before Task execution so navigation, deletion, context assembly,
and transcript isolation are correct before Tasks begin reporting into chats.

## Canonical documents

- `docs/aloy-vision.md` — canonical product definition, version 2.1.
- `docs/aloy-v1-plan.md` — active R0–R7 delivery plan and gates.
- `docs/aloy-wedge-spec.md` — delivered foundation architecture; its old
  remaining phase order is superseded.
- `.agent/progress/current.md` — current repository state and immediate action.

## Locked V1 decisions

- Life is the permanent user–Aloy personal Event and may own many fresh
  Conversations. A dedicated Event owns one canonical lifetime Conversation.
- New Conversation enters Life; New Event creates a dedicated workspace.
- A fresh Life Conversation receives accepted Life state, not every sibling
  transcript. Older chats are retrieved explicitly when relevant.
- Tasks retain optional origin provenance, and each Run retains its selected
  Conversation so progress returns to the right Life thread or dedicated Event
  Conversation.
- A Task is durable executable work, not merely a checklist row.
- Work starts explicitly in V1; automatic Task selection is deferred.
- One active Task Run per Event and one foreground Run per Conversation in V1,
  subject to an account-wide cap; additional work queues.
- The Surface is trusted server-derived state in the context pane.
- The Trail is semantic Event history; Run Replay remains low-level detail.
- Career OS research is the hero flow. It creates a sourced report artifact,
  stages a Gmail summary Proposal, and finishes only after receipt evidence.
- Desktop is primary, web is a responsive fallback.

## Immediate next action: R0 only

Remain on `aloy-v1-phase-5-surfaces` and close the existing foundation:

1. apply the formatting-only Black correction to migration
   `v8f9a0b1c2d3_event_primary_conversation.py`;
2. complete signed-in visual QA for the three-region Event workspace;
3. verify visible streaming, canonical Session reopen, delete protection, and
   Surface loading against the local stack;
4. make PR #168 green and merge it into `aloy-v1`.

After R0 merges, create `aloy-v1-r1-life-conversations` from the latest `aloy-v1`.
Do not create R1 from the current unmerged branch.

## Known state and blockers

- PR #168 remains draft. The migration was formatted in `1cbd7a5`, the plan was
  committed in `98d1710`, and all seven GitHub checks are green.
- The previous visual QA report is blocked because in-app browser control was
  unavailable. R0 is not complete until signed-in visual and interaction QA
  passes.
- The local stack was last healthy at API `http://127.0.0.1:8000` and web
  `http://localhost:5173`; recheck it before relying on that state.
- Local R0 verification is green: Black (`153 files`), `225` backend tests,
  backend mypy (`83 source files`), app ESLint/build, API health, web health,
  and the worker chain. The initial pytest attempt hit a Windows permission
  error in the shared system temp directory; rerunning with a unique
  workspace-local `--basetemp` passed all tests.

## Verification for this planning pass

- Markdown/source-of-truth consistency checks and `git diff --check` only.
- Code tests are intentionally skipped because this pass changes documentation
  and progress records only.
