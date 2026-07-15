# Aloy V1 R1 handoff - Life conversations and Event sessions

## State

- Base: `aloy-v1` at R0 squash merge
  `069f173ec59dad02b0f9bbb26cf3598b51c10c47` (#168).
- Phase branch: `aloy-v1-r1-life-conversations`.
- Scope: R1 in `docs/aloy-v1-plan.md`.
- Status: implementation and local automated verification complete; draft-PR
  CI and signed-in manual QA remain before merge.

## Implemented

- Default Conversation listing is scoped to the signed-in user's singleton
  Life Event. An explicit `event_id` scope remains available to trusted Event
  consumers, so dedicated Event Conversations never leak into Chat history.
- New Life chats become Life's current resume target. Deleting that target
  selects the most recently active remaining Life chat or leaves a safe empty
  state without deleting Event-owned Tasks, files, Trail, memory, or receipts.
- Dedicated Event canonical Conversation deletion remains protected with
  `409`.
- Runtime hydration includes only the active Conversation transcript. All
  owning-Event Conversations are still indexed for explicit scoped history
  search, preserving continuity without automatic sibling transcript leakage.
- Event creation accepts an optional user-owned Life Conversation origin. The
  Event metadata and `event_created` Trail entry preserve that provenance; the
  source chat and its existing durable state are not moved.
- Reading/listing Life no longer creates an empty Conversation implicitly.
- The app shell now separates **Chat**, **New chat**, **New event**, and the
  dedicated Event rail. Today opens Life through Chat and dedicated Events
  through their workspace.
- Chat can explicitly create an Event from the active Life Conversation and
  displays creation failures inline.

## Verification

- Full backend: `228 passed` using a unique workspace-local `--basetemp`.
- Final affected Event/context regression set: `14 passed`.
- Backend Black: `153 files` clean.
- Backend mypy: clean across `83 source files`.
- Aloy app ESLint: passed.
- Aloy app production TypeScript/Vite build: passed.
- `git diff --check`: passed before documentation finalization.
- Non-blocking known warning: the existing Vite main chunk exceeds 500 kB.

## Manual merge gate

Against the signed-in local stack, verify:

1. New chat creates a fresh Life Conversation and both Life chats retain
   separate transcripts.
2. The Chat switcher shows Life chats only.
3. New event creates a dedicated Event and reopening it returns to the same
   canonical Conversation.
4. Deleting Life's current chat safely selects another chat or the empty Chat
   state.
5. A dedicated Event's canonical Conversation cannot be deleted.
6. Create Event from conversation preserves the Life chat and exposes origin
   provenance on the Event.

Require green PR CI before merging into `aloy-v1`. After merge, create
`aloy-v1-r2-task-model` from the updated integration branch.
