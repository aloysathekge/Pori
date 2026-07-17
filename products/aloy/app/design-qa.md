# Today/Life design QA

- Source visual truth: `C:\Users\sathe\.codex\generated_images\019f61c8-b311-7972-9b39-85e3f722aa20\exec-643e7979-11ac-494f-9c5a-71ff9d89248c.png`
- Implementation route: `http://localhost:5173/today`
- Implementation screenshot: blocked — the available in-app and Chrome browser-control connection fails before capture.
- Target viewport: 1920 × 1080 desktop, light theme, authenticated Today state.
- State: profile greeting, Life band, priority sections, notification rail open.

## Full-view comparison evidence

The selected source image was opened and inspected. The implementation has not
yet been captured at the matching authenticated route and viewport, so no
evidence-based full-view comparison can be claimed.

## Focused-region comparison evidence

Blocked with the full-view capture. Required focused regions for the next pass:

- greeting and header actions;
- Life band;
- Needs you / Aloy is working / Coming up hierarchy;
- notification rail and expanded approval state;
- narrow responsive layout.

## Findings

- [P1] Rendered implementation evidence is missing.
  Location: authenticated `/today` route.
  Evidence: source visual is available, but browser capture could not initialize.
  Impact: typography, spacing, responsive behavior, and runtime content cannot
  be compared honestly against the selected design.
  Fix: capture the refreshed authenticated Today page at 1920 × 1080, compare it
  with the source in one visual input, then fix all P0/P1/P2 drift.

## Required fidelity surfaces

- Fonts and typography: blocked pending implementation capture.
- Spacing and layout rhythm: blocked pending implementation capture.
- Colors and visual tokens: source uses existing Aloy tokens; rendered mapping
  remains unverified.
- Image and icon fidelity: no raster content is required; the implementation
  uses Aloy's existing icon family and the established app icon dependency.
- Copy and content: implemented from durable profile, Event, Task, Proposal,
  Trail, and Conversation state; visible wrapping remains unverified.

## Comparison history

No visual iteration can begin until the first implementation capture exists.
Code verification is green but does not substitute for design QA.

## Implementation checklist

1. Capture the refreshed authenticated Today route at the target viewport.
2. Compare the source and implementation together.
3. Fix all P0/P1/P2 findings and capture again.
4. Exercise New conversation, New Event, Work on this, notification review,
   Mark all read, notification toggle, and narrow reflow.

final result: blocked (accepted for this merge; signed-in visual capture remains follow-up work)

---

# Event setup design QA

- Source visual truth: `C:\Users\sathe\.codex\generated_images\019f61c8-b311-7972-9b39-85e3f722aa20\exec-0b19e96c-6f33-4572-b669-6915c71cd5f0.png`
- Implementation route: `http://localhost:5173/events/new`
- Target state: Start simple by default; Ask Aloy changes the prompting posture
  while the same model-independent context composer remains usable.
- Implementation screenshot: blocked — the browser runtime failed before a tab could be captured.

## Findings

- [P1] Rendered comparison evidence is missing. The production build and
  interaction contracts pass, but spacing, focus states, drag/drop feedback,
  connection selection, and exact visual fidelity still require a signed-in
  capture against the source.
- [P2] The selected reference depicts a model-populated assisted proposal. With
  no setup model configured, the implementation keeps the same durable context
  composer usable instead of hardcoding University content or fake suggestions.

## Required next visual pass

1. Capture default simple mode at 1440 × 1000.
2. Exercise file drop/select, link entry, connection selection, item removal,
   and server-backed refresh recovery.
3. Switch to Ask Aloy and verify that context remains intact.
4. Create a name-only Event and a context-rich Event; inspect Today and
   Workbench placeholders while asynchronous cover work remains queued.
5. Fix all P0/P1/P2 visible drift and capture again.

final result: blocked

### Feedback iteration — 2026-07-17

- Removed the automatic-cover timing explanation from the creation action.
- Moved **Ask Aloy** beside the Event name input so the two setup paths are visible at the same decision point.
- Compressed the header, context composer, and actions so the default desktop
  creation path keeps **Create Event** visible without scrolling.
- Rendered verification remains blocked until browser control can capture the authenticated route.
- Embedded **Ask Aloy** inside the Event name field as a quiet trailing assistance action and removed generic sparkle iconography in favor of Aloy's own mark.
- Removed cover design from setup. Files, links, notes, and existing connections
  now provide the useful context from which later bootstrap and cover work can
  be derived.
