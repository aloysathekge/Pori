# Aloy Event Workbench design QA

- source visual truth: `C:\Users\sathe\OneDrive\Pictures\Screenshots\Screenshot (7).png`
- implementation URL: `http://localhost:5173/events/<event-id>`
- target viewport: 1920 × 1080 desktop
- state: signed-in Event with Conversation, Workbench, and Event context

## Intended comparison

The source was inspected at original resolution. It establishes the behavior,
not a literal theme clone: a persistent app sidebar, a central first-class
workspace, an adjacent utility/context region, and flexible panes that do not
reduce documents or generated UI to modal drawers. Aloy retains its own visual
tokens, type, mark, and icon system.

## Implemented fidelity surfaces

- Conversation, Split, and Workbench focus modes.
- Draggable Conversation/Workbench and Surface/resource dividers.
- Persistent Workbench tabs for Surface, message artifacts, Event files, and
  Run replay.
- A collapsible Event context rail grouping Tasks, Approvals, Receipts, Files,
  and Trail.
- A full global sidebar that can stay open or auto-hide and reveal from the
  left edge.
- Per-Event persistence for layout mode, ratios, open tabs, active tab,
  context visibility, and the optional Surface/resource split.
- Narrow-screen focus behavior and existing Surface-ready conversation card.
- Host-owned file viewers and an Ask Aloy action that attaches the trusted
  stored-file reference to the Event composer.

## Verification completed

- ESLint passed.
- TypeScript and the production Vite build passed.
- The existing Vite large-chunk warning remains non-blocking.
- No backend code changed in this phase.

## Visual comparison blocker

The required in-app browser connection fails while initializing, before Aloy
can be opened or captured. Therefore no same-viewport implementation screenshot,
combined reference comparison, or signed-in interaction pass can be claimed in
this run. The founder must inspect the already-running local URL before this PR
is treated as visually accepted.

final result: blocked — automated signed-in capture and comparison unavailable
