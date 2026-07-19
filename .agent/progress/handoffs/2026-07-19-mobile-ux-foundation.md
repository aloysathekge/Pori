# Aloy mobile UX foundation - 2026-07-19

## Outcome

The `aloy-v1-mobile-ux` branch reorganizes Aloy's existing web product for
phones without creating a separate mobile product or weakening the Event
model. Mobile users can move among Today, Life, creation, Events, and utility
navigation with one-handed controls, then work inside an Event one primary
region at a time.

## Product behavior

- The mobile bottom bar owns the five frequent destinations: Today, Life, New,
  Events, and More.
- New distinguishes a Life conversation from a dedicated Event in a bottom
  sheet. Events opens a visual Event switcher; More opens the full global
  sidebar.
- The mobile Event header separates identity, live state, Event context, and
  Conversation/Workbench mode controls into usable rows. Split mode remains a
  wider-screen capability.
- Event context is a full-screen mobile panel with six equal, non-overlapping
  trusted tabs. Workbench resources use horizontally scrollable tabs.
- Shared dialogs become bottom sheets on phones. Long utility pages own their
  vertical scrolling, and setup forms stack controls when horizontal space is
  limited.
- Dynamic viewport units, safe-area padding, 44-pixel touch targets, and
  16-pixel mobile inputs handle browser chrome, device cutouts, touch accuracy,
  and iOS focus zoom.

## Scope changed

Runtime changes are frontend-only under `products/aloy/app`. The parent vision,
Surface Builder guidance, and this progress handoff now also define the mobile
and Expo compatibility contract. No backend API, Event semantics, authority, or
persisted data model changed.

## Planned Expo compatibility

The responsive web shell is the mobile information-architecture foundation,
not the Expo implementation. The native app should share product API/domain
contracts and durable replay behavior while implementing its shell with native
controls. Generated Surfaces remain sandboxed React web bundles embedded in a
WebView; a mobile transport adapter carries the same session-bound Surface
protocol. Native device capabilities are exposed only through validated typed
intents and existing authority rails.

## Verification

- `npm run lint`: passed.
- `npm run build`: passed; the existing large-chunk warning remains.
- `npm test`: `7 passed`, `0 failed`.
- `git diff --check`: passed after the final documentation update.
- Automated responsive browser screenshots were attempted through the approved
  in-app browser workflow but remain blocked by the known bootstrap failure
  `Cannot redefine property: process`.

## Next review

Run a manual device-width smoke at approximately 390 x 844 and tablet width,
covering Today, Life, Event setup, Event Conversation, Workbench/Surface, Event
context, and one long utility page. Confirm keyboard/composer behavior and both
portrait and landscape rotation before merging.
