# Today focus stream design QA

- Source visual truth: `C:\Users\sathe\.codex\generated_images\019f61c8-b311-7972-9b39-85e3f722aa20\exec-0792bf76-63d2-43de-8c71-db23dd303815.png`
- Implementation route: `http://localhost:5173/today`
- Intended viewport: 1440 × 1024
- Intended state: authenticated Today screen with the Notifications modal open
- Implementation screenshot: not captured

## Full-view comparison evidence

The source visual was opened at original resolution. The local API, worker, and
Vite app are running, but the desktop browser-control bridge could not be
initialized in this environment. An authenticated implementation screenshot
therefore could not be captured and no visual comparison is claimed.

## Focused region comparison evidence

Blocked with the full-view capture. The required focus regions are the primary
attention block, Notifications modal, Important emails table, and responsive
header actions.

## Findings

- [P1] Browser-rendered fidelity is unverified.
  - Location: Today screen at 1440 × 1024, Notifications open.
  - Evidence: source image is available; implementation screenshot is missing.
  - Impact: typography, spacing, modal placement, overflow, and authenticated
    data states cannot be accepted from build output alone.
  - Fix: capture the authenticated local screen at 1440 × 1024 with the bell
    modal open, compare it with the source in one visual input, then correct all
    P0–P2 differences.

## Primary interactions tested

Static verification covers TypeScript, lint, production build, backend typing,
and the Today/email API tests. Browser clicks, focus trapping, Escape dismissal,
mark-all-read, approval review, and external email links remain visually
unverified.

## Console errors checked

Not checked because browser capture is blocked.

## Comparison history

- Initial pass: blocked before implementation capture; no visual fixes claimed.

## Implementation checklist

- Capture the authenticated screen with Notifications open.
- Compare the source and implementation at the same viewport.
- Verify modal focus, Escape, backdrop dismissal, and focus restoration.
- Verify connected, disconnected, empty, loading, and provider-error email states.
- Fix all P0–P2 visual and interaction findings and repeat the comparison.

## Follow-up polish

None classified until the first visual comparison is available.

final result: blocked
