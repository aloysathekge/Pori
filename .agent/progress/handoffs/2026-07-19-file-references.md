# Aloy file references — 2026-07-19

## Branch

`aloy-v1-file-references` from `aloy-v1`. Work remains uncommitted.

## What changed

- Added direct uploads to **My files** with sequential progress and immediate
  retained-file availability.
- Added a scoped Conversation file-search endpoint and reusable frontend hook.
- Added composer `@` search and a `+` menu for device upload or choosing an
  existing file. Chosen files become removable trusted-reference chips.
- Dedicated Events list and expose only their own uploads and artifacts. Life
  can intentionally choose retained files across the user's Events.
- Message assembly resolves file IDs against tenant, user, and Event ownership.
  Only accepted choices become message metadata and selected-file task context.
- Durable worker and team paths now retain assembled file context instead of
  falling back to the user's plain message text.
- Run library capabilities and saved-file memory pointers now respect the Event
  boundary. Legacy unscoped file-library pointers are ignored.
- The composer supports ten file chips total, with subtype ceilings of three
  inline text files and three native documents, plus three images.
- Recorded the product contract in `docs/aloy-vision.md`.

## Verification

- Backend upload/library focus: `27 passed`.
- Worker, queue/resume, approval, Surface SDK, Event memory, and Event context
  cross-path regression set: `53 passed`.
- Aloy backend mypy: `117` source files clean.
- Pori kernel mypy: `109` source files clean.
- Changed backend Black and Ruff checks pass.
- Aloy app ESLint and production build pass. Vite retains the existing
  large-chunk warning.
- `git diff --check` passes.

The complete backend suite was attempted with a five-minute command deadline,
but the quiet run exceeded that deadline before emitting a summary; it is not
claimed as a pass. Automated browser inspection remains blocked by the known
desktop bootstrap error `Cannot redefine property: process`.

## Recommended manual smoke

1. In a dedicated Event, upload two files with `+`; verify `@` lists those
   files and does not list files from another dedicated Event.
2. Select one file with `@`, send a question, and confirm the chip persists on
   the message and Aloy receives the selected filename/path context.
3. Open Life and verify `@` can find retained files from multiple Events.
4. Upload directly in **My files**, then choose it from a Life composer through
   both `@` and **+ → Choose existing file**.
5. Check the composer at a narrow mobile width: picker, action menu, chips, and
   upload progress must remain reachable without horizontal overflow.

## Next step

Perform the manual visual smoke, address any interaction polish discovered,
then commit, push, and open a PR only when requested.
