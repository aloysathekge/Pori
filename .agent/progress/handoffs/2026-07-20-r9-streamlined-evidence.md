# R9 Streamlined Surface Evidence

## Outcome

The trusted browser gate still renders and inspects all 19 required Surface
compositions, but it no longer stores 19 PNG files or depends on a second model
before publication.

- Five baseline layouts retain screenshots: wide, split, tablet, mobile, and
  narrow mobile.
- Seven host-owned resource states are rendered at wide and mobile sizes. Their
  DOM, layout, accessibility, focus, and contrast observations are retained in
  the signed quality receipt with a deterministic fingerprint rather than a
  PNG.
- The receipt records runtime bootstrap, viewport matrix, state matrix,
  declared-interaction, and total inspection durations. Publication fails
  closed if the timing evidence is absent, invalid, or internally inconsistent.
- Focus Visible and deterministic text contrast remain blocking checks. The
  optional visual Critic is deferred and is not part of the V1 publish path.

## Why

This preserves broad runtime coverage while removing unnecessary image I/O,
object-store growth, and model latency. Human-readable visual evidence remains
available at the five layouts where it is most useful. Machine-verifiable state
coverage remains exact and content-bound.

## Verification

- 30 focused quality, publication, and build tests pass.
- The browser gate accepts a valid interactive Surface and rejects hidden focus
  plus low-contrast text.
- Ruff and diff integrity pass.

## Next

Add long-content and approval fixtures to the host-owned state contract, then
implement primary-job simulation. Use the new stage timings to optimize the
measured bottleneck rather than removing safety checks speculatively.
