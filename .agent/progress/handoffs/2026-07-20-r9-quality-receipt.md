# R9 exact-build Surface quality receipt

## Outcome

Started `aloy-v1-r9-surface-quality` from the PR #198 merge on `aloy-v1` and
completed the first R9 slice. Trusted preview now writes a
fingerprinted `aloy-surface-quality@1` receipt into the durable Surface Build
metrics. It binds the build id, revision id, source checksum, retained bundle
hash, deterministic validation, executable runtime inspection, declared
interaction inspection, diagnostics, policy version, and inspection time.

New publication fails closed when the receipt is absent, failed, altered, or
bound to different source/bundle content. Rollback deliberately continues to
accept a previously published legacy last-good build, so tightening the new
publication policy cannot remove recovery.

The receipt now also requires a host-owned viewport matrix covering wide,
split, tablet, 390px mobile, and 360px narrow mobile. The browser records layout
and basic DOM accessibility observations, captures each viewport, retains local
PNGs beside the bundle, and binds capture hashes into the receipt. Page
overflow, clipped controls, missing main landmarks, unnamed controls, missing
image alternatives, keyboard-unreachable custom controls, duplicate ids, or
capture/storage failure block publication. State variants, focus-indicator and
contrast analysis, remote capture transport, Critic, and primary-job evidence
remain explicit follow-ups.

## Files

- `products/aloy/backend/aloy_backend/surface_quality.py`
- `products/aloy/backend/aloy_backend/surface_builds.py`
- `products/aloy/backend/aloy_backend/surface_pipeline.py`
- `products/aloy/backend/aloy_backend/surface_publication.py`
- `products/aloy/backend/aloy_backend/surface_runtime_inspection.py`
- `products/aloy/backend/aloy_backend/storage.py`
- `products/aloy/backend/aloy_backend/product_skills/surface-builder/SKILL.md`
- `products/aloy/backend/tests/test_surface_quality.py`
- `products/aloy/backend/tests/test_surface_publication.py`
- `products/aloy/backend/tests/test_surface_builds.py`
- `products/aloy/backend/tests/test_storage.py`
- `docs/aloy-v1-plan.md`
- `.agent/progress/current.md`

## Verification

- Black, Ruff, and isort pass on every changed Python file.
- The backend's exact CI mypy gate passes: `125 source files`.
- Surface build/pipeline/publication focused suite: `31 passed`.
- All Surface-related backend tests plus Event Surface, R4 live Surface, and
  Release-B compatibility: `82 passed`.
- The full backend suite exceeded the bounded six-minute command deadline
  without reporting a failure; do not record it as passing or failing.

## Next

Add the host-owned state-fixture contract and deterministic focus-indicator and
contrast evidence. Only after those artifacts are durable should the independent
vision-capable Surface Critic be added.
