# R9 host-owned Surface resource states

## Outcome

The Surface runtime no longer collapses absent, empty, and failed data into the
same empty collection. Every manifest capability now receives a versioned
host-owned resource state. Generated React reads that state through
`useSurfaceResourceState` and binds `feedbackProps` to a visible primary region.

Trusted inspection uses the same public context to exercise loading, empty,
stale, error, permission-denied, pending, and indeterminate states at wide and
mobile sizes. The gate requires the bound region to transition to each state,
checks runtime/layout/basic accessibility, retains 14 state captures, and binds
their hashes and observations into `aloy-surface-quality@2`. Interaction checks
are reset to canonical Event context after state inspection.

## Architecture decisions

- No inspection-only flag is exposed to generated code.
- The host owns state truth; an empty array alone never means failure.
- State fixtures transform the same context shape used by a live Event Surface.
- Remote builders must return explicit state-inspection evidence before their
  build can become publication-eligible.

## Next

Add deterministic keyboard focus-indicator and contrast evidence, followed by
long-content and approval fixtures. Only then add the independent Critic.
