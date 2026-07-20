# R9 long-content and approval scenarios

## Outcome

Quality policy `aloy-surface-quality@4` now requires state policy
`aloy-surface-states@2`. The trusted browser renders two additional scenarios
at wide and mobile sizes without exposing an inspection-only switch:

- `long_content` fills ordinary declared Event resources with bounded dense
  records and long copy, then applies the existing overflow, clipping,
  accessibility, and contrast audits;
- `approval_required` projects a pending Proposal and a durable
  `waiting_approval` Interaction. Generated UI binds a visible summary with
  `useSurfaceApprovalState`, while Approve and Reject remain host-owned.

Action-only `ask_aloy` is no longer misclassified as a data resource. The gate
now inspects 23 compositions, retains only five baseline PNGs, and stores 18
compact state/scenario fingerprints.

## Verification

- 47 focused resource-state, quality, publication, and build tests pass.
- Real-browser proofs accept the SDK-bound approval summary and reject missing
  approval binding plus long-content horizontal overflow.
- `@aloy/surface` TypeScript build, Ruff, and mypy across 126 backend files pass.

## Next

Implement primary-job simulation with accessible semantic paths and typed
outcome assertions over canonical Event state. Keep any optional visual Critic
outside the V1 publication path.
