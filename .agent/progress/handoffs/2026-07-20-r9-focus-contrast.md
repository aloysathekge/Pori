# R9 deterministic focus and contrast evidence

## Outcome

Trusted Surface inspection now performs real keyboard traversal at wide, split,
tablet, mobile, and narrow-mobile sizes. It records the visited order and focus
styles, blocks controls that Tab cannot reach, premature focus cycles, and any
focused control without a visible indicator. A 2px indicator with at least 3:1
outline contrast is recorded as stronger evidence, but publication claims only
the WCAG 2.2 AA Focus Visible requirement.

The browser measures visible text against its effective solid background in all
five baseline compositions and all 14 public-state compositions. Normal text
must reach 4.5:1 and large text 3:1. Image/gradient backdrops that prevent a
deterministic measurement fail closed; the Builder is instructed to place text
on a solid provable backdrop.

Both evidence blocks are bound into `aloy-surface-quality@3`. Remote builders
must provide explicit focus and contrast inspection flags plus the complete
evidence before publication.

## Next

Add host-owned long-content and approval fixtures, then primary-job simulation.
An optional visual Critic remains deferred and outside the publication gate.
