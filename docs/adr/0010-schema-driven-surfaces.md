# 0010 — Surfaces are schema-composed, not generated code

Date: 2026-07-11 (vision v1.0) · Status: **superseded 2026-07-16**

Superseded by the model-authored, sandboxed application decision in
[`../aloy-vision.md`](../aloy-vision.md) and its complete contract in
[`../aloy-surface-spec.md`](../aloy-surface-spec.md). This file remains only as
the record of the earlier decision; it is not active implementation guidance.

## Context
The Aloy vision's Surfaces could be AI-generated UI code in a sandbox
(flexible, unsafe, unreviewable) or model-composed typed schemas rendered by
trusted components.

## Decision
Schema DSL: the model composes from a component vocabulary; our components
render; DESIGN.md-style token files brand the output. No code execution in
the client. The vocabulary is a floor, not a ceiling.

## Consequences
Surfaces are versionable, diffable, validatable — agents can EDIT them, and
Verifiable Reality can gate what they assert. Freeform codegen artifacts were
deliberately deferred at the time. The later review retained schema validation,
truth rails, and trusted host components as the SDK/runtime boundary while
moving Event-specific composition into isolated model-authored React.
