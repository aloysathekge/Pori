# 0010 — Surfaces are schema-composed, not generated code

Date: 2026-07-11 (vision v1.0) · Status: accepted

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
Verifiable Reality can gate what they assert. Freeform codegen artifacts are
deliberately deferred (docs/aloy-vision.md §8).
