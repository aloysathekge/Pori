# 0002 — The agent loop stays whole in `agent/core.py`

Date: 2026-07 (agent package split) · Status: accepted

## Context
`agent/core.py` grew past 1900 lines. Cohesive method groups were extracted,
but scattering the Plan→Act→Reflect→Evaluate loop across files would make
the kernel's most important control flow unreadable.

## Decision
`run()` and `step()` never leave core.py. Everything else (prompting,
planning, artifacts, authorization, dispatch, completion gates) may live in
sibling modules bound onto the class (see ADR 0003).

## Consequences
An agent (human or AI) reads the loop top-to-bottom in one file. Extractions
must respect the boundary: policy moves out, control flow stays.
