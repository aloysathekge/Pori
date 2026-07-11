# 0003 — Sibling-module method binding over mixins

Date: 2026-07 · Status: accepted

## Context
Splitting a god-class usually means mixins (MRO opacity, scattered state) or
composition (breaking `self` access for tightly-coupled state).

## Decision
Cohesive method groups live in sibling modules as plain functions taking
`self`, bound onto `Agent` in core.py:
`execute_actions = _dispatch.execute_actions`. One class, many files, zero
MRO.

## Consequences
Extraction is transparent to callers and tests. The binding block in core.py
is the single map of what lives where. Type checkers see through it.
