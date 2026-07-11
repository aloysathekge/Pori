# 0006 — Memory is an index over durable things

Date: 2026-07 (storage arc) · Status: accepted

## Context
Stuffing file contents or transcripts into memory bloats every prompt and
rots. An external brain needs recall over a growing life without growing
context.

## Decision
Memory holds POINTERS (typed records with provenance) — to stored files,
events, facts. Bytes live in the object store; agents materialize what a
task needs into their sandbox (page-fault style). Typed MemoryRecord
contract with scope, retention, supersession.

## Consequences
A 200MB file costs ~20 tokens of memory. Scales to the Aloy vision's
context architecture (docs/aloy-vision.md §5): context is a cache; state
lives as state; receipts win over memory on conflict.
