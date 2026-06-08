---
name: repo-workflow
description: Work safely in this repo using Aloy rules, source-of-truth notes, and verification guidance.
---

# Pori Core Workflow

Use this skill for tasks inside $repoRelative.

## Must Read First

- $repoRelative/AGENTS.md
- $repoRelative/.agent/agent.json
- $repoRelative/.agent/rules/repo-safety.md
- $repoRelative/.agent/rules/source-of-truth.md
- $repoRelative/.agent/rules/verification.md

## Workflow

1. Classify the task by area: app, backend, data, docs, integrations, deployment, or generated assets.
2. Read the smallest relevant file set before editing.
3. Preserve source-of-truth boundaries.
4. Apply the safe verification path in .agent/rules/verification.md.
5. Report skipped checks, approval-gated actions, and remaining risks.

## Approval Required

env/credential changes, package publishing, production-impacting framework changes

## Output

Report files changed, verification commands run, skipped checks, and assumptions.
