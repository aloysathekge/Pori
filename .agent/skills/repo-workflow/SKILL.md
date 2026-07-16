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

Lead every implementation handoff with a plain-language product explanation:

1. **What changed** — the concrete behavior or capability introduced.
2. **What it does** — walk through the real user/system flow after the change.
3. **How it makes the system better** — name the reliability, safety, speed,
   usability, or product capability gained, including what remains unchanged.

Then report files changed, verification commands run, skipped checks,
assumptions, and the recommended next step. Do this automatically at the end of
every implementation phase; do not wait for the user to ask.
