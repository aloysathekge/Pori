# Pori Core Explore

Explore this repo in read-only mode.

## Must Read

1. AGENTS.md
2. .agent/agent.json
3. .agent/rules
4. .agent/skills/repo-workflow/SKILL.md
5. repo manifests such as package.json, pyproject.toml, README files, and config files when present

## Workflow

1. Identify whether the task touches app code, backend logic, data, docs, integrations, deployment, or generated assets.
2. Map the smallest relevant file set before recommending edits.
3. Report safe verification commands from .agent/rules/verification.md.
4. Call out external sources of truth from .agent/rules/source-of-truth.md.
5. Do not modify files during exploration.

## Rules

- Do not read secret values.
- Do not run servers, deploys, migrations, paid jobs, or destructive cleanup during exploration.
- Treat verification command detection as a signal, not proof that verification is complete.
