# 0005 — The tool footprint ladder

Date: 2026 · Status: accepted (see CLAUDE.md for the full ladder)

## Context
Every registered tool's schema ships on EVERY LLM call. A new always-on tool
taxes the whole system's context budget forever.

## Decision
New capability climbs the lowest rung that works: extend an existing tool →
CLI+skill → gated tool (capability group / check_fn / per-run denied_tools)
→ plugin → MCP → core tool (last resort). Gated tools vanish from the model
surface when unusable (e.g. `fetch_my_file` with an empty library).

## Consequences
Context stays lean as capability grows. Per-run gating (the run surface)
is the product-side expression of the same rule.
