# 0001 — Direct SDK calls, no LangChain

Date: 2025 (project inception) · Status: accepted

## Context
Agent frameworks built on LangChain inherit its abstractions, breaking
changes, and debugging opacity. Pori's value is the loop, memory, and trust
rails — not chain plumbing.

## Decision
Pori talks to Anthropic/OpenAI/Google through their own SDKs behind one
small interface (`pori/llm/base.py::BaseChatModel`); shared message types in
`pori/llm/messages.py`.

## Consequences
We own provider quirks (normalization lives in our adapters) and gain full
control of streaming, tool-calling, and token accounting. Adding a provider
= one adapter file.
