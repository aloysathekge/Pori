# pori/llm — direct-SDK provider wrappers (no LangChain)

## What this package owns
The one interface the rest of Pori talks to a model through
(`BaseChatModel`), the shared message types, and the resilience layer
around provider calls: retry, error classification, cross-provider
failover, prompt caching, and context-length awareness.

## Files
- `base.py` — `BaseChatModel` protocol: `ainvoke(messages, ...)`,
  `with_structured_output(model)`, tool binding. Everything upstream
  (agent, teams, evals) depends only on this.
- `messages.py` — `UserMessage` / `AssistantMessage` / `SystemMessage`,
  content blocks (`ImageBlock`, `DocumentBlock`), and tool-call/turn types.
  This is the provider-neutral wire format.
- `anthropic.py`, `openai.py`, `google.py` — the direct SDK wrappers; each
  also has a `StructuredWrapper` for structured output.
- `openrouter.py`, `fireworks.py` — OpenAI-compatible hosts; subclass
  `ChatOpenAI` and only override client construction.
  `openrouter_models.py` holds model metadata.
- `retry.py` — bounded exponential backoff + jitter for transient errors;
  detects by exception class name/status so it imports no provider SDKs.
- `error_classifier.py` — `classify_error` maps a failure to a
  `FailoverReason` with recovery hints (`retryable`, `should_compress`,
  provider-unavailable, ...). Consumed by the agent loop and failover.
- `failover.py` — `FailoverChatModel`: ordered chain of models; advances on
  provider-unavailability (after that model's own retries) and the switch
  is sticky. Context overflow deliberately does NOT fail over.
- `prompt_caching.py` — pure functions adding Anthropic `cache_control`
  breakpoints (tools → system → messages prefix order, max 4).
- `model_context.py` — context-length lookup by model-id substring;
  conservative default for unknown models. Sizes the history budget.
- `reasoning.py` — streaming `<think>...</think>` scrubber for tagged
  reasoning models (DeepSeek-R1 style), chunk-boundary safe.

## Key contracts
- Build instances via `pori.config.create_llm(LLMConfig(...))`, not by
  constructing wrappers directly.
- Providers normalize token usage (`normalize_usage`) so
  `pori/metrics.py` can price any call the same way.
- Retry handles *transient* faults inside one model; failover handles a
  *dead* model; classification decides which is which — keep those layers
  distinct.

## Change X → look at Y
- New provider → wrapper module implementing `BaseChatModel` + wiring in
  `__init__.py` and `pori/config.py`; OpenAI-compatible hosts should
  subclass `ChatOpenAI` like `openrouter.py`.
- Retries burning on permanent errors (or not firing) →
  `error_classifier.py` first, then `retry.py`.
- Cache hit-rate regressions → `prompt_caching.py` and whatever changed
  the stable prefix (tool schemas, system prompt).
- History too small / overflow loops → `model_context.py` +
  `pori/compression.py`.
