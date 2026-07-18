"""Logging wrapper for structured LLM calls: ``ainvoke_structured`` runs
``llm.with_structured_output(...).ainvoke(...)`` and logs start / success /
failure with timing on the ``pori.llm`` logger. Prefer it over calling
structured output directly so the call shows up in LLM logs.
"""

import time
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from pori.llm import BaseMessage

from .logging_config import ensure_logger_configured

_llm_logger = ensure_logger_configured("pori.llm")


async def ainvoke_structured(
    llm,
    output_model: Type[BaseModel],
    messages: List[BaseMessage],
    *,
    include_raw: bool = False,
    meta: Optional[Dict[str, Any]] = None,
):
    meta = meta or {}
    start = time.perf_counter()
    _llm_logger.info("LLM call start", extra={"meta": meta})
    try:
        structured = llm.with_structured_output(output_model, include_raw=include_raw)
        resp = await structured.ainvoke(messages)
        elapsed = time.perf_counter() - start
        wrapped = isinstance(resp, dict)
        has_parsed = resp.get("parsed") is not None if wrapped else True
        log = _llm_logger.info if has_parsed else _llm_logger.warning
        log(
            "LLM call success" if has_parsed else "LLM structured output rejected",
            extra={
                "meta": {
                    **meta,
                    "elapsed_s": round(elapsed, 3),
                    "has_parsed": has_parsed,
                    "has_raw": wrapped and "raw" in resp,
                }
            },
        )
        return resp
    except Exception as e:
        elapsed = time.perf_counter() - start
        _llm_logger.error(
            f"LLM call failed: {e}",
            exc_info=True,
            extra={"meta": {**meta, "elapsed_s": round(elapsed, 3)}},
        )
        raise
