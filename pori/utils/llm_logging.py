import time
from typing import Any, Dict, List, Optional, Type
from langchain.schema import BaseMessage
from pydantic import BaseModel
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
        _llm_logger.info(
            "LLM call success",
            extra={
                "meta": {
                    **meta,
                    "elapsed_s": round(elapsed, 3),
                    "has_parsed": bool(resp.get("parsed")),
                    "has_raw": "raw" in resp,
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
