from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger("pori_cloud")

# Context var so any code can access the current request ID
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assign a unique request ID to every request."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        rid = request.headers.get("X-Request-ID", uuid.uuid4().hex[:16])
        request_id_var.set(rid)
        request.state.request_id = rid

        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled error", extra={"request_id": rid})
            return Response(
                content=f'{{"detail":"Internal server error","request_id":"{rid}"}}',
                status_code=500,
                media_type="application/json",
            )

        response.headers["X-Request-ID"] = rid
        return response
