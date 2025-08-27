import uuid
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseFunction
from starlette.requests import Request
from starlette.responses import Response
from pori.utils.context import request_id_var


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Injects a unique request_id into every incoming request's context.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseFunction
    ) -> Response:
        # Generate a new request ID
        request_id = str(uuid.uuid4())

        # Set the request ID in the context variable
        request_id_var.set(request_id)

        # Add the request ID to the response headers
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response
