"""
Context variables for the Pori framework.

This module contains context variables that can be used across different
parts of the application without creating circular imports.
"""

from contextvars import ContextVar

# A context variable to hold the request ID.
# This allows the ID to be accessed by any part of the application
# during the lifecycle of a single request.
request_id_var: ContextVar[str] = ContextVar("request_id", default=None)
