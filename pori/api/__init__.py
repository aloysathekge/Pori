from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from pori.utils.logging_config import setup_logging
from pori.api.routers import agents
from pori.api.deps import build_orchestrator
from pori.api.middleware import RequestIdMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the application.
    """
    # Startup:
    # Build the orchestrator and store it in the application state.
    # This is where slow initializations (like loading models or tools) should happen.
    app.state.orchestrator = build_orchestrator()
    yield
    # Shutdown:
    # (No shutdown actions needed for now)
    pass


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    setup_logging()
    logger = logging.getLogger("pori.api")

    app = FastAPI(title="Pori API", version="0.1.0", lifespan=lifespan)

    app.add_middleware(RequestIdMiddleware)

    app.include_router(agents.router, prefix="/v1", tags=["Agents"])

    @app.get("/v1/health")
    async def health() -> dict:
        return {"status": "ok"}

    logger.info("FastAPI app created: /v1/health and /v1/tasks ready")
    return app


# Convenience: allow `uvicorn pori.api:app` in addition to factory usage
app = create_app()
