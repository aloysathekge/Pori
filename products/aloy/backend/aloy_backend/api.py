"""The ASGI application: lifespan wiring (logging, DB init, sandbox
configuration — streamed runs execute in this process), request-ID and CORS
middleware, and the aggregate ``routes.router`` mounted under ``/v1``. Served
by ``main.run()`` as ``aloy_backend.api:app``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import init_db
from .logging_config import setup_logging
from .middleware import RequestIdMiddleware
from .routes import router

logger = logging.getLogger("aloy_backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Starting Pori Cloud")
    await init_db()
    logger.info("Database initialized")
    # Streamed runs execute in THIS process (not just the worker), so the
    # sandbox backend must be configured here too.
    from .orchestrator import configure_sandbox

    configure_sandbox()
    yield
    logger.info("Shutting down Pori Cloud")


app = FastAPI(title="Pori Cloud", version="0.1", lifespan=lifespan)

# Middleware (order matters — outermost first)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router, prefix="/v1")


@app.get("/v1/health")
async def health():
    return {"status": "ok", "version": "0.1"}
