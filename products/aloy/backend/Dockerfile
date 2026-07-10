# Aloy backend image (monorepo build).
#
# Build context MUST be the MONOREPO ROOT (the Pori repo), because the
# backend's pyproject declares pori as an editable path dep (../../..):
#
#   docker build -f products/aloy/backend/Dockerfile -t aloy-backend:latest .
#
# or via products/aloy/backend/docker-compose.yml (context: ../../..).

# ---------- builder ----------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Preserve the monorepo layout inside the image so the backend's editable
# path dep (../../.. -> /app) resolves exactly as it does in dev.
WORKDIR /app
COPY pyproject.toml uv.lock* README.md /app/
COPY pori /app/pori
COPY products/aloy/backend /app/products/aloy/backend

WORKDIR /app/products/aloy/backend
# --frozen uses the committed uv.lock; --extra s3 bakes in boto3 so flipping
# STORAGE_BACKEND=s3 is config-only, no rebuild.
RUN uv sync --frozen --no-dev --extra s3

# ---------- runtime ----------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/products/aloy/backend/.venv/bin:$PATH" \
    # Durable dirs live on the /data volume (see docker-compose.yml):
    # blobs (when STORAGE_BACKEND=local) + per-conversation sandbox jails.
    STORAGE_DIR=/data/storage \
    SANDBOX_BASE_DIR=/data/sandbox

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash --uid 1000 app \
    && mkdir -p /data/storage /data/sandbox \
    && chown -R app:app /data

WORKDIR /app
COPY --from=builder --chown=app:app /app /app

USER app
WORKDIR /app/products/aloy/backend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/v1/health || exit 1

# ONE worker — non-negotiable. Live-run re-attach, clarify bridges, stop, and
# warm resume are in-process registries; a second worker splits them and those
# features fail intermittently. Scale by adding worker-service replicas (the
# durable-run executor), not API processes.
CMD ["uvicorn", "aloy_backend.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "127.0.0.1"]
