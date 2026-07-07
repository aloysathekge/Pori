# Pori Cloud API image.
#
# Build context MUST be the parent directory containing both `Pori/` and
# `aloy_backend/` side-by-side, because aloy_backend's pyproject.toml declares
# pori as an editable local path dep (`../Pori`).
#
#   docker build -f aloy_backend/Dockerfile -t aloy-backend:latest ..
#
# or equivalently via docker-compose.yml (context: ..).

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

WORKDIR /app
COPY Pori /app/Pori
COPY aloy_backend /app/aloy_backend

WORKDIR /app/aloy_backend
# --frozen uses the committed uv.lock. --no-dev skips the [dev] extras.
RUN uv sync --frozen --no-dev

# ---------- runtime ----------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/aloy_backend/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash --uid 1000 app

WORKDIR /app
COPY --from=builder --chown=app:app /app /app

USER app
WORKDIR /app/aloy_backend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/v1/health || exit 1

CMD ["uvicorn", "aloy_backend.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "127.0.0.1"]
