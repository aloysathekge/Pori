# Pori AI Agent API - Production Dockerfile
FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Create non-root user for security
RUN groupadd --gid 1000 pori \
    && useradd --uid 1000 --gid pori --shell /bin/bash --create-home pori

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY pori/ ./pori/
COPY pyproject.toml .
COPY config.example.yaml .

# Install the package (no deps - requirements.txt has them)
RUN pip install --no-cache-dir --no-deps .

# Create config.yaml from example (API uses env vars; config is fallback for CLI)
RUN cp config.example.yaml config.yaml

# Ensure pori can read/write
RUN chown -R pori:pori /app

USER pori

EXPOSE 8000

# Run the API server
CMD ["uvicorn", "pori.api:app", "--host", "0.0.0.0", "--port", "8000"]
