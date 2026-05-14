# ============================================================================
# Multi-stage Dockerfile for Video RAG Chatbot API
# Base: python:3.11.9-slim-bookworm  (pinned, no "latest")
# ============================================================================

# ── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.11.9-slim-bookworm AS builder

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install only the build-time OS deps needed to compile native wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment so we can copy it cleanly to the runtime stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies (cached unless requirements change)
COPY requirements-api.txt /tmp/requirements-api.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements-api.txt


# ── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.11.9-slim-bookworm AS runtime

LABEL maintainer="saigonfilm" \
      description="Video RAG Chatbot API — FastAPI + HuggingFace Embeddings + Qdrant (in-memory)" \
      version="1.0.0"

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install minimal runtime OS deps (curl for health checks, sqlite3 for debug)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── Non-root user for security ──────────────────────────────────────────────
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# ── Application directory ───────────────────────────────────────────────────
WORKDIR /app

# Copy application source code
COPY app/ ./app/
COPY chat.py ./chat.py
COPY main.py ./main.py
COPY entrypoint.sh ./entrypoint.sh

# Ensure entrypoint is executable
RUN chmod +x ./entrypoint.sh

# ── Directories for runtime data (will be mounted as volumes) ───────────────
RUN mkdir -p /app/data /app/db && \
    chown -R appuser:appuser /app

# ── HuggingFace cache directory ─────────────────────────────────────────────
# Model downloads go here; mount as a volume to persist across rebuilds
ENV HF_HOME="/app/.hf_cache"
RUN mkdir -p /app/.hf_cache && chown -R appuser:appuser /app/.hf_cache

# Switch to non-root user
USER appuser

# ── Expose API port ─────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entrypoint ──────────────────────────────────────────────────────────────
ENTRYPOINT ["./entrypoint.sh"]
