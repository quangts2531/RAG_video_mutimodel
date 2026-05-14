#!/usr/bin/env bash
# ============================================================================
# entrypoint.sh — Startup script for the Video RAG Chatbot API container
#
# Steps:
#   1. Wait for Ollama LLM server to become reachable
#   2. Verify the RAG knowledge base file exists
#   3. Launch Uvicorn (FastAPI)
# ============================================================================

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
DOCUMENT_PATH="${DOCUMENT_PATH:-result_document.json}"
MAX_RETRIES=30
RETRY_INTERVAL=2

# ── Colors for logging ──────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[entrypoint]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[entrypoint]${NC} $*"; }
log_error() { echo -e "${RED}[entrypoint]${NC} $*"; }

# ── Step 1: Wait for Ollama ──────────────────────────────────────────────────
log_info "Waiting for Ollama at ${OLLAMA_HOST} ..."

retries=0
until python -c "import urllib.request; urllib.request.urlopen('${OLLAMA_HOST}/api/tags', timeout=3)" > /dev/null 2>&1; do
    retries=$((retries + 1))
    if [ "$retries" -ge "$MAX_RETRIES" ]; then
        log_error "Ollama not reachable after ${MAX_RETRIES} attempts. Aborting."
        exit 1
    fi
    log_warn "Ollama not ready (attempt ${retries}/${MAX_RETRIES}). Retrying in ${RETRY_INTERVAL}s..."
    sleep "$RETRY_INTERVAL"
done

log_info "Ollama is reachable at ${OLLAMA_HOST}"

# ── Step 2: Verify knowledge base ───────────────────────────────────────────
if [ ! -f "/app/data/${DOCUMENT_PATH}" ]; then
    log_error "Knowledge base file not found at /app/data/${DOCUMENT_PATH}"
    log_error "Mount it via: -v /path/to/result_document.json:/app/data/result_document.json:ro"
    exit 1
fi

log_info "Knowledge base found: /app/data/${DOCUMENT_PATH}"

# ── Step 3: Launch Uvicorn ───────────────────────────────────────────────────
log_info "Starting Uvicorn on 0.0.0.0:8000 ..."

exec uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    --access-log
