"""
FastAPI application entry point.

Responsibilities:
  - Create the FastAPI app with metadata.
  - Register the ``lifespan`` event to load the AI Agent exactly once.
  - Create database tables on startup.
  - Attach CORS middleware and global exception handlers.
  - Include all API routers.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from app.api.routes import chat as chat_routes
from app.api.routes import conversations as conversation_routes
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.models.database import Base, engine
from app.services.ai_service import AIService

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)

# ── Settings ─────────────────────────────────────────────────────────────────

settings = get_settings()


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown.

    Startup:
      1. Create all database tables (idempotent).
      2. Load the AI Agent once via ``AIService.initialize()``.
         Runs in a threadpool because Agent.__init__ is blocking.

    Shutdown:
      Placeholder for future cleanup (e.g. closing connections).
    """
    # ── STARTUP ──────────────────────────────────────────────────
    logger.info("Creating database tables…")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready.")

    logger.info("Initialising AI Agent (blocking — running in threadpool)…")
    await run_in_threadpool(AIService.initialize)
    logger.info("AI Agent initialised and ready to serve requests.")

    yield

    # ── SHUTDOWN ─────────────────────────────────────────────────
    logger.info("Application shutting down — goodbye.")


# ── Application Factory ──────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "REST API for the Video RAG Chatbot. "
        "Performs similarity search over video segments using "
        "HuggingFace embeddings and Qdrant vector store."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Exception Handlers ───────────────────────────────────────────────────────

register_exception_handlers(app)

# ── Routers ──────────────────────────────────────────────────────────────────

app.include_router(chat_routes.router, prefix="/api")
app.include_router(conversation_routes.router, prefix="/api")


# ── Health Check ─────────────────────────────────────────────────────────────

@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
    description="Returns 200 if the service is running.",
)
async def health_check() -> dict[str, str]:
    """Simple liveness probe."""
    return {"status": "healthy", "version": settings.APP_VERSION}
