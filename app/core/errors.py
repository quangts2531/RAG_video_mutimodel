"""
Standardized error handling for the AI Chatbot API.

Defines custom exception classes and a global exception handler that
returns a consistent JSON structure while keeping raw error details
in server logs only.
"""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger: logging.Logger = logging.getLogger(__name__)


# ── Error Code Registry ─────────────────────────────────────────────────────

class ErrorCode:
    """Centralised error code constants."""

    AI_TIMEOUT: str = "AI_001"
    AI_OUT_OF_MEMORY: str = "AI_002"
    AI_GENERIC: str = "AI_003"
    DB_ERROR: str = "DB_001"
    VALIDATION_ERROR: str = "VAL_001"
    NOT_FOUND: str = "NOT_FOUND"


# ── Custom Exceptions ────────────────────────────────────────────────────────

class AIServiceError(Exception):
    """Raised when the AI Agent encounters an error."""

    def __init__(
        self,
        message: str = "An error occurred in the AI service",
        code: str = ErrorCode.AI_GENERIC,
        original_error: Exception | None = None,
    ) -> None:
        self.message = message
        self.code = code
        self.original_error = original_error
        super().__init__(self.message)


class DatabaseError(Exception):
    """Raised when a database operation fails."""

    def __init__(
        self,
        message: str = "A database error occurred",
        original_error: Exception | None = None,
    ) -> None:
        self.message = message
        self.code = ErrorCode.DB_ERROR
        self.original_error = original_error
        super().__init__(self.message)


class NotFoundError(Exception):
    """Raised when a requested resource is not found."""

    def __init__(self, resource: str = "Resource", identifier: str = "") -> None:
        self.message = f"{resource} '{identifier}' not found"
        self.code = ErrorCode.NOT_FOUND
        super().__init__(self.message)


# ── Standardised Error Response ──────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """JSON schema returned to the client on error."""

    error: str
    message: str
    code: str


# ── Exception Handlers ──────────────────────────────────────────────────────

def register_exception_handlers(app: FastAPI) -> None:
    """Attach global exception handlers to the FastAPI application."""

    @app.exception_handler(AIServiceError)
    async def ai_service_error_handler(
        request: Request, exc: AIServiceError
    ) -> JSONResponse:
        # Log the full traceback for debugging — never expose to client
        logger.exception(
            "AIServiceError [%s]: %s | path=%s",
            exc.code,
            exc.message,
            request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal Server Error",
                message="The AI service encountered an error. Please try again later.",
                code=exc.code,
            ).model_dump(),
        )

    @app.exception_handler(DatabaseError)
    async def database_error_handler(
        request: Request, exc: DatabaseError
    ) -> JSONResponse:
        logger.exception(
            "DatabaseError [%s]: %s | path=%s",
            exc.code,
            exc.message,
            request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal Server Error",
                message="A database error occurred. Please try again later.",
                code=exc.code,
            ).model_dump(),
        )

    @app.exception_handler(NotFoundError)
    async def not_found_error_handler(
        request: Request, exc: NotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error="Not Found",
                message=exc.message,
                code=exc.code,
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all for any unhandled exceptions — prevents raw tracebacks
        from leaking to the client."""
        logger.exception(
            "Unhandled exception: %s | path=%s",
            str(exc),
            request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal Server Error",
                message="An unexpected error occurred. Please try again later.",
                code=ErrorCode.AI_GENERIC,
            ).model_dump(),
        )
