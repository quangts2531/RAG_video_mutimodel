"""
FastAPI dependency injection helpers.

Provides:
  - ``get_db``:         Yields a SQLAlchemy session (auto-closes).
  - ``get_ai_service``: Returns the singleton AIService instance.
"""

from typing import Generator

from sqlalchemy.orm import Session

from app.models.database import SessionLocal
from app.services.ai_service import AIService


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and guarantee cleanup via ``finally``."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_ai_service() -> AIService:
    """Return the pre-initialised AIService singleton.

    This is safe to call inside ``Depends(...)`` because the singleton
    was created during the ``lifespan`` startup event.
    """
    return AIService.get_instance()
