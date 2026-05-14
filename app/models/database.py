"""
SQLAlchemy database engine, session factory, and ORM models.

Tables:
  - conversations: Tracks chat sessions.
  - messages:      Stores individual user / AI messages within a conversation.
"""

import uuid
from datetime import datetime, timezone
from typing import List

from sqlalchemy import (
    Column,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from app.core.config import get_settings

# ── Engine & Session ─────────────────────────────────────────────────────────

_settings = get_settings()

engine = create_engine(
    _settings.DATABASE_URL,
    connect_args={"check_same_thread": False},  # Required for SQLite
    echo=_settings.DEBUG,
)

SessionLocal: sessionmaker[Session] = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# ── Declarative Base ─────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# ── Helper ───────────────────────────────────────────────────────────────────

def _generate_uuid() -> str:
    """Generate a UUID4 string for use as a primary key."""
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


# ── ORM Models ───────────────────────────────────────────────────────────────

class Conversation(Base):
    """Represents a chat session / conversation."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_generate_uuid
    )
    session_id: Mapped[str] = mapped_column(
        String(36), unique=True, index=True, default=_generate_uuid
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    # Relationship
    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.timestamp",
    )

    def __repr__(self) -> str:
        return f"<Conversation(id={self.id!r}, session_id={self.session_id!r})>"


class Message(Base):
    """A single message (user or AI) within a conversation."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(
        SAEnum("user", "ai", name="message_role"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    # Relationship
    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="messages"
    )

    def __repr__(self) -> str:
        return (
            f"<Message(id={self.id!r}, role={self.role!r}, "
            f"conversation_id={self.conversation_id!r})>"
        )
