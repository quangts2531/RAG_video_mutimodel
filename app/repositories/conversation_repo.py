"""
Repository layer for Conversation and Message CRUD operations.

All database interactions are isolated here so that the service and
route layers remain decoupled from SQLAlchemy specifics.
"""

import logging
from typing import Sequence

from sqlalchemy.orm import Session, joinedload

from app.models.database import Conversation, Message

logger: logging.Logger = logging.getLogger(__name__)


# ── Conversation CRUD ────────────────────────────────────────────────────────

def create_conversation(db: Session, session_id: str | None = None) -> Conversation:
    """Create a new conversation. If *session_id* is ``None``, one is auto-generated."""
    conversation = Conversation()
    if session_id is not None:
        conversation.session_id = session_id
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    logger.info("Created conversation id=%s session_id=%s", conversation.id, conversation.session_id)
    return conversation


def get_conversation_by_session_id(
    db: Session, session_id: str
) -> Conversation | None:
    """Fetch a conversation by its public *session_id*, eagerly loading messages."""
    return (
        db.query(Conversation)
        .options(joinedload(Conversation.messages))
        .filter(Conversation.session_id == session_id)
        .first()
    )


def get_conversation_by_id(db: Session, conversation_id: str) -> Conversation | None:
    """Fetch a conversation by internal *id*."""
    return (
        db.query(Conversation)
        .options(joinedload(Conversation.messages))
        .filter(Conversation.id == conversation_id)
        .first()
    )


def list_conversations(
    db: Session, skip: int = 0, limit: int = 20
) -> tuple[Sequence[Conversation], int]:
    """Return a paginated list of conversations and the total count."""
    total: int = db.query(Conversation).count()
    conversations: Sequence[Conversation] = (
        db.query(Conversation)
        .order_by(Conversation.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return conversations, total


# ── Message CRUD ─────────────────────────────────────────────────────────────

def add_message(
    db: Session,
    conversation_id: str,
    role: str,
    content: str,
) -> Message:
    """Add a message to an existing conversation."""
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    logger.debug(
        "Added message id=%s role=%s to conversation=%s",
        message.id,
        role,
        conversation_id,
    )
    return message


def get_messages_by_conversation(
    db: Session, conversation_id: str
) -> Sequence[Message]:
    """Return all messages for a conversation, ordered by timestamp."""
    return (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.timestamp.asc())
        .all()
    )
