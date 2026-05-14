"""
Conversation routes — read-only endpoints for browsing past conversations.

Endpoints:
  GET /api/conversations              — Paginated list of conversations.
  GET /api/conversations/{session_id} — Single conversation with all messages.
"""

import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.errors import NotFoundError
from app.repositories import conversation_repo
from app.schemas.conversation import (
    ConversationListResponse,
    ConversationSchema,
)

logger: logging.Logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["Conversations"])


@router.get(
    "",
    response_model=ConversationListResponse,
    summary="List all conversations",
    description="Returns a paginated list of conversations, most recent first.",
)
async def list_conversations(
    skip: int = Query(0, ge=0, description="Number of records to skip."),
    limit: int = Query(20, ge=1, le=100, description="Max records to return."),
    db: Session = Depends(get_db),
) -> ConversationListResponse:
    """Return a paginated list of conversations."""
    conversations, total = conversation_repo.list_conversations(db, skip=skip, limit=limit)
    return ConversationListResponse(
        conversations=[
            ConversationSchema.model_validate(c) for c in conversations
        ],
        total=total,
        skip=skip,
        limit=limit,
    )


@router.get(
    "/{session_id}",
    response_model=ConversationSchema,
    summary="Get a conversation by session ID",
    description="Returns a single conversation with all its messages.",
)
async def get_conversation(
    session_id: str,
    db: Session = Depends(get_db),
) -> ConversationSchema:
    """Return a single conversation with its messages."""
    conversation = conversation_repo.get_conversation_by_session_id(db, session_id)
    if conversation is None:
        raise NotFoundError(resource="Conversation", identifier=session_id)
    return ConversationSchema.model_validate(conversation)
