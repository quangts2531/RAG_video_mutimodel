"""
Chat route — POST /api/chat

Flow:
  1. Validate request (ChatRequest).
  2. Resolve or create a Conversation by session_id.
  3. Persist the user message.
  4. Query the AI Agent (async, threadpool-offloaded).
  5. Persist the AI response.
  6. Return ChatResponse.
"""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_ai_service, get_db
from app.core.errors import DatabaseError
from app.repositories import conversation_repo
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.ai_service import AIService

logger: logging.Logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "",
    response_model=ChatResponse,
    summary="Send a message to the Video RAG chatbot",
    description=(
        "Sends a query to the AI Agent which performs similarity search "
        "against the video knowledge base and returns a generated answer."
    ),
)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    ai_service: AIService = Depends(get_ai_service),
) -> ChatResponse:
    """Handle an incoming chat message."""

    # ── 1. Resolve or create Conversation ────────────────────────
    try:
        conversation = None
        if request.session_id:
            conversation = conversation_repo.get_conversation_by_session_id(
                db, request.session_id
            )

        if conversation is None:
            conversation = conversation_repo.create_conversation(
                db, session_id=request.session_id
            )
    except Exception as exc:
        raise DatabaseError(
            message=f"Failed to resolve conversation: {exc}",
            original_error=exc,
        ) from exc

    # ── 2. Save user message ─────────────────────────────────────
    try:
        conversation_repo.add_message(
            db,
            conversation_id=conversation.id,
            role="user",
            content=request.message,
        )
    except Exception as exc:
        raise DatabaseError(
            message=f"Failed to save user message: {exc}",
            original_error=exc,
        ) from exc

    # ── 3. Query AI Agent (non-blocking) ─────────────────────────
    # AIServiceError is raised internally and handled by the global handler
    answer: str = await ai_service.chat(request.message)

    # ── 4. Save AI response ──────────────────────────────────────
    try:
        ai_message = conversation_repo.add_message(
            db,
            conversation_id=conversation.id,
            role="ai",
            content=answer,
        )
    except Exception as exc:
        raise DatabaseError(
            message=f"Failed to save AI response: {exc}",
            original_error=exc,
        ) from exc

    # ── 5. Build response ────────────────────────────────────────
    return ChatResponse(
        session_id=conversation.session_id,
        answer=answer,
        message_id=ai_message.id,
    )

