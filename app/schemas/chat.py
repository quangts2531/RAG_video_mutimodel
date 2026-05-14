"""
Pydantic v2 schemas for the chat endpoint.
"""

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat request body."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's query to search against the video knowledge base.",
        examples=["a person walking on the street"],
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Existing session ID to continue a conversation. "
            "If omitted, a new conversation is created."
        ),
    )


class SearchResult(BaseModel):
    """A single search result returned by the AI Agent."""

    content: str = Field(..., description="Matched document content (audio + visual text).")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata: video_name, segment_id, start_time, end_time.",
    )


class ChatResponse(BaseModel):
    """Response returned by the POST /api/chat endpoint."""

    session_id: str = Field(..., description="The conversation session ID.")
    answer: str = Field(
        ...,
        description="The AI-generated answer based on the video knowledge base.",
    )
    message_id: int = Field(..., description="Database ID of the stored AI message.")
