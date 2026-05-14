"""
Pydantic v2 schemas for Conversation and Message resources.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class MessageSchema(BaseModel):
    """Schema for a single message within a conversation."""

    id: int
    role: str = Field(..., description="Message author: 'user' or 'ai'.")
    content: str
    timestamp: datetime

    model_config = {"from_attributes": True}


class ConversationSchema(BaseModel):
    """Schema for a conversation with its messages."""

    id: str
    session_id: str
    created_at: datetime
    messages: list[MessageSchema] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class ConversationListResponse(BaseModel):
    """Paginated list of conversations."""

    conversations: list[ConversationSchema]
    total: int
    skip: int
    limit: int
