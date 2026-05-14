"""
Automated tests for the Video RAG Chatbot API.

Strategy:
  - The heavy AI model (Agent) is fully MOCKED — no model is loaded into RAM.
  - An in-memory SQLite database is used (not the production DB).
  - Tests verify both HTTP responses and database state.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# ── Patch the Agent BEFORE any app module tries to import it ─────────────────
# This prevents chat.py's Agent class from loading the real model.

_mock_agent_instance = MagicMock()


def _mock_chat(query: str):
    """Simulated chat that sleeps 1s and returns a fake response."""
    time.sleep(1)
    return "This is a simulated response from the AI"


_mock_agent_instance.chat = _mock_chat


# ── Test-scoped database (in-memory SQLite) ──────────────────────────────────

TEST_DATABASE_URL = "sqlite:///file::memory:?cache=shared&uri=true"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)
TestSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=test_engine,
)


def override_get_db():
    """Yield a test database session."""
    db: Session = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Build the app with mocks in place ────────────────────────────────────────

# Patch AIService so it never touches the real Agent
with patch("app.services.ai_service.AIService") as MockAIServiceClass:
    mock_ai_service = MagicMock()

    # Make the async chat method return the mocked data
    async def _async_mock_chat(query: str):
        time.sleep(1)
        return "This is a simulated response from the AI"

    mock_ai_service.chat = _async_mock_chat
    MockAIServiceClass.get_instance.return_value = mock_ai_service
    MockAIServiceClass.initialize.return_value = mock_ai_service
    MockAIServiceClass._instance = mock_ai_service

    # Now import the app — the lifespan will use the mocked AIService
    from app.api.deps import get_ai_service, get_db
    from app.models.database import Base

    # Import the app but override lifespan to avoid real Agent loading
    from app.main import app

# ── Override dependencies ────────────────────────────────────────────────────

app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[get_ai_service] = lambda: mock_ai_service

# Create tables in the test database
Base.metadata.create_all(bind=test_engine)

# ── TestClient ───────────────────────────────────────────────────────────────

client = TestClient(app, raise_server_exceptions=False)


# ══════════════════════════════════════════════════════════════════════════════
# TEST CASES
# ══════════════════════════════════════════════════════════════════════════════


class TestChatEndpoint:
    """Tests for POST /api/chat."""

    def test_chat_returns_200_with_mocked_response(self):
        """
        TC-01: Send a standard text message to the chat endpoint.
        Assert status 200 and the response body contains the mocked answer.
        """
        payload = {"message": "a person walking on the street"}

        response = client.post("/api/chat", json=payload)

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )

        data = response.json()

        # Verify response structure
        assert "session_id" in data, "Response must contain session_id"
        assert "answer" in data, "Response must contain answer"
        assert "message_id" in data, "Response must contain message_id"

        # Verify mocked AI content is present
        assert data["answer"] == "This is a simulated response from the AI"

        print(f"\n✅ TC-01 PASSED — status=200, session_id={data['session_id']}")

    def test_chat_with_existing_session_id(self):
        """
        TC-02: Send two messages to the same session and verify
        the same session_id is returned.
        """
        # First message — creates a new conversation
        resp1 = client.post("/api/chat", json={"message": "first message"})
        assert resp1.status_code == 200
        session_id = resp1.json()["session_id"]

        # Second message — re-uses the same conversation
        resp2 = client.post(
            "/api/chat",
            json={"message": "second message", "session_id": session_id},
        )
        assert resp2.status_code == 200
        assert resp2.json()["session_id"] == session_id

        print(f"\n✅ TC-02 PASSED — session reuse verified for {session_id}")

    def test_chat_validates_empty_message(self):
        """
        TC-03: An empty message should be rejected with 422 (validation error).
        """
        response = client.post("/api/chat", json={"message": ""})

        assert response.status_code == 422, (
            f"Expected 422 for empty message, got {response.status_code}"
        )

        print("\n✅ TC-03 PASSED — empty message rejected with 422")


class TestDatabaseRecords:
    """Tests that verify DB state after a chat request."""

    def test_conversation_and_messages_inserted(self):
        """
        TC-04: After a chat request, verify that the corresponding
        Conversation and Message records were inserted into the test DB.
        """
        # Import ORM models for direct DB queries
        from app.models.database import Conversation, Message

        payload = {"message": "test database insertion"}
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 200

        data = response.json()
        session_id = data["session_id"]
        message_id = data["message_id"]

        # ── Query the test database directly ─────────────────────
        db: Session = TestSessionLocal()
        try:
            # Verify Conversation record exists
            conversation = (
                db.query(Conversation)
                .filter(Conversation.session_id == session_id)
                .first()
            )
            assert conversation is not None, (
                f"Conversation with session_id={session_id} not found in DB"
            )
            print(f"\n  📦 Conversation found: id={conversation.id}")

            # Verify User message exists
            user_msg = (
                db.query(Message)
                .filter(
                    Message.conversation_id == conversation.id,
                    Message.role == "user",
                )
                .first()
            )
            assert user_msg is not None, "User message not found in DB"
            assert user_msg.content == "test database insertion"
            print(f"  📩 User message: id={user_msg.id}, content={user_msg.content!r}")

            # Verify AI message exists
            ai_msg = (
                db.query(Message)
                .filter(
                    Message.conversation_id == conversation.id,
                    Message.role == "ai",
                )
                .first()
            )
            assert ai_msg is not None, "AI message not found in DB"
            assert ai_msg.id == message_id
            # The AI content is JSON-serialised results
            assert "simulated response" in ai_msg.content
            print(f"  🤖 AI message:   id={ai_msg.id}, content={ai_msg.content[:60]}…")

            print(f"\n✅ TC-04 PASSED — Conversation + 2 Messages verified in SQLite")
        finally:
            db.close()

    def test_multiple_messages_in_same_conversation(self):
        """
        TC-05: Send 3 messages to the same session, then verify
        that 6 messages total (3 user + 3 AI) exist for that conversation.
        """
        from app.models.database import Conversation, Message

        # Create conversation with first message
        resp = client.post("/api/chat", json={"message": "msg 1"})
        session_id = resp.json()["session_id"]

        # Send 2 more messages
        client.post("/api/chat", json={"message": "msg 2", "session_id": session_id})
        client.post("/api/chat", json={"message": "msg 3", "session_id": session_id})

        # Verify DB
        db: Session = TestSessionLocal()
        try:
            conversation = (
                db.query(Conversation)
                .filter(Conversation.session_id == session_id)
                .first()
            )
            assert conversation is not None

            messages = (
                db.query(Message)
                .filter(Message.conversation_id == conversation.id)
                .all()
            )
            assert len(messages) == 6, (
                f"Expected 6 messages (3 user + 3 AI), got {len(messages)}"
            )

            user_msgs = [m for m in messages if m.role == "user"]
            ai_msgs = [m for m in messages if m.role == "ai"]
            assert len(user_msgs) == 3
            assert len(ai_msgs) == 3

            print(f"\n✅ TC-05 PASSED — 6 messages (3u + 3ai) in session {session_id}")
        finally:
            db.close()


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_check(self):
        """TC-06: Health check should return 200 with status 'healthy'."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

        print(f"\n✅ TC-06 PASSED — health={data['status']}, version={data['version']}")
