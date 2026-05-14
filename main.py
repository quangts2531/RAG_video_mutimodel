"""
Main entry point for the FastAPI Backend.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers from the app.api.routes module (thư mục api nằm trong app)
from app.api.routes import chat as chat_routes
from app.api.routes import conversations as conversation_routes

# Import config, errors and lifespan from the core app
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.main import lifespan  # Giữ lại lifespan để load DB và AI Agent một lần duy nhất lúc startup

settings = get_settings()

app = FastAPI(
    title="Video RAG Chatbot API",
    description="Backend API for Video RAG with AI Agent",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Thiết lập middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register exception handlers
register_exception_handlers(app)

# Include routers trỏ đúng vào module api.routes
app.include_router(chat_routes.router, prefix="/api")
app.include_router(conversation_routes.router, prefix="/api")

@app.get("/health", tags=["System"])
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "version": settings.APP_VERSION}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
