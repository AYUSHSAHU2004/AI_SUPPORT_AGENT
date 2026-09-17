"""
app/main.py — FastAPI Application Entry Point for Amazon Customer Support AI Agent.

Configured on port 8088 (or via FASTAPI_PORT environment variable).
Provides automatic Swagger documentation at /docs.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from app.agent import agent
from app.config import settings
from app.database import AsyncSessionLocal, ConversationLog, init_db, log_conversation
from app.models import (
    AgentRequest,
    AgentResponse,
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    ReplyRequest,
    ReplyResponse,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager: connects to DB and sets up services."""
    log.info(
        "Starting up Support Agent API server (port: %d)...", settings.fastapi_port
    )
    db_ok = await init_db()
    if db_ok:
        log.info("Database connection active.")
    else:
        log.info("Database running in offline/standalone mode.")
    yield
    log.info("Shutting down Support Agent API server.")


app = FastAPI(
    title="Amazon Customer Support AI Agent API",
    description=(
        "Production-grade AI Customer Support Agent for @AmazonHelp. "
        "Performs structured intent classification, grounded few-shot Twitter RAG reply generation, "
        "and safety/severity escalation management with audit logging."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Endpoint ──────────────────────────────────────────────────────────


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check & system status",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Returns server health, active model name, and vector store status."""
    return HealthResponse(
        status="healthy",
        model=settings.model_name,
        vector_store=settings.chroma_persist_dir,
    )


# ── Full Agent Pipeline ──────────────────────────────────────────────────────


@app.post(
    "/agent",
    response_model=AgentResponse,
    status_code=status.HTTP_200_OK,
    summary="Full Support Agent Pipeline",
    description="Classifies intent, drafts grounded response, and decides auto-handling vs human escalation.",
    tags=["Agent"],
)
async def process_message(request: AgentRequest) -> AgentResponse:
    """End-to-end support pipeline processing for customer tweet."""
    if not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message field cannot be empty.",
        )

    response = agent.process(
        message=request.message,
        conversation_id=request.conversation_id,
    )

    # Record turn to database for audit trail
    await log_conversation(
        message=request.message,
        intent=response.intent,
        confidence=response.intent_confidence,
        reply_draft=response.reply_draft,
        reply_short=response.reply_short,
        should_escalate=response.should_escalate,
        escalation_reason=response.escalation_reason,
        conversation_id=request.conversation_id,
    )

    return response


# ── Modular Sub-Endpoints ────────────────────────────────────────────────────


@app.post(
    "/classify",
    response_model=ClassifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify Intent Only",
    tags=["Sub-Tasks"],
)
async def classify_message(request: ClassifyRequest) -> ClassifyResponse:
    """Classifies a customer message into one of the 8 support intents."""
    if not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message field cannot be empty.",
        )
    return agent.classify_intent(request.message)


@app.post(
    "/reply",
    response_model=ReplyResponse,
    status_code=status.HTTP_200_OK,
    summary="Draft Grounded Reply Only",
    tags=["Sub-Tasks"],
)
async def generate_reply(request: ReplyRequest) -> ReplyResponse:
    """Generates full and short (<=280 chars) draft replies grounded in historical Twitter resolutions."""
    if not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message field cannot be empty.",
        )
    return agent.draft_reply(
        message=request.message,
        intent=request.intent or "general_inquiry",
    )


# ── Conversation Logs Audit Endpoint ─────────────────────────────────────────


@app.get(
    "/logs",
    summary="Get recent conversation logs",
    tags=["Audit"],
)
async def get_logs(limit: int = 20) -> list[dict]:
    """Retrieve recent conversation logs stored in SQLite."""
    try:
        async with AsyncSessionLocal() as session:
            stmt = (
                select(ConversationLog).order_by(ConversationLog.id.desc()).limit(limit)
            )
            result = await session.execute(stmt)
            logs = result.scalars().all()
            return [
                {
                    "id": l.id,
                    "conversation_id": l.conversation_id,
                    "message": l.message,
                    "intent": l.intent,
                    "confidence": l.confidence,
                    "should_escalate": l.should_escalate,
                    "escalation_reason": l.escalation_reason,
                    "reply_short": l.reply_short,
                    "created_at": l.created_at.isoformat() if l.created_at else None,
                }
                for l in logs
            ]
    except Exception as e:
        log.warning("Could not fetch DB logs: %s", e)
        return []


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.fastapi_port, reload=True)
