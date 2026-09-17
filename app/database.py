"""
app/database.py — SQLite database connection, ORM models, and logging helpers.

Uses aiosqlite (async SQLite) — zero infrastructure needed.
The DB file lives at data/support_agent.db inside the container/project.

Stores:
1. conversation_logs: Full audit trail of every agent request and response.
2. golden_set:        Ground-truth evaluation dataset.
3. eval_runs:         Metrics from each evaluation harness run.
"""

import logging
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from app.config import settings

log = logging.getLogger(__name__)

Base = declarative_base()


# ── ORM Table Models ──────────────────────────────────────────────────────────


class ConversationLog(Base):
    """Stores every incoming user message and agent decision for audit."""

    __tablename__ = "conversation_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(128), default="")
    message = Column(Text, nullable=False)
    intent = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    reply_draft = Column(Text, nullable=False)
    reply_short = Column(String(300), nullable=False)
    should_escalate = Column(Boolean, nullable=False)
    escalation_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class GoldenSetItem(Base):
    """Stores persistent golden evaluation examples."""

    __tablename__ = "golden_set"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message = Column(Text, nullable=False)
    label_intent = Column(String(64), nullable=False)
    label_escalate = Column(Boolean, nullable=False)
    label_reason = Column(Text, nullable=True)
    ideal_reply_summary = Column(Text, nullable=True)
    source = Column(String(64), default="twitter_sample")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class EvalRun(Base):
    """Stores summary metrics from evaluation harness executions."""

    __tablename__ = "eval_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_name = Column(String(128), nullable=False)
    system = Column(String(64), nullable=False)
    metrics_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ── SQLite Engine & Session Factory ──────────────────────────────────────────

# Ensure the data/ directory exists before SQLite tries to create the file
Path(settings.sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)

engine = create_async_engine(
    f"sqlite+aiosqlite:///{settings.sqlite_db_path}",
    echo=False,
    future=True,
    connect_args={"check_same_thread": False},
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def init_db() -> bool:
    """Create all tables if they don't exist. Always succeeds with SQLite."""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        log.info("SQLite database initialised at %s.", settings.sqlite_db_path)
        return True
    except Exception as e:
        log.error("Failed to initialise SQLite database: %s", e)
        return False


async def log_conversation(
    message: str,
    intent: str,
    confidence: float,
    reply_draft: str,
    reply_short: str,
    should_escalate: bool,
    escalation_reason: str | None,
    conversation_id: str = "",
) -> None:
    """Record a conversation turn to SQLite. Silently skips on error."""
    try:
        async with AsyncSessionLocal() as session, session.begin():
            entry = ConversationLog(
                conversation_id=conversation_id,
                message=message,
                intent=intent,
                confidence=confidence,
                reply_draft=reply_draft,
                reply_short=reply_short,
                should_escalate=should_escalate,
                escalation_reason=escalation_reason,
            )
            session.add(entry)
    except Exception as e:
        log.debug("Skipping DB logging: %s", e)
