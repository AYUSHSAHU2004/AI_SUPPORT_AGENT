"""
app/models.py — Pydantic schemas for API request and response models.
Strict type validation and documentation for all endpoints.
"""

from enum import Enum

from pydantic import BaseModel, Field


class IntentEnum(str, Enum):
    """Defined customer support intents derived from Amazon Twitter dataset."""

    ORDER_TRACKING = "order_tracking"
    REFUND_RETURN = "refund_return"
    ACCOUNT_ACCESS = "account_access"
    PRODUCT_ISSUE = "product_issue"
    DELIVERY_ISSUE = "delivery_issue"
    BILLING_PAYMENT = "billing_payment"
    GENERAL_INQUIRY = "general_inquiry"
    OTHER = "other"


class RetrievedExample(BaseModel):
    """Historical customer question and agent reply pair from ChromaDB."""

    customer_message: str
    agent_reply: str
    score: float | None = None


# ── Full Agent Pipeline Models ───────────────────────────────────────────────


class AgentRequest(BaseModel):
    """Input message for the full end-to-end support agent."""

    message: str = Field(
        ..., min_length=1, description="Incoming customer tweet or message"
    )
    conversation_id: str = Field(
        default="", description="Optional conversation/thread identifier"
    )


class AgentResponse(BaseModel):
    """Complete output produced by the agent pipeline."""

    intent: str = Field(..., description="Classified intent category")
    intent_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for intent (0-1)"
    )
    should_escalate: bool = Field(
        ..., description="Whether message needs human agent escalation"
    )
    escalation_reason: str | None = Field(
        None, description="Reason for escalation or auto-handling rationale"
    )
    reply_draft: str = Field(
        ..., description="Full drafted response grounded in historical resolutions"
    )
    reply_short: str = Field(
        ...,
        max_length=280,
        description="Twitter-compatible concise response (<= 280 chars)",
    )
    retrieved_examples: list[RetrievedExample] = Field(
        default_factory=list,
        description="Top historical similar resolutions retrieved from vector store",
    )


# ── Individual Endpoint Models ───────────────────────────────────────────────


class ClassifyRequest(BaseModel):
    """Request payload for intent classification only."""

    message: str = Field(..., min_length=1, description="Customer message to classify")


class ClassifyResponse(BaseModel):
    """Response payload for intent classification."""

    intent: str
    intent_confidence: float
    reasoning: str | None = None


class ReplyRequest(BaseModel):
    """Request payload to generate a grounded reply."""

    message: str = Field(..., min_length=1, description="Customer message")
    intent: str | None = Field(
        default=None, description="Optional pre-classified intent"
    )


class ReplyResponse(BaseModel):
    """Response payload containing generated replies."""

    reply_draft: str
    reply_short: str
    retrieved_examples: list[RetrievedExample]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model: str
    vector_store: str
