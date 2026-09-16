"""
app/agent.py — Core AI Customer Support Agent for Amazon (@AmazonHelp).

Implements three LangChain LCEL chains:
1. Intent Classification Chain (structured JSON output with confidence)
2. RAG Reply Drafting Chain (grounded in historical Twitter resolutions)
3. Escalation Decider (business rules + sentiment & urgency detection)
"""

import logging
import re

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.models import (
    AgentResponse,
    ClassifyResponse,
    IntentEnum,
    ReplyResponse,
    RetrievedExample,
)
from app.retrieval import SupportRetriever, retriever

log = logging.getLogger(__name__)

# ── Intent System Prompts ────────────────────────────────────────────────────

INTENT_DESCRIPTIONS = """
- order_tracking: Questions about order status, tracking links, where is my package (WISMO).
- refund_return: Requests for returns, refunds, item replacements, or return labels.
- account_access: Password reset, OTP issues, login difficulties, account suspension or lock.
- product_issue: Damaged, defective, wrong item received, missing parts, or quality problems.
- delivery_issue: Late delivery, missed delivery window, delivered to wrong address, courier problems.
- billing_payment: Unrecognized charges, double charges, payment declined, gift card or discount issues.
- general_inquiry: Questions about store policies, Prime benefits, product availability, hours.
- other: Anything else that does not clearly fit into the categories above.
"""

CLASSIFY_PROMPT_TEMPLATE = """You are an expert intent classification system for Amazon Customer Support (@AmazonHelp).
Analyze the incoming customer tweet and classify it into exactly one of the following 8 categories:

{intent_descriptions}

Customer Tweet:
"{message}"

Return your classification as valid JSON matching this exact structure:
{{
    "intent": "<one of the 8 category names in lowercase>",
    "confidence": <float between 0.0 and 1.0 representing your certainty>,
    "reasoning": "<concise 1-sentence justification>"
}}
"""

REPLY_PROMPT_TEMPLATE = """You are a helpful, professional, and empathetic customer support agent for Amazon (@AmazonHelp) on Twitter.
Your task is to draft a reply to the customer's message based on how Amazon historically resolved similar issues.

Customer Message:
"{message}"

Identified Intent:
"{intent}"

Historical Amazon Resolution Examples:
{retrieved_context}

Guidelines:
1. Be polite, empathetic, and professional.
2. If asking for details, remind the user NOT to share sensitive/personal info publicly (e.g., "Without sharing personal details...").
3. Do NOT hallucinate order numbers, tracking numbers, or personal accounts.
4. Ground your response in the tone and resolution steps from the historical examples.
5. Provide a full reply, and a short version that strictly fits within 280 characters for Twitter.

Return your answer as valid JSON matching this exact structure:
{{
    "reply_draft": "<full detailed drafted response>",
    "reply_short": "<concise Twitter reply strictly under 280 characters>"
}}
"""


class SupportAgent:
    """End-to-end Support Agent managing classification, retrieval, and response generation."""

    def __init__(self, vector_retriever: SupportRetriever | None = None) -> None:
        self.retriever = vector_retriever or retriever
        self.llm = ChatGoogleGenerativeAI(
            model=settings.model_name,
            google_api_key=settings.google_api_key,
            temperature=0.1,
        )
        self._build_chains()

    def _build_chains(self) -> None:
        """Construct LangChain LCEL runnable chains."""
        # 1. Classification Chain
        classify_prompt = ChatPromptTemplate.from_template(CLASSIFY_PROMPT_TEMPLATE)
        self.classify_chain = classify_prompt | self.llm | JsonOutputParser()

        # 2. Reply Generation Chain
        reply_prompt = ChatPromptTemplate.from_template(REPLY_PROMPT_TEMPLATE)
        self.reply_chain = reply_prompt | self.llm | JsonOutputParser()

    def classify_intent(self, message: str) -> ClassifyResponse:
        """Classify customer message intent and confidence score."""
        try:
            result = self.classify_chain.invoke(
                {
                    "intent_descriptions": INTENT_DESCRIPTIONS.strip(),
                    "message": message,
                }
            )
            intent = result.get("intent", "other").strip().lower()
            confidence = float(result.get("confidence", 0.7))
            reasoning = result.get("reasoning", "")

            # Fallback if unknown intent generated
            valid_intents = [i.value for i in IntentEnum]
            if intent not in valid_intents:
                intent = "other"

            return ClassifyResponse(
                intent=intent,
                intent_confidence=min(max(confidence, 0.0), 1.0),
                reasoning=reasoning,
            )
        except Exception as e:
            log.error("Intent classification chain failed: %s", e)
            return ClassifyResponse(
                intent="other",
                intent_confidence=0.5,
                reasoning=f"Fallback due to classification error: {e!s}",
            )

    def draft_reply(
        self,
        message: str,
        intent: str = "general_inquiry",
        retrieved_examples: list[RetrievedExample] | None = None,
    ) -> ReplyResponse:
        """Draft a grounded reply using retrieved historical Amazon resolutions."""
        examples = retrieved_examples or self.retriever.retrieve_similar(
            message, k=settings.max_retrieved_examples
        )

        # Format context from examples
        if examples:
            context_blocks = []
            for i, ex in enumerate(examples, 1):
                context_blocks.append(
                    f"Example {i}:\nCustomer: {ex.customer_message}\nAmazonHelp: {ex.agent_reply}"
                )
            context_str = "\n\n".join(context_blocks)
        else:
            context_str = "No direct historical matches found. Use standard Amazon customer service tone."

        try:
            res = self.reply_chain.invoke(
                {
                    "message": message,
                    "intent": intent,
                    "retrieved_context": context_str,
                }
            )
            reply_draft = res.get("reply_draft", "").strip()
            reply_short = res.get("reply_short", "").strip()

            # Ensure short reply is strictly <= 280 chars
            if len(reply_short) > 280:
                reply_short = reply_short[:277] + "..."

            return ReplyResponse(
                reply_draft=reply_draft,
                reply_short=reply_short,
                retrieved_examples=examples,
            )
        except Exception as e:
            log.error("Reply generation chain failed: %s", e)
            fallback = "I'm sorry to hear about the issue! Please send us a direct message so we can look into this for you."
            return ReplyResponse(
                reply_draft=fallback,
                reply_short=fallback,
                retrieved_examples=examples,
            )

    def decide_escalation(
        self, message: str, intent: str, confidence: float
    ) -> tuple[bool, str]:
        """
        Determine if the issue requires escalation to a human specialist.

        Rules:
        - Low confidence (< 0.60) -> Escalate
        - High-risk intents (refund_return, billing_payment, account_access) -> Escalate
        - Frustration / Anger / Legal / Media keywords -> Escalate
        - Simple queries (order_tracking, general_inquiry, delivery_issue) with normal sentiment -> Auto-handle
        """
        msg_lower = message.lower()

        # 1. High urgency / legal / escalatory keywords
        escalate_keywords = [
            "lawyer",
            "attorney",
            "legal",
            "sue",
            "court",
            "fraud",
            "police",
            "scam",
            "chargeback",
            "dispute",
            "better business bureau",
            "bbb",
            "ridiculous",
            "unacceptable",
            "furious",
            "stolen",
            "unauthorized",
            "hacked",
            "locked out",
        ]
        for kw in escalate_keywords:
            if re.search(rf"\b{re.escape(kw)}\b", msg_lower):
                return True, f"High severity or legal/fraud keyword detected ('{kw}')"

        # 2. Confidence threshold
        if confidence < 0.60:
            return True, f"Low classification confidence ({confidence:.2f} < 0.60)"

        # 3. Intent-based escalation
        escalate_intents = {
            "refund_return": "Refund/return requests involve financial transactions and policy authorizations",
            "billing_payment": "Billing disputes and payment investigations require human agent tools",
            "account_access": "Account security and credential recovery must be handled by human agents",
        }
        if intent in escalate_intents:
            return True, escalate_intents[intent]

        # 4. Standard auto-handled intents
        auto_intents = {
            "order_tracking",
            "general_inquiry",
            "delivery_issue",
            "product_issue",
            "other",
        }
        if intent in auto_intents:
            return (
                False,
                f"Standard informational issue for intent '{intent}' with high confidence",
            )

        return False, "Default auto-handled with standard support response"

    def process(self, message: str, conversation_id: str = "") -> AgentResponse:
        """Run the full end-to-end agent pipeline."""
        # 1. Intent classification
        classify_res = self.classify_intent(message)
        intent = classify_res.intent
        confidence = classify_res.intent_confidence

        # 2. Retrieve relevant historical examples
        retrieved = self.retriever.retrieve_similar(
            message, k=settings.max_retrieved_examples
        )

        # 3. Draft grounded response
        reply_res = self.draft_reply(
            message=message, intent=intent, retrieved_examples=retrieved
        )

        # 4. Decide escalation
        should_escalate, reason = self.decide_escalation(
            message=message,
            intent=intent,
            confidence=confidence,
        )

        return AgentResponse(
            intent=intent,
            intent_confidence=confidence,
            should_escalate=should_escalate,
            escalation_reason=reason,
            reply_draft=reply_res.reply_draft,
            reply_short=reply_res.reply_short,
            retrieved_examples=retrieved,
        )


# Global singleton agent
agent = SupportAgent()
