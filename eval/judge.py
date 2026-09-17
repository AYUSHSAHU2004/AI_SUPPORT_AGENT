"""
eval/judge.py — LLM-as-judge for reply quality evaluation.

Uses gemini-3.1-flash-lite via LangChain to score agent replies on 4 dimensions:
  - relevance:     Does the reply address the customer's actual problem? (1–5)
  - groundedness:  Is the reply grounded in real Amazon policies/tone, no hallucinations? (1–5)
  - tone:          Is the reply empathetic, professional, not defensive? (1–5)
  - conciseness:   Is the reply appropriately concise for Twitter support? (1–5)

Also provides a composite score (mean of 4 dimensions).

Standalone — does NOT import from app/. Reads config from environment variables.
"""

import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

log = logging.getLogger(__name__)

JUDGE_PROMPT = """You are an objective evaluator of customer support replies for Amazon (@AmazonHelp) on Twitter.
Score the following agent reply on 4 dimensions, each from 1 (poor) to 5 (excellent).

Customer Message:
"{message}"

Customer's Intent:
"{intent}"

Agent Reply Being Evaluated:
"{reply}"

Scoring Rubric:
- relevance (1–5): Does the reply directly address the customer's specific problem? 1=completely off-topic, 5=directly addresses all concerns.
- groundedness (1–5): Is the reply factually grounded? Does it avoid hallucinating order numbers, policies, or promises Amazon can't keep? 1=fabricates facts, 5=accurate and careful.
- tone (1–5): Is the reply empathetic, professional, and non-defensive? 1=cold/defensive/dismissive, 5=warm, understanding, and helpful.
- conciseness (1–5): Is the reply appropriately brief for Twitter support? 1=far too long or too short to be useful, 5=perfectly sized with no filler.

Return ONLY valid JSON, no markdown fences, no extra text:
{{
    "relevance": <integer 1–5>,
    "groundedness": <integer 1–5>,
    "tone": <integer 1–5>,
    "conciseness": <integer 1–5>,
    "rationale": "<1–2 sentences summarising your overall assessment>"
}}"""


class ReplyJudge:
    """LLM-as-judge for evaluating support reply quality."""

    def __init__(self) -> None:
        """Initialise LLM from environment variables."""
        api_key = os.getenv("GOOGLE_API_KEY", "")
        model = os.getenv("MODEL_NAME", "gemini-3.1-flash-lite")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment.")
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.0,
        )
        self.chain = (
            ChatPromptTemplate.from_template(JUDGE_PROMPT)
            | self.llm
            | JsonOutputParser()
        )

    def score(
        self,
        message: str,
        intent: str,
        reply: str,
        max_retries: int = 3,
    ) -> dict[str, float | str]:
        """
        Score a single agent reply.

        Returns dict with keys: relevance, groundedness, tone, conciseness,
        composite (mean of 4), and rationale.
        Returns all zeros + error string on failure.
        """
        for attempt in range(max_retries):
            try:
                result = self.chain.invoke(
                    {
                        "message": message,
                        "intent": intent,
                        "reply": reply,
                    }
                )
                scores = {
                    "relevance": float(result.get("relevance", 0)),
                    "groundedness": float(result.get("groundedness", 0)),
                    "tone": float(result.get("tone", 0)),
                    "conciseness": float(result.get("conciseness", 0)),
                    "rationale": result.get("rationale", ""),
                }
                scores["composite"] = round(
                    sum(
                        scores[k]
                        for k in ["relevance", "groundedness", "tone", "conciseness"]
                    )
                    / 4,
                    3,
                )
                return scores
            except Exception as e:
                err = str(e)
                if "RESOURCE_EXHAUSTED" in err or "429" in err:
                    wait = 30 * (attempt + 1)
                    log.warning(
                        "Judge rate limited. Waiting %ds (attempt %d/%d)...",
                        wait,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(wait)
                elif "503" in err or "UNAVAILABLE" in err:
                    wait = 15 * (attempt + 1)
                    log.warning("Judge service unavailable. Waiting %ds...", wait)
                    time.sleep(wait)
                else:
                    log.warning(
                        "Judge scoring failed (attempt %d/%d): %s",
                        attempt + 1,
                        max_retries,
                        err[:120],
                    )
                    time.sleep(3.0)

        log.error(
            "Judge gave up after %d retries for message: %s...",
            max_retries,
            message[:60],
        )
        return {
            "relevance": 0.0,
            "groundedness": 0.0,
            "tone": 0.0,
            "conciseness": 0.0,
            "composite": 0.0,
            "rationale": "FAILED: judge could not score this reply.",
        }

    def score_batch(
        self,
        examples: list[dict],
        delay: float = 1.0,
    ) -> list[dict[str, float | str]]:
        """
        Score a batch of examples.
        Each example dict must have: message, intent, reply.
        Returns list of score dicts in the same order.
        """
        results = []
        for i, ex in enumerate(examples):
            log.info("Judging example %d/%d...", i + 1, len(examples))
            score = self.score(
                message=ex["message"],
                intent=ex["intent"],
                reply=ex["reply"],
            )
            results.append(score)
            time.sleep(delay)
        return results
