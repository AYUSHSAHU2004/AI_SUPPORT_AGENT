"""
scripts/build_golden_set.py — Build stratified golden evaluation set with LLM pre-labelling.

Strategy:
1. Read data/processed/amazon_conversations.jsonl
2. Keyword-bucket messages into 8 intent categories (fast, no API cost)
3. Sample 25 per intent → 200 total, stratified
4. Use gemini-3.1-flash-lite (via LangChain) to pre-label each:
   intent, escalation, reasoning, ideal reply summary
5. Write to data/golden_set/golden_set.jsonl for human review & correction

After running this script, YOU must open golden_set.jsonl, review the labels, and
correct any obvious mistakes before running the evaluation harness. This is the
"hand-labelling" component required by the assignment.

Usage:
    .venv/bin/python scripts/build_golden_set.py
    .venv/bin/python scripts/build_golden_set.py --per-intent 25
"""

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from app.config import settings

IN_JSONL = ROOT / "data" / "processed" / "amazon_conversations.jsonl"
OUT_JSONL = ROOT / "data" / "golden_set" / "golden_set.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

INTENTS = [
    "order_tracking",
    "refund_return",
    "account_access",
    "product_issue",
    "delivery_issue",
    "billing_payment",
    "general_inquiry",
    "other",
]

# ── Keyword buckets for fast, free stratified sampling ────────────────────────
# Ordered from most-specific → least-specific. First match wins.

KEYWORD_BUCKETS: list[tuple[str, list[str]]] = [
    (
        "account_access",
        [
            "log in",
            "login",
            "sign in",
            "password",
            "account locked",
            "can't access",
            "cannot access",
            "account suspended",
            "verify account",
            "authentication",
            "two-factor",
            "2fa",
        ],
    ),
    (
        "billing_payment",
        [
            "charged",
            "charge",
            "payment",
            "invoice",
            "bill",
            "credit card",
            "refunded to card",
            "gift card",
            "price match",
            "overcharged",
            "unauthorized charge",
            "fee",
            "subscription fee",
        ],
    ),
    (
        "refund_return",
        [
            "refund",
            "return",
            "exchange",
            "send back",
            "replacement",
            "cancel order",
            "cancellation",
            "money back",
        ],
    ),
    (
        "order_tracking",
        [
            "where is my order",
            "track",
            "tracking",
            "order status",
            "hasn't arrived",
            "still waiting",
            "estimated delivery",
            "when will",
            "order number",
            "wismo",
        ],
    ),
    (
        "delivery_issue",
        [
            "delivered",
            "not delivered",
            "delivery failed",
            "couldn't deliver",
            "wrong address",
            "left at door",
            "stolen",
            "missing package",
            "courier",
            "driver",
            "ups",
            "usps",
            "fedex",
        ],
    ),
    (
        "product_issue",
        [
            "broken",
            "damaged",
            "defective",
            "wrong item",
            "not working",
            "quality",
            "missing part",
            "doesn't work",
            "stopped working",
            "counterfeit",
            "fake",
        ],
    ),
    (
        "general_inquiry",
        [
            "prime",
            "membership",
            "policy",
            "how do i",
            "how to",
            "question about",
            "information",
            "store hours",
            "availability",
            "does amazon",
            "can amazon",
        ],
    ),
]


def keyword_bucket(message: str) -> str:
    """Assign a message to an intent bucket using keyword heuristics. O(1) cost."""
    lower = message.lower()
    for intent, keywords in KEYWORD_BUCKETS:
        if any(kw in lower for kw in keywords):
            return intent
    return "other"


def load_all_pairs(path: Path) -> list[dict]:
    """Load all conversation pairs from processed JSONL."""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    log.info("Loaded %d total conversation pairs.", len(pairs))
    return pairs


def stratified_sample(pairs: list[dict], per_intent: int) -> list[dict]:
    """
    Bucket messages using keyword heuristics, then sample evenly per intent.
    No LLM calls needed — instant and free.
    """
    log.info("Bucketing %d pairs with keyword heuristics...", len(pairs))
    buckets: dict[str, list[dict]] = defaultdict(list)

    for pair in pairs:
        intent = keyword_bucket(pair["customer_message"])
        buckets[intent].append(pair)

    for intent in INTENTS:
        log.info("  Bucket '%s': %d pairs", intent, len(buckets.get(intent, [])))

    # Shuffle each bucket for randomness, then take per_intent
    samples = []
    for intent in INTENTS:
        pool = buckets.get(intent, [])
        random.shuffle(pool)
        chosen = pool[:per_intent]
        for item in chosen:
            item["_bucket_intent"] = intent  # Store heuristic bucket for reference
        samples.extend(chosen)
        log.info(
            "Sampled %d from intent '%s' (had %d available)",
            len(chosen),
            intent,
            len(pool),
        )

    random.shuffle(samples)
    log.info("Total samples after stratified selection: %d", len(samples))
    return samples


# ── Pre-labelling Prompt ──────────────────────────────────────────────────────

PRE_LABEL_PROMPT = """You are an expert annotator for customer support data from Amazon (@AmazonHelp) on Twitter.
Your job is to label the following customer message with precise, objective labels.

Customer Message:
"{message}"

Historical Amazon Reply (for context only — do NOT directly copy it):
"{agent_reply}"

Available intent categories:
- order_tracking: Where is my order? Tracking status queries.
- refund_return: Return, refund, exchange, or cancellation requests.
- account_access: Login issues, password reset, account locked/suspended.
- product_issue: Wrong item, damaged item, missing parts, quality complaints.
- delivery_issue: Late delivery, failed delivery attempt, wrong address, courier problems.
- billing_payment: Unrecognized charges, payment failures, price disputes, gift card issues.
- general_inquiry: Policy questions, product information, store hours, Prime membership questions.
- other: Anything that does not clearly fit the categories above.

Escalation rules:
- Escalate if: billing/payment involved, account security issue, explicit anger or frustration, legal threat, or any safety concern.
- Auto-handle if: simple status inquiry, general question, or product question with no strong negative emotion.

Return ONLY valid JSON, no markdown fences, no extra text:
{{
    "label_intent": "<one of the 8 category names>",
    "label_escalate": <true or false>,
    "label_reason": "<1-2 sentence rationale for the escalation decision>",
    "ideal_reply_summary": "<what an ideal agent response should accomplish in 1-2 sentences>"
}}"""


def extract_content(result: object) -> str:
    """Safely extract text content from a LangChain message object."""
    content = getattr(result, "content", result)
    if isinstance(content, list):
        # Some SDK versions return a list of content parts
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content)


def pre_label_sample(sample: dict, chain: object) -> dict | None:
    """Call the LLM to pre-label a single sample. Returns None on failure."""
    max_retries = 4
    for attempt in range(max_retries):
        try:
            result = chain.invoke(
                {
                    "message": sample["customer_message"],
                    "agent_reply": sample["agent_reply"],
                }
            )
            return {
                "id": sample.get("customer_tweet_id", ""),
                "message": sample["customer_message"],
                "original_agent_reply": sample["agent_reply"],
                "label_intent": result.get("label_intent", "other"),
                "label_escalate": bool(result.get("label_escalate", False)),
                "label_reason": result.get("label_reason", ""),
                "ideal_reply_summary": result.get("ideal_reply_summary", ""),
                "source": "twitter_amazon_sample",
                # Set this to true after you have personally reviewed/corrected this row
                "human_verified": False,
            }
        except Exception as e:
            err = str(e)
            if "RESOURCE_EXHAUSTED" in err or "429" in err:
                wait = 30 * (attempt + 1)
                log.warning(
                    "Rate limited. Waiting %ds (attempt %d/%d)...",
                    wait,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(wait)
            elif "503" in err or "UNAVAILABLE" in err:
                wait = 15 * (attempt + 1)
                log.warning(
                    "Service unavailable. Waiting %ds (attempt %d/%d)...",
                    wait,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(wait)
            else:
                log.warning(
                    "Pre-label failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    err[:150],
                )
                time.sleep(3.0)
    return None


def main(per_intent: int = 25) -> None:
    """Build, pre-label, and write the golden evaluation set."""
    if not IN_JSONL.exists():
        log.error(
            "Processed data not found at %s. Run scripts/prepare_data.py first.",
            IN_JSONL,
        )
        sys.exit(1)

    if (
        not settings.google_api_key
        or settings.google_api_key == "your_google_api_key_here"
    ):
        log.error("GOOGLE_API_KEY not set in .env.")
        sys.exit(1)

    llm = ChatGoogleGenerativeAI(
        model=settings.model_name,
        google_api_key=settings.google_api_key,
        temperature=0.0,  # Deterministic labels
    )

    label_chain = (
        ChatPromptTemplate.from_template(PRE_LABEL_PROMPT) | llm | JsonOutputParser()
    )

    log.info("Loading dataset from %s ...", IN_JSONL)
    all_pairs = load_all_pairs(IN_JSONL)

    samples = stratified_sample(all_pairs, per_intent=per_intent)

    log.info(
        "Pre-labelling %d samples with LLM (this will take ~10 minutes at free tier)...",
        len(samples),
    )
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    labelled: list[dict] = []
    failed = 0

    with open(OUT_JSONL, "w", encoding="utf-8") as out_f:
        for i, sample in enumerate(tqdm(samples, desc="Pre-labelling")):
            result = pre_label_sample(sample, label_chain)
            if result:
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()  # Write progressively — don't lose progress on crash
                labelled.append(result)
            else:
                failed += 1
                log.warning("Skipped sample %d (pre-label failed after retries).", i)
            time.sleep(1.0)  # ~60 RPM — safe for free tier

    log.info(
        "Done! Wrote %d labelled examples to %s (%d failed).",
        len(labelled),
        OUT_JSONL,
        failed,
    )

    # Distribution summary
    intent_counts: dict[str, int] = defaultdict(int)
    escalate_count = 0
    for item in labelled:
        intent_counts[item["label_intent"]] += 1
        if item["label_escalate"]:
            escalate_count += 1

    print("\n── Golden Set Summary ─────────────────────────────────────────────")
    for intent in INTENTS:
        print(f"  {intent:<20} : {intent_counts.get(intent, 0)}")
    print(f"  {'TOTAL':<20} : {len(labelled)}")
    print(
        f"  {'Escalate=True':<20} : {escalate_count} ({100 * escalate_count / max(len(labelled), 1):.1f}%)"
    )
    print("──────────────────────────────────────────────────────────────────")
    print(f"\n⚠️  HUMAN REVIEW REQUIRED: Open {OUT_JSONL} and review labels.")
    print("   Set 'human_verified': true for each row you have confirmed or corrected.")
    print("   This is the 'hand-labelled' component of the assignment.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build stratified golden evaluation set for Amazon support agent.",
    )
    parser.add_argument(
        "--per-intent",
        type=int,
        default=25,
        help="Examples per intent category (default: 25, total = 25 × 8 = 200)",
    )
    args = parser.parse_args()
    main(per_intent=args.per_intent)
