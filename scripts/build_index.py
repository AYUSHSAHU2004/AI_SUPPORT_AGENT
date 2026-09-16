"""
scripts/build_index.py — Build ChromaDB vector index from processed JSONL.

Embeds the *customer_message* so that at inference time, we can match
new customer queries against past similar customer queries. The corresponding
agent_reply is stored in metadata to be used as few-shot examples for the LLM.

Usage:
    python scripts/build_index.py
"""

import json
import logging
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))  # allow importing app.config

from app.config import settings

IN_JSONL = ROOT / "data" / "processed" / "amazon_conversations.jsonl"
OUT_CHROMA = ROOT / settings.chroma_persist_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def build_index() -> None:
    if not IN_JSONL.exists():
        log.error("Processed data not found at %s", IN_JSONL)
        sys.exit(1)

    log.info("Loading documents from %s", IN_JSONL)
    docs = []
    with open(IN_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if (
                i >= 5000
            ):  # Limit to 5k for speed/cost if desired, though our limit was 6000
                break
            record = json.loads(line)

            # The document is the customer message (what we will search against)
            doc = Document(
                page_content=record["customer_message"],
                metadata={
                    "customer_tweet_id": record.get("customer_tweet_id", ""),
                    "agent_tweet_id": record.get("agent_tweet_id", ""),
                    "agent_reply": record[
                        "agent_reply"
                    ],  # The answer we want to retrieve
                },
            )
            docs.append(doc)

    log.info("Loaded %d documents. Initializing embeddings...", len(docs))

    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model, google_api_key=settings.google_api_key
    )

    log.info("Building ChromaDB at %s...", OUT_CHROMA)
    # Chroma.from_documents handles batching automatically
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(OUT_CHROMA),
        collection_name="amazon_support",
    )

    log.info(
        "Finished building index. Collection size: %d", vectorstore._collection.count()
    )


if __name__ == "__main__":
    if (
        not settings.google_api_key
        or settings.google_api_key == "your_google_api_key_here"
    ):
        log.error("Please set GOOGLE_API_KEY in .env first.")
        sys.exit(1)

    build_index()
