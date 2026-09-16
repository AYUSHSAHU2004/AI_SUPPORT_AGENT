"""
scripts/build_index.py — Build ChromaDB vector index from processed JSONL.

Embeds the *customer_message* so that at inference time, we can match
new customer queries against past similar customer queries. The corresponding
agent_reply is stored in metadata to be used as few-shot examples for the LLM.

Usage:
    .venv/bin/python scripts/build_index.py [--limit 2000] [--batch-size 50]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tqdm import tqdm

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


def build_index(limit: int = 2000, batch_size: int = 50) -> None:
    if not IN_JSONL.exists():
        log.error("Processed data not found at %s", IN_JSONL)
        sys.exit(1)

    log.info("Loading documents from %s (limit=%d)...", IN_JSONL, limit)
    docs: list[Document] = []
    with open(IN_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            record = json.loads(line)

            # The document text is the customer message (what we query against)
            doc = Document(
                page_content=record["customer_message"],
                metadata={
                    "customer_tweet_id": record.get("customer_tweet_id", ""),
                    "agent_tweet_id": record.get("agent_tweet_id", ""),
                    "agent_reply": record["agent_reply"],  # The historical resolution
                },
            )
            docs.append(doc)

    log.info(
        "Loaded %d documents. Initializing embeddings model: %s...",
        len(docs),
        settings.embedding_model,
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model, google_api_key=settings.google_api_key
    )

    log.info("Initializing ChromaDB at %s (collection='amazon_support')...", OUT_CHROMA)
    vectorstore = Chroma(
        collection_name="amazon_support",
        embedding_function=embeddings,
        persist_directory=str(OUT_CHROMA),
    )

    log.info("Embedding and indexing in batches of %d...", batch_size)
    for i in tqdm(range(0, len(docs), batch_size), desc="Indexing"):
        batch = docs[i : i + batch_size]
        max_retries = 5
        for attempt in range(max_retries):
            try:
                vectorstore.add_documents(batch)
                time.sleep(1.0)  # Gentle spacing to stay under free tier RPM
                break
            except Exception as e:
                err_str = str(e)
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    wait_time = 25.0
                    log.warning(
                        "Rate limit hit on batch %d. Waiting %.1fs before retrying (attempt %d/%d)...",
                        i // batch_size,
                        wait_time,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(wait_time)
                else:
                    log.error("Batch %d failed with error: %s", i // batch_size, e)
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(3.0)

    log.info(
        "Finished building index. Total collection size: %d",
        vectorstore._collection.count(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build ChromaDB vector index from Amazon dataset."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Number of conversation pairs to index (default: 2000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for embeddings (default: 50)",
    )
    args = parser.parse_args()

    if (
        not settings.google_api_key
        or settings.google_api_key == "your_google_api_key_here"
    ):
        log.error("Please set GOOGLE_API_KEY in .env first.")
        sys.exit(1)

    build_index(limit=args.limit, batch_size=args.batch_size)
