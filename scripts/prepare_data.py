"""
scripts/prepare_data.py — Filter and clean the raw Kaggle Twitter dataset.

Reads data/raw/twcs/twcs.csv, extracts all @AmazonHelp conversations,
reconstructs (customer_message, agent_reply) pairs, cleans tweet text,
and writes data/processed/amazon_conversations.jsonl.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --limit 5000   # cap output rows
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "twcs" / "twcs.csv"
OUT_JSONL = ROOT / "data" / "processed" / "amazon_conversations.jsonl"

# ── Brand filter ─────────────────────────────────────────────────────────────
BRAND_AUTHOR_ID = "AmazonHelp"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── Text cleaning ─────────────────────────────────────────────────────────────


def clean_tweet(text: str) -> str:
    """Remove @mentions, URLs, extra whitespace from a tweet."""
    text = re.sub(r"@\w+", "", text)  # strip @handles
    text = re.sub(r"http\S+|www\.\S+", "", text)  # strip URLs
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return text.strip()


def is_english(text: str, threshold: float = 0.80) -> bool:
    """Return True if >threshold fraction of chars are printable ASCII.

    Fast heuristic — avoids langdetect dependency. Works well enough
    to filter Japanese, Arabic, Korean, Chinese etc.
    """
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / len(text)) >= threshold


# ── Core logic ────────────────────────────────────────────────────────────────


def load_raw(path: Path) -> pd.DataFrame:
    """Load the raw CSV, keeping only the columns we need."""
    log.info("Loading raw CSV from %s …", path)
    df = pd.read_csv(
        path,
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
            "text",
            "response_tweet_id",
            "in_response_to_tweet_id",
        ],
        dtype=str,
    )
    df.fillna("", inplace=True)
    log.info("Loaded %d rows total.", len(df))
    return df


def build_id_index(df: pd.DataFrame) -> dict[str, dict]:
    """Build a dict of tweet_id → row for fast parent lookup."""
    return {row["tweet_id"]: row for _, row in df.iterrows()}


def extract_amazon_pairs(df: pd.DataFrame, index: dict[str, dict]) -> list[dict]:
    """
    Find every AmazonHelp reply and pair it with the customer message it
    responds to. Returns a list of conversation dicts.
    """
    # All outbound tweets authored by AmazonHelp
    agent_tweets = df[
        (df["author_id"].str.strip() == BRAND_AUTHOR_ID)
        & (df["inbound"].str.strip().str.lower() == "false")
    ]
    log.info("Found %d AmazonHelp outbound tweets.", len(agent_tweets))

    pairs: list[dict] = []
    skipped = 0

    for _, agent_row in agent_tweets.iterrows():
        parent_id = agent_row["in_response_to_tweet_id"].strip()
        if not parent_id or parent_id not in index:
            skipped += 1
            continue

        customer_row = index[parent_id]

        # Only keep actual inbound (customer) messages
        if customer_row["inbound"].strip().lower() != "true":
            skipped += 1
            continue

        customer_text = clean_tweet(customer_row["text"])
        agent_text = clean_tweet(agent_row["text"])

        # Skip empty or very short messages
        if len(customer_text) < 10 or len(agent_text) < 10:
            skipped += 1
            continue

        # Skip non-English tweets
        if not is_english(customer_text) or not is_english(agent_text):
            skipped += 1
            continue

        pairs.append(
            {
                "customer_tweet_id": customer_row["tweet_id"],
                "agent_tweet_id": agent_row["tweet_id"],
                "customer_message": customer_text,
                "agent_reply": agent_text,
            }
        )

    log.info("Extracted %d valid pairs (%d skipped).", len(pairs), skipped)
    return pairs


def write_jsonl(pairs: list[dict], path: Path, limit: int | None = None) -> None:
    """Write conversation pairs to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if limit:
        pairs = pairs[:limit]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(record) + "\n" for record in pairs)
    log.info("Wrote %d records to %s", len(pairs), path)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Run the data preparation pipeline."""
    parser = argparse.ArgumentParser(description="Prepare Amazon Twitter support data.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of output pairs (default: all).",
    )
    args = parser.parse_args()

    if not RAW_CSV.exists():
        log.error("Raw CSV not found at %s", RAW_CSV)
        log.error(
            "Run: kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw --unzip"
        )
        sys.exit(1)

    df = load_raw(RAW_CSV)
    index = build_id_index(df)
    pairs = extract_amazon_pairs(df, index)

    if not pairs:
        log.error("No pairs extracted. Check BRAND_AUTHOR_ID filter.")
        sys.exit(1)

    write_jsonl(pairs, OUT_JSONL, limit=args.limit)
    log.info("Done. Output: %s", OUT_JSONL)

    # Quick sanity print
    sample = pairs[0]
    print("\n── Sample pair ─────────────────────────────────────────")
    print(f"Customer : {sample['customer_message']}")
    print(f"AmazonHelp: {sample['agent_reply']}")
    print("────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
