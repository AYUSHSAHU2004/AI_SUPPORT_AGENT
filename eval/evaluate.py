"""
eval/evaluate.py — Main evaluation harness. Runs 3 systems against the golden set.

Systems compared:
  1. Trivial baseline:  Majority class intent (order_tracking) + hardcoded template reply.
  2. Simple baseline:   TF-IDF + cosine similarity for intent matching + nearest reply lookup.
  3. Our agent:         Full LangChain classify → RAG retrieve → Gemini reply → escalation pipeline.

Outputs a JSON report to eval/results/<run_name>.json.

Standalone — does NOT import from app/. Reads config from environment variables directly.

Usage:
    .venv/bin/python eval/evaluate.py
    .venv/bin/python eval/evaluate.py --system agent   # run only one system
    .venv/bin/python eval/evaluate.py --limit 50       # smoke-test on first 50 examples
    .venv/bin/python eval/evaluate.py --skip-judge     # skip LLM judge (fast, metrics only)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.append(str(ROOT))

# Eval-internal imports (standalone — no app/ imports)
from eval.judge import ReplyJudge
from eval.metrics import (
    accuracy,
    macro_f1,
    mean_judge_scores,
    per_class_f1,
    precision_recall_f1,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

GOLDEN_JSONL = ROOT / "data" / "golden_set" / "golden_set.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"

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

# ── Trivial Baseline ──────────────────────────────────────────────────────────

MAJORITY_INTENT = "order_tracking"

TRIVIAL_REPLY = (
    "Hi! Thank you for reaching out to Amazon. "
    "Please DM us your order details and we'll look into it right away. ^AH"
)


def trivial_predict(message: str) -> dict:
    """Always predict majority class; always return the same template reply."""
    return {
        "intent": MAJORITY_INTENT,
        "escalate": False,
        "reply": TRIVIAL_REPLY,
    }


# ── Simple TF-IDF Baseline ────────────────────────────────────────────────────


class TfidfBaseline:
    """
    Simple intent classifier using TF-IDF + cosine similarity against training data.
    Nearest-neighbour lookup also provides the reply (no generation).
    """

    def __init__(self, training_jsonl: Path) -> None:
        """Load training data and build TF-IDF index."""
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        self._cosine = cosine_similarity
        self._np = np

        log.info(
            "Loading training data for TF-IDF baseline from %s ...", training_jsonl
        )
        self.training: list[dict] = []
        with open(training_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.training.append(json.loads(line))

        self.corpus = [ex["customer_message"] for ex in self.training]
        self.vectorizer = TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        log.info(
            "TF-IDF index built: %d documents, %d features.",
            len(self.corpus),
            self.tfidf_matrix.shape[1],
        )

    def predict(self, message: str) -> dict:
        """Find most similar training example; use its intent label and reply."""
        vec = self.vectorizer.transform([message])
        sims = self._cosine(vec, self.tfidf_matrix).flatten()
        idx = int(self._np.argmax(sims))
        best = self.training[idx]
        # The training data has no intent labels, so we apply the same keyword heuristic
        from scripts.build_golden_set import keyword_bucket

        intent = keyword_bucket(best["customer_message"])
        return {
            "intent": intent,
            "escalate": intent
            in {"refund_return", "billing_payment", "account_access"},
            "reply": best["agent_reply"],
        }


# ── Our Agent ─────────────────────────────────────────────────────────────────


class AgentSystem:
    """Wraps app.agent.SupportAgent for use in the eval harness."""

    def __init__(self) -> None:
        """Initialise the agent (imports app/ only here, not at module level)."""
        from app.agent import SupportAgent

        self._agent = SupportAgent()

    def predict(self, message: str) -> dict:
        """Run the full agent pipeline and return standardised prediction dict."""
        time.sleep(1.0)  # Pace API calls
        try:
            resp = self._agent.process(message)
            return {
                "intent": resp.intent,
                "escalate": resp.should_escalate,
                "reply": resp.reply_short,
            }
        except Exception as e:
            log.warning("Agent failed on message: %s — %s", message[:60], str(e)[:100])
            return {"intent": "other", "escalate": False, "reply": ""}


# ── Golden Set Loader ─────────────────────────────────────────────────────────


def load_golden_set(path: Path, limit: int | None = None) -> list[dict]:
    """Load the human-verified golden set from JSONL."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                examples.append(ex)
                if limit and len(examples) >= limit:
                    break
    log.info("Loaded %d golden set examples from %s.", len(examples), path)
    return examples


# ── Per-system Evaluation ─────────────────────────────────────────────────────


def evaluate_system(
    name: str,
    predictor,
    golden: list[dict],
    judge: ReplyJudge | None,
    skip_judge: bool = False,
) -> dict:
    """
    Run a single system (trivial / simple / agent) against the golden set.
    Returns a metrics dict for that system.
    """
    log.info("─── Evaluating system: %s (%d examples) ───", name, len(golden))
    y_true_intent = [ex["label_intent"] for ex in golden]
    y_true_escalate = [ex["label_escalate"] for ex in golden]

    y_pred_intent: list[str] = []
    y_pred_escalate: list[bool] = []
    judge_inputs: list[dict] = []

    for ex in tqdm(golden, desc=f"{name}"):
        pred = (
            predictor(ex["message"])
            if callable(predictor)
            else predictor.predict(ex["message"])
        )
        y_pred_intent.append(pred["intent"])
        y_pred_escalate.append(bool(pred["escalate"]))
        judge_inputs.append(
            {
                "message": ex["message"],
                "intent": pred["intent"],
                "reply": pred.get("reply", ""),
            }
        )

    intent_metrics = {
        "accuracy": accuracy(y_true_intent, y_pred_intent),
        "macro_f1": macro_f1(y_true_intent, y_pred_intent, INTENTS),
        "per_class_f1": per_class_f1(y_true_intent, y_pred_intent, INTENTS),
    }

    escalation_metrics = precision_recall_f1(y_true_escalate, y_pred_escalate)

    judge_metrics: dict = {}
    if not skip_judge and judge is not None:
        log.info("Running LLM judge for %s...", name)
        scores = judge.score_batch(judge_inputs, delay=1.0)
        judge_metrics = mean_judge_scores(scores)

    return {
        "system": name,
        "n_examples": len(golden),
        "intent": intent_metrics,
        "escalation": escalation_metrics,
        "reply_quality": judge_metrics,
        "predictions": [
            {
                "message": golden[i]["message"],
                "true_intent": y_true_intent[i],
                "pred_intent": y_pred_intent[i],
                "true_escalate": y_true_escalate[i],
                "pred_escalate": y_pred_escalate[i],
                "reply": judge_inputs[i]["reply"],
            }
            for i in range(len(golden))
        ],
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main(
    system: str = "all", limit: int | None = None, skip_judge: bool = False
) -> None:
    """Run evaluation harness and write results JSON."""
    if not GOLDEN_JSONL.exists():
        log.error(
            "Golden set not found at %s. Run scripts/build_golden_set.py first.",
            GOLDEN_JSONL,
        )
        sys.exit(1)

    golden = load_golden_set(GOLDEN_JSONL, limit=limit)

    if not golden:
        log.error("Golden set is empty. Did you forget to run the build script?")
        sys.exit(1)

    judge = None
    if not skip_judge:
        try:
            judge = ReplyJudge()
        except ValueError as e:
            log.warning("Judge disabled: %s", e)
            skip_judge = True

    # Initialise systems
    systems_to_run: dict[str, object] = {}

    training_jsonl = ROOT / "data" / "processed" / "amazon_conversations.jsonl"

    if system in ("all", "trivial"):
        systems_to_run["trivial"] = trivial_predict

    if system in ("all", "simple"):
        if training_jsonl.exists():
            systems_to_run["simple"] = TfidfBaseline(training_jsonl)
        else:
            log.warning("Training JSONL not found — skipping simple baseline.")

    if system in ("all", "agent"):
        log.info("Initialising full agent (this may take a moment)...")
        systems_to_run["agent"] = AgentSystem()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    all_results: list[dict] = []

    for sys_name, predictor in systems_to_run.items():
        result = evaluate_system(
            name=sys_name,
            predictor=predictor,
            golden=golden,
            judge=judge,
            skip_judge=skip_judge,
        )
        all_results.append(result)

        # Print summary
        print(f"\n{'═' * 60}")
        print(f"  System: {sys_name.upper()}")
        print(f"  Intent Accuracy : {result['intent']['accuracy']:.4f}")
        print(f"  Intent Macro-F1 : {result['intent']['macro_f1']:.4f}")
        print(f"  Escalation F1   : {result['escalation']['f1']:.4f}")
        if result["reply_quality"]:
            print(
                f"  Reply Composite : {result['reply_quality'].get('composite', 'N/A'):.3f}/5"
            )
        print(f"{'═' * 60}")

    # Save full results
    out_path = RESULTS_DIR / f"{run_name}_{system}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_name,
                "system_filter": system,
                "n_examples": len(golden),
                "skip_judge": skip_judge,
                "results": all_results,
            },
            f,
            indent=2,
        )
    log.info("Results saved to %s", out_path)
    print(f"\n✅ Evaluation complete. Results saved to: {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run evaluation harness for the Amazon support agent."
    )
    parser.add_argument(
        "--system",
        choices=["all", "trivial", "simple", "agent"],
        default="all",
        help="Which system(s) to evaluate (default: all).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to first N golden set examples (for smoke testing).",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM-as-judge reply quality scoring (faster, metrics only).",
    )
    args = parser.parse_args()
    main(system=args.system, limit=args.limit, skip_judge=args.skip_judge)
