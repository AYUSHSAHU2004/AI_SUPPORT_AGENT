"""
eval/metrics.py — Metric calculations for the evaluation harness.

Computes:
- Intent classification: Accuracy, macro-F1, per-class F1
- Escalation: Precision, Recall, F1 (binary)
- Cohen's Kappa: For human-judge agreement on reply quality scores
"""

from collections import defaultdict


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Compute accuracy for classification labels."""
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def precision_recall_f1(y_true: list[bool], y_pred: list[bool]) -> dict[str, float]:
    """
    Compute binary precision, recall, and F1 for escalation.
    Positive class = True (should escalate).
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    """
    Compute macro-averaged F1 across all intent classes.
    Macro-F1 penalises poor performance on rare classes equally.
    """
    f1_scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)

    return round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0


def per_class_f1(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> dict[str, float]:
    """Return per-class F1 scores as a dict keyed by intent label."""
    results = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        results[label] = round(f1, 4)
    return results


def cohens_kappa(ratings_a: list[int], ratings_b: list[int]) -> float:
    """
    Compute Cohen's Kappa for inter-rater agreement.

    Used to measure agreement between human reply quality scores and LLM judge scores.
    Scale: <0=no agreement, 0–0.2=slight, 0.2–0.4=fair, 0.4–0.6=moderate,
           0.6–0.8=substantial, 0.8–1.0=almost perfect.

    Both rating lists must be the same length and use integer scores (e.g., 1–5).
    """
    if len(ratings_a) != len(ratings_b) or not ratings_a:
        raise ValueError("Both rating lists must be non-empty and equal length.")

    categories = sorted(set(ratings_a) | set(ratings_b))
    n = len(ratings_a)

    # Observed agreement
    observed = sum(1 for a, b in zip(ratings_a, ratings_b) if a == b) / n

    # Expected agreement under chance
    count_a: dict[int, int] = defaultdict(int)
    count_b: dict[int, int] = defaultdict(int)
    for a, b in zip(ratings_a, ratings_b):
        count_a[a] += 1
        count_b[b] += 1

    expected = sum((count_a[c] / n) * (count_b[c] / n) for c in categories)

    if expected == 1.0:
        return 1.0  # Perfect agreement trivially

    kappa = (observed - expected) / (1.0 - expected)
    return round(kappa, 4)


NUMERIC_JUDGE_KEYS = {"relevance", "groundedness", "tone", "conciseness", "composite"}


def mean_judge_scores(scores: list[dict[str, float]]) -> dict[str, float]:
    """
    Aggregate a list of per-example LLM judge score dicts into mean values.
    Only averages numeric keys (relevance, groundedness, tone, conciseness, composite).
    Skips string fields like 'rationale'.
    """
    if not scores:
        return {}
    return {
        k: round(sum(float(int(s.get(k, 0))) for s in scores) / len(scores), 3)
        for k in NUMERIC_JUDGE_KEYS
        if k in scores[0] and isinstance(scores[0][k], (int, float))
    }
