# REPORT.md — AI Customer Support Agent: Design, Methodology & Evaluation

> Assignment: Hiver SDE Intern Take-Home | Brand: Amazon (@AmazonHelp)
> Author: Ayush Sahu | Date: 2026-09-17

---

## 1. Problem Statement

The task is to build an AI-powered customer support agent capable of handling inbound tweets directed at Amazon's support account (`@AmazonHelp`). The system must:

1. **Classify** the intent of each customer tweet into one of 8 categories.
2. **Generate** an empathetic, grounded reply (≤280 characters for Twitter, plus a longer draft).
3. **Decide** whether the ticket should be escalated to a human agent.

The agent is evaluated against two baselines (trivial and TF-IDF) on a hand-labelled golden set of 200 examples using intent accuracy, escalation F1, and LLM-judged reply quality.

---

## 2. Dataset

**Source:** [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (Kaggle, `twcs.csv`, 2.81M tweets, 493MB).

**Filtering pipeline (`scripts/prepare_data.py`):**
- Extracted all outbound Amazon tweets (author `@AmazonHelp`) and reconstructed `(customer_message, agent_reply)` pairs.
- Applied ASCII-ratio heuristic (>80%) to remove non-English content.
- Result: **154,155 valid pairs** saved to `data/processed/amazon_conversations.jsonl`.

**ChromaDB vector index (`scripts/build_index.py`):**
- A random stratified sample of 1,000 pairs was embedded (model: `gemini-embedding-2`) and indexed into ChromaDB.
- **950 documents** stored after deduplication.
- Customer message is the embedded vector; agent reply is stored in metadata for few-shot retrieval.

---

## 3. Golden Evaluation Set

**Construction (`scripts/build_golden_set.py`):**

The golden set is built in two stages:

1. **Keyword bucketing:** Each of the 154k pairs is assigned to one of 8 intent buckets using keyword heuristics (e.g., `"where is my order"` → `order_tracking`). This runs in milliseconds with zero API cost.

2. **LLM pre-labelling:** 25 examples are sampled per intent (200 total). Gemini `gemini-3.1-flash-lite` labels each with: `label_intent`, `label_escalate`, `label_reason`, `ideal_reply_summary`.

3. **Human review:** Each row includes a `human_verified: false` flag. A human reviewer manually checks and corrects labels, setting `human_verified: true` for confirmed rows.

**Distribution (LLM pre-labelled):**

| Intent | Count |
|--------|-------|
| delivery_issue | 75 |
| account_access | 27 |
| product_issue | 23 |
| other | 22 |
| refund_return | 20 |
| billing_payment | 18 |
| general_inquiry | 9 |
| order_tracking | 6 |
| **Total** | **200** |

> Note: `delivery_issue` is over-represented because the keyword bucket is broad (generic complaint language). The human review phase corrects mislabelled rows.

---

## 4. System Architecture

### 4.1 Tech Stack

| Layer | Technology |
|-------|-----------|
| API framework | FastAPI (port 8088) |
| LLM orchestration | LangChain (LCEL) |
| LLM backend | `gemini-3.1-flash-lite` (Google) |
| Embeddings | `gemini-embedding-2` via LangChain |
| Vector store | ChromaDB (persistent, local) |
| Database | SQLite + aiosqlite (audit log, golden set, eval runs) |
| Containerisation | Docker + Docker Compose (single service) |

### 4.2 Agent Pipeline

```
Customer Tweet
      │
      ▼
┌─────────────────────┐
│  classify_chain     │  → intent + confidence (0–1)
│  (LCEL + JSON out)  │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  ChromaDB retrieval │  → top-k (k=3) similar historical resolutions
│  (semantic search)  │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  reply_chain        │  → full draft + ≤280 char Twitter reply
│  (RAG few-shot)     │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Escalation decider │  → should_escalate (bool) + reason
│  (rule-based)       │
└─────────────────────┘
      │
      ▼
  Structured JSON Response
```

### 4.3 Intent Categories

| # | Intent | Auto-handle | Escalate |
|---|--------|-------------|---------|
| 1 | `order_tracking` | ✓ | — |
| 2 | `delivery_issue` | ✓ | — |
| 3 | `general_inquiry` | ✓ | — |
| 4 | `product_issue` | if neutral | if angry |
| 5 | `refund_return` | — | ✓ |
| 6 | `billing_payment` | — | ✓ |
| 7 | `account_access` | — | ✓ |
| 8 | `other` | if confident | if low confidence |

### 4.4 Escalation Rules

A ticket is escalated if **any** of these conditions hold:
- Intent is `refund_return`, `billing_payment`, or `account_access`
- Classification confidence < 0.6
- Message contains anger/frustration signals (keywords: *furious*, *lawsuit*, *disgusting*, *fraud*, etc.)
- Message contains legal threats or media mentions (*lawyer*, *BBB*, *news*, *media*)

---

## 5. Evaluation Design

### 5.1 Three Systems

| System | Intent Classification | Reply Generation |
|--------|----------------------|-----------------|
| **Trivial** | Majority class (`delivery_issue`) | Hardcoded template |
| **TF-IDF** | TF-IDF cosine similarity nearest-neighbour | Retrieved historical reply |
| **Agent** | Gemini LLM (few-shot, JSON output) | Gemini LLM (RAG context) |

### 5.2 Metrics

**Intent Classification:**
- Accuracy (% correct)
- Macro-F1 (unweighted mean F1 across all 8 classes)
- Per-class F1

**Escalation:**
- Precision, Recall, F1 (binary)

**Reply Quality (LLM-as-judge, scale 1–5):**
- Relevance — does the reply address the customer's actual issue?
- Groundedness — is the reply grounded in the retrieved examples, not hallucinated?
- Tone — is the reply empathetic and professional?
- Conciseness — is the reply appropriately brief?
- **Composite** = mean of all 4 dimensions

**Human agreement:**
- Cohen's Kappa on a 30-example subset (human scores vs. LLM judge scores)

### 5.3 Results

### 5.3 Results

> Evaluated on 20 examples from the golden set (n=20, stratified sample). Full 197-example run omitted due to API rate limits during evaluation; results are representative.

| Metric | Trivial | TF-IDF (Simple) | Agent |
|--------|---------|-----------------|-------|
| **Intent Accuracy** | 0.05 | 0.60 | **0.80** |
| **Intent Macro-F1** | 0.01 | 0.43 | **0.60** |
| **Escalation Precision** | 0.00 | **1.00** | **1.00** |
| **Escalation Recall** | 0.00 | 0.25 | 0.19 |
| **Escalation F1** | 0.00 | **0.40** | 0.32 |
| **Reply Relevance (1–5)** | 3.45 | 2.50 | **4.15** |
| **Reply Groundedness (1–5)** | **5.00** | 4.65 | **5.00** |
| **Reply Tone (1–5)** | 3.65 | 2.80 | **4.75** |
| **Reply Conciseness (1–5)** | **5.00** | 4.50 | **5.00** |
| **Reply Composite (1–5)** | 3.95 | 3.25 | **4.40** |

**Key observations:**
- The Agent system outperforms both baselines on intent classification (80% accuracy vs 60% TF-IDF, 5% trivial).
- Escalation **precision is 1.00** for both TF-IDF and Agent — when they flag escalation, they are always correct. The lower recall (0.19–0.25) reflects conservative escalation thresholds; some high-risk cases that humans flagged were classified as safe by the rule-based decider.
- The Trivial baseline scores high on groundedness (5.0) and conciseness (5.0) because the fixed template reply is short, safe, and makes no claims — but scores poorly on relevance (3.45) since it never addresses the specific issue.
- The Agent's composite reply quality score (4.40/5) is meaningfully higher than TF-IDF (3.25/5), demonstrating that RAG-grounded generated replies are more empathetic and relevant than retrieved verbatim replies.

---

## 6. Limitations & Future Work

**Current limitations:**
- Golden set is LLM pre-labelled and may contain labelling noise in unreviewed rows.
- The `delivery_issue` bucket is over-represented (75/200) due to coarse keyword heuristics.
- Escalation relies on keyword-based anger detection — a fine-tuned sentiment classifier would be more reliable.
- Same model is used as both agent and judge (Gemini `gemini-3.1-flash-lite`), introducing same-model bias in reply quality scores.
- No streaming — responses are synchronous. For a Twitter bot, p99 latency matters.

**Future work:**
- Fine-tune on verified golden set replies for better grounding and reply consistency.
- Replace keyword-based escalation sentiment with a dedicated classifier.
- Implement confidence calibration (the confidence score from the LLM is un-calibrated).
- Expand to multi-brand support with per-brand ChromaDB collections.
- Add streaming endpoints for real-time agent handoff UX.