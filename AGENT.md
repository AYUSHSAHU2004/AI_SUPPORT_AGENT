# AGENT.md — Custom Instructions & Project Rules

> This file defines the coding conventions, architectural decisions, and agent behavior rules for the AI Support Agent project.
> All contributors (human or AI coding assistants) MUST read and follow these rules.
> Last updated: 2026-09-13

---

## 1. Project Philosophy

- **Simplicity over cleverness.** If two approaches work, pick the simpler one.
- **Prove it works.** The evaluation harness and golden set matter more than the agent itself.
- **No over-engineering.** No microservices, no complex dependency injection, no unnecessary abstractions.
- **4-5 hour build budget.** Every design decision must be justified against this constraint.

---

## 2. Tech Stack (Non-Negotiable)

| Layer | Choice | Reason |
|---|---|---|
| API framework | FastAPI (port 8088) | Simple, async, auto-docs |
| LLM orchestration | LangChain — used for ALL LLM/embedding calls | Consistency across the codebase |
| LLM backend | gemini-2.0-flash-lite (Google) | As specified by user |
| Embeddings | LangChain GoogleGenerativeAIEmbeddings | Same provider, consistent |
| Vector store | ChromaDB (persistent, via LangChain Chroma) | No separate infra needed |
| Relational DB | PostgreSQL | For conversation logs, audit trail, golden set storage |
| Data processing | Pandas + Python stdlib | Keep it simple |
| Containerisation | Docker + Docker Compose | Required by assignment |
| Evaluation | Python scripts + LLM-as-judge | Custom harness |
| Config | .env + pydantic-settings | All tweakable values via env |

### .env Tweakable Variables (always use .env, never hardcode)
```
GOOGLE_API_KEY=...
MODEL_NAME=gemini-2.0-flash-lite
EMBEDDING_MODEL=models/text-embedding-004
FASTAPI_PORT=8088
CHROMA_PERSIST_DIR=./data/chroma
POSTGRES_URL=postgresql://user:pass@db:5432/support_agent
MAX_RETRIEVED_EXAMPLES=3
GOLDEN_SET_SIZE=200
LOG_LEVEL=INFO
```

---

## 3. Coding Conventions

### General
- Use **Python 3.11+**.
- Use **type hints** everywhere.
- All functions must have **docstrings** (one-liner minimum).
- Max file length: **300 lines**. Split logically only if it makes sense.
- No dead code. No commented-out blocks.

### Naming
- Files: snake_case.py
- Classes: PascalCase
- Functions/variables: snake_case
- Constants: UPPER_SNAKE_CASE
- Env vars: UPPER_SNAKE_CASE (no prefix required, keep names obvious)

### Imports
- Standard library first, then third-party, then local.
- No wildcard imports.

### Error Handling
- Use specific exception types, not bare except.
- All API endpoints must return structured error responses.
- Log errors with context using Python's stdlib `logging`.

---

## 4. Project Structure

```
AI_support_agent/
├── app/                    # FastAPI application
│   ├── main.py             # App entry point, route registration, lifespan
│   ├── agent.py            # Core agent logic (classify + reply + escalate)
│   ├── retrieval.py        # ChromaDB vector store + retrieval via LangChain
│   ├── database.py         # PostgreSQL connection + models (SQLAlchemy)
│   ├── models.py           # Pydantic request/response models
│   └── config.py           # Settings (pydantic-settings, reads .env)
├── data/                   # Data directory (large files gitignored)
│   ├── raw/                # Raw CSV from Kaggle (gitignored)
│   ├── processed/          # Cleaned brand-specific JSONL (gitignored)
│   ├── chroma/             # ChromaDB persistent storage (gitignored)
│   └── golden_set/         # Hand-labelled evaluation examples
├── eval/                   # Evaluation harness
│   ├── evaluate.py         # Main evaluation script (runs all 3 systems)
│   ├── judge.py            # LLM-as-judge (LangChain, gemini-2.0-flash-lite)
│   ├── metrics.py          # Accuracy, F1, Kappa calculations
│   └── results/            # JSON output from evaluation runs
├── scripts/                # One-off data prep scripts (standalone)
│   ├── prepare_data.py     # Filter + clean brand data from raw CSV
│   └── build_golden_set.py # Stratified sample + LLM pre-label golden set
├── DEVLOG.md               # Development timeline with timestamps
├── AGENT.md                # This file
├── DECISION_LOG.md         # 10-15 non-obvious decisions
├── REPORT.md               # Assignment report (<=6 pages)
├── docker-compose.yml      # Orchestrates api + db services
├── Dockerfile              # API server image
├── requirements.txt        # Pinned dependencies
├── .env.example            # Template for .env
├── .gitignore
└── README.md               # Reproduction guide (<15 min)
```

**Rules:**
- `app/` is the only place that runs during the API server.
- `eval/` and `scripts/` are standalone — they do NOT import from `app/`.
  They share `config.py` via a direct path import or env vars only.
- Data files > 1MB are gitignored.
- No Jupyter notebooks in the final submission.

---

## 5. LangChain Usage Rules

**All LLM and embedding calls MUST go through LangChain. No direct SDK calls.**

### LLM
```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=...)
```

### Embeddings
```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
```

### Vector Store
```python
from langchain_chroma import Chroma

vectorstore = Chroma(persist_directory=..., embedding_function=embeddings)
```

### Chains
- Use `ChatPromptTemplate` + `|` pipe syntax (LCEL) for all chains.
- Use `JsonOutputParser` or `PydanticOutputParser` for structured outputs.
- No deprecated `LLMChain` class.

---

## 6. Database (PostgreSQL) Rules

PostgreSQL is used for:
1. **Conversation logs** — every agent request/response stored for audit.
2. **Golden set persistence** — golden set stored in DB, exportable to JSONL.
3. **Evaluation run results** — each eval run stored with timestamp and scores.

Use **SQLAlchemy 2.0** with async sessions.
Migrations via **Alembic** (one `alembic upgrade head` in Docker entrypoint).

Tables:
- `conversation_logs(id, message, intent, confidence, reply_draft, should_escalate, escalation_reason, created_at)`
- `golden_set(id, message, label_intent, label_escalate, label_reason, ideal_reply_summary, source)`
- `eval_runs(id, run_name, system, metrics_json, created_at)`

---

## 7. Agent Behaviour Rules

### Brand Selection
- **Brand: Amazon** (`@AmazonHelp`) — largest volume, diverse intents, well-known resolution patterns.

### Intent Categories
1. `order_tracking` — Where is my order? WISMO queries.
2. `refund_return` — Return, refund, or exchange requests.
3. `account_access` — Login issues, password, account locked.
4. `product_issue` — Damaged, wrong item, quality complaints.
5. `delivery_issue` — Late delivery, missed delivery, address problems.
6. `billing_payment` — Charges, payment failures, price disputes.
7. `general_inquiry` — Info requests, store hours, policy questions.
8. `other` — Anything that doesn't fit the above.

### Escalation Rules
Auto-handle if:
- Intent is `order_tracking`, `general_inquiry`, or `delivery_issue`.
- Sentiment is neutral or positive.

Escalate if:
- Intent is `refund_return`, `billing_payment`, or `account_access`.
- Message contains anger/frustration signals.
- Confidence in intent classification is < 0.6.
- Message contains legal threats or media mentions.

### Reply Constraints
- Max 280 characters for Twitter-style replies (plus a full draft).
- Always be empathetic, never defensive.
- Ground replies in retrieved historical examples.
- Never hallucinate order numbers, case IDs, or specific policies.

---

## 8. Docker Rules

- `Dockerfile` — builds the API server image (Python 3.11-slim).
- `docker-compose.yml` — two services: `api` (port 8088) + `db` (PostgreSQL).
- All secrets and tweakable values come from `.env` file.
- `docker-compose up --build` must start the full stack cleanly.
- Health check endpoint: `GET /health`.
- Data volumes:
  - `./data/chroma` mounted into container for ChromaDB persistence.
  - PostgreSQL data in named Docker volume `pgdata`.

---

## 9. Evaluation Rules

### Golden Set
- 200 examples, stratified by intent (~25 per intent).
- LLM pre-labelled, manually corrected by user.
- Stored in both `data/golden_set/golden_set.jsonl` AND the `golden_set` DB table.
- Fields: message, label_intent, label_escalate, label_reason, ideal_reply_summary.

### Metrics
- **Intent classification**: Accuracy, macro-F1.
- **Escalation**: Precision, Recall, F1.
- **Reply quality (LLM-as-judge)**: 1-5 on relevance, groundedness, tone, conciseness.
- **Human-judge agreement**: Cohen's Kappa on a 30-example subset.

### Baselines
1. **Trivial**: Majority class intent + static template reply.
2. **Simple**: TF-IDF + cosine similarity for intent + template lookup for reply.

---

## 10. What NOT to Build

- No frontend.
- No authentication/JWT.
- No streaming endpoints.
- No fine-tuning — prompt engineering only.
- No multi-brand support — Amazon only.
- No real-time Twitter API integration.

---

## 11. AI Coding Assistant Rules

- Cite any non-trivial logic borrowed.
- Understand every line. Live code review will happen.
- Do not add complexity not in the spec above.
- Prefer readable over clever.

---

## 12. Definition of Done

A feature is done when:
- [ ] Code is written with type hints and docstrings.
- [ ] It works end-to-end (tested via curl / FastAPI /docs).
- [ ] Edge cases handled (empty input, bad API key, DB down, etc.).
- [ ] It is reflected in DEVLOG.md with a timestamp.