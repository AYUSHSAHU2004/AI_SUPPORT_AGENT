# DECISION_LOG.md — Non-Obvious Architectural Decisions


---

## 1. SQLite over PostgreSQL

**Decision:** Use SQLite + aiosqlite instead of PostgreSQL.

**Why:** The assignment requires storing conversation logs, golden set rows, and eval run results. PostgreSQL would require a second Docker service, a network between containers, health check waiting logic in the entrypoint, and Alembic migrations. For this scale (hundreds of rows), SQLite is completely sufficient. Removing Postgres eliminated ~100 lines of infrastructure glue and made `docker compose up` a single-service operation with no startup sequencing.

**Trade-off:** Cannot horizontally scale (only one writer at a time). Acceptable for this scope.

---

## 2. Keyword Bucketing for Golden Set Stratification

**Decision:** Use keyword heuristics to bucket 154k conversations into 8 intent classes before sampling, rather than LLM classification per row.

**Why:** Calling the Gemini API 154k times would cost significant money and take 4+ hours. Instead, fast keyword matching (e.g., `"track"`, `"where is"` → `order_tracking`) runs in milliseconds for the entire dataset. We then sample 25 per bucket and call the LLM only for those 200 rows to generate labels — reducing API calls by ~750x.

**Trade-off:** Keyword buckets are imprecise. Some buckets (e.g., `delivery_issue`) over-capture because generic complaint language triggers multiple keywords. The golden set distribution is uneven as a result: 75 delivery_issue vs 6 order_tracking. Acceptable since human review corrects the labels.

---

## 3. No Fine-Tuning — Prompt Engineering Only

**Decision:** Do not fine-tune the Gemini model. Use few-shot prompting with RAG context.

**Why:** Fine-tuning requires labelled training data, a training pipeline, hosting a custom model endpoint, and significant compute budget — none of which are feasible in a 4-5 hour build. Prompt engineering with retrieved historical examples achieves good grounding and produces empathetic replies without any training infrastructure.

**Trade-off:** In production, fine-tuning on verified Amazon reply data would likely improve consistency and reduce hallucination risk.

---

## 4. LangChain as the Sole LLM Abstraction Layer

**Decision:** All LLM and embedding calls go through LangChain. No direct Gemini SDK (`google-genai`) usage.

**Why:** Mandated by project spec (AGENTS.md). LangChain's LCEL pipe syntax makes chains readable and composable. `JsonOutputParser` eliminates boilerplate for structured outputs. Using one abstraction throughout the codebase ensures every LLM call is consistently configured (model, temperature, API key injection).

**Trade-off:** LangChain adds a dependency layer. API shape changes in upstream Gemini SDK can be masked until they break in unexpected ways (we hit this: `result.content` became a list in newer SDK versions).

---

## 5. ChromaDB over Pinecone / Weaviate

**Decision:** Use ChromaDB (local, persistent) as the vector store.

**Why:** ChromaDB requires zero external infrastructure. It runs embedded in the same Python process, persists to disk, and supports LangChain's `Chroma` wrapper natively. Pinecone and Weaviate require API keys, external accounts, and network round-trips — all unnecessary overhead for a local assignment.

**Trade-off:** Cannot scale beyond a single machine. Fine for this use case.

---

## 6. Customer Message as the Embedded Document (Not Agent Reply)

**Decision:** Index the **customer's message** as the vector, store the **agent's reply** in metadata.

**Why:** At query time, we have a new customer message. Semantic similarity should be measured between incoming messages (what the customer said) to retrieve the most relevant historical agent reply. Embedding replies instead would match on reply language, which is less meaningful for retrieval.

**Trade-off:** Long, compound customer messages with mixed intents may retrieve less precise neighbours.

---

## 7. Three-Tier Evaluation Design (Trivial → TF-IDF → Agent)

**Decision:** Evaluate three systems: a majority-class trivial baseline, a TF-IDF nearest-neighbour system, and the full LangChain agent.

**Why:** A single evaluation of the main agent tells you nothing without context. The trivial baseline sets the floor — if the agent doesn't beat majority-class, the pipeline is broken. The TF-IDF system tests whether semantic retrieval (ChromaDB) adds value over simple keyword similarity. Only the agent evaluation uses the Gemini LLM.

---

## 8. LLM-as-Judge on 4 Dimensions

**Decision:** Score reply quality on four dimensions — relevance, groundedness, tone, conciseness — scored 1–5, then averaged.

**Why:** A single composite score conflates very different failure modes (a reply can be perfectly toned but completely unhallucinated nonsense). Four orthogonal dimensions allow targeted diagnosis. Gemini `gemini-3.1-flash-lite` is used as judge since it's the same model as the agent — this introduces same-model bias, but it's the most available option.

---

## 9. Escalation via Rule-Based Heuristic (not LLM)

**Decision:** Escalation is decided by a deterministic rule: intent type + confidence threshold + keyword scan, not an additional LLM call.

**Why:** Adding an LLM call purely for escalation doubles latency and cost with marginal gain. The rules cover the assignment's requirements exactly:
- High-risk intents (`refund_return`, `billing_payment`, `account_access`) → escalate
- Low confidence (< 0.6) → escalate
- Legal/anger keywords → escalate

This is transparent, auditable, and fast.

---

## 10. `--skip-judge` Flag on Evaluation Harness

**Decision:** Add an optional `--skip-judge` flag to `eval/evaluate.py` that bypasses LLM-as-judge scoring.

**Why:** Running the full judge on 200 examples takes significant time and API quota. During development, fast sanity checks (intent accuracy, escalation F1) can be done in seconds without waiting for the judge. The flag makes the harness useful at all stages of development, not just final evaluation.

---

## 11. Multi-Stage Docker Build

**Decision:** Use a two-stage `Dockerfile` (builder + runtime) instead of a single-stage image.

**Why:** ChromaDB and other packages have compiled C extensions requiring `build-essential` and `gcc`. Installing these in the final image bloats it unnecessarily. The builder stage installs all dependencies into `/install`, then the slim runtime stage copies only the installed packages — keeping the final image small without `build-essential`.

---

## 12. Amazon (`@AmazonHelp`) as the Single Brand

**Decision:** Filter and index only Amazon conversations, not multiple brands.

**Why:** The assignment explicitly calls for a single brand. Amazon was chosen because it has the highest tweet volume (154k pairs) and the broadest mix of intents in the dataset, making retrieval more meaningful. A multi-brand agent would require per-brand vector stores and routing logic that adds complexity with no benefit for a single-brand evaluation.

---

## 13. `human_verified` Flag in Golden Set

**Decision:** Every golden set row is written with `human_verified: false` and must be manually set to `true` by a human reviewer before eval runs.

**Why:** LLM pre-labelling is fast but makes mistakes, especially on borderline cases. Treating all rows as unverified by default forces the reviewer to actively assess each label rather than passively accept it. The eval harness can be configured to run on only verified rows, filtering out any rows the reviewer flagged as incorrect.

---

## 14. `eval/` is Fully Standalone — No Import from `app/`

**Decision:** The `trivial` and `tfidf` baseline systems in `eval/evaluate.py` do not import anything from `app/`. Only the `AgentSystem` class imports from `app/`.

**Why:** The eval harness must be runnable without starting the FastAPI server or having the full app environment wired up. Standalone evaluation also means the baselines are immune to changes in `app/` — they will always produce consistent, repeatable results.

---

## 15. Single `uvicorn --workers 1` in Docker

**Decision:** Run uvicorn with a single worker in Docker.

**Why:** Multiple workers with async SQLite and ChromaDB in a single container creates file locking contention. One worker is safe, and for this evaluation workload, throughput is irrelevant. In production, you'd use multiple containers behind a load balancer instead of multiple workers.