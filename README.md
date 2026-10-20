# FidelityMem

**Recall-constrained adaptive hybrid memory for LLM agents.**

FidelityMem is a Python memory layer that decides what an agent keeps in
active context, what moves to long-term memory, how hard each memory may be
compressed, and when the original evidence must be restored. Its control
variable is not a vague "memory percentage" but a constraint:

```python
from fidelitymem import FidelityMem, MemoryPolicy, Message

policy = MemoryPolicy(
    max_context_tokens=4000,
    minimum_recall_retention=0.95,   # keep >= 95% of raw-index recall
    maximum_contradiction_rate=0.01, # in retrieved context
    latency_budget_ms=300,
    mode="balanced",                 # fast | balanced | high_recall
)
fm = FidelityMem(policy)

fm.observe(Message(role="user", content="On 2026-03-04 the deployment of billing-api failed because migration 18 removed a required column."))
fm.remember("billing-api depends on auth-service for request validation.")
fm.remember("auth-service runs on PostgreSQL and is owned by Priya.")

ctx = fm.build_context("which database does the service that billing-api depends on run on")
print(ctx.render())      # retrieved memories (compressed or raw-expanded) + working window
fm.feedback([r.memory.id for r in ctx.retrieved], useful=True)

fm.optimise()            # guarded search for the cheapest policy that keeps recall >= 0.95
print(fm.stats())
```

No LLM is required. Every stage is deterministic by default and accepts an
LLM through a one-method protocol (`Compressor`, `Extractor`, `Embedder`).

## What is in the box

| Module | Role |
| --- | --- |
| `window.py` | Adaptive working window: per-message retention score, boundary triggers (token pressure, topic shift, task completion, inactivity), eviction into extraction. |
| `extraction.py` | Candidate memories with types (semantic / episodic / procedural / relational), entities, numbers, dates, negations, relations. |
| `validation.py` | Stage 1 source & consistency (trust, contradiction, supersede-not-delete, untrusted-instruction guard). Stage 2 relevance & utility (durability score, fact-aware dedupe). |
| `compression.py` | Four-level compression ladder and the **fidelity gate**: probes built from the raw source must survive, or the level backs off. Stage 3. |
| `stores/` | Immutable archival store (JSONL-persistable), BM25, dense index with a hermetic hashing embedder, relation graph with relation-following expansion and hub penalty, trie for exact/prefix identifiers. |
| `retrieval/` | Query classification, per-type plans, RRF fusion + ranking function, relation chaining for multi-hop, sufficiency check and raw-source expansion, token-budgeted context. |
| `controller.py` | Recall-constrained policy search over (compression level, top-k, expansion budget, routes) with tuning/hold-out split, promotion, rejected-candidate audit and rollback. |
| `eval/` | Metrics (Recall@K, complete-set recall, MRR, nDCG, retention), a consistent-world synthetic benchmark with gold evidence ids and hard distractors, classic baselines, and architecture-faithful re-implementations of Mem0, Letta/MemGPT, Zep/Graphiti, A-MEM, HippoRAG and a sliding-window + trie variant. |

## Benchmark

```bash
pip install -e ".[dev]"
pytest -q
fidelitymem-bench --memories 1000 --queries 100 --k 5 10
fidelitymem-bench --memories 10000 --queries 200 --k 10 --quick
```

Result tables for 1k and 10k memories are in `docs/results/`. The design
discussion, literature review, comparison matrix against current memory
systems and an explicit statement of what is and is not demonstrated are in
`docs/DESIGN.md`. The original research concept is `docs/CONCEPT.md`.

## Status

Phases 1–5 of the concept roadmap are implemented in-process (retrieval
baseline, hybrid memory, adaptive window and compression, three-stage
validation, guarded policy optimisation). Phase 6 (LLM-judged LoCoMo /
LongMemEval, durable stores, million-memory scale) is not.
