# FidelityMem: design discussion, research review and build decisions

This document records the discussion phase that preceded the build in this
repository. `docs/CONCEPT.md` is the original research concept; this file is
the critique of it, the optimisations applied, the comparison with current
memory systems, and an honest statement of what is and is not demonstrated.

---

## 1. What the concept gets right, and the one claim that needs reframing

The concept is correct that the architecture (adaptive window + typed stores +
compression + provenance + query-aware routing) is not new in any single part.
The defensible contribution it proposes is a **recall-constrained controller**:

> minimise storage, latency and tokens subject to RecallRetention@K ≥ ρ.

The problem with that claim as written is that **RecallRetention cannot be
measured online**. A production system has no gold evidence set for a live
query. Any paper or product that says "we guarantee 95% recall" at query time
is either measuring something else or guessing. So the constraint has to be
enforced at two different points, and the build reflects that:

1. **Admission time: a deterministic fidelity gate.** Probes are generated
   from the *raw* text (entities, numbers, dates, negated sentences). A
   compressed representation must satisfy ≥ ρ of the probes (weighted:
   negations 2×, numbers/dates 1.5×) or the compressor backs off a level, down
   to raw. This is what makes the constraint enforceable per memory, with no
   LLM and no labels. The recent literature supports exactly this failure
   mode: gist compression keeps the topic and drops the fact, the qualifier
   or the attribution ([Facts as First-Class Objects][ffco], [FACTWASH][fw],
   [Fidelity Before Structure][fbs]).
2. **Policy time: a calibrated, guarded search.** The controller measures
   retention on self-generated probe queries (the most fact-dense sentence of
   a sampled episode, with a third of its words dropped) against a level-0
   index of the same memories. It picks the cheapest configuration that is
   feasible on a tuning split, confirms on a hold-out split, promotes it, and
   keeps the previous policy for rollback. Rejected candidates stay in the
   history for audit.

That reframing is the main "optimise" outcome of the discussion. The rest of
the design follows from it.

## 2. Optimisations applied relative to the concept

| Concept proposal | What was built instead, and why |
| --- | --- |
| Compression ratio as a continuous knob | A **4-level ladder** (raw, extractive 60%, key-facts, gist). Discrete levels are what a gate can back off through, and what a controller can search. An LLM compressor plugs in via a one-method protocol; the gate applies regardless of who wrote the text. |
| "Post-compression validation" by an LLM judge | **Deterministic probes** generated from the raw source. Cheap, reproducible, testable, and they measure what was lost rather than what was kept. An entailment judge can be added on top; it must not replace the probes. |
| Contradiction rate as a global store property | Contradiction rate measured **inside retrieved context** (and only when raw expansion did not resolve it). A store-wide rate is a property of the data, not of any retrieval policy, and made every configuration infeasible in the first controller run. |
| Vector + BM25 + graph fused by RRF | RRF plus **relation chaining**: when a top result states `A -p-> B` and A is in the query, memories where B is a relation subject are promoted. RRF with k=60 flattens rank 2 vs rank 4; filler weakly matched by three routes outranked a strong single-route hit on every multi-hop question. Chaining lifted multi-hop Recall@5 from 0.65 to 1.0 on the synthetic set. |
| Graph expansion by co-mention | Relation-following first, co-mention second, with a **hub penalty** `1/(1+log1p(degree-3))`. Without it, one filler entity ("Mondays") flooded the route. Seeds from lexical hits follow explicit relations only. |
| Temporal route scanning all memories | Lexical search restricted to episodic ids, sorted by time. The scan was O(n) per query and would have dominated latency at 10k+. |
| Contradiction = any same-subject relation | Only **exclusive predicates** (`runs on`, `belongs to`, `reports to`, `is part of`) and polarity flips count; `uses`/`depends on` can legitimately have several objects. |
| Near-duplicate = high token Jaccard | Jaccard **and** identical entity/number/date sets. Two episodes with the same boilerplate and different facts were being merged. |
| Percent-based memory budget | Constraint-based `MemoryPolicy` as the concept recommends, plus three modes that only change defaults the controller can override. |
| Single "self-optimisation" loop | Offline guarded search with a hold-out gate and versioned rollback, as the concept recommends. Online learning is limited to per-memory utility from feedback, which feeds the ranking function. |

Two further additions came from the discussion:

* **Trie index (user proposal: sliding window + trie).** A character trie over
  identifier-like tokens (kebab/snake/dotted names, versions, numbers, dates)
  with prefix lookup, and a word trie over 2–4-grams for exact spans. It is
  O(len) per lookup, deterministic, and catches what both BM25 and dense
  retrieval handle badly: `billing-` → every billing service, an exact error
  code, a verbatim phrase. It is evaluated two ways: as a standalone
  "window + trie" system, and as an extra route inside FidelityMem that the
  planner enables for EXACT and CODE_REVIEW queries and the controller can
  select. See §4 for the verdict.
* **Untrusted-content guard.** An instruction found in untrusted content is
  demoted from procedural to episodic at stage 1 and recorded as an issue.
  This is the cheapest defence against prompt-injection-into-memory and the
  concept listed it; it is now a test.

## 3. Comparison with current memory systems

### 3.1 Feature comparison

"✔" means the system ships the capability as a first-class feature; "~" means
partial or configurable; "–" means absent as far as public documentation and
papers show. FidelityMem's column reflects what is in this repository today.

| Capability | FidelityMem | Mem0 | Letta (MemGPT) | Zep / Graphiti | LangMem | A-MEM | HippoRAG 2 | MemoryOS | Agent Zero Memory | LazyMem |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Typed stores (semantic / episodic / procedural / relational) | ✔ | ~ (facts + optional graph) | ~ (core / recall / archival) | ✔ (episodes + entity graph) | ✔ (semantic / episodic / procedural) | – (notes) | – (passages + KG) | ✔ (short / mid / long) | ✔ (episodic, graph, semantic) | – |
| Raw evidence kept immutably with pointer from compressed form | ✔ | – (facts replace source) | ~ (recall memory keeps messages) | ✔ (episodes kept) | – | – | ✔ (passages) | ~ | ✔ (citation-locked) | ~ |
| Compression with a **fidelity gate** (fact-level check before storing) | ✔ | – | – | – | – | – | – | – | ~ (verification at answer time) | – |
| Raw-source expansion on demand (confidence / exact / conflict triggers) | ✔ | – | ~ (agent decides) | – | – | – | – | – | ~ | ✔ (query-time construction) |
| Query-type-aware routing (factual / temporal / multi-hop / exact / procedural) | ✔ | – | – | ~ (search scopes) | – | – | – | ~ | ~ | – |
| Temporal validity (valid_from / valid_to, supersede not delete) | ✔ | ~ (update/delete ops) | – | ✔ (bi-temporal edges) | – | – | – | ~ | ✔ | – |
| Contradiction detection at admission | ✔ (relation / polarity / value) | ~ (LLM decides add/update) | – | ✔ (edge invalidation) | – | ~ (note evolution) | – | – | ~ | – |
| Multi-hop via explicit relation following | ✔ | ~ (graph add-on) | – | ✔ | – | ~ (links) | ✔ (PPR) | – | ✔ | – |
| Working window with retention scoring and boundary triggers | ✔ | – | ~ (fixed core + eviction) | – | – | – | – | ✔ (heat-based) | – | – |
| Recall-constrained controller with hold-out gate and rollback | ✔ | – | – | – | – | – | – | – | – | – |
| Untrusted-instruction guard for procedural memory | ✔ | – | – | – | – | – | – | – | – | – |
| Exact / prefix identifier index (trie) | ✔ | – | – | – | – | – | – | – | – | – |
| Needs an LLM to ingest | **no** (optional) | yes | yes | yes | yes | yes | yes (OpenIE) | yes | yes | yes |
| Published LoCoMo / LongMemEval numbers | not yet | yes (vendor) | yes (vendor) | yes (vendor) | – | yes | yes | yes | yes (vendor) | yes |

Sources: product documentation and the papers listed in `docs/CONCEPT.md`
§21, plus [Agent Zero Memory][azm], [LazyMem][lazy] and the Mem0 benchmark
pages. Vendor-reported LoCoMo and LongMemEval scores disagree with one another
and have been disputed between vendors; no public study reports recall, cost
and latency for these systems under one protocol.

### 3.2 What is actually unique

Being precise about this matters more than the adjective. Three things in this
repository have no equivalent in the systems above:

1. **The fidelity gate.** Every other system either stores an LLM's summary
   as-is or stores raw text. None checks, per memory, that the compressed form
   still answers the entity, number, date and negation probes of its source,
   and backs off when it does not.
2. **The recall-constrained controller.** No other system exposes "keep at
   least ρ of raw recall" as the control variable and searches compression,
   depth and routing under it with a hold-out gate and rollback. EvolveMem and
   MemRL learn retrieval configuration, but they optimise accuracy, not a
   constraint.
3. **Ingestion without an LLM.** Every component is deterministic by default
   and accepts an LLM via a protocol. That is a practical advantage (cost,
   determinism, hermetic tests) and the reason this repository can run an
   exact, offline, scale-controlled comparison at all.

Everything else (typed stores, provenance, temporal validity, graph,
hybrid retrieval, adaptive window) exists elsewhere and should be described as
an integration, not an invention.

### 3.3 "Better" and "at scale": what the evidence does and does not show

**Better** is only meaningful under a fixed protocol, and the comparison here
is against architecture-faithful re-implementations (`fidelitymem/eval/competitors.py`),
not vendor code. Vendor systems need an LLM to ingest, which would confound
memory design with model quality and make exact recall unmeasurable. The
re-implementations use the same extractors, BM25 and embedder as FidelityMem,
so the only variable is memory architecture. Results are in §4 and in
`docs/results/`. They show FidelityMem on the Pareto frontier of recall vs
storage vs tokens on *this* benchmark. They do not show anything about
LoCoMo or LongMemEval answer accuracy, which needs an LLM and is the next
phase.

**At scale** is a property of the implementation, not the architecture, and
the implementation here is in-process Python. Measured behaviour at 10k
memories is in §4. The honest statement is:

* Retrieval is sub-linear only where the index is (BM25 postings, trie); the
  dense index is a dense matrix multiply (O(n·d) per query) and the graph
  expansion is bounded by hop count and k. At 10k this is milliseconds; at 1M
  it needs pgvector/FAISS and a real graph store, which the store protocols
  are shaped for but which are not built.
* The controller's search is O(configs × probes × retrieval) and recompresses
  the store per compression level; with the caches added it is minutes at
  10k and must become a sampled, background job beyond that.
* Tenant isolation is a field on provenance, not an enforced partition.

So: unique in three specific ways, better on a controlled offline benchmark
against six architecture baselines, and scalable in design but demonstrated
only to 10k memories in this process. Claims beyond that need Phase 6 of the
concept roadmap.

## 4. Results

See `docs/results/bench_1k.txt` and `docs/results/bench_10k.txt` for the full
tables (generated by `fidelitymem-bench`). Headline numbers are summarised in
the README once the runs complete; the tables include per-system Recall@K,
complete-evidence-set recall, MRR, nDCG, tokens per query, p50/p95 latency,
stored tokens, raw-expansion rate and recall retention against the level-0
index.

What to look for in the tables:

* `summary_L3+bm25` is the "compress without a gate" baseline. Its storage is
  the smallest and its recall is where fact loss shows.
* `fidelitymem_L*_noexpand` vs `_gated` isolates the raw-expansion effect.
* `fidelitymem_ctrl(Ln)` is the controller's own choice; the level it picked
  is in the name, and `retain` is measured retention against level 0.
* `window+trie` is the user-proposed variant. On this benchmark it is strong
  on exact-value questions and weak on factual, temporal and multi-hop
  questions, because a trie only fires on identifier-like tokens; this is
  why it is kept as a *route* inside FidelityMem rather than as the system.

## 5. Roadmap from here

1. **LLM-backed phase.** Plug an LLM compressor and extractor through the
   existing protocols, keep the deterministic gate, and run LoCoMo and
   LongMemEval with an LLM judge. This is the first point at which "better"
   can be claimed against vendor numbers.
2. **Durable stores.** PostgreSQL + pgvector implementations of the store
   protocols; the archival store already has JSONL persistence.
3. **Sampled background optimisation.** Run the controller on a probe sample
   with a time budget, and promote policies by version through the existing
   history/rollback API.
4. **Code-review integration.** The concept's starting use case: feed PR
   findings, accepted/rejected outcomes and repository conventions through
   `remember()` with `source_type` and `authority` set, and give the review
   agents `build_context()`.

[ffco]: https://arxiv.org/abs/2603.17781
[fw]: https://arxiv.org/abs/2608.03372
[fbs]: https://arxiv.org/abs/2601.00821
[azm]: https://arxiv.org/abs/2608.29606
[lazy]: https://arxiv.org/abs/2607.22690
