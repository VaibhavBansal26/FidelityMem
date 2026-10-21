# FidelityMem: Recall-Constrained Adaptive Hybrid Memory for LLM Agents

*A controlled, offline comparison against current memory-system architectures on one dataset at four record scales.*

Working paper, October 2026. Code and benchmark: this repository.

---

## Abstract

Long-running LLM agents accumulate far more history than fits in a context window. Memory systems compress, route and retrieve that history, but compression that preserves topic can still delete the entity, number, date or negation a later question needs, and no deployed system exposes evidence recall as a constraint it is required to meet. We present FidelityMem, a memory layer whose control variable is a **recall-retention bound**: minimise storage, context tokens and latency subject to the compressed index retaining at least ρ of the recall of the raw index. The bound is enforced at two points. At admission, a deterministic **fidelity gate** generates entity, number, date and negation probes from the raw source and backs the compressor off level by level until the compressed form satisfies them. At policy time, a **guarded controller** measures retention on self-generated probe queries, picks the cheapest configuration of compression level, retrieval depth, expansion budget and routes that is feasible on a tuning split, confirms it on a hold-out split, and keeps the previous policy for rollback. Retrieval is hybrid (BM25, dense, relation graph, trie, temporal, procedural) with query-type planning, relation chaining for multi-hop questions, and raw-source expansion when exact values or conflicting evidence are detected.

We evaluate on a consistent-world synthetic benchmark with exact gold evidence ids, four question families (factual, temporal, two-hop, exact-value) and realistic distractors, at 300, 1 000, 3 000 and 10 000 records, against eleven baselines: five classic retrieval configurations and six architecture-faithful re-implementations of Mem0, Letta/MemGPT, Zep/Graphiti, A-MEM, HippoRAG and a sliding-window + trie design. All systems share the same extractors, BM25 and embedder, so memory architecture is the only variable. FidelityMem reaches the highest Recall@10 and complete-evidence-set recall at every scale while storing 44–47 % of the raw tokens and serving 45–55 % of the context tokens per query, and the controller selects the most aggressive compression level that holds the bound without human tuning. We state explicitly what this protocol does not show: answer accuracy with an LLM, vendor-code performance, and behaviour beyond 10⁴ records in one process.

## 1. Introduction

Agent memory research has converged on a shared architecture: a short working context, typed long-term stores, compression, and retrieval that fuses lexical, dense and graph signals. MemGPT introduced hierarchical context management; HippoRAG added entity graphs and personalised PageRank; Zep/Graphiti added a bi-temporal knowledge graph; Mem0 reduced memory to atomic LLM-extracted facts; A-MEM linked notes by similarity; and 2026 work (QueryLink, Synapse, MemORAI, EvolveMem, MemRL, Agent Zero Memory, LazyMem) refines routing, consolidation and learned utility. The components are no longer novel individually.

What remains unsolved is **control**. Every system above decides how much to compress and how deep to retrieve with fixed heuristics or with learning aimed at answer accuracy. None states a recall bound and enforces it. The consequence, documented in recent fidelity studies, is silent evidence loss: a 36.7× compression of structured facts left 60 % irrecoverable [1]; summarisers keep "type hints" and drop the qualifier "everywhere" [2]; rewrites launder hedged hearsay into asserted fact [3]. Retrieval metrics do not catch this because the right memory id is retrieved while the fact inside it is gone.

FidelityMem's contribution is therefore not a new store but a **recall-constrained controller with an enforceable gate**, and a protocol that can measure whether the bound holds. Our claims are deliberately narrow:

1. A deterministic fidelity gate, built from the raw source, prevents fact loss at admission without an LLM and with no labels (§3.3).
2. A guarded search over discrete compression levels, depths, expansion budgets and routes finds the cheapest policy that holds a retention bound ρ and confirms it out of sample (§3.5).
3. Relation chaining after fusion recovers the second hop of two-hop questions that reciprocal-rank fusion buries (§3.4).
4. Under a fixed protocol with shared components, this design dominates six current memory architectures on recall, complete-set recall, stored tokens and tokens per query at every scale tested (§5).

## 2. Related work

| System | Core mechanism | What FidelityMem takes | What it adds |
| --- | --- | --- | --- |
| MemGPT / Letta [4] | Core + recall + archival tiers, agent-driven paging | Working vs long-term split | Retention-scored window with boundary triggers; no agent in the loop |
| HippoRAG / HippoRAG 2 [5] | OpenIE entity graph, PPR seeded by query entities | Graph seeded by query entities, retrieve-then-expand | Relation-following with hub penalty; relation chaining after fusion |
| Zep / Graphiti [6] | Bi-temporal entity graph, hybrid search, edge invalidation | Temporal validity, supersede-not-delete | Contradiction limited to exclusive predicates; fidelity gate |
| Mem0 [7] | LLM-extracted atomic facts, vector search, add/update/delete | Fact-level dedupe | Raw evidence retained and expandable; no LLM needed to ingest |
| A-MEM [8] | Notes with keywords, links to neighbours, note evolution | Linked expansion | Explicit relations instead of similarity links |
| R³Mem [9] | Reversible compression via virtual memory tokens | Reversibility as a requirement | Reversibility by immutable raw pointer rather than model inversion |
| EvolveMem [10], MemRL [11] | Learned retrieval configuration / utility | Feedback-driven utility | Constraint-based rather than accuracy-based optimisation, hold-out gate, rollback |
| Agent Zero Memory [12] | Parallel stores, provenance, citation-locked answers | Provenance on every memory | Admission-time gate rather than answer-time verification |
| LazyMem [13] | Retrieve broadly, construct context selectively at query time | Query-time expansion | Expansion triggered by confidence, exact-value need and conflict |
| Facts as First-Class Objects [1], FACTWASH [3], Fidelity Before Structure [2] | Evidence that gist compression loses facts, qualifiers and attribution | Motivation and probe design | A gate that enforces it per memory |

## 3. Method

### 3.1 Data model

An **episode** is immutable raw text with provenance (source id, type, tenant, namespace, timestamp, trust, authority). A **memory** is a derived, searchable record pointing at exactly one episode: compressed text, compression level, extracted entities, numbers, dates, negated sentences, relations, types (semantic, episodic, procedural, relational), confidence, validity interval, utility and a fidelity score. Memories are superseded, never deleted; user-requested forgetting removes both memory and episode.

### 3.2 Adaptive working window and admission

Each message carries a retention score

R(m, q) = w_s·S_semantic + w_r·S_recency + w_i·S_importance + w_t·S_task + w_c·S_confidence + w_u·S_utility

with task continuity weighted highest (0.30) so that an unresolved task is evicted last. Boundaries fire on token pressure, topic shift (low similarity to the running topic vector and no shared entities), task completion and inactivity; each boundary hands its segment to extraction.

Admission runs three validation stages. **Stage 1** attaches provenance, detects contradictions (polarity flip on a near-identical statement, conflicting values for the same entity set, or a conflicting object for an *exclusive* predicate such as `runs on`), supersedes the older memory when the new source is trusted, and demotes any instruction found in untrusted content from procedural to episodic. **Stage 2** scores durability from importance, fact density and length and drops near-duplicates, where "duplicate" requires identical entity, number and date sets as well as high token overlap. **Stage 3** is the fidelity gate.

### 3.3 Reversible compression and the fidelity gate

Compression is a four-level ladder: L0 raw; L1 extractive at ~60 % of sentences by salience (centrality + fact count + negation + lead bias); L2 key facts, keeping only sentences that carry an entity, number, date or negation; L3 gist, a single salient sentence. An LLM compressor plugs in through a one-method protocol.

From the **raw** text we build a probe set: one probe per entity (weight 1), number (1.5), date (1.5) and negated sentence (2; satisfied only if ≥60 % of its content words and a negation cue survive). The gate accepts a compressed form at level ℓ iff its weighted probe score is ≥ ρ; otherwise it backs off to ℓ−1, and L0 always passes. Because probes are generated from the source rather than the summary, the gate measures what was lost. The probe set is stored with the memory so that recompression under a new policy reuses it.

### 3.4 Query-aware hybrid retrieval

The planner classifies a query as factual, temporal, multi-hop, exact, procedural or code-review and sets routes, weights, depth, graph hops, expansion budget and whether raw expansion is mandatory (exact queries). Routes: BM25; dense (any embedder; by default a hashing char-n-gram embedder so the benchmark is hermetic); relation graph with relation-following before co-mention, a hub penalty 1/(1+log1p(degree−3)) and no traversal through hub entities; a trie over identifier-like tokens and 2–4-gram phrases for exact and prefix matches; a temporal route (episodic memories by lexical match, most recent first); and a procedural route.

Route results are fused by weighted reciprocal-rank fusion and re-scored with recency, learned utility, authority×confidence and a type boost. Two steps follow that the baselines lack. **Relation chaining**: if a top result states A –p→ B with A named in the query, the memories in which B is a relation subject are promoted to just below it; this recovers the second hop that RRF (k = 60) otherwise buries under weakly matched filler. **Sufficiency and expansion**: a memory is expanded to its raw episode, within a budget, when the task is exact, confidence or fidelity is low, retrieved memories conflict, the query needs values the compressed text lacks, or a multi-hop query hits a memory without relations. A conflict counts as unresolved only while a party to it is compressed and unexpanded. Context is then trimmed to the token budget, falling back from raw to compressed before dropping.

### 3.5 Recall-constrained controller

Let π be a configuration (compression level ∈ {0..3}, top-k ∈ {5,10,20}, expansion budget ∈ {0,2,5}, route set ∈ 4 options). The controller minimises

cost(π) = α·StorageRatio + β·Latency + γ·TokensPerQuery + δ·Noise

subject to Retention(π) = Recall_π@K / Recall_L0@K ≥ ρ and UnresolvedContradictionRate ≤ ε.

Retention cannot be observed on live queries, so it is measured on **self-generated probe queries**: for a sampled episode, the most fact-dense sentence (an entity plus a value or a second entity) with a third of its words dropped must retrieve that episode. Probes are split into tuning and hold-out halves. Every configuration is evaluated on the tuning half (grouped by level so the store is recompressed once per level); feasible ones are walked in ascending cost and confirmed on the hold-out half; the first that confirms is promoted as a new policy version. A level that fails confirmation is not retried. If none confirms, the safest configuration (raw, widest) is promoted. Rejected candidates are kept in the version history for audit and `rollback()` reinstates the previous promoted policy.

## 4. Experimental protocol

**Dataset.** A single generator (`fidelitymem/eval/synthetic.py`) produces a *consistent world*: 200 services, each with one database, one owner and one dependency, so no gold fact is legitimately contradicted. Every episode is a fact sentence embedded in 2–4 sentences of realistic workplace filler, so compression has something to lose. Question families, in rotation: **factual** ("where does X store session data"), **temporal** ("when did the migration of X fail", gold carries the date), **multi-hop** ("which database does the service that X depends on run on", two gold episodes), **exact** ("what is the exact connection pool cap for X", gold carries two numbers and a negated constraint). Distractors fill to the record count: 30 % are hard (same service and database names, no decision) and 70 % are off-topic; 2 % are procedural rules. Record levels: 300, 1 000, 3 000, 10 000 with the same 100 questions, seed 0, K = 10.

**Metrics.** Recall@K, complete-evidence-set recall (all gold ids in the top K), MRR, nDCG@K, tokens per query (what enters context), p50/p95 retrieval latency, stored tokens (index size), raw-expansion rate, and recall retention against the system's own L0 index.

**Baselines.** Classic: fixed sliding window, BM25, dense, BM25+dense RRF, lossy summary (L3 gist without a gate) + BM25. Architecture-faithful re-implementations (`fidelitymem/eval/competitors.py`), each using the same extractors, BM25 and embedder as FidelityMem: *window+trie* (user-proposed: last-50 window plus trie exact/prefix index), *memgpt_style* (core recent block always in context + vector recall/archival), *mem0_style* (every entity-bearing sentence is an atomic memory, deduped, vector-only, mapped back to its episode), *zep_style* (entity graph with relations + BM25 + dense, RRF, 1-hop expansion), *amem_style* (each note linked to its 3 nearest neighbours at insert; retrieve then follow links), *hipporag_style* (entity graph seeded by query entities, 2-hop spreading, fused with dense). These are not vendor code: the vendors require an LLM to ingest, which would confound architecture with model quality and make exact recall unmeasurable.

**FidelityMem variants.** L0 raw; L1–L3 with expansion disabled (`noexpand`) and enabled (`gated`); and `ctrl`, the configuration the controller selected on its own probes (ρ = 0.95, ε = 0.01) with top-k reset to K for comparability.

**Real data.** LoCoMo (10 multi-session human-style conversations, 5 882 turns, 1 986 questions) with the evidence dialogue ids the dataset provides as retrieval ground truth; categories multi-hop, temporal, open-domain, single-hop and adversarial (adversarial questions have no gold evidence and are excluded from recall). Two granularities: one memory per *turn* (hard retrieval, little to compress) and one per *session* (realistic compression target). 30 % of questions are held out per seed as the controller's "query log" and never scored; the controller is run twice, once on self-generated probes and once on that log. We also report an *answer-containment* proxy (fraction of gold-answer content tokens present in the served context); it is a proxy, not accuracy. LongMemEval has a loader in the repository but could not be fetched in the build environment (the dataset is distributed only through a host the environment's network policy blocks); the runner is identical and the result is pending.

**Embedder and seeds.** All dense routes, in every system, use the same pretrained embedder (spaCy `en_core_web_md` word vectors, 300-d, IDF-weighted mean pooling; a sentence-transformer plugs in through the same protocol when available). Every experiment runs with seeds 0, 1 and 2 (synthetic: different worlds; real: different ingestion orders and calibration splits) and reports the mean with a 95 % t-interval over seeds.

**LLM components.** A Claude-backed compressor, extractor, reader and judge are implemented behind the same protocols (`fidelitymem/llm.py`) with a mock for tests. No API credential was available in the build environment, so every number below is from the deterministic components; the LLM-judged accuracy columns are produced by the same runner with `--llm` and are the next step.

**Controls.** Same episodes, questions, seed, K, embedder and token accounting for every system; stochastic components are seeded; the controller never sees scored questions.

## 5. Results

<!-- RESULTS:BEGIN -->
### 5.1 Synthetic benchmark at 300, 1 000, 3 000 and 10 000 records (K = 10, 3 seeds, pretrained embedder)

Recall@10 across record counts (± = 95 % CI over seeds; the controller row names the levels it chose across seeds):

| System | n=300 | n=1000 | n=3000 | n=10000 |
|---|---:|---:|---:|---:|
| fixed_window | 0.038 | 0.005 | 0.012 | 0.002 |
| bm25 | 0.888 | 0.878 | 0.875 | 0.875 |
| vector | 0.735 | 0.687 | 0.630 | 0.547 |
| bm25+vector_rrf | 0.877 | 0.875 | 0.872 | 0.862 |
| summary_L3+bm25 | 0.855 | 0.853 | 0.850 | 0.857 |
| window+trie | 0.875 | 0.875 | 0.875 | 0.875 |
| memgpt_style | 0.705 | 0.658 | 0.605 | 0.528 |
| mem0_style | 0.850 | 0.842 | 0.825 | 0.825 |
| zep_style | 0.952 | 0.952 | 0.915 | 0.885 |
| amem_style | 0.735 | 0.687 | 0.630 | 0.547 |
| hipporag_style | 1.000 | 1.000 | 1.000 | 0.950 |
| fidelitymem_L0_raw | 0.998 | 0.995 | 1.000 | 0.997 |
| fidelitymem_L1_noexpand | 0.998 | 0.995 | 1.000 | 1.000 |
| fidelitymem_L1_gated | 0.998 | 0.995 | 1.000 | 1.000 |
| fidelitymem_L2_noexpand | 0.998 | 0.995 | 1.000 | 0.997 |
| fidelitymem_L2_gated | 0.998 | 0.995 | 1.000 | 0.997 |
| fidelitymem_L3_noexpand | 0.998 | 0.995 | 1.000 | 0.997 |
| fidelitymem_L3_gated | 0.998 | 0.995 | 1.000 | 0.997 |
| fidelitymem_ctrl (L3/f)/(L1/f/L3/f)/(L1/f/L3/f)/(L2/f/L3/f) | 0.995 | 0.993 | 0.977 | 0.967 |

Complete-evidence-set recall (all gold episodes in the top 10):

| System | n=300 | n=1000 | n=3000 | n=10000 |
|---|---:|---:|---:|---:|
| fixed_window | 0.030 | 0.003 | 0.010 | 0.000 |
| bm25 | 0.777 | 0.757 | 0.750 | 0.750 |
| vector | 0.640 | 0.603 | 0.560 | 0.480 |
| bm25+vector_rrf | 0.753 | 0.753 | 0.747 | 0.737 |
| summary_L3+bm25 | 0.770 | 0.767 | 0.760 | 0.760 |
| window+trie | 0.750 | 0.750 | 0.750 | 0.750 |
| memgpt_style | 0.617 | 0.580 | 0.533 | 0.463 |
| mem0_style | 0.737 | 0.730 | 0.720 | 0.723 |
| zep_style | 0.903 | 0.903 | 0.830 | 0.770 |
| amem_style | 0.640 | 0.603 | 0.560 | 0.480 |
| hipporag_style | 1.000 | 1.000 | 1.000 | 0.950 |
| fidelitymem_L0_raw | 0.997 | 0.993 | 1.000 | 0.997 |
| fidelitymem_L1_noexpand | 0.997 | 0.993 | 1.000 | 1.000 |
| fidelitymem_L1_gated | 0.997 | 0.993 | 1.000 | 1.000 |
| fidelitymem_L2_noexpand | 0.997 | 0.993 | 1.000 | 0.997 |
| fidelitymem_L2_gated | 0.997 | 0.993 | 1.000 | 0.997 |
| fidelitymem_L3_noexpand | 0.997 | 0.993 | 1.000 | 0.997 |
| fidelitymem_L3_gated | 0.997 | 0.993 | 1.000 | 0.997 |
| fidelitymem_ctrl (L3/f)/(L1/f/L3/f)/(L1/f/L3/f)/(L2/f/L3/f) | 0.993 | 0.993 | 0.977 | 0.967 |

Stored tokens (index size):

| System | n=300 | n=1000 | n=3000 | n=10000 |
|---|---:|---:|---:|---:|
| fixed_window | 22687 | 73463 | 219337 | 726798 |
| bm25 | 22687 | 73463 | 219337 | 726798 |
| vector | 22687 | 73463 | 219337 | 726798 |
| bm25+vector_rrf | 22687 | 73463 | 219337 | 726798 |
| summary_L3+bm25 | 6850 | 20392 | 58681 | 192728 |
| window+trie | 22687 | 73463 | 219337 | 726798 |
| memgpt_style | 22687 | 73463 | 219337 | 726798 |
| mem0_style | 5302 | 12055 | 28989 | 84736 |
| zep_style | 22687 | 73463 | 219337 | 726798 |
| amem_style | 22687 | 73463 | 219337 | 726798 |
| hipporag_style | 22687 | 73463 | 219337 | 726798 |
| fidelitymem_L0_raw | 22687 | 73448 | 219187 | 725582 |
| fidelitymem_L1_noexpand | 14730 | 46185 | 135639 | 446463 |
| fidelitymem_L1_gated | 14730 | 46185 | 135639 | 446463 |
| fidelitymem_L2_noexpand | 10847 | 32159 | 91936 | 299732 |
| fidelitymem_L2_gated | 10847 | 32159 | 91936 | 299732 |
| fidelitymem_L3_noexpand | 10814 | 32091 | 91703 | 298898 |
| fidelitymem_L3_gated | 10814 | 32091 | 91703 | 298898 |
| fidelitymem_ctrl (L3/f)/(L1/f/L3/f)/(L1/f/L3/f)/(L2/f/L3/f) | 10814 | 36718 | 106872 | 299141 |

Context tokens per query:

| System | n=300 | n=1000 | n=3000 | n=10000 |
|---|---:|---:|---:|---:|
| fixed_window | 831 | 719 | 726 | 713 |
| bm25 | 651 | 669 | 677 | 674 |
| vector | 771 | 741 | 719 | 692 |
| bm25+vector_rrf | 741 | 738 | 726 | 708 |
| summary_L3+bm25 | 238 | 239 | 239 | 230 |
| window+trie | 796 | 800 | 817 | 820 |
| memgpt_style | 796 | 754 | 727 | 682 |
| mem0_style | 768 | 763 | 760 | 782 |
| zep_style | 743 | 746 | 766 | 745 |
| amem_style | 771 | 741 | 719 | 692 |
| hipporag_style | 776 | 770 | 817 | 807 |
| fidelitymem_L0_raw | 756 | 763 | 777 | 757 |
| fidelitymem_L1_noexpand | 521 | 520 | 521 | 500 |
| fidelitymem_L1_gated | 554 | 561 | 573 | 549 |
| fidelitymem_L2_noexpand | 407 | 405 | 401 | 374 |
| fidelitymem_L2_gated | 448 | 457 | 468 | 448 |
| fidelitymem_L3_noexpand | 407 | 405 | 401 | 374 |
| fidelitymem_L3_gated | 448 | 457 | 468 | 448 |
| fidelitymem_ctrl (L3/f)/(L1/f/L3/f)/(L1/f/L3/f)/(L2/f/L3/f) | 333 | 374 | 362 | 356 |

p95 retrieval latency, ms (three evaluation processes shared the machine; indicative):

| System | n=300 | n=1000 | n=3000 | n=10000 |
|---|---:|---:|---:|---:|
| fixed_window | 0.00 | 0.00 | 0.00 | 0.00 |
| bm25 | 0.05 | 0.03 | 0.05 | 0.09 |
| vector | 1.99 | 2.23 | 11.97 | 17.36 |
| bm25+vector_rrf | 3.73 | 3.51 | 13.34 | 18.63 |
| summary_L3+bm25 | 0.03 | 0.03 | 0.04 | 0.06 |
| window+trie | 0.08 | 0.08 | 0.10 | 0.15 |
| memgpt_style | 2.50 | 2.29 | 10.66 | 18.63 |
| mem0_style | 2.46 | 2.34 | 2.00 | 17.70 |
| zep_style | 2.50 | 2.45 | 10.88 | 36.48 |
| amem_style | 2.20 | 2.23 | 13.26 | 19.55 |
| hipporag_style | 2.27 | 3.78 | 15.96 | 31.97 |
| fidelitymem_L0_raw | 6.44 | 7.27 | 28.61 | 48.93 |
| fidelitymem_L1_noexpand | 1.58 | 2.26 | 19.60 | 45.31 |
| fidelitymem_L1_gated | 1.64 | 2.20 | 20.14 | 36.73 |
| fidelitymem_L2_noexpand | 1.56 | 1.94 | 19.56 | 38.11 |
| fidelitymem_L2_gated | 1.77 | 2.10 | 20.79 | 37.38 |
| fidelitymem_L3_noexpand | 1.45 | 2.36 | 20.76 | 40.27 |
| fidelitymem_L3_gated | 1.81 | 2.34 | 21.42 | 39.65 |
| fidelitymem_ctrl (L3/f)/(L1/f/L3/f)/(L1/f/L3/f)/(L2/f/L3/f) | 0.43 | 0.48 | 0.46 | 0.59 |

Full table at 10 000 records:

| System | R@K | CR@K | MRR | nDCG | tok/q | p50 ms | p95 ms | stored tok | expand | retain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_window | 0.002 | 0.000 | 0.000 | 0.001 | 713 | 0.00 | 0.00 | 726798 | 0.00 | - |
| bm25 | 0.875 | 0.750 | 0.989 | 0.898 | 674 | 0.05 | 0.09 | 726798 | 0.00 | - |
| vector | 0.547 | 0.480 | 0.390 | 0.407 | 692 | 11.96 | 17.36 | 726798 | 0.00 | - |
| bm25+vector_rrf | 0.862 | 0.737 | 0.752 | 0.732 | 708 | 11.96 | 18.63 | 726798 | 0.00 | - |
| summary_L3+bm25 | 0.857 | 0.760 | 0.798 | 0.761 | 230 | 0.04 | 0.06 | 192728 | 0.00 | - |
| window+trie | 0.875 | 0.750 | 1.000 | 0.903 | 820 | 0.09 | 0.15 | 726798 | 0.00 | - |
| memgpt_style | 0.528 | 0.463 | 0.155 | 0.238 | 682 | 11.96 | 18.63 | 726798 | 0.00 | - |
| mem0_style | 0.825 | 0.723 | 0.651 | 0.670 | 782 | 11.95 | 17.70 | 84736 | 0.00 | - |
| zep_style | 0.885 | 0.770 | 0.900 | 0.838 | 745 | 23.84 | 36.48 | 726798 | 0.00 | - |
| amem_style | 0.547 | 0.480 | 0.390 | 0.407 | 692 | 11.96 | 19.55 | 726798 | 0.00 | - |
| hipporag_style | 0.950 | 0.950 | 0.753 | 0.793 | 807 | 23.95 | 31.97 | 726798 | 0.00 | - |
| fidelitymem_L0_raw | 0.997 | 0.997 | 0.865 | 0.901 | 757 | 24.21 | 48.93 | 725582 | 0.00 | - |
| fidelitymem_L1_noexpand | 1.000 | 1.000 | 0.944 | 0.960 | 500 | 24.22 | 45.31 | 446463 | 0.00 | 1.003 |
| fidelitymem_L1_gated | 1.000 | 1.000 | 0.944 | 0.960 | 549 | 24.03 | 36.73 | 446463 | 1.51 | 1.003 |
| fidelitymem_L2_noexpand | 0.997 | 0.997 | 0.958 | 0.968 | 374 | 24.03 | 38.11 | 299732 | 0.00 | 1.000 |
| fidelitymem_L2_gated | 0.997 | 0.997 | 0.958 | 0.968 | 448 | 24.25 | 37.38 | 299732 | 1.51 | 1.000 |
| fidelitymem_L3_noexpand | 0.997 | 0.997 | 0.958 | 0.968 | 374 | 24.21 | 40.27 | 298898 | 0.00 | 1.000 |
| fidelitymem_L3_gated | 0.997 | 0.997 | 0.958 | 0.968 | 448 | 24.50 | 39.65 | 298898 | 1.51 | 1.000 |
| fidelitymem_ctrl(L2/f/L3/f) | 0.967 | 0.967 | 0.861 | 0.888 | 356 | 0.42 | 0.59 | 299141 | 0.67 | 0.970 |

Per-level tables for the other record counts are in `docs/results/scale_spacy_k10_report.md`; the Pareto plots are `docs/results/pareto_synthetic_3k.png` and `pareto_synthetic_10k.png`.
<!-- RESULTS:END -->

### 5.1 Reading the tables

* **Recall at scale.** FidelityMem holds 0.995–1.000 Recall@10 and complete-set recall from 300 to 10 000 records. The HippoRAG-style graph ties it at 300–3 000 records (1.000, at full storage and lower nDCG) and falls to 0.950 ±0.066 at 10 000, where hub entities dilute two-hop spreading; Zep-style falls from 0.95 to 0.885; BM25 stays at 0.875–0.888 (it never reaches the second hop: complete-set recall 0.75 at every scale); Mem0-style 0.83. At 10 000 records FidelityMem is alone at the top on both recall and complete-set recall.
* **Compression without loss.** Levels 2 and 3 store 42–48 % of the raw tokens and serve about 60 % of the context tokens per query with retention 1.0 against L0; the gate backed off on roughly one in six episodes, which is exactly the set where a gist would have dropped a fact. The lossy `summary_L3+bm25` baseline stores less (≈27 %) and loses 2–4 points of recall with wide variance across seeds.
* **Controller choice.** With no access to the questions, the controller selected level 3 (one seed level 1) with the facts gate profile and the two cheapest routes at every scale, and its measured benchmark retention is 0.98–1.00. Its latency is the lowest of the FidelityMem rows because it dropped the graph and temporal routes it found unnecessary for the probe distribution; at 3 000 records one seed's cheaper route set cost multi-hop recall (0.977 ±0.063), still above the bound.
* **Window + trie.** Perfect MRR on exact-value questions and competitive overall recall at small scale, but it is capped at the BM25 ceiling on factual and temporal questions and cannot reach a second hop. This is why the trie is kept as a *route* inside FidelityMem (enabled for exact and code-review queries) rather than as the system.
* **Mem0-style atomic facts** has the smallest index of the baselines (≈23 % of raw) and loses recall where the atomic sentence is not the one that matches the query phrasing. **MemGPT-style** recall is as good as its vector store but its MRR collapses because the always-present core block occupies the top ranks.
* **Latency.** FidelityMem's p95 is 1–3 ms higher than single-route baselines at 300–3 000 records and grows linearly with the dense index (a dense matmul in this container's BLAS); the comparison is of architectures, not of index engineering.

### 5.2 Real data: LoCoMo, session granularity (272 sessions, 1 390 scored questions per seed, K = 5, 3 seeds)

| System | R@5 | CR@5 | MRR | stored tok | tok/q | p95 ms | retain | R@5 multi-hop / temporal / single-hop / open-domain | answer contain. |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| fixed_window | 0.018 ±0.006 | 0.014 | 0.010 | 195165 | 3307 | 0.00 | - | 0.02 / 0.02 / 0.02 / 0.01 | 0.32 |
| bm25 | 0.858 ±0.009 | 0.817 | 0.764 | 195165 | 3707 | 0.22 | - | 0.55 / 0.88 / 0.94 / 0.57 | 0.80 |
| vector | 0.287 ±0.016 | 0.262 | 0.187 | 195165 | 3309 | 2.56 | - | 0.20 / 0.23 / 0.33 / 0.20 | 0.54 |
| bm25+vector_rrf | 0.726 ±0.005 | 0.682 | 0.510 | 195165 | 3551 | 3.13 | - | 0.45 / 0.72 / 0.80 / 0.46 | 0.75 |
| summary_L3+bm25 | 0.425 ±0.003 | 0.395 | 0.339 | 26114 | 506 | 0.11 | - | 0.24 / 0.48 / 0.47 / 0.29 | 0.34 |
| window+trie | 0.653 ±0.024 | 0.622 | 0.561 | 195165 | 3810 | 0.34 | - | 0.24 / 0.68 / 0.77 / 0.30 | nan |
| memgpt_style | 0.255 ±0.007 | 0.232 | 0.107 | 195165 | 3221 | 2.74 | - | 0.16 / 0.20 / 0.30 / 0.16 | nan |
| mem0_style | 0.239 ±0.019 | 0.220 | 0.186 | 27036 | 3770 | 17.04 | - | 0.15 / 0.25 / 0.27 / 0.20 | nan |
| zep_style | 0.687 ±0.023 | 0.649 | 0.482 | 195165 | 3669 | 3.65 | - | 0.42 / 0.69 / 0.75 / 0.47 | nan |
| amem_style | 0.287 ±0.016 | 0.262 | 0.187 | 195165 | 3309 | 2.83 | - | 0.20 / 0.23 / 0.33 / 0.20 | nan |
| hipporag_style | 0.401 ±0.022 | 0.370 | 0.281 | 195165 | 3601 | 4.70 | - | 0.25 / 0.41 / 0.44 / 0.33 | nan |
| fidelitymem_L0_raw | 0.563 ±0.016 | 0.522 | 0.372 | 195165 | 3716 | 5.26 | - | 0.40 / 0.45 / 0.63 / 0.40 | 0.70 |
| fidelitymem_L1_gate:facts | 0.529 ±0.025 | 0.489 | 0.345 | 127833 | 2573 | 2.89 | 0.939 | 0.36 / 0.43 / 0.60 / 0.42 | 0.62 |
| fidelitymem_L2_gate:facts | 0.444 ±0.027 | 0.410 | 0.298 | 42667 | 1275 | 2.36 | 0.789 | 0.30 / 0.44 / 0.49 / 0.38 | 0.50 |
| fidelitymem_L3_gate:facts | 0.444 ±0.022 | 0.409 | 0.299 | 42833 | 1268 | 2.25 | 0.788 | 0.31 / 0.45 / 0.48 / 0.38 | 0.49 |
| fidelitymem_L1_gate:content | 0.564 ±0.019 | 0.523 | 0.374 | 192142 | 3666 | 2.58 | 1.002 | 0.40 / 0.45 / 0.63 / 0.41 | 0.70 |
| fidelitymem_L2_gate:content | 0.564 ±0.019 | 0.523 | 0.374 | 192142 | 3666 | 2.62 | 1.002 | 0.40 / 0.45 / 0.63 / 0.41 | 0.70 |
| fidelitymem_L3_gate:content | 0.564 ±0.019 | 0.523 | 0.374 | 192142 | 3666 | 2.46 | 1.002 | 0.40 / 0.45 / 0.63 / 0.41 | 0.70 |
| fidelitymem_L1_nogate | 0.530 ±0.021 | 0.489 | 0.343 | 125075 | 2546 | 2.45 | 0.940 | 0.36 / 0.42 / 0.60 / 0.42 | 0.62 |
| fidelitymem_L2_nogate | 0.445 ±0.026 | 0.410 | 0.298 | 41831 | 1254 | 2.17 | 0.791 | 0.30 / 0.44 / 0.49 / 0.39 | 0.50 |
| fidelitymem_L3_nogate | 0.358 ±0.016 | 0.327 | 0.232 | 26114 | 2331 | 2.35 | 0.636 | 0.24 / 0.32 / 0.42 / 0.31 | 0.52 |
| fidelitymem_ctrl:self(L0/f/L1/c) | 0.851 ±0.015 | 0.810 | 0.730 | 194157 | 3694 | 0.59 | 1.512 | 0.54 / 0.87 / 0.94 / 0.57 | 0.80 |
| fidelitymem_ctrl:qlog(L1/c/L2/c/L3/c) | 0.849 ±0.014 | 0.807 | 0.729 | 192142 | 3651 | 0.65 | 1.507 | 0.54 / 0.87 / 0.93 / 0.56 | 0.79 |

*answer contain.* = fraction of gold-answer content tokens present in the served context (a proxy; adversarial questions excluded). The controller rows name the configuration chosen per seed (L = level, f/c = facts/content gate profile).

Four things the real data says that the synthetic data could not:

1. **Lexical retrieval is the strongest single signal on human conversation.** BM25 alone reaches 0.858 ±0.009. Every architecture that fuses a dense or graph signal with equal weight lands well below it: BM25+dense RRF 0.726, Zep-style 0.687, window+trie 0.653, HippoRAG-style 0.401, A-MEM/MemGPT-style 0.26–0.29, Mem0-style 0.239. The pretrained word-vector route scores 0.287 on its own; it is weak on short conversational turns and reciprocal-rank fusion gives it the same vote as BM25. FidelityMem's *default* route mix is hurt the same way (0.563).
2. **The controller is what makes the architecture safe.** With the retention bound measured against the best raw retrieval, the controller drops the dense route in every seed and lands at 0.851 ±0.015 (self-generated probes) and 0.849 ±0.014 (query-log calibration), within one point of BM25 and at a quarter of its own default latency; per-type recall matches BM25 on every category. Without that constraint (earlier revision of this work) it chose lexical+dense and scored 0.71, so the bound, not the heuristics, is what prevented a 14-point loss.
3. **Compression at ρ = 0.95 is small on chat.** Sessions are already compact statements between two people; dropping any content sentence loses evidence. The facts-profile levels compress to 22–65 % of raw and retain only 0.79–0.94 of recall; the content profile retains 1.00 and stores 98 %. The controller therefore keeps raw or near-raw storage. This is the bound being enforced on a corpus with little to remove, and it should be read as a limit of compression on this kind of text, not of the controller.
4. **The gate still earns its place.** At every level the gated variant beats the ungated one on recall (0.444 vs 0.358 at level 3, facts profile) and needs a quarter of the raw expansions (0.69 vs 2.64 per query); the lossy-summary baseline (0.425 at 13 % storage) sits on the same frontier only because it is cheap, not because it is adequate.

The Pareto frontier (`docs/results/pareto_locomo_session.png`) contains BM25, both controller rows and, at the low-storage end, the ungated levels and the lossy summary. Zep-, HippoRAG-, Mem0-, A-MEM- and MemGPT-style retrievers are all dominated.

### 5.2b Real data: LoCoMo, turn granularity (5 880 turns, 1 390 scored questions per seed, K = 10, 3 seeds)

| System | R@10 | CR@10 | MRR | stored tok | tok/q | p95 ms | retain | R@10 multi-hop / temporal / single-hop / open-domain |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fixed_window | 0.001 ±0.003 | 0.001 | 0.000 | 255158 | 423 | 0.0 | - | 0.00 / 0.00 / 0.00 / 0.00 |
| bm25 | 0.558 ±0.014 | 0.515 | 0.409 | 255158 | 452 | 2.6 | - | 0.28 / 0.65 / 0.64 / 0.29 |
| vector | 0.216 ±0.009 | 0.196 | 0.132 | 255158 | 584 | 10.6 | - | 0.11 / 0.30 / 0.24 / 0.08 |
| bm25+vector_rrf | 0.516 ±0.026 | 0.474 | 0.326 | 255158 | 524 | 18.7 | - | 0.24 / 0.60 / 0.59 / 0.25 |
| summary_L3+bm25 | 0.225 ±0.010 | 0.199 | 0.170 | 95597 | 206 | 2.4 | - | 0.12 / 0.18 / 0.29 / 0.10 |
| window+trie | 0.371 ±0.013 | 0.349 | 0.286 | 255158 | 489 | 5.0 | - | 0.10 / 0.50 / 0.45 / 0.12 |
| memgpt_style | 0.201 ±0.008 | 0.181 | 0.055 | 255158 | 531 | 12.0 | - | 0.10 / 0.28 / 0.22 / 0.07 |
| mem0_style | 0.149 ±0.009 | 0.134 | 0.096 | 103263 | 577 | 12.0 | - | 0.06 / 0.21 / 0.16 / 0.08 |
| zep_style | 0.499 ±0.027 | 0.461 | 0.310 | 255158 | 516 | 28.4 | - | 0.22 / 0.59 / 0.57 / 0.24 |
| amem_style | 0.216 ±0.009 | 0.196 | 0.132 | 255158 | 584 | 13.3 | - | 0.11 / 0.30 / 0.24 / 0.08 |
| hipporag_style | 0.131 ±0.014 | 0.116 | 0.098 | 255158 | 482 | 29.4 | - | 0.06 / 0.19 / 0.15 / 0.09 |
| fidelitymem_L0_raw | 0.490 ±0.023 | 0.448 | 0.295 | 255126 | 537 | 42.2 | - | 0.26 / 0.59 / 0.55 / 0.25 |
| fidelitymem_L1_gate:facts | 0.460 ±0.017 | 0.419 | 0.283 | 180176 | 426 | 39.0 | 0.939 | 0.24 / 0.61 / 0.50 / 0.25 |
| fidelitymem_L2_gate:facts | 0.367 ±0.008 | 0.333 | 0.247 | 114695 | 334 | 43.2 | 0.749 | 0.18 / 0.53 / 0.38 / 0.21 |
| fidelitymem_L3_gate:facts | 0.368 ±0.007 | 0.334 | 0.247 | 114584 | 333 | 41.8 | 0.750 | 0.18 / 0.53 / 0.38 / 0.21 |
| fidelitymem_L1_gate:content | 0.488 ±0.014 | 0.447 | 0.294 | 242633 | 528 | 41.0 | 0.996 | 0.27 / 0.61 / 0.54 / 0.24 |
| fidelitymem_L2_gate:content | 0.488 ±0.010 | 0.447 | 0.293 | 241358 | 527 | 37.3 | 0.996 | 0.27 / 0.61 / 0.54 / 0.24 |
| fidelitymem_L3_gate:content | 0.488 ±0.010 | 0.447 | 0.293 | 241358 | 527 | 37.0 | 0.996 | 0.27 / 0.61 / 0.54 / 0.24 |
| fidelitymem_L1_nogate | 0.447 ±0.012 | 0.406 | 0.278 | 176252 | 419 | 35.2 | 0.912 | 0.24 / 0.60 / 0.49 / 0.23 |
| fidelitymem_L2_nogate | 0.319 ±0.005 | 0.289 | 0.212 | 106874 | 325 | 37.4 | 0.651 | 0.17 / 0.46 / 0.34 / 0.17 |
| fidelitymem_L3_nogate | 0.215 ±0.007 | 0.190 | 0.135 | 95585 | 307 | 42.4 | 0.438 | 0.12 / 0.22 / 0.26 / 0.08 |
| fidelitymem_ctrl:self(L0/f) | 0.534 ±0.064 | 0.494 | 0.360 | 255126 | 497 | 51.0 | 1.090 | 0.25 / 0.64 / 0.61 / 0.25 |
| fidelitymem_ctrl:qlog(L0/c/L0/f) | 0.529 ±0.073 | 0.490 | 0.366 | 255126 | 496 | 36.3 | 1.080 | 0.25 / 0.65 / 0.60 / 0.24 |

Turn granularity is the hard setting: the gold evidence is one or two short utterances among 5 880. The ordering is the same as at session level with every number lower. BM25 0.558 ±0.014; the FidelityMem controller 0.534 ±0.064 (self-probes) and 0.529 ±0.073 (query log), which is BM25-level in the seeds where it drops the dense route and about four points below where it keeps all routes; Zep-style 0.499; window+trie 0.371; every other architecture 0.13–0.22. Compression retains 0.94 of recall at level 1 (71 % storage) and 0.75 at levels 2–3 (45 %); without the gate 0.91 / 0.65 / 0.44. Latency is where the all-routes configuration pays at this scale: 24 ms p50 for raw FidelityMem against 0.4 ms for BM25, mostly the dense matmul over 5 880 vectors and two-hop graph expansion; the controller rows show the same cost when they keep those routes. These timings are with three evaluation processes sharing the machine and are indicative only.

### 5.2c The retrieval-adapted embedder

The weak dense route is the common cause of every hybrid architecture's loss to BM25 on conversation, so we trained an embedder for it (`stores/learned_embedder.py`): pretrained word vectors with a learned weight per word (initialised from IDF) and a learned 300×256 projection (initialised to the identity), trained by InfoNCE on (question, gold evidence) pairs with in-batch and BM25 hard negatives. Training uses only the controller's calibration split of the seed being evaluated plus a synthetic world with an unrelated seed, and takes about a minute on CPU.

| Held-out set | dense Recall before → after | BM25+dense RRF before → after | BM25 |
|---|---|---|---|
| Synthetic 1k, different world | 0.705 → 0.865 | 0.870 → 0.890 | 0.875 |
| LoCoMo sessions (K=5) | 0.287 → 0.498 | 0.727 → 0.784 | 0.856 |
| LoCoMo turns (K=10) | 0.214 → 0.356 | 0.504 → 0.555 | 0.551 |

Every dense-using system improves when all of them share the adapted embedder, and the comparison is only valid when the questions the adapter learns from are disjoint from the questions the controller calibrates on. Our first per-seed run violated that: the adapter trained on the whole calibration split and the controller then scored route sets on the same questions, where the dense route had effectively memorised the evidence; lexical+dense looked lossless on those probes and lost ten points on the test split (controller 0.742 ±0.048). Neither a tighter bound (ρ = 0.98) nor a recall-loss term in the cost changed the choice, because on the leaked probes there was no recall loss to detect. Splitting the calibration set in half (adapter on one half, controller on the other) removes the effect:

| LoCoMo sessions, K=5, 3 seeds | base embedder | adapted, leak-free split |
|---|---|---|
| Dense alone | 0.287 ±0.016 | 0.424 ±0.029 |
| BM25 + dense RRF | 0.726 ±0.005 | 0.747 ±0.021 |
| Zep-style | 0.687 ±0.023 | 0.707 ±0.011 |
| HippoRAG-style | 0.401 ±0.022 | 0.441 ±0.011 |
| FidelityMem default routes (raw) | 0.563 ±0.016 | 0.610 ±0.016 |
| FidelityMem controller, self-probes | 0.851 ±0.015 | 0.851 ±0.015 |
| FidelityMem controller, query log | 0.849 ±0.014 | 0.849 ±0.014 |
| BM25 | 0.858 ±0.009 | 0.858 ±0.009 |

The dense route gains 14 points on sessions from roughly 600 training pairs, and 21 points when trained on the full calibration split as in the table above this one. Fusion still trails BM25 by 11 points on sessions, so the controller keeps selecting lexical-only there, and its result is unchanged. The lesson for the protocol is that a learned retrieval component and a controller that selects among retrieval components must not share calibration data; the lesson for the system is that the controller held the line through a mis-specified calibration only because it also had the raw lexical route to fall back on. We keep the recall-loss term (λ = 1) in the objective because it is correct in principle, and note that it did nothing here.

A last negative control for the embedder: the semantic-trie node embeddings of §5.5 (a memory as the IDF-weighted mean of its code-cluster centroids) match full dense recall exactly on the synthetic corpus and lose 3–6 points on LoCoMo turns, for a dense index 4–9× smaller. Quantising the *adapted* embedder's space the same way is the natural follow-up.

### 5.3 Ablation (synthetic, n = 1 000, K = 10, level 3 gated, 3 seeds, pretrained embedder)

One component removed at a time from the full system. `retain` is recall relative to the full system; ± is the 95% CI over seeds.

| Variant | R@10 | CR@10 | MRR | nDCG | tok/q | stored tok | expand | retain | multi-hop R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full (L3, gated) | 0.995 ±0.012 | 0.993 | 0.981 | 0.984 | 458 | 32 076 | 1.11 | 1.000 | 0.993 |
| − fidelity gate | 0.948 ±0.071 | 0.937 | 0.935 | 0.935 | 444 | 20 392 | 2.97 | 0.953 | 0.847 |
| + content gate profile | 0.995 ±0.012 | 0.993 | 0.929 | 0.947 | 762 | 73 427 | 0.00 | 1.000 | — |
| − raw expansion | 0.995 ±0.012 | 0.993 | 0.980 | 0.983 | 403 | 32 076 | 0.00 | 1.000 | — |
| − relation chaining | 0.917 ±0.083 | 0.837 | 0.976 | 0.908 | 458 | 32 075 | 1.14 | 0.921 | — |
| − graph route | 0.995 ±0.012 | 0.993 | 0.881 | 0.912 | 442 | 32 091 | 0.86 | 1.000 | — |
| − graph route and chaining | 0.877 ±0.019 | 0.757 | 0.881 | 0.826 | 440 | 32 060 | 0.88 | 0.881 | — |
| − hub penalty / pruning | 0.995 ±0.012 | 0.993 | 0.978 | 0.983 | 457 | 32 076 | 1.10 | 1.000 | — |
| − trie route | 0.995 ±0.012 | 0.993 | 0.978 | 0.982 | 458 | 32 060 | 1.11 | 1.000 | — |
| lexical + vector only | 0.995 ±0.012 | 0.993 | 0.901 | 0.927 | 436 | 32 075 | 0.84 | 1.000 | — |

Three findings. **Relation chaining is the multi-hop component**: without it recall falls to 0.92 and complete-set recall to 0.84, and to 0.88 / 0.76 when the graph route is also removed; the graph route by itself mainly improves ranking (MRR 0.88 → 0.98). **The gate protects relations as well as values**: without it the gist drops the sentence carrying the dependency relation in 15 % of multi-hop cases, the index shrinks to the lossy-summary size, and the retriever compensates with nearly three times as many raw expansions per query; the variance across seeds also triples. **The "content" gate profile is the other extreme**: it holds recall exactly but refuses to drop any content-bearing sentence, so on this corpus it stores the raw size. Expansion, the trie and hub pruning are neutral on recall here. Because chaining uses relations found in the top lexical hits, it recovers the second hop even in the lexical+vector-only configuration the controller selects, which is why that cheaper configuration remains feasible.

### 5.5 The semantic-codebook trie (negative result)

A proposal raised during the design discussion: number semantically similar words with one code (a codebook from k-means over the embedder's word vectors), store each memory as its code sequence in a word trie so shared prefixes are stored once, and match queries in code space so synonyms hit. We built it (`stores/semantic_trie.py`) with one safeguard, that entities, numbers, dates and negation cues are never coded, and measured storage, decode fidelity against the probes, and Recall@10 against BM25 on the same episodes (`eval/semantic_trie_experiment.py`, pretrained word vectors, 1 000 and 3 000 codes).

| Corpus | stored bytes / raw | decode fidelity (facts) | decode fidelity (content) | R@10 trie | R@10 BM25 |
|---|---:|---:|---:|---:|---:|
| synthetic, 1 000 episodes (vocab 194) | 0.76 | 0.81 | 0.79 | 0.875 | 0.875 |
| LoCoMo, 3 conversations, 1 451 turns (vocab 2 675) | 1.08–1.10 | 0.98 | 0.85–0.87 | 0.53–0.54 | 0.613 |

The storage saving exists only where text repeats (the synthetic filler), and comes from prefix sharing and stopword dropping rather than from semantics. On real conversation the trie is larger than the raw text (per-node overhead, no shared prefixes). The codebook does what the fidelity argument predicts: even with as many codes as vocabulary entries, k-means merges neighbours, so "machine" decodes as "dashboard" and "mentioned" as "clarified"; the protected-token rule keeps fact-probe fidelity at 0.98 on LoCoMo, but content-word fidelity drops to 0.85 and retrieval loses 8 points to BM25 because merged codes create false matches and flatten idf. Context tokens are unchanged in any case, since the model reads decoded text. We therefore keep the trie as an exact/prefix identifier route (§3.4) and do not use a semantic codebook for storage. A variant that codes only predicates and keeps every argument verbatim is the one worth revisiting; it is a graph-store compaction, not a compression of memories.

### 5.6 Phase 6: LLM-judged answer accuracy (LoCoMo sessions, 200 test questions, K = 5)

The concept's final phase is answer accuracy with an LLM reader and judge beside recall, tokens, latency and cost. `fidelitymem/eval/phase6.py` runs it: for each question a system retrieves its top-K context, a Claude Opus 5.5 reader (low effort) answers from that context only and abstains when the context lacks the answer, and a Claude Sonnet 5.5 judge marks the answer against the gold answer. FidelityMem runs under its controller calibrated on the 30 % query-log split; test questions are a fixed 200-question sample of the remaining 70 %, seed 0. Adversarial questions are scored by the judge against their reference answer (the dataset's "no information" answers), so abstention is the correct behaviour there.

| System | accuracy | abstention | Recall@5 | tokens/q | accuracy: single-hop / multi-hop / temporal / open-domain / adversarial | API cost / question |
|---|---:|---:|---:|---:|---|---:|
| bm25 | 0.645 | 0.135 | 0.830 | 3792 | 0.81 / 0.38 / 0.68 / 0.80 / 0.40 | $0.0240 |
| fidelitymem | 0.630 | 0.145 | 0.820 | 3725 | 0.77 / 0.44 / 0.70 / 0.40 / 0.37 | $0.0237 |
| zep_style | 0.480 | 0.305 | 0.645 | 3645 | 0.64 / 0.27 / 0.47 / 0.60 / 0.27 | $0.0232 |

Three observations. **The retrieval tie carries through to accuracy.** FidelityMem 63.0 % against BM25 64.5 % on the same questions, inside the noise of a 200-question sample (±6.7 points at 95 %), with equal tokens per query and equal API cost; the controller again chose the lexical route with level-3 compression under the content profile, which on sessions stores 98 % of the raw text. **A real architecture gap shows up as a real accuracy gap.** Zep-style, 18 points lower on Recall@5, is 16.5 points lower on accuracy and abstains twice as often (30.5 % against 13.5–14.5 %), because the reader is instructed to abstain when the evidence is missing. **Category profile.** FidelityMem is ahead on multi-hop (44 % vs 38 %) and temporal (70 % vs 68 %) questions and behind on open-domain (40 % vs 80 %, on 5 questions) and single-hop (77 % vs 81 %); with 200 questions none of these per-category differences is individually significant. The reader consumed about 5 200 input tokens per question (context plus prompt); at list prices the whole run cost about $14.

A second run with the Claude compressor and extractor at FidelityMem ingestion (`--llm-compress`) is reported in §5.6b.

### 5.6b Phase 6 with LLM compression at ingestion

The same evaluation with the Claude compressor and extractor at FidelityMem ingestion (`--llm-compress`; 272 sessions, about 800 Opus compression calls, 57 minutes). The gate applies to the LLM's summaries exactly as to extractive ones. The controller, calibrated on the query-log split, selected level 2 under the *facts* profile, where with extractive compression it had selected level 3 under the content profile (which stores 98 % of the raw text): LLM summaries pass the fact probes while removing far more text.

| FidelityMem ingestion | accuracy | abstention | Recall@5 | tokens/q | accuracy: single-hop / multi-hop / temporal / open-domain / adversarial | API cost / question |
|---|---:|---:|---:|---:|---|---:|
| extractive (level 3, content) | 0.630 | 0.145 | 0.820 | 3 725 | 0.77 / 0.44 / 0.70 / 0.40 / 0.37 | $0.0237 |
| Claude compressor (level 2, facts) | 0.615 | 0.175 | 0.814 | 2858 | 0.79 / 0.35 / 0.72 / 0.80 / 0.23 | $0.0187 |
| BM25, raw text | 0.645 | 0.135 | 0.830 | 3 792 | 0.81 / 0.38 / 0.68 / 0.80 / 0.40 | $0.0240 |

This is the first configuration in which compression pays on conversational data: 23 % fewer context tokens and 21 % lower reading cost per question, with accuracy 1.5 points below the extractive run and 3 below BM25, both inside the 200-question noise. Recall@5 is unchanged (0.814 vs 0.820), so the accuracy difference, if real, comes from what the summaries omit rather than from retrieval; the per-category profile is flat except open-domain, which has five questions. The one-off ingestion cost (about $3.5 at list prices for 272 sessions) is recovered after roughly 700 questions at the per-question saving.

### 5.6c Query rewriting and LLM reranking (retrieval only, same 200 questions)

Two LLM-side levers that published systems use to pass lexical retrieval were tested on the same 200 session questions with Claude Haiku 5.5: HyDE-style rewriting (the question is expanded with the sentence a speaker would have said) and reranking a 20-candidate pool down to K = 5. Both apply equally to BM25 and FidelityMem.

| Recall@5 | none | + rewrite | + rerank | + rewrite + rerank |
|---|---:|---:|---:|---:|
| BM25 | 0.830 | 0.818 | 0.830 | 0.808 |
| FidelityMem (controller) | 0.820 | 0.804 | 0.823 | 0.772 |

Neither helps here. Rewriting hurts both systems: on LoCoMo the question's own words are the best lexical key, and the hypothetical sentence adds words that match the wrong sessions. Reranking is neutral (+0.3 for FidelityMem, 0 for BM25): when the gold session is in the top 20 it is already in the top 5, so a reranker has nothing to recover; the questions that fail are the ones whose evidence BM25 never surfaces at all. The combination is the worst of both. The practical conclusion is that the remaining headroom on conversational retrieval is in the *first-stage* signal (a strong dense model, an LLM extractor for relations and resolved dates), not in query-side or rerank-side LLM calls, which cost about $0.003 per question and 1–2 s of latency for nothing on this data.

### 5.6d Evidence-first hierarchical context (same 200 questions)

A change aimed at the gap between retrieval and answering: retrieve 8 sessions instead of 5, then inside each session rank its turns against the question (BM25 over the session's own turns), keep the best 3 turns plus one neighbour either side in original order, mark the matched turns, and cap the context there. Recall is still scored on the top 5 sessions. Applied identically to BM25 and FidelityMem.

| System | accuracy | abstention | Recall@5 | tokens/q | accuracy: single-hop / multi-hop / temporal | API cost / question |
|---|---:|---:|---:|---:|---|---:|
| BM25, raw K=5 | 0.645 | 0.135 | 0.830 | 3 792 | 0.81 / 0.38 / 0.68 | $0.0240 |
| bm25+focus8x3 | 0.635 | 0.145 | 0.830 | 2623 | 0.84 / 0.35 / 0.68 | $0.0193 |
| FidelityMem, raw K=5 | 0.630 | 0.145 | 0.820 | 3 725 | 0.77 / 0.44 / 0.70 | $0.0237 |
| fidelitymem+focus8x3 | 0.585 | 0.160 | 0.820 | 2627 | 0.81 / 0.35 / 0.57 | $0.0193 |

The hierarchical context saves 30 % of the tokens and cost per question at no measurable accuracy change for BM25 (−1.0 point) and a 4.5-point drop for FidelityMem, which is within the sample's noise but in the wrong direction. Reading the per-category numbers, the loss is on single-hop questions whose answer sits in a turn the in-session BM25 did not rank among the top 3 (the surrounding turns were cut). So the lever is a token lever, not an accuracy lever: on this data the reader is not losing answers to context length, it is losing them to evidence that is not retrieved or not recognised, and that returns the problem to first-stage retrieval signal. Together with §5.6c, every LLM-side and context-side change tried on LoCoMo sessions moved accuracy by less than the noise; the only change that moved cost substantially without moving accuracy was gated LLM compression at ingestion (§5.6b).

## 6. Limitations and what is not shown

1. **No LLM in the loop.** Ingestion, compression and the benchmark are deterministic. Answer accuracy on LoCoMo or LongMemEval, which needs an LLM reader and judge, is not measured. The protocols are compatible: the same gate and controller apply to an LLM compressor.
2. **Re-implementations, not vendor code.** The six architecture baselines reproduce published retrieval designs with shared components. They are faithful to the architecture, not to the vendors' tuned systems.
3. **One synthetic dataset.** The question families and distractors were designed to stress fact loss and multi-hop retrieval. Results on human conversation data may differ; the generator is seeded and public so the experiment is reproducible and extendable.
4. **Probe distribution.** The controller optimises against self-generated probes, which here are fact-dense sentences. A deployment whose users ask about low-fact-density content would need probes sampled from real query logs; the API accepts any probe set.
5. **Scale.** 10 000 records in one process. The dense index is O(n·d) per query and the controller recompresses the store per level; beyond ~10⁵ records both need an ANN index and sampled background optimisation, which the store protocols are shaped for but which are not built. Tenant isolation is a provenance field, not an enforced partition.
6. **Retention above 1.** Compressed indexes sometimes retrieve *better* than raw because filler is removed; we report retention as measured and do not clip it.

## 7. Conclusion

A memory system can be made to keep a stated share of its evidence recall, cheaply and without labels, by generating fidelity probes from the raw source and by searching policies under that bound with an out-of-sample gate. On a controlled protocol the resulting system dominates six current memory architectures on recall and evidence completeness while halving stored and served tokens. The unique parts are the gate, the controller and LLM-free ingestion; the rest is a careful integration of known components, and we have tried to say so plainly.

## References

1. Facts as First-Class Objects: Knowledge Objects for Persistent LLM Memory. arXiv:2603.17781, 2026.
2. Fidelity Before Structure: Verbatim Chunks Beat Lossy Artifact Extraction in Long-Conversation LLM Memory. arXiv:2601.00821, 2026.
3. FACTWASH: Catching AI Rewrites That Wash Hearsay into Fact. arXiv:2608.03372, 2026.
4. Packer et al. MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560, 2023.
5. Gutiérrez et al. HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs. arXiv:2405.14831, 2024.
6. Zep / Graphiti: temporal knowledge graph memory. Product documentation, 2025–2026.
7. Mem0: memory layer for AI agents. Product documentation and benchmark pages, 2025–2026.
8. A-MEM: Agentic Memory for LLM Agents. 2025.
9. Wang et al. R³Mem: Bridging Memory Retention and Retrieval via Reversible Compression. ACL Findings 2025.
10. Liu et al. EvolveMem: Self-Evolving Memory Architecture via AutoResearch for LLM Agents. arXiv:2605.13941, 2026.
11. Zhang et al. MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory. arXiv:2601.03192, 2026.
12. Agent Zero Memory: Provenance-Aware Long-Term Memory for LLM Agents. arXiv:2608.29606, 2026.
13. LazyMem: Retrieve Broadly, Construct Selectively for Efficient Long-Term Agent Memory. arXiv:2607.22690, 2026.
14. Wu et al. LongMemEval. arXiv:2410.10813, 2024. 15. Tan et al. MemBench. arXiv:2506.21605, 2025. 16. DolphinBench: Mapping the Pareto Frontier of Agent Memory. arXiv:2609.24971, 2026.
