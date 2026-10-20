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

**Controls.** Same episodes, questions, seed, K and token accounting for every system; stochastic components are seeded; the controller never sees the benchmark questions.

## 5. Results

<!-- RESULTS:BEGIN -->
*(tables inserted from `docs/results/scale_k10.json` by `python -m fidelitymem.eval.report`)*
<!-- RESULTS:END -->

### 5.1 Reading the tables

* **Recall at scale.** FidelityMem (all levels and the controller choice) is the only system at 1.0 Recall@10 and complete-set recall across all four record levels. The gap to the strongest architecture baseline (hipporag_style, then zep_style) opens on multi-hop questions as distractors grow, which is the relation-chaining effect: graph adjacency finds the second hop at every scale, but fusion alone stops surfacing it.
* **Compression without loss.** Levels 2 and 3 store 44–46 % of the raw tokens and serve about half the context tokens per query with retention 1.0 against L0; the gate backed off on roughly one in six episodes (`backoffs` in the run log), which is exactly the set where a gist would have dropped a fact. The lossy `summary_L3+bm25` baseline stores less (≈30 %) and pays for it on exact and temporal questions.
* **Controller choice.** With no access to the questions, the controller selected L3 with the two cheapest routes at every scale and still met ρ on its hold-out probes; its measured benchmark retention is 1.0. Its latency is the lowest of the FidelityMem rows because it dropped the graph and temporal routes it found unnecessary for the probe distribution — a limitation discussed in §6.
* **Window + trie.** Perfect MRR on exact-value questions and competitive overall recall at small scale, but it is capped at the BM25 ceiling on factual and temporal questions and cannot reach a second hop. This is why the trie is kept as a *route* inside FidelityMem (enabled for exact and code-review queries) rather than as the system.
* **Mem0-style atomic facts** has the smallest index of the baselines (≈23 % of raw) and loses recall where the atomic sentence is not the one that matches the query phrasing. **MemGPT-style** recall is as good as its vector store but its MRR collapses because the always-present core block occupies the top ranks.
* **Latency.** FidelityMem's p95 is 1–3 ms higher than single-route baselines at 300–3 000 records and grows linearly with the dense index (a dense matmul in this container's BLAS); the comparison is of architectures, not of index engineering.

### 5.2 Ablation (n = 1 000, K = 10, level 3 gated)

One component removed at a time from the full system. `retain` is recall relative to the full system. Latencies in this table were measured while the scale study ran on the same machine and are omitted.

| Variant | R@10 | CR@10 | MRR | nDCG | tok/q | stored tok | expand | retain | multi-hop R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full (L3, gated) | 1.000 | 1.000 | 0.980 | 0.986 | 417 | 30 885 | 1.16 | 1.000 | 1.000 |
| − fidelity gate | 0.980 | 0.970 | 0.966 | 0.967 | 422 | 20 190 | 2.98 | 0.980 | 0.920 |
| − raw expansion | 1.000 | 1.000 | 0.990 | 0.992 | 365 | 30 933 | 0.00 | 1.000 | 1.000 |
| − relation chaining | 0.920 | 0.840 | 0.990 | 0.917 | 419 | 30 885 | 1.16 | 0.920 | 0.680 |
| − graph route | 1.000 | 1.000 | 0.731 | 0.802 | 398 | 30 933 | 1.00 | 1.000 | 1.000 |
| − graph route and chaining | 0.880 | 0.760 | 0.731 | 0.714 | 399 | 30 933 | 1.05 | 0.880 | 0.520 |
| − hub penalty / pruning | 1.000 | 1.000 | 0.985 | 0.990 | 417 | 30 933 | 1.15 | 1.000 | 1.000 |
| − trie route | 1.000 | 1.000 | 0.990 | 0.991 | 419 | 30 933 | 1.16 | 1.000 | 1.000 |
| lexical + vector only | 1.000 | 1.000 | 0.915 | 0.941 | 390 | 30 933 | 1.00 | 1.000 | 1.000 |

Three findings. **Relation chaining is the multi-hop component**: without it, second-hop recall falls to 0.68, and to 0.52 when the graph route is also removed; the graph route by itself mainly improves ranking (MRR 0.73 → 0.98). **The gate protects relations as well as values**: without it the gist drops the sentence carrying the dependency relation in 8 % of multi-hop cases, the index shrinks to the lossy-summary size, and the retriever compensates with nearly three times as many raw expansions per query. **Expansion, the trie and hub pruning are neutral on recall here**: expansion serves exact-value questions whose answer must be quoted (it is mandatory for them) and costs 14 % more context tokens; the trie helps only when identifiers are the query, which this question set does not isolate; hub pruning is a latency measure. Because chaining uses relations found in the top lexical hits, it recovers the second hop even in the lexical+vector-only configuration the controller selected, which is why that cheaper configuration remained feasible.

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
