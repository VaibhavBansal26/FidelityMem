"""Hybrid retriever: parallel route search, fusion, rerank, sufficiency check, raw expansion."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

from ..stores import ArchivalStore, BM25Index, GraphStore, TrieIndex, VectorIndex
from ..text import content_tokens, estimate_tokens, extract_dates, extract_entities, extract_numbers, overlap
from ..types import Memory, MemoryType, QueryType, RetrievedMemory, now
from .fusion import final_score, rrf
from .planner import RetrievalPlan


@dataclass
class RetrievalStats:
    latency_ms: float = 0.0
    candidates: int = 0
    expansions: int = 0
    tokens: int = 0
    route_hits: dict[str, int] = field(default_factory=dict)
    conflicts: int = 0
    unresolved_conflicts: int = 0


class HybridRetriever:
    def __init__(self, memories: dict[str, Memory], archive: ArchivalStore, lexical: BM25Index, vector: VectorIndex, graph: GraphStore, trie: TrieIndex | None = None):
        self.trie = trie
        self.semantic_trie = None  # optional SemanticTrie (stores.semantic_trie), attached by the facade
        self.memories = memories
        self.archive = archive
        self.lexical = lexical
        self.vector = vector
        self.graph = graph
        # maintained by the facade on index / supersede / forget
        self.active: set[str] = set()
        self.enable_chaining = True  # ablation switch
        self.by_type: dict[MemoryType, set[str]] = {t: set() for t in MemoryType}

    def track(self, m: Memory) -> None:
        self.active.add(m.id)
        for t in m.types:
            self.by_type[t].add(m.id)

    def untrack(self, m: Memory) -> None:
        self.active.discard(m.id)
        for t in m.types:
            self.by_type[t].discard(m.id)

    # ---- routes ------------------------------------------------------
    def _active_ids(self) -> set[str]:
        return self.active

    def _type_ids(self, allowed: set[str], t: MemoryType) -> set[str]:
        return self.by_type[t] if allowed is self.active else self.by_type[t] & allowed

    def _route(self, route: str, query: str, plan: RetrievalPlan, allowed: set[str]) -> list[tuple[str, float]]:
        k = plan.per_route_k
        if route == "lexical":
            return self.lexical.search(query, k, None if allowed is self.active else allowed)
        if route == "vector":
            return self.vector.search(query, k, None if allowed is self.active else allowed)
        if route == "trie":
            return self.trie.search(query, k, allowed) if self.trie is not None else []
        if route == "semantic_trie":
            if self.semantic_trie is None:
                return []
            return [(i, sc) for i, sc in self.semantic_trie.search(query, k * 4) if i in allowed][:k]
        if route == "graph":
            known = self.graph.known_entities()
            seeds: dict[str, float] = {}
            for e in extract_entities(query) + [t.strip("?.,") for t in query.lower().split() if t.strip("?.,") in known]:
                seeds[e.lower()] = 1.0
            # iterative retrieval ("retrieve then expand"): follow explicit relations out of
            # the top lexical hits whose subject is a query entity, so a bridge entity the
            # query never names becomes a seed. Co-mentions are *not* used here: filler
            # entities in a hit would otherwise flood the route.
            for mid, _ in self.lexical.search(query, 3, allowed):
                for r in self.memories[mid].relations:
                    if r.subject.lower() in seeds and not r.negated:
                        seeds.setdefault(r.object.lower(), 0.9)
            hits = [(i, s) for i, s in self.graph.expand(seeds, plan.graph_hops, k) if i in allowed]
            # bridged second pass: re-issue the query anchored on each bridge entity so the
            # second hop is found by lexical/vector evidence, not only by graph adjacency.
            if plan.graph_hops >= 2:
                bridges = [e for e, w in seeds.items() if w < 1.0][:3]
                extra: dict[str, float] = {}
                for b_ent in bridges:
                    sub = f"{b_ent} {query}"
                    for rank, (i, _) in enumerate(self.lexical.search(sub, 5, allowed)):
                        extra[i] = max(extra.get(i, 0.0), 0.8 / (1 + rank))
                    for rank, (i, _) in enumerate(self.vector.search(sub, 5, allowed)):
                        extra[i] = max(extra.get(i, 0.0), 0.6 / (1 + rank))
                merged = dict(hits)
                for i, sc in extra.items():
                    merged[i] = max(merged.get(i, 0.0), sc)
                hits = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:k]
            return hits
        if route == "temporal":
            # episodic memories matching the query; recency ordering only when the query
            # asks for recent/latest state, otherwise "when did X happen" is answered by
            # the best lexical match however old it is
            eps = self._type_ids(allowed, MemoryType.EPISODIC)
            hits = self.lexical.search(query, k, eps)
            if plan.wants_recent:
                hits = [(i, 1.0) for i in sorted((i for i, _ in hits), key=lambda i: self.memories[i].valid_from, reverse=True)]
            return hits
        if route == "procedural":
            procs = {i for i, m in self.memories.items() if i in allowed and MemoryType.PROCEDURAL in m.types}
            return self.lexical.search(query, k, procs)
        return []

    # ---- main --------------------------------------------------------
    def retrieve(self, query: str, plan: RetrievalPlan, at: datetime | None = None) -> tuple[list[RetrievedMemory], RetrievalStats]:
        t0 = time.perf_counter()
        at = at or now()
        allowed = self._active_ids()
        route_results = {r: self._route(r, query, plan, allowed) for r in plan.routes}
        stats = RetrievalStats(route_hits={r: len(v) for r, v in route_results.items()})
        fused = rrf(route_results, plan.weights)
        stats.candidates = len(fused)
        scored = []
        fused_max = max(fused.values(), default=1.0)
        for mid, f in fused.items():
            m = self.memories[mid]
            s = final_score(m, f, plan.weights, at, plan.recency_halflife_days, plan.type_boost, fused_max)
            scored.append(RetrievedMemory(memory=m, score=s, sources={r: dict(v).get(mid, 0.0) for r, v in route_results.items() if mid in dict(v)}))
        scored.sort(key=lambda r: r.score, reverse=True)
        if plan.graph_hops >= 2 and self.enable_chaining:
            scored = self._chain_relations(query, scored, allowed, plan.top_k)
        top = scored[: plan.top_k]
        # sufficiency + raw expansion
        conflict_ids = self._conflicts(top)
        stats.conflicts = len(conflict_ids)
        stats.expansions = self._expand(query, plan, top, bool(conflict_ids), conflict_ids)
        stats.unresolved_conflicts = sum(1 for r in top if r.memory.id in conflict_ids and r.memory.compression_level > 0 and not r.expanded)
        # token budget trim
        kept, used = [], 0
        for r in top:
            t = estimate_tokens(r.context_text)
            if used + t > plan.memory_token_budget:
                if r.expanded:  # fall back to compressed form before dropping
                    r.expanded = False
                    t = estimate_tokens(r.context_text)
                    if used + t > plan.memory_token_budget:
                        continue
                else:
                    continue
            kept.append(r)
            used += t
            r.memory.retrieval_count += 1
        stats.tokens = used
        stats.latency_ms = (time.perf_counter() - t0) * 1000
        return kept, stats

    def _chain_relations(self, query: str, scored: list[RetrievedMemory], allowed: set[str], top_k: int, max_inject: int = 3) -> list[RetrievedMemory]:
        """Deterministic relation following for multi-hop questions.

        If a top result states `A --p--> B` and A is named in the query, the memories
        in which B is itself the subject of a relation are the second hop. They are
        promoted to just below the result that led to them (injected if absent), so
        fusion noise cannot bury a hop the graph has already found."""
        q_ents = {e.lower() for e in extract_entities(query)} | {t.strip("?.,") for t in query.lower().split()}
        by_id = {r.memory.id: r for r in scored}
        promoted: list[tuple[int, RetrievedMemory]] = []  # (anchor index, result)
        used: set[str] = set()
        for idx, r in enumerate(scored[:top_k]):
            if len(promoted) >= max_inject:
                break
            for rel in r.memory.relations:
                if rel.negated or rel.subject.lower() not in q_ents:
                    continue
                for mid in self.graph._subject_mems.get(rel.object.lower(), ()):
                    if mid not in allowed or mid in used or len(promoted) >= max_inject:
                        continue
                    hop = by_id.get(mid) or RetrievedMemory(memory=self.memories[mid], score=0.0, sources={})
                    hop.sources["chain"] = 1.0
                    hop.score = r.score * 0.98
                    promoted.append((idx, hop))
                    used.add(mid)
        if not promoted:
            return scored
        out: list[RetrievedMemory] = []
        for idx, r in enumerate(scored):
            if r.memory.id in used:
                continue
            out.append(r)
            for a_idx, hop in promoted:
                if a_idx == idx:
                    out.append(hop)
        return out

    # ---- expansion ---------------------------------------------------
    def _conflicts(self, results: list[RetrievedMemory]) -> set[str]:
        """Ids of retrieved memories that contradict another retrieved memory."""
        from ..validation import _relation_conflict

        ids: set[str] = set()
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                a, b = results[i].memory, results[j].memory
                hit = False
                if a.entities and set(a.entities) == set(b.entities) and a.numbers and b.numbers and set(a.numbers) != set(b.numbers):
                    hit = True
                if not hit:
                    hit = any(_relation_conflict(ra, rb) for ra in a.relations for rb in b.relations)
                if hit:
                    ids.add(a.id)
                    ids.add(b.id)
        return ids

    def _needs_expansion(self, query: str, r: RetrievedMemory, plan: RetrievalPlan, conflict: bool, conflict_ids: set[str] | None = None) -> tuple[bool, str]:
        m = r.memory
        if m.compression_level == 0:
            return False, ""
        if plan.expand_always:
            return True, "exact_task"
        if m.confidence < 0.5:
            return True, "low_confidence"
        if m.fidelity_score < 0.9:
            return True, "low_fidelity"
        if conflict and (conflict_ids is None or m.id in conflict_ids):
            return True, "conflict"
        q_nums, q_dates = set(extract_numbers(query)), set(extract_dates(query))
        if (q_nums or q_dates) and plan.query_type in (QueryType.EXACT, QueryType.TEMPORAL):
            return True, "query_needs_values"
        # evidence looks incomplete: query content words barely covered by compressed text
        qt = set(content_tokens(query))
        if qt and overlap(qt, set(content_tokens(m.compressed_text))) < 0.2 and overlap(qt, set(content_tokens(self.archive.get(m.episode_id).text if self.archive.get(m.episode_id) else ""))) >= 0.4:
            return True, "incomplete_evidence"
        if plan.query_type == QueryType.MULTI_HOP and not m.relations:
            return True, "multi_hop_relations"
        return False, ""

    def _expand(self, query: str, plan: RetrievalPlan, results: list[RetrievedMemory], conflict: bool, conflict_ids: set[str] | None = None) -> int:
        budget = plan.expand_budget
        n = 0
        for r in results:
            if budget <= 0:
                break
            need, _ = self._needs_expansion(query, r, plan, conflict, conflict_ids)
            if need:
                ep = self.archive.get(r.memory.episode_id)
                if ep:
                    r.expanded, r.raw_text = True, ep.text
                    budget -= 1
                    n += 1
        return n
