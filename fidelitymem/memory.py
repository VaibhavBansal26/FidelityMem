"""FidelityMem facade: the single object an agent talks to."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .compression import Compressor, ProbeSet, build_probes, compress_with_gate
from .controller import Config, Controller, Measurement, ProbeQuery, make_probe_queries
from .extraction import Candidate, Extractor, RuleExtractor
from .policy import MemoryPolicy
from .retrieval.planner import plan_for
from .retrieval.retriever import HybridRetriever, RetrievalStats
from .stores import ArchivalStore, BM25Index, Embedder, GraphStore, TrieIndex, VectorIndex
from .text import estimate_tokens
from .types import ContextBundle, Episode, Memory, MemoryType, Message, Provenance, ValidationReport, now
from .validation import stage1_source_consistency, stage2_relevance_utility
from .window import AdaptiveWindow, Boundary


@dataclass
class Telemetry:
    admitted: int = 0
    rejected: int = 0
    superseded: int = 0
    compression_backoffs: int = 0
    queries: int = 0
    expansions: int = 0
    tokens_served: int = 0
    latency_ms: list[float] = field(default_factory=list)
    contradictions_detected: int = 0

    def summary(self) -> dict:
        lat = sorted(self.latency_ms)
        p = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))] if lat else 0.0
        return {
            "admitted": self.admitted,
            "rejected": self.rejected,
            "superseded": self.superseded,
            "compression_backoffs": self.compression_backoffs,
            "queries": self.queries,
            "expansions": self.expansions,
            "avg_tokens": self.tokens_served / max(1, self.queries),
            "p50_ms": p(0.5),
            "p95_ms": p(0.95),
            "contradictions": self.contradictions_detected,
        }


class FidelityMem:
    def __init__(
        self,
        policy: MemoryPolicy | None = None,
        embedder: Embedder | None = None,
        compressor: Compressor | None = None,
        extractor: Extractor | None = None,
        archive_path: str | None = None,
    ):
        self.policy = policy or MemoryPolicy()
        self.archive = ArchivalStore(archive_path)
        self.memories: dict[str, Memory] = {}
        self.probes: dict[str, ProbeSet] = {}
        self.lexical = BM25Index()
        self.vector = VectorIndex(embedder)  # None -> process-wide default (stores.set_default_embedder)
        self.graph = GraphStore()
        self.trie = TrieIndex()
        self.compressor = compressor
        self.extractor = extractor or RuleExtractor()
        self.controller = Controller(self.policy)
        self.retriever = HybridRetriever(self.memories, self.archive, self.lexical, self.vector, self.graph, self.trie)
        self.window = AdaptiveWindow(
            min_tokens=self.policy.working_window_min_tokens,
            max_tokens=self.policy.working_window_max_tokens,
            on_boundary=self._on_boundary,
        )
        self.telemetry = Telemetry()
        self.reports: list[ValidationReport] = []
        self._level_cache: dict[tuple[str, int, str], tuple[str, int, float]] = {}
        self._raw_text: dict[str, str] = {}
        self.gate_enabled = True  # ablation switch: False compresses without the fidelity gate
        self._raw_recall_cache: dict[tuple, tuple] = {}

    # ---- ingestion -----------------------------------------------------
    def observe(self, msg: Message) -> None:
        """Add an interaction to the working window. Boundaries trigger extraction."""
        self.window.add(msg)

    def flush(self) -> None:
        """Force a boundary on the whole window (e.g. session end)."""
        msgs = self.window.messages
        if msgs:
            self._on_boundary(Boundary("flush", msgs))
            self.window._messages.clear()

    def _on_boundary(self, b: Boundary) -> None:
        for cand in self.extractor.extract(b.messages):
            self.admit(cand, source_type="message")

    def remember(self, text: str, *, trusted: bool = True, importance: float = 0.6, source_type: str = "user", authority: float = 0.6, timestamp: Optional[datetime] = None, types: list[MemoryType] | None = None) -> Optional[Memory]:
        """Directly store a piece of evidence (bypasses the window)."""
        msg = Message(role=source_type, content=text, trusted=trusted, importance=importance, timestamp=timestamp or now())
        cands = RuleExtractor().extract([msg]) or [Candidate(text=text, types=types or [MemoryType.SEMANTIC], message_ids=[msg.id], trusted=trusted, importance=importance)]
        c = cands[0]
        if types:
            c.types = types
        return self.admit(c, source_type=source_type, authority=authority, timestamp=timestamp)

    def _neighbours(self, cand: Candidate, limit: int = 64) -> list[Memory]:
        """Memories worth comparing a candidate against: those sharing an entity, plus the
        candidate's own top lexical hits. Keeps admission sub-linear in store size."""
        ids: dict[str, None] = {}
        for e in cand.entities:
            for mid in self.graph._entity_mems.get(e.lower(), ()):
                ids.setdefault(mid)
        # a short, rare-term query (identifiers and values) is both cheaper and more
        # selective than the full candidate text
        probe = " ".join(cand.entities + cand.numbers + cand.dates) or cand.text
        for mid, _ in self.lexical.search(probe, 16):
            ids.setdefault(mid)
        out = [self.memories[i] for i in list(ids)[:limit] if i in self.memories]
        return out

    def admit(self, cand: Candidate, *, source_type: str = "message", authority: float = 0.5, timestamp: Optional[datetime] = None) -> Optional[Memory]:
        existing = self._neighbours(cand)
        s1 = stage1_source_consistency(cand, existing, authority)
        s2 = stage2_relevance_utility(cand, existing)
        self.telemetry.contradictions_detected += len(s1.contradictions)
        if not s2.accept:
            self.telemetry.rejected += 1
            self.reports.append(ValidationReport(accepted=False, issues=s1.issues + s2.issues, confidence=s1.confidence))
            return None
        ts = timestamp or now()
        prov = Provenance(source_id=cand.message_ids[0] if cand.message_ids else "direct", source_type=source_type, tenant=self.policy.tenant, namespace=self.policy.namespace, timestamp=ts, trusted=cand.trusted, authority=authority)
        ep = self.archive.put(Episode(text=cand.text, provenance=prov, message_ids=cand.message_ids))
        profile = self.controller.current.gate_profile
        probes = build_probes(cand.text, profile)
        level = self.controller.current.compression_level
        cr = compress_with_gate(cand.text, level, self.policy.minimum_recall_retention if self.gate_enabled else 0.0, self.compressor, probes)
        self._raw_recall_cache.clear()
        if cr.level < min(level, 3):
            self.telemetry.compression_backoffs += 1
        mem = Memory(
            episode_id=ep.id, types=s1.types, compressed_text=cr.text, compression_level=cr.level,
            entities=cand.entities, numbers=cand.numbers, dates=cand.dates, negations=cand.negations, relations=cand.relations,
            provenance=prov, confidence=s1.confidence, valid_from=ts, fidelity_score=cr.fidelity,
        )
        self._index(mem, probes)
        self._raw_text[mem.id] = cand.text
        self._level_cache[(mem.id, level, profile)] = (cr.text, cr.level, cr.fidelity)
        for old_id in s1.supersedes:
            self._supersede(old_id, mem.id, ts)
        self.telemetry.admitted += 1
        self.reports.append(ValidationReport(accepted=True, issues=s1.issues + s2.issues, confidence=s1.confidence, fidelity_score=cr.fidelity, compression_level=cr.level))
        return mem

    def _index(self, mem: Memory, probes: ProbeSet) -> None:
        self.memories[mem.id] = mem
        self.probes[mem.id] = probes
        self.retriever.track(mem)
        self.lexical.add(mem.id, mem.compressed_text)
        self.vector.add(mem.id, mem.compressed_text)
        self.trie.add(mem.id, mem.compressed_text)
        self.graph.add_memory(mem.id, mem.entities, mem.relations)

    def _supersede(self, old_id: str, new_id: str, at: datetime) -> None:
        old = self.memories.get(old_id)
        if old and old.active:
            old.valid_to = at
            old.superseded_by = new_id
            self.retriever.untrack(old)
            self.lexical.remove(old_id)
            self.vector.remove(old_id)
            self.trie.remove(old_id)
            self.graph.remove_memory(old_id)
            self.telemetry.superseded += 1

    def forget(self, memory_id: str) -> bool:
        """User-requested forgetting: remove derived memory and raw episode text."""
        m = self.memories.pop(memory_id, None)
        if not m:
            return False
        self.retriever.untrack(m)
        self.lexical.remove(memory_id)
        self.vector.remove(memory_id)
        self.trie.remove(memory_id)
        self.graph.remove_memory(memory_id)
        self.probes.pop(memory_id, None)
        self.archive._items.pop(m.episode_id, None)
        return True

    # ---- recompression (policy change) ------------------------------
    def recompress_all(self, level: int, profile: str | None = None) -> int:
        """Rebuild compressed representations from raw evidence at a new level / gate profile."""
        profile = profile or self.controller.current.gate_profile
        n = 0
        for mem in list(self.memories.values()):
            key = (mem.id, level, profile)
            if key not in self._level_cache:
                ep = self.archive.get(mem.episode_id)
                if not ep:
                    continue
                probes = build_probes(ep.text, profile)
                cr = compress_with_gate(ep.text, level, self.policy.minimum_recall_retention if self.gate_enabled else 0.0, self.compressor, probes, profile)
                self._level_cache[key] = (cr.text, cr.level, cr.fidelity)
            text, lvl, fid = self._level_cache[key]
            if text != mem.compressed_text:
                mem.compressed_text, mem.compression_level, mem.fidelity_score = text, lvl, fid
                if mem.active:
                    self.lexical.add(mem.id, text)
                    self.vector.add(mem.id, text)
                    self.trie.add(mem.id, text)
                n += 1
        return n

    # ---- retrieval -----------------------------------------------------
    def build_context(self, query: str, at: datetime | None = None, overrides: dict | None = None) -> ContextBundle:
        ov = dict(self.controller.current.as_overrides())
        ov.update(overrides or {})
        plan = plan_for(query, self.policy, ov)
        retrieved, stats = self.retriever.retrieve(query, plan, at)
        remaining = max(0, self.policy.max_context_tokens - stats.tokens)
        working = self.window.select(query, min(remaining, self.policy.working_window_max_tokens))
        used = stats.tokens + sum(estimate_tokens(m.content) for m in working)
        self.telemetry.queries += 1
        self.telemetry.expansions += stats.expansions
        self.telemetry.tokens_served += used
        self.telemetry.latency_ms.append(stats.latency_ms)
        return ContextBundle(query=query, query_type=plan.query_type, working=working, retrieved=retrieved, tokens_used=used, token_budget=self.policy.max_context_tokens, expansions=stats.expansions, plan={"routes": plan.routes, "top_k": plan.top_k, "latency_ms": stats.latency_ms, "route_hits": stats.route_hits, "conflicts": stats.conflicts})

    def search(self, query: str, k: int | None = None, at: datetime | None = None):
        ov = dict(self.controller.current.as_overrides())
        if k:
            ov["top_k"] = k
            ov["per_route_k"] = k * 3
        plan = plan_for(query, self.policy, ov)
        return self.retriever.retrieve(query, plan, at)

    # ---- feedback ------------------------------------------------------
    def feedback(self, memory_ids: list[str], useful: bool, weight: float = 1.0) -> None:
        for mid in memory_ids:
            m = self.memories.get(mid)
            if m:
                m.utility += weight if useful else -0.5 * weight
                if useful:
                    m.used_count += 1

    # ---- self-optimisation --------------------------------------------
    def evaluate_config(self, cfg: Config, probes: list[ProbeQuery]) -> Measurement:
        """Measure a configuration on probe queries: retention vs a level-0 index."""
        import time

        from .eval.metrics import recall_at_k

        original = {mid: (m.compressed_text, m.compression_level, m.fidelity_score) for mid, m in self.memories.items()}
        raw_tokens = sum(estimate_tokens(self.archive.get(m.episode_id).text) for m in self.memories.values()) or 1

        def run(level: int, routes: tuple[str, ...] | None = None) -> tuple[float, float, float, float, float]:
            self.recompress_all(level, cfg.gate_profile)
            ov = cfg.as_overrides()
            if routes is not None:
                ov["routes"] = list(routes)
            recalls, toks, lat, prec, conflicts = [], [], [], [], 0
            for p in probes:
                plan = plan_for(p.query, self.policy, ov)
                t0 = time.perf_counter()
                res, st = self.retriever.retrieve(p.query, plan)
                lat.append((time.perf_counter() - t0) * 1000)
                got = [r.memory.episode_id for r in res]
                recalls.append(recall_at_k(got, p.gold_episode_ids, cfg.top_k))
                toks.append(st.tokens)
                prec.append(len(set(got) & set(p.gold_episode_ids)) / max(1, len(got)))
                # a contradiction counts only while a party to it is compressed and unexpanded
                if st.unresolved_conflicts:
                    conflicts += 1
            n = max(1, len(probes))
            return sum(recalls) / n, sum(toks) / n, sum(lat) / n, sum(prec) / n, conflicts / n

        # Reference: the *best* raw (level-0) retrieval over the route sets in the search
        # space, at this candidate's depth and expansion budget. The bound is relative to
        # the uncompressed system, not to a candidate's own raw twin, so a route choice
        # that loses recall is infeasible rather than merely cheap.
        rk = (cfg.top_k, cfg.expand_budget, id(probes))
        if rk not in self._raw_recall_cache:
            best = 0.0
            for routes in self.controller.search_space["routes"]:
                best = max(best, run(0, tuple(routes))[0])
            self._raw_recall_cache[rk] = (best,)
        r_raw = self._raw_recall_cache[rk][0]
        r_c, toks, lat, prec, contradiction_rate = run(cfg.compression_level)
        stored = sum(estimate_tokens(m.compressed_text) for m in self.memories.values())
        # restore
        for mid, (t, lvl, f) in original.items():
            m = self.memories[mid]
            m.compressed_text, m.compression_level, m.fidelity_score = t, lvl, f
            if m.active:
                self.lexical.add(mid, t)
                self.vector.add(mid, t)
                self.trie.add(mid, t)
        return Measurement(cfg, r_raw, r_c, (r_c / r_raw) if r_raw > 0 else 1.0, contradiction_rate, toks, lat, stored / raw_tokens, 1 - prec)

    def optimise(self, n_probes: int = 120, seed: int = 0, probes: list[ProbeQuery] | None = None) -> Measurement | None:
        """Guarded policy search. `probes` may be supplied from a query log (question,
        gold episode ids); otherwise self-generated probes are sampled from the archive."""
        probes = probes if probes is not None else make_probe_queries(self, n_probes, seed)
        if not probes:
            return None
        self._raw_recall_cache.clear()
        m = self.controller.optimise(self.evaluate_config, probes, seed=seed)
        self.recompress_all(self.controller.current.compression_level, self.controller.current.gate_profile)
        return m

    # ---- introspection -------------------------------------------------
    def stats(self) -> dict:
        s = self.telemetry.summary()
        s.update({"episodes": len(self.archive), "memories": len(self.memories), "active": sum(1 for m in self.memories.values() if m.active), "raw_bytes": self.archive.storage_bytes(), "compressed_tokens": sum(estimate_tokens(m.compressed_text) for m in self.memories.values()), "controller": self.controller.describe()})
        return s
