"""Relational memory: entity graph with bounded multi-hop expansion."""
from __future__ import annotations

import math
from collections import defaultdict, deque

from ..types import Relation


class GraphStore:
    hub_degree: int = 40

    def __init__(self):
        self._adj: dict[str, set[str]] = defaultdict(set)  # entity -> entities
        self._entity_mems: dict[str, set[str]] = defaultdict(set)  # entity -> memory ids
        self._edges: list[tuple[Relation, str]] = []
        self._subject_mems: dict[str, set[str]] = defaultdict(set)  # entity as relation subject -> memory ids
        self._out: dict[str, list[tuple[str, str]]] = defaultdict(list)  # subject -> [(predicate, object)]

    def add_memory(self, mem_id: str, entities: list[str], relations: list[Relation]) -> None:
        for e in entities:
            self._entity_mems[self._norm(e)].add(mem_id)
        for r in relations:
            s, o = self._norm(r.subject), self._norm(r.object)
            self._adj[s].add(o)
            self._adj[o].add(s)
            self._entity_mems[s].add(mem_id)
            self._entity_mems[o].add(mem_id)
            self._edges.append((r, mem_id))
            if not r.negated:
                self._subject_mems[s].add(mem_id)
                self._out[s].append((r.predicate, o))
        # co-mention edges
        ns = [self._norm(e) for e in entities]
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                self._adj[ns[i]].add(ns[j])
                self._adj[ns[j]].add(ns[i])

    def remove_memory(self, mem_id: str) -> None:
        for e in list(self._entity_mems):
            self._entity_mems[e].discard(mem_id)
        for e in list(self._subject_mems):
            self._subject_mems[e].discard(mem_id)
        self._edges = [(r, m) for r, m in self._edges if m != mem_id]

    def relations_for(self, mem_id: str) -> list[Relation]:
        return [r for r, m in self._edges if m == mem_id]

    def successors(self, entity: str) -> list[str]:
        """Objects of outgoing relations from `entity` (relation-following, not co-mention)."""
        return [o for _, o in self._out.get(self._norm(entity), ())]

    def degree(self, entity: str) -> int:
        e = self._norm(entity)
        return len(self._entity_mems.get(e, ())) + len(self._adj.get(e, ()))

    def expand(self, seeds: list[str] | dict[str, float], hops: int = 2, k: int = 20) -> list[tuple[str, float]]:
        """Return memory ids reachable from seed entities.

        Scoring follows relations, not just co-mentions: a memory in which a
        reached entity is the *subject* of a relation scores 0.9/(1+hop); a
        memory that merely mentions the entity scores 0.5/(1+hop). Seeds (hop 0)
        score 1.0. This is what lets a two-hop question surface the second hop
        above the many memories that merely mention the bridge entity."""
        if not isinstance(seeds, dict):
            seeds = {s: 1.0 for s in seeds}
        seen: dict[str, tuple[int, float]] = {}  # entity -> (hop, seed weight)
        q: deque[tuple[str, int, float]] = deque((self._norm(s), 0, w) for s, w in seeds.items())
        while q:
            e, d, w = q.popleft()
            if e in seen or d > hops:
                continue
            seen[e] = (d, w)
            # follow outgoing relations first (they are the multi-hop path), then co-mentions;
            # never traverse *through* a hub entity reached by co-mention: it connects everything
            for _, o in self._out.get(e, ()):
                if o not in seen:
                    q.append((o, d + 1, w))
            if d > 0 and len(self._adj.get(e, ())) > self.hub_degree:
                continue
            for nb in self._adj.get(e, ()):
                if nb not in seen:
                    q.append((nb, d + 1, w * 0.5))
        scores: dict[str, float] = defaultdict(float)
        for e, (d, w) in seen.items():
            # hub penalty: an entity mentioned everywhere carries little evidence
            hub = 1.0 / (1.0 + math.log1p(max(0, len(self._entity_mems.get(e, ())) - 3)))
            for m in self._entity_mems.get(e, ()):
                base = 1.0 if d == 0 else 0.5
                scores[m] = max(scores[m], w * hub * base / (1 + d))
            for m in self._subject_mems.get(e, ()):
                base = 1.0 if d == 0 else 0.9
                scores[m] = max(scores[m], w * base / (1 + d))
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    def known_entities(self) -> set[str]:
        return set(self._entity_mems)

    @staticmethod
    def _norm(e: str) -> str:
        return e.strip().lower()
