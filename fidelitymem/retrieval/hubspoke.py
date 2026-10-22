"""Hub-and-spoke reranking over a fused candidate set.

Candidates (memories) are spokes attached to hubs: extracted entities, the
session or conversation they came from, and their time window (day). Hubs are
linked by co-membership. Scores propagate through a personalised bipartite walk:
memories vote for hubs, hubs exchange score with linked hubs, hubs return score
to memories; a restart term anchored to the fused retrieval scores keeps the
ranking query-specific. High-degree hubs are damped by an IDF-style penalty.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Spoke:
    id: str
    entities: list[str]
    session: str | None
    day: str | None


def hub_spoke_rerank(cands: list[tuple[str, float]], spokes: dict[str, Spoke], alpha: float = 0.5, beta: float = 0.3,
                     iters: int = 6, use_entities: bool = True, use_sessions: bool = True, use_days: bool = True) -> list[tuple[str, float]]:
    """cands: [(memory id, fused score)] in current order. Returns the reranked list.
    alpha = share of score that flows through the graph (1-alpha restarts to fused scores);
    beta = share of hub score exchanged with linked hubs per iteration."""
    if len(cands) < 3:
        return cands
    ids = [i for i, _ in cands]
    s0 = {i: max(s, 0.0) for i, s in cands}
    z = sum(s0.values()) or 1.0
    s0 = {i: v / z for i, v in s0.items()}
    # bipartite membership
    hubs_of: dict[str, list[str]] = {}
    members: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        sp = spokes.get(i)
        hs = []
        if sp is not None:
            if use_entities:
                hs += [f"e:{e.lower()}" for e in sp.entities]
            if use_sessions and sp.session:
                hs.append(f"s:{sp.session}")
            if use_days and sp.day:
                hs.append(f"d:{sp.day}")
        hs = list(dict.fromkeys(hs))
        hubs_of[i] = hs
        for h in hs:
            members[h].append(i)
    if not members:
        return cands
    n = len(ids)
    damp = {h: 1.0 / (1.0 + math.log1p(max(0, len(ms) - 1))) for h, ms in members.items()}  # IDF-style penalty on big hubs
    # hub-hub links by co-membership (two hubs linked if a memory belongs to both)
    links: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for i, hs in hubs_of.items():
        for a in hs:
            for b in hs:
                if a != b:
                    links[a][b] += 1.0
    m = dict(s0)
    for _ in range(iters):
        # memories vote for hubs (each memory splits its score over its hubs)
        h: dict[str, float] = defaultdict(float)
        for i, hs in hubs_of.items():
            if hs:
                share = m[i] / len(hs)
                for x in hs:
                    h[x] += share
        for x in h:
            h[x] *= damp[x]
        # hubs exchange score with linked hubs
        if beta > 0:
            h2 = {x: (1 - beta) * v for x, v in h.items()}
            for x, v in h.items():
                nb = links.get(x)
                if nb:
                    tot = sum(nb.values())
                    for y, w in nb.items():
                        h2[y] = h2.get(y, 0.0) + beta * v * w / tot
            h = h2
        # hubs return score to members
        back: dict[str, float] = defaultdict(float)
        for x, v in h.items():
            ms = members.get(x, ())
            if ms:
                share = v / len(ms)
                for i in ms:
                    back[i] += share
        zb = sum(back.values()) or 1.0
        m = {i: (1 - alpha) * s0[i] + alpha * back.get(i, 0.0) / zb for i in ids}
    return sorted(m.items(), key=lambda x: x[1], reverse=True)


__all__ = ["Spoke", "hub_spoke_rerank"]
