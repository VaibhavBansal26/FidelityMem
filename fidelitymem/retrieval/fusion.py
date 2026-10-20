"""Score fusion: reciprocal-rank fusion plus the weighted linear ranking function."""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime

from ..types import Memory


def rrf(route_results: dict[str, list[tuple[str, float]]], weights: dict[str, float], k: int = 60) -> dict[str, float]:
    fused: dict[str, float] = defaultdict(float)
    for route, results in route_results.items():
        w = weights.get(route, 1.0)
        for rank, (doc_id, _) in enumerate(results):
            fused[doc_id] += w / (k + rank + 1)
    return dict(fused)


def final_score(
    mem: Memory,
    fused: float,
    weights: dict[str, float],
    now: datetime,
    halflife_days: float,
    type_boost: dict[str, float],
    fused_max: float = 1.0,
) -> float:
    """Rank-fused relevance, gently modulated by recency, learned utility, authority and type.

    RRF differences between adjacent ranks are small (1/60 vs 1/61), so additive bonuses
    would dominate the ordering. The bonuses therefore multiply a max-normalised fused
    score, and their weights are kept small enough that they break ties and nudge, but
    never outrank a clearly better lexical/dense/graph match."""
    age_days = max(0.0, (now - mem.valid_from).total_seconds() / 86400)
    recency = math.exp(-math.log(2) * age_days / max(1e-6, halflife_days))
    utility = math.tanh(mem.utility)
    authority = mem.provenance.authority * mem.confidence
    boost = max((type_boost.get(t.value, 0.0) for t in mem.types), default=0.0)
    base = fused / fused_max if fused_max > 0 else fused
    mod = 1.0 + weights.get("recency", 0.05) * recency + weights.get("utility", 0.2) * utility + weights.get("authority", 0.05) * authority + boost
    return base * mod
