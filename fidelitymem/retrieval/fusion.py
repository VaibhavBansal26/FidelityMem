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
) -> float:
    age_days = max(0.0, (now - mem.valid_from).total_seconds() / 86400)
    recency = math.exp(-math.log(2) * age_days / max(1e-6, halflife_days))
    utility = math.tanh(mem.utility)
    authority = mem.provenance.authority * mem.confidence
    boost = max((type_boost.get(t.value, 0.0) for t in mem.types), default=0.0)
    return (
        fused * 100.0  # RRF values are ~1/60; rescale so the other terms matter
        + weights.get("recency", 0.3) * recency
        + weights.get("utility", 0.4) * utility
        + weights.get("authority", 0.4) * authority
        + boost
    )
