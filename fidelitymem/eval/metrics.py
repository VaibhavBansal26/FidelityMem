"""Retrieval, compression and operational metrics."""
from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not gold:
        return 1.0
    top = set(retrieved[:k])
    return len(top & set(gold)) / len(set(gold))


def precision_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    top = list(retrieved[:k])
    if not top:
        return 0.0
    return len(set(top) & set(gold)) / len(top)


def mrr(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    g = set(gold)
    for i, r in enumerate(retrieved):
        if r in g:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    g = set(gold)
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(retrieved[:k]) if r in g)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(g))))
    return dcg / ideal if ideal else 0.0


def complete_set_recall(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """1.0 only if *every* gold item is in the top-k (multi-hop needs all of them)."""
    return 1.0 if set(gold) <= set(retrieved[:k]) else 0.0


def duplicate_rate(retrieved: Sequence[str]) -> float:
    if not retrieved:
        return 0.0
    return 1 - len(set(retrieved)) / len(retrieved)


def recall_retention(recall_compressed: float, recall_raw: float) -> float:
    return 1.0 if recall_raw == 0 else recall_compressed / recall_raw


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]
