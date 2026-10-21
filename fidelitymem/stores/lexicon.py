"""Semantic lexicon: a hash map from each indexed (stemmed) term to its nearest
semantic neighbours among the *indexed vocabulary*, built from word vectors.

Used for query-side expansion of BM25: exact query terms keep weight 1.0, each
neighbour gets alpha * similarity. Everything stays inside one BM25 score, so a
semantic neighbour can never outrank an exact match on its own, which is the
failure mode of fusing a separate dense route by rank.
"""
from __future__ import annotations

import numpy as np

from ..text import STOPWORDS, content_tokens, stem, tokenize


class SemanticLexicon:
    def __init__(self, word_embedder, neighbours: int = 3, min_sim: float = 0.55):
        self.embedder = word_embedder
        self.m = neighbours
        self.min_sim = min_sim
        self.map: dict[str, list[tuple[str, float]]] = {}  # stem -> [(stem, sim)]

    def fit(self, texts: list[str]) -> "SemanticLexicon":
        words = sorted({t for text in texts for t in tokenize(text) if t not in STOPWORDS and t.isalpha() and len(t) > 2})
        if not words:
            return self
        vecs = self.embedder.embed(words)
        norms = np.linalg.norm(vecs, axis=1)
        keep = norms > 0
        words = [w for w, k in zip(words, keep) if k]
        vecs = vecs[keep] / norms[keep, None]
        stems = [stem(w) for w in words]
        by_stem: dict[str, dict[str, float]] = {}
        sims = vecs @ vecs.T
        np.fill_diagonal(sims, -1.0)
        top = np.argpartition(-sims, min(self.m * 4, len(words) - 1), axis=1)[:, : self.m * 4]
        for i, w in enumerate(words):
            cand = sorted(((sims[i, j], stems[j]) for j in top[i] if stems[j] != stems[i]), reverse=True)
            d = by_stem.setdefault(stems[i], {})
            for s, st in cand:
                if s < self.min_sim:
                    break
                d[st] = max(d.get(st, 0.0), float(s))
        self.map = {k: sorted(v.items(), key=lambda x: -x[1])[: self.m] for k, v in by_stem.items()}
        return self

    def expand(self, query: str, alpha: float = 0.4) -> dict[str, float]:
        terms = {t: 1.0 for t in set(content_tokens(query))}
        for t in list(terms):
            for nb, sim in self.map.get(t, ()):
                if nb not in terms:
                    terms[nb] = max(terms.get(nb, 0.0), alpha * sim)
        return terms


__all__ = ["SemanticLexicon"]
