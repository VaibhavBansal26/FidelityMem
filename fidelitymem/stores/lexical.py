"""Pure-Python BM25 index with incremental add/remove."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from ..text import content_tokens


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._df: Counter = Counter()
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)  # term -> {doc_id: tf}
        self._len: dict[str, int] = {}
        self._total_len = 0

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._len:
            self.remove(doc_id)
        toks = content_tokens(text)
        tf = Counter(toks)
        for t, c in tf.items():
            self._postings[t][doc_id] = c
            self._df[t] += 1
        self._len[doc_id] = len(toks)
        self._total_len += len(toks)

    def remove(self, doc_id: str) -> None:
        if doc_id not in self._len:
            return
        for t, docs in list(self._postings.items()):
            if doc_id in docs:
                del docs[doc_id]
                self._df[t] -= 1
                if not docs:
                    del self._postings[t]
                    del self._df[t]
        self._total_len -= self._len.pop(doc_id)

    def __len__(self) -> int:
        return len(self._len)

    def search(self, query: str, k: int = 10, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        n = len(self._len)
        if n == 0:
            return []
        avgdl = self._total_len / n
        scores: dict[str, float] = defaultdict(float)
        for t in set(content_tokens(query)):
            docs = self._postings.get(t)
            if not docs:
                continue
            idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
            for d, tf in docs.items():
                if allowed is not None and d not in allowed:
                    continue
                dl = self._len[d]
                scores[d] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / avgdl))
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
