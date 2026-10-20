"""Dense index with a pluggable embedder.

`HashingEmbedder` is an offline, deterministic char-n-gram + word hashing
embedder. It is *not* a semantic model, but it gives dense retrieval behaviour
(fuzzy matching, typo tolerance) with zero downloads, which keeps the test
suite and benchmark harness hermetic. Swap in a sentence-transformer or API
embedder via the `Embedder` protocol for real deployments.
"""
from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from ..text import content_tokens


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = 512, ngram: tuple[int, int] = (3, 5)):
        self.dim = dim
        self.ngram = ngram

    def _features(self, text: str) -> list[str]:
        toks = content_tokens(text)
        feats = list(toks)
        for t in toks:
            padded = f"#{t}#"
            for n in range(self.ngram[0], self.ngram[1] + 1):
                feats.extend(padded[i : i + n] for i in range(max(0, len(padded) - n + 1)))
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for f in self._features(text):
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
                idx = h % self.dim
                sign = 1.0 if (h >> 63) & 1 else -1.0
                out[row, idx] += sign
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


_DEFAULT_EMBEDDER: "Embedder | None" = None


def set_default_embedder(e: "Embedder | None") -> None:
    """Process-wide default used by every VectorIndex() created without an explicit
    embedder, so a benchmark can give all systems the same semantic model."""
    global _DEFAULT_EMBEDDER
    _DEFAULT_EMBEDDER = e


def get_default_embedder() -> "Embedder":
    return _DEFAULT_EMBEDDER or HashingEmbedder()


class VectorIndex:
    def __init__(self, embedder: Embedder | None = None, cache_size: int = 50_000):
        self.embedder = embedder or get_default_embedder()
        self._cache: dict[str, np.ndarray] = {}
        self._cache_size = cache_size
        self._ids: list[str] = []
        self._pos: dict[str, int] = {}
        self._mat = np.zeros((1024, self.embedder.dim), dtype=np.float32)  # capacity-doubling buffer
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def _embed(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            for t, v in zip(missing, self.embedder.embed(missing)):
                if len(self._cache) >= self._cache_size:
                    self._cache.clear()
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts])

    def _grow(self, need: int) -> None:
        if self._n + need <= self._mat.shape[0]:
            return
        cap = self._mat.shape[0]
        while cap < self._n + need:
            cap *= 2
        new = np.zeros((cap, self.embedder.dim), dtype=np.float32)
        new[: self._n] = self._mat[: self._n]
        self._mat = new

    def add(self, doc_id: str, text: str) -> None:
        self.add_many([(doc_id, text)])

    def add_many(self, items: list[tuple[str, str]]) -> None:
        if not items:
            return
        vecs = self._embed([t for _, t in items])
        new_rows = []
        for (i, _), v in zip(items, vecs):
            if i in self._pos:
                self._mat[self._pos[i]] = v
            else:
                new_rows.append((i, v))
        if new_rows:
            self._grow(len(new_rows))
            for i, v in new_rows:
                self._pos[i] = self._n
                self._ids.append(i)
                self._mat[self._n] = v
                self._n += 1

    def remove(self, doc_id: str) -> None:
        """Swap-remove: O(1), keeps the buffer dense."""
        p = self._pos.pop(doc_id, None)
        if p is None:
            return
        last = self._n - 1
        if p != last:
            last_id = self._ids[last]
            self._mat[p] = self._mat[last]
            self._ids[p] = last_id
            self._pos[last_id] = p
        self._ids.pop()
        self._n -= 1

    def search(self, query: str, k: int = 10, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        if self._n == 0:
            return []
        q = self._embed([query])[0]
        if allowed is not None and len(allowed) < self._n:
            rows = [self._pos[i] for i in allowed if i in self._pos]
            if not rows:
                return []
            idx = np.fromiter(rows, dtype=np.int64, count=len(rows))
            sims = self._mat[idx] @ q
            ids = [self._ids[r] for r in rows]
        else:
            sims = self._mat[: self._n] @ q
            ids = self._ids
        k = min(k, len(ids))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(ids[i], float(sims[i])) for i in top]
