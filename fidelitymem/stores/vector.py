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


class VectorIndex:
    def __init__(self, embedder: Embedder | None = None, cache_size: int = 50_000):
        self.embedder = embedder or HashingEmbedder()
        self._cache: dict[str, np.ndarray] = {}
        self._cache_size = cache_size
        self._ids: list[str] = []
        self._pos: dict[str, int] = {}
        self._mat = np.zeros((0, self.embedder.dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._ids)

    def _embed(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            for t, v in zip(missing, self.embedder.embed(missing)):
                if len(self._cache) >= self._cache_size:
                    self._cache.clear()
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts])

    def add(self, doc_id: str, text: str) -> None:
        vec = self._embed([text])[0]
        if doc_id in self._pos:
            self._mat[self._pos[doc_id]] = vec
            return
        self._pos[doc_id] = len(self._ids)
        self._ids.append(doc_id)
        self._mat = np.vstack([self._mat, vec[None, :]])

    def add_many(self, items: list[tuple[str, str]]) -> None:
        if not items:
            return
        vecs = self._embed([t for _, t in items])
        new = [(i, v) for (i, _), v in zip(items, vecs) if i not in self._pos]
        for (i, _), v in zip(items, vecs):
            if i in self._pos:
                self._mat[self._pos[i]] = v
        if new:
            for i, _ in new:
                self._pos[i] = len(self._ids)
                self._ids.append(i)
            self._mat = np.vstack([self._mat, np.stack([v for _, v in new])])

    def remove(self, doc_id: str) -> None:
        if doc_id not in self._pos:
            return
        p = self._pos.pop(doc_id)
        self._ids.pop(p)
        self._mat = np.delete(self._mat, p, axis=0)
        self._pos = {i: n for n, i in enumerate(self._ids)}

    def search(self, query: str, k: int = 10, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        q = self.embedder.embed([query])[0]
        sims = self._mat @ q
        if allowed is not None:
            mask = np.array([i in allowed for i in self._ids])
            sims = np.where(mask, sims, -np.inf)
        k = min(k, len(self._ids))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self._ids[i], float(sims[i])) for i in top if np.isfinite(sims[i])]
