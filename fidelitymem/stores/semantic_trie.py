"""Semantic-codebook trie (prototype of the user's proposal).

Idea: number semantically similar words with one code (a codebook built by
clustering the embedder's word vectors), store every memory as its code
sequence in a word trie, and look memories up by the code sequence of a query.
Shared prefixes are stored once, repeated phrases cost nothing, and a query
phrased with a synonym still hits.

The protected-token rule is what keeps it from destroying evidence: entities,
numbers, dates and negation cues are never coded; they keep their own verbatim
node. Everything else (content words) may share a code with its neighbours.

What this buys and what it costs is measured, not assumed:
  * storage ratio:   bytes of the coded trie vs raw text (dedup + codebook)
  * decode fidelity: decode the code sequence back to text (each code -> its
                     representative word) and score it with the fidelity probes
  * retrieval:       recall@K of sequence/bag-of-codes matching vs BM25

This is a storage/index technique: context tokens are unchanged, because the
model still reads decoded text, and the decoded text is lossy wherever a code
merged two words with different meanings.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..text import NEGATION_CUES, STOPWORDS, extract_dates, extract_entities, extract_numbers, tokenize

_PROTECT = re.compile(r"[\d\-_.:/]")  # identifiers, numbers, dates keep their own node


@dataclass
class _Node:
    children: dict[str, "_Node"] = field(default_factory=dict)
    ids: set[str] = field(default_factory=set)


class SemanticCodebook:
    """k-means over word vectors; code = cluster id; representative = word nearest the centroid."""

    def __init__(self, embedder, n_codes: int = 2000, seed: int = 0):
        self.embedder = embedder
        self.n_codes = n_codes
        self.seed = seed
        self.code_of: dict[str, int] = {}
        self.rep: dict[int, str] = {}
        self._vocab: list[str] = []

    def fit(self, texts: list[str]) -> "SemanticCodebook":
        counts = Counter(t for text in texts for t in tokenize(text) if t not in STOPWORDS and t.isalpha() and len(t) > 2)
        vocab = [w for w, _ in counts.most_common()]
        if not vocab:
            return self
        vecs = self.embedder.embed(vocab)
        k = min(self.n_codes, len(vocab))
        rng = np.random.default_rng(self.seed)
        centroids = vecs[rng.choice(len(vocab), size=k, replace=False)].copy()
        assign = np.zeros(len(vocab), dtype=int)
        for _ in range(8):  # a few Lloyd iterations are enough for a codebook
            sims = vecs @ centroids.T
            assign = sims.argmax(axis=1)
            for c in range(k):
                members = vecs[assign == c]
                if len(members):
                    m = members.mean(axis=0)
                    n = np.linalg.norm(m)
                    centroids[c] = m / n if n > 0 else m
        self.code_of = {w: int(c) for w, c in zip(vocab, assign)}
        sims = vecs @ centroids.T
        best: dict[int, tuple[float, str]] = {}
        for i, (w, c) in enumerate(zip(vocab, assign)):
            s = float(sims[i, c])
            if int(c) not in best or s > best[int(c)][0]:
                best[int(c)] = (s, w)
        self.rep = {c: w for c, (_, w) in best.items()}
        self._vocab = vocab
        return self

    def encode_token(self, t: str) -> str:
        if t in STOPWORDS:
            return ""  # dropped: stopwords carry no evidence and are restored generically
        if t in NEGATION_CUES or t.endswith("n't") or _PROTECT.search(t) or t not in self.code_of:
            return t  # protected / unknown: verbatim
        return f"#{self.code_of[t]}"

    def decode_token(self, c: str) -> str:
        if c.startswith("#") and c[1:].isdigit():
            return self.rep.get(int(c[1:]), c)
        return c


class SemanticTrie:
    def __init__(self, codebook: SemanticCodebook):
        self.cb = codebook
        self.root = _Node()
        self._seq: dict[str, list[str]] = {}
        self._protected: dict[str, set[str]] = {}
        self._nodes = 1
        self._code_docs: dict[str, set[str]] = defaultdict(set)

    def encode(self, text: str) -> list[str]:
        protected = {e.lower() for e in extract_entities(text)} | {n.lower() for n in extract_numbers(text)} | {d.lower() for d in extract_dates(text)}
        out = []
        for t in tokenize(text):
            if t in protected:
                out.append(t)
                continue
            c = self.cb.encode_token(t)
            if c:
                out.append(c)
        return out

    def add(self, doc_id: str, text: str) -> None:
        seq = self.encode(text)
        self._seq[doc_id] = seq
        node = self.root
        for c in seq:
            nxt = node.children.get(c)
            if nxt is None:
                nxt = _Node()
                node.children[c] = nxt
                self._nodes += 1
            node = nxt
            self._code_docs[c].add(doc_id)
        node.ids.add(doc_id)

    def remove(self, doc_id: str) -> None:
        seq = self._seq.pop(doc_id, None)
        if seq is None:
            return
        for c in set(seq):
            self._code_docs[c].discard(doc_id)
        node = self.root
        for c in seq:
            node = node.children.get(c)
            if node is None:
                return
        node.ids.discard(doc_id)

    def decode(self, doc_id: str) -> str:
        return " ".join(self.cb.decode_token(c) for c in self._seq.get(doc_id, []))

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Bag-of-codes match with idf weighting (synonyms share a code, so a query
        phrased differently still matches), plus a bonus for the longest shared
        code n-gram (exact phrase in code space)."""
        q = self.encode(query)
        if not q:
            return []
        n_docs = max(1, len(self._seq))
        scores: dict[str, float] = defaultdict(float)
        for c in set(q):
            docs = self._code_docs.get(c, ())
            if not docs:
                continue
            idf = np.log(1 + n_docs / len(docs))
            for d in docs:
                scores[d] += idf
        for n in (3, 2):
            for i in range(len(q) - n + 1):
                node = self.root
                # walk any position: use code_docs intersection as an approximation of phrase presence
                gram = q[i : i + n]
                cand = set.intersection(*(self._code_docs.get(c, set()) for c in gram)) if all(c in self._code_docs for c in gram) else set()
                for d in cand:
                    seq = self._seq[d]
                    if any(seq[j : j + n] == gram for j in range(len(seq) - n + 1)):
                        scores[d] += 0.5 * n
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    def storage_bytes(self) -> int:
        """Trie nodes (one code string + child map entry each) plus the codebook."""
        node_bytes = sum(len(c) + 8 for n in self._walk() for c in n.children)
        cb_bytes = sum(len(w) + 4 for w in self.cb.code_of) + sum(len(w) + 4 for w in self.cb.rep.values())
        return node_bytes + cb_bytes

    def _walk(self):
        stack = [self.root]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(n.children.values())


# --------------------------------------------------------------------------- node embeddings
class NodeEmbeddedTrie(SemanticTrie):
    """Semantic trie whose nodes carry embeddings.

    * A code node's embedding is its cluster centroid, stored once in the codebook.
    * A protected (verbatim) token keeps its own word vector.
    * A memory's embedding is the IDF-weighted mean of the node vectors along its path,
      so the dense index stores one short integer code sequence per memory instead of a
      float vector: vector quantisation of the dense route, with the query still embedded
      in full and compared asymmetrically.
    """

    def __init__(self, codebook: SemanticCodebook, token_vectors, idf: dict[str, float] | None = None):
        super().__init__(codebook)
        self.tv = token_vectors  # exposes vector_of(token) -> vector | None
        self.idf = idf or {}
        self.centroids = self._centroids()
        self._emb: dict[str, np.ndarray] = {}

    def _centroids(self) -> dict[int, np.ndarray]:
        sums: dict[int, list] = defaultdict(list)
        for w, c in self.cb.code_of.items():
            v = self.tv.vector_of(w)
            if v is not None:
                sums[c].append(v)
        return {c: np.mean(np.stack(vs), axis=0) for c, vs in sums.items() if vs}

    def node_vector(self, code: str):
        if code.startswith("#") and code[1:].isdigit():
            return self.centroids.get(int(code[1:]))
        return self.tv.vector_of(code)

    def add(self, doc_id: str, text: str) -> None:
        super().add(doc_id, text)
        self._emb.pop(doc_id, None)

    def embedding(self, doc_id: str) -> np.ndarray:
        if doc_id not in self._emb:
            vecs, ws = [], []
            for c in self._seq.get(doc_id, []):
                v = self.node_vector(c)
                if v is None:
                    continue
                word = self.cb.rep.get(int(c[1:]), c) if c.startswith("#") and c[1:].isdigit() else c
                vecs.append(v)
                ws.append(self.idf.get(word, 1.0))
            if not vecs:
                e = np.zeros(self.tv.dim, dtype=np.float32)
            else:
                e = np.average(np.stack(vecs), axis=0, weights=np.asarray(ws, dtype=np.float32)).astype(np.float32)
                n = np.linalg.norm(e)
                if n > 0:
                    e /= n
            self._emb[doc_id] = e
        return self._emb[doc_id]

    def dense_search(self, query_vec: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        ids = list(self._seq)
        if not ids:
            return []
        M = np.stack([self.embedding(i) for i in ids])
        sims = M @ query_vec
        top = np.argsort(-sims)[:k]
        return [(ids[i], float(sims[i])) for i in top]

    def index_bytes(self) -> int:
        """Codebook centroids (float16) + one 2-byte code per token per memory."""
        return len(self.centroids) * self.tv.dim * 2 + sum(2 * len(s) for s in self._seq.values())
