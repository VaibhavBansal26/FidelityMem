"""A retrieval-adapted embedder trained in NumPy.

    text  ->  pretrained word vectors (spaCy)  ->  learned term weights  ->  pooled
          ->  learned linear projection W (d_in x d_out)  ->  unit-norm embedding

Training: InfoNCE over (query, gold passage) pairs with in-batch negatives plus
BM25 hard negatives, Adam, a handful of epochs. Term weights are a learned
scalar per vocabulary word (initialised from IDF) so frequent chat words stop
dominating the mean; W is initialised to the identity (truncated) so the
untrained model equals the base embedder and training can only move it
towards the task.

It implements the `Embedder` protocol, so it drops into every system in the
benchmark, and it is saved/loaded as .npz so an experiment can reuse it.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..text import content_tokens


@dataclass
class AdaptedEmbedder:
    base: object  # an Embedder exposing .vector_of(token) -> np.ndarray | None  (see SpacyTokenVectors)
    W: np.ndarray
    term_logw: dict[str, float] = field(default_factory=dict)
    default_logw: float = 0.0
    name: str = "adapted"

    @property
    def dim(self) -> int:
        return int(self.W.shape[1])

    def _pool(self, text: str) -> np.ndarray:
        toks = content_tokens(text)
        vecs, ws = [], []
        for t in toks:
            v = self.base.vector_of(t)
            if v is None:
                continue
            vecs.append(v)
            ws.append(math.exp(self.term_logw.get(t, self.default_logw)))
        if not vecs:
            return np.zeros(self.base.dim, dtype=np.float32)
        return np.average(np.stack(vecs), axis=0, weights=np.asarray(ws, dtype=np.float32))

    def embed(self, texts: list[str]) -> np.ndarray:
        P = np.stack([self._pool(t) for t in texts]).astype(np.float32)
        Z = P @ self.W
        n = np.linalg.norm(Z, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return (Z / n).astype(np.float32)

    def save(self, path: str) -> None:
        words = list(self.term_logw)
        np.savez(path, W=self.W, words=np.array(words, dtype=object), logw=np.array([self.term_logw[w] for w in words], dtype=np.float32), default_logw=self.default_logw)

    @classmethod
    def load(cls, path: str, base) -> "AdaptedEmbedder":
        z = np.load(path, allow_pickle=True)
        return cls(base, z["W"], dict(zip(z["words"].tolist(), z["logw"].tolist())), float(z["default_logw"]), name=f"adapted:{path}")


class SpacyTokenVectors:
    """Token-level access to spaCy's pretrained vectors (the base of the adapter)."""

    def __init__(self, model: str = "en_core_web_md"):
        import spacy

        self._nlp = spacy.load(model, exclude=["parser", "ner", "lemmatizer", "tagger", "attribute_ruler"])
        self.dim = int(self._nlp.vocab.vectors.shape[1])
        self._cache: dict[str, np.ndarray | None] = {}

    def vector_of(self, tok: str):
        """Vector for a token; an identifier without one ("billing-api", "ledger_core",
        "v2.1") takes the mean of its sub-token vectors so entity names are not lost."""
        if tok not in self._cache:
            lex = self._nlp.vocab[tok]
            if lex.has_vector:
                self._cache[tok] = lex.vector.astype(np.float32)
            else:
                parts = [p for p in re.split(r"[-_./:]+", tok) if p]
                vecs = [self._nlp.vocab[p].vector for p in parts if self._nlp.vocab[p].has_vector]
                self._cache[tok] = np.mean(np.stack(vecs), axis=0).astype(np.float32) if vecs else None
        return self._cache[tok]


def _idf(texts: list[str]) -> dict[str, float]:
    df: Counter = Counter()
    for t in texts:
        df.update(set(content_tokens(t)))
    n = len(texts)
    return {w: math.log((n + 1) / (c + 0.5)) for w, c in df.items()}


def train_adapter(
    base: SpacyTokenVectors,
    pairs: list[tuple[str, str]],
    corpus: list[str],
    d_out: int = 256,
    epochs: int = 12,
    batch: int = 64,
    lr: float = 2e-3,
    temperature: float = 0.05,
    hard_negatives: int = 2,
    seed: int = 0,
    log=print,
) -> AdaptedEmbedder:
    """Contrastive training of term weights + projection on (query, gold text) pairs.

    Hard negatives come from BM25 over `corpus` (the top non-gold hits for each query),
    so the model learns to separate lexically similar wrong passages, which is exactly
    where a mean-of-word-vectors embedder fails."""
    from .lexical import BM25Index

    rng = np.random.default_rng(seed)
    idf = _idf(corpus)
    vocab = sorted({t for q, p in pairs for t in content_tokens(q) + content_tokens(p)} | set(idf))
    widx = {w: i for i, w in enumerate(vocab)}
    logw = np.array([math.log(max(1e-3, idf.get(w, 1.0))) for w in vocab], dtype=np.float32)
    default_logw = float(np.median(logw))
    d_in = base.dim
    W = np.eye(d_in, d_out, dtype=np.float32) if d_out <= d_in else np.eye(d_in, dtype=np.float32)
    d_out = W.shape[1]

    # token matrices per text (cached): list of (indices into vocab, vectors)
    def prep(text: str):
        toks = [t for t in content_tokens(text) if base.vector_of(t) is not None]
        if not toks:
            return np.zeros(0, dtype=np.int64), np.zeros((0, d_in), dtype=np.float32)
        return np.array([widx.get(t, -1) for t in toks]), np.stack([base.vector_of(t) for t in toks])

    bm = BM25Index()
    for i, t in enumerate(corpus):
        bm.add(str(i), t)
    gold_text = {q: p for q, p in pairs}
    negs: dict[str, list[str]] = {}
    for q, p in pairs:
        hits = [corpus[int(d)] for d, _ in bm.search(q, hard_negatives + 3)]
        negs[q] = [h for h in hits if h != p][:hard_negatives]
    cache: dict[str, tuple] = {}

    def enc(text: str, lw: np.ndarray):
        if text not in cache:
            cache[text] = prep(text)
        idx, V = cache[text]
        if len(idx) == 0:
            return np.zeros(d_in, dtype=np.float32), idx, V, np.zeros(0, dtype=np.float32)
        w = np.exp(np.where(idx >= 0, lw[np.maximum(idx, 0)], default_logw))
        pooled = (w[:, None] * V).sum(0) / w.sum()
        return pooled, idx, V, w

    # Adam state
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mL, vL = np.zeros_like(logw), np.zeros_like(logw)
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    order = np.arange(len(pairs))
    for ep in range(epochs):
        rng.shuffle(order)
        total, nb = 0.0, 0
        for s in range(0, len(order), batch):
            idxs = order[s : s + batch]
            qs = [pairs[i][0] for i in idxs]
            ps = [pairs[i][1] for i in idxs]
            ns = [n for q in qs for n in negs.get(q, [])]
            docs = ps + ns
            Q = [enc(q, logw) for q in qs]
            D = [enc(d, logw) for d in docs]
            Pq = np.stack([x[0] for x in Q])
            Pd = np.stack([x[0] for x in D])
            Zq, Zd = Pq @ W, Pd @ W
            nq = np.linalg.norm(Zq, axis=1, keepdims=True) + 1e-8
            nd = np.linalg.norm(Zd, axis=1, keepdims=True) + 1e-8
            Uq, Ud = Zq / nq, Zd / nd
            S = (Uq @ Ud.T) / temperature  # (B, B+N)
            S -= S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            B = len(qs)
            loss = -np.mean(np.log(P[np.arange(B), np.arange(B)] + 1e-12))
            total += float(loss)
            nb += 1
            # gradients
            G = P.copy()
            G[np.arange(B), np.arange(B)] -= 1.0
            G /= B * temperature
            dUq = G @ Ud  # (B, d_out)
            dUd = G.T @ Uq  # (B+N, d_out)
            # back through normalisation
            dZq = (dUq - Uq * (dUq * Uq).sum(1, keepdims=True)) / nq
            dZd = (dUd - Ud * (dUd * Ud).sum(1, keepdims=True)) / nd
            dW = Pq.T @ dZq + Pd.T @ dZd
            dPq = dZq @ W.T
            dPd = dZd @ W.T
            dlogw = np.zeros_like(logw)
            for (pooled, idx, V, w), dP in list(zip(Q, dPq)) + list(zip(D, dPd)):
                if len(idx) == 0:
                    continue
                # pooled = sum(w_i V_i)/sum(w); d pooled/d w_i = (V_i - pooled)/sum(w); d w_i/d logw_i = w_i
                g = ((V - pooled) @ dP) * w / w.sum()
                valid = idx >= 0
                np.add.at(dlogw, idx[valid], g[valid])
            step += 1
            for param, grad, m, v in ((W, dW, mW, vW), (logw, dlogw, mL, vL)):
                m *= b1
                m += (1 - b1) * grad
                v *= b2
                v += (1 - b2) * grad * grad
                mhat = m / (1 - b1**step)
                vhat = v / (1 - b2**step)
                param -= lr * mhat / (np.sqrt(vhat) + eps)
        log(f"[adapter] epoch {ep + 1}/{epochs} loss={total / max(1, nb):.4f}")
    return AdaptedEmbedder(base, W.astype(np.float32), {w: float(logw[i]) for w, i in widx.items()}, default_logw, name=f"adapted(d={d_out},pairs={len(pairs)})")
