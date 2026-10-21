"""Pretrained embedders.

`best_available()` returns the strongest embedder installed, in this order:
  1. SentenceTransformerEmbedder (sentence-transformers, any model id)
  2. SpacyVectorEmbedder (spaCy pipeline with pretrained word vectors,
     e.g. en_core_web_md: 300-d GloVe-style vectors, mean-pooled over content
     tokens). Real pretrained semantics, no network needed once installed.
  3. HashingEmbedder (hermetic fallback; not semantic)

All implement the `Embedder` protocol (dim, embed(list[str]) -> np.ndarray of
unit-norm rows).
"""
from __future__ import annotations

import numpy as np

from .vector import HashingEmbedder


class SpacyVectorEmbedder:
    name = "spacy"

    def __init__(self, model: str = "en_core_web_md", idf_weighted: bool = True):
        import spacy  # noqa: F401 - imported lazily so the package stays optional

        self._nlp = spacy.load(model, exclude=["parser", "ner", "lemmatizer", "tagger", "attribute_ruler"])
        self.dim = int(self._nlp.vocab.vectors.shape[1])
        self.idf_weighted = idf_weighted
        self.name = f"spacy:{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, doc in enumerate(self._nlp.pipe(texts, batch_size=256)):
            vecs, weights = [], []
            for tok in doc:
                if tok.is_stop or tok.is_punct or tok.is_space or not tok.has_vector:
                    continue
                vecs.append(tok.vector)
                # rarer words carry more meaning; spaCy's prob is a log-probability (<= 0)
                weights.append(max(0.5, -tok.prob / 10.0) if self.idf_weighted and tok.prob < 0 else 1.0)
            if vecs:
                v = np.average(np.stack(vecs), axis=0, weights=np.asarray(weights, dtype=np.float32))
                n = np.linalg.norm(v)
                if n > 0:
                    out[row] = v / n
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # noqa: F401

        self._m = SentenceTransformer(model)
        self.dim = int(self._m.get_sentence_embedding_dimension())
        self.name = f"st:{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._m.encode(texts, normalize_embeddings=True, batch_size=64), dtype=np.float32)


def best_available(prefer: str | None = None):
    """Return the best installed embedder. `prefer` may force 'st', 'spacy', 'hashing' or
    'adapted:<path.npz>' (a retrieval-adapted embedder saved by eval.train_embedder)."""
    if prefer and prefer.startswith("adapted:"):
        from .learned_embedder import AdaptedEmbedder, SpacyTokenVectors

        return AdaptedEmbedder.load(prefer.split(":", 1)[1], SpacyTokenVectors())
    order = [prefer] if prefer else ["st", "spacy", "hashing"]
    for kind in order:
        try:
            if kind == "st":
                return SentenceTransformerEmbedder()
            if kind == "spacy":
                return SpacyVectorEmbedder()
            if kind == "hashing":
                return HashingEmbedder()
        except Exception:  # noqa: BLE001 - optional dependency missing
            continue
    return HashingEmbedder()


def embedder_name(e) -> str:
    return getattr(e, "name", type(e).__name__)
