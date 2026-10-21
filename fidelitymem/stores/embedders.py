"""Pretrained embedders.

`best_available()` returns the strongest embedder installed, in this order:
  0. OnnxEmbedder (data/models/all-MiniLM-L6-v2-onnx, onnxruntime, CPU)
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

        self._m = SentenceTransformer(model, device="cpu")
        self.dim = int(self._m.get_sentence_embedding_dimension())
        self.name = f"st:{model.rstrip('/').rsplit('/', 1)[-1]}"

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._m.encode(texts, normalize_embeddings=True, batch_size=64), dtype=np.float32)


class OnnxEmbedder:
    """Sentence-transformer checkpoint exported to ONNX (tokenizer.json + model.onnx in one
    directory), run with onnxruntime on CPU: mean pooling over the attention mask, unit norm.
    Works without torch or network access once the directory is in place."""

    def __init__(self, path: str = "data/models/all-MiniLM-L6-v2-onnx", max_length: int = 256, batch_size: int = 64, chunk_words: int = 120):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(f"{path}/tokenizer.json")
        self._tok.enable_truncation(max_length)
        self._tok.enable_padding(pad_id=0, pad_token="[PAD]")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self._sess = ort.InferenceSession(f"{path}/model.onnx", opts, providers=["CPUExecutionProvider"])
        self._inputs = [i.name for i in self._sess.get_inputs()]
        self.batch_size = batch_size
        self.chunk_words = chunk_words  # long records are embedded as the mean of overlapping word windows
        self.dim = int(self.embed(["dim probe"]).shape[1])
        self.name = f"onnx:{path.rstrip('/').rsplit('/', 1)[-1]}"

    def _chunks(self, text: str) -> list[str]:
        words = text.split()
        if len(words) <= self.chunk_words:
            return [text]
        step = max(1, self.chunk_words // 2)
        return [" ".join(words[i : i + self.chunk_words]) for i in range(0, max(1, len(words) - step), step)]

    def embed(self, texts: list[str]) -> np.ndarray:
        pieces, owner = [], []
        for n, t in enumerate(texts):
            for c in self._chunks(t):
                pieces.append(c)
                owner.append(n)
        flat = self._embed_flat(pieces)
        if len(pieces) == len(texts):
            return flat
        out = np.zeros((len(texts), flat.shape[1]), dtype=np.float32)
        np.add.at(out, np.asarray(owner), flat)
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return out

    def _embed_flat(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self._tok.encode_batch(texts[i : i + self.batch_size])
            ids = np.asarray([e.ids for e in enc], dtype=np.int64)
            mask = np.asarray([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self._sess.run(None, feed)[0]  # (b, t, d)
            m = mask[..., None].astype(np.float32)
            v = (hidden * m).sum(1) / np.maximum(m.sum(1), 1e-6)
            v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
            out.append(v.astype(np.float32))
        return np.concatenate(out) if out else np.zeros((0, getattr(self, "dim", 384)), dtype=np.float32)


class CachingEmbedder:
    """Memoises any embedder by text so that systems rebuilt per question (LongMemEval
    protocol) never embed the same session twice."""

    def __init__(self, inner, max_size: int = 200_000):
        self.inner = inner
        self.dim = inner.dim
        self.name = embedder_name(inner)
        self._cache: dict[str, np.ndarray] = {}
        self._max = max_size

    def embed(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            if len(self._cache) + len(missing) > self._max:
                self._cache.clear()
            for t, v in zip(missing, self.inner.embed(missing)):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)


def best_available(prefer: str | None = None):
    """Return the best installed embedder. `prefer` may force 'st', 'spacy', 'hashing' or
    'adapted:<path.npz>' (a retrieval-adapted embedder saved by eval.train_embedder)."""
    if prefer and prefer.startswith("adapted:"):
        from .learned_embedder import AdaptedEmbedder, SpacyTokenVectors

        return AdaptedEmbedder.load(prefer.split(":", 1)[1], SpacyTokenVectors())
    if prefer and prefer.startswith("onnx:"):
        return OnnxEmbedder(prefer.split(":", 1)[1])
    if prefer and prefer.startswith("st:"):
        return SentenceTransformerEmbedder(prefer.split(":", 1)[1])
    order = [prefer] if prefer else ["onnx", "st", "spacy", "hashing"]
    for kind in order:
        try:
            if kind == "onnx":
                return OnnxEmbedder()
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
