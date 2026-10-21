"""Dry run: BM25 with hash-map semantic expansion (exact words weight 1.0, word-vector
neighbours alpha*sim) against plain BM25 and lexical+dense RRF, on LoCoMo sessions.

    python -m fidelitymem.eval.lexical_expansion --path data/locomo/locomo10.json --conversations 0
"""
from __future__ import annotations

import argparse
import json
import time

from ..stores import BM25Index, VectorIndex
from ..stores.lexicon import SemanticLexicon
from ..text import estimate_tokens
from .competitors import _run
from .datasets import load
from .harness import aggregate, format_table
from .synthetic import from_real


def run_seed(real, seed, k, alphas, ms, min_sim, word_embedder):
    bench = from_real(real, seed=seed)
    texts = [t for t, _ in bench.episodes]
    ids = [f"ep{i}" for i in range(len(texts))]
    bm = BM25Index()
    for i, t in zip(ids, texts):
        bm.add(i, t)
    stored = sum(estimate_tokens(t) for t in texts)
    out = [_run("bm25", bench, k, texts, stored, lambda q: [i for i, _ in bm.search(q, k)])]
    for m in ms:
        t0 = time.perf_counter()
        lex = SemanticLexicon(word_embedder, neighbours=m, min_sim=min_sim).fit(texts)
        print(f"[seed {seed}] lexicon m={m}: {len(lex.map)} keys, {time.perf_counter() - t0:.0f}s", flush=True)
        for a in alphas:
            r = _run(f"bm25+lexicon(m{m},a{a})", bench, k, texts, stored, lambda q, a=a, lex=lex: [i for i, _ in bm.search_weighted(lex.expand(q, a), k)])
            print(f"[seed {seed}] m={m} alpha={a}: R@{k}={r.recall:.3f} MRR={r.mrr:.3f} " + " ".join(f"{t}={v:.3f}" for t, v in sorted(r.per_type.items())), flush=True)
            out.append(r)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--session-level", action="store_true", default=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--conversations", type=int, nargs="*", default=None)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.2, 0.4, 0.7])
    ap.add_argument("--neighbours", type=int, nargs="+", default=[3, 6])
    ap.add_argument("--min-sim", type=float, default=0.55)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from ..stores.embedders import SpacyVectorEmbedder

    we = SpacyVectorEmbedder(idf_weighted=False)
    kw = {"per_turn": not a.session_level}
    if a.conversations is not None:
        kw["conversations"] = a.conversations
    real = load("locomo", a.path, **kw)
    print(f"episodes={len(real.episodes)} questions={len(real.questions)} seeds={a.seeds}", flush=True)
    per_seed = [run_seed(real, sd, a.k, a.alphas, a.neighbours, a.min_sim, we) for sd in a.seeds]
    agg = aggregate(per_seed) if len(per_seed) > 1 else per_seed[0]
    print(f"\n### BM25 + semantic lexicon expansion, LoCoMo sessions, seeds {a.seeds}\n{format_table(agg)}", flush=True)
    if a.out:
        json.dump({"aggregate": [{k2: v for k2, v in r.__dict__.items() if k2 != "per_query_recall"} for r in agg]}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
