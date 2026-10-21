"""Train the retrieval-adapted embedder and measure what it changes.

Training data: the controller's calibration split of LoCoMo (30 % of questions,
never scored) paired with their gold evidence, plus (query, gold) pairs from a
synthetic world generated with a *different* seed from the evaluation world.
Evaluation: dense-only Recall@K and BM25+dense fusion, base vs adapted, on the
LoCoMo test split (session and turn granularity) and on a synthetic world.

    python -m fidelitymem.eval.train_embedder --locomo data/locomo/locomo10.json --out docs/results/adapter.npz
"""
from __future__ import annotations

import argparse
import json
import random
import time

from ..retrieval.fusion import rrf
from ..stores import BM25Index, VectorIndex
from ..stores.embedders import SpacyVectorEmbedder
from ..stores.learned_embedder import AdaptedEmbedder, SpacyTokenVectors, train_adapter
from .datasets import load_locomo
from .metrics import recall_at_k
from .synthetic import Benchmark, from_real, generate


def split(bench: Benchmark, frac: float, seed: int):
    items = list(bench.items)
    random.Random(seed).shuffle(items)
    cut = int(len(items) * frac)
    return Benchmark(bench.episodes, items[:cut], {}, name=bench.name + "-calib"), Benchmark(bench.episodes, items[cut:], {}, name=bench.name)


def pairs_from(bench: Benchmark) -> list[tuple[str, str]]:
    out = []
    for it in bench.items:
        for gi in bench.gold_indices(it):
            out.append((it.query, bench.episodes[gi][0]))
    return out


def evaluate(bench: Benchmark, embedder, k: int) -> dict:
    texts = [t for t, _ in bench.episodes]
    ids = [f"ep{i}" for i in range(len(texts))]
    vi = VectorIndex(embedder)
    vi.add_many(list(zip(ids, texts)))
    bm = BM25Index()
    for i, t in zip(ids, texts):
        bm.add(i, t)
    dense, fused, lex = [], [], []
    for it in bench.items:
        gold = [f"ep{i}" for i in bench.gold_indices(it)]
        if not gold:
            continue
        d = [i for i, _ in vi.search(it.query, k)]
        dense.append(recall_at_k(d, gold, k))
        f = rrf({"lexical": bm.search(it.query, k * 3), "vector": vi.search(it.query, k * 3)}, {"lexical": 1, "vector": 1})
        fused.append(recall_at_k([i for i, _ in sorted(f.items(), key=lambda x: x[1], reverse=True)[:k]], gold, k))
        lex.append(recall_at_k([i for i, _ in bm.search(it.query, k)], gold, k))
    n = max(1, len(dense))
    return {"dense": round(sum(dense) / n, 3), "bm25+dense_rrf": round(sum(fused) / n, 3), "bm25": round(sum(lex) / n, 3), "n": n}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--locomo", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    base_tok = SpacyTokenVectors()
    base = SpacyVectorEmbedder()
    # training data
    sess = from_real(load_locomo(a.locomo, per_turn=False), seed=a.seed)
    turn = from_real(load_locomo(a.locomo, per_turn=True), seed=a.seed)
    sess_cal, sess_test = split(sess, 0.3, a.seed)
    turn_cal, turn_test = split(turn, 0.3, a.seed)
    syn_train = generate(1000, 200, seed=a.seed + 100)  # a different world from the evaluation one
    syn_test = generate(1000, 100, seed=a.seed)
    pairs = pairs_from(sess_cal) + pairs_from(turn_cal) + pairs_from(syn_train)
    corpus = [t for t, _ in sess.episodes] + [t for t, _ in turn.episodes] + [t for t, _ in syn_train.episodes]
    print(f"training pairs={len(pairs)} corpus={len(corpus)}", flush=True)
    adapted = train_adapter(base_tok, pairs, corpus, d_out=a.dim, epochs=a.epochs, seed=a.seed, log=lambda s: print(s, flush=True))
    print(f"trained in {time.perf_counter() - t0:.0f}s", flush=True)
    if a.out:
        adapted.save(a.out)
    report = {"pairs": len(pairs), "epochs": a.epochs, "dim": a.dim, "results": {}}
    for name, bench, k in (("locomo_session_test", sess_test, 5), ("locomo_turn_test", turn_test, 10), ("synthetic_1k", syn_test, 10)):
        before = evaluate(bench, base, k)
        after = evaluate(bench, adapted, k)
        report["results"][name] = {"k": k, "base": before, "adapted": after}
        print(f"{name:<22} K={k}  dense {before['dense']:.3f} -> {after['dense']:.3f}   bm25+dense {before['bm25+dense_rrf']:.3f} -> {after['bm25+dense_rrf']:.3f}   bm25 {before['bm25']:.3f}", flush=True)
    if a.out:
        json.dump(report, open(a.out.replace(".npz", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
