"""Dry run of hub-and-spoke reranking on LoCoMo (turn and session granularity).

    python -m fidelitymem.eval.hubspoke_experiment --path data/locomo/locomo10.json --conversations 0
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

from ..retrieval.fusion import rrf
from ..retrieval.hubspoke import Spoke, hub_spoke_rerank
from ..stores import BM25Index, VectorIndex, set_default_embedder
from ..text import estimate_tokens, extract_entities
from .competitors import _run
from .datasets import load
from .harness import format_table
from .synthetic import from_real


def run(real, k: int, pool: int, alphas, betas, per_turn: bool, log=print):
    bench = from_real(real, seed=0)
    texts = [t for t, _ in bench.episodes]
    ids = [f"ep{i}" for i in range(len(texts))]
    # map bench index -> original episode id (shuffled by from_real); recover session/day from the text prefix and timestamp
    eid_of = {}
    orig = {t: (e, ts) for e, t, ts in real.episodes}
    for i, (t, ts) in enumerate(bench.episodes):
        eid_of[ids[i]] = orig[t][0]
    spokes = {}
    for i, t in zip(ids, texts):
        eid, ts = orig[t]
        parts = eid.split(":")  # turn: sid:D3:12  session: sid:session_3
        session = (parts[1] if per_turn else parts[0]) if len(parts) >= 2 else None
        day = ts.strftime("%Y-%m-%d") if isinstance(ts, datetime) else None
        spokes[i] = Spoke(i, extract_entities(t)[:12], session, day)
    bm, vi = BM25Index(), VectorIndex()
    for i, t in zip(ids, texts):
        bm.add(i, t)
    t0 = time.perf_counter()
    vi.add_many(list(zip(ids, texts)))
    log(f"embedded {len(texts)} episodes in {time.perf_counter() - t0:.0f}s")
    stored = sum(estimate_tokens(t) for t in texts)

    def fused(q, wl=1.0, wv=1.0, n=pool):
        f = rrf({"lexical": bm.search(q, n), "vector": vi.search(q, n)}, {"lexical": wl, "vector": wv})
        return sorted(f.items(), key=lambda x: x[1], reverse=True)[:n]

    systems = {
        "bm25": lambda q: [i for i, _ in bm.search(q, k)],
        "rrf lex+dense": lambda q: [i for i, _ in fused(q)[:k]],
        "rrf lex1.2+dense0.5": lambda q: [i for i, _ in fused(q, 1.2, 0.5)[:k]],
    }
    for a in alphas:
        for b in betas:
            systems[f"bm25+hub(a{a},b{b})"] = (lambda q, a=a, b=b: [i for i, _ in hub_spoke_rerank(bm.search(q, pool), spokes, alpha=a, beta=b)[:k]])
            systems[f"rrf+hub(a{a},b{b})"] = (lambda q, a=a, b=b: [i for i, _ in hub_spoke_rerank(fused(q), spokes, alpha=a, beta=b)[:k]])
    # hub-type ablation at the middle setting
    a, b = alphas[len(alphas) // 2], betas[len(betas) // 2]
    systems[f"rrf+hub entities only"] = lambda q: [i for i, _ in hub_spoke_rerank(fused(q), spokes, alpha=a, beta=b, use_sessions=False, use_days=False)[:k]]
    systems[f"rrf+hub sessions only"] = lambda q: [i for i, _ in hub_spoke_rerank(fused(q), spokes, alpha=a, beta=b, use_entities=False, use_days=False)[:k]]
    systems[f"rrf+hub days only"] = lambda q: [i for i, _ in hub_spoke_rerank(fused(q), spokes, alpha=a, beta=b, use_entities=False, use_sessions=False)[:k]]
    rows = []
    for name, fn in systems.items():
        t0 = time.perf_counter()
        r = _run(name, bench, k, texts, stored, fn)
        rows.append(r)
        log(f"  {name:<28} R@{k}={r.recall:.3f} MRR={r.mrr:.3f} nDCG={r.ndcg:.3f} p95={r.p95_ms:.1f}ms  " + " ".join(f"{t}={v:.2f}" for t, v in sorted(r.per_type.items())))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--conversations", type=int, nargs="*", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pool", type=int, default=50)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.3])
    ap.add_argument("--embedder", default="onnx")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from ..stores.embedders import best_available

    set_default_embedder(best_available(a.embedder))
    out = {}
    for gran, per_turn in (("turns", True), ("sessions", False)):
        kw = {"per_turn": per_turn}
        if a.conversations is not None:
            kw["conversations"] = a.conversations
        real = load("locomo", a.path, **kw)
        print(f"\n### LoCoMo {gran}: episodes={len(real.episodes)} questions={len(real.questions)} pool={a.pool} K={a.k}", flush=True)
        rows = run(real, a.k, a.pool, a.alphas, a.betas, per_turn, log=lambda s: print(s, flush=True))
        print(format_table(rows), flush=True)
        out[gran] = [{k2: v for k2, v in r.__dict__.items() if k2 != "per_query_recall"} for r in rows]
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
