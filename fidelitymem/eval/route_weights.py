"""Fixed route-weight study: FidelityMem at raw storage (L0) with hand-set RRF weights
and route pools, including the semantic-codebook trie as a route, against BM25.

    python -m fidelitymem.eval.route_weights --path data/locomo/locomo10.json --session-level \
        --embedder onnx --seeds 0 1 2 --out docs/results/route_weights_locomo_session.json
"""
from __future__ import annotations

import argparse
import json
import time

from ..policy import MemoryPolicy, Mode
from .datasets import load
from .harness import Result, _ingest, _set_cfg, aggregate, format_table
from .real import _baseline_real, _measure_real
from .synthetic import from_real

CONFIGS: list[tuple[str, dict[str, float]]] = [
    ("fm:lexical", {"lexical": 1.0}),
    ("fm:lex1.2+vec1.0", {"lexical": 1.2, "vector": 1.0}),
    ("fm:lex1.0+vec1.0", {"lexical": 1.0, "vector": 1.0}),
    ("fm:lex1.2+vec0.5", {"lexical": 1.2, "vector": 0.5}),
    ("fm:lex1.2+vec1.0+strie1.2", {"lexical": 1.2, "vector": 1.0, "semantic_trie": 1.2}),
    ("fm:lex1.2+strie1.2", {"lexical": 1.2, "semantic_trie": 1.2}),
    ("fm:strie", {"semantic_trie": 1.0}),
    ("fm:lex1.2+vec1.0+trie1.2", {"lexical": 1.2, "vector": 1.0, "trie": 1.2}),
    ("fm:lex1.2+vec1.0+graph1.2", {"lexical": 1.2, "vector": 1.0, "graph": 1.2}),
    ("fm:all-planner-weights", {}),
]


def run_seed(real, seed: int, k: int, n_codes: int, configs=CONFIGS) -> list[Result]:
    bench = from_real(real, seed=seed)
    out = [_baseline_real("bm25", bench, k, "bm25"), _baseline_real("bm25+vector_rrf", bench, k, "hybrid")]
    pol = MemoryPolicy(minimum_recall_retention=0.95, mode=Mode.BALANCED, max_context_tokens=100_000)
    fm, ep_ids, ingest_s = _ingest(bench, pol)
    t0 = time.perf_counter()
    fm.enable_semantic_trie(n_codes=n_codes)
    print(f"[seed {seed}] ingest {ingest_s:.0f}s, semantic trie {time.perf_counter() - t0:.0f}s ({n_codes} codes)", flush=True)
    _set_cfg(fm, 0, k, True)
    base = fm.controller.current
    for name, weights in configs:
        routes = tuple(sorted(weights)) if weights else ("lexical", "vector", "graph", "trie", "temporal", "procedural", "semantic_trie")
        fm.controller.current = base.__class__(compression_level=0, top_k=k, expand_budget=base.expand_budget, routes=routes, gate_profile=base.gate_profile)
        fm.policy.route_weights = dict(weights)
        t0 = time.perf_counter()
        r = _measure_real(name, fm, bench, ep_ids, k)
        print(f"[seed {seed}] {name:<28} R@{k}={r.recall:.3f} CR={r.complete_recall:.3f} MRR={r.mrr:.3f} ({time.perf_counter() - t0:.0f}s)", flush=True)
        out.append(r)
    fm.policy.route_weights = {}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--dataset", default="locomo")
    ap.add_argument("--session-level", action="store_true")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--conversations", type=int, nargs="*", default=None)
    ap.add_argument("--codes", type=int, default=2000)
    ap.add_argument("--embedder", default="best")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from ..stores import set_default_embedder
    from ..stores.embedders import best_available, embedder_name

    emb = best_available(None if a.embedder == "best" else a.embedder)
    set_default_embedder(emb)
    kw = {}
    if a.dataset == "locomo":
        kw["per_turn"] = not a.session_level
        if a.conversations is not None:
            kw["conversations"] = a.conversations
    real = load(a.dataset, a.path, **kw)
    print(f"dataset={real.name} episodes={len(real.episodes)} questions={len(real.questions)} embedder={embedder_name(emb)} seeds={a.seeds}", flush=True)
    per_seed = [run_seed(real, sd, a.k, a.codes) for sd in a.seeds]
    agg = aggregate(per_seed) if len(per_seed) > 1 else per_seed[0]
    print(f"\n### route weights, {real.name}: mean over seeds {a.seeds} (± = 95% CI)\n{format_table(agg)}", flush=True)
    print("\nper-type Recall@K:")
    for r in agg:
        print(f"  {r.system:<28} " + ", ".join(f"{t}={v:.3f}" for t, v in sorted(r.per_type.items()) if t != "answer_containment"))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"dataset": real.name, "granularity": real.meta.get("granularity"), "embedder": embedder_name(emb), "seeds": a.seeds, "k": a.k, "codes": a.codes,
                       "configs": {n: w for n, w in CONFIGS},
                       "aggregate": [{k2: v for k2, v in r.__dict__.items() if k2 != "per_query_recall"} for r in agg],
                       "per_seed": [[{k2: v for k2, v in r.__dict__.items() if k2 != "per_query_recall"} for r in rs] for rs in per_seed]}, f, indent=1)


if __name__ == "__main__":
    main()
