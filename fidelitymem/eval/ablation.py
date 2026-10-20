"""Ablation at a fixed record count: remove one component at a time from the L3 gated system."""
from __future__ import annotations

import json
import sys

from ..policy import MemoryPolicy, Mode
from .harness import _ingest, _measure, _set_cfg, format_table
from .synthetic import generate


def run(n: int = 1000, n_queries: int = 100, k: int = 10, seed: int = 0, level: int = 3):
    bench = generate(n, n_queries, seed)
    pol = MemoryPolicy(minimum_recall_retention=0.95, mode=Mode.BALANCED, max_context_tokens=100_000)
    results = []

    def variant(name, gate=True, chaining=True, routes=None, expansion=True, hub=None, profile="facts"):
        from ..stores.graph import GraphStore

        old_hub = GraphStore.hub_degree
        if hub is not None:
            GraphStore.hub_degree = hub
        fm, ep_ids, _ = _ingest(bench, pol) if gate else _ingest_nogate(bench, pol)
        fm.retriever.enable_chaining = chaining
        _set_cfg(fm, level, k, expansion, profile)
        if routes is not None:
            c = fm.controller.current
            fm.controller.current = c.__class__(compression_level=c.compression_level, top_k=c.top_k, expand_budget=c.expand_budget, routes=tuple(routes))
        r = _measure(name, fm, bench, ep_ids, k)
        GraphStore.hub_degree = old_hub
        results.append(r)
        return r

    def _ingest_nogate(bench, pol):
        from ..memory import FidelityMem
        import time

        fm = FidelityMem(pol)
        fm.gate_enabled = False
        ep_ids = {}
        for idx, (text, ts) in enumerate(bench.episodes):
            m = fm.remember(text, timestamp=ts, importance=0.6)
            if m:
                ep_ids[idx] = m.episode_id
        return fm, ep_ids, 0.0

    full = variant(f"full_L{level}_gated")
    variant("- fidelity gate", gate=False)
    variant("+ content gate profile", profile="content")
    variant("- raw expansion", expansion=False)
    variant("- relation chaining", chaining=False)
    variant("- graph route", routes=("lexical", "vector", "temporal"))
    variant("- graph & chaining", chaining=False, routes=("lexical", "vector", "temporal"))
    variant("- hub penalty/pruning", hub=10**9)
    variant("- trie route", routes=("lexical", "vector", "graph", "temporal"))
    variant("lexical+vector only", routes=("lexical", "vector"))
    for r in results:
        r.retention = r.recall / full.recall if full.recall else 1.0
    return results


if __name__ == "__main__":
    import argparse

    from ..stores import set_default_embedder
    from ..stores.embedders import best_available, embedder_name
    from .harness import aggregate

    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?", default=1000)
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--embedder", default="best")
    a = ap.parse_args()
    emb = best_available(None if a.embedder == "best" else a.embedder)
    set_default_embedder(emb)
    per_seed = [run(a.n, seed=sd) for sd in a.seeds]
    res = aggregate(per_seed) if len(per_seed) > 1 else per_seed[0]
    print(f"ablation n={a.n} K=10 seeds={a.seeds} embedder={embedder_name(emb)} (retain = recall relative to full; ± = 95% CI)")
    print(format_table(res))
    print("\nper-type Recall@10:")
    for r in res:
        print(f"  {r.system:<24} " + ", ".join(f"{t}={v:.3f}" for t, v in sorted(r.per_type.items())))
    if a.out:
        json.dump({"embedder": embedder_name(emb), "seeds": a.seeds, "n": a.n, "aggregate": [{k: v for k, v in r.__dict__.items() if k != "per_query_recall"} for r in res]}, open(a.out, "w"), indent=1)
