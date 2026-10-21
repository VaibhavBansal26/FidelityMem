"""Does the recall-constrained controller pick the right retrieval regime from the
query log alone?

Three query regimes on LoCoMo sessions: literal questions, half-paraphrased and
fully paraphrased (WordNet synonyms, names/dates/numbers kept). For each regime the
controller is calibrated on a 30% calibration split of that regime's questions
(its query log) and scored on the other 70%. Fixed configurations are scored on the
same held-out split so the controller can be compared with the fixed-weight oracle.

    python -m fidelitymem.eval.controller_regimes --path data/locomo/locomo10.json --embedder onnx \
        --out docs/results/controller_regimes_locomo_session.json
"""
from __future__ import annotations

import argparse
import json
import random
import time

from ..controller import Config, ProbeQuery
from ..policy import MemoryPolicy, Mode
from .datasets import load
from .harness import _ingest, _set_cfg, format_table
from .paraphrase import paraphrase
from .real import _measure_real
from .synthetic import Benchmark, BenchItem, from_real

FIXED = [
    ("fixed:lexical", ("lexical",), 1.0),
    ("fixed:lex+vec(1.0)", ("lexical", "vector"), 1.0),
    ("fixed:lex+vec(0.5)", ("lexical", "vector"), 0.5),
    ("fixed:lex+vec+graph+temporal", ("lexical", "vector", "graph", "temporal"), 1.0),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--calib-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embedder", default="onnx")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from nltk.corpus import wordnet as wn

    from ..stores import set_default_embedder
    from ..stores.embedders import best_available, embedder_name

    emb = best_available(a.embedder)
    set_default_embedder(emb)
    real = load("locomo", a.path, per_turn=False)
    bench = from_real(real, seed=a.seed)
    pol = MemoryPolicy(minimum_recall_retention=0.95, mode=Mode.BALANCED, max_context_tokens=100_000)
    t0 = time.perf_counter()
    fm, ep_ids, ingest_s = _ingest(bench, pol)
    print(f"episodes={len(bench.episodes)} questions={len(bench.items)} embedder={embedder_name(emb)} ingest={ingest_s:.0f}s", flush=True)
    k = a.k
    results = {}
    for rate in a.rates:
        rng = random.Random(a.seed)
        items = list(bench.items) if rate == 0 else [BenchItem(paraphrase(it.query, rate, rng, wn)[0], it.gold, it.qtype, it.answer, it.gold_idx) for it in bench.items]
        random.Random(a.seed).shuffle(items)
        cut = int(len(items) * a.calib_frac)
        calib = Benchmark(bench.episodes, items[:cut], {}, name="calib")
        test = Benchmark(bench.episodes, items[cut:], {}, name=f"test-r{rate}")
        regime = "literal" if rate == 0 else f"paraphrased r={rate}"
        rows = []
        for name, routes, vw in FIXED:
            _set_cfg(fm, 0, k, True)
            fm.controller.current = Config(compression_level=0, top_k=k, expand_budget=fm.controller.current.expand_budget, routes=routes, gate_profile="content", vector_weight=vw)
            rows.append(_measure_real(name, fm, test, ep_ids, k))
        # controller: fresh search, dense weight in the space, top-k fixed at K
        _set_cfg(fm, 0, k, True)
        fm.controller.__init__(fm.policy)
        fm.controller.search_space["top_k"] = [k]
        fm.controller.search_space["expand_budget"] = [0, 2]
        fm.controller.search_space["vector_weight"] = [0.5, 1.0]
        probes = [ProbeQuery(it.query, [ep_ids[i] for i in calib.gold_indices(it) if i in ep_ids]) for it in calib.items if calib.gold_indices(it)]
        t1 = time.perf_counter()
        fm.optimise(probes=probes)
        cfg = fm.controller.current
        fm.controller.current = Config(compression_level=cfg.compression_level, top_k=k, expand_budget=cfg.expand_budget, routes=cfg.routes, gate_profile=cfg.gate_profile, vector_weight=cfg.vector_weight)
        fm.recompress_all(cfg.compression_level, cfg.gate_profile)
        r = _measure_real(f"controller(L{cfg.compression_level}/{'+'.join(x[:3] for x in cfg.routes)}/vw{cfg.vector_weight})", fm, test, ep_ids, k)
        rows.append(r)
        print(f"\n### {regime}: controller calibrated on {len(probes)} probes in {time.perf_counter() - t1:.0f}s -> {fm.controller.describe()}", flush=True)
        print(format_table(rows), flush=True)
        results[regime] = {"controller": fm.controller.describe(), "rows": [{k2: v for k2, v in x.__dict__.items() if k2 != "per_query_recall"} for x in rows]}
    print("\n### Summary: held-out Recall@K, controller vs best fixed")
    for regime, d in results.items():
        best = max(d["rows"][:-1], key=lambda x: x["recall"])
        ctrl = d["rows"][-1]
        print(f"  {regime:<22} controller={ctrl['recall']:.3f} ({ctrl['system']})  best fixed={best['recall']:.3f} ({best['system']})  lexical={d['rows'][0]['recall']:.3f}", flush=True)
    if a.out:
        json.dump({"k": k, "embedder": embedder_name(emb), "seed": a.seed, "regimes": results}, open(a.out, "w"), indent=1)
    print(f"total {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
