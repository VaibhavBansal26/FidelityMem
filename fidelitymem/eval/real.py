"""Real-dataset evaluation: every system on LoCoMo (or LongMemEval) with exact
evidence ids, a gate-on/gate-off comparison, several seeds with confidence
intervals, and an optional LLM reader + judge for answer accuracy.

Without an LLM credential the runner still reports retrieval recall, evidence
completeness, tokens, latency, storage and an *answer-containment* proxy: the
fraction of gold-answer content tokens present in the served context. It is a
proxy, not accuracy; it is reported as such.

    python -m fidelitymem.eval.real --dataset locomo --path data/locomo/locomo10.json \
        --k 10 --seeds 0 1 2 --questions 300 --out docs/results/locomo.json
"""
from __future__ import annotations

import argparse
import json
import time

from ..memory import FidelityMem
from ..policy import MemoryPolicy, Mode
from ..text import content_tokens, estimate_tokens
from .datasets import load
from .harness import Result, _baseline, _ingest, _set_cfg, _summarise, aggregate, format_table
from .synthetic import Benchmark, from_real


def _containment(answer: str, context: str) -> float:
    a = set(content_tokens(answer))
    if not a:
        return 1.0
    c = set(content_tokens(context))
    return len(a & c) / len(a)


def _measure_real(name: str, fm: FidelityMem, bench: Benchmark, ep_ids: dict[int, str], k: int, reader=None, judge=None) -> Result:
    runs, toks, lats, exp, contain, correct, n_judged = [], [], [], 0, [], 0, 0
    for item in bench.items:
        gold = [ep_ids.get(i, "missing") for i in bench.gold_indices(item)]
        t0 = time.perf_counter()
        res, st = fm.search(item.query, k=k)
        lats.append((time.perf_counter() - t0) * 1000)
        got = [r.memory.episode_id for r in res]
        toks.append(st.tokens)
        exp += st.expansions
        runs.append((got, gold, item.qtype))
        ctx = "\n".join(r.context_text for r in res)
        if item.qtype != "adversarial":
            contain.append(_containment(item.answer, ctx))
        if reader is not None and judge is not None:
            pred = reader.answer(item.query, ctx)
            correct += judge.judge(item.query, item.answer, pred)
            n_judged += 1
    r = _summarise(name, k, runs, toks, lats, sum(estimate_tokens(m.compressed_text) for m in fm.memories.values()), exp)
    r.per_type["answer_containment"] = sum(contain) / max(1, len(contain))
    if n_judged:
        r.per_type["answer_accuracy_llm"] = correct / n_judged
    return r


def _baseline_real(name: str, bench: Benchmark, k: int, mode: str) -> Result:
    r = _baseline(name, bench, k, mode)
    # containment proxy for baselines: recompute from their stored texts
    from ..stores import BM25Index, VectorIndex

    texts = [t for t, _ in bench.episodes]
    stored = texts
    if mode == "summary":
        from ..compression import ExtractiveCompressor

        c = ExtractiveCompressor()
        stored = [c.compress(t, 3) for t in texts]
    ids = [f"ep{i}" for i in range(len(texts))]
    bm = BM25Index()
    for i, t in zip(ids, stored):
        bm.add(i, t)
    vi = VectorIndex()
    if mode in ("vector", "hybrid"):
        vi.add_many(list(zip(ids, stored)))
    contain = []
    for item in bench.items:
        if item.qtype == "adversarial":
            continue
        if mode == "window":
            got = ids[-k:]
        elif mode == "vector":
            got = [i for i, _ in vi.search(item.query, k)]
        elif mode == "hybrid":
            from ..retrieval.fusion import rrf

            fused = rrf({"lexical": bm.search(item.query, k * 3), "vector": vi.search(item.query, k * 3)}, {"lexical": 1, "vector": 1})
            got = [i for i, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]]
        else:
            got = [i for i, _ in bm.search(item.query, k)]
        contain.append(_containment(item.answer, "\n".join(stored[int(i[2:])] for i in got)))
    r.per_type["answer_containment"] = sum(contain) / max(1, len(contain))
    return r


def run_once(bench: Benchmark, k: int, rho: float, reader=None, judge=None, levels=(1, 2, 3), with_controller: bool = True, calib: Benchmark | None = None, fast_controller: bool = False) -> list[Result]:
    """`calib` holds questions (with gold) the controller may calibrate on; they are never
    scored. `bench` is the test split."""
    from ..controller import ProbeQuery
    from .competitors import ALL as COMPETITORS

    results: list[Result] = []

    def timed(label, fn):
        t0 = time.perf_counter()
        r = fn()
        print(f"[timing] {label}: {time.perf_counter() - t0:.1f}s", flush=True)
        return r

    for name, mode in (("fixed_window", "window"), ("bm25", "bm25"), ("vector", "vector"), ("bm25+vector_rrf", "hybrid"), ("summary_L3+bm25", "summary")):
        results.append(timed(name, lambda: _baseline_real(name, bench, k, mode)))
    for fn in COMPETITORS:
        results.append(timed(fn.__name__, lambda: fn(bench, k)))
    pol = MemoryPolicy(minimum_recall_retention=rho, mode=Mode.BALANCED, max_context_tokens=100_000)
    fm, ep_ids, ingest_s = _ingest(bench, pol)
    if fast_controller:
        fm.controller.search_space["top_k"] = [k]
        fm.controller.search_space["expand_budget"] = [0, 2]
    _set_cfg(fm, 0, k, True)
    raw = timed("fidelitymem_L0_raw", lambda: _measure_real("fidelitymem_L0_raw", fm, bench, ep_ids, k, reader, judge))
    results.append(raw)
    for profile in ("facts", "content"):
        for lvl in levels:
            t0 = time.perf_counter()
            _set_cfg(fm, lvl, k, True, profile)
            print(f"[timing] recompress L{lvl}/{profile}: {time.perf_counter() - t0:.1f}s", flush=True)
            r = timed(f"fidelitymem_L{lvl}_gate:{profile}", lambda: _measure_real(f"fidelitymem_L{lvl}_gate:{profile}", fm, bench, ep_ids, k, reader, judge))
            r.retention = r.recall / raw.recall if raw.recall else 1.0
            results.append(r)
    # gate off: same levels, compressor never backs off
    fm.gate_enabled = False
    fm._level_cache.clear()
    for lvl in levels:
        _set_cfg(fm, lvl, k, True)
        r = _measure_real(f"fidelitymem_L{lvl}_nogate", fm, bench, ep_ids, k, reader, judge)
        r.retention = r.recall / raw.recall if raw.recall else 1.0
        results.append(r)
    fm.gate_enabled = True
    fm._level_cache.clear()
    if with_controller:
        variants = [("self", None)]
        if calib is not None and calib.items:
            variants.append(("qlog", [ProbeQuery(it.query, [ep_ids[i] for i in calib.gold_indices(it) if i in ep_ids]) for it in calib.items if calib.gold_indices(it)]))
        for src, probes in variants:
            _set_cfg(fm, 0, k, True)
            space = fm.controller.search_space
            fm.controller.__init__(fm.policy)  # fresh search per probe source
            fm.controller.search_space = space
            t0 = time.perf_counter()
            fm.optimise(n_probes=120, probes=probes)
            print(f"[timing] controller:{src}: {time.perf_counter() - t0:.1f}s", flush=True)
            cfg = fm.controller.current
            fm.controller.current = cfg.__class__(compression_level=cfg.compression_level, top_k=k, expand_budget=cfg.expand_budget, routes=cfg.routes, gate_profile=cfg.gate_profile)
            r = _measure_real(f"fidelitymem_ctrl:{src}(L{cfg.compression_level}/{cfg.gate_profile[:1]})", fm, bench, ep_ids, k, reader, judge)
            r.retention = r.recall / raw.recall if raw.recall else 1.0
            results.append(r)
            print(f"[controller:{src}] {fm.controller.describe()}", flush=True)
    print(f"[fidelitymem {bench.name}] ingest_s={ingest_s:.1f} memories={len(fm.memories)} backoffs={fm.telemetry.compression_backoffs} controller={fm.controller.describe()}", flush=True)
    return results


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="locomo")
    ap.add_argument("--path", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--questions", type=int, default=None, help="subsample questions (same subset every seed)")
    ap.add_argument("--conversations", type=int, nargs="*", default=None, help="LoCoMo conversation indices to include")
    ap.add_argument("--rho", type=float, default=0.95)
    ap.add_argument("--embedder", default="best")
    ap.add_argument("--llm", action="store_true", help="use Claude reader+judge for answer accuracy (needs credentials)")
    ap.add_argument("--no-controller", action="store_true")
    ap.add_argument("--session-level", action="store_true", help="LoCoMo: one episode per session instead of per turn")
    ap.add_argument("--calib-frac", type=float, default=0.3, help="fraction of questions held out as the controller's query log")
    ap.add_argument("--fast-controller", action="store_true", help="controller search with depth fixed at K and expansion in {0,2}")
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
    real = load(a.dataset, a.path, **kw).subset(a.questions, seed=0)
    reader = judge = None
    if a.llm:
        from ..llm import ClaudeJudge, ClaudeReader, available

        if available():
            reader, judge = ClaudeReader(), ClaudeJudge()
        else:
            print("LLM credentials not available; running without reader/judge", flush=True)
    print(f"dataset={real.name} episodes={len(real.episodes)} questions={len(real.questions)} embedder={embedder_name(emb)} seeds={a.seeds}", flush=True)
    per_seed = []
    for sd in a.seeds:
        bench = from_real(real, seed=sd)
        calib = None
        if a.calib_frac > 0:
            import random

            items = list(bench.items)
            random.Random(sd).shuffle(items)
            cut = int(len(items) * a.calib_frac)
            calib = Benchmark(bench.episodes, items[:cut], {}, name=bench.name + "-calib")
            bench = Benchmark(bench.episodes, items[cut:], {}, name=bench.name)
        per_seed.append(run_once(bench, a.k, a.rho, reader, judge, with_controller=not a.no_controller, calib=calib, fast_controller=a.fast_controller))
        print(f"\n### seed {sd}\n{format_table(per_seed[-1])}", flush=True)
    agg = aggregate(per_seed) if len(per_seed) > 1 else per_seed[0]
    print(f"\n### {real.name}: mean over seeds {a.seeds} (± = 95% CI)\n{format_table(agg)}", flush=True)
    print("\nper-type Recall@K and answer containment:")
    for r in agg:
        print(f"  {r.system:<26} " + ", ".join(f"{t}={v:.3f}" for t, v in sorted(r.per_type.items())))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"dataset": real.name, "granularity": real.meta.get("granularity"), "embedder": embedder_name(emb), "seeds": a.seeds, "k": a.k, "episodes": len(real.episodes), "questions": len(real.questions), "calib_frac": a.calib_frac,
                       "aggregate": [{k2: v for k2, v in r.__dict__.items() if k2 != "per_query_recall"} for r in agg],
                       "per_seed": [[{k2: v for k2, v in r.__dict__.items() if k2 != "per_query_recall"} for r in rs] for rs in per_seed]}, f, indent=1)


if __name__ == "__main__":
    main()
