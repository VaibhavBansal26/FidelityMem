"""Run baselines and FidelityMem on a synthetic benchmark and report metrics together."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..memory import FidelityMem
from ..policy import MemoryPolicy, Mode
from ..retrieval.planner import plan_for
from ..stores import BM25Index, VectorIndex
from ..text import estimate_tokens
from ..types import MemoryType
from .metrics import complete_set_recall, mrr, ndcg_at_k, percentile, recall_at_k
from .synthetic import Benchmark, generate


@dataclass
class Result:
    system: str
    k: int
    recall: float
    complete_recall: float
    mrr: float
    ndcg: float
    tokens_per_query: float
    p50_ms: float
    p95_ms: float
    storage_tokens: int
    expansion_rate: float = 0.0
    retention: float | None = None
    per_type: dict[str, float] = field(default_factory=dict)
    per_query_recall: list[float] = field(default_factory=list, repr=False)
    ci95: dict[str, float] = field(default_factory=dict)  # half-widths when aggregated over seeds / bootstrap

    def row(self) -> str:
        ret = f"{self.retention:.3f}" if self.retention is not None else "  -  "
        ci = f" ±{self.ci95['recall']:.3f}" if "recall" in self.ci95 else ""
        return f"{self.system:<28}{self.k:>3} {self.recall:>7.3f}{ci} {self.complete_recall:>7.3f} {self.mrr:>6.3f} {self.ndcg:>6.3f} {self.tokens_per_query:>8.0f} {self.p50_ms:>7.2f} {self.p95_ms:>7.2f} {self.storage_tokens:>9} {self.expansion_rate:>6.2f} {ret:>7}"


HEADER = f"{'system':<28}{'K':>3} {'R@K':>7} {'CR@K':>7} {'MRR':>6} {'nDCG':>6} {'tok/q':>8} {'p50ms':>7} {'p95ms':>7} {'storage':>9} {'expand':>6} {'retain':>7}"


def _summarise(system: str, k: int, runs: list[tuple[list[str], list[str], str]], toks: list[int], lats: list[float], storage: int, expansions: int) -> Result:
    scored = [(r, g, t) for r, g, t in runs if g]  # questions without gold evidence (adversarial) are excluded from recall
    scored = scored or runs
    rec = [recall_at_k(r, g, k) for r, g, _ in scored]
    cr = [complete_set_recall(r, g, k) for r, g, _ in scored]
    per_type: dict[str, list[float]] = {}
    for (r, g, t), v in zip(scored, rec):
        per_type.setdefault(t, []).append(v)
    res = Result(
        system, k, sum(rec) / len(rec), sum(cr) / len(cr), sum(mrr(r, g) for r, g, _ in scored) / len(scored),
        sum(ndcg_at_k(r, g, k) for r, g, _ in scored) / len(scored), sum(toks) / len(toks), percentile(lats, 0.5), percentile(lats, 0.95), storage,
        expansions / len(runs), None, {t: sum(v) / len(v) for t, v in per_type.items()},
    )
    res.per_query_recall = rec
    return res


def _baseline(name: str, bench: Benchmark, k: int, mode: str) -> Result:
    """mode: bm25 | vector | hybrid | window | summary"""
    ids = [f"ep{i}" for i in range(len(bench.episodes))]
    texts = [t for t, _ in bench.episodes]
    bm, vi = BM25Index(), VectorIndex()
    stored = texts
    if mode == "summary":
        # conversation summarisation baseline: a 3-level gist per episode (lossy)
        from ..compression import ExtractiveCompressor

        c = ExtractiveCompressor()
        stored = [c.compress(t, 3) for t in texts]
    for i, t in zip(ids, stored):
        bm.add(i, t)
    if mode in ("vector", "hybrid"):
        vi.add_many(list(zip(ids, stored)))
    runs, toks, lats = [], [], []
    for item in bench.items:
        gold = [f"ep{i}" for i in bench.gold_indices(item)]
        t0 = time.perf_counter()
        if mode == "window":
            # fixed sliding window = last k episodes in ingestion order
            got = ids[-k:][::-1]
        elif mode == "vector":
            got = [i for i, _ in vi.search(item.query, k)]
        elif mode == "hybrid":
            from ..retrieval.fusion import rrf

            fused = rrf({"lexical": bm.search(item.query, k * 3), "vector": vi.search(item.query, k * 3)}, {"lexical": 1, "vector": 1})
            got = [i for i, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]]
        else:
            got = [i for i, _ in bm.search(item.query, k)]
        lats.append((time.perf_counter() - t0) * 1000)
        toks.append(sum(estimate_tokens(stored[int(i[2:])]) for i in got))
        runs.append((got, gold, item.qtype))
    return _summarise(name, k, runs, toks, lats, sum(estimate_tokens(t) for t in stored), 0)


def _ingest(bench: Benchmark, policy: MemoryPolicy) -> tuple[FidelityMem, dict[int, str], float]:
    fm = FidelityMem(policy)
    t0 = time.perf_counter()
    ep_ids: dict[int, str] = {}
    for idx, (text, ts) in enumerate(bench.episodes):
        m = fm.remember(text, timestamp=ts, importance=0.6)
        if m:
            ep_ids[idx] = m.episode_id
    return fm, ep_ids, time.perf_counter() - t0


def _measure(name: str, fm: FidelityMem, bench: Benchmark, ep_ids: dict[int, str], k: int) -> Result:
    runs, toks, lats, exp = [], [], [], 0
    for item in bench.items:
        gold = [ep_ids.get(i, "missing") for i in bench.gold_indices(item)]
        t0 = time.perf_counter()
        res, st = fm.search(item.query, k=k, at=None)
        lats.append((time.perf_counter() - t0) * 1000)
        got = [r.memory.episode_id for r in res]
        toks.append(st.tokens)
        exp += st.expansions
        runs.append((got, gold, item.qtype))
    return _summarise(name, k, runs, toks, lats, sum(estimate_tokens(m.compressed_text) for m in fm.memories.values()), exp)


def _set_cfg(fm: FidelityMem, level: int, k: int, expansion: bool, profile: str = "facts") -> None:
    cur = fm.controller.current
    budget = fm.policy.mode_defaults()["expand_budget"]
    fm.controller.current = cur.__class__(compression_level=level, top_k=k, expand_budget=budget if expansion else 0, routes=cur.routes, gate_profile=profile)
    fm.recompress_all(level, profile)


def run_fidelitymem(bench: Benchmark, k: int, policy: MemoryPolicy, levels=(1, 2, 3), optimise: bool = True) -> tuple[list[Result], dict]:
    """One ingestion, then every compression level with and without expansion, then the controller."""
    fm, ep_ids, ingest_s = _ingest(bench, policy)
    out: list[Result] = []
    _set_cfg(fm, 0, k, True)
    raw = _measure("fidelitymem_L0_raw", fm, bench, ep_ids, k)
    out.append(raw)
    for lvl in levels:
        for expansion in (False, True):
            _set_cfg(fm, lvl, k, expansion)
            r = _measure(f"fidelitymem_L{lvl}_{'gated' if expansion else 'noexpand'}", fm, bench, ep_ids, k)
            r.retention = r.recall / raw.recall if raw.recall else 1.0
            out.append(r)
    info = {"ingest_s": ingest_s, "memories": len(fm.memories), "backoffs": fm.telemetry.compression_backoffs}
    if optimise:
        _set_cfg(fm, 0, k, True)
        t0 = time.perf_counter()
        meas = fm.optimise(n_probes=120)
        info["optimise_s"] = time.perf_counter() - t0
        info["controller"] = fm.controller.describe()
        info["controller_measurement"] = None if meas is None else {"retention": meas.retention, "cost": meas.cost(policy)}
        cfg = fm.controller.current
        fm.controller.current = cfg.__class__(compression_level=cfg.compression_level, top_k=k, expand_budget=cfg.expand_budget, routes=cfg.routes, gate_profile=cfg.gate_profile)
        r = _measure(f"fidelitymem_ctrl(L{cfg.compression_level}/{cfg.gate_profile[:1]})", fm, bench, ep_ids, k)
        r.retention = r.recall / raw.recall if raw.recall else 1.0
        out.append(r)
    return out, info


def run_suite(n_memories: int = 1000, n_queries: int = 100, ks: tuple[int, ...] = (5, 10), seed: int = 0, rho: float = 0.95, quick: bool = False) -> list[Result]:
    bench = generate(n_memories, n_queries, seed)
    results: list[Result] = []
    from .competitors import ALL as COMPETITORS

    for k in ks:
        results.append(_baseline("fixed_window", bench, k, "window"))
        results.append(_baseline("bm25", bench, k, "bm25"))
        results.append(_baseline("vector", bench, k, "vector"))
        results.append(_baseline("bm25+vector_rrf", bench, k, "hybrid"))
        results.append(_baseline("summary_L3+bm25", bench, k, "summary"))
        for fn in COMPETITORS:
            results.append(fn(bench, k))
        pol = MemoryPolicy(minimum_recall_retention=rho, mode=Mode.BALANCED, max_context_tokens=100_000)
        res, info = run_fidelitymem(bench, k, pol, levels=((2,) if quick else (1, 2, 3)))
        results.extend(res)
        print(f"[fidelitymem n={n_memories} k={k}] {info}", flush=True)
    return results


def aggregate(runs: list[list[Result]]) -> list[Result]:
    """Mean over seeds per system with 95% CI half-widths (t-interval over seeds;
    bootstrap over questions when only one seed)."""
    import math
    import random

    by: dict[str, list[Result]] = {}
    order: list[str] = []
    for rs in runs:
        for r in rs:
            key = r.system.split("(")[0]
            if key not in by:
                by[key] = []
                order.append(key)
            by[key].append(r)
    out: list[Result] = []
    T95 = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
    for key in order:
        rs = by[key]
        n = len(rs)
        agg = Result(rs[0].system if n == 1 else key + ("" if "(" not in rs[0].system else "(" + "/".join(sorted({r.system.split("(")[1].rstrip(")") for r in rs})) + ")"), rs[0].k, 0, 0, 0, 0, 0, 0, 0, 0)
        ci: dict[str, float] = {}
        for m in ("recall", "complete_recall", "mrr", "ndcg", "tokens_per_query", "p50_ms", "p95_ms", "storage_tokens", "expansion_rate"):
            vals = [getattr(r, m) for r in rs]
            mean = sum(vals) / n
            setattr(agg, m, mean if m != "storage_tokens" else int(mean))
            if n > 1:
                sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
                ci[m] = T95.get(n - 1, 1.96) * sd / math.sqrt(n)
            elif m == "recall" and rs[0].per_query_recall:
                rng = random.Random(0)
                pq = rs[0].per_query_recall
                boots = sorted(sum(rng.choice(pq) for _ in pq) / len(pq) for _ in range(1000))
                ci[m] = (boots[974] - boots[25]) / 2
        rets = [r.retention for r in rs if r.retention is not None]
        agg.retention = sum(rets) / len(rets) if rets else None
        types = {t for r in rs for t in r.per_type}
        agg.per_type = {t: sum(r.per_type.get(t, 0.0) for r in rs) / n for t in types}
        agg.ci95 = ci
        out.append(agg)
    return out


def run_scale(levels=(300, 1000, 3000, 10000), n_queries: int = 100, k: int = 10, seed: int = 0, rho: float = 0.95, seeds: tuple[int, ...] | None = None) -> dict:
    """Same question families and distractor generator at increasing record counts.
    With several seeds, each level reports the mean over seeds with 95% CI half-widths."""
    report = {}
    seeds = seeds or (seed,)
    for n in levels:
        per_seed = [run_suite(n, n_queries, (k,), sd, rho, quick=False) for sd in seeds]
        results = aggregate(per_seed) if len(per_seed) > 1 else per_seed[0]
        report[n] = [{k2: v for k2, v in r.__dict__.items() if k2 != "per_query_recall"} for r in results]
        print(f"\n### {n} memories, K={k}, seeds={list(seeds)}\n{format_table(results)}", flush=True)
    return report


def format_table(results: list[Result]) -> str:
    lines = [HEADER]
    lines += [r.row() for r in results]
    return "\n".join(lines)
