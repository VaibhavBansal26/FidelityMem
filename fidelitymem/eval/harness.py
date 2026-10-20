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

    def row(self) -> str:
        ret = f"{self.retention:.3f}" if self.retention is not None else "  -  "
        return f"{self.system:<28}{self.k:>3} {self.recall:>7.3f} {self.complete_recall:>7.3f} {self.mrr:>6.3f} {self.ndcg:>6.3f} {self.tokens_per_query:>8.0f} {self.p50_ms:>7.2f} {self.p95_ms:>7.2f} {self.storage_tokens:>9} {self.expansion_rate:>6.2f} {ret:>7}"


HEADER = f"{'system':<28}{'K':>3} {'R@K':>7} {'CR@K':>7} {'MRR':>6} {'nDCG':>6} {'tok/q':>8} {'p50ms':>7} {'p95ms':>7} {'storage':>9} {'expand':>6} {'retain':>7}"


def _summarise(system: str, k: int, runs: list[tuple[list[str], list[str], str]], toks: list[int], lats: list[float], storage: int, expansions: int) -> Result:
    rec = [recall_at_k(r, g, k) for r, g, _ in runs]
    cr = [complete_set_recall(r, g, k) for r, g, _ in runs]
    per_type: dict[str, list[float]] = {}
    for (r, g, t), v in zip(runs, rec):
        per_type.setdefault(t, []).append(v)
    return Result(
        system, k, sum(rec) / len(rec), sum(cr) / len(cr), sum(mrr(r, g) for r, g, _ in runs) / len(runs),
        sum(ndcg_at_k(r, g, k) for r, g, _ in runs) / len(runs), sum(toks) / len(toks), percentile(lats, 0.5), percentile(lats, 0.95), storage,
        expansions / len(runs), None, {t: sum(v) / len(v) for t, v in per_type.items()},
    )


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
        gold = [f"ep{bench.text_to_idx[g]}" for g in item.gold]
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


def _fidelitymem(name: str, bench: Benchmark, k: int, policy: MemoryPolicy, level: int | None, optimise: bool, expansion: bool = True) -> tuple[Result, FidelityMem]:
    fm = FidelityMem(policy)
    if level is not None:
        fm.controller.current = fm.controller.current.__class__(compression_level=level, top_k=k, expand_budget=fm.controller.current.expand_budget if expansion else 0, routes=fm.controller.current.routes)
    ep_ids: dict[int, str] = {}
    for idx, (text, ts) in enumerate(bench.episodes):
        m = fm.remember(text, timestamp=ts, importance=0.6)
        if m:
            ep_ids[idx] = m.episode_id
    if optimise:
        fm.optimise(n_probes=40)
    runs, toks, lats, exp = [], [], [], 0
    cfg = fm.controller.current
    for item in bench.items:
        gold = [ep_ids.get(bench.text_to_idx[g], "missing") for g in item.gold]
        t0 = time.perf_counter()
        res, st = fm.search(item.query, k=k, at=None)
        lats.append((time.perf_counter() - t0) * 1000)
        got = [r.memory.episode_id for r in res]
        toks.append(st.tokens)
        exp += st.expansions
        runs.append((got, gold, item.qtype))
    r = _summarise(name, k, runs, toks, lats, sum(estimate_tokens(m.compressed_text) for m in fm.memories.values()), exp)
    return r, fm


def run_suite(n_memories: int = 1000, n_queries: int = 100, ks: tuple[int, ...] = (5, 10), seed: int = 0, rho: float = 0.95, quick: bool = False) -> list[Result]:
    bench = generate(n_memories, n_queries, seed)
    results: list[Result] = []
    for k in ks:
        results.append(_baseline("fixed_window", bench, k, "window"))
        results.append(_baseline("bm25", bench, k, "bm25"))
        results.append(_baseline("vector", bench, k, "vector"))
        results.append(_baseline("bm25+vector_rrf", bench, k, "hybrid"))
        results.append(_baseline("summary_L3+bm25", bench, k, "summary"))
        from .competitors import ALL as COMPETITORS

        for fn in COMPETITORS:
            results.append(fn(bench, k))
        pol = MemoryPolicy(minimum_recall_retention=rho, mode=Mode.BALANCED, max_context_tokens=100_000)
        raw, _ = _fidelitymem("fidelitymem_L0_raw", bench, k, pol, 0, False)
        results.append(raw)
        for lvl in ([2] if quick else [1, 2, 3]):
            r, _ = _fidelitymem(f"fidelitymem_L{lvl}_noexpand", bench, k, pol, lvl, False, expansion=False)
            r.retention = r.recall / raw.recall if raw.recall else 1.0
            results.append(r)
            r2, _ = _fidelitymem(f"fidelitymem_L{lvl}_gated", bench, k, pol, lvl, False)
            r2.retention = r2.recall / raw.recall if raw.recall else 1.0
            results.append(r2)
        opt, fm = _fidelitymem("fidelitymem_controller", bench, k, pol, None, True)
        opt.retention = opt.recall / raw.recall if raw.recall else 1.0
        opt.system = f"fidelitymem_ctrl(L{fm.controller.current.compression_level})"
        results.append(opt)
    return results


def format_table(results: list[Result]) -> str:
    lines = [HEADER]
    lines += [r.row() for r in results]
    return "\n".join(lines)
