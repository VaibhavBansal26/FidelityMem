"""LongMemEval (Wu et al. 2024) under its standard protocol: every question is
answered against *its own* haystack (about 48 sessions, ~115k tokens), and the
retrieval target is the question's `answer_session_ids`.

Every system is rebuilt per question. FidelityMem's controller is calibrated once,
on a separate set of calibration questions (their haystacks pooled into one store,
their questions used as the query log), and the chosen configuration is then
applied to every test question unchanged. Calibration questions are never scored.

    python -m fidelitymem.eval.longmemeval --path data/longmemeval/longmemeval_s_cleaned.json \
        --k 5 --questions 150 --calib 12 --embedder onnx --out docs/results/longmemeval_s.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from datetime import datetime, timezone

from ..memory import FidelityMem
from ..policy import MemoryPolicy, Mode
from .datasets import RealBench, RealQ
from .harness import Result, _ingest, _set_cfg
from .real import _baseline_real, _measure_real
from .synthetic import from_real

BASELINES = (("bm25", "bm25"), ("vector", "vector"), ("bm25+vector_rrf", "hybrid"), ("summary_L3+bm25", "summary"))
COMPETITOR_NAMES = ("window_trie", "mem0_style", "zep_style", "hipporag_style")
METRICS = ("recall", "complete_recall", "mrr", "ndcg", "tokens_per_query", "p50_ms", "p95_ms", "storage_tokens", "expansion_rate")


def _session_episode(sid: str, sess: list[dict], date: str) -> tuple[str, str, datetime]:
    try:
        ts = datetime.strptime(date.split(" (")[0], "%Y/%m/%d").replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001
        ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
    return sid, f"[{date}] " + " ".join(f"{t['role']}: {t['content']}" for t in sess), ts


def question_bench(item: dict) -> RealBench:
    eps = [_session_episode(sid, s, d) for sid, s, d in zip(item["haystack_session_ids"], item["haystack_sessions"], item["haystack_dates"])]
    q = RealQ(item["question_id"], item["question"], str(item["answer"]), list(item["answer_session_ids"]), item["question_type"])
    return RealBench("longmemeval", eps, [q], {})


def pooled_bench(items: list[dict]) -> RealBench:
    eps: dict[str, tuple] = {}
    qs = []
    for it in items:
        for sid, s, d in zip(it["haystack_session_ids"], it["haystack_sessions"], it["haystack_dates"]):
            eps.setdefault(sid, _session_episode(sid, s, d))
        qs.append(RealQ(it["question_id"], it["question"], str(it["answer"]), list(it["answer_session_ids"]), it["question_type"]))
    return RealBench("longmemeval-calib", list(eps.values()), qs, {})


def stratified(data: list[dict], n: int, seed: int, exclude: set[str] = frozenset()) -> list[dict]:
    """Equal share per question type (remainder to the largest types), fixed seed."""
    by = defaultdict(list)
    for it in data:
        if it["question_id"] not in exclude:
            by[it["question_type"]].append(it)
    rng = random.Random(seed)
    for v in by.values():
        rng.shuffle(v)
    types = sorted(by, key=lambda t: -len(by[t]))
    share = {t: min(len(by[t]), n // len(types)) for t in types}
    left = n - sum(share.values())
    for t in types:
        add = min(left, len(by[t]) - share[t])
        share[t] += add
        left -= add
    return [it for t in types for it in by[t][: share[t]]]


class Acc:
    """Accumulates one-question Results per system and reports means with bootstrap CIs."""

    def __init__(self):
        self.rows: dict[str, list[Result]] = defaultdict(list)
        self.order: list[str] = []
        self.types: dict[str, list[str]] = defaultdict(list)

    def add(self, r: Result, qtype: str) -> None:
        if r.system not in self.rows:
            self.order.append(r.system)
        self.rows[r.system].append(r)
        self.types[r.system].append(qtype)

    def summary(self, raw_system: str = "fidelitymem_L0_raw") -> list[Result]:
        out = []
        raw_rec = None
        for s in self.order:
            rs = self.rows[s]
            n = len(rs)
            agg = Result(s, rs[0].k, 0, 0, 0, 0, 0, 0, 0, 0)
            for m in METRICS:
                vals = [getattr(r, m) for r in rs]
                setattr(agg, m, (sum(vals) / n) if m != "storage_tokens" else int(sum(vals) / n))
            # bootstrap over questions
            rng = random.Random(0)
            rec = [r.recall for r in rs]
            boots = sorted(sum(rng.choice(rec) for _ in range(n)) / n for _ in range(1000))
            agg.ci95 = {"recall": (boots[974] - boots[24]) / 2}
            per_type = defaultdict(list)
            for r, t in zip(rs, self.types[s]):
                per_type[t].append(r.recall)
            agg.per_type = {t: sum(v) / len(v) for t, v in per_type.items()}
            agg.per_query_recall = rec
            if s == raw_system:
                raw_rec = agg.recall
            if s.startswith("fidelitymem") and raw_rec:
                agg.retention = agg.recall / raw_rec
            out.append(agg)
        return out


def calibrate(items: list[dict], k: int, rho: float, fast: bool = True):
    """Pool the calibration haystacks, run the controller with the calibration questions as
    the query log, return the chosen Config."""
    from ..controller import ProbeQuery

    real = pooled_bench(items)
    bench = from_real(real, seed=0)
    pol = MemoryPolicy(minimum_recall_retention=rho, mode=Mode.BALANCED, max_context_tokens=100_000)
    t0 = time.perf_counter()
    fm, ep_ids, ingest_s = _ingest(bench, pol)
    if fast:
        fm.controller.search_space["top_k"] = [k]
        fm.controller.search_space["expand_budget"] = [0, 2]
    _set_cfg(fm, 0, k, True)
    probes = [ProbeQuery(it.query, [ep_ids[i] for i in bench.gold_indices(it) if i in ep_ids]) for it in bench.items if bench.gold_indices(it)]
    fm.optimise(probes=probes)
    cfg = fm.controller.current
    print(f"[calibration] sessions={len(bench.episodes)} probes={len(probes)} ingest={ingest_s:.0f}s total={time.perf_counter() - t0:.0f}s -> {fm.controller.describe()}", flush=True)
    return cfg


def run(data: list[dict], k: int, rho: float, cfg=None, seed: int = 0, levels=(2, 3), log=print) -> Acc:
    from .competitors import ALL as COMPETITORS

    comp = [fn for fn in COMPETITORS if fn.__name__ in COMPETITOR_NAMES]
    acc = Acc()
    pol = MemoryPolicy(minimum_recall_retention=rho, mode=Mode.BALANCED, max_context_tokens=100_000)
    t_start = time.perf_counter()
    for qi, item in enumerate(data):
        bench = from_real(question_bench(item), seed=seed)
        qt = item["question_type"]
        for name, mode in BASELINES:
            acc.add(_baseline_real(name, bench, k, mode), qt)
        for fn in comp:
            acc.add(fn(bench, k), qt)
        fm, ep_ids, _ = _ingest(bench, pol)
        _set_cfg(fm, 0, k, True)
        acc.add(_measure_real("fidelitymem_L0_raw", fm, bench, ep_ids, k), qt)
        for lvl in levels:
            _set_cfg(fm, lvl, k, True, "content")
            acc.add(_measure_real(f"fidelitymem_L{lvl}_gate:content", fm, bench, ep_ids, k), qt)
        fm.gate_enabled = False
        fm._level_cache.clear()
        _set_cfg(fm, 3, k, True)
        acc.add(_measure_real("fidelitymem_L3_nogate", fm, bench, ep_ids, k), qt)
        fm.gate_enabled = True
        fm._level_cache.clear()
        if cfg is not None:
            fm.controller.current = cfg.__class__(compression_level=cfg.compression_level, top_k=k, expand_budget=cfg.expand_budget, routes=cfg.routes, gate_profile=cfg.gate_profile)
            fm.recompress_all(cfg.compression_level, cfg.gate_profile)
            acc.add(_measure_real(f"fidelitymem_ctrl:qlog(L{cfg.compression_level}/{cfg.gate_profile[:1]})", fm, bench, ep_ids, k), qt)
        if (qi + 1) % 5 == 0 or qi == 0:
            el = time.perf_counter() - t_start
            cur = {s: round(sum(r.recall for r in rs) / len(rs), 3) for s, rs in acc.rows.items() if s in ("bm25", "bm25+vector_rrf", "fidelitymem_L0_raw") or s.startswith("fidelitymem_ctrl")}
            log(f"[{qi + 1}/{len(data)}] {el / (qi + 1):.0f}s/question, running Recall@{k}: {cur}")
    return acc


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--questions", type=int, default=150)
    ap.add_argument("--calib", type=int, default=12, help="calibration questions for the controller (never scored)")
    ap.add_argument("--rho", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, nargs=2, default=None, metavar=("I", "N"), help="run test questions i::N only (merge shards with eval.merge)")
    ap.add_argument("--embedder", default="best")
    ap.add_argument("--no-controller", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    from ..stores import set_default_embedder
    from ..stores.embedders import CachingEmbedder, best_available, embedder_name
    from .harness import format_table

    emb = CachingEmbedder(best_available(None if a.embedder == "best" else a.embedder))
    set_default_embedder(emb)
    data = json.load(open(a.path))
    calib_items = stratified(data, a.calib, seed=a.seed + 1000) if a.calib and not a.no_controller else []
    test_items = stratified(data, a.questions, seed=a.seed, exclude={it["question_id"] for it in calib_items})
    if a.shard:
        i, n = a.shard
        test_items = test_items[i::n]
    print(f"dataset=longmemeval questions={len(test_items)} calib={len(calib_items)} k={a.k} embedder={embedder_name(emb)} sessions/question={sum(len(it['haystack_sessions']) for it in test_items) / max(1, len(test_items)):.1f}", flush=True)
    cfg = calibrate(calib_items, a.k, a.rho) if calib_items else None
    acc = run(test_items, a.k, a.rho, cfg, seed=a.seed, log=lambda s: print(s, flush=True))
    agg = acc.summary()
    print(f"\n### longmemeval-s ({len(test_items)} questions, Recall@{a.k} against gold sessions, ± = bootstrap 95% CI)\n{format_table(agg)}", flush=True)
    print("\nper-type Recall@K:")
    for r in agg:
        print(f"  {r.system:<34} " + ", ".join(f"{t}={v:.3f}" for t, v in sorted(r.per_type.items())))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"dataset": "longmemeval", "protocol": "per-question haystack", "embedder": embedder_name(emb), "k": a.k, "questions": len(test_items), "calib": len(calib_items),
                       "controller": None if cfg is None else cfg.__dict__ | {"routes": list(cfg.routes)}, "question_ids": [it["question_id"] for it in test_items],
                       "aggregate": [{k2: v for k2, v in r.__dict__.items() if k2 != "per_query_recall"} for r in agg],
                       "per_question": {s: [r.recall for r in rs] for s, rs in acc.rows.items()}, "types": {s: t for s, t in acc.types.items()}}, f, indent=1)


if __name__ == "__main__":
    main()
