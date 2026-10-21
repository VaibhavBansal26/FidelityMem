"""Paraphrase-robustness test. The same LoCoMo questions are rewritten with WordNet
synonyms (names, numbers, dates and stop words untouched) and every retrieval
strategy is scored on the original and on the paraphrased questions against the
same store. The drop from original to paraphrased is what each strategy loses when
the asker does not reuse the evidence's words.

    python -m fidelitymem.eval.paraphrase --path data/locomo/locomo10.json --embedder onnx \
        --out docs/results/paraphrase_locomo_session.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time

from ..retrieval.fusion import rrf
from ..stores import BM25Index, VectorIndex
from ..stores.lexicon import SemanticLexicon
from ..text import STOPWORDS, extract_dates, extract_entities, extract_numbers, stem
from .competitors import _run
from .datasets import load
from .harness import format_table
from .synthetic import Benchmark, BenchItem, from_real

_TOK = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d[\w./:-]*|[^\sA-Za-z\d]")


def paraphrase(question: str, rate: float, rng: random.Random, wn) -> tuple[str, int, int]:
    """Replace eligible content words by a WordNet synonym with probability `rate`.
    Returns (text, eligible words, replaced words)."""
    protected = {w.lower() for e in extract_entities(question) for w in e.split()}
    protected |= {w.lower() for d in extract_dates(question) for w in d.split()} | {n.lower() for n in extract_numbers(question)}
    toks = _TOK.findall(question)
    out, eligible, replaced = [], 0, 0
    for i, t in enumerate(toks):
        low = t.lower()
        ok = t.isalpha() and len(t) > 3 and low not in STOPWORDS and low not in protected and not (t[0].isupper() and i > 0)
        if not ok:
            out.append(t)
            continue
        eligible += 1
        cands = []
        for s in wn.synsets(low):
            for l in s.lemmas():
                name = l.name().lower()
                if "_" in name or name == low or stem(name) == stem(low) or not name.isalpha():
                    continue
                if name not in cands:
                    cands.append(name)
        if cands and rng.random() < rate:
            new = cands[0]
            out.append(new.capitalize() if t[0].isupper() else new)
            replaced += 1
        else:
            out.append(t)
    text = ""
    for t in out:
        text += t if (not text or re.match(r"[^\w]", t)) else " " + t
    return text, eligible, replaced


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rates", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--embedder", default="onnx")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from nltk.corpus import wordnet as wn

    from ..stores import set_default_embedder
    from ..stores.embedders import SpacyVectorEmbedder, best_available, embedder_name

    emb = best_available(a.embedder)
    set_default_embedder(emb)
    real = load("locomo", a.path, per_turn=False)
    bench = from_real(real, seed=0)
    texts = [t for t, _ in bench.episodes]
    ids = [f"ep{i}" for i in range(len(texts))]
    bm, vi = BM25Index(), VectorIndex()
    for i, t in zip(ids, texts):
        bm.add(i, t)
    t0 = time.perf_counter()
    vi.add_many(list(zip(ids, texts)))
    print(f"episodes={len(texts)} questions={len(bench.items)} embedder={embedder_name(emb)} embed={time.perf_counter() - t0:.0f}s", flush=True)
    lex = SemanticLexicon(SpacyVectorEmbedder(idf_weighted=False), neighbours=3, min_sim=0.55).fit(texts)
    k = a.k

    def fused(q, w_lex, w_vec):
        f = rrf({"lexical": bm.search(q, k * 3), "vector": vi.search(q, k * 3)}, {"lexical": w_lex, "vector": w_vec})
        return [i for i, _ in sorted(f.items(), key=lambda x: x[1], reverse=True)[:k]]

    systems = {
        "bm25": lambda q: [i for i, _ in bm.search(q, k)],
        "bm25+lexicon(m3,a0.4)": lambda q: [i for i, _ in bm.search_weighted(lex.expand(q, 0.4), k)],
        "dense": lambda q: [i for i, _ in vi.search(q, k)],
        "rrf lex1.0+vec1.0": lambda q: fused(q, 1.0, 1.0),
        "rrf lex1.2+vec1.0": lambda q: fused(q, 1.2, 1.0),
        "rrf lex1.2+vec0.5": lambda q: fused(q, 1.2, 0.5),
    }
    stored = sum(len(t) // 4 for t in texts)
    variants = [("original", 0.0, bench)]
    for rate in a.rates:
        rng = random.Random(0)
        items, el, rp, changed = [], 0, 0, 0
        for it in bench.items:
            q2, e, r = paraphrase(it.query, rate, rng, wn)
            el += e
            rp += r
            changed += q2 != it.query
            items.append(BenchItem(q2, it.gold, it.qtype, it.answer, it.gold_idx))
        print(f"[paraphrase rate={rate}] {changed}/{len(items)} questions changed, {rp}/{el} eligible words replaced", flush=True)
        for it0, it1 in list(zip(bench.items, items))[:3]:
            print(f"    {it0.query!r} -> {it1.query!r}", flush=True)
        variants.append((f"paraphrased r={rate}", rate, Benchmark(bench.episodes, items, {}, name="locomo-para")))
    results = {}
    for vname, rate, b in variants:
        rows = [_run(name, b, k, texts, stored, fn) for name, fn in systems.items()]
        results[vname] = rows
        print(f"\n### LoCoMo sessions, questions: {vname}\n{format_table(rows)}", flush=True)
    print("\n### Recall@K drop from original to paraphrased")
    base = {r.system: r.recall for r in results["original"]}
    for vname in results:
        if vname == "original":
            continue
        print(f"  {vname}: " + ", ".join(f"{r.system}={r.recall:.3f} ({r.recall - base[r.system]:+.3f})" for r in results[vname]), flush=True)
    if a.out:
        json.dump({v: [{k2: val for k2, val in r.__dict__.items() if k2 != "per_query_recall"} for r in rows] for v, rows in results.items()}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
