"""Measure the semantic-codebook trie: storage, decode fidelity, retrieval.

    python -m fidelitymem.eval.semantic_trie_experiment --locomo data/locomo/locomo10.json --codes 500 1000 2000
"""
from __future__ import annotations

import argparse
import json
import time

from ..compression import build_probes
from ..stores import BM25Index, set_default_embedder
from ..stores.embedders import best_available, embedder_name
from ..stores.semantic_trie import SemanticCodebook, SemanticTrie
from ..text import estimate_tokens
from .datasets import load_locomo
from .metrics import recall_at_k
from .synthetic import from_real, generate


def run(bench, n_codes: int, embedder, k: int = 10) -> dict:
    texts = [t for t, _ in bench.episodes]
    ids = [f"ep{i}" for i in range(len(texts))]
    t0 = time.perf_counter()
    cb = SemanticCodebook(embedder, n_codes=n_codes).fit(texts)
    trie = SemanticTrie(cb)
    for i, t in zip(ids, texts):
        trie.add(i, t)
    build_s = time.perf_counter() - t0
    raw_bytes = sum(len(t.encode()) for t in texts)
    # decode fidelity: probes from the raw text must survive the decode
    fid_facts, fid_content = [], []
    for i, t in zip(ids, texts):
        dec = trie.decode(i)
        fid_facts.append(build_probes(t, "facts").score(dec))
        fid_content.append(build_probes(t, "content").score(dec))
    bm = BM25Index()
    for i, t in zip(ids, texts):
        bm.add(i, t)
    rec_trie, rec_bm, lat = [], [], []
    for item in bench.items:
        gold = [f"ep{g}" for g in bench.gold_indices(item)]
        if not gold:
            continue
        t0 = time.perf_counter()
        got = [d for d, _ in trie.search(item.query, k)]
        lat.append((time.perf_counter() - t0) * 1000)
        rec_trie.append(recall_at_k(got, gold, k))
        rec_bm.append(recall_at_k([d for d, _ in bm.search(item.query, k)], gold, k))
    n = max(1, len(rec_trie))
    return {
        "n_codes": n_codes, "episodes": len(texts), "vocab": len(cb.code_of), "build_s": round(build_s, 1),
        "raw_bytes": raw_bytes, "trie_bytes": trie.storage_bytes(), "storage_ratio": round(trie.storage_bytes() / raw_bytes, 3),
        "decode_fidelity_facts": round(sum(fid_facts) / len(fid_facts), 3), "decode_fidelity_content": round(sum(fid_content) / len(fid_content), 3),
        "recall@10_semantic_trie": round(sum(rec_trie) / n, 3), "recall@10_bm25": round(sum(rec_bm) / n, 3), "p50_ms": round(sorted(lat)[len(lat) // 2], 2) if lat else 0,
        "example_decode": trie.decode(ids[0])[:160], "example_raw": texts[0][:160],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--locomo", default=None)
    ap.add_argument("--synthetic", type=int, default=1000)
    ap.add_argument("--codes", type=int, nargs="+", default=[300, 1000, 3000])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    emb = best_available("spacy")
    set_default_embedder(emb)
    print("embedder", embedder_name(emb))
    out = {"embedder": embedder_name(emb), "runs": []}
    benches = []
    if a.synthetic:
        benches.append(("synthetic", generate(a.synthetic, 100, 0)))
    if a.locomo:
        benches.append(("locomo", from_real(load_locomo(a.locomo, conversations=[0, 1, 2]), seed=0, shuffle=False)))
    for name, bench in benches:
        for nc in a.codes:
            r = run(bench, nc, emb)
            r["dataset"] = name
            out["runs"].append(r)
            print(json.dumps(r), flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
