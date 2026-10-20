"""Architecture-faithful re-implementations of current memory systems' retrieval
strategies, for a controlled comparison.

These are NOT the vendors' code. Mem0, Letta, Zep and friends all need an LLM
to extract memories, which makes an offline, exact-recall comparison impossible
and would confound memory architecture with model quality. Each baseline here
reproduces the *retrieval architecture* of the system using the same
deterministic extractors, the same BM25 and the same embedder as FidelityMem,
so the only variable is the memory design.

  window_trie      sliding window of recent episodes + trie exact/prefix index (user proposal)
  memgpt_style     Letta/MemGPT: core (recent) + recall/archival vector search over raw text
  mem0_style       Mem0: atomic fact sentences as separate memories, vector search, dedupe/update
  zep_style        Zep/Graphiti: temporal entity graph + BM25 + vector, RRF, 1-hop expansion
  amem_style       A-MEM: notes linked to their nearest neighbours at insert; retrieve + follow links
  hipporag_style   HippoRAG: entity graph seeded by query entities, 2-hop spreading, fused with dense
"""
from __future__ import annotations

import time
from collections import defaultdict

from ..extraction import extract_relations
from ..retrieval.fusion import rrf
from ..stores import BM25Index, GraphStore, TrieIndex, VectorIndex
from ..text import estimate_tokens, extract_entities, sentences
from .synthetic import Benchmark


def _run(name, bench: Benchmark, k, texts, stored_tokens, retrieve):
    from .harness import _summarise

    runs, toks, lats = [], [], []
    for item in bench.items:
        gold = [f"ep{bench.text_to_idx[g]}" for g in item.gold]
        t0 = time.perf_counter()
        got = retrieve(item.query)[:k]
        lats.append((time.perf_counter() - t0) * 1000)
        toks.append(sum(estimate_tokens(texts[int(i[2:])]) for i in got))
        runs.append((got, gold, item.qtype))
    return _summarise(name, k, runs, toks, lats, stored_tokens, 0)


def window_trie(bench: Benchmark, k: int, window: int = 50):
    ids = [f"ep{i}" for i in range(len(bench.episodes))]
    texts = [t for t, _ in bench.episodes]
    trie = TrieIndex()
    for i, t in zip(ids, texts):
        trie.add(i, t)
    recent = ids[-window:][::-1]

    def retrieve(q):
        hits = [i for i, _ in trie.search(q, k)]
        for r in recent:
            if len(hits) >= k:
                break
            if r not in hits:
                hits.append(r)
        return hits

    return _run("window+trie", bench, k, texts, sum(estimate_tokens(t) for t in texts), retrieve)


def memgpt_style(bench: Benchmark, k: int, core: int = 10):
    ids = [f"ep{i}" for i in range(len(bench.episodes))]
    texts = [t for t, _ in bench.episodes]
    vi = VectorIndex()
    vi.add_many(list(zip(ids, texts)))
    core_ids = ids[-core:][::-1]

    def retrieve(q):
        # core memory is always in context; archival/recall search fills the rest
        hits = list(core_ids[: max(1, k // 5)])
        for i, _ in vi.search(q, k):
            if i not in hits:
                hits.append(i)
        return hits

    return _run("memgpt_style", bench, k, texts, sum(estimate_tokens(t) for t in texts), retrieve)


def mem0_style(bench: Benchmark, k: int):
    """Atomic facts: every fact-bearing sentence becomes its own memory; near-duplicate
    facts update in place. Retrieval is vector-only over facts, mapped back to episodes."""
    ids = [f"ep{i}" for i in range(len(bench.episodes))]
    texts = [t for t, _ in bench.episodes]
    vi = VectorIndex()
    fact_to_ep: dict[str, str] = {}
    items, seen = [], set()
    stored = 0
    for i, t in zip(ids, texts):
        for n, s in enumerate(sentences(t)):
            if not extract_entities(s):
                continue
            key = " ".join(sorted(set(s.lower().split())))
            if key in seen:
                continue
            seen.add(key)
            fid = f"{i}#{n}"
            fact_to_ep[fid] = i
            items.append((fid, s))
            stored += estimate_tokens(s)
    vi.add_many(items)

    def retrieve(q):
        out = []
        for fid, _ in vi.search(q, k * 3):
            ep = fact_to_ep[fid]
            if ep not in out:
                out.append(ep)
        return out

    return _run("mem0_style", bench, k, texts, stored, retrieve)


def zep_style(bench: Benchmark, k: int):
    ids = [f"ep{i}" for i in range(len(bench.episodes))]
    texts = [t for t, _ in bench.episodes]
    bm, vi, g = BM25Index(), VectorIndex(), GraphStore()
    for i, t in zip(ids, texts):
        bm.add(i, t)
        g.add_memory(i, extract_entities(t), extract_relations(t))
    vi.add_many(list(zip(ids, texts)))

    def retrieve(q):
        routes = {"lexical": bm.search(q, k * 3), "vector": vi.search(q, k * 3), "graph": g.expand(extract_entities(q), 1, k * 3)}
        fused = rrf(routes, {"lexical": 1, "vector": 1, "graph": 1})
        return [i for i, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]

    return _run("zep_style", bench, k, texts, sum(estimate_tokens(t) for t in texts), retrieve)


def amem_style(bench: Benchmark, k: int, links: int = 3):
    ids = [f"ep{i}" for i in range(len(bench.episodes))]
    texts = [t for t, _ in bench.episodes]
    vi = VectorIndex()
    link: dict[str, list[str]] = defaultdict(list)
    for i, t in zip(ids, texts):
        for nb, _ in vi.search(t, links):
            link[i].append(nb)
            link[nb].append(i)
        vi.add(i, t)

    def retrieve(q):
        out = [i for i, _ in vi.search(q, k)]
        for i in list(out):
            for nb in link.get(i, ()):
                if nb not in out:
                    out.append(nb)
        return out

    return _run("amem_style", bench, k, texts, sum(estimate_tokens(t) for t in texts), retrieve)


def hipporag_style(bench: Benchmark, k: int):
    ids = [f"ep{i}" for i in range(len(bench.episodes))]
    texts = [t for t, _ in bench.episodes]
    vi, g = VectorIndex(), GraphStore()
    for i, t in zip(ids, texts):
        g.add_memory(i, extract_entities(t), extract_relations(t))
    vi.add_many(list(zip(ids, texts)))

    def retrieve(q):
        ents = extract_entities(q)
        routes = {"graph": g.expand(ents, 2, k * 3) if ents else [], "vector": vi.search(q, k * 3)}
        fused = rrf(routes, {"graph": 1.5, "vector": 1})
        return [i for i, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]

    return _run("hipporag_style", bench, k, texts, sum(estimate_tokens(t) for t in texts), retrieve)


ALL = [window_trie, memgpt_style, mem0_style, zep_style, amem_style, hipporag_style]
