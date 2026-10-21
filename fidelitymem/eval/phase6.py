"""Phase 6: LLM-judged answer accuracy next to recall, tokens, latency and API cost.

For each question: a memory system retrieves its context, a Claude *reader* answers
from that context only, a Claude *judge* compares the answer with the gold answer.
Reported per system: answer accuracy (overall and per question category),
Recall@K of the gold evidence, context tokens per query, retrieval latency, and the
reader/judge API usage with an estimated cost.

Systems compared: BM25 (the lexical baseline), FidelityMem under its controller
(optionally with the Claude compressor and extractor at ingestion), and any of the
architecture re-implementations. LoCoMo is loaded from the repository's data
directory; LongMemEval from a local file when present.

    # with credentials (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or `ant auth login`):
    python -m fidelitymem.eval.phase6 --dataset locomo --path data/locomo/locomo10.json \
        --questions 300 --k 5 --session-level --out docs/results/phase6_locomo.json

    # plumbing check without credentials (mock reader/judge: containment heuristic):
    python -m fidelitymem.eval.phase6 --dataset locomo --path data/locomo/locomo10.json --mock --questions 100
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field

from ..memory import FidelityMem
from ..policy import MemoryPolicy, Mode
from ..stores import BM25Index, VectorIndex, set_default_embedder
from ..stores.embedders import best_available, embedder_name
from ..text import estimate_tokens
from .datasets import load
from .harness import _ingest, _set_cfg
from .metrics import recall_at_k
from .synthetic import Benchmark, from_real

# Anthropic first-party API list prices, USD per million tokens (input, output); see the
# claude-api skill's model table. Used only to *estimate* the cost column.
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50), "claude-fable-5-1": (10.0, 50.0)}


@dataclass
class SystemResult:
    system: str
    n: int = 0
    correct: int = 0
    abstained: int = 0
    recall: list[float] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    latency_ms: list[float] = field(default_factory=list)
    by_type: dict[str, list[int]] = field(default_factory=dict)  # type -> [correct, n]
    reader_in: int = 0
    reader_out: int = 0
    judge_in: int = 0
    judge_out: int = 0

    def summary(self, reader_model: str, judge_model: str) -> dict:
        n = max(1, self.n)
        pi, po = PRICES.get(reader_model, (0, 0))
        ji, jo = PRICES.get(judge_model, (0, 0))
        cost = (self.reader_in * pi + self.reader_out * po + self.judge_in * ji + self.judge_out * jo) / 1e6
        return {
            "system": self.system, "questions": self.n, "accuracy": round(self.correct / n, 3), "abstention_rate": round(self.abstained / n, 3),
            "recall@k": round(sum(self.recall) / max(1, len(self.recall)), 3), "tokens_per_query": round(sum(self.tokens) / n),
            "p50_ms": round(sorted(self.latency_ms)[len(self.latency_ms) // 2], 2) if self.latency_ms else 0.0,
            "accuracy_by_type": {t: round(c / max(1, m), 3) for t, (c, m) in sorted(self.by_type.items())},
            "api_tokens": {"reader_in": self.reader_in, "reader_out": self.reader_out, "judge_in": self.judge_in, "judge_out": self.judge_out},
            "estimated_cost_usd": round(cost, 4), "cost_per_question_usd": round(cost / n, 5),
        }


def _bm25_system(bench: Benchmark, k: int):
    texts = [t for t, _ in bench.episodes]
    bm = BM25Index()
    for i, t in enumerate(texts):
        bm.add(f"ep{i}", t)

    def retrieve(q: str, kk: int = k):
        hits = [i for i, _ in bm.search(q, kk)]
        return hits, "\n".join(texts[int(i[2:])] for i in hits)

    return retrieve


def _with_llm_steps(retrieve, texts_of, k: int, rewriter=None, reranker=None, pool: int = 20):
    """Wrap a retrieval closure with optional query rewriting and reranking of a wider pool."""

    def wrapped(q: str, kk: int = k):
        q2 = rewriter.rewrite(q) if rewriter else q
        if reranker is None:
            return retrieve(q2, kk)
        hits, _ = retrieve(q2, pool)
        top = reranker.rerank(q, [(h, texts_of(h)) for h in hits], kk)
        return top, "\n".join(texts_of(h) for h in top)

    return wrapped


def _competitor_system(name: str, bench: Benchmark, k: int):
    from . import competitors

    texts = [t for t, _ in bench.episodes]
    fn = getattr(competitors, name)
    # competitors expose a Result; re-create the retrieval closure by calling the module's internals
    ids = [f"ep{i}" for i in range(len(texts))]
    if name == "zep_style":
        from ..extraction import extract_relations
        from ..retrieval.fusion import rrf
        from ..stores import GraphStore
        from ..text import extract_entities

        bm, vi, g = BM25Index(), VectorIndex(), GraphStore()
        for i, t in zip(ids, texts):
            bm.add(i, t)
            g.add_memory(i, extract_entities(t), extract_relations(t))
        vi.add_many(list(zip(ids, texts)))

        def retrieve(q: str, kk: int = k):
            fused = rrf({"lexical": bm.search(q, kk * 3), "vector": vi.search(q, kk * 3), "graph": g.expand(extract_entities(q), 1, kk * 3)}, {"lexical": 1, "vector": 1, "graph": 1})
            hits = [i for i, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:kk]]
            return hits, "\n".join(texts[int(i[2:])] for i in hits)

        return retrieve
    raise ValueError(f"no phase-6 retrieval for {name}; supported: bm25, fidelitymem, zep_style")


def _fidelitymem_system(bench: Benchmark, calib: Benchmark | None, k: int, rho: float, compressor=None, extractor=None, fast: bool = True):
    pol = MemoryPolicy(minimum_recall_retention=rho, mode=Mode.BALANCED, max_context_tokens=100_000)
    fm = FidelityMem(pol, compressor=compressor, extractor=extractor)
    ep_ids: dict[int, str] = {}
    for idx, (text, ts) in enumerate(bench.episodes):
        m = fm.remember(text, timestamp=ts, importance=0.6)
        if m:
            ep_ids[idx] = m.episode_id
    if fast:
        fm.controller.search_space["top_k"] = [k]
        fm.controller.search_space["expand_budget"] = [0, 2]
    _set_cfg(fm, 0, k, True)
    probes = None
    if calib is not None and calib.items:
        from ..controller import ProbeQuery

        probes = [ProbeQuery(it.query, [ep_ids[i] for i in calib.gold_indices(it) if i in ep_ids]) for it in calib.items if calib.gold_indices(it)]
    fm.optimise(n_probes=120, probes=probes)
    cfg = fm.controller.current
    fm.controller.current = cfg.__class__(cfg.compression_level, k, cfg.expand_budget, cfg.routes, cfg.gate_profile)
    inv = {v: i for i, v in ep_ids.items()}

    def retrieve(q: str, kk: int = k):
        res, st = fm.search(q, k=kk)
        return [f"ep{inv[r.memory.episode_id]}" for r in res if r.memory.episode_id in inv], "\n".join(r.context_text for r in res)

    return retrieve, fm


def run(bench: Benchmark, calib: Benchmark | None, systems: list[str], k: int, rho: float, reader, judge, llm_compress: bool, log=print, rewriter=None, reranker=None, retrieval_only: bool = False) -> list[dict]:
    out = []
    texts = [t for t, _ in bench.episodes]
    texts_of = lambda h: texts[int(h[2:])]
    for name in systems:
        t0 = time.perf_counter()
        if name == "bm25":
            retrieve = _bm25_system(bench, k)
        elif name == "fidelitymem":
            comp = ext = None
            if llm_compress:
                from ..llm import ClaudeCompressor, ClaudeExtractor

                comp, ext = ClaudeCompressor(reader.llm), ClaudeExtractor(reader.llm)
            retrieve, fm = _fidelitymem_system(bench, calib, k, rho, comp, ext)
            log(f"[phase6] fidelitymem controller: {fm.controller.describe()}")
        else:
            retrieve = _competitor_system(name, bench, k)
        if rewriter or reranker:
            retrieve = _with_llm_steps(retrieve, texts_of, k, rewriter, reranker)
        log(f"[phase6] {name}: built in {time.perf_counter() - t0:.0f}s")
        sr = SystemResult(name + ("+rewrite" if rewriter else "") + ("+rerank" if reranker else ""))
        for it in bench.items:
            gold = [f"ep{i}" for i in bench.gold_indices(it)]
            t1 = time.perf_counter()
            hits, ctx = retrieve(it.query)
            sr.latency_ms.append((time.perf_counter() - t1) * 1000)
            sr.tokens.append(estimate_tokens(ctx))
            if gold:
                sr.recall.append(recall_at_k(hits, gold, k))
            if retrieval_only:
                sr.n += 1
                continue
            r_in0, r_out0 = reader.llm.input_tokens, reader.llm.output_tokens
            answer = reader.answer(it.query, ctx)
            sr.reader_in += reader.llm.input_tokens - r_in0
            sr.reader_out += reader.llm.output_tokens - r_out0
            j_in0, j_out0 = judge.llm.input_tokens, judge.llm.output_tokens
            ok = judge.judge(it.query, it.answer, answer)
            sr.judge_in += judge.llm.input_tokens - j_in0
            sr.judge_out += judge.llm.output_tokens - j_out0
            sr.n += 1
            sr.correct += int(ok)
            if "don't know" in answer.lower() or "do not know" in answer.lower():
                sr.abstained += 1
            c, m = sr.by_type.get(it.qtype, [0, 0])
            sr.by_type[it.qtype] = [c + int(ok), m + 1]
        out.append(sr)
        log(f"[phase6] {name}: accuracy={sr.correct / max(1, sr.n):.3f} recall@{k}={sum(sr.recall) / max(1, len(sr.recall)):.3f} tokens/q={sum(sr.tokens) / max(1, sr.n):.0f}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="locomo")
    ap.add_argument("--path", required=True)
    ap.add_argument("--session-level", action="store_true")
    ap.add_argument("--questions", type=int, default=300)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rho", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--systems", nargs="+", default=["bm25", "fidelitymem", "zep_style"])
    ap.add_argument("--reader-model", default="claude-opus-5-5")
    ap.add_argument("--judge-model", default="claude-sonnet-5-5")
    ap.add_argument("--llm-compress", action="store_true", help="use the Claude compressor/extractor at FidelityMem ingestion")
    ap.add_argument("--embedder", default="spacy")
    ap.add_argument("--mock", action="store_true", help="mock reader/judge (containment heuristic); validates the pipeline without credentials")
    ap.add_argument("--rewrite", action="store_true", help="HyDE-style query rewriting with a Haiku call per question")
    ap.add_argument("--rerank", action="store_true", help="Haiku reranking of a 20-candidate pool down to K")
    ap.add_argument("--retrieval-only", action="store_true", help="skip reader and judge; report recall only")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    set_default_embedder(best_available(a.embedder))
    kw = {"per_turn": not a.session_level} if a.dataset == "locomo" else {}
    real = load(a.dataset, a.path, **kw)
    bench = from_real(real, seed=a.seed)
    items = list(bench.items)
    random.Random(a.seed).shuffle(items)
    cut = int(len(items) * 0.3)
    calib = Benchmark(bench.episodes, items[:cut], {}, name="calib")
    test = Benchmark(bench.episodes, items[cut : cut + a.questions], {}, name=bench.name)

    if a.mock:
        from ..llm import ClaudeJudge, ClaudeReader, MockLLM

        llm = MockLLM()
        reader, judge = ClaudeReader(llm), ClaudeJudge(llm)
        # the mock reader returns "unknown"; for a plumbing check, answer with the context itself so
        # the containment judge measures whether the gold answer was *retrieved*
        reader.answer = lambda q, ctx: ctx[:4000]
        for l in (reader.llm, judge.llm):
            l.input_tokens = l.output_tokens = 0
        mode = "mock"
    else:
        from ..llm import ClaudeJudge, ClaudeLLM, ClaudeReader, available

        if not available():
            raise SystemExit("No Anthropic credential available (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or `ant auth login`). Use --mock to validate the pipeline.")
        reader = ClaudeReader(ClaudeLLM(model=a.reader_model, effort="low"))
        judge = ClaudeJudge(ClaudeLLM(model=a.judge_model, effort="low"))
        mode = "claude"
    print(f"phase6 dataset={real.name} granularity={real.meta.get('granularity')} questions={len(test.items)} calib={len(calib.items)} K={a.k} mode={mode} embedder={embedder_name(set_default_embedder.__globals__['get_default_embedder']())}", flush=True)
    rewriter = reranker = None
    if a.rewrite or a.rerank:
        from ..llm import ClaudeLLM, ClaudeReranker, ClaudeRewriter, MockLLM

        aux = MockLLM() if a.mock else ClaudeLLM(model="claude-haiku-5-5", effort="low")
        rewriter = ClaudeRewriter(aux) if a.rewrite else None
        reranker = ClaudeReranker(aux) if a.rerank else None
    results = run(test, calib, a.systems, a.k, a.rho, reader, judge, a.llm_compress, log=lambda s: print(s, flush=True), rewriter=rewriter, reranker=reranker, retrieval_only=a.retrieval_only)
    if rewriter or reranker:
        aux = (rewriter or reranker).llm
        print(f"[phase6] rewrite/rerank API tokens: in={aux.input_tokens} out={aux.output_tokens} (claude-haiku-5-5)", flush=True)
    rows = [r.summary(a.reader_model, a.judge_model) for r in results]
    print("\n| System | accuracy | abstain | recall@K | tokens/q | p50 ms | est. cost / q |\n|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['system']} | {r['accuracy']:.3f} | {r['abstention_rate']:.3f} | {r['recall@k']:.3f} | {r['tokens_per_query']} | {r['p50_ms']} | ${r['cost_per_question_usd']:.4f} |")
    for r in rows:
        print(f"  {r['system']} by type: {r['accuracy_by_type']}")
    if a.out:
        json.dump({"dataset": real.name, "granularity": real.meta.get("granularity"), "mode": mode, "k": a.k, "questions": len(test.items), "reader_model": a.reader_model, "judge_model": a.judge_model, "llm_compress": a.llm_compress, "results": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
