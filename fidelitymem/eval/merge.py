"""Merge per-seed result JSON files (from eval.real or eval.cli --scale) into one
aggregated report with 95% CIs, and parse the plain-text tables that
`format_table` prints (useful when a run was interrupted before writing JSON).

    python -m fidelitymem.eval.merge real a_s0.json a_s1.json a_s2.json --out merged.json
    python -m fidelitymem.eval.merge scale s0.json s1.json s2.json --out merged.json
    python -m fidelitymem.eval.merge text run.log --out parsed.json
"""
from __future__ import annotations

import argparse
import json
import re

from .harness import Result, aggregate, format_table

_COLS = ["recall", "complete_recall", "mrr", "ndcg", "tokens_per_query", "p50_ms", "p95_ms", "storage_tokens", "expansion_rate", "retention"]


def to_result(d: dict) -> Result:
    r = Result(d["system"], d["k"], d["recall"], d["complete_recall"], d["mrr"], d["ndcg"], d["tokens_per_query"], d["p50_ms"], d["p95_ms"], d["storage_tokens"], d.get("expansion_rate", 0.0), d.get("retention"), d.get("per_type", {}))
    r.ci95 = d.get("ci95", {})
    return r


def _dump(r: Result) -> dict:
    return {k: v for k, v in r.__dict__.items() if k != "per_query_recall"}


def merge_real(paths: list[str]) -> dict:
    reports = [json.load(open(p)) for p in paths]
    per_seed = []
    for rep in reports:
        per_seed.extend([[to_result(d) for d in rs] for rs in rep["per_seed"]])
    agg = aggregate(per_seed)
    base = dict(reports[0])
    base["seeds"] = sorted({s for rep in reports for s in rep["seeds"]})
    base["aggregate"] = [_dump(r) for r in agg]
    base["per_seed"] = [[_dump(r) for r in rs] for rs in per_seed]
    return base


def merge_scale(paths: list[str]) -> dict:
    reports = [json.load(open(p)) for p in paths]
    levels: dict[str, list[list[Result]]] = {}
    for rep in reports:
        for lvl, rows in rep["levels"].items():
            levels.setdefault(str(lvl), []).append([to_result(d) for d in rows])
    out = {"embedder": reports[0].get("embedder"), "seeds": sorted({s for rep in reports for s in rep["seeds"]}), "levels": {}}
    for lvl, per_seed in levels.items():
        agg = aggregate(per_seed) if len(per_seed) > 1 else per_seed[0]
        out["levels"][lvl] = [_dump(r) for r in agg]
    return out


_ROW = re.compile(r"^(?P<system>\S+)\s+(?P<k>\d+)\s+(?P<recall>[\d.]+)(?:\s+±(?P<ci>[\d.]+))?\s+(?P<cr>[\d.]+)\s+(?P<mrr>[\d.]+)\s+(?P<ndcg>[\d.]+)\s+(?P<tok>\d+)\s+(?P<p50>[\d.]+)\s+(?P<p95>[\d.]+)\s+(?P<stor>\d+)\s+(?P<exp>[\d.]+)\s+(?P<ret>[\d.]+|-)\s*$")


def parse_text(path: str) -> dict:
    """Parse '### <n> memories' sections (scale study) or '### <name>: mean' sections (real)."""
    levels: dict[str, list[dict]] = {}
    current = None
    for line in open(path):
        m = re.match(r"^### (\d+) memories", line)
        if m:
            current = m.group(1)
            levels[current] = []
            continue
        m2 = re.match(r"^### (\S+): mean over seeds", line)
        if m2:
            current = "aggregate"
            levels[current] = []
            continue
        if current is None:
            continue
        mm = _ROW.match(line.rstrip("\n"))
        if not mm:
            if line.startswith("per-type") or line.startswith("###"):
                current = None
            continue
        g = mm.groupdict()
        row = {"system": g["system"], "k": int(g["k"]), "recall": float(g["recall"]), "complete_recall": float(g["cr"]), "mrr": float(g["mrr"]), "ndcg": float(g["ndcg"]), "tokens_per_query": float(g["tok"]), "p50_ms": float(g["p50"]), "p95_ms": float(g["p95"]), "storage_tokens": int(g["stor"]), "expansion_rate": float(g["exp"]), "retention": None if g["ret"] == "-" else float(g["ret"]), "per_type": {}, "ci95": ({"recall": float(g["ci"])} if g["ci"] else {})}
        levels[current].append(row)
    if "aggregate" in levels:
        return {"aggregate": levels["aggregate"]}
    return {"levels": levels}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["real", "scale", "text"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = {"real": merge_real, "scale": merge_scale}.get(a.kind, lambda ps: parse_text(ps[0]))(a.paths)
    json.dump(rep, open(a.out, "w"), indent=1)
    if "aggregate" in rep:
        print(format_table([to_result(d) for d in rep["aggregate"]]))
    else:
        for lvl, rows in rep["levels"].items():
            print(f"\n### {lvl}\n" + format_table([to_result(d) for d in rows]))


if __name__ == "__main__":
    main()
