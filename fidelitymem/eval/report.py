"""Render a scale-study JSON report (from `fidelitymem-bench --scale ... --out`) as
Markdown tables for the paper: one table per record level, plus a cross-scale
summary of Recall@K, tokens/query, p95 latency and stored tokens per system."""
from __future__ import annotations

import json
import sys


def _fmt(v, nd=3):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def per_level_table(rows: list[dict]) -> str:
    head = "| System | R@K | CR@K | MRR | nDCG | tok/q | p50 ms | p95 ms | stored tok | expand | retain |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    out = [head]
    for r in rows:
        out.append(
            f"| {r['system']} | {_fmt(r['recall'])} | {_fmt(r['complete_recall'])} | {_fmt(r['mrr'])} | {_fmt(r['ndcg'])} | "
            f"{r['tokens_per_query']:.0f} | {_fmt(r['p50_ms'], 2)} | {_fmt(r['p95_ms'], 2)} | {r['storage_tokens']} | {_fmt(r['expansion_rate'], 2)} | {_fmt(r['retention'])} |"
        )
    return "\n".join(out)


def per_type_table(rows: list[dict]) -> str:
    types = ["factual", "temporal", "multi_hop", "exact"]
    out = ["| System | " + " | ".join(types) + " |", "|---|" + "---:|" * len(types)]
    for r in rows:
        out.append(f"| {r['system']} | " + " | ".join(_fmt(r["per_type"].get(t)) for t in types) + " |")
    return "\n".join(out)


def cross_scale(report: dict, metric: str, nd=3) -> str:
    levels = sorted(report, key=int)
    systems = []
    for lvl in levels:
        for r in report[lvl]:
            name = r["system"].split("(")[0]
            if name not in systems:
                systems.append(name)
    out = ["| System | " + " | ".join(f"n={lvl}" for lvl in levels) + " |", "|---|" + "---:|" * len(levels)]
    for sname in systems:
        cells = []
        for lvl in levels:
            match = [r for r in report[lvl] if r["system"].split("(")[0] == sname]
            v = match[0][metric] if match else None
            if v is None:
                cells.append("-")
            elif isinstance(v, float):
                cells.append(f"{v:.{nd}f}" if metric not in ("tokens_per_query",) else f"{v:.0f}")
            else:
                cells.append(str(v))
        label = sname
        for lvl in levels:
            m = [r for r in report[lvl] if r["system"].split("(")[0] == sname]
            if m and "(" in m[0]["system"]:
                label = sname + " " + "/".join(r["system"][len(sname):] for lvl2 in levels for r in report[lvl2] if r["system"].startswith(sname + "("))
                break
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(path: str) -> None:
    report = json.load(open(path))
    levels = sorted(report, key=int)
    print("## Cross-scale Recall@K\n")
    print(cross_scale(report, "recall"))
    print("\n## Cross-scale complete-evidence-set recall\n")
    print(cross_scale(report, "complete_recall"))
    print("\n## Cross-scale tokens per query\n")
    print(cross_scale(report, "tokens_per_query"))
    print("\n## Cross-scale p95 latency (ms)\n")
    print(cross_scale(report, "p95_ms", 2))
    print("\n## Cross-scale stored tokens\n")
    print(cross_scale(report, "storage_tokens"))
    for lvl in levels:
        print(f"\n## n = {lvl} memories\n")
        print(per_level_table(report[lvl]))
        print("\nRecall@K by question type\n")
        print(per_type_table(report[lvl]))


if __name__ == "__main__":
    main(sys.argv[1])
