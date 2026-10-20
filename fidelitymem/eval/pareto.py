"""Pareto frontier over (recall, stored tokens, tokens per query, p95 latency).

A system is dominated if another system is at least as good on every objective
and strictly better on one (recall maximised, the rest minimised). Produces a
Markdown table and a two-panel PNG (recall vs stored tokens, recall vs tokens
per query) with the frontier highlighted.

    python -m fidelitymem.eval.pareto docs/results/scale_k10.json --level 10000 --png docs/results/pareto_10k.png
    python -m fidelitymem.eval.pareto docs/results/locomo.json --png docs/results/pareto_locomo.png
"""
from __future__ import annotations

import argparse
import json


def _rows(report: dict, level: str | None):
    if "aggregate" in report:  # real-dataset report
        return report["aggregate"]
    levels = report.get("levels", report)
    key = level or sorted(levels, key=int)[-1]
    return levels[str(key)] if str(key) in levels else levels[int(key)]


def frontier(rows: list[dict], objectives=("recall", "storage_tokens", "tokens_per_query", "p95_ms")) -> list[dict]:
    def better_or_equal(a, b):
        return a["recall"] >= b["recall"] and all(a[o] <= b[o] for o in objectives[1:])

    def strictly(a, b):
        return a["recall"] > b["recall"] or any(a[o] < b[o] for o in objectives[1:])

    out = []
    for r in rows:
        dominated = any(better_or_equal(o, r) and strictly(o, r) for o in rows if o is not r)
        out.append({**r, "pareto": not dominated})
    return out


def table(rows: list[dict]) -> str:
    head = "| System | on frontier | R@K | CR@K | stored tok | tok/q | p95 ms | retain |\n|---|:---:|---:|---:|---:|---:|---:|---:|"
    lines = [head]
    for r in sorted(rows, key=lambda x: (-x["pareto"], -x["recall"], x["storage_tokens"])):
        ret = "-" if r.get("retention") is None else f"{r['retention']:.3f}"
        lines.append(f"| {r['system']} | {'**yes**' if r['pareto'] else 'no'} | {r['recall']:.3f} | {r['complete_recall']:.3f} | {r['storage_tokens']} | {r['tokens_per_query']:.0f} | {r['p95_ms']:.2f} | {ret} |")
    return "\n".join(lines)


def plot(rows: list[dict], png: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, xkey, xlabel in ((axes[0], "storage_tokens", "stored tokens (index size)"), (axes[1], "tokens_per_query", "context tokens per query")):
        for r in rows:
            fm = r["system"].startswith("fidelitymem")
            ax.scatter(r[xkey], r["recall"], s=70 if r["pareto"] else 35, marker="o" if fm else "s",
                       facecolors=("#1f77b4" if fm else "#7f7f7f") if r["pareto"] else "none",
                       edgecolors="#1f77b4" if fm else "#7f7f7f", linewidths=1.2, zorder=3 if r["pareto"] else 2)
            ax.annotate(r["system"].replace("fidelitymem_", "fm_"), (r[xkey], r["recall"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
        front = sorted([r for r in rows if r["pareto"]], key=lambda x: x[xkey])
        if len(front) > 1:
            ax.plot([r[xkey] for r in front], [r["recall"] for r in front], "-", color="#1f77b4", alpha=0.4, zorder=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Recall@K")
        ax.grid(alpha=0.3)
    fig.suptitle(title + "  (filled = on Pareto frontier; circles = FidelityMem, squares = baselines)")
    fig.tight_layout()
    fig.savefig(png, dpi=130)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--level", default=None)
    ap.add_argument("--png", default=None)
    a = ap.parse_args(argv)
    rep = json.load(open(a.report))
    rows = frontier(_rows(rep, a.level))
    print(table(rows))
    if a.png:
        plot(rows, a.png, f"{a.report.split('/')[-1]}" + (f" n={a.level}" if a.level else ""))
        print(f"\nwrote {a.png}")


if __name__ == "__main__":
    main()
