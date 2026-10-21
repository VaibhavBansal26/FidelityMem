"""Export the result JSON files under docs/results/ into the two CSV tables the
dashboard reads (recall by system and view; LoCoMo recall by question type).

    python -m fidelitymem.eval.dashboard_export docs/results docs/results/dashboard
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

LABEL = {
    "fixed_window": "Fixed window", "bm25": "BM25", "vector": "Dense", "bm25+vector_rrf": "BM25 + dense RRF",
    "summary_L3+bm25": "Lossy summary + BM25", "window+trie": "Window + trie", "memgpt_style": "MemGPT-style",
    "mem0_style": "Mem0-style", "zep_style": "Zep-style", "amem_style": "A-MEM-style", "hipporag_style": "HippoRAG-style",
    "fidelitymem_L0_raw": "FidelityMem raw (L0)", "fidelitymem_L3_gated": "FidelityMem L3 gated",
    "fidelitymem_L3_gate:facts": "FidelityMem L3 gated (facts)", "fidelitymem_L3_nogate": "FidelityMem L3 no gate",
    "fidelitymem_L2_gate:content": "FidelityMem L2 gated (content)",
}
KEEP_FM = set(k for k in LABEL if k.startswith("fidelitymem"))


def label(s: str) -> str:
    if s in LABEL:
        return LABEL[s]
    if s.startswith("fidelitymem_ctrl:self"):
        return "FidelityMem controller (self-probes)"
    if s.startswith("fidelitymem_ctrl:qlog"):
        return "FidelityMem controller (query log)"
    if s.startswith("fidelitymem_ctrl"):
        return "FidelityMem controller"
    return s


def family(s: str) -> str:
    if s.startswith("fidelitymem"):
        return "FidelityMem"
    if s in ("fixed_window", "bm25", "vector", "bm25+vector_rrf", "summary_L3+bm25"):
        return "Classic baseline"
    return "Memory-system architecture"


def _row(dataset: str, records: int, r: dict, raw_storage: int, embedder: str) -> dict:
    return {
        "dataset": dataset, "records": records, "k": r["k"], "embedder": embedder, "system": label(r["system"]), "family": family(r["system"]),
        "recall": round(r["recall"], 3), "ci95": round((r.get("ci95") or {}).get("recall", 0) or 0, 3), "complete_recall": round(r["complete_recall"], 3),
        "mrr": round(r["mrr"], 3), "storage_pct": round(100 * r["storage_tokens"] / max(1, raw_storage), 1), "tokens_per_query": round(r["tokens_per_query"]), "p95_ms": round(r["p95_ms"], 2),
    }


def _keep(s: str) -> bool:
    return not s.startswith("fidelitymem") or s in KEEP_FM or s.startswith("fidelitymem_ctrl")


def export(results_dir: Path, out_dir: Path) -> tuple[int, int]:
    rows, types = [], []
    for path in sorted(results_dir.glob("scale_*.json")):
        rep = json.load(open(path))
        for lvl, rs in rep.get("levels", {}).items():
            raw = next((r["storage_tokens"] for r in rs if r["system"] == "fidelitymem_L0_raw"), 1)
            rows += [_row("Synthetic", int(lvl), r, raw, "pretrained word vectors") for r in rs if _keep(r["system"])]
    for path in sorted(results_dir.glob("locomo_*.json")):
        rep = json.load(open(path))
        if "aggregate" not in rep:
            continue
        gran = rep.get("granularity", "turn")
        emb = "adapted (trained per seed)" if "adapted" in rep.get("embedder", "") else "pretrained word vectors"
        name = f"LoCoMo {'sessions' if gran == 'session' else 'turns'}" + (" · adapted embedder" if emb.startswith("adapted") else "")
        raw = next((r["storage_tokens"] for r in rep["aggregate"] if r["system"] == "fidelitymem_L0_raw"), 1)
        rows += [_row(name, rep["episodes"], r, raw, emb) for r in rep["aggregate"] if _keep(r["system"])]
        for r in rep["aggregate"]:
            s = r["system"]
            if s in ("bm25", "zep_style", "hipporag_style", "mem0_style", "window+trie", "fidelitymem_L0_raw") or s.startswith("fidelitymem_ctrl"):
                for t, v in r.get("per_type", {}).items():
                    if t in ("answer_containment", "answer_accuracy_llm"):
                        continue
                    types.append({"dataset": name, "system": label(s), "question_type": t.replace("_", " "), "recall": round(v, 3)})
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "recall_by_system.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "recall_by_type.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(types[0]))
        w.writeheader()
        w.writerows(types)
    return len(rows), len(types)


if __name__ == "__main__":
    n, m = export(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"{n} system rows, {m} per-type rows")
